import sys
import os

# this fixes a problem with openmp https://github.com/dmlc/xgboost/issues/1715
os.environ['KMP_DUPLICATE_LIB_OK']='True'

sys.path.append("../tools")
sys.path.append("../graphs")

import io_utils as io
import dgl
import torch
import preprocessing as pp
from graphnet import GraphNet
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dgl.data.utils import load_graphs
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import time
import preprocessing as pp
import json
import training
import plot_tools as ptools
import matplotlib.cm as cm
import shutil
import pathlib
import normalization as nrmz
from rollout import rollout

def get_color_nodes(graph, cmap = cm.get_cmap("plasma")):
    nnodes = graph.nodes['inner'].data['x'].shape[0]

    color_node = np.zeros((nnodes,4))
    node_type = graph.nodes['inner'].data['node_type'].detach().numpy()

    colors = np.zeros((node_type.shape[0]))

    for i in range(node_type.shape[0]):
        colors[i] = np.where(node_type[i,:] == 1)[0]

    node_type = np.copy(colors)

    colors = colors / np.max(colors)


    color_node = cmap(colors)
    color_node[graph.nodes['inlet'].data['mask']] = np.array([1,0,0,1])
    color_node[graph.nodes['outlet'].data['mask']] = np.array([0,1,0,1])

    return color_node, node_type

def compute_min_max_list(tlist, field_name, coefs):
    mm = np.infty
    MM = np.NINF

    for el in tlist:
        eln = nrmz.invert_normalize_function(el, field_name, coefs)
        mm = np.min([np.min(eln), mm])
        MM = np.max([np.max(eln), MM])

    return {'min': mm, 'max': MM}

def get_color_nodes(graph, cmap = cm.get_cmap("plasma")):
    nbranch = graph.nodes['branch'].data['x'].shape[0]
    njunction = graph.nodes['junction'].data['x'].shape[0]

    nnodes = nbranch + njunction

    node_type = graph.nodes['junction'].data['node_type'].detach().numpy()
    junction_colors = np.zeros((node_type.shape[0]))

    for i in range(node_type.shape[0]):
        junction_colors[i] = np.where(node_type[i,:] == 1)[0]


    junction_colors = junction_colors / np.max(junction_colors)
    junction_colors = cmap(junction_colors)

    color_node = np.zeros((nnodes,4))
    color_node = cmap(np.zeros(nbranch + njunction))
    color_node[nbranch:,:] = junction_colors

    return color_node

def print_rollout_errors(errors):
    print('Error pressure branches = {:.5e}'.format(np.sqrt(errors['p_branch']/errors['norm_p'])))
    print('Error flowrate branches = {:.5e}'.format(np.sqrt(errors['q_branch']/errors['norm_q'])))
    print('Global error branches = {:.5e}'.format(np.sqrt((errors['p_branch'] + errors['q_branch'])/errors['norm_t'])))
    print('Error pressure junctions = {:.5e}'.format(np.sqrt(errors['p_junction']/errors['norm_p'])))
    print('Error flowrate junctions = {:.5e}'.format(np.sqrt(errors['q_junction']/errors['norm_q'])))
    print('Global error junctions = {:.5e}'.format(np.sqrt((errors['p_junction'] + errors['p_junction'])/errors['norm_t'])))
    print('Error pressure = {:.5e}'.format(np.sqrt(errors['p']/errors['norm_p'])))
    print('Error flowrate = {:.5e}'.format(np.sqrt(errors['q']/errors['norm_q'])))
    print('Global error = {:.5e}'.format(np.sqrt((errors['p'] + errors['q'])/errors['norm_t'])))
    print('Relative flowrate loss = {:.5e}'.format(errors['continuity']))

def plot_rollout(solutions, coefs_dict, graph, true_graph, out_folder):
    pressures_branch_real = solutions['p_branch_real']
    pressures_junction_real = solutions['p_junction_real']
    flowrates_branch_real = solutions['q_branch_real']
    flowrates_junction_real = solutions['q_junction_real']
    pressures_branch_pred = solutions['p_branch_pred']
    pressures_junction_pred = solutions['p_junction_pred']
    flowrates_branch_pred = solutions['q_branch_pred']
    flowrates_junction_pred = solutions['q_junction_pred']

    p_bounds = compute_min_max_list(pressures_branch_real + \
                                    pressures_junction_real, 'pressure', coefs_dict)
    q_bounds = compute_min_max_list(flowrates_branch_real + \
                                    flowrates_junction_real, 'flowrate', coefs_dict)

    bounds = {'pressure': p_bounds, 'flowrate': q_bounds}

    ptools.plot_static(graph, pressures_branch_pred, flowrates_branch_pred,
                       pressures_branch_real, flowrates_branch_real,
                       nrmz.get_actual_times(true_graph, coefs_dict),
                       coefs_dict, npoints=3, outdir=out_folder)

    color_nodes = get_color_nodes(graph)

    ptools.plot_linear(pressures_branch_pred, flowrates_branch_pred,
                       pressures_junction_pred, flowrates_junction_pred,
                       pressures_branch_real, flowrates_branch_real,
                       pressures_junction_real, flowrates_junction_real,
                       color_nodes,
                       nrmz.get_actual_times(true_graph, coefs_dict),
                       coefs_dict, bounds, out_folder + '/linear.mp4', time = 5)

    ptools.plot_node_types(graph, color_nodes, out_folder + '/node_types.mp4', time = 5)

def evaluate_all_models(dataset, split_name, gnn_model, params):
    num_examples = len(params['dataset_parameters']['split'][split_name])
    coefs_dict = dataset.coefs_dict

    print('==========' + split_name + '==========')
    tot_err_p_branch = 0
    tot_err_q_branch = 0
    tot_err_global = 0
    tot_err_p_junction = 0
    tot_err_q_junction = 0
    tot_err_p = 0
    tot_err_q = 0
    tot_continuity = 0
    if os.path.exists('results_' + split_name):
        shutil.rmtree('results_' + split_name)
    pathlib.Path('results_' + split_name).mkdir(parents=True, exist_ok=True)
    for i in range(num_examples):
        model_name = params['dataset_parameters']['split'][split_name][i]
        print('model name = ' + model_name)
        pathlib.Path('results_' + split_name + '/' + model_name).mkdir(parents=True, exist_ok=True)
        errors, solutions, graph, true_graph = rollout(gnn_model, params,
                                    dataset, index_graph = i,
                                    split = split_name)
        print_rollout_errors(errors)
        plot_rollout(solutions,
                     params['normalization_coefficients']['features'],
                     graph, true_graph,
                     out_folder = 'results_' + split_name + '/' + model_name)

        tot_err_p_branch = tot_err_p_branch + np.sqrt(errors['p_branch'] / errors['norm_p'])
        tot_err_q_branch = tot_err_q_branch + np.sqrt(errors['q_branch'] / errors['norm_q'])
        tot_err_p_junction = tot_err_p_junction + np.sqrt(errors['p_junction'] / errors['norm_p'])
        tot_err_q_junction = tot_err_q_junction + np.sqrt(errors['q_junction'] / errors['norm_q'])
        tot_err_p = tot_err_p + np.sqrt(errors['p'] / errors['norm_p'])
        tot_err_q = tot_err_q + np.sqrt(errors['q'] / errors['norm_q'])
        tot_err_global = tot_err_global + np.sqrt((errors['p'] + errors['q']) / errors['norm_t'])
        tot_continuity = tot_continuity + errors['continuity']

    print('----------------------------')
    print('Global statistics')
    print('Error pressure branches = ' + str(tot_err_p_branch / num_examples))
    print('Error flowrate branches = ' + str(tot_err_q_branch / num_examples))
    print('Error pressure junction = ' + str(tot_err_p_junction / num_examples))
    print('Error flowrate junction = ' + str(tot_err_q_junction / num_examples))
    print('Error pressure = ' + str(tot_err_p / num_examples))
    print('Error flowrate = ' + str(tot_err_q / num_examples))
    print('Error global = ' + str(tot_err_global / num_examples))
    print('Error continuity = ' + str(tot_continuity / num_examples))

def check_loss(gnn_model, dataset, loss, parameters):
    num_examples = int(len(dataset))
    sampler = SubsetRandomSampler(torch.arange(num_examples))

    dataloader = GraphDataLoader(dataset, sampler=sampler,
                                   batch_size=1,
                                   # batch_size=parameters['train_parameters']['batch_size'],
                                   drop_last=False)

    results, _, elapsed = training.evaluate_model(gnn_model,
                                         dataloader, loss)

    msg = 'results: '
    msg = msg + 'train_loss = {:.2e} '.format(results['global_loss']/results['count'])
    msg = msg + 'train_con_loss = {:.2e} '.format(results['continuity_loss']/results['count'])
    msg = msg + 'time = {:.2f} s'.format(elapsed)
    print(msg)

if __name__ == "__main__":
    # trained model folder
    path = sys.argv[1]

    params = json.load(open(path + '/parameters.json'))

    gnn_model = GraphNet(params['hyperparameters'])
    gnn_model.load_state_dict(torch.load(path + '/trained_gnn.pms'))
    gnn_model.eval()

    dataset = pp.generate_dataset(params['dataset_parameters']['split']['train'], \
                                  io.data_location() + 'normalized_graphs', 'train')
    # check_loss(gnn_model, dataset, training.mse, params)
    evaluate_all_models(dataset, 'train', gnn_model, params)

    dataset = pp.generate_dataset(params['dataset_parameters']['split']['test'], \
                                  io.data_location() + 'normalized_graphs', 'test')

    evaluate_all_models(dataset, 'test', gnn_model, params)
