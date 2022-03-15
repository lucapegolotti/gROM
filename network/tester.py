import sys
import os

# this fixes a problem with openmp https://github.com/dmlc/xgboost/issues/1715
os.environ['KMP_DUPLICATE_LIB_OK']='True'

sys.path.append("../tools")

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

def test_train(gnn_model, model_name, dataset):
    num_examples = len(dataset)
    num_train = int(num_examples)
    train_sampler = SubsetRandomSampler(torch.arange(num_train))
    train_dataloader = GraphDataLoader(dataset, sampler=train_sampler,
                                       batch_size=1,
                                       drop_last=False)

    global_loss, count, elapsed, global_mae = training.evaluate_model(gnn_model, \
                                              train_dataloader, \
                                              training.mse, \
                                              training.mae)
    print('\tloss = {:.2e}\tmae = {:.2e}\ttime = {:.2f} s'.format(global_loss/count,
                                                                  global_mae/count,
                                                                  elapsed))

    return coefs_dict

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
        eln = pp.invert_normalize_function(el, field_name, coefs)
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


def test_rollout(model, params, dataset, index_graph, split, out_folder):
    model.eval()
    graph = dataset.lightgraphs[index_graph]
    model_name = params['dataset_parameters']['split'][split][index_graph]
    true_graph = load_graphs('../graphs/data/' + model_name + '.grph')[0][0]
    times = pp.get_times(true_graph)

    coefs_dict = params['normalization_coefficients']['features']
    label_coefs = params['normalization_coefficients']['labels']

    if label_coefs['normalization_type'] == 'min_max':
        minp = label_coefs['min'][0]
        maxp = label_coefs['max'][0]
        minq = label_coefs['min'][1]
        maxq = label_coefs['max'][1]
        def bring_to_range_p(pressure):
            return minp + (maxp - minp) * pressure
        def bring_to_range_q(flowrate):
            return minq + (maxq - minq) * flowrate
    elif label_coefs['normalization_type'] == 'standard':
        meanp = label_coefs['mean'][0]
        stdvp = label_coefs['std'][0]
        meanq = label_coefs['mean'][1]
        stdvq = label_coefs['std'][1]
        def bring_to_range_p(pressure):
            return pressure * stdvp + meanp
        def bring_to_range_q(flowrate):
            return flowrate * stdvq + meanq
    else:
        def bring_to_range_p(pressure):
            return pressure
        def bring_to_range_q(flowrate):
            return flowrate

    pressure_dict = {'branch': pp.normalize_function(true_graph.nodes['branch'].data['pressure_0'],
                              'pressure', coefs_dict),
                     'junction': pp.normalize_function(true_graph.nodes['junction'].data['pressure_0'],
                              'pressure', coefs_dict),
                     'inlet': pp.normalize_function(true_graph.nodes['inlet'].data['pressure_0'],
                             'pressure', coefs_dict),
                     'outlet': pp.normalize_function(true_graph.nodes['outlet'].data['pressure_0'],
                             'pressure', coefs_dict)}
    flowrate_dict = {'branch': pp.normalize_function(true_graph.nodes['branch'].data['flowrate_0'],
                             'flowrate', coefs_dict),
                     'junction': pp.normalize_function(true_graph.nodes['junction'].data['flowrate_0'],
                             'flowrate', coefs_dict),
                     'inlet': pp.normalize_function(true_graph.nodes['inlet'].data['flowrate_0'],
                             'flowrate', coefs_dict),
                     'outlet': pp.normalize_function(true_graph.nodes['outlet'].data['flowrate_0'],
                             'flowrate', coefs_dict)}

    new_state = {'pressure': pressure_dict, 'flowrate': flowrate_dict}

    err_p = 0
    err_q = 0
    err_p_branch = 0
    err_q_branch = 0
    err_p_junction = 0
    err_q_junction = 0

    norm_p = 0
    norm_q = 0
    norm_p_branch = 0
    norm_q_branch = 0
    norm_p_junction = 0
    norm_q_junction = 0
    pred_states = [new_state]
    real_states = [new_state]

    pressures_branch_pred = [new_state['pressure']['branch'].detach().numpy()]
    pressures_junction_pred = [new_state['pressure']['junction'].detach().numpy()]
    flowrates_branch_pred = [new_state['flowrate']['branch'].detach().numpy()]
    flowrates_junction_pred = [new_state['flowrate']['junction'].detach().numpy()]
    pressures_branch_real = [new_state['pressure']['branch'].detach().numpy()]
    pressures_junction_real = [new_state['pressure']['junction'].detach().numpy()]
    flowrates_branch_real = [new_state['flowrate']['branch'].detach().numpy()]
    flowrates_junction_real = [new_state['flowrate']['junction'].detach().numpy()]
    start = time.time()
    c_loss_total = 0
    total_flowrate = 0
    for t in range(len(times)-1):
        tp1 = t+1
        next_pressure_branch = pp.normalize_function(true_graph.nodes['branch'].data['pressure_' + str(tp1)], 'pressure', coefs_dict)
        next_flowrate_branch = pp.normalize_function(true_graph.nodes['branch'].data['flowrate_' + str(tp1)], 'flowrate', coefs_dict)
        next_pressure_junction = pp.normalize_function(true_graph.nodes['junction'].data['pressure_' + str(tp1)], 'pressure', coefs_dict)
        next_flowrate_junction = pp.normalize_function(true_graph.nodes['junction'].data['flowrate_' + str(tp1)], 'flowrate', coefs_dict)

        pressure_dict = {'inlet': pp.normalize_function(true_graph.nodes['inlet'].data['pressure_' + str(tp1)], 'pressure', coefs_dict),
                         'outlet': pp.normalize_function(true_graph.nodes['outlet'].data['pressure_' + str(tp1)], 'pressure', coefs_dict)}
        flowrate_dict = {'inlet': pp.normalize_function(true_graph.nodes['inlet'].data['flowrate_' + str(tp1)], 'flowrate', coefs_dict),
                         'outlet': pp.normalize_function(true_graph.nodes['outlet'].data['flowrate_' + str(tp1)], 'flowrate', coefs_dict)}

        new_bcs = {'pressure': pressure_dict, 'flowrate': flowrate_dict}

        pp.set_bcs(graph, new_bcs)
        pp.set_state(graph, new_state)

        average_flowrate = True
        pred_branch, pred_junction = model(graph, graph.nodes['branch'].data['n_features'].float(),
                                           graph.nodes['junction'].data['n_features'].float(),
                                           average_flowrate=average_flowrate)

        pred_branch = pred_branch.squeeze()
        pred_junction = pred_junction.squeeze()

        c_loss = gnn_model.compute_continuity_loss(graph, pred_branch, pred_junction, label_coefs, coefs_dict)
        c_loss_total = c_loss_total + float(c_loss.detach().numpy())
        total_flowrate = total_flowrate + float(true_graph.nodes['inlet'].data['flowrate_' + str(tp1)].detach().numpy())

        dp_branch = bring_to_range_p(pred_branch[:,0].detach().numpy())

        prev_p = graph.nodes['branch'].data['pressure'].detach().numpy().squeeze()

        p_branch = dp_branch + prev_p

        dq_branch = bring_to_range_q(pred_branch[:,1].detach().numpy())

        prev_q = graph.nodes['branch'].data['flowrate'].detach().numpy().squeeze()

        q_branch = dq_branch + prev_q

        err_p_branch = err_p_branch + np.linalg.norm(p_branch - next_pressure_branch.detach().numpy().squeeze())**2
        norm_p_branch = norm_p_branch + np.linalg.norm(next_pressure_branch.detach().numpy().squeeze())**2
        err_q_branch = err_q_branch + np.linalg.norm(q_branch - next_flowrate_branch.detach().numpy().squeeze())**2
        norm_q_branch = norm_q_branch + np.linalg.norm(next_flowrate_branch.detach().numpy().squeeze())**2

        dp_junction = bring_to_range_p(pred_junction[:,0].detach().numpy())

        prev_p = graph.nodes['junction'].data['pressure'].detach().numpy().squeeze()

        p_junction = dp_junction + prev_p

        dq_junction = bring_to_range_q(pred_junction[:,1].detach().numpy())

        prev_q = graph.nodes['junction'].data['flowrate'].detach().numpy().squeeze()

        q_junction = dq_junction + prev_q

        err_p_junction = err_p_junction + np.linalg.norm(p_junction - next_pressure_junction.detach().numpy().squeeze())**2
        norm_p_junction = norm_p_junction + np.linalg.norm(next_pressure_junction.detach().numpy().squeeze())**2
        err_q_junction = err_q_junction + np.linalg.norm(q_junction - next_flowrate_junction.detach().numpy().squeeze())**2
        norm_q_junction = norm_q_junction + np.linalg.norm(next_flowrate_junction.detach().numpy().squeeze())**2

        err_p = err_p + err_p_branch + err_p_junction
        norm_p = norm_p + norm_p_branch + norm_p_junction

        err_q = err_q + err_q_branch + err_q_junction
        norm_q = norm_q + norm_q_branch + norm_q_junction

        pressure_dict_exact = {'branch': next_pressure_branch,
                               'junction': next_pressure_junction,
                               'inlet': graph.nodes['inlet'].data['pressure_next'],
                               'outlet': graph.nodes['outlet'].data['pressure_next'],}
        flowrate_dict_exact = {'branch': next_flowrate_branch,
                               'junction': next_flowrate_junction,
                               'inlet': graph.nodes['inlet'].data['flowrate_next'],
                               'outlet': graph.nodes['outlet'].data['flowrate_next']}

        exact_state = {'pressure': pressure_dict_exact, 'flowrate': flowrate_dict_exact}


        pressures_branch_real.append(next_pressure_branch.detach().numpy())
        pressures_junction_real.append(next_pressure_junction.detach().numpy())
        flowrates_branch_real.append(next_flowrate_branch.detach().numpy())
        flowrates_junction_real.append(next_flowrate_junction.detach().numpy())

        pressure_dict = {'branch': torch.from_numpy(np.expand_dims(p_branch,axis=1)),
                         'junction': torch.from_numpy(np.expand_dims(p_junction,axis=1)),
                         'inlet': graph.nodes['inlet'].data['pressure_next'],
                         'outlet': graph.nodes['outlet'].data['pressure_next'],}
        flowrate_dict = {'branch': torch.from_numpy(np.expand_dims(q_branch,axis=1)),
                         'junction': torch.from_numpy(np.expand_dims(q_junction,axis=1)),
                         'inlet': graph.nodes['inlet'].data['flowrate_next'],
                         'outlet': graph.nodes['outlet'].data['flowrate_next']}

        new_state = {'pressure': pressure_dict, 'flowrate': flowrate_dict}

        pressures_branch_pred.append(np.expand_dims(p_branch,axis=1))
        pressures_junction_pred.append(np.expand_dims(p_junction,axis=1))
        flowrates_branch_pred.append(np.expand_dims(q_branch,axis=1))
        flowrates_junction_pred.append(np.expand_dims(q_junction,axis=1))

        pred_states.append(new_state)
        real_states.append(exact_state)

    end = time.time()

    print('Rollout time = {:.2f} s for {:.0f} timesteps'.format(end - start,
                                                                len(times)))

    err_p_branch = np.sqrt(err_p_branch / norm_p_branch)
    err_q_branch = np.sqrt(err_q_branch / norm_q_branch)
    err_p_junction = np.sqrt(err_p_junction / norm_p_junction)
    err_q_junction = np.sqrt(err_q_junction / norm_q_junction)
    err_p = np.sqrt(err_p / norm_p)
    err_q = np.sqrt(err_q / norm_q)

    print('Error pressure branches = {:.5e}'.format(err_p_branch))
    print('Error flowrate branches = {:.5e}'.format(err_q_branch))
    print('Global error branches = {:.5e}'.format(np.sqrt(err_p_branch**2 + err_q_branch**2)))
    print('Error pressure junctions = {:.5e}'.format(err_p_junction))
    print('Error flowrate junctions = {:.5e}'.format(err_q_junction))
    print('Global error junctions = {:.5e}'.format(np.sqrt(err_p_junction**2 + err_q_junction**2)))
    print('Error pressure = {:.5e}'.format(err_p))
    print('Error flowrate = {:.5e}'.format(err_q))
    print('Global error = {:.5e}'.format(np.sqrt(err_p**2 + err_q**2)))
    print('Relative flowrate loss = {:.5e}'.format(c_loss_total / total_flowrate))

    save_plots = True
    if save_plots:
        p_bounds = compute_min_max_list(pressures_branch_real + \
                                        pressures_junction_real, 'pressure', coefs_dict)
        q_bounds = compute_min_max_list(flowrates_branch_real + \
                                        flowrates_junction_real, 'flowrate', coefs_dict)

        bounds = {'pressure': p_bounds, 'flowrate': q_bounds}

        ptools.plot_static(graph, pressures_branch_pred, flowrates_branch_pred,
                           pressures_branch_real, flowrates_branch_real,
                           graph.nodes['params'].data['times'].detach().numpy(),
                           coefs_dict, npoints=3, outdir=out_folder)

        print3D = False
        if print3D:
            ptools.plot_3D(model_name, pred_states, graph.nodes['params'].data['times'].detach().numpy(),
                            coefs_dict, bounds, 'pressure', outfile_name=out_folder + '/3d_pressure_pred.mp4',
                            time = 5)

            ptools.plot_3D(model_name, real_states, graph.nodes['params'].data['times'].detach().numpy(),
                            coefs_dict, bounds, 'pressure', outfile_name=out_folder + '/3d_pressure_real.mp4',
                            time = 5)

            ptools.plot_3D(model_name, pred_states, graph.nodes['params'].data['times'].detach().numpy(),
                            coefs_dict, bounds, 'flowrate', outfile_name=out_folder + '/3d_flowrate_pred.mp4',
                            time = 5)

            ptools.plot_3D(model_name, real_states, graph.nodes['params'].data['times'].detach().numpy(),
                            coefs_dict, bounds, 'flowrate', outfile_name=out_folder + '/3d_flowrate_real.mp4',
                            time = 5)

        color_nodes = get_color_nodes(graph)

        ptools.plot_linear(pressures_branch_pred, flowrates_branch_pred,
                           pressures_junction_pred, flowrates_junction_pred,
                           pressures_branch_real, flowrates_branch_real,
                           pressures_junction_real, flowrates_junction_real,
                           color_nodes,
                           graph.nodes['params'].data['times'].detach().numpy(),
                           coefs_dict, bounds, out_folder + '/linear.mp4', time = 5)

        ptools.plot_node_types(graph, color_nodes, out_folder + '/node_types.mp4', time = 5)

    return err_p_branch, err_q_branch, err_p_junction, err_q_junction, \
           err_p, err_q, c_loss_total / total_flowrate

if __name__ == "__main__":
    path = sys.argv[1]

    params = json.load(open(path + '/parameters.json'))

    gnn_model = GraphNet(params['hyperparameters'])
    gnn_model.load_state_dict(torch.load(path + '/trained_gnn.pms'))
    gnn_model.eval()

    num_validation = len(params['dataset_parameters']['split']['validation'])

    dataset = pp.generate_dataset(params['dataset_parameters']['split']['validation'],
                                  dataset_params = params['dataset_parameters'],
                                  coefs_dict = params['normalization_coefficients']['features'])

    coefs_dict = dataset.coefs_dict

    print('==========VALIDATION==========')
    tot_err_p_branch = 0
    tot_err_q_branch = 0
    tot_err_p_junction = 0
    tot_err_q_junction = 0
    tot_err_p = 0
    tot_err_q = 0
    tot_continuity = 0
    if os.path.exists('results_validation'):
        shutil.rmtree('results_validation')
    training.create_directory('results_validation')
    for i in range(num_validation):
        model_name = params['dataset_parameters']['split']['validation'][i]
        print('model name = ' + model_name)
        training.create_directory('results_validation/' + model_name)
        err_p_branch, err_q_branch, err_p_junction, \
        err_q_junction, err_p, err_q, cont = test_rollout(gnn_model, params,
                                                   dataset, index_graph = i,
                                                   split = 'validation',
                                                   out_folder = 'results_validation/' + model_name)
        tot_err_p_branch = tot_err_p_branch + err_p_branch
        tot_err_q_branch = tot_err_q_branch + err_q_branch
        tot_err_p_junction = tot_err_p_junction + err_p_junction
        tot_err_q_junction = tot_err_q_junction + err_q_junction
        tot_err_p = tot_err_p + err_p
        tot_err_q = tot_err_q + err_q
        tot_continuity = tot_continuity + cont

    print('----------------------------')
    print('Global statistics')
    print('Error pressure branches = ' + str(tot_err_p_branch / num_validation))
    print('Error flowrate branches = ' + str(tot_err_q_branch / num_validation))
    print('Error pressure junction = ' + str(tot_err_p_junction / num_validation))
    print('Error flowrate junction = ' + str(tot_err_q_junction / num_validation))
    print('Error pressure = ' + str(tot_err_p / num_validation))
    print('Error flowrate = ' + str(tot_err_q / num_validation))
    print('Error continuity = ' + str(tot_continuity / num_validation))

    num_train = len(params['dataset_parameters']['split']['train'])

    dataset = pp.generate_dataset(params['dataset_parameters']['split']['train'],
                                  dataset_params = params['dataset_parameters'],
                                  coefs_dict = params['normalization_coefficients']['features'])
    coefs_dict = dataset.coefs_dict

    print('==========TRAIN==========')
    tot_err_p_branch = 0
    tot_err_q_branch = 0
    tot_err_p_junction = 0
    tot_err_q_junction = 0
    tot_err_p = 0
    tot_err_q = 0
    tot_continuity = 0
    if os.path.exists('results_train'):
        shutil.rmtree('results_train')
    training.create_directory('results_train')
    for i in range(num_train):
        model_name = params['dataset_parameters']['split']['train'][i]
        print('model name = ' + model_name)
        training.create_directory('results_train/' + model_name)
        err_p_branch, err_q_branch, err_p_junction, \
        err_q_junction, err_p, err_q, cont = test_rollout(gnn_model, params,
                                                  dataset, index_graph = i,
                                                  split = 'train',
                                                  out_folder = 'results_train/' + model_name)
        tot_err_p_branch = tot_err_p_branch + err_p_branch
        tot_err_q_branch = tot_err_q_branch + err_q_branch
        tot_err_p_junction = tot_err_p_junction + err_p_junction
        tot_err_q_junction = tot_err_q_junction + err_q_junction
        tot_err_p = tot_err_p + err_p
        tot_err_q = tot_err_q + err_q
        tot_continuity = tot_continuity + cont

    print('----------------------------')
    print('Global statistics')
    print('Error pressure branches = ' + str(tot_err_p_branch / num_train))
    print('Error flowrate branches = ' + str(tot_err_q_branch / num_train))
    print('Error pressure junction = ' + str(tot_err_p_junction / num_train))
    print('Error flowrate junction = ' + str(tot_err_q_junction / num_train))
    print('Error pressure = ' + str(tot_err_p / num_train))
    print('Error flowrate = ' + str(tot_err_q / num_train))
    print('Error continuity = ' + str(tot_continuity / num_train))
