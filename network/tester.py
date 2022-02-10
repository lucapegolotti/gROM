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

def get_mask(graph,node_type):
    return np.squeeze(graph.nodes[node_type].data['global_mask'].detach().numpy())

def get_color_nodes(graph, cmap = cm.get_cmap("plasma")):
    nnodes = graph.nodes['inner'].data['global_mask'].shape[0] + \
             graph.nodes['inlet'].data['global_mask'].shape[0] + \
             graph.nodes['outlet'].data['global_mask'].shape[0]

    color_node = np.zeros((nnodes,4))
    node_type = graph.nodes['inner'].data['node_type'].detach().numpy()

    colors = np.zeros((node_type.shape[0]))

    for i in range(node_type.shape[0]):
        colors[i] = np.where(node_type[i,:] == 1)[0]

    colors = colors / np.max(colors)

    color_node[get_mask(graph,'inner')] = cmap(colors)
    color_node[get_mask(graph,'inlet')] = np.array([1,0,0,1])
    color_node[get_mask(graph,'outlet')] = np.array([0,1,0,1])

    return color_node

def get_solution_all_nodes(state, graph):

    nnodes = state['pressure']['inner'].shape[0] + \
             state['pressure']['inlet'].shape[0] + \
             state['pressure']['outlet'].shape[0]

    pressure = np.zeros((nnodes,1))
    pressure[get_mask(graph,'inlet')] = state['pressure']['inlet']
    pressure[get_mask(graph,'inner')] = state['pressure']['inner']
    pressure[get_mask(graph,'outlet')] = state['pressure']['outlet']

    flowrate = np.zeros((nnodes,1))
    flowrate[get_mask(graph,'inlet')] = state['flowrate']['inlet']
    flowrate[get_mask(graph,'inner')] = state['flowrate']['inner']
    flowrate[get_mask(graph,'outlet')] = state['flowrate']['outlet']

    return pressure, flowrate

def test_rollout(model, params, dataset, index_graph, split, out_folder):
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

    pressure_dict = {'inner': pp.normalize_function(true_graph.nodes['inner'].data['pressure_0'],
                             'pressure', coefs_dict),
                     'inlet': pp.normalize_function(true_graph.nodes['inlet'].data['pressure_0'],
                             'pressure', coefs_dict),
                     'outlet': pp.normalize_function(true_graph.nodes['outlet'].data['pressure_0'],
                             'pressure', coefs_dict)}
    flowrate_dict = {'inner': pp.normalize_function(true_graph.nodes['inner'].data['flowrate_0'],
                             'flowrate', coefs_dict),
                     'inlet': pp.normalize_function(true_graph.nodes['inlet'].data['flowrate_0'],
                             'flowrate', coefs_dict),
                     'outlet': pp.normalize_function(true_graph.nodes['outlet'].data['flowrate_0'],
                             'flowrate', coefs_dict)}

    new_state = {'pressure': pressure_dict, 'flowrate': flowrate_dict}

    pressure_exact, flowrate_exact = get_solution_all_nodes(new_state, graph)
    pressure_pred, flowrate_pred = get_solution_all_nodes(new_state, graph)

    err_p = 0
    err_q = 0
    norm_p = 0
    norm_q = 0
    pred_states = [new_state]
    real_states = [new_state]
    pressures_pred = [pressure_pred]
    pressures_real = [pressure_exact]
    flowrates_pred = [flowrate_pred]
    flowrates_real = [flowrate_exact]
    start = time.time()
    c_loss_total = 0
    total_flowrate = 0
    for t in range(len(times)-1):
        tp1 = t+1
        next_pressure = pp.normalize_function(true_graph.nodes['inner'].data['pressure_' + str(tp1)], 'pressure', coefs_dict)
        next_flowrate = pp.normalize_function(true_graph.nodes['inner'].data['flowrate_' + str(tp1)], 'flowrate', coefs_dict)

        pressure_dict = {'inlet': pp.normalize_function(true_graph.nodes['inlet'].data['pressure_' + str(tp1)], 'pressure', coefs_dict),
                         'outlet': pp.normalize_function(true_graph.nodes['outlet'].data['pressure_' + str(tp1)], 'pressure', coefs_dict)}
        flowrate_dict = {'inlet': pp.normalize_function(true_graph.nodes['inlet'].data['flowrate_' + str(tp1)], 'flowrate', coefs_dict),
                         'outlet': pp.normalize_function(true_graph.nodes['outlet'].data['flowrate_' + str(tp1)], 'flowrate', coefs_dict)}

        new_bcs = {'pressure': pressure_dict, 'flowrate': flowrate_dict}

        pp.set_bcs(graph, new_bcs)
        pp.set_state(graph, new_state)

        average_flowrate = True
        pred = model(graph, graph.nodes['inner'].data['n_features'].float(),
                     average_flowrate=average_flowrate).squeeze()

        c_loss = gnn_model.compute_continuity_loss(graph, pred, label_coefs, coefs_dict)
        c_loss_total = c_loss_total + float(c_loss.detach().numpy())
        total_flowrate = total_flowrate + float(true_graph.nodes['inlet'].data['flowrate_' + str(tp1)].detach().numpy())

        dp = bring_to_range_p(pred[:,0].detach().numpy())

        prev_p = graph.nodes['inner'].data['pressure'].detach().numpy().squeeze()

        p = dp + prev_p

        dq = bring_to_range_q(pred[:,1].detach().numpy())

        prev_q = graph.nodes['inner'].data['flowrate'].detach().numpy().squeeze()

        q = dq + prev_q

        err_p = err_p + np.linalg.norm(p - next_pressure.detach().numpy().squeeze())**2
        norm_p = norm_p + np.linalg.norm(next_pressure.detach().numpy().squeeze())**2
        err_q = err_q + np.linalg.norm(q - next_flowrate.detach().numpy().squeeze())**2
        norm_q = norm_q + np.linalg.norm(next_flowrate.detach().numpy().squeeze())**2

        pressure_dict_exact = {'inner': next_pressure,
                               'inlet': graph.nodes['inlet'].data['pressure_next'],
                               'outlet': graph.nodes['outlet'].data['pressure_next'],}
        flowrate_dict_exact = {'inner': next_flowrate,
                               'inlet': graph.nodes['inlet'].data['flowrate_next'],
                               'outlet': graph.nodes['outlet'].data['flowrate_next']}

        exact_state = {'pressure': pressure_dict_exact, 'flowrate': flowrate_dict_exact}

        pressure_exact, flowrate_exact = get_solution_all_nodes(exact_state, graph)

        pressures_real.append(pressure_exact)
        flowrates_real.append(flowrate_exact)

        pressure_dict = {'inner': torch.from_numpy(np.expand_dims(p,axis=1)),
                         'inlet': graph.nodes['inlet'].data['pressure_next'],
                         'outlet': graph.nodes['outlet'].data['pressure_next'],}
        flowrate_dict = {'inner': torch.from_numpy(np.expand_dims(q,axis=1)),
                         'inlet': graph.nodes['inlet'].data['flowrate_next'],
                         'outlet': graph.nodes['outlet'].data['flowrate_next']}

        new_state = {'pressure': pressure_dict, 'flowrate': flowrate_dict}
        pressure, flowrate = get_solution_all_nodes(new_state, graph)

        pressures_pred.append(pressure)
        flowrates_pred.append(flowrate)

        pred_states.append(new_state)
        real_states.append(exact_state)

    end = time.time()

    print('Rollout time = {:.2f} s for {:.0f} timesteps'.format(end - start,
                                                                len(times)))

    err_p = np.sqrt(err_p / norm_p)
    err_q = np.sqrt(err_q / norm_q)

    print('Error pressure = {:.5e}'.format(err_p))
    print('Error flowrate = {:.5e}'.format(err_q))
    print('Global error = {:.5e}'.format(np.sqrt(err_p**2 + err_q**2)))
    print('Relative flowrate loss = {:.5e}'.format(c_loss_total / total_flowrate))

    ptools.plot_3D(model_name, pred_states, graph.nodes['params'].data['times'].detach().numpy(),
                    coefs_dict, 'pressure', outfile_name=out_folder + '/3d_pressure_pred.mp4',
                    time = 5)

    ptools.plot_3D(model_name, real_states, graph.nodes['params'].data['times'].detach().numpy(),
                    coefs_dict, 'pressure', outfile_name=out_folder + '/3d_pressure_real.mp4',
                    time = 5)

    ptools.plot_3D(model_name, pred_states, graph.nodes['params'].data['times'].detach().numpy(),
                    coefs_dict, 'flowrate', outfile_name=out_folder + '/3d_flowrate_pred.mp4',
                    time = 5)

    ptools.plot_3D(model_name, real_states, graph.nodes['params'].data['times'].detach().numpy(),
                    coefs_dict, 'flowrate', outfile_name=out_folder + '/3d_flowrate_real.mp4',
                    time = 5)

    ptools.plot_linear(pressures_pred, flowrates_pred, pressures_real, flowrates_real,
                       get_color_nodes(graph, cmap = cm.get_cmap("plasma")),
                       graph.nodes['params'].data['times'].detach().numpy(),
                       coefs_dict, out_folder + '/linear.mp4', time = 5)

    ptools.plot_node_types(graph, out_folder + '/node_types.mp4', time = 5,
                           cmap = cm.get_cmap("plasma"))

    return err_p, err_q, np.sqrt(err_p**2 + err_q**2)

if __name__ == "__main__":
    path = sys.argv[1]

    params = json.load(open(path + '/parameters.json'))

    gnn_model = GraphNet(params['hyperparameters'])
    gnn_model.load_state_dict(torch.load(path + '/trained_gnn.pms'))
    gnn_model.eval()

    training.create_directory('results_train')
    num_train = len(params['dataset_parameters']['split']['train'])

    dataset = pp.generate_dataset(params['dataset_parameters']['split']['train'],
                                  dataset_params = params['dataset_parameters'],
                                  coefs_dict = params['normalization_coefficients']['features'])
    coefs_dict = dataset.coefs_dict

    for i in range(num_train):
        training.create_directory('results_train/' + str(i))
        err_p, err_q, global_error = test_rollout(gnn_model, params,
                                                  dataset, index_graph = i,
                                                  split = 'train',
                                                  out_folder = 'results_train/' + str(i))


    training.create_directory('results_validation')
    num_validation = len(params['dataset_parameters']['split']['validation'])

    dataset = pp.generate_dataset(params['dataset_parameters']['split']['validation'],
                                  dataset_params = params['dataset_parameters'],
                                  coefs_dict = params['normalization_coefficients']['features'])

    coefs_dict = dataset.coefs_dict

    for i in range(num_validation):
        training.create_directory('results_validation/' + str(i))
        err_p, err_q, global_error = test_rollout(gnn_model, params,
                                                  dataset, index_graph = i,
                                                  split = 'validation',
                                                  out_folder = 'results_validation/' + str(i))
