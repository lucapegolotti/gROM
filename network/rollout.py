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
import time
import preprocessing as pp
import json
import plot_tools as ptools
import matplotlib.cm as cm
import shutil
import pathlib
import normalization as nrmz

def rollout(gnn_model, params, dataset, index_graph, split, print_time = True):
    gnn_model.eval()
    graph = dataset.lightgraphs[index_graph]
    model_name = params['dataset_parameters']['split'][split][index_graph]
    true_graph = load_graphs(io.data_location() + 'normalized_graphs/' + model_name + '.0.grph')[0][0]
    times = nrmz.get_times(true_graph)

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

    pressure_dict = {'branch': true_graph.nodes['branch'].data['pressure_0'],
                     'inlet': true_graph.nodes['inlet'].data['pressure_0'],
                     'outlet': true_graph.nodes['outlet'].data['pressure_0']}
    flowrate_dict = {'branch': true_graph.nodes['branch'].data['flowrate_0'],
                     'inlet': true_graph.nodes['inlet'].data['flowrate_0'],
                     'outlet': true_graph.nodes['outlet'].data['flowrate_0']}

    new_state = {'pressure': pressure_dict, 'flowrate': flowrate_dict}

    err_p = 0
    err_q = 0
    err_p_branch = 0
    err_q_branch = 0

    norm_t = 0
    norm_p = 0
    norm_q = 0
    pred_states = [new_state]
    real_states = [new_state]

    pressures_branch_pred = [new_state['pressure']['branch'].detach().numpy()]
    flowrates_branch_pred = [new_state['flowrate']['branch'].detach().numpy()]
    pressures_branch_real = [new_state['pressure']['branch'].detach().numpy()]
    flowrates_branch_real = [new_state['flowrate']['branch'].detach().numpy()]
    start = time.time()
    c_loss_total = 0
    total_flowrate = 0
    for t in range(len(times)-1):
        tp1 = t+1
        next_pressure_branch = true_graph.nodes['branch'].data['pressure_' + str(tp1)]
        next_flowrate_branch = true_graph.nodes['branch'].data['flowrate_' + str(tp1)]

        pressure_dict = {'inlet': true_graph.nodes['inlet'].data['pressure_' + str(tp1)],
                         'outlet': true_graph.nodes['outlet'].data['pressure_' + str(tp1)]}
        flowrate_dict = {'inlet': true_graph.nodes['inlet'].data['flowrate_' + str(tp1)],
                         'outlet': true_graph.nodes['outlet'].data['flowrate_' + str(tp1)]}

        new_bcs = {'pressure': pressure_dict, 'flowrate': flowrate_dict}

        pp.set_bcs(graph, new_bcs)
        pp.set_state(graph, new_state)

        average_flowrate = True
        pred_branch, pred_junction = gnn_model(graph, graph.nodes['branch'].data['n_features'].float(),
                                               graph.nodes['junction'].data['n_features'].float(),
                                               average_flowrate=average_flowrate)

        pred_branch = pred_branch.squeeze()

        try:
            c_loss = gnn_model.compute_continuity_loss(graph, pred_branch, pred_junction, label_coefs, coefs_dict)
        except AttributeError:
            c_loss = gnn_model.module.compute_continuity_loss(graph, pred_branch, pred_junction, label_coefs, coefs_dict)

        c_loss_total = c_loss_total + float(c_loss.detach().numpy())
        fr = float(true_graph.nodes['inlet'].data['flowrate_' + str(tp1)].detach().numpy())
        total_flowrate = total_flowrate + nrmz.invert_normalize_function(fr, 'flowrate', coefs_dict)

        dp_branch = bring_to_range_p(pred_branch[:,0].detach().numpy())

        prev_p = graph.nodes['branch'].data['pressure'].detach().numpy().squeeze()

        p_branch = dp_branch + prev_p

        dq_branch = bring_to_range_q(pred_branch[:,1].detach().numpy())

        prev_q = graph.nodes['branch'].data['flowrate'].detach().numpy().squeeze()

        q_branch = dq_branch + prev_q

        curr_err_p_branch = np.linalg.norm(p_branch - next_pressure_branch.detach().numpy().squeeze())**2
        err_p_branch = err_p_branch + curr_err_p_branch
        curr_norm_p_branch = np.linalg.norm(next_pressure_branch.detach().numpy().squeeze())**2
        norm_t = norm_t + curr_norm_p_branch
        curr_err_q_branch = np.linalg.norm(q_branch - next_flowrate_branch.detach().numpy().squeeze())**2
        err_q_branch = err_q_branch + curr_err_q_branch
        curr_norm_q_branch = np.linalg.norm(next_flowrate_branch.detach().numpy().squeeze())**2
        norm_t = norm_t + curr_norm_q_branch

        norm_p = norm_p + curr_norm_p_branch
        norm_q = norm_q + curr_norm_q_branch

        pressure_dict_exact = {'branch': next_pressure_branch,
                               'inlet': graph.nodes['inlet'].data['pressure_next'],
                               'outlet': graph.nodes['outlet'].data['pressure_next'],}
        flowrate_dict_exact = {'branch': next_flowrate_branch,
                               'inlet': graph.nodes['inlet'].data['flowrate_next'],
                               'outlet': graph.nodes['outlet'].data['flowrate_next']}

        exact_state = {'pressure': pressure_dict_exact, 'flowrate': flowrate_dict_exact}

        pressures_branch_real.append(next_pressure_branch.detach().numpy())
        flowrates_branch_real.append(next_flowrate_branch.detach().numpy())

        pressure_dict = {'branch': torch.from_numpy(np.expand_dims(p_branch,axis=1)),
                         'inlet': graph.nodes['inlet'].data['pressure_next'],
                         'outlet': graph.nodes['outlet'].data['pressure_next']}
        flowrate_dict = {'branch': torch.from_numpy(np.expand_dims(q_branch,axis=1)),
                         'inlet': graph.nodes['inlet'].data['flowrate_next'],
                         'outlet': graph.nodes['outlet'].data['flowrate_next']}

        new_state = {'pressure': pressure_dict, 'flowrate': flowrate_dict}

        pressures_branch_pred.append(np.expand_dims(p_branch,axis=1))
        flowrates_branch_pred.append(np.expand_dims(q_branch,axis=1))

        pred_states.append(new_state)
        real_states.append(exact_state)

    end = time.time()

    if print_time:
        print('Rollout time = {:.2f} s for {:.0f} timesteps'.format(end - start,
                                                                    len(times)))

    err_p = err_p_branch
    err_q = err_q_branch
    err_p_branch = err_p_branch
    err_q_branch = err_q_branch
    err_p = err_p
    err_q = err_q

    errors = {'p_branch': err_p_branch,
              'q_branch': err_q_branch,
              'p': err_p,
              'q': err_q,
              'norm_t': norm_t,
              'norm_p': norm_p,
              'norm_q': norm_q,
              'continuity': c_loss_total / total_flowrate}

    solutions = {'p_branch_real': pressures_branch_real,
                 'q_branch_real': flowrates_branch_real,
                 'p_branch_pred': pressures_branch_pred,
                 'q_branch_pred': flowrates_branch_pred}

    return errors, solutions, graph, true_graph
