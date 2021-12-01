import sys
import os

# this fixes a problem with openmp https://github.com/dmlc/xgboost/issues/1715
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import dgl
import torch
import preprocessing as pp
from graphnet import GraphNet
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dgl.data.utils import load_graphs
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from datetime import datetime
import json

def weighted_mse_loss(input, target, weight):
    return (weight * (input - target) ** 2).mean()

def generate_gnn_model(params_dict):
    return GraphNet(params_dict)

def train_gnn_model(gnn_model, model_name, optimizer_name, train_params):
    dataset, coefs_dict = pp.generate_dataset(model_name,
                                              train_params['resample_freq_timesteps'])
    num_examples = len(dataset)
    num_train = int(num_examples)
    train_sampler = SubsetRandomSampler(torch.arange(num_train))
    train_dataloader = GraphDataLoader(dataset, sampler=train_sampler,
                                       batch_size=train_params['batch_size'],
                                       drop_last=False)

    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(gnn_model.parameters(),
                                     train_params['learning_rate'])
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(gnn_model.parameters(),
                                    train_params['learning_rate'],
                                    momentum=train_params['momentum'])
    else:
        raise ValueError('Optimizer ' + optimizerizer_name + ' not implemented')

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                       gamma=train_params['weight_decay'])
    nepochs = train_params['nepochs']
    for epoch in range(nepochs):
        print('ep = ' + str(epoch))
        global_loss = 0
        count = 0
        for batched_graph in train_dataloader:
            pred = gnn_model(batched_graph,
                             batched_graph.ndata['n_features'].float()).squeeze()
            weight = torch.ones(pred.shape)
            # mask out values corresponding to boundary conditions
            inlets = np.where(batched_graph.ndata['inlet_mask'].detach().numpy() == 1)[0]
            outlets = np.where(batched_graph.ndata['outlet_mask'].detach().numpy() == 1)[0]
            weight[inlets,1] = 0
            weight[outlets,0] = 0
            loss = weighted_mse_loss(pred,
                                     torch.reshape(batched_graph.ndata['n_labels'].float(),
                                     pred.shape), weight)
            global_loss = global_loss + loss.detach().numpy()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            count = count + 1
        scheduler.step()
        print('\tloss = ' + str(global_loss / count))

    return gnn_model, train_dataloader, global_loss / count, coefs_dict

def evaluate_error(model, model_name, train_dataloader, coefs_dict, do_plot, out_folder):
    it = iter(train_dataloader)
    batch = next(it)
    batched_graph = batch
    graph = dgl.unbatch(batched_graph)[0]

    true_graph = load_graphs('../dataset/data/' + model_name + '.grph')[0][0]
    times = pp.get_times(true_graph)
    # times = times[0:]
    initial_pressure = pp.min_max(true_graph.ndata['pressure_' + str(times[0])],
                                  coefs_dict['pressure'])
    initial_flowrate = pp.min_max(true_graph.ndata['flowrate_' + str(times[0])],
                                  coefs_dict['flowrate'])
    graph = pp.set_state(graph, initial_pressure, initial_flowrate)

    err_p = 0
    err_q = 0
    norm_p = 0
    norm_q = 0
    pressures_pred = []
    pressures_real = []
    flowrates_pred = []
    flowrates_real = []
    for tind in range(len(times)-1):
        t = times[tind]
        tp1 = times[tind+1]

        next_pressure = true_graph.ndata['pressure_' + str(tp1)]
        next_flowrate = true_graph.ndata['flowrate_' + str(tp1)]
        np_normalized = pp.min_max(next_pressure, coefs_dict['pressure'])
        nf_normalized = pp.min_max(next_flowrate, coefs_dict['flowrate'])
        graph = pp.set_bcs(graph, np_normalized, nf_normalized)
        pred = model(graph, graph.ndata['n_features'].float()).squeeze()

        dp = pp.invert_min_max(pred[:,0].detach().numpy(), coefs_dict['dp'])
        prev_p = pp.invert_min_max(graph.ndata['pressure'].detach().numpy().squeeze(), coefs_dict['pressure'])

        p = dp + prev_p
        # print(np.linalg.norm(p))
        pressures_pred.append(p)
        pressures_real.append(next_pressure.detach().numpy())

        dq = pp.invert_min_max(pred[:,1].detach().numpy(), coefs_dict['dq'])
        prev_q = pp.invert_min_max(graph.ndata['flowrate'].detach().numpy().squeeze(),
                                   coefs_dict['flowrate'])

        q = dq + prev_q

        flowrates_pred.append(q)
        flowrates_real.append(next_flowrate.detach().numpy())

        err_p = err_p + np.linalg.norm(p - next_pressure.detach().numpy())**2
        norm_p = norm_p + np.linalg.norm(next_pressure.detach().numpy())**2
        err_q = err_q + np.linalg.norm(q - next_flowrate.detach().numpy())**2
        norm_q = norm_q + np.linalg.norm(next_flowrate.detach().numpy())**2

        new_pressure = pp.min_max(p, coefs_dict['pressure'])
        new_flowrate = pp.min_max(q, coefs_dict['flowrate'])
        graph = pp.set_state(graph, torch.unsqueeze(torch.from_numpy(new_pressure),1),
                                    torch.unsqueeze(torch.from_numpy(new_flowrate),1))

    err_p = np.sqrt(err_p / norm_p)
    err_q = np.sqrt(err_q / norm_q)

    if do_plot:
        fig, ax = plt.subplots(2)
        line_pred_p, = ax[0].plot([],[],'r')
        line_real_p, = ax[0].plot([],[],'--b')
        line_pred_q, = ax[1].plot([],[],'r')
        line_real_q, = ax[1].plot([],[],'--b')

        def animation_frame(i):
            line_pred_p.set_xdata(range(0,len(pressures_pred[i])))
            line_pred_p.set_ydata(pressures_pred[i])
            line_real_p.set_xdata(range(0,len(pressures_pred[i])))
            line_real_p.set_ydata(pressures_real[i])
            line_pred_q.set_xdata(range(0,len(flowrates_pred[i])))
            line_pred_q.set_ydata(flowrates_pred[i])
            line_real_q.set_xdata(range(0,len(flowrates_pred[i])))
            line_real_q.set_ydata(flowrates_real[i])
            ax[0].set_xlim(0,len(pressures_pred[i]))
            ax[0].set_ylim(coefs_dict['pressure'][0],coefs_dict['pressure'][1])
            ax[1].set_xlim(0,len(flowrates_pred[i]))
            ax[1].set_ylim(coefs_dict['flowrate'][0],coefs_dict['flowrate'][1])
            return line_pred_p, line_real_p, line_pred_q, line_real_q

        anim = animation.FuncAnimation(fig, animation_frame,
                                       frames=len(pressures_pred),
                                       interval=20)
        writervideo = animation.FFMpegWriter(fps=60)
        anim.save(out_folder + '/plot.mp4', writer = writervideo)

    return np.sqrt(err_p), np.sqrt(err_q), np.sqrt(err_p + err_q)

def create_directory(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        pass

def launch_training(model_name, optimizer_name, params_dict,
                    train_params, plot_validation = True):
    create_directory('models')
    gnn_model = generate_gnn_model(params_dict)
    gnn_model, train_loader, loss, coefs_dict = train_gnn_model(gnn_model,
                                                                model_name,
                                                                optimizer_name,
                                                                train_params)

    now = datetime.now()
    dt_string = now.strftime("%d.%m.%Y_%H.%M.%S")
    folder = 'models/' + dt_string
    create_directory(folder)
    torch.save(gnn_model.state_dict(), folder + '/gnn.pms')
    json_params = json.dumps(params_dict, indent = 4)
    json_train = json.dumps(train_params, indent = 4)
    with open(folder + '/hparams.json', 'w') as outfile:
        json.dump(json_params, outfile)
    with open(folder + '/train.json', 'w') as outfile:
        json.dump(json_train, outfile)
    return gnn_model, loss, train_loader, coefs_dict, folder

if __name__ == "__main__":
    params_dict = {'infeat_nodes': 6,
                   'infeat_edges': 5,
                   'latent_size_gnn': 32,
                   'latent_size_mlp': 64,
                   'out_size': 2,
                   'process_iterations': 10,
                   'hl_mlp': 2,
                   'normalize': True}
    train_params = {'learning_rate': 0.05,
                    'weight_decay': 0.999,
                    'momentum': 0.0,
                    'resample_freq_timesteps': -1,
                    'batch_size': 10,
                    'nepochs': 10}

    gnn_model, _, train_dataloader, coefs_dict, out_fdr = launch_training(sys.argv[1],
                                                                          'sgd',
                                                                           params_dict,
                                                                           train_params)
    err_p, err_q, global_err = evaluate_error(gnn_model, sys.argv[1],
                                              train_dataloader,
                                              coefs_dict,
                                              do_plot = True,
                                              out_folder = out_fdr)

    print('Error pressure ' + str(err_p))
    print('Error flowrate ' + str(err_q))
    print('Global error ' + str(global_err))