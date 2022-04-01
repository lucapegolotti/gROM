import sys
import os

# this fixes a problem with openmp https://github.com/dmlc/xgboost/issues/1715
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

sys.path.append("../tools")

import dgl
import torch
import torch.distributed as dist
import preprocessing as pp
from graphnet import GraphNet
from dgl.data import DGLDataset
from preprocessing import DGL_Dataset
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
from dgl.data.utils import load_graphs
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from datetime import datetime
import random
import time
import json
import pathlib
from rollout import rollout
import plot_tools as ptools
import pickle

# try to use mse + param * torch.abs((input - target)).mean()
def mse(input, target):
    return ((input - target) ** 2).mean()

def mae(input, target, weight = None):
    if weight == None:
        return (torch.abs(input - target)).mean()
    return (weight * (torch.abs(input - target))).mean()

def generate_gnn_model(params_dict):
    return GraphNet(params_dict)

def evaluate_model(gnn_model, train_dataloader, loss, metric = None,
                   optimizer = None, continuity_coeff = 0.0,
                   bc_coeff = 0.0,
                   validation_dataloader = None,
                   train = True):

    try:
        average_flowrate = gnn_model.module.params['average_flowrate_training']
    except AttributeError:
        average_flowrate = gnn_model.params['average_flowrate_training']
    label_coefs = train_dataloader.dataloader.dataset.label_coefs
    coefs_dict = train_dataloader.dataloader.dataset.coefs_dict

    # this is to turn on or off dropout layers
    if train:
        gnn_model.train()
    else:
        gnn_model.eval()

    def loop_over(dataloader, c_optimizer = None):
        global_loss = 0
        global_metric = 0
        count = 0
        c_loss_global = 0
        for batched_graph in dataloader:
            pred_branch, pred_junction = gnn_model(batched_graph, \
                                                   batched_graph.nodes['branch'].data['n_features'].float(), \
                                                   batched_graph.nodes['junction'].data['n_features'].float())

            pred_branch = pred_branch.squeeze()
            pred_junction = pred_junction.squeeze()

            train_on_junctions = True

            if train_on_junctions:
                pred = torch.cat((pred_branch, pred_junction), 0)
                label = torch.cat((batched_graph.nodes['branch'].data['n_labels'].float(),
                                   batched_graph.nodes['junction'].data['n_labels'].float()), 0)
            else:
                pred = pred_branch
                label = batched_graph.nodes['branch'].data['n_labels'].float()

            try:
                if continuity_coeff > 1e-5:
                    c_loss = gnn_model.module.compute_continuity_loss(batched_graph, pred_branch, pred_junction, label_coefs, coefs_dict)
                bc_loss = gnn_model.module.compute_bc_loss(batched_graph,
                                                    batched_graph.nodes['branch'].data['n_features'],
                                                    batched_graph.nodes['junction'].data['n_features'],
                                                    batched_graph.nodes['inlet'].data['n_features'],
                                                    batched_graph.nodes['outlet'].data['n_features'],
                                                    pred_branch,
                                                    pred_junction,
                                                    label_coefs)
            except AttributeError:
                if continuity_coeff > 1e-5:
                    c_loss = gnn_model.compute_continuity_loss(batched_graph, pred_branch, pred_junction, label_coefs, coefs_dict)
                bc_loss = gnn_model.compute_bc_loss(batched_graph,
                                                    batched_graph.nodes['branch'].data['n_features'],
                                                    batched_graph.nodes['junction'].data['n_features'],
                                                    batched_graph.nodes['inlet'].data['n_features'],
                                                    batched_graph.nodes['outlet'].data['n_features'],
                                                    pred_branch,
                                                    pred_junction,
                                                    label_coefs)

            loss_v = loss(pred, label)
            if continuity_coeff > 1e-5:
                # real = gnn_model.compute_continuity_loss(batched_graph, batched_graph.nodes['inner'].data['n_labels'], label_coefs, coefs_dict)
                # print(real)
                loss_v =  loss_v + c_loss * continuity_coeff
                c_loss_global = c_loss_global + c_loss

            if bc_coeff > 1e-5:
                loss_v =  loss_v + bc_loss * bc_coeff

            global_loss = global_loss + loss_v.detach().numpy()

            if metric != None:
                metric_v = metric(pred, label)

                global_metric = global_metric + metric_v.detach().numpy()

            if c_optimizer != None:
               optimizer.zero_grad()
               loss_v.backward()
               optimizer.step()
            count = count + 1

        return {'global_loss': global_loss, 'count': count,
                'continuity_loss': c_loss_global, 'global_metric': global_metric}

    validation_results = None
    start = time.time()
    if validation_dataloader:
        validation_results = loop_over(validation_dataloader)
    train_results = loop_over(train_dataloader, optimizer)
    end = time.time()

    return train_results, validation_results, end - start

def train_gnn_model(gnn_model, optimizer_name, train_params,
                    checkpoint_fct = None):

    train_dataset = pickle.load(open('datasets/d0/train.dts', 'rb'))
    coefs_dict = train_dataset.coefs_dict
    dataset_params = train_dataset.dataset_params
    print('Dataset contains {:.0f} graphs'.format(len(train_dataset)), flush=True)

    validation_dataset = pickle.load(open('datasets/d0/test.dts', 'rb'))

    if dataset_params['label_normalization'] == 'min_max':
        def weighted_mae(input, target):
            label_coefs = train_dataset.label_coefs
            shapein = input.shape
            weight = torch.ones(shapein)
            for i in range(shapein[1]):
                weight[:,i] = (label_coefs['max'][i] - label_coefs['min'][i])

            return mae(input, target, weight)
    elif dataset_params['label_normalization'] == 'standard':
        def weighted_mae(input, target):
            label_coefs = train_dataset.label_coefs
            shapein = input.shape
            weight = torch.ones(shapein)
            for i in range(shapein[1]):
                weight[:,i] = label_coefs['std'][i]

            return mae(input, target, weight)
    else:
        def weighted_mae(input, target):
            return mae(input, target)

    try:
        gnn_model.module.set_normalization_coefs(coefs_dict)
        train_sampler = DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
        validation_sampler = DistributedSampler(validation_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
    except AttributeError:
        gnn_model.set_normalization_coefs(coefs_dict)
        num_examples = len(train_dataset)
        num_train = int(num_examples)
        train_sampler = SubsetRandomSampler(torch.arange(num_train))
        num_examples = len(validation_dataset)
        num_validation = int(num_examples)
        validation_sampler = SubsetRandomSampler(torch.arange(num_validation))

    train_dataloader = GraphDataLoader(train_dataset, sampler=train_sampler,
                                       batch_size=train_params['batch_size'],
                                       drop_last=False)
    validation_dataloader = GraphDataLoader(validation_dataset, sampler=validation_sampler,
                                            batch_size=train_params['batch_size'],
                                            drop_last=False)
    try:
        print("my rank = %d, world = %d, train_dataloader_len = %d."
              % (dist.get_rank(), dist.get_world_size(), len(train_dataloader)), flush=True)
    except RuntimeError:
        pass

    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(gnn_model.parameters(),
                                     train_params['learning_rate'],
                                     weight_decay=train_params['weight_decay'])
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(gnn_model.parameters(),
                                    train_params['learning_rate'],
                                    momentum=train_params['momentum'])
    else:
        raise ValueError('Optimizer ' + optimizer_name + ' not implemented')

    nepochs = train_params['nepochs']
    scheduler_name = 'cosine'
    if scheduler_name == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                           gamma=train_params['weight_decay'])
    elif scheduler_name == 'cosine':
        eta_min = train_params['learning_rate'] * train_params['weight_decay']
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=nepochs,
                                                               eta_min=eta_min)

    if checkpoint_fct != None:
        # 200 is the maximum number of sigopt checkpoint
        chckp_epochs = list(np.floor(np.linspace(0, nepochs, 200)))

    noise_start = nepochs / 10
    ramp_epochs = int(np.floor(nepochs/2))

    history = {}
    history['train_loss'] = [[],[]]
    history['train_metric'] = [[],[]]
    history['train_rollout_error'] = [[],[]]
    history['validation_loss'] = [[],[]]
    history['validation_metric'] = [[],[]]
    history['validation_rollout_error'] = [[],[]]

    for epoch in range(nepochs):
        if epoch >= noise_start:
            if epoch >= noise_start + ramp_epochs:
                train_dataset.sample_noise(dataset_params['rate_noise'])
                validation_dataset.sample_noise(dataset_params['rate_noise'])
            else:
                train_dataset.sample_noise(dataset_params['rate_noise'] * (epoch - noise_start)/ramp_epochs)
                validation_dataset.sample_noise(dataset_params['rate_noise'] * (epoch - noise_start)/ramp_epochs)

        train_results, val_results, elapsed = evaluate_model(gnn_model, train_dataloader,
                                                             mse, weighted_mae, optimizer,
                                                             validation_dataloader = validation_dataloader,
                                                             continuity_coeff = 10**train_params['continuity_coeff'],
                                                             bc_coeff = 10**train_params['bc_coeff'])

        msg = '{:.0f}\t'.format(epoch)
        msg = msg + 'train_loss = {:.2e} '.format(train_results['global_loss']/train_results['count'])
        msg = msg + 'train_mae = {:.2e} '.format(train_results['global_metric']/train_results['count'])
        msg = msg + 'train_con_loss = {:.2e} '.format(train_results['continuity_loss']/train_results['count'])
        msg = msg + 'val_loss = {:.2e} '.format(val_results['global_loss']/val_results['count'])
        msg = msg + 'val_mae = {:.2e} '.format(val_results['global_metric']/val_results['count'])
        msg = msg + 'val_con_loss = {:.2e} '.format(val_results['continuity_loss']/val_results['count'])
        msg = msg + 'time = {:.2f} s'.format(elapsed)

        history['train_loss'][0].append(epoch)
        history['train_loss'][1].append(train_results['global_loss']/train_results['count'])
        history['train_metric'][0].append(epoch)
        history['train_metric'][1].append(train_results['global_metric']/train_results['count'])
        history['validation_loss'][0].append(epoch)
        history['validation_loss'][1].append(val_results['global_loss']/val_results['count'])
        history['validation_metric'][0].append(epoch)
        history['validation_metric'][1].append(val_results['global_metric']/val_results['count'])

        print(msg, flush=True)

        if checkpoint_fct != None:
            if epoch in chckp_epochs:
                checkpoint_fct(global_loss/count)

    # # compute final loss
    # train_results, val_results, elapsed = evaluate_model(gnn_model, train_dataloader,
    #                                                      mse, weighted_mae,
    #                                                      validation_dataloader = validation_dataloader,
    #                                                      continuity_coeff = 10**train_params['continuity_coeff'])
    # msg = 'end\t'
    # msg = msg + 'train_loss = {:.2e} '.format(train_results['global_loss']/train_results['count'])
    # msg = msg + 'train_mae = {:.2e} '.format(train_results['global_metric']/train_results['count'])
    # msg = msg + 'train_con_loss = {:.2e} '.format(train_results['continuity_loss']/train_results['count'])
    # msg = msg + 'val_loss = {:.2e} '.format(val_results['global_loss']/val_results['count'])
    # msg = msg + 'val_mae = {:.2e} '.format(val_results['global_metric']/val_results['count'])
    # msg = msg + 'val_con_loss = {:.2e} '.format(val_results['continuity_loss']/val_results['count'])
    # msg = msg + 'time = {:.2f} s'.format(elapsed)

    print(msg, flush=True)

    return gnn_model, train_dataloader, train_results['global_loss']/train_results['count'], \
           train_results['global_metric']/train_results['count'], coefs_dict, train_dataset, \
           dataset_params, history

def launch_training(dataset_json, optimizer_name, params_dict,
                    train_params, plot_validation = True, checkpoint_fct = None):
    now = datetime.now()
    folder = 'models/' + now.strftime("%d.%m.%Y_%H.%M.%S")
    def save_model(filename):
        try:
            # we call the method on .module because otherwise the pms file
            # cannot be read serially
            torch.save(gnn_model.module.state_dict(), folder + '/' + filename)
        except AttributeError:
            torch.save(gnn_model.state_dict(),  folder + '/' + filename)

    gnn_model = generate_gnn_model(params_dict)
    save_data = True
    # check if MPI is supported
    try:
        gnn_model = torch.nn.parallel.DistributedDataParallel(gnn_model,
                                                              find_unused_parameters=True)
        save_data = (dist.get_rank() == 0)
    except RuntimeError:
        pass
    if save_data:
        pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
        if save_data:
            save_model('initial_gnn.pms')

    gnn_model, train_loader, loss, mae, \
    coefs_dict, dataset, dataset_params, history = train_gnn_model(gnn_model,
                                                                   optimizer_name,
                                                                   train_params,
                                                                   checkpoint_fct)

    if save_data:
        save_model('trained_gnn.pms')

    coefs = {'features': coefs_dict,
             'labels': dataset.label_coefs}

    parameters = {'hyperparameters': params_dict,
                  'train_parameters': train_params,
                  'dataset_parameters': dataset_params,
                  'normalization_coefficients': coefs}

    if save_data:
        ptools.plot_history(history['train_loss'],
                        history['validation_loss'],
                        'loss', folder)

        ptools.plot_history(history['train_metric'],
                            history['validation_metric'],
                            'mae', folder)

    def default(obj):
        if isinstance(obj, torch.Tensor):
            return default(obj.detach().numpy())
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        print(obj)
        raise TypeError('Not serializable')

    if save_data:
        with open(folder + '/parameters.json', 'w') as outfile:
            json.dump(parameters, outfile, default=default, indent=4)
    return gnn_model, loss, mae, dataset, coefs_dict, folder, parameters

if __name__ == "__main__":
    try:
        dist.init_process_group(backend='mpi')
        print("my rank = %d, world = %d." % (dist.get_rank(), dist.get_world_size()), flush=True)
    except RuntimeError:
        print("MPI not supported. Running serially.")

    dataset_json = json.load(open('../graphs/normalized_data/dataset_list.json'))

    # params_dict = {'infeat_nodes': 13,
    #                'infeat_edges': 4,
    #                'latent_size_gnn': 16,
    #                'latent_size_mlp': 64,
    #                'out_size': 2,
    #                'process_iterations': 3,
    #                'hl_mlp': 1,
    #                'normalize': 1,
    #                'average_flowrate_training': 0}
    params_dict = {'infeat_nodes': 27,
                   'infeat_edges': 4,
                   'latent_size_gnn': 8,
                   'latent_size_mlp': 32,
                   'out_size': 2,
                   'process_iterations': 3,
                   'hl_mlp': 1,
                   'normalize': 1,
                   'average_flowrate_training': 0}
    train_params = {'learning_rate': 0.0008,
                    'weight_decay': 0.6,
                    'momentum': 0.0,
                    'batch_size': 1,
                    'nepochs': 400,
                    'continuity_coeff': -5,
                    'bc_coeff': -5}

    start = time.time()
    launch_training(dataset_json,  'adam', params_dict, train_params,
                    checkpoint_fct = None)

    end = time.time()
    elapsed_time = end - start
    print('Training time = ' + str(elapsed_time))
