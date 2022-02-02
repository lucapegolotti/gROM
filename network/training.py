import sys
import os

# this fixes a problem with openmp https://github.com/dmlc/xgboost/issues/1715
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

import dgl
import torch
import torch.distributed as dist
import preprocessing as pp
from graphnet import GraphNet
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

def mse(input, target):
    return ((input - target) ** 2).mean()

def mae(input, target, weight = None):
    if weight == None:
        return (torch.abs(input - target)).mean()
    return (weight * (torch.abs(input - target))).mean()

def generate_gnn_model(params_dict):
    return GraphNet(params_dict)

def evaluate_model(gnn_model, train_dataloader, loss, metric = None, optimizer = None):
    global_loss = 0
    global_metric = 0
    count = 0
    start = time.time()
    for batched_graph in train_dataloader:
        pred = gnn_model(batched_graph,
                         batched_graph.nodes['inner'].data['n_features'].float()).squeeze()

        loss_v = loss(pred, torch.reshape(batched_graph.nodes['inner'].data['n_labels'].float(),
                      pred.shape))

        global_loss = global_loss + loss_v.detach().numpy()

        if metric != None:
            metric_v = metric(pred, torch.reshape(batched_graph.nodes['inner'].data['n_labels'].float(),
                              pred.shape))

            global_metric = global_metric + metric_v.detach().numpy()

        if optimizer != None:
            optimizer.zero_grad()
            loss_v.backward()
            optimizer.step()
        count = count + 1

    end = time.time()

    return global_loss, count, end - start, global_metric

def train_gnn_model(gnn_model, train, validation, optimizer_name, train_params,
                    checkpoint_fct = None, dataset_params = None):
    # we only compute the coefs_dict on the train_dataset
    train_dataset, coefs_dict = pp.generate_dataset(train, dataset_params = dataset_params)
    validation_dataset, _ = pp.generate_dataset(validation, coefs_dict, dataset_params)

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

    gnn_model.module.set_normalization_coefs(coefs_dict)
    train_sampler = DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
    train_dataloader = GraphDataLoader(train_dataset, sampler=train_sampler,
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


    print("my rank = %d, world = %d, train_dataloader_len = %d."
          % (dist.get_rank(), dist.get_world_size(), len(train_dataloader)), flush=True)
    for epoch in range(nepochs):
        global_loss, count, elapsed, global_mae = evaluate_model(gnn_model, train_dataloader, mse, weighted_mae, optimizer)
        scheduler.step()
        print('{:.0f}\tloss = {:.4e} mae = {:.4e} time = {:.2f} s'.format(epoch,
                                                                      global_loss/count,
                                                                      global_mae/count,
                                                                      elapsed),
              flush=True)

        if checkpoint_fct != None:
            if epoch in chckp_epochs:
                checkpoint_fct(global_loss/count)

        if epoch >= 6:
            train_dataset.sample_noise(dataset_params['rate_noise'])

    # compute final loss
    global_loss, count, _, global_mae = evaluate_model(gnn_model, train_dataloader, mse, weighted_mae)
    print('\tFinal loss = {:.2e}\tfinal mae = {:.2e}'.format(global_loss/count,
                                                             global_mae/count))

    return gnn_model, train_dataloader, global_loss / count,  global_mae / count,coefs_dict, train_dataset

def create_directory(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        pass

def prepare_dataset(dataset_json):
    dataset = dataset_json['dataset']
    random.shuffle(dataset)
    ndata = len(dataset)

    train_perc = dataset_json['training']
    valid_perc = dataset_json['validation']

    train, validation, test = np.split(dataset, [int(ndata*train_perc), \
                                                 int(ndata*(valid_perc + train_perc))])

    return train, validation, test


def launch_training(dataset_json, optimizer_name, params_dict,
                    train_params, plot_validation = True, checkpoint_fct = None,
                    dataset_params = None):
    now = datetime.now()
    train, validation, test = prepare_dataset(dataset_json)
    gnn_model = generate_gnn_model(params_dict)
    gnn_model = torch.nn.parallel.DistributedDataParallel(gnn_model)
    folder = 'models/' + now.strftime("%d.%m.%Y_%H.%M.%S")
    if dist.get_rank() == 0:
        pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
        torch.save(gnn_model.state_dict(), folder + '/initial_gnn.pms')
    gnn_model, train_loader, loss, mae, coefs_dict, dataset = train_gnn_model(gnn_model,
                                                                            train,
                                                                            validation,
                                                                            optimizer_name,
                                                                            train_params,
                                                                            checkpoint_fct,
                                                                            dataset_params)

    split = {'train': train.tolist(), 'validation': validation.tolist(), 'test': test.tolist()}

    if dist.get_rank() == 0:
        torch.save(gnn_model.state_dict(), folder + '/trained_gnn.pms')

    dataset_params['split'] = split

    coefs = {'features': coefs_dict,
             'labels': dataset.label_coefs}

    parameters = {'hyperparameters': params_dict,
                  'train_parameters': train_params,
                  'dataset_parameters': dataset_params,
                  'normalization_coefficients': coefs}

    def default(obj):
        if isinstance(obj, torch.Tensor):
            return default(obj.detach().numpy())
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        print(obj)
        raise TypeError('Not serializable')

    if dist.get_rank() == 0:
        with open(folder + '/parameters.json', 'w') as outfile:
            json.dump(parameters, outfile, default=default)
    return gnn_model, loss, mae, dataset, coefs_dict, folder, parameters

if __name__ == "__main__":
    dist.init_process_group(backend='mpi')
    print("my rank = %d, world = %d." % (dist.get_rank(), dist.get_world_size()), flush=True)
    dataset_json = json.load(open('training_dataset.json'))

    params_dict = {'infeat_nodes': 12,
                   'infeat_edges': 4,
                   'latent_size_gnn': 18,
                   'latent_size_mlp': 64,
                   'out_size': 2,
                   'process_iterations': 3,
                   'hl_mlp': 1,
                   'normalize': 1}
    train_params = {'learning_rate': 0.008223127794360673,
                    'weight_decay': 0.36984122162067234,
                    'momentum': 0.0,
                    'batch_size': 359,
                    'nepochs': 30}
    dataset_params = {'normalization': 'standard',
                      'rate_noise': 0.006,
                      'label_normalization': 'min_max'}

    start = time.time()
    launch_training(dataset_json,  'adam', params_dict, train_params,
                    checkpoint_fct = None, dataset_params = dataset_params)

    end = time.time()
    elapsed_time = end - start
    print('Training time = ' + str(elapsed_time))
