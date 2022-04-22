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
import io_utils as io
import argparse

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
                   test_dataloader = None,
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
            start = time.time()
            pred_branch, pred_junction = gnn_model(batched_graph, \
                                                   batched_graph.nodes['branch'].data['n_features'].float(), \
                                                   batched_graph.nodes['junction'].data['n_features'].float())

            pred_branch = pred_branch.squeeze()
            pred_junction = pred_junction.squeeze()
            end = time.time()
            # print('pred = {:.3f} s'.format(end - start))

            train_on_junctions = True
            start = time.time()

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

            end = time.time()
            # print('losses = {:.3f} s'.format(end - start))

            start = time.time()
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

            end = time.time()
            # print('boh = {:.3f} s'.format(end - start))

            start = time.time()
            if c_optimizer != None:
               optimizer.zero_grad()
               loss_v.backward()
               optimizer.step()
            end = time.time()
            # print('optimizer = {:.3f} s'.format(end - start))
            count = count + 1

        return {'global_loss': global_loss, 'count': count,
                'continuity_loss': c_loss_global, 'global_metric': global_metric}

    test_results = None
    start = time.time()
    if test_dataloader:
        test_results = loop_over(test_dataloader)
    train_results = loop_over(train_dataloader, optimizer)
    end = time.time()

    return train_results, test_results, end - start

def train_gnn_model(gnn_model, optimizer_name, parameters,
                    checkpoint_fct = None):

    train_params = parameters['train_parameters']
    train_dataset = pickle.load(open(io.data_location() + 'datasets/d0/train.dts', 'rb'))
    train_dataset.set_number_model_copies(train_params['n_copies_models'])
    train_dataset_rollout = pickle.load(open(io.data_location() + 'datasets/d0/train_not_augmented.dts', 'rb'))
    coefs_dict = train_dataset.coefs_dict
    dataset_params = train_dataset.dataset_params
    coefs = {'features': coefs_dict,
             'labels': train_dataset.label_coefs}
    parameters['dataset_parameters'] = dataset_params
    parameters['normalization_coefficients'] = coefs

    print('Dataset contains {:.0f} graphs'.format(len(train_dataset)), flush=True)

    test_dataset = pickle.load(open(io.data_location() + 'datasets/d0/test.dts', 'rb'))

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
        test_sampler = DistributedSampler(test_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
    except AttributeError:
        gnn_model.set_normalization_coefs(coefs_dict)
        num_examples = len(train_dataset)
        num_train = int(num_examples)
        train_sampler = SubsetRandomSampler(torch.arange(num_train))
        num_examples = len(test_dataset)
        num_validation = int(num_examples)
        test_sampler = SubsetRandomSampler(torch.arange(num_validation))

    train_dataloader = GraphDataLoader(train_dataset, sampler=train_sampler,
                                       batch_size=train_params['batch_size'],
                                       drop_last=False)
    test_dataloader = GraphDataLoader(test_dataset, sampler=test_sampler,
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
        eta_min = train_params['learning_rate'] * train_params['lr_decay']
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=nepochs,
                                                               eta_min=eta_min)

    if checkpoint_fct != None:
        # 200 is the maximum number of sigopt checkpoint
        chckp_epochs = list(np.floor(np.linspace(0, nepochs, 200)))

    history = {}
    history['train_loss'] = [[],[]]
    history['train_metric'] = [[],[]]
    history['train_rollout_error'] = [[],[]]
    history['test_loss'] = [[],[]]
    history['test_metric'] = [[],[]]
    history['test_rollout_error'] = [[],[]]

    for epoch in range(nepochs):
        train_dataset.sample_noise(parameters['train_parameters']['rate_noise'])
        test_dataset.sample_noise(parameters['train_parameters']['rate_noise'])

        train_results, val_results, elapsed = evaluate_model(gnn_model, train_dataloader,
                                                             mse, weighted_mae, optimizer,
                                                             test_dataloader = test_dataloader,
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

        print(msg, flush=True)

        if epoch % 5 == 0 or epoch == nepochs-1:
            nrollout = 2
            random.seed(10)
            indices = random.sample(list(range(len(parameters['dataset_parameters']['split']['train']))), nrollout)

            error_branches_train = 0
            error_junctions_train = 0
            error_global_train = 0
            for index in indices:
                errors, _, _, _ = rollout(gnn_model, parameters,
                                          train_dataset_rollout,
                                          index_graph = index,
                                          split = 'train',
                                          print_time = False)

                error_branches_train = error_branches_train + \
                                       (errors['p_branch'] + errors['q_branch']) / errors['norm_t']

                error_junctions_train = error_junctions_train + \
                                        (errors['p_junction'] + errors['q_junction']) / errors['norm_t']

                error_global_train = error_global_train + \
                                     (errors['p'] + errors['q']) / errors['norm_t']

            error_branches_train = error_branches_train / nrollout
            error_junctions_train = error_junctions_train / nrollout
            error_global_train = error_global_train / nrollout

            history['train_rollout_error'][0].append(epoch)
            history['train_rollout_error'][1].append(np.sqrt(error_global_train))

            msg = 'Rollout train: '
            msg = msg + 'rollout_error_branch = {:.5e} '.format(error_branches_train)
            msg = msg + 'rollout_error_junctions = {:.5e} '.format(error_junctions_train)
            msg = msg + 'rollout_error_global = {:.5e} '.format(error_global_train)
            msg = msg + 'sqrt = {:.5e} '.format(np.sqrt(error_global_train))

            print(msg)

            indices = random.sample(list(range(len(parameters['dataset_parameters']['split']['test']))), nrollout)

            error_branches_test = 0
            error_junctions_test = 0
            error_global_test = 0
            for index in indices:
                errors, _, _, _ = rollout(gnn_model, parameters,
                                          test_dataset,
                                          index_graph = index,
                                          split = 'test',
                                          print_time = False)

                error_branches_test = error_branches_test + \
                                      (errors['p_branch'] + errors['q_branch']) / errors['norm_t']

                error_junctions_test = error_junctions_test + \
                                       (errors['p_junction'] + errors['q_junction']) / errors['norm_t']

                error_global_test = error_global_test + \
                                    (errors['p'] + errors['q']) / errors['norm_t']

            error_branches_test = error_branches_test / nrollout
            error_junctions_test = error_junctions_test / nrollout
            error_global_test = error_global_test / nrollout

            history['test_rollout_error'][0].append(epoch)
            history['test_rollout_error'][1].append(np.sqrt(error_global_test))

            msg = 'Rollout test: '
            msg = msg + 'rollout_error_branch = {:.5e} '.format(error_branches_test)
            msg = msg + 'rollout_error_junctions = {:.5e} '.format(error_junctions_test)
            msg = msg + 'rollout_error_global = {:.5e} '.format(error_global_test)
            msg = msg + 'sqrt = {:.5e} '.format(np.sqrt(error_global_test))

            print(msg)

        if epoch == nepochs-1:
            print("Final rollout error on test = {:.5}".format(error_global_test))

        # def get_lr(optimizer):
        #     for param_group in optimizer.param_groups:
        #         return param_group['lr']
        # print(get_lr(optimizer))

        history['train_loss'][0].append(epoch)
        history['train_loss'][1].append(train_results['global_loss']/train_results['count'])
        history['train_metric'][0].append(epoch)
        history['train_metric'][1].append(train_results['global_metric']/train_results['count'])
        history['test_loss'][0].append(epoch)
        history['test_loss'][1].append(val_results['global_loss']/val_results['count'])
        history['test_metric'][0].append(epoch)
        history['test_metric'][1].append(val_results['global_metric']/val_results['count'])

        if checkpoint_fct != None:
            if epoch in chckp_epochs:
                checkpoint_fct(history['test_rollout_error'][1][-1])

        scheduler.step()

    return gnn_model, train_dataloader, train_results['global_loss']/train_results['count'], \
           train_results['global_metric']/train_results['count'], coefs_dict, train_dataset, history

def launch_training(dataset_json, optimizer_name, params_dict,
                    train_params, plot_validation = True, checkpoint_fct = None,
                    out_dir = 'models/'):
    now = datetime.now()
    folder = out_dir + now.strftime("%d.%m.%Y_%H.%M.%S")
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
        gnn_model = torch.nn.parallel.DistributedDataParallel(gnn_model)
        save_data = (dist.get_rank() == 0)
    except RuntimeError:
        pass
    if save_data:
        pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
        if save_data:
            save_model('initial_gnn.pms')

    parameters = {'hyperparameters': params_dict,
                  'train_parameters': train_params}

    def default(obj):
        if isinstance(obj, torch.Tensor):
            return default(obj.detach().numpy())
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        print(obj)
        raise TypeError('Not serializable')

    gnn_model, train_loader, loss, mae, \
    coefs_dict, dataset, history = train_gnn_model(gnn_model,
                                                   optimizer_name,
                                                   parameters,
                                                   checkpoint_fct)

    if save_data:
        ptools.plot_history(history['train_loss'],
                        history['test_loss'],
                        'loss', folder)

        ptools.plot_history(history['train_metric'],
                            history['test_metric'],
                            'mae', folder)

        ptools.plot_history(history['train_rollout_error'],
                            history['test_rollout_error'],
                            'rollout', folder)

    if save_data:
        with open(folder + '/parameters.json', 'w') as outfile:
            json.dump(parameters, outfile, default=default, indent=4)

    if save_data:
        save_model('trained_gnn.pms')

    if save_data:
        with open(folder + '/history.bnr', 'wb') as outfile:
            pickle.dump(history, outfile)

    return gnn_model, history, dataset, coefs_dict, folder, parameters

if __name__ == "__main__":
    try:
        dist.init_process_group(backend='mpi')
        print("my rank = %d, world = %d." % (dist.get_rank(), dist.get_world_size()), flush=True)
    except RuntimeError:
        print("MPI not supported. Running serially.")

    dataset_json = json.load(open(io.data_location() + 'normalized_graphs/dataset_list.json'))

    parser = argparse.ArgumentParser(description='gROM SimVascular Project.')

    parser.add_argument('--bs', help='batch size', type=int, default=100)
    parser.add_argument('--epochs', help='total number of epochs', type=int, default=200)
    parser.add_argument('--lr_decay', help='learning rate decay', type=float, default=0.1)
    parser.add_argument('--lr', help='learning rate', type=float, default=0.008)
    parser.add_argument('--rate_noise', help='rate noise', type=float, default=600)
    parser.add_argument('--continuity_coeff', help='continuity coefficient', type=int, default=-3)
    parser.add_argument('--bc_coeff', help='boundary conditions coefficient', type=int, default=-5)
    parser.add_argument('--momentum', help='momentum', type=float, default=0)
    parser.add_argument('--weight_decay', help='weight decay for l2 regularization', type=float, default=1e-5)
    parser.add_argument('--ls_gnn', help='latent size gnn', type=int, default=16)
    parser.add_argument('--ls_mlp', help='latent size mlps', type=int, default=64)
    parser.add_argument('--process_iterations', help='gnn layers', type=int, default=2)
    parser.add_argument('--hl_mlp', help='hidden layers mlps', type=int, default=1)
    parser.add_argument('--nmc', help='copies per model', type=int, default=1)

    args = parser.parse_args()

    params_dict = {'infeat_edges': 4,
                   'latent_size_gnn': args.ls_gnn,
                   'latent_size_mlp': args.ls_mlp,
                   'out_size': 2,
                   'process_iterations': args.process_iterations,
                   'hl_mlp': args.hl_mlp,
                   'normalize': 1,
                   'average_flowrate_training': 0}
    train_params = {'learning_rate': args.lr,
                    'momentum': args.momentum,
                    'batch_size': args.bs,
                    'lr_decay': args.lr_decay,
                    'nepochs': args.epochs,
                    'continuity_coeff': args.continuity_coeff,
                    'bc_coeff': args.bc_coeff,
                    'weight_decay': args.weight_decay,
                    'rate_noise': args.rate_noise,
                    'n_copies_models': args.nmc}

    start = time.time()
    launch_training(dataset_json, 'adam', params_dict, train_params,
                    checkpoint_fct = None)

    end = time.time()
    elapsed_time = end - start
    print('Training time = ' + str(elapsed_time))
