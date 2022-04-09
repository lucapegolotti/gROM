import sys
import os
from random import sample
import numpy as np
import training as tr
import torch.distributed as dist
import random
import time
import json
import io_utils as io
from preprocessing import DGL_Dataset
import pickle
from os.path import exists

# np.random.seed(10)
# random.seed(10)

def float_in_range(minv, maxv):
    return minv + (maxv - minv) * np.random.random(1)[0]

def sample_parameters():
    params_dict = {'latent_size_gnn': np.random.randint(10, 32),
                   'latent_size_mlp': np.random.randint(32, 80),
                   'process_iterations': np.random.randint(1, 3),
                   'hl_mlp': np.random.randint(1, 2),
                   'normalize': 1,
                   'average_flowrate_training': 0}
    train_params = {'learning_rate': float_in_range(0.0001, 0.05),
                    'momentum': 0,
                    'batch_size': np.random.randint(10, 300),
                    'lr_decay': float_in_range(0.01, 0.9),
                    'nepochs': 50,
                    'continuity_coeff': -3,
                    'bc_coeff': -5,
                    'weight_decay': 1e-5,
                    'rate_noise': float_in_range(10, 1000),
                    'n_copies_models': sample([1,8,64,128], 1)[0]}

    return params_dict, train_params

def load_histories(folder):
    hdir = os.listdir(folder)
    histories = {}
    for cdir in hdir:
        if os.path.isdir(folder + '/' + cdir):
            if  exists(folder + '/' + cdir + '/history.bnr'):
                histories[cdir] = pickle.load(open(folder + '/' + cdir + '/history.bnr', 'rb'))

    return histories

def print_results(folder = 'hpo'):
    histories = load_histories(folder)

    fields = ['train_rollout_error', 'test_rollout_error', 'train_loss']

    for fn in histories:
        msg = fn + ': '
        for field in fields:
            msg = msg + field +' {:.3f} '.format(histories[fn][field][1][-1])

        msg = msg +'overfitting {:.3f} '.format(histories[fn]['test_rollout_error'][1][-1]/
                                                histories[fn]['train_rollout_error'][1][-1])

        print(msg, flush = True)

if __name__ == "__main__":
    try:
        dist.init_process_group(backend='mpi')
        print("my rank = %d, world = %d." % (dist.get_rank(), dist.get_world_size()), flush=True)
    except RuntimeError:
        print("MPI not supported. Running serially.")

    dataset_json = json.load(open(io.data_location() + 'normalized_graphs/dataset_list.json'))

    nopt = 40
    start = time.time()
    for i in range(nopt):
        print('HPO iteration ' + str(i))
        params_dict, train_params = sample_parameters()
        print('hyperparameters = ')
        print(params_dict)
        print('train parameters = ')
        print(train_params)
        tr.launch_training(dataset_json, 'adam', params_dict, train_params,
                        checkpoint_fct = None,
                        out_dir = 'hpo/')

    end = time.time()
    elapsed_time = end - start
    print('Training time = ' + str(elapsed_time))
