
import sigopt
import sys

sys.path.append("../network")
sys.path.append("../tools")

import training as tr
import numpy as np
import time
import tester as test
import json
import preprocessing as pp
import io_utils as io
from preprocessing import DGL_Dataset

def log_checkpoint(loss):
    sigopt.log_checkpoint({'loss': loss})
    sigopt.log_metric(name="loss", value=loss)

if __name__ == "__main__":
    dataset_json = json.load(open(io.data_location() + 'normalized_graphs/dataset_list.json'))

    sigopt.params.setdefaults(
        latent_size_gnn=32,
        latent_size_mlp=64,
        learning_rate=0.001,
        lr_decay=0.999,
        momentum=0.0,
        process_iterations=1,
        hl_mlp=2,
        normalize=1,
        nepochs=30,
        batch_size=100,
        rate_noise=600,
        normalization='standard',
        optimizer='adam',
        label_normalization='min_max',
        continuity_coeff=-5,
        average_flowrate_training=0,
        weight_decay=1e-5,
        bc_coeff=-5,
        ncopies=4
    )
    network_params = {'infeat_nodes': 12,
                    'infeat_edges': 4,
                    'latent_size_gnn': sigopt.params.latent_size_gnn,
                    'latent_size_mlp': sigopt.params.latent_size_mlp,
                    'out_size': 2,
                    'process_iterations': sigopt.params.process_iterations,
                    'hl_mlp': sigopt.params.hl_mlp,
                    'normalize': sigopt.params.normalize,
                    'average_flowrate_training': sigopt.params.average_flowrate_training}
    train_params = {'learning_rate': sigopt.params.learning_rate,
                    'lr_decay': sigopt.params.lr_decay,
                    'momentum': sigopt.params.momentum,
                    'nepochs': sigopt.params.nepochs,
                    'batch_size': sigopt.params.batch_size,
                    'continuity_coeff': sigopt.params.continuity_coeff,
                    'bc_coeff': sigopt.params.bc_coeff,
                    'weight_decay': sigopt.params.weight_decay,
                    'rate_noise': sigopt.params.rate_noise,
                    'n_copies_models': sigopt.params.ncopies}

    start = time.time()
    gnn_model, history, dataset, \
    coefs_dict, out_fdr, parameters = tr.launch_training(dataset_json,
                                                         'adam', network_params, train_params,
                                                         checkpoint_fct = log_checkpoint)

    end = time.time()
    elapsed_time = end - start
    print('Training time = ' + str(elapsed_time))

    sigopt.log_metadata('folder', out_fdr)
    sigopt.log_metric(name="loss", value=history['train_loss'][1][-1])
    sigopt.log_metric(name="mae", value=history['train_metric'][1][-1])
    sigopt.log_metric(name="train_rollout", value=history['train_rollout_error'][1][-1])
    sigopt.log_metric(name="test_rollout", value=history['test_rollout_error'][1][-1])
    sigopt.log_metric(name="overfitting", value=np.abs(1-history['test_rollout_error'][1][-1]/
                      history['train_rollout_error'][1][-1]))
