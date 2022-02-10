
import sigopt
import sys

sys.path.append("../network")

import training as tr
import numpy as np
import time
import tester as test
import json
import preprocessing as pp

def log_checkpoint(loss):
    sigopt.log_checkpoint({'loss': loss})
    sigopt.log_metric(name="loss", value=loss)

if __name__ == "__main__":
    dataset_json = json.load(open('../network/training_dataset.json'))

    sigopt.params.setdefaults(
        latent_size_gnn=32,
        latent_size_mlp=64,
        learning_rate=0.001,
        weight_decay=0.999,
        momentum=0.0,
        process_iterations=1,
        hl_mlp=2,
        normalize=1,
        nepochs=400,
        batch_size=100,
        rate_noise=0.1,
        random_walks=0,
        normalization='standard',
        optimizer='adam',
        label_normalization='min_max',
        continuity_coeff=-5,
        average_flowrate_training=0
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
                    'weight_decay': sigopt.params.weight_decay,
                    'momentum': sigopt.params.momentum,
                    'nepochs': sigopt.params.nepochs,
                    'batch_size': sigopt.params.batch_size,
                    'continuity_coeff': sigopt.params.continuity_coeff}
    dataset_params = {'normalization': sigopt.params.normalization,
                      'rate_noise': sigopt.params.rate_noise,
                      'label_normalization': sigopt.params.label_normalization}

    start = time.time()
    gnn_model, loss, mae, dataset, \
    coefs_dict, out_fdr, parameters = tr.launch_training(dataset_json,
                                                         'adam', network_params, train_params,
                                                         checkpoint_fct = log_checkpoint,
                                                         dataset_params = dataset_params)

    end = time.time()
    elapsed_time = end - start
    print('Training time = ' + str(elapsed_time))

    dataset = pp.generate_dataset(parameters['dataset_parameters']['split']['validation'],
                                  dataset_params = parameters['dataset_parameters'],
                                  coefs_dict = parameters['normalization_coefficients']['features'])

    err_p, err_q, global_err = test.test_rollout(gnn_model, parameters,
                                                 dataset,
                                                 index_graph = 0,
                                                 split = 'validation',
                                                 out_folder = out_fdr)

    sigopt.log_metadata('folder', out_fdr)
    sigopt.log_metric(name="loss", value=loss)
    sigopt.log_metric(name="mae", value=mae)

    if err_p != err_p or err_p > 1e10:
        sys.exit()

    if err_q != err_q or err_q > 1e10:
        sys.exit()

    if global_err != global_err or global_err > 1e10:
        sys.exit()

    sigopt.log_metric(name="error_pressure", value=err_p)
    sigopt.log_metric(name="error_flowrate", value=err_q)
    sigopt.log_metric(name="global_error", value=global_err)
    sigopt.log_metric(name="training_time", value=elapsed_time)
