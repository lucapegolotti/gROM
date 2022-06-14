import sys
import os

# this fixes a problem with openmp https://github.com/dmlc/xgboost/issues/1715
os.environ['KMP_DUPLICATE_LIB_OK']='True'

sys.path.append("../tools")

import dgl
import torch
import numpy as np
from dgl.data.utils import load_graphs
from dgl.data import DGLDataset
import generate_graphs as gg
import copy
import random
import pathlib
import json
import time
import matplotlib.pyplot as plt
import io_utils as io

def get_times(graph):
    times = []

    features = graph.nodes['branch'].data
    for feature in features:
        if 'pressure' in feature:
            ind  = feature.find('_')
            times.append(float(feature[ind+1:]))
    times.sort()

    return times

def get_actual_times(graph, coefs_dict, denormalize = True):
    times =np.array(get_times(graph))
    dt = graph.nodes['branch'].data['dt'][0].detach().numpy()

    if denormalize:
        dt = invert_normalize_function(dt, 'dt', coefs_dict)

    times = (times - times[0]) * dt

    return times

def free_fields(graph, times):
    def per_node_type(node_type):
        for t in times:
            del(graph.nodes[node_type].data['pressure_' + str(t)])
            del(graph.nodes[node_type].data['noise_p_' + str(t)])
            del(graph.nodes[node_type].data['flowrate_' + str(t)])
            del(graph.nodes[node_type].data['noise_q_' + str(t)])
    per_node_type('branch')
    per_node_type('junction')
    per_node_type('inlet')
    per_node_type('outlet')

def set_timestep(targetgraph, allgraph, t, tp1):
    def per_node_type(node_type):
        targetgraph.nodes[node_type].data['pressure'] = allgraph.nodes[node_type].data['pressure_' + str(t)] + \
                                                        allgraph.nodes[node_type].data['noise_p_' + str(t)]
        targetgraph.nodes[node_type].data['dp'] = allgraph.nodes[node_type].data['pressure_' + str(tp1)] - \
                                allgraph.nodes[node_type].data['pressure_' + str(t)] - \
                                allgraph.nodes[node_type].data['noise_p_' + str(t)]

        targetgraph.nodes[node_type].data['flowrate'] = allgraph.nodes[node_type].data['flowrate_' + str(t)] + \
                                      allgraph.nodes[node_type].data['noise_q_' + str(t)]
        targetgraph.nodes[node_type].data['dq'] = allgraph.nodes[node_type].data['flowrate_' + str(tp1)] - \
                                allgraph.nodes[node_type].data['flowrate_' + str(t)] - \
                                allgraph.nodes[node_type].data['noise_q_' + str(t)]

        targetgraph.nodes[node_type].data['pressure_next'] = allgraph.nodes[node_type].data['pressure_' + str(tp1)]
        targetgraph.nodes[node_type].data['flowrate_next'] = allgraph.nodes[node_type].data['flowrate_' + str(tp1)]

    per_node_type('branch')
    per_node_type('junction')
    per_node_type('inlet')
    per_node_type('outlet')

    # we add also current time to graph(for debugging()
    targetgraph.nodes['inlet'].data['time'] = torch.from_numpy(np.array([t]))

def min_max(field, bounds):
    ncomponents = bounds['min'].size
    if ncomponents == 1:
        scaling = (bounds['max'] - bounds['min'])
        if scaling < 1e-12:
            return field - bounds['min']
        return (field - bounds['min']) / scaling
    for i in range(ncomponents):
        scaling = bounds['max'][i] - bounds['min'][i]
        if scaling < 1e-12:
            field[:,i] = (field[:,i] - bounds['min'][i]) / scaling
    return field

def invert_min_max(field, bounds):
    return bounds['min'] + field * (bounds['max'] - bounds['min'])

def min_max_normalization(graph, fields, bounds_dict):
    def per_node_type(node_type):
        node_features = graph.nodes[node_type].data
        for feat in node_features:
            for field in fields:
                if field in feat:
                    if np.linalg.norm(np.min(graph.nodes[node_type].data[feat].detach().numpy()) - 0) > 1e-5 and \
                       np.linalg.norm(np.max(graph.nodes[node_type].data[feat].detach().numpy()) - 1) > 1e-5:
                           graph.nodes[node_type].data[feat] = min_max(graph.nodes[node_type].data[feat], bounds_dict[field]).float()

    def per_edge_type(edge_type):
        edge_features = graph.edges[edge_type].data
        for feat in edge_features:
            for field in fields:
                if field in feat:
                    if np.linalg.norm(np.min(graph.edges[edge_type].data[feat].detach().numpy()) - 0) > 1e-5 and \
                       np.linalg.norm(np.max(graph.edges[edge_type].data[feat].detach().numpy()) - 1) > 1e-5:
                           graph.edges[edge_type].data[feat] = min_max(graph.edges[edge_type].data[feat], bounds_dict[field]).float()

    per_node_type('branch')
    per_node_type('junction')
    per_node_type('inlet')
    per_node_type('outlet')
    per_edge_type('branch_to_branch')
    per_edge_type('junction_to_junction')
    per_edge_type('branch_to_junction')
    per_edge_type('junction_to_branch')
    per_edge_type('in_to_branch')
    per_edge_type('in_to_junction')
    per_edge_type('out_to_branch')
    per_edge_type('out_to_junction')

def standardize(field, coeffs):
    if type(coeffs['mean']) == list:
        coeffs['mean'] = np.asarray(coeffs['mean'])
    if type(coeffs['std']) == list:
        coeffs['std'] = np.asarray(coeffs['std'])
    try:
        ncomponents = coeffs['mean'].size
    except AttributeError:
        ncomponents = 1
    if ncomponents == 1:
        if coeffs['std'] < 1e-6:
            return (field - coeffs['mean'])
        else:
            return (field - coeffs['mean']) / coeffs['std']
    for i in range(ncomponents):
        if coeffs['std'][i] < 1e-6:
            field[:,i] = (field[:,i] - coeffs['mean'][i])
        else:
            field[:,i] = (field[:,i] - coeffs['mean'][i]) / coeffs['std'][i]
    return field

def invert_standardize(field, coeffs):
    return coeffs['mean'] + field * coeffs['std']

def standard_normalization(graph, fields, coeffs_dict):
    def per_node_type(node_type):
        node_features = graph.nodes[node_type].data
        for feat in node_features:
            for field in fields:
                if field in feat:
                    graph.nodes[node_type].data[feat] = standardize(graph.nodes[node_type].data[feat], coeffs_dict[field]).float()

    def per_edge_type(edge_type):
        edge_features = graph.edges[edge_type].data
        for feat in edge_features:
            for field in fields:
                if field in feat:
                    graph.edges[edge_type].data[feat] = standardize(graph.edges[edge_type].data[feat], coeffs_dict[field]).float()

    per_node_type('branch')
    per_node_type('junction')
    per_node_type('inlet')
    per_node_type('outlet')
    per_edge_type('branch_to_branch')
    per_edge_type('junction_to_junction')
    per_edge_type('branch_to_junction')
    per_edge_type('junction_to_branch')
    per_edge_type('in_to_branch')
    per_edge_type('in_to_junction')
    per_edge_type('out_to_branch')
    per_edge_type('out_to_junction')

def normalize_function(field, field_name, coefs_dict):
    if coefs_dict['type'] == 'min_max':
        return min_max(field, coefs_dict[field_name])
    elif coefs_dict['type'] == 'standard':
        return standardize(field, coefs_dict[field_name])
    return []

def invert_normalize_function(field, field_name, coefs_dict):
    if coefs_dict['type'] == 'min_max':
        return invert_min_max(field, coefs_dict[field_name])
    if coefs_dict['type'] == 'standard':
        return invert_standardize(field, coefs_dict[field_name])
    return []

def graph_statistics(graph, field):
    def transform_scalar(value):
        try:
            return float(value)
        except TypeError:
            return value
    def per_node_type(node_type):
        N = 0
        sumv = 0
        minv = None
        maxv = None
        node_features = graph.nodes[node_type].data
        for feat in node_features:
            if field in feat:
                value = graph.nodes[node_type].data[feat].detach().numpy()
                if minv is None:
                    minv = transform_scalar(np.min(value, axis = 0))
                else:
                    minv = np.min([minv,transform_scalar(np.min(value,axis = 0))])
                if maxv is None:
                    maxv = transform_scalar(np.max(value, axis = 0))
                else:
                    maxv = np.max([maxv,transform_scalar(np.max(value,axis = 0))])
                N = N + value.shape[0]
                sumv = sumv + np.sum(value, axis = 0)
        return minv, maxv, sumv, N

    def per_edge_type(edge_type):
        N = 0
        sumv = 0
        minv = None
        maxv = None
        edge_features = graph.edges[edge_type].data
        for feat in edge_features:
            if field in feat:
                value = graph.edges[edge_type].data[feat].detach().numpy()
                if minv is None:
                    minv = transform_scalar(np.min(value, axis = 0))
                else:
                    minv = np.min([minv,transform_scalar(np.min(value,axis = 0))])
                if maxv is None:
                    maxv = transform_scalar(np.max(value, axis = 0))
                else:
                    maxv = np.max([maxv,transform_scalar(np.max(value,axis = 0))])
                N = N + value.shape[0]
                sumv = sumv + np.sum(value, axis = 0)
        return  minv, maxv, sumv, N
    N = 0
    sumv = 0
    minv = None
    maxv = None
    node_types = ['branch', 'junction', 'inlet', 'outlet']
    edge_types = ['branch_to_branch', 'junction_to_junction',
                  'branch_to_junction', 'junction_to_branch',
                  'in_to_branch', 'in_to_junction',
                  'out_to_branch', 'out_to_junction']
    for nt in node_types:
        cmin, cmax, csum, cN = per_node_type(nt)
        if minv is None:
            minv = cmin
        elif cmin is not None:
            minv = np.min([minv,cmin], axis = 0)
        if maxv is None:
            maxv = cmax
        elif cmax is not None:
            maxv = np.max([maxv,cmax], axis = 0)
        sumv = sumv + csum
        N = N + cN

    for et in edge_types:
        cmin, cmax, csum, cN = per_edge_type(et)
        if minv is None:
            minv = cmin
        elif cmin is not None:
            minv = np.min([minv,cmin], axis = 0)
        if maxv is None:
            maxv = cmax
        elif cmax is not None:
            maxv = np.max([maxv,cmax], axis = 0)
        sumv = sumv + csum
        N = N + cN

    return minv, maxv, sumv, N

def graph_squared_diff(graph, field, mean):
    def per_node_type(node_type, mean):
        sumv = 0
        node_features = graph.nodes[node_type].data
        for feat in node_features:
            if field in feat:
                value = graph.nodes[node_type].data[feat].detach().numpy()
                sumv = sumv + np.sum((value - mean)**2, axis = 0)
        return sumv

    def per_edge_type(edge_type, mean):
        sumv = 0
        edge_features = graph.edges[edge_type].data
        for feat in edge_features:
            if field in feat:
                value = graph.edges[edge_type].data[feat].detach().numpy()
                sumv = sumv + np.sum((value - mean)**2, axis = 0)
        return  sumv
    sumv = 0
    node_types = ['branch', 'junction', 'inlet', 'outlet']
    edge_types = ['branch_to_branch', 'junction_to_junction',
                  'branch_to_junction', 'junction_to_branch',
                  'in_to_branch', 'in_to_junction',
                  'out_to_branch', 'out_to_junction']

    for nt in node_types:
        csum = per_node_type(nt, mean)
        sumv = sumv + csum

    for et in edge_types:
        csum = per_edge_type(et, mean)
        sumv = sumv + csum

    return sumv

def compute_statistics(graphs, fields, coefs_dict):
    for field in fields:
        # first we compute min, max and mean
        N = 0
        sumv = 0
        minv = None
        maxv = None
        count = 0
        for graph in graphs:
            gmin, gmax, gsum, gN = graph_statistics(graph, field)
            if minv is None:
                minv = gmin
            else:
                minv = np.min([minv, gmin], axis = 0)
            if maxv is None:
                maxv = gmax
            else:
                maxv = np.max([maxv, gmax], axis = 0)
            sumv = sumv + gsum
            N = N + gN

            # basic correctness check
            # if field in ['pressure', 'area'] and gmin < 0:
            #     print(field + ' can not be negative! Model ' + str(count))
            count = count + 1

        cmean = sumv / N
        sumv = 0

        # compute sum of diffs squared for std
        for graph in graphs:
            gdiff = graph_squared_diff(graph, field, cmean)
            sumv = sumv + gdiff

        # we use an unbiased estimator
        cstdv = np.sqrt(1 / (N-1) * sumv)

        if type(cmean) == list and len(cmean) == 1:
            cmean = cmean[0]
        if type(cstdv) == list and len(cstdv) == 1:
            cstdv = cstdv[0]
        coefs_dict[field] = {'min': minv,
                             'max': maxv,
                             'mean': cmean,
                             'std': cstdv}

    return coefs_dict

def normalize_graphs(graphs, fields, coefs_dict):
    norm_graphs = []

    ntype = coefs_dict['type']

    for name in graphs:
        graph = graphs[name]
        if ntype == 'min_max':
            min_max_normalization(graph, fields, coefs_dict)
        if ntype == 'standard':
            standard_normalization(graph, fields, coefs_dict)

def normalize(graphs, ntype, coefs_dict = None):
    fields = {'pressure', 'flowrate', 'area', 'rel_position_norm', 'distance', 'dt'}

    start = time.time()
    if coefs_dict == None:
        coefs_dict = {}
        coefs_dict['type'] = ntype
        coefs_dict = compute_statistics(graphs.values(), fields, coefs_dict)
    end = time.time()

    elapsed_time = end - start
    print('\tstatistics computed in {:0.2f} s'.format(elapsed_time))

    normalize_graphs(graphs, fields, coefs_dict)

    return coefs_dict

def rotate_graph(graph, identity = False):
    # we want to the keep the scale close to one otherwise flowrate and pressure
    # don't make sense
    if not identity:
        # scale = round(np.random.normal(1, 0.001) , 10)
        scale = 1

        # random rotation matrix
        R, _ = np.linalg.qr(np.random.rand(3,3))
    else:
        scale = 1
        R = np.eye(3)

    def rotate_array(inarray):
        return  (np.matmul(inarray,R) * scale).float()

    def scale_array(inarray):
        return inarray * scale

    graph.edges['branch_to_branch'].data['rel_position'] = \
        rotate_array(graph.edges['branch_to_branch'].data['rel_position'])
    graph.edges['branch_to_branch'].data['rel_position'] = \
        scale_array(graph.edges['branch_to_branch'].data['rel_position'])
    graph.edges['junction_to_junction'].data['rel_position'] = \
        rotate_array(graph.edges['junction_to_junction'].data['rel_position'])
    graph.edges['junction_to_junction'].data['rel_position'] = \
        scale_array(graph.edges['junction_to_junction'].data['rel_position'])
    graph.edges['junction_to_branch'].data['rel_position'] = \
        rotate_array(graph.edges['junction_to_branch'].data['rel_position'])
    graph.edges['junction_to_branch'].data['rel_position'] = \
        scale_array(graph.edges['junction_to_branch'].data['rel_position'])
    graph.edges['branch_to_junction'].data['rel_position'] = \
        rotate_array(graph.edges['branch_to_junction'].data['rel_position'])
    # graph.edges['branch_to_junction'].data['rel_position'][:,3] = \
    #     scale_array(graph.edges['branch_to_junction'].data['rel_position'])
    # graph.edges['in_to_branch'].data['distance'] = \
    #     scale_array(graph.edges['in_to_branch'].data['distance'])
    # graph.edges['in_to_junction'].data['distance'] = \
    #     scale_array(graph.edges['in_to_junction'].data['distance'])
    # graph.edges['out_to_branch'].data['distance'] = \
    #     scale_array(graph.edges['out_to_branch'].data['distance'])
    # graph.edges['out_to_junction'].data['distance'] = \
    #     scale_array(graph.edges['out_to_junction'].data['distance'])

    # fig = plt.figure()
    # ax = plt.axes(projection='3d')


    # ax.scatter(graph.nodes['branch'].data['x'][:,0],
    #             graph.nodes['branch'].data['x'][:,1],
    #             graph.nodes['branch'].data['x'][:,2], c='black')

    graph.nodes['branch'].data['x'] = \
        rotate_array(graph.nodes['branch'].data['x'])
    # scale_array(graph.nodes['branch'].data['area'])
    graph.nodes['branch'].data['tangent'] = \
        rotate_array(graph.nodes['branch'].data['tangent'])
    # ax.scatter(graph.nodes['branch'].data['x'][:,0],
    #             graph.nodes['branch'].data['x'][:,1],
    #             graph.nodes['branch'].data['x'][:,2], c='black')

    # ax.scatter(graph.nodes['junction'].data['x'][:,0],
    #             graph.nodes['junction'].data['x'][:,1],
    #             graph.nodes['junction'].data['x'][:,2], c='red')
    graph.nodes['junction'].data['x'] = \
        rotate_array(graph.nodes['junction'].data['x'])
    # scale_array(graph.nodes['junction'].data['area'])
    graph.nodes['junction'].data['tangent'] = \
        rotate_array(graph.nodes['junction'].data['tangent'])
    # ax.scatter(graph.nodes['junction'].data['x'][:,0],
    #             graph.nodes['junction'].data['x'][:,1],
    #             graph.nodes['junction'].data['x'][:,2], c='red')
    # plt.show()

    graph.nodes['inlet'].data['x'] = \
        rotate_array(graph.nodes['inlet'].data['x'])
    graph.nodes['outlet'].data['x'] = \
        rotate_array(graph.nodes['outlet'].data['x'])

def normalize_dataset(data_folder, dataset_params,  output_dir):
    graphs_names = os.listdir(data_folder)
    graphs_names = graphs_names

    graphs = {}
    start = time.time()
    count = 0
    models = set()
    prevbincount = -1
    for name in graphs_names:
        if '.grph' in name:
            bincount = 0
            for name2 in graphs_names:
                p1 = name.find('.')
                p2 = name2.find('.')
                if name[0:p1] == name2[0:p2]:
                    models.add(name[0:p1])
                    bincount = bincount + 1
            if prevbincount != -1 and bincount != prevbincount:
                print(name)
                raise RuntimeError('Number of examples is not constant')
            print('Loading ' + str(count) + ': ' + name + ', bincount = ' + str(bincount))
            graphs[name] = load_graphs(data_folder + '/' + name)[0][0]
            count = count + 1
            prevbincount = bincount

    end = time.time()
    elapsed_time = end - start
    print('Graphs loaded in {:0.2f} s'.format(elapsed_time))

    models_dict = {}
    models_dict['dataset'] = list(models)
    models_dict['training'] = 0.9
    models_dict['validation'] = 1 - 0.9
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(output_dir + '/dataset_list.json', 'w') as outfile:
        json.dump(models_dict, outfile, indent=4)

    # print dataset statistics
    nnodes = 0
    nedges = 0
    for name in graphs:
        graph = graphs[name]
        nnodes = nnodes + graph.nodes['branch'].data['x'].shape[0]
        nnodes = nnodes + graph.nodes['junction'].data['x'].shape[0]
        nedges = nedges + graph.edges['branch_to_branch'].data['rel_position'].shape[0]
        nedges = nedges + graph.edges['branch_to_junction'].data['rel_position'].shape[0]
        nedges = nedges + graph.edges['junction_to_junction'].data['rel_position'].shape[0]
        nedges = nedges + graph.edges['junction_to_branch'].data['rel_position'].shape[0]

    numgraphs = len(graphs)
    print('n. graphs = ' + str(numgraphs))
    print('average n. nodes = ' + str(nnodes / len(graphs)))
    print('average n. edges = ' + str(nedges / len(graphs)))

    print('Rotating graphs')
    start = time.time()
    for name in graphs:
        # we only rotate the models with model version != 0
        if '.0.' not in name:
            rotate_graph(graphs[name])
        else:
            rotate_graph(graphs[name], identity=True)
    end = time.time()
    elapsed_time = end - start
    print('Graphs rotated in {:0.2f} s'.format(elapsed_time))

    print('Normalizing graphs')
    start = time.time()
    normalization_type = dataset_params['normalization']
    coefs_dict = normalize(graphs, normalization_type)
    end = time.time()
    elapsed_time = end - start
    print('Graphs normalized in {:0.2f} s'.format(elapsed_time))

    for name in graphs:
        dgl.save_graphs(output_dir + '/' + name, graphs[name])

    dataset_params['augmented_graphs'] = bincount

    with open(output_dir + '/dataset_parameters.json', 'w') as outfile:
        json.dump(dataset_params, outfile, indent=4)

    def flatten_dict(dicti):
        for c in dicti:
            cd = dicti[c]
            if isinstance(cd, dict):
                flatten_dict(cd)
            if isinstance(cd, torch.Tensor):
                dicti[c] = cd.detach().numpy()
            if isinstance(cd, np.ndarray):
                if dicti[c].size == 1:
                    dicti[c] = float(cd[0])
                else:
                    dicti[c] = cd.tolist()

    flatten_dict(coefs_dict)

    with open(output_dir + '/normalization_coefficients.json', 'w') as outfile:
        json.dump(coefs_dict, outfile, indent=4)

if __name__ == "__main__":
    dataset_params = {'normalization': 'standard',
                      'label_normalization': 'min_max'}

    normalize_dataset(io.data_location() + 'graphs/', dataset_params,
                      io.data_location() + 'normalized_graphs/')
