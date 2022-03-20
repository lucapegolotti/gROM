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

def get_times(graph):
    times = []

    features = graph.nodes['branch'].data
    for feature in features:
        if 'pressure' in feature:
            ind  = feature.find('_')
            times.append(float(feature[ind+1:]))
    times.sort()

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
        return (field - bounds['min']) / (bounds['max'] - bounds['min'])
    for i in range(ncomponents):
        field[:,i] = (field[:,i] - bounds['min'][i]) / (bounds['max'][i] - bounds['min'][i])
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
                           graph.nodes[node_type].data[feat] = min_max(graph.nodes[node_type].data[feat], bounds_dict[field])

    def per_edge_type(edge_type):
        edge_features = graph.edges[edge_type].data
        for feat in edge_features:
            for field in fields:
                if field in feat:
                    if np.linalg.norm(np.min(graph.edges[edge_type].data[feat].detach().numpy()) - 0) > 1e-5 and \
                       np.linalg.norm(np.max(graph.edges[edge_type].data[feat].detach().numpy()) - 1) > 1e-5:
                           graph.edges[edge_type].data[feat] = min_max(graph.edges[edge_type].data[feat], bounds_dict[field])

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
        if coeffs['std'] < 1e-12:
            return (field - coeffs['mean'])
        else:
            return (field - coeffs['mean']) / coeffs['std']
    for i in range(ncomponents):
        if coeffs['std'][i] < 1e-12:
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
                    if feat == 'dt':
                        graph.nodes[node_type].data['original_dt'] = graph.nodes[node_type].data[feat]
                    graph.nodes[node_type].data[feat] = standardize(graph.nodes[node_type].data[feat], coeffs_dict[field])

    def per_edge_type(edge_type):
        edge_features = graph.edges[edge_type].data
        for feat in edge_features:
            for field in fields:
                if field in feat:
                    graph.edges[edge_type].data[feat] = standardize(graph.edges[edge_type].data[feat], coeffs_dict[field])

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
            if field in ['pressure', 'area'] and gmin < 0:
                print(field + ' can not be negative! Model ' + str(count))
            count = count + 1

        cmean = sumv / N
        sumv = 0

        # compute sum of diffs squared for std
        for graph in graphs:
            gdiff = graph_squared_diff(graph, field, cmean)
            sumv = sumv + gdiff

        # we use an unbiased estimator
        cstdv = np.sqrt(1 / (N-1) * sumv)


        if len(cmean) == 1:
            cmean = cmean[0]
        if len(cstdv) == 1:
            cstdv = cstdv[0]
        coefs_dict[field] = {'min': minv,
                             'max': maxv,
                             'mean': cmean,
                             'std': cstdv}

    return coefs_dict

def normalize_graphs(graphs, fields, coefs_dict):
    norm_graphs = []

    ntype = coefs_dict['type']

    for graph in graphs:
        if ntype == 'min_max':
            min_max_normalization(graph, fields, coefs_dict)
        if ntype == 'standard':
            standard_normalization(graph, fields, coefs_dict)

        norm_graphs.append(graph)

    return norm_graphs

def normalize(graphs, ntype, coefs_dict = None):
    fields = {'pressure', 'flowrate', 'area', 'position', 'distance', 'dt'}

    start = time.time()
    if coefs_dict == None:
        coefs_dict = {}
        coefs_dict['type'] = ntype
        coefs_dict = compute_statistics(graphs, fields, coefs_dict)
    end = time.time()
    elapsed_time = end - start
    print('\tstatistics computed in {:0.2f} s'.format(elapsed_time))

    norm_graphs = normalize_graphs(graphs, fields, coefs_dict)

    return norm_graphs, coefs_dict

def rotate_graph(graph):
    # we want to the keep the scale close to one otherwise flowrate and pressure
    # don't make sense
    minscale = 1
    maxscale = 1
    scale = minscale + np.random.rand(1) * (maxscale - minscale)

    # random rotation matrix
    R, _ = np.linalg.qr(np.random.rand(3,3))

    def rotate_array(inarray):
        inarray = np.matmul(inarray,R) * scale
    #
    # def scale_array(inarray):
    #     inarray = inarray * scale

    rotate_array(graph.edges['branch_to_branch'].data['position'][:,0:3])
    # scale_array(newgraph.edges['branch_to_branch'].data['position'][:,3])
    rotate_array(graph.edges['junction_to_junction'].data['position'][:,0:3])
    # scale_array(newgraph.edges['junction_to_junction'].data['position'][:,3])
    rotate_array(graph.edges['junction_to_branch'].data['position'][:,0:3])
    # scale_array(newgraph.edges['junction_to_branch'].data['position'][:,3])
    rotate_array(graph.edges['branch_to_junction'].data['position'][:,0:3])
    # scale_array(newgraph.edges['branch_to_junction'].data['position'][:,3])
    # scale_array(newgraph.edges['in_to_branch'].data['distance'])
    # scale_array(newgraph.edges['in_to_junction'].data['distance'])
    # scale_array(newgraph.edges['out_to_branch'].data['distance'])
    # scale_array(newgraph.edges['out_to_junction'].data['distance'])

    rotate_array(graph.nodes['branch'].data['x'])
    # scale_array(newgraph.nodes['branch'].data['area'])
    rotate_array(graph.nodes['branch'].data['tangent'])

    rotate_array(graph.nodes['junction'].data['x'])
    # scale_array(newgraph.nodes['junction'].data['area'])
    rotate_array(graph.nodes['junction'].data['tangent'])

    rotate_array(graph.nodes['inlet'].data['x'])
    rotate_array(graph.nodes['outlet'].data['x'])

def normalize_dataset(data_folder, dataset_params,  output_dir = 'normalized_data/'):
    graphs_names = os.listdir(data_folder)

    graphs = []
    start = time.time()
    count = 0
    models = set()
    for name in graphs_names:
        if '.grph' in name:
            bincount = 0
            for name2 in graphs_names:
                if name[0:9] == name2[0:9]:
                    models.add(name[0:9])
                    bincount = bincount + 1
            print('Loading ' + str(count) + ': ' + name + ', bincount = ' + str(bincount))
            graphs.append(load_graphs(data_folder + '/' + name)[0][0])
            count = count + 1

    end = time.time()
    elapsed_time = end - start
    print('Graphs loaded in {:0.2f} s'.format(elapsed_time))

    models_dict = {}
    models_dict['dataset'] = list(models)
    models_dict['training'] = 0.9
    models_dict['validation'] = 1 - 0.9
    with open(output_dir + '/dataset_list.json', 'w') as outfile:
        json.dump(models_dict, outfile, indent=4)

    # print dataset statistics
    nnodes = 0
    nedges = 0
    for graph in graphs:
        nnodes = nnodes + graph.nodes['branch'].data['x'].shape[0]
        nnodes = nnodes + graph.nodes['junction'].data['x'].shape[0]
        nedges = nedges + graph.edges['branch_to_branch'].data['position'].shape[0]
        nedges = nedges + graph.edges['branch_to_junction'].data['position'].shape[0]
        nedges = nedges + graph.edges['junction_to_junction'].data['position'].shape[0]
        nedges = nedges + graph.edges['junction_to_branch'].data['position'].shape[0]

    numgraphs = len(graphs)
    print('n. graphs = ' + str(numgraphs))
    print('average n. nodes = ' + str(nnodes / len(graphs)))
    print('average n. edges = ' + str(nedges / len(graphs)))

    print('Rotating graphs')
    start = time.time()
    for igraph in range(numgraphs):
        # we only rotate the models with model version != 0
        if '.0.' not in graphs_names[igraph]:
            rotate_graph(graphs[igraph])
    end = time.time()
    elapsed_time = end - start
    print('Graphs rotated in {:0.2f} s'.format(elapsed_time))

    print('Normalizing graphs')
    start = time.time()
    normalization_type = dataset_params['normalization']
    graphs, coefs_dict = normalize(graphs, normalization_type)
    end = time.time()
    elapsed_time = end - start
    print('Graphs normalized in {:0.2f} s'.format(elapsed_time))

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    for igraph in range(numgraphs):
        dgl.save_graphs(output_dir + '/' + graphs_names[igraph],
                        graphs[igraph])

    def default(obj):
        if isinstance(obj, torch.Tensor):
            return default(obj.detach().numpy())
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        print(obj)
        raise TypeError('Not serializable')

    with open(output_dir + '/dataset_parameters.json', 'w') as outfile:
        json.dump(dataset_params, outfile, indent=4)

    with open(output_dir + '/normalization_coefficients.json', 'w') as outfile:
        json.dump(coefs_dict, outfile, default=default, indent=4)

if __name__ == "__main__":
    dataset_params = {'normalization': 'standard',
                      'rate_noise': 1e-5,
                      'label_normalization': 'min_max',
                      'augment_data': 0,
                      'add_noise': False,
                      'noise_stdv': 1e-10}

    normalize_dataset('data/', dataset_params)
