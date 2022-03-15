import sys
import os

# this fixes a problem with openmp https://github.com/dmlc/xgboost/issues/1715
os.environ['KMP_DUPLICATE_LIB_OK']='True'

sys.path.append("../graphs")
sys.path.append("../graphs/core")

import dgl
import torch
import numpy as np
from dgl.data.utils import load_graphs
from dgl.data import DGLDataset
import generate_graphs as gg
import copy
import random

def set_state(graph, state_dict, next_state_dict = None, noise_dict = None, coefs_label = None):
    def per_node_type(node_type):
        graph.nodes[node_type].data['pressure'] = state_dict['pressure'][node_type]
        graph.nodes[node_type].data['flowrate'] = state_dict['flowrate'][node_type]

        if next_state_dict != None:
            graph.nodes[node_type].data['pressure_next'] = next_state_dict['pressure'][node_type]
            graph.nodes[node_type].data['flowrate_next'] = next_state_dict['flowrate'][node_type]
            if noise_dict == None:
                graph.nodes[node_type].data['n_labels'] = torch.cat((graph.nodes[node_type].data['pressure_next'] - \
                                                                     graph.nodes[node_type].data['pressure'], \
                                                                     graph.nodes[node_type].data['flowrate_next'] - \
                                                                     graph.nodes[node_type].data['flowrate']), 1).float()

            else:
                if node_type == 'branch':
                    graph.nodes[node_type].data['n_labels'] = torch.cat((graph.nodes[node_type].data['pressure_next'] - \
                                                                         graph.nodes[node_type].data['pressure'] - \
                                                                         noise_dict['pressure_branch'], \
                                                                         graph.nodes[node_type].data['flowrate_next'] - \
                                                                         graph.nodes[node_type].data['flowrate'] - \
                                                                         noise_dict['flowrate_branch']), 1).float()
                elif node_type == 'junction':
                    graph.nodes[node_type].data['n_labels'] = torch.cat((graph.nodes[node_type].data['pressure_next'] - \
                                                                         graph.nodes[node_type].data['pressure'] - \
                                                                         noise_dict['pressure_junct'], \
                                                                         graph.nodes[node_type].data['flowrate_next'] - \
                                                                         graph.nodes[node_type].data['flowrate'] - \
                                                                         noise_dict['flowrate_junct']), 1).float()

        if (node_type == 'branch' or node_type == 'junction') and coefs_label != None:
            nlabels = graph.nodes[node_type].data['n_labels'].shape[1]
            for i in range(nlabels):
                colmn = graph.nodes[node_type].data['n_labels'][:,i]
                if coefs_label['normalization_type'] == 'standard':
                    graph.nodes[node_type].data['n_labels'][:,i] = (colmn - coefs_label['mean'][i]) / coefs_label['std'][i]
                elif coefs_label['normalization_type'] == 'min_max':
                    graph.nodes[node_type].data['n_labels'][:,i] = (colmn - coefs_label['min'][i]) / (coefs_label['max'][i] - coefs_label['min'][i])
                elif coefs_label['normalization_type'] == 'none':
                    pass
                else:
                    print('Label normalization {} does not exist'.format(coefs_label['normalization_type']))

        if node_type == 'branch':
            if noise_dict == None:
                graph.nodes[node_type].data['n_features'] = torch.cat((graph.nodes[node_type].data['pressure'], \
                                                                       graph.nodes[node_type].data['flowrate'], \
                                                                       graph.nodes[node_type].data['area'], \
                                                                       graph.nodes[node_type].data['tangent'],
                                                                       graph.nodes[node_type].data['dt']), 1).float()
            else:
                graph.nodes[node_type].data['n_features'] = torch.cat((graph.nodes[node_type].data['pressure'] + \
                                                                       noise_dict['pressure_branch'], \
                                                                       graph.nodes[node_type].data['flowrate'] + \
                                                                       noise_dict['flowrate_branch'], \
                                                                       graph.nodes[node_type].data['area'], \
                                                                       graph.nodes[node_type].data['tangent'],
                                                                       graph.nodes[node_type].data['dt']), 1).float()
        elif node_type == 'junction':
            if noise_dict == None:
                graph.nodes[node_type].data['n_features'] = torch.cat((graph.nodes[node_type].data['pressure'], \
                                                                       graph.nodes[node_type].data['flowrate'], \
                                                                       graph.nodes[node_type].data['area'], \
                                                                       graph.nodes[node_type].data['node_type'], \
                                                                       graph.nodes[node_type].data['tangent'],
                                                                       graph.nodes[node_type].data['dt']), 1).float()
            else:
                graph.nodes[node_type].data['n_features'] = torch.cat((graph.nodes[node_type].data['pressure'] + \
                                                                       noise_dict['pressure_junct'], \
                                                                       graph.nodes[node_type].data['flowrate'] + \
                                                                       noise_dict['flowrate_junct'], \
                                                                       graph.nodes[node_type].data['area'], \
                                                                       graph.nodes[node_type].data['node_type'], \
                                                                       graph.nodes[node_type].data['tangent'],
                                                                       graph.nodes[node_type].data['dt']), 1).float()
        else:
            graph.nodes[node_type].data['n_features'] = torch.cat((graph.nodes[node_type].data['pressure'], \
                                                                   graph.nodes[node_type].data['pressure_next'], \
                                                                   graph.nodes[node_type].data['flowrate'], \
                                                                   graph.nodes[node_type].data['flowrate_next'], \
                                                                   graph.nodes[node_type].data['area']), 1).float()

    def per_edge_type(edge_type):
        if edge_type == 'branch_to_branch' or \
           edge_type == 'junction_to_junction' or \
           edge_type == 'branch_to_junction' or \
           edge_type == 'junction_to_branch':
            graph.edges[edge_type].data['e_features'] = graph.edges[edge_type].data['position'].float()
        else:
            graph.edges[edge_type].data['e_features'] = torch.cat((graph.edges[edge_type].data['distance'][:,None],
                                                                   graph.edges[edge_type].data['physical_same'][:,None]), 1).float()
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


def set_bcs(graph, state_dict):
    def per_node_type(node_type):
        graph.nodes[node_type].data['pressure_next'] = state_dict['pressure'][node_type]
        graph.nodes[node_type].data['flowrate_next'] = state_dict['flowrate'][node_type]
    per_node_type('inlet')
    per_node_type('outlet')

class DGL_Dataset(DGLDataset):
    def __init__(self, graphs = None, label_normalization = 'none', coefs_dict = None):
        self.graphs = graphs
        self.set_times()
        self.label_normalization = label_normalization
        self.coefs_dict = coefs_dict
        super().__init__(name='dgl_dataset')

    def set_times(self):
        self.times = []
        for graph in self.graphs:
            self.times.append(graph.nodes['params'].data['times'])

    def i_to_graph_index(self, i):
        grph_index = 0
        sum = 0
        prev_sum = 0
        while i >= sum:
            prev_sum = sum
            # minus 1 because we don't consider the last timestep for each rollout
            sum = sum + self.times[grph_index].shape[1] - 1
            grph_index = grph_index + 1

        grph_index = grph_index - 1
        time_index = i - prev_sum

        return grph_index, time_index

    def process(self):
        def per_node_type(graph, ntype, field):
            todelete = []
            for data_name in graph.nodes[ntype].data:
                if field in data_name:
                    todelete.append(data_name)

            for data_name in todelete:
                del graph.nodes[ntype].data[data_name]

        self.lightgraphs = []
        self.noise_pressures_branch = []
        self.noise_flowrates_branch = []
        self.noise_pressures_junct = []
        self.noise_flowrates_junct = []
        all_labels = None
        for igraph in range(len(self.graphs)):
            lightgraph = copy.deepcopy(self.graphs[igraph])

            for ntype in ['branch', 'junction', 'inlet', 'outlet']:
                per_node_type(lightgraph, ntype, 'pressure')
                per_node_type(lightgraph, ntype, 'flowrate')

            self.lightgraphs.append(lightgraph)

            nbranch_nodes = self.graphs[igraph].nodes['branch'].data['pressure_0'].shape[0]
            noise_pressure = np.zeros((nbranch_nodes,self.times[igraph].shape[1]))
            noise_flowrate = np.zeros((nbranch_nodes,self.times[igraph].shape[1]))
            self.noise_pressures_branch.append(noise_pressure)
            self.noise_flowrates_branch.append(noise_flowrate)
            njunction_nodes = self.graphs[igraph].nodes['junction'].data['pressure_0'].shape[0]
            noise_pressure = np.zeros((njunction_nodes,self.times[igraph].shape[1]))
            noise_flowrate = np.zeros((njunction_nodes,self.times[igraph].shape[1]))
            self.noise_pressures_junct.append(noise_pressure)
            self.noise_flowrates_junct.append(noise_flowrate)

            for itime in range(self.times[igraph].shape[1] - 1):
                self.prep_item(igraph, itime)
                curlabels = self.lightgraphs[igraph].nodes['branch'].data['n_labels']
                if all_labels == None:
                    all_labels = curlabels
                else:
                    all_labels = torch.cat((all_labels, curlabels), axis = 0)

            for itime in range(self.times[igraph].shape[1] - 1):
                self.prep_item(igraph, itime)
                curlabels = self.lightgraphs[igraph].nodes['junction'].data['n_labels']
                if all_labels == None:
                    all_labels = curlabels
                else:
                    all_labels = torch.cat((all_labels, curlabels), axis = 0)

        all_labels = all_labels.detach().numpy()
        self.label_coefs = {'min': torch.from_numpy(np.min(all_labels, axis=0)),
                            'max': torch.from_numpy(np.max(all_labels, axis=0)),
                            'mean': torch.from_numpy(np.mean(all_labels, axis=0)),
                            'std': torch.from_numpy(np.std(all_labels, axis=0)),
                            'normalization_type': self.label_normalization}

    def sample_noise(self, rate):
        ngraphs = len(self.noise_pressures_branch)
        for igraph in range(ngraphs):
            dt = float(self.graphs[igraph].nodes['branch'].data['dt'][0])
            dt = invert_normalize_function(dt, 'dt', self.coefs_dict)
            actual_rate = rate * dt
            nnodes_branch = self.noise_pressures_branch[igraph].shape[0]
            nnodes_junction = self.noise_pressures_junct[igraph].shape[0]
            for index in range(1,self.times[igraph].shape[1]-1):
                self.noise_pressures_branch[igraph][:,index] = np.random.normal(0, actual_rate, (nnodes_branch)) + self.noise_pressures_branch[igraph][:,index-1]
                self.noise_flowrates_branch[igraph][:,index] = np.random.normal(0, actual_rate, (nnodes_branch)) + self.noise_flowrates_branch[igraph][:,index-1]
                self.noise_pressures_junct[igraph][:,index] = np.random.normal(0, actual_rate, (nnodes_junction)) + self.noise_pressures_junct[igraph][:,index-1]
                self.noise_flowrates_junct[igraph][:,index] = np.random.normal(0, actual_rate, (nnodes_junction)) + self.noise_flowrates_junct[igraph][:,index-1]

    def get_state_dict(self, gindex, tindex):
        pressure_dict = {'branch': self.graphs[gindex].nodes['branch'].data['pressure_' + str(tindex)],
                         'junction': self.graphs[gindex].nodes['junction'].data['pressure_' + str(tindex)],
                         'inlet': self.graphs[gindex].nodes['inlet'].data['pressure_' + str(tindex)],
                         'outlet': self.graphs[gindex].nodes['outlet'].data['pressure_' + str(tindex)]}
        flowrate_dict = {'branch': self.graphs[gindex].nodes['branch'].data['flowrate_' + str(tindex)],
                         'junction': self.graphs[gindex].nodes['junction'].data['flowrate_' + str(tindex)],
                         'inlet': self.graphs[gindex].nodes['inlet'].data['flowrate_' + str(tindex)],
                         'outlet': self.graphs[gindex].nodes['outlet'].data['flowrate_' + str(tindex)]}
        return {'pressure': pressure_dict, 'flowrate': flowrate_dict}

    def prep_item(self, gindex, tindex, label_coefs = None):
        state_dict = self.get_state_dict(gindex, tindex)
        next_state_dict = self.get_state_dict(gindex, tindex+1)
        noise_dict = {'pressure_branch': torch.from_numpy(np.expand_dims(self.noise_pressures_branch[gindex][:,tindex],1)),
                      'pressure_junct': torch.from_numpy(np.expand_dims(self.noise_pressures_junct[gindex][:,tindex],1)),
                      'flowrate_branch': torch.from_numpy(np.expand_dims(self.noise_flowrates_branch[gindex][:,tindex],1)),
                      'flowrate_junct': torch.from_numpy(np.expand_dims(self.noise_flowrates_junct[gindex][:,tindex],1))}
        set_state(self.lightgraphs[gindex], state_dict, next_state_dict, noise_dict, label_coefs)

    def __getitem__(self, i):
        try:
            gindex, tindex =  self.i_to_graph_index(i.detach().numpy())
        except AttributeError:
            gindex, tindex =  self.i_to_graph_index(i)
        self.prep_item(gindex, tindex, self.label_coefs)
        return self.lightgraphs[gindex]

    def __len__(self):
        sum = 0
        for itime in range(len(self.times)):
            # remove last timestep
            sum = sum + self.times[itime].shape[1] - 1
        return sum

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

        cmean = sumv / N
        sumv = 0

        # compute sum of diffs squared for std
        for graph in graphs:
            gdiff = graph_squared_diff(graph, field, cmean)
            sumv = sumv + gdiff

        # we use an unbiased estimator
        cstdv = np.sqrt(1 / (N-1) * sumv)

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

    if coefs_dict == None:
        coefs_dict = {}
        coefs_dict['type'] = ntype
        coefs_dict = compute_statistics(graphs, fields, coefs_dict)

    norm_graphs = normalize_graphs(graphs, fields, coefs_dict)

    return norm_graphs, coefs_dict

def randomize_graph(graph):
    # we want to the keep the scale close to one otherwise flowrate and pressure
    # don't make sense
    minscale = 1/1.04
    maxscale = 1.04
    scale = minscale + np.random.rand(1) * (maxscale - minscale)

    # random rotation matrix
    R, _ = np.linalg.qr(np.random.rand(3,3))

    def rotate_array(inarray):
        inarray = np.matmul(inarray,R) * scale

    def scale_array(inarray):
        inarray = inarray * scale

    newgraph = copy.deepcopy(graph)

    rotate_array(newgraph.edges['branch_to_branch'].data['position'][:,0:3])
    scale_array(newgraph.edges['branch_to_branch'].data['position'][:,3])
    rotate_array(newgraph.edges['junction_to_junction'].data['position'][:,0:3])
    scale_array(newgraph.edges['junction_to_junction'].data['position'][:,3])
    rotate_array(newgraph.edges['junction_to_branch'].data['position'][:,0:3])
    scale_array(newgraph.edges['junction_to_branch'].data['position'][:,3])
    rotate_array(newgraph.edges['branch_to_junction'].data['position'][:,0:3])
    scale_array(newgraph.edges['branch_to_junction'].data['position'][:,3])
    scale_array(newgraph.edges['in_to_branch'].data['distance'])
    scale_array(newgraph.edges['in_to_junction'].data['distance'])
    scale_array(newgraph.edges['out_to_branch'].data['distance'])
    scale_array(newgraph.edges['out_to_junction'].data['distance'])

    rotate_array(newgraph.nodes['branch'].data['x'])
    scale_array(newgraph.nodes['branch'].data['area'])
    rotate_array(newgraph.nodes['branch'].data['tangent'])

    rotate_array(newgraph.nodes['junction'].data['x'])
    scale_array(newgraph.nodes['junction'].data['area'])
    rotate_array(newgraph.nodes['junction'].data['tangent'])

    rotate_array(newgraph.nodes['inlet'].data['x'])
    rotate_array(newgraph.nodes['outlet'].data['x'])

    return newgraph

def generate_dataset(model_names, coefs_dict = None, dataset_params = None, augment = False):
    graphs = []
    for model_name in model_names:
        graphs.append(load_graphs('../graphs/data/' + model_name + '.grph')[0][0])

    numgraphs = len(graphs)

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

    print('n. graphs = ' + str(len(graphs)))
    print('average n. nodes = ' + str(nnodes / len(graphs)))
    print('average n. edges = ' + str(nedges / len(graphs)))

    if augment:
        num_augmentation = dataset_params['augment_data']
        for i in range(num_augmentation):
            for igraph in range(numgraphs):
                graphs.append(randomize_graph(graphs[igraph]))

    normalization_type = 'standard'
    if dataset_params != None:
        normalization_type = dataset_params['normalization']

    label_normalization = 'none'
    if dataset_params != None:
        label_normalization = dataset_params['label_normalization']

    graphs, coefs_dict = normalize(graphs, normalization_type, coefs_dict)

    return DGL_Dataset(graphs, label_normalization, coefs_dict)
