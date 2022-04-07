import sys
import os

# this fixes a problem with openmp https://github.com/dmlc/xgboost/issues/1715
os.environ['KMP_DUPLICATE_LIB_OK']='True'

sys.path.append("../graphs")
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
from os.path import exists
import json
import time
import normalization as nrmz
import pickle
import pathlib
import io_utils as io

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
                if node_type == 'branch' or node_type == 'junction':
                    graph.nodes[node_type].data['n_labels'] = torch.cat((graph.nodes[node_type].data['pressure_next'] - \
                                                                         graph.nodes[node_type].data['pressure'] - \
                                                                         noise_dict['pressure'][node_type], \
                                                                         graph.nodes[node_type].data['flowrate_next'] - \
                                                                         graph.nodes[node_type].data['flowrate'] - \
                                                                         noise_dict['flowrate'][node_type]), 1).float()


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
                                                                       graph.nodes[node_type].data['tangent']), 1).float()
            else:
                graph.nodes[node_type].data['n_features'] = torch.cat((graph.nodes[node_type].data['pressure'] + \
                                                                       noise_dict['pressure'][node_type], \
                                                                       graph.nodes[node_type].data['flowrate'] + \
                                                                       noise_dict['flowrate'][node_type], \
                                                                       graph.nodes[node_type].data['area'], \
                                                                       graph.nodes[node_type].data['tangent']), 1).float()
        elif node_type == 'junction':
            if noise_dict == None:
                graph.nodes[node_type].data['n_features'] = torch.cat((graph.nodes[node_type].data['pressure'], \
                                                                       graph.nodes[node_type].data['flowrate'], \
                                                                       graph.nodes[node_type].data['area'], \
                                                                       graph.nodes[node_type].data['tangent']), 1).float()
            else:
                graph.nodes[node_type].data['n_features'] = torch.cat((graph.nodes[node_type].data['pressure'] + \
                                                                       noise_dict['pressure'][node_type], \
                                                                       graph.nodes[node_type].data['flowrate'] + \
                                                                       noise_dict['flowrate'][node_type], \
                                                                       graph.nodes[node_type].data['area'], \
                                                                       graph.nodes[node_type].data['tangent']), 1).float()
        # These would be physical bcs but they work worse
        # elif node_type == 'inlet':
        #     if noise_dict == None:
        #         graph.nodes[node_type].data['n_features'] = torch.cat((graph.nodes[node_type].data['flowrate'], \
        #                                                                graph.nodes[node_type].data['flowrate_next'], \
        #                                                                graph.nodes[node_type].data['area']), 1).float()
        #     else:
        #         graph.nodes[node_type].data['n_features'] = torch.cat((graph.nodes[node_type].data['flowrate'] + noise_dict['flowrate'][node_type], \
        #                                                                graph.nodes[node_type].data['flowrate_next'], \
        #                                                                graph.nodes[node_type].data['area']), 1).float()
        # elif node_type == 'outlet':
        #     if noise_dict == None:
        #         graph.nodes[node_type].data['n_features'] = torch.cat((graph.nodes[node_type].data['pressure'], \
        #                                                                graph.nodes[node_type].data['pressure_next'], \
        #                                                                graph.nodes[node_type].data['area']), 1).float()
        #     else:
        #         graph.nodes[node_type].data['n_features'] = torch.cat((graph.nodes[node_type].data['pressure'] + noise_dict['pressure'][node_type], \
        #                                                                graph.nodes[node_type].data['pressure_next'], \
        #                                                                graph.nodes[node_type].data['area']), 1).float()

        elif node_type == 'inlet':
            if noise_dict == None:
                graph.nodes[node_type].data['n_features'] = torch.cat((graph.nodes[node_type].data['pressure'], \
                                                                       graph.nodes[node_type].data['pressure_next'],
                                                                       graph.nodes[node_type].data['flowrate'], \
                                                                       graph.nodes[node_type].data['flowrate_next'], \
                                                                       graph.nodes[node_type].data['area']), 1).float()
            else:
                graph.nodes[node_type].data['n_features'] = torch.cat((graph.nodes[node_type].data['pressure'] + noise_dict['pressure'][node_type], \
                                                                       graph.nodes[node_type].data['pressure_next'],
                                                                       graph.nodes[node_type].data['flowrate'] + noise_dict['flowrate'][node_type], \
                                                                       graph.nodes[node_type].data['flowrate_next'], \
                                                                       graph.nodes[node_type].data['area']), 1).float()
        elif node_type == 'outlet':
            if noise_dict == None:
                graph.nodes[node_type].data['n_features'] = torch.cat((graph.nodes[node_type].data['pressure'], \
                                                                       graph.nodes[node_type].data['pressure_next'], \
                                                                       graph.nodes[node_type].data['flowrate'], \
                                                                       graph.nodes[node_type].data['flowrate_next'], \
                                                                       graph.nodes[node_type].data['area']), 1).float()
            else:
                graph.nodes[node_type].data['n_features'] = torch.cat((graph.nodes[node_type].data['pressure'] + noise_dict['pressure'][node_type], \
                                                                       graph.nodes[node_type].data['pressure_next'], \
                                                                       graph.nodes[node_type].data['flowrate'] + noise_dict['flowrate'][node_type], \
                                                                       graph.nodes[node_type].data['flowrate_next'], \
                                                                       graph.nodes[node_type].data['area']), 1).float()

    def per_edge_type(edge_type):
        if edge_type == 'branch_to_branch' or \
           edge_type == 'junction_to_junction' or \
           edge_type == 'branch_to_junction' or \
           edge_type == 'junction_to_branch':
            graph.edges[edge_type].data['e_features'] = torch.cat((graph.edges[edge_type].data['rel_position'], \
                                                                   graph.edges[edge_type].data['rel_position_norm']), 1)
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
    def __init__(self, graphs, dataset_params, coefs_dict, filenames):
        self.graphs = graphs
        self.dataset_params = dataset_params
        self.label_normalization = dataset_params['label_normalization']
        self.coefs_dict = coefs_dict
        self.set_times()
        self.filenames = filenames
        super().__init__(name='dgl_dataset')

    def set_times(self):
        self.times = []
        for graph in self.graphs:
            dt = float(graph.nodes['branch'].data['dt'][0])
            dt = nrmz.invert_normalize_function(dt, 'dt', self.coefs_dict)
            # find max time index
            keys = graph.nodes['branch'].data.keys()
            maxi = 0
            for key in keys:
                if 'pressure_' in key:
                    maxi = np.max([maxi, int(key[9:])])
            times = np.arange(maxi + 1) * dt
            self.times.append(np.expand_dims(times,0))

    def get_model_version(self, model_name):
        model_name_sub = model_name[model_name.find('.')+1:]
        model_name_v = model_name_sub[:model_name_sub.find('.')]
        return int(model_name_v)

    def set_max_model_version(self, max_model_version):
        graphs_to_delete = []

        numgraphs = len(self.graphs)
        for igraph in range(numgraphs):
            if self.get_model_version(self.filenames[igraph]) > max_model_version:
                graphs_to_delete.append(igraph)

        for index in sorted(graphs_to_delete, reverse=True):
            del self.filenames[index]
            del self.graphs[index]
            del self.lightgraphs[index]
            del self.times[index]
            del self.noise_pressures[index]
            del self.noise_flowrates[index]

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
        start = time.time()
        def per_node_type(graph, ntype, field):
            todelete = []
            for data_name in graph.nodes[ntype].data:
                if field in data_name:
                    todelete.append(data_name)

            for data_name in todelete:
                del graph.nodes[ntype].data[data_name]

        self.lightgraphs = []
        self.noise_pressures = []
        self.noise_flowrates = []
        N = 0
        for igraph in range(len(self.graphs)):
            lightgraph = copy.deepcopy(self.graphs[igraph])

            for ntype in ['branch', 'junction', 'inlet', 'outlet']:
                per_node_type(lightgraph, ntype, 'pressure')
                per_node_type(lightgraph, ntype, 'flowrate')

            self.lightgraphs.append(lightgraph)

            nbranch_nodes = self.graphs[igraph].nodes['branch'].data['pressure_0'].shape[0]
            njunction_nodes = self.graphs[igraph].nodes['junction'].data['pressure_0'].shape[0]
            nnodes = nbranch_nodes + njunction_nodes
            noise_pressure = np.zeros((nnodes,self.times[igraph].shape[1]))
            noise_flowrate = np.zeros((nnodes,self.times[igraph].shape[1]))
            self.noise_pressures.append(noise_pressure)
            self.noise_flowrates.append(noise_flowrate)

            all_labels = np.array([])
            for itime in range(self.times[igraph].shape[1] - 1):
                self.prep_item(igraph, itime)
                curlabels = self.lightgraphs[igraph].nodes['branch'].data['n_labels']
                if all_labels.size == 0:
                    all_labels = curlabels
                else:
                    all_labels = torch.cat((all_labels, curlabels), axis = 0)
                curlabels = self.lightgraphs[igraph].nodes['junction'].data['n_labels']
                all_labels = torch.cat((all_labels, curlabels), axis = 0)
            all_labels = all_labels.detach().numpy()
            sumsquared = np.sum(all_labels**2, axis=0)
            sumx = np.sum(all_labels, axis=0)
            N = N + all_labels.shape[0]
            if igraph == 0:
                minlabels = np.min(all_labels, axis=0)
                maxlabels = np.max(all_labels, axis=0)
            else:
                curmin = np.min(all_labels, axis=0)
                curmax = np.max(all_labels, axis=0)
                for i in range(minlabels.size):
                    minlabels[i] = np.min([minlabels[i], curmin[i]])
                    maxlabels[i] = np.max([maxlabels[i], curmax[i]])

        self.label_coefs = {'min': torch.from_numpy(minlabels),
                            'max': torch.from_numpy(maxlabels),
                            'mean': torch.from_numpy(sumx / N),
                            'std': torch.tensor(np.sqrt(sumsquared/N - (sumx/N)**2)),
                            'normalization_type': self.label_normalization}

        end = time.time()
        elapsed_time = end - start
        print('\tDGLDataset generated in {:0.2f} s'.format(elapsed_time))

    def sample_noise(self, rate):
        ngraphs = len(self.noise_pressures)
        for igraph in range(ngraphs):
            dt = float(self.graphs[igraph].nodes['branch'].data['dt'][0])
            dt = nrmz.invert_normalize_function(dt, 'dt', self.coefs_dict)
            actual_rate = rate * dt
            nnodes = self.noise_pressures[igraph].shape[0]
            self.noise_pressures[igraph] = np.random.normal(0, actual_rate, (nnodes, self.times[igraph].shape[1]))
            self.noise_flowrates[igraph] = np.random.normal(0, actual_rate, (nnodes, self.times[igraph].shape[1]))
            # this would be brownian noise
            # for index in range(1,self.times[igraph].shape[1]-1):
            #     self.noise_pressures[igraph][:,index] = np.random.normal(0, actual_rate, (nnodes)) + self.noise_pressures[igraph][:,index-1]
            #     self.noise_flowrates[igraph][:,index] = np.random.normal(0, actual_rate, (nnodes)) + self.noise_flowrates[igraph][:,index-1]

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
        mb = self.graphs[gindex].nodes['branch'].data['mask']
        mj = self.graphs[gindex].nodes['junction'].data['mask']
        mi = np.array(self.graphs[gindex].nodes['inlet'].data['mask'])
        mo = self.graphs[gindex].nodes['outlet'].data['mask']

        pressure_noise_dict = {'branch': torch.from_numpy(np.expand_dims(self.noise_pressures[gindex][mb,tindex],1)),
                               'junction': torch.from_numpy(np.expand_dims(self.noise_pressures[gindex][mj,tindex],1)),
                               'inlet': torch.from_numpy(self.noise_pressures[gindex][mi,tindex]),
                               'outlet': torch.from_numpy(np.expand_dims(self.noise_pressures[gindex][mo,tindex],1))}

        flowrate_noise_dict = {'branch': torch.from_numpy(np.expand_dims(self.noise_flowrates[gindex][mb,tindex],1)),
                               'junction': torch.from_numpy(np.expand_dims(self.noise_flowrates[gindex][mj,tindex],1)),
                               'inlet': torch.from_numpy(self.noise_flowrates[gindex][mi,tindex]),
                               'outlet': torch.from_numpy(np.expand_dims(self.noise_flowrates[gindex][mo,tindex],1))}

        noise_dict = {'pressure': pressure_noise_dict,
                      'flowrate': flowrate_noise_dict}

        set_state(self.lightgraphs[gindex], state_dict,
                  next_state_dict, noise_dict, label_coefs)

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

def generate_dataset(model_names, normalized_data_dir, split, train = False):
    graphs = []
    filenames = []
    start = time.time()
    for model in model_names:
        mv = 0
        while True:
            filename = normalized_data_dir + '/' + model + '.' + str(mv) + '.grph'
            if not exists(filename):
                break
            else:
                filenames.append(model + '.' + str(mv) + '.grph')
                graphs.append(load_graphs(filename)[0][0])
            if not train:
                break
            mv = mv + 1
    end = time.time()
    elapsed_time = end - start
    print('Dataset loaded in {:0.2f} s'.format(elapsed_time), flush=True)

    coefs_dict = json.load(open(normalized_data_dir + '/normalization_coefficients.json'))
    dataset_params = json.load(open(normalized_data_dir + '/dataset_parameters.json'))
    dataset_params['split'] = split

    return DGL_Dataset(graphs, dataset_params, coefs_dict, filenames)

def prepare_dataset(dataset_json, nsets):
    def chunks(lst, n):
        n = int(np.floor(len(lst)/nsets))
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    dataset = dataset_json['dataset']
    if len(dataset) == 1:
        datasets = [{'train': [dataset[0]],
                     'test': [dataset[0]]}]
        return datasets

    random.seed(10)
    random.shuffle(dataset)

    sets = list(chunks(dataset, nsets))

    datasets = []

    for i in range(nsets):
        newdata = {'test': sets[i]}
        train_s = []
        for j in range(nsets):
            if j != i:
                train_s = train_s + sets[j]
        newdata['train'] = train_s
        datasets.append(newdata)

    return datasets

if __name__ == "__main__":
    dataset_dir = io.data_location() + 'datasets/'
    dataset_json = json.load(open(io.data_location() + 'normalized_graphs/dataset_list.json'))

    nsets = 10
    datasets_models = prepare_dataset(dataset_json, nsets = 10)

    pathlib.Path(dataset_dir).mkdir(parents=True, exist_ok=True)

    for i in range(1):
        curfolder = dataset_dir + 'd' + str(i)
        pathlib.Path(curfolder).mkdir(parents=True, exist_ok=True)
        train_dataset = generate_dataset(datasets_models[i]['train'],
                                         io.data_location() + 'normalized_graphs',
                                         datasets_models[i],
                                         train = True)
        pickle.dump(train_dataset, open(curfolder + '/train.dts', 'wb'))
        train_dataset = generate_dataset(datasets_models[i]['train'],
                                         io.data_location() + 'normalized_graphs',
                                         datasets_models[i])
        pickle.dump(train_dataset, open(curfolder + '/train_not_augmented.dts', 'wb'))
        test_dataset = generate_dataset(datasets_models[i]['test'],
                                        io.data_location() + 'normalized_graphs',
                                        datasets_models[i])
        pickle.dump(test_dataset, open(curfolder + '/test.dts', 'wb'))
