import sys
import os

# this fixes a problem with openmp https://github.com/dmlc/xgboost/issues/1715
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# add path to core
sys.path.append("../tools/")

import matplotlib.pyplot as plt
import io_utils as io
import dgl
import numpy as np
import torch
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import copy
from scipy import interpolate
from matplotlib import animation
import json
from raw_graph import RawGraph

DTYPE = np.float32

def create_fixed_graph(raw_graph, area):
    inner_dict, inlet_dict, outlet_dict, macro_dict = raw_graph.create_heterogeneous_graph()

    graph_data = {('inner', 'inner_to_inner', 'inner'): \
                  (inner_dict['edges'][:,0], inner_dict['edges'][:,1]),
                  ('inlet', 'in_to_inner', 'inner'): \
                  (inlet_dict['edges'][:,0], inlet_dict['edges'][:,1]), \
                  ('outlet', 'out_to_inner', 'inner'): \
                  (outlet_dict['edges'][:,0],outlet_dict['edges'][:,1]), \
                  ('params', 'dummy', 'params'): \
                  (np.array([0]), np.array([0])), \
                  ('inner', 'inner_to_macro', 'macro'): \
                  (macro_dict['inner_to_macro'][:,0],
                   macro_dict['inner_to_macro'][:,1]),
                  ('macro', 'macro_to_inner', 'inner'): \
                  (macro_dict['macro_to_inner'][:,0],
                   macro_dict['macro_to_inner'][:,1]),
                  ('macro', 'macro_to_junction_positive', 'junction'): \
                  (macro_dict['macro_to_junction_positive'][:,0],
                   macro_dict['macro_to_junction_positive'][:,1]),
                  ('macro', 'macro_to_junction_negative', 'junction'): \
                  (macro_dict['macro_to_junction_negative'][:,0],
                   macro_dict['macro_to_junction_negative'][:,1])}

    graph = dgl.heterograph(graph_data)

    graph.edges['inner_to_inner'].data['position'] = \
                        torch.from_numpy(inner_dict['position'].astype(DTYPE))
    graph.edges['inner_to_inner'].data['edges'] = \
                        torch.from_numpy(inner_dict['edges'])
    graph.edges['in_to_inner'].data['distance'] = \
                        torch.from_numpy(inlet_dict['distance'].astype(DTYPE))
    graph.edges['in_to_inner'].data['physical_contiguous'] = \
                        torch.from_numpy(inlet_dict['physical_contiguous'])
    graph.edges['in_to_inner'].data['edges'] = \
                        torch.from_numpy(inlet_dict['edges'])
    graph.edges['out_to_inner'].data['distance'] = \
                        torch.from_numpy(outlet_dict['distance'].astype(DTYPE))
    graph.edges['out_to_inner'].data['physical_contiguous'] = \
                        torch.from_numpy(outlet_dict['physical_contiguous'])
    graph.edges['out_to_inner'].data['edges'] = \
                        torch.from_numpy(outlet_dict['edges'])

    graph.nodes['inner'].data['x'] = torch.from_numpy(inner_dict['x'])
    graph.nodes['inner'].data['global_mask'] = torch.from_numpy(inner_dict['mask'])
    graph.nodes['inner'].data['area'] = torch.from_numpy(area[inner_dict['mask']].astype(DTYPE))
    graph.nodes['inner'].data['node_type'] = torch.nn.functional.one_hot(
                                             torch.from_numpy(
                                             np.squeeze(
                                             inner_dict['node_type'].astype(int))))
    graph.nodes['inner'].data['dt'] = torch.ones(graph.nodes['inner'].data['area'].shape)
    graph.nodes['inner'].data['tangent'] = torch.from_numpy(inner_dict['tangent'])

    graph.nodes['inlet'].data['global_mask'] = torch.from_numpy(inlet_dict['mask'])
    graph.nodes['inlet'].data['area'] = torch.from_numpy(area[inlet_dict['mask']].astype(DTYPE))
    graph.nodes['inlet'].data['x'] = torch.from_numpy(inlet_dict['x'])

    graph.nodes['outlet'].data['global_mask'] = torch.from_numpy(outlet_dict['mask'])
    graph.nodes['outlet'].data['area'] = torch.from_numpy(area[outlet_dict['mask']].astype(DTYPE))
    graph.nodes['outlet'].data['x'] = torch.from_numpy(outlet_dict['x'])

    print('Graph generated:')
    print(' n. inner nodes = ' + str(inner_dict['x'].shape[0]))
    print(' n. inner edges = ' + str(inner_dict['edges'].shape[0]))
    print(' n. inlet edges = ' + str(inlet_dict['edges'].shape[0]))
    print(' n. outlet edges = ' + str(outlet_dict['edges'].shape[0]))

    return graph

def set_field(graph, name_field, field):
    def set_in_node(node_type):
        mask = graph.nodes[node_type].data['global_mask'].detach().numpy().astype(int)
        masked_field = torch.from_numpy(field[mask].astype(DTYPE))
        graph.nodes[node_type].data[name_field] = masked_field
    set_in_node('inner')
    set_in_node('inlet')
    set_in_node('outlet')

def add_fields(graph, pressure, velocity):
    print('Writing fields:')
    graphs = []
    times = [t for t in pressure]
    times.sort()
    nP = pressure[times[0]].shape[0]
    nQ = velocity[times[0]].shape[0]
    print('  n. times = ' + str(len(times)))

    newgraph = copy.deepcopy(graph)

    for t in range(len(times)):
        set_field(newgraph, 'pressure_' + str(t), pressure[times[t]])
        set_field(newgraph, 'flowrate_' + str(t), velocity[times[t]])

    newgraph.nodes['params'].data['times'] = \
                        torch.from_numpy(np.expand_dims(np.array(times),axis=0))

    newgraph.nodes['inner'].data['dt'] = graph.nodes['inner'].data['dt'] * (np.array(times[1] - times[0]))

    return newgraph

def augment_time(field, period, ntimepoints):
    times_before = [t for t in field]
    times_before.sort()
    ntimes = len(times_before)

    npoints = field[times_before[0]].shape[0]

    times_scaled = np.linspace(0, period, ntimes)
    times_new = np.linspace(0, period, ntimepoints)

    Y = np.zeros((npoints, ntimepoints))
    for ipoint in range(npoints):
        y = []
        for t in times_before:
            y.append(field[t][ipoint])

        tck = interpolate.splrep(times_scaled, y, s=0)
        Y[ipoint,:] = interpolate.splev(times_new, tck, der=0)

    newfield = {}
    count = 0
    for t in times_new:
        newfield[t] = np.expand_dims(Y[:,count],axis=1)
        count = count + 1

    return newfield

def generate_graphs(model_name, model_params, input_dir, output_dir, save = True):
    print('Create geometry: ' + model_name)
    soln = io.read_geo(input_dir + '/' + model_name + '.vtp').GetOutput()
    fields, _, p_array = io.get_all_arrays(soln)

    raw_graph = RawGraph(p_array, model_params)
    area = raw_graph.project(fields['area'])
    raw_graph.set_node_types(fields['BifurcationId'])

    g_pressure, g_flowrate = io.gather_pressures_flowrates(fields)

    pressure = {}
    for t in g_pressure:
        pressure[t] = raw_graph.partition_and_stack_field(g_pressure[t])

    flowrate = {}
    for t in g_flowrate:
        flowrate[t] = raw_graph.partition_and_stack_field(g_flowrate[t])

    print('Augmenting timesteps')
    pressure = augment_time(pressure, model_params['period'],
                                      model_params['n_time_points'])
    flowrate = augment_time(flowrate, model_params['period'],
                                      model_params['n_time_points'])

    print('Generating graphs')
    fixed_graph = create_fixed_graph(raw_graph, raw_graph.stack(area))

    print('Adding fields')
    graphs = add_fields(fixed_graph, pressure, flowrate)
    if save:
        dgl.save_graphs(output_dir + model_name + '.grph', graphs)
    return graphs

if __name__ == "__main__":
    input_dir = 'vtps'
    output_dir = 'data/'
    params = json.load(open(input_dir + '/dataset_info.json'))
    for model in params:
        print('Processing {}'.format(model))
        generate_graphs(model, params[model], input_dir, output_dir)
