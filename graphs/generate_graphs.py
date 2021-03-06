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
import plot_tools as pt
import pathlib
from os.path import exists
import random
import time

DTYPE = np.float32

def create_fixed_graph(raw_graph, area):
    branch_dict, junct_dict, inlet_dict, outlet_dict, macro_dict = \
        raw_graph.create_heterogeneous_graph()

    graph_data = {('branch', 'branch_to_branch', 'branch'): \
                  (branch_dict['edges'][:,0], branch_dict['edges'][:,1]), \
                  ('junction', 'junction_to_junction', 'junction'): \
                  (junct_dict['edges'][:,0], junct_dict['edges'][:,1]), \
                  ('branch', 'branch_to_junction', 'junction'): \
                  (branch_dict['branch_to_junct'][:,0], branch_dict['branch_to_junct'][:,1]), \
                  ('junction', 'junction_to_branch', 'branch'): \
                  (junct_dict['junct_to_branch'][:,0], junct_dict['junct_to_branch'][:,1]), \
                  ('inlet', 'in_to_branch', 'branch'): \
                  (inlet_dict['edges_branch'][:,0], inlet_dict['edges_branch'][:,1]), \
                  ('inlet', 'in_to_junction', 'junction'): \
                  (inlet_dict['edges_junct'][:,0], inlet_dict['edges_junct'][:,1]), \
                  ('outlet', 'out_to_branch', 'branch'): \
                  (outlet_dict['edges_branch'][:,0], outlet_dict['edges_branch'][:,1]), \
                  ('outlet', 'out_to_junction', 'junction'): \
                  (outlet_dict['edges_junct'][:,0], outlet_dict['edges_junct'][:,1]), \
                  ('branch', 'branch_to_macro', 'macro'): \
                  (macro_dict['branch_to_macro'][:,0],
                   macro_dict['branch_to_macro'][:,1]),
                  ('macro', 'macro_to_branch', 'branch'): \
                  (macro_dict['macro_to_branch'][:,0],
                   macro_dict['macro_to_branch'][:,1]),
                  ('macro', 'macro_to_junction_positive', 'junction'): \
                  (macro_dict['macro_to_junction_positive'][:,0],
                   macro_dict['macro_to_junction_positive'][:,1]),
                  ('macro', 'macro_to_junction_negative', 'junction'): \
                  (macro_dict['macro_to_junction_negative'][:,0],
                   macro_dict['macro_to_junction_negative'][:,1])}

    if macro_dict['inlet_to_junction_positive'].size > 0:
        graph_data[('inlet', 'inlet_to_junction_positive', 'junction')] = \
                  (macro_dict['inlet_to_junction_positive'][:,0],
                   macro_dict['inlet_to_junction_positive'][:,1])
    else:
        graph_data[('inlet', 'inlet_to_junction_positive', 'junction')] = \
                  (np.array([]), np.array([]))

    if macro_dict['outlet_to_junction_negative'].size > 0:
        graph_data[('outlet', 'outlet_to_junction_negative', 'junction')] = \
                  (macro_dict['outlet_to_junction_negative'][:,0],
                   macro_dict['outlet_to_junction_negative'][:,1])
    else:
        graph_data[('outlet', 'outlet_to_junction_negative', 'junction')] = \
                  (np.array([]), np.array([]))

    graph = dgl.heterograph(graph_data)

    graph.edges['branch_to_branch'].data['rel_position'] = \
                        torch.from_numpy(branch_dict['pos'][:,0:3].astype(DTYPE))
    graph.edges['branch_to_branch'].data['rel_position_norm'] = \
                        torch.from_numpy(np.expand_dims(branch_dict['pos'][:,3],1).astype(DTYPE))
    graph.edges['branch_to_branch'].data['edges'] = \
                        torch.from_numpy(branch_dict['edges'])
    graph.edges['junction_to_junction'].data['rel_position'] = \
                        torch.from_numpy(junct_dict['pos'][:,0:3].astype(DTYPE))
    graph.edges['junction_to_junction'].data['rel_position_norm'] = \
                        torch.from_numpy(np.expand_dims(junct_dict['pos'][:,3],1).astype(DTYPE))
    graph.edges['junction_to_junction'].data['edges'] = \
                        torch.from_numpy(junct_dict['edges'])
    graph.edges['branch_to_junction'].data['rel_position'] = \
                        torch.from_numpy(branch_dict['pos_branch_to_junct'][:,0:3].astype(DTYPE))
    graph.edges['branch_to_junction'].data['rel_position_norm'] = \
                        torch.from_numpy(np.expand_dims(branch_dict['pos_branch_to_junct'][:,3],1).astype(DTYPE))
    graph.edges['junction_to_branch'].data['rel_position'] = \
                        torch.from_numpy(junct_dict['pos_junct_to_branch'][:,0:3].astype(DTYPE))
    graph.edges['junction_to_branch'].data['rel_position_norm'] = \
                        torch.from_numpy(np.expand_dims(junct_dict['pos_junct_to_branch'][:,3],1).astype(DTYPE))
    graph.edges['in_to_branch'].data['distance'] = \
                        torch.from_numpy(inlet_dict['distance_branch'].astype(DTYPE))
    graph.edges['in_to_junction'].data['distance'] = \
                        torch.from_numpy(inlet_dict['distance_junct'].astype(DTYPE))
    graph.edges['in_to_branch'].data['physical_same'] = \
                        torch.from_numpy(inlet_dict['physical_same_branch'])
    graph.edges['in_to_junction'].data['physical_same'] = \
                        torch.from_numpy(inlet_dict['physical_same_junct'])
    graph.edges['in_to_branch'].data['edges'] = \
                        torch.from_numpy(inlet_dict['edges_branch'])
    graph.edges['in_to_junction'].data['edges'] = \
                        torch.from_numpy(inlet_dict['edges_junct'])
    graph.edges['out_to_branch'].data['distance'] = \
                        torch.from_numpy(outlet_dict['distance_branch'].astype(DTYPE))
    graph.edges['out_to_junction'].data['distance'] = \
                        torch.from_numpy(outlet_dict['distance_junct'].astype(DTYPE))
    graph.edges['out_to_branch'].data['physical_same'] = \
                        torch.from_numpy(outlet_dict['physical_same_branch'])
    graph.edges['out_to_junction'].data['physical_same'] = \
                        torch.from_numpy(outlet_dict['physical_same_junct'])
    graph.edges['out_to_branch'].data['edges'] = \
                        torch.from_numpy(outlet_dict['edges_branch'])
    graph.edges['out_to_junction'].data['edges'] = \
                        torch.from_numpy(outlet_dict['edges_junct'])

    graph.nodes['branch'].data['mask'] = torch.from_numpy(branch_dict['mask'])
    graph.nodes['branch'].data['x'] = torch.from_numpy(branch_dict['x'])
    graph.nodes['branch'].data['area'] = torch.from_numpy(area[branch_dict['mask']].astype(DTYPE))
    graph.nodes['branch'].data['dt'] = torch.ones(area[branch_dict['mask']].shape)
    graph.nodes['branch'].data['tangent'] = torch.from_numpy(branch_dict['tangent'])

    graph.nodes['junction'].data['mask'] = torch.from_numpy(junct_dict['mask'])
    graph.nodes['junction'].data['x'] = torch.from_numpy(junct_dict['x'])
    graph.nodes['junction'].data['area'] = torch.from_numpy(area[junct_dict['mask']].astype(DTYPE))
    graph.nodes['junction'].data['dt'] = torch.ones(area[junct_dict['mask']].shape)
    graph.nodes['junction'].data['tangent'] = torch.from_numpy(junct_dict['tangent'])
    max_bif_degree = 20
    try:
        graph.nodes['junction'].data['node_type'] = torch.nn.functional.one_hot(
                                                    torch.from_numpy(
                                                    np.squeeze(
                                                    junct_dict['node_type'].astype(int))),
                                                    num_classes=max_bif_degree)
    except RuntimeError:
        print("max_bif_degree < " + str(np.max(np.squeeze(junct_dict['node_type'].astype(int)))))
        raise RuntimeError

    graph.nodes['inlet'].data['mask'] = torch.from_numpy(inlet_dict['mask'])
    graph.nodes['inlet'].data['area'] = torch.from_numpy(area[inlet_dict['mask']].astype(DTYPE))
    graph.nodes['inlet'].data['x'] = torch.from_numpy(inlet_dict['x'])

    graph.nodes['outlet'].data['mask'] = torch.from_numpy(outlet_dict['mask'])
    graph.nodes['outlet'].data['area'] = torch.from_numpy(area[outlet_dict['mask']].astype(DTYPE))
    graph.nodes['outlet'].data['x'] = torch.from_numpy(outlet_dict['x'])

    print('Graph generated:')
    print(' n. branch nodes = ' + str(branch_dict['x'].shape[0]))
    print(' n. junction nodes = ' + str(junct_dict['x'].shape[0]))
    print(' n. nodes = ' + str(branch_dict['x'].shape[0] + junct_dict['x'].shape[0]))

    return graph

def set_field(graph, name_field, field):
    def set_in_node(node_type):
        mask = graph.nodes[node_type].data['mask'].detach().numpy().astype(int)
        masked_field = torch.from_numpy(field[mask].astype(DTYPE))
        graph.nodes[node_type].data[name_field] = masked_field
    set_in_node('branch')
    set_in_node('junction')
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

    newgraph.nodes['branch'].data['dt'] = graph.nodes['branch'].data['dt'] * (np.array(times[1] - times[0]))
    newgraph.nodes['junction'].data['dt'] = graph.nodes['junction'].data['dt'] * (np.array(times[1] - times[0]))

    return newgraph

def augment_time(field, model_params, dt_new):
    timestep = model_params['timestep']

    times_before = [t for t in field]
    times_before.sort()
    ntimes = len(times_before)

    npoints = field[times_before[0]].shape[0]

    times_scaled = np.array([t * timestep for t in times_before])
    times_scaled = times_scaled - times_scaled[0]
    period = times_scaled[-1]

    # dt_new = 0.002
    # times_new = np.linspace(0, period, ntimepoints)
    times_new = [0]
    t = 0
    while t < period:
        t = t + dt_new
        times_new.append(t)

    times_new = np.array(times_new)
    ntimepoints = times_new.size

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
        tround = round(t, 10)
        newfield[tround] = np.expand_dims(Y[:,count],axis=1)
        count = count + 1

    return newfield

def generate_graphs(model_name, model_params, input_dir, output_dir,
                    n_graphs_per_model, save = True):
    print('Create geometry: ' + model_name)
    versions = []
    max_model_version = 5
    for mv in range(0, max_model_version):
        model_name_v = model_name + '.' + str(mv)
        if exists(input_dir + '/' + model_name_v + '.vtp'):
            versions.append(mv)

    # delete this
    if 1 not in versions:
        return True

    failures = 0
    max_failures = 100
    n_graph = 0
    while n_graph < n_graphs_per_model:
        print('Adding graph ' + str(n_graph))
        # pick a random version
        mv = random.choice(versions)
        # delete this
        mv = 1
        model_name_v = model_name + '.' + str(mv)
        soln = io.read_geo(input_dir + '/' + model_name_v + '.vtp').GetOutput()
        fields, _, p_array = io.get_all_arrays(soln)

        debug = False

        try:
            start = time.time()
            raw_graph = RawGraph(p_array, model_params, debug)
            area, _ = raw_graph.project(fields['area'])
            for a in area:
                if (np.min(a) < 0):
                    raise ValueError('Interpolated area is negative!')

            raw_graph.set_node_types(fields['BifurcationId'])
            end = time.time()
            elapsed_time = end - start
            print('Graph generated in = {:0.2f} s'.format(elapsed_time))

            g_pressure, g_flowrate = io.gather_pressures_flowrates(fields)

            times = [t for t in g_pressure]
            times.sort()

            # if we consider simulation from the start we skip a few timesteps
            # because the first ones are noisy
            if mv == 0:
                for i in range(0,10):
                    del g_pressure[times[i]]
                    del g_flowrate[times[i]]

                times = times[10:]

            start = time.time()

            pressure, g_pressure = raw_graph.partition_and_stack_fields(g_pressure)
            flowrate, g_flowrate = raw_graph.partition_and_stack_fields(g_flowrate)

            # we just check that there are no negative pressures after interpolation
            for t in pressure:
                if np.min(pressure[t]) < 0:
                    raise ValueError("Min pressure is negative")

            end = time.time()
            elapsed_time = end - start
            print('Fields projected in = {:0.2f} s'.format(elapsed_time))

            check_interpolation = False
            if check_interpolation:
                pathlib.Path('check_interpolation/').mkdir(parents=True, exist_ok=True)
                folder = 'check_interpolation/' + model_name + '.' + str(n_graph)
                pathlib.Path(folder).mkdir(parents=True, exist_ok=True)

                arclength_interpolated = raw_graph.compute_resampled_arclenght()
                arclength_original = raw_graph.compute_arclenght()

                pt.plot_interpolated(pressure, g_pressure,
                                  arclength_interpolated, arclength_original,
                                  folder + '/pressure.mp4')

                pt.plot_interpolated(flowrate, g_flowrate,
                                  arclength_interpolated, arclength_original,
                                  folder + '/flowrate.mp4')

            print('Augmenting timesteps')
            # dt_new = round(np.random.normal(0.002, 0.0001) , 10)
            dt_new = 0.002

            pressure = augment_time(pressure, model_params, dt_new)
            flowrate = augment_time(flowrate, model_params, dt_new)

            for t in pressure:
                if np.min(pressure[t]) < 0:
                    raise ValueError("Min pressure is negative")

            print('Generating graphs')
            fixed_graph = create_fixed_graph(raw_graph, raw_graph.stack(area))

            print('Adding fields')
            graphs = add_fields(fixed_graph, pressure, flowrate)
            if save:
                dgl.save_graphs(output_dir + model_name + '.' + str(n_graph) + '.grph', graphs)
            n_graph = n_graph + 1

        except Exception as e:
            print('Failed to generate: ' + str(e))
            failures = failures + 1
            if failures < max_failures:
                print('Retrying...')
                continue
            else:
                print('Exiting...')
                return False
    return True

if __name__ == "__main__":
    data_location = io.data_location()
    input_dir = data_location + 'vtps'
    output_dir = data_location + 'graphs/'
    params = json.load(open(input_dir + '/dataset_aortas.json'))

    timesteps = json.load(open(input_dir + '/timesteps.json'))
    for model in params:
        tmodel = model
        if model[-1] == '_':
            tmodel = model[:-1]
        params[model]['timestep'] = timesteps[tmodel]

    failed_models = []
    n_graphs_per_model = 5
    for model in params:
        print('Processing {}'.format(model))
        success = generate_graphs(model, params[model], input_dir, output_dir,
                        n_graphs_per_model)
        if not success:
            failed_models.append(model)

    print('Failed models:')
    print(failed_models)
