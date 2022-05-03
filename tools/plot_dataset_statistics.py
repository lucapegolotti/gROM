import io_utils as io
import matplotlib.pyplot as plt
import os
import numpy as np
from dgl.data.utils import load_graphs

if __name__ == "__main__":
    data_location = io.data_location()
    graphs_location = data_location + 'graphs'
    graphs_names = os.listdir(graphs_location)

    data_pressure = []
    data_flowrate = []
    count = 0
    for graph_name in graphs_names:
        if '.0.' in graph_name:
            count = count + 1
            print(count)
            print(graph_name)
            graph = load_graphs(graphs_location + '/' + graph_name)[0][0]
            fields = graph.nodes['branch'][0]

            pressures_timesteps = []
            first = True
            for f in fields:
                if 'pressure' in f:
                    pressures_timesteps.append(f)

            ntsteps = len(pressures_timesteps)
            for t in range(0, ntsteps):
                timestamp_p = 'pressure_' + str(t)
                timestamp_q = 'flowrate_' + str(t)
                if first:
                    pressure = graph.nodes['branch'].data[timestamp_p].detach().numpy()
                    flowrate = graph.nodes['branch'].data[timestamp_q].detach().numpy()
                    first = False
                else:
                    pressure = np.concatenate((pressure, graph.nodes['branch'].data[timestamp_p]), axis = 0)
                    flowrate = np.concatenate((flowrate, graph.nodes['branch'].data[timestamp_q]), axis = 0)

                pressure = np.concatenate((pressure, graph.nodes['junction'].data[timestamp_p]), axis = 0)
                # flowrate = np.concatenate((flowrate, graph.nodes['junction'].data[timestamp_q]), axis = 0)

            data_pressure.append(pressure.squeeze() / 1333.2)
            data_flowrate.append(flowrate.squeeze())

    plt.boxplot(data_flowrate)

    plt.show()
