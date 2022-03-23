import sys

sys.path.append("../graphs")
sys.path.append("../network")

import io_utils as io
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import interpolate
from matplotlib.colors import Normalize
from matplotlib import animation
import preprocessing as pp
import matplotlib.cm as cm
import random
import normalization as nrmz

def circle3D(center, normal, radius, npoints):
    theta = np.linspace(0, 2 * np.pi, npoints)

    axis = np.array([normal[2],0,-normal[0]])
    axis = axis / np.linalg.norm(axis)
    if (axis[0] < 0):
        axis = - axis

    axis_p = np.cross(normal, axis)
    axis_p = axis_p / np.linalg.norm(axis_p)
    if (axis_p[1] < 0):
        axis_p = - axis_p

    circle = np.zeros((npoints, 3))

    for ipoint in range(npoints):
        circle[ipoint,:] = center + np.sin(theta[ipoint]) * axis * radius + np.cos(theta[ipoint]) * axis_p * radius

    return circle

def plot_3D_graph(graph, state = None, coefs = None, bounds = None, field_name = None, cmap = cm.get_cmap("viridis")):
    fig = plt.figure(figsize=(8, 8), dpi=200)
    # fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax = plt.axes(projection='3d')
    ax._axis3don = False

    scatterpts = None
    if state == None:
        # plot inlet
        xin = graph.nodes['inlet'].data['x'].detach().numpy()
        ax.scatter(xin[:,0], xin[:,1], xin[:,2], color='red', depthshade=0)

        # plot outlet
        xout = graph.nodes['outlet'].data['x'].detach().numpy()
        ax.scatter(xout[:,0], xout[:,1], xout[:,2], color='blue', depthshade=0)

        # plot inner
        xinner = graph.nodes['inner'].data['x'].detach().numpy()
        ax.scatter(xinner[:,0], xinner[:,1], xinner[:,2], color='black', depthshade=0)

        x = np.concatenate((xin,xout,xinner),axis=0)

    else:
        xin = graph.nodes['inlet'].data['x'].detach().numpy()
        xout = graph.nodes['outlet'].data['x'].detach().numpy()
        xinner = graph.nodes['inner'].data['x'].detach().numpy()
        x = np.concatenate((xin,xout,xinner),axis=0)

        fin = state[field_name]['inlet']
        fout = state[field_name]['outlet']
        finner = state[field_name]['inner']
        field = np.concatenate((fin,fout,finner),axis=0)

        field = nrmz.invert_normalize_function(field, field_name, coefs)

        if type(bounds[field_name]['min']) == list:
            bounds[field_name]['min'] = np.asarray(bounds[field_name]['min'])

        if type(bounds[field_name]['max']) == list:
            bounds[field_name]['max'] = np.asarray(bounds[field_name]['max'])

        C = (field - bounds[field_name]['min']) / \
            (bounds[field_name]['max'] - bounds[field_name]['min'])

        scatterpts = ax.scatter(x[:,0], x[:,1], x[:,2],
                                color=cmap(C), depthshade=0)

    minx = np.min(x, axis=0)
    maxx = np.max(x, axis=0)

    m = np.min(minx)
    M = np.max(maxx)

    padding = np.max([np.abs(m),np.abs(M)]) * 0.1

    minx = minx - padding
    maxx = maxx + padding

    ax.set_box_aspect((maxx[0]-minx[0], maxx[1]-minx[1], maxx[2]-minx[2]))

    ax.set_xlim((minx[0],maxx[0]))
    ax.set_ylim((minx[1],maxx[1]))
    ax.set_zlim((minx[2],maxx[2]))

    cbar = fig.colorbar(scatterpts)

    cbar.set_ticks([0, 1])
    if field_name == 'pressure':
        minv = str('{:.0f}'.format(float(coefs[field_name]['min']/1333.2)))
        maxv = str('{:.0f}'.format(float(coefs[field_name]['max']/1333.2)))
    else:
        minv = str('{:.0f}'.format(float(coefs[field_name]['min'])))
        maxv = str('{:.0f}'.format(float(coefs[field_name]['max'])))

    cbar.set_ticklabels([minv, maxv])

    return fig, ax, scatterpts

def plot_node_types(graph, color_nodes, outfile_name = None, time = 5):
    framerate = 60
    nframes = time * framerate
    fig = plt.figure(figsize=(8, 8), dpi=284)
    # fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax = plt.axes(projection='3d')
    ax._axis3don = False

    # plot inner
    xbranch = graph.nodes['branch'].data['x'].detach().numpy()
    xjunct = graph.nodes['junction'].data['x'].detach().numpy()
    x = np.concatenate((xbranch, xjunct), axis = 0)
    ax.scatter(x[:,0], x[:,1], x[:,2], color=color_nodes, depthshade=0)

    minx = np.min(x, axis=0)
    maxx = np.max(x, axis=0)

    m = np.min(minx)
    M = np.max(maxx)

    padding = np.max([np.abs(m),np.abs(M)]) * 0.1

    minx = minx - padding
    maxx = maxx + padding

    ax.set_box_aspect((maxx[0]-minx[0], maxx[1]-minx[1], maxx[2]-minx[2]))

    ax.set_xlim((minx[0],maxx[0]))
    ax.set_ylim((minx[1],maxx[1]))
    ax.set_zlim((minx[2],maxx[2]))

    if outfile_name != None:
        angles = np.floor(np.linspace(0,360,nframes)).astype(int)
        def animation_frame(i):
            ax.view_init(elev=10., azim=angles[i])
            return

        anim = animation.FuncAnimation(fig, animation_frame,
                                       frames=nframes,
                                       interval=20)
        writervideo = animation.FFMpegWriter(fps=framerate)
        anim.save(outfile_name, writer = writervideo)

def plot_geo_with_circles(graph,cmap = cm.get_cmap("viridis")):
    fig = plt.figure()
    # fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax = plt.axes(projection='3d')
    ax._axis3don = False

    node_types = graph.nodes['inner'].data['node_type'].detach().numpy()

    colors = np.zeros((node_types.shape[0]))

    for i in range(node_types.shape[0]):
        colors[i] = np.where(node_types[i,:] == 1)[0]

    colors = colors / np.max(colors)

    scatterpts = None
    # plot inlet
    xin = graph.nodes['inlet'].data['x'].detach().numpy()
    ax.scatter(xin[:,0], xin[:,1], xin[:,2], color='red', depthshade=0)

    # plot outlet
    xout = graph.nodes['outlet'].data['x'].detach().numpy()
    ax.scatter(xout[:,0], xout[:,1], xout[:,2], color='green', depthshade=0)

    # plot inner
    xinner = graph.nodes['inner'].data['x'].detach().numpy()
    tangent = graph.nodes['inner'].data['tangent'].detach().numpy()
    area = graph.nodes['inner'].data['area'].detach().numpy()
    ax.scatter(xinner[:,0], xinner[:,1], xinner[:,2], color=cmap(colors), depthshade=0)

    for i in range(xinner.shape[0]):
        radius = np.sqrt(area[i] / np.pi)
        if colors[i] == 0:
            circle = circle3D(xinner[i,:], tangent[i,:], radius, 60)
            ax.plot(circle[:,0], circle[:,1], circle[:,2], color='red')

    x = np.concatenate((xin,xout,xinner),axis=0)

    minx = np.min(x, axis=0)
    maxx = np.max(x, axis=0)

    m = np.min(minx)
    M = np.max(maxx)

    padding = np.max([np.abs(m),np.abs(M)]) * 0.1

    minx = minx - padding
    maxx = maxx + padding

    ax.set_box_aspect((maxx[0]-minx[0], maxx[1]-minx[1], maxx[2]-minx[2]))

    ax.set_xlim((minx[0],maxx[0]))
    ax.set_ylim((minx[1],maxx[1]))
    ax.set_zlim((minx[2],maxx[2]))

def plot_interpolated(field_interpolated, field_original,
                      arclength_interpolated, arclength_original,
                      filename):

    times = []
    for t in field_interpolated:
        times.append(t)

    nframes = 3 * 30
    indices = np.floor(np.linspace(0,len(field_interpolated)-1,nframes)).astype(int)

    selected_times = []
    selected_field_interpolated = []
    selected_field_original = []
    for ind in indices:
        selected_field_interpolated.append(field_interpolated[times[ind]])
        selected_field_original.append(field_original[times[ind]])
        selected_times.append(times[ind])

    times = selected_times

    fig, ax = plt.subplots(2, dpi=284)

    scatter_original = ax[0].scatter(arclength_original, \
                                   selected_field_original[0], color='black', \
                                   s = 1.5, alpha = 0.3)
    scatter_interpol = ax[0].scatter(arclength_interpolated, \
                                   selected_field_interpolated[0], color='red', \
                                   s = 1)
    scatter_original_2 = ax[1].scatter(arclength_original, \
                                   selected_field_original[0], color='black', \
                                   s = 1.5, alpha = 1)

    def animation_frame(i):
        df = selected_field_interpolated[i]
        df = np.concatenate((np.expand_dims(arclength_interpolated,axis=1), df),axis = 1)
        scatter_interpol.set_offsets(df)
        df = selected_field_original[i]
        df = np.concatenate((np.expand_dims(arclength_original,axis=1), df),axis = 1)
        scatter_original.set_offsets(df)
        scatter_original_2.set_offsets(df)
        ax[0].set_title('{:.2f} s'.format(float(times[i])))
        ax[0].set_xlim(arclength_interpolated[0],arclength_interpolated[-1])
        ax[0].set_ylim(np.min(df[:,1]),np.max(df[:,1]))
        ax[1].set_xlim(arclength_interpolated[0],arclength_interpolated[-1])
        ax[1].set_ylim(np.min(df[:,1]),np.max(df[:,1]))
        return scatter_original,

    anim = animation.FuncAnimation(fig, animation_frame,
                                   frames=len(times),
                                   interval=20)

    writervideo = animation.FFMpegWriter(fps=60)
    anim.save(filename, writer = writervideo)

def plot_linear(pressures_branch_pred, flowrates_branch_pred,
                pressures_junction_pred, flowrates_junction_pred,
                pressures_branch_real, flowrates_branch_real,
                pressures_junction_real, flowrates_junction_real,
                colors, times, coefs_dict, bounds, outfile_name, time, framerate = 60):

    nframes = time * framerate
    indices = np.floor(np.linspace(0,len(pressures_branch_pred)-1,nframes)).astype(int)

    selected_times = []
    selected_pred_pressure = []
    selected_real_pressure = []
    selected_pred_flowrate = []
    selected_real_flowrate = []
    for ind in indices:
        selected_pred_pressure.append(np.concatenate((pressures_branch_pred[ind],
                                      pressures_junction_pred[ind]), axis=0))
        selected_real_pressure.append(np.concatenate((pressures_branch_real[ind],
                                      pressures_junction_real[ind]), axis=0))
        selected_pred_flowrate.append(np.concatenate((flowrates_branch_pred[ind],
                                      flowrates_junction_pred[ind]), axis=0))
        selected_real_flowrate.append(np.concatenate((flowrates_branch_real[ind],
                                      flowrates_junction_real[ind]), axis=0))
        selected_times.append(times[ind])

    pressures_pred = selected_pred_pressure
    pressures_real = selected_real_pressure
    flowrates_pred = selected_pred_flowrate
    flowrates_real = selected_real_flowrate
    times = selected_times

    nnodes = len(pressures_pred[0])
    fig, ax = plt.subplots(2, dpi=284)

    scatter_real_p = ax[0].scatter(range(0,nnodes), \
                                   nrmz.invert_normalize_function(pressures_real[0], \
                                   'pressure',coefs_dict) / 1333.2, color='black', \
                                   s = 1.5, alpha = 0.3)
    scatter_pred_p = ax[0].scatter(range(0,nnodes), \
                                   nrmz.invert_normalize_function(pressures_pred[0], \
                                   'pressure',coefs_dict) / 1333.2, color=colors, \
                                   s = 1)
    scatter_real_q = ax[1].scatter(range(0,nnodes), \
                                   nrmz.invert_normalize_function(flowrates_real[0], \
                                   'flowrate',coefs_dict),
                                   color='black', s = 1.5, alpha = 0.3)
    scatter_pred_q = ax[1].scatter(range(0,nnodes), \
                                   nrmz.invert_normalize_function(flowrates_pred[0], \
                                   'flowrate',coefs_dict), color=colors, s = 1)

    nodesidxs = np.expand_dims(np.arange(nnodes),axis=1)
    ax[1].set_xlabel('graph node index')
    ax[0].set_ylabel('pressure [mmHg]')
    ax[1].set_ylabel('flowrate [cc^3/s]')
    def animation_frame(i):
        dp = nrmz.invert_normalize_function(pressures_pred[i],'pressure',coefs_dict) / 1333.2
        dp = np.concatenate((nodesidxs, dp),axis = 1)
        scatter_pred_p.set_offsets(dp)
        dp = nrmz.invert_normalize_function(pressures_real[i],'pressure',coefs_dict) / 1333.2
        dp = np.concatenate((nodesidxs, dp),axis = 1)
        scatter_real_p.set_offsets(dp)
        dq = nrmz.invert_normalize_function(flowrates_pred[i],'flowrate',coefs_dict)
        dq = np.concatenate((nodesidxs, dq),axis = 1)
        scatter_pred_q.set_offsets(dq)
        dq = nrmz.invert_normalize_function(flowrates_real[i],'flowrate',coefs_dict)
        dq = np.concatenate((nodesidxs, dq),axis = 1)
        scatter_real_q.set_offsets(dq)
        ax[0].set_title('{:.2f} s'.format(float(times[i])))
        ax[0].set_xlim(0,len(pressures_pred[i]))
        ax[0].set_ylim((bounds['pressure']['min']-np.abs(bounds['pressure']['min'])*0.1)/1333.2,(bounds['pressure']['max']+np.abs(bounds['pressure']['max'])*0.1) / 1333.2)
        ax[1].set_xlim(0,len(flowrates_pred[i]))
        ax[1].set_ylim(bounds['flowrate']['min']-np.abs(bounds['flowrate']['min'])*0.1,bounds['flowrate']['max']+np.abs(bounds['flowrate']['max'])*0.1)
        return scatter_pred_p,

    anim = animation.FuncAnimation(fig, animation_frame,
                                   frames=len(pressures_pred),
                                   interval=20)
    writervideo = animation.FFMpegWriter(fps=framerate)
    anim.save(outfile_name, writer = writervideo)

def plot_3D(model_name, states, times,
            coefs_dict, bounds, field_name, outfile_name, time, framerate = 60):

    nframes = time * framerate
    indices = np.floor(np.linspace(0,len(states)-1,nframes)).astype(int)

    selected_times = []
    selected_states = []
    for ind in indices:
        selected_states.append(states[ind])
        selected_times.append(times[ind])

    states = selected_states
    times = selected_times

    graph = pp.load_graphs('../graphs/data/' + model_name + '.grph')[0][0]

    cmap = cm.get_cmap("viridis")
    fig, ax, points = plot_3D_graph(graph, states[0], coefs_dict, bounds, field_name, cmap)

    angles = np.floor(np.linspace(0,360,len(states))).astype(int)

    def animation_frame(i):
        ax.view_init(elev=10., azim=angles[i])
        state = states[i]
        fin = state[field_name]['inlet']
        fout = state[field_name]['outlet']
        finner = state[field_name]['inner']
        field = np.concatenate((fin,fout,finner),axis=0)
        field = nrmz.invert_normalize_function(field, field_name, coefs_dict)

        C = (field - coefs_dict[field_name]['min']) / \
            (coefs_dict[field_name]['max'] - coefs_dict[field_name]['min'])

        points.set_color(cmap(C))
        ax.set_title('{:.2f} s'.format(float(times[i])))
        return

    anim = animation.FuncAnimation(fig, animation_frame,
                                   frames=len(states),
                                   interval=20)
    writervideo = animation.FFMpegWriter(fps=framerate, bitrate=-1)
    anim.save(outfile_name, writer = writervideo)

def plot_static(graph, pressures_branch_pred, flowrates_branch_pred,
                pressures_branch_real, flowrates_branch_real,
                times, coefs_dict, npoints, outdir):
    # randomly sample one point per portion
    fig = plt.figure()
    ax = plt.axes()

    idxs = np.arange(pressures_branch_pred[0].shape[0])
    idxs = random.sample(idxs.tolist(), npoints)

    pred_p = []
    pred_q = []
    real_p = []
    real_q = []
    for i in range(times.size):
        p = nrmz.invert_normalize_function(pressures_branch_pred[i],'pressure',coefs_dict) / 1333.2
        pred_p.append(p[idxs,0])
        p = nrmz.invert_normalize_function(pressures_branch_real[i],'pressure',coefs_dict) / 1333.2
        real_p.append(p[idxs,0])
        q = nrmz.invert_normalize_function(flowrates_branch_pred[i],'flowrate',coefs_dict)
        pred_q.append(q[idxs,0])
        q = nrmz.invert_normalize_function(flowrates_branch_real[i],'flowrate',coefs_dict)
        real_q.append(q[idxs,0])

    cmap = cm.get_cmap("cool")
    color = iter(cmap(np.linspace(0, 1, npoints)))
    pred_p = np.array(pred_p)
    pred_q = np.array(pred_q)
    real_p = np.array(real_p)
    real_q = np.array(real_q)
    for i in range(len(idxs)):
        c = next(color)
        plt.plot(times.squeeze(), real_p[:,i], color=c, linestyle='dashed', linewidth=1)
        plt.plot(times.squeeze(), pred_p[:,i], color=c, linewidth=2)

    ax.set_xlim(times[0],times[-1])
    ax.set_xlabel('time [s]')
    ax.set_ylabel('pressure [mmHg]')
    plt.savefig(outdir + '/pressure_vs_time.png')

    fig = plt.figure()
    ax = plt.axes()
    color = iter(cmap(np.linspace(0, 1, npoints)))

    for i in range(len(idxs)):
        c = next(color)
        plt.plot(times.squeeze(), real_q[:,i], color=c, linestyle='dashed', linewidth=1)
        plt.plot(times.squeeze(), pred_q[:,i], color=c, linewidth=2)

    ax.set_xlim(times[0],times[-1])
    ax.set_xlabel('time [s]')
    ax.set_ylabel('flowrate [cc^3/s]')
    plt.savefig(outdir + '/flowrate_vs_time.png')

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    xin = graph.nodes['inlet'].data['x'].detach().numpy()
    xout = graph.nodes['outlet'].data['x'].detach().numpy()
    xbranch = graph.nodes['branch'].data['x'].detach().numpy()
    xjunction = graph.nodes['junction'].data['x'].detach().numpy()

    x = np.concatenate((xin,xout,xbranch,xjunction),axis=0)

    ax.scatter(x[:,0], x[:,1], x[:,2], color='black', depthshade=0, s = 0.5)

    color = iter(cmap(np.linspace(0, 1, npoints)))
    for i in range(len(idxs)):
        c = next(color)
        ax.scatter(x[idxs[i],0], x[idxs[i],1], x[idxs[i],2], color=c, depthshade=0, s = 14)

    minx = np.min(x, axis=0)
    maxx = np.max(x, axis=0)

    m = np.min(minx)
    M = np.max(maxx)

    padding = np.max([np.abs(m),np.abs(M)]) * 0.1

    minx = minx - padding
    maxx = maxx + padding

    ax.set_box_aspect((maxx[0]-minx[0], maxx[1]-minx[1], maxx[2]-minx[2]))

    ax.set_xlim((minx[0],maxx[0]))
    ax.set_ylim((minx[1],maxx[1]))
    ax.set_zlim((minx[2],maxx[2]))

    plt.savefig(outdir + '/static_3D.png')

if __name__ == "__main__":
    model_name = '0091_0001'
    print('Create geometry: ' + model_name)
    graphs = pp.load_graphs('../graphs/data/' + model_name + '.grph')[0]
    plot_geo_with_circles(graphs[0])
    plt.show()
