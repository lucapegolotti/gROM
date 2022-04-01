import os
import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy as v2n

def data_location():
    f = open('../data_location.txt', 'r')
    location = f.readline().strip()
    f.close()
    return location + '/'

def save_sequential_data(X, Y, mins, maxs, min_pressure, max_pressure, min_velocity, max_velocity,
                         model, stencil_size, center, var):
    directory = 'training_data/S_' + var + 'st' + str(stencil_size) + 'c' + str(center) + '/'
    create_directory('training_data/')
    create_directory(directory)
    np.save(directory + 'X.npy', X)
    np.save(directory + 'Y.npy', Y)
    np.save(directory + 'mins_x.npy', mins)
    np.save(directory + 'maxs_x.npy', maxs)
    np.save('training_data/bounds_pressure.npy', np.array((min_pressure,max_pressure)))
    np.save('training_data/bounds_velocity.npy', np.array((min_velocity,max_velocity)))
    model.save(directory)

def save_junctions_data(X, Y, mins, maxs, my, My, model, stencil_size, njunctions, var):
    directory = 'training_data/J_' + var + 'st' + str(stencil_size) + 'nj' + str(njunctions) + '/'
    create_directory('training_data/')
    create_directory(directory)
    np.save(directory + 'X.npy', X)
    np.save(directory + 'Y.npy', Y)
    np.save(directory + 'mins_x.npy', mins)
    np.save(directory + 'maxs_x.npy', maxs)
    np.save(directory + 'mins_y.npy', my)
    np.save(directory + 'maxs_y.npy', My)
    model.save(directory)

def create_directory(fdr_name):
    try:
        os.mkdir(fdr_name)
    except OSError as error:
        print('Directory ' + fdr_name + ' exists')

def collect_arrays(output, components = None):
    res = {}
    for i in range(output.GetNumberOfArrays()):
        name = output.GetArrayName(i)
        data = output.GetArray(i)
        if components == None:
            res[name] = v2n(data).astype(np.float32)
        else:
            res[name] = v2n(data)[:components].astype(np.float32)
    return res

def collect_points(output, components = None):
    if components == None:
        return v2n(output.GetData()).astype(np.float32)
    else:
        return v2n(output.GetData())[:components].astype(np.float32)

def get_all_arrays(geo, components = None):
    # collect all arrays
    cell_data = collect_arrays(geo.GetCellData(), components)
    point_data = collect_arrays(geo.GetPointData(), components)
    points = collect_points(geo.GetPoints(), components)
    return point_data, cell_data, points

def read_geo(fname):
    _, ext = os.path.splitext(fname)
    if ext == ".vtp":
        reader = vtk.vtkXMLPolyDataReader()
    elif ext == ".vtu":
        reader = vtk.vtkXMLUnstructuredGridReader()
    else:
        raise ValueError("File extension " + ext + " unknown.")
    reader.SetFileName(fname)
    reader.Update()
    return reader

def gather_pressures_flowrates(arrays):
    pressures   = {}
    velocities  = {}
    for array in arrays:
        if array[0:8] == 'pressure':
            time = float(array[9:])
            pressures[time] = arrays[array]
        if array[0:8] == 'velocity':
            time = float(array[9:])
            velocities[time] = arrays[array]

    return pressures, velocities
