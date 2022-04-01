import numpy as np
import scipy
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random

class RawGraph:
    def __init__(self, points, params, debug = False):
        self.points = points

        resample_original_coeff = random.randint(params['min_resample_original'],
                                                 params['max_resample_original'])
        self.resample_original(resample_original_coeff, points)
        self.points = points[self.resampled_indices , :]
        self.params = params
        self.debug = debug

        self.divide()

        self.compute_h()
        self.find_junctions()
        self.resample()
        self.find_connectivity()
        self.construct_interpolation_matrices()

    def resample_original(self, coeff, points):
        npoints = points.shape[0]

        npoints_to_exclude = int(npoints - np.floor(npoints/coeff))

        indices = np.arange(0, npoints)
        for i in range(npoints_to_exclude):
            diff = np.linalg.norm(points[indices[1:],:] - points[indices[:-1],:], axis = 1)
            mindiff = np.min(diff)

            idx = np.where(np.abs(diff - mindiff) < 1e-14)[0]
            indices = np.delete(indices, idx)

        self.resampled_indices = indices

    def find_connectivity(self):
        # find connectivity
        nportions = len(self.resampled_portions)
        # every component stores idx of parent branch
        self.connectivity = [None] * nportions
        # we start from 1 because we assume that 0 is the global inlet
        for ipor1 in range(1,nportions):
            portion1 = self.resampled_portions[ipor1]
            mindist = np.infty
            minidxpor = None
            minidxpoint = None
            for ipor2 in range(0,nportions):
                if ipor1 != ipor2:
                    portion2 = self.resampled_portions[ipor2]
                    diffs = np.linalg.norm(portion2 - portion1[0,:],axis=1)
                    curmindist = np.min(diffs)
                    if curmindist < mindist:
                        minidxpoint = np.argmin(diffs)
                        mindist = curmindist
                        minidx = ipor2
            self.connectivity[ipor1] = [minidx, minidxpoint]

    def divide(self):
        # we assume inlet is the first point
        self.inlet = 0
        npoints = self.points.shape[0]

        dists = np.linalg.norm(self.points[1:,:] - self.points[:-1,:], axis = 1)

        self.outlets = np.where(dists > self.params['tol'])[0]

        if self.outlets[0] == self.inlet:
            # remove inlet from outlets if necessary
            self.outlets = np.delete(self.outlets, 0)

        # we assume last point is an outlet
        self.outlets = np.append(self.outlets, self.points.shape[0]-1)
        self.portions = self.partition(self.points)

        # use this piece of code to check if outlets are fine, if not
        # tune tol
        if self.debug:
            fig = plt.figure()
            ax = plt.axes(projection='3d')

            ax.scatter(self.points[:,0],
                       self.points[:,1],
                       self.points[:,2], s = 0.5)

            for i in self.outlets:
                ax.scatter(self.points[i,0],
                           self.points[i,1],
                           self.points[i,2], color = 'red')

            ax.scatter(self.points[0,0],
                       self.points[0,1],
                       self.points[0,2], color = 'blue')

            plt.show()

    def find_junctions(self):
        # there is one junction for each portion after 1 (the inlet)
        self.junctions = []
        nportions = len(self.portions)
        for rs_idx in range(1, nportions):
            mindists = []
            minidxs = []

            for other_idx in range(0, nportions):
                if rs_idx != other_idx:
                    diff = np.linalg.norm(self.portions[other_idx] - \
                                          self.portions[rs_idx][0, :], axis = 1)
                    # if the first node is the closest one than we'll form a closed
                    # portion of the geometry (this can happen in here -<) and
                    # we want to avoid this
                    diff[0] = np.infty
                    mindist = np.min(diff)
                    minidx = np.where(np.abs(diff - mindist) < 1e-14)[0][0]
                else:
                    mindist = np.infty
                    minidx = -1

                mindists.append(mindist)
                minidxs.append(minidx)

            mindists = np.array(mindists)
            totalmin = np.min(mindists)
            portionidx = np.where(np.abs(mindists - totalmin) < 1e-14)[0][0]
            junction = {}
            junction['portion_1'] = portionidx
            junction['portion_2'] = rs_idx
            junction['point_1'] = minidxs[portionidx]
            junction['point_2'] = 0

            self.junctions.append(junction)

    def compute_portion_length(self, portion):
        return np.sum(np.linalg.norm(portion[1:,:] - portion[:-1,:], axis=1))

    def compute_h(self):
        tot_length = 0
        tot_points = 0
        for portion in self.portions:
            tot_points += portion.shape[0]
            tot_length += self.compute_portion_length(portion)

        self.h = tot_length / tot_points

    def partition(self, array):
        if len(array.shape) == 1:
            array = np.expand_dims(array, axis = 1)
        slices = []
        points_to_remove = 3
        prev = 0
        for outlet in self.outlets:
            offset_in = 0
            if self.params['remove_caps'] and prev == 0:
                offset_in = points_to_remove
            offset_out = 0
            if self.params['remove_caps']:
                offset_out = points_to_remove
            slices.append(array[prev + offset_in:outlet+1 - offset_out,:])
            prev = outlet+1
        return slices

    def set_node_types(self, bifurcation_id):
        def fix_singular_points(slices, value0):
            for i in range(len(slices)):
                curslice = np.squeeze(slices[i])

                nslice = curslice.size

                i1 = []
                i2 = []
                invalue = True
                for j in range(nslice):
                    if curslice[j] != value0:
                        if invalue:
                            i1.append(j)
                        invalue = False
                    else:
                        if not invalue:
                            i2.append(j)
                        invalue = True

                # if this is the case then the last values are a chunk
                if len(i1) != len(i2):
                    i2.append(nslice)


                # set chunk to most common value
                for chunki in range(len(i1)):
                    cv = np.bincount(curslice[i1[chunki]:i2[chunki]]).argmax()
                    slices[i][i1[chunki]:i2[chunki]] = cv

        # when resampling, two bifurcations with different ids could be merged
        # into one if they're separate by a small branch. Here we loop over all
        # portions to check if bifurcation ids are consistent
        def check_inlets(ids):
            something_changed = True

            nportions = len(ids)
            while something_changed:
                something_changed = False
                for i in range(1, nportions):
                    conn = self.connectivity[i]
                    fatherid = ids[conn[0]][conn[1]]
                    if ids[i][0] != fatherid:
                        something_changed = True
                        j = 0
                        initialv = np.copy(ids[i][0])
                        while j != ids[i].size and ids[i][j] == initialv:
                            # set id equal to father
                            ids[i][j] = fatherid
                            j = j + 1

        bid, _ = self.project(bifurcation_id)
        for i in range(len(bid)):
            bid[i] = np.round(bid[i]).astype(int)
        fix_singular_points(bid, -1)
        check_inlets(bid)
        self.bifurcation_id, _ = self.project(bifurcation_id)
        for i in range(len(bid)):
            self.bifurcation_id[i] = np.round(self.bifurcation_id[i]).astype(int)
        fix_singular_points(self.bifurcation_id, -1)
        check_inlets(self.bifurcation_id)

        allids = set()
        for ids in bid:
            for id in ids:
                allids.add(int(id))

        degrees = {}
        count = 0
        for id in allids:
            degrees[id] = 0
            for bifid in bid:
                if id in np.squeeze(bifid).tolist():
                    degrees[id] = degrees[id] + 1

            count = count + 1

        degrees[-1] = 0

        for id in allids:
            for i in range(len(bid)):
                idxs = np.where(self.bifurcation_id[i] == id)[0]
                bid[i][idxs] = degrees[id]

        # fix singular points
        for i in range(len(bid)):
            curbid = np.squeeze(bid[i])

            # find consecutive chunks
            idxs = np.where(curbid != 0)[0]

            if len(idxs > 0):
                i1 = [idxs[0]]
                i2 = []
                for j in range(len(idxs)-2):
                    if idxs[j] + 1 != idxs[j+1]:
                        i2.append(idxs[j])
                        i1.append(idxs[j+1])

                i2.append(idxs[len(idxs)-1])

                # set chunk to most common value
                for chunki in range(len(i1)):
                    cv = np.bincount(curbid[i1[chunki]:i2[chunki]+1]).argmax()
                    bid[i][i1[chunki]:i2[chunki]+1] = cv

        self.nodes_type = bid

    def project(self, field):
        field = field[self.resampled_indices]
        proj_field = []
        sliced_field = self.partition(field)

        if self.debug:
            sliced_points = self.partition(self.points)
        for ifield in range(len(sliced_field)):
            weights = np.linalg.solve(self.interpolation_matrices[ifield],
                                      sliced_field[ifield])
            if weights[0] != weights[0]:
                msg = "Interpolation weights became nan. Maybe points in " + \
                      "original geometry were too close. Try increasing the " + \
                      "resample_original parameter for this geometry."
                raise ValueError(msg)
            proj_values = np.matmul(self.projection_matrices[ifield], weights)
            proj_field.append(proj_values)

            # check interpolation
            if self.debug:
                s = [0]
                for ip in range(sliced_points[ifield].shape[0]-1):
                    s.append(s[-1] + np.linalg.norm(sliced_points[ifield][ip+1,:] - sliced_points[ifield][ip,:]))

                s1 = [0]
                for ip in range(self.resampled_portions[ifield].shape[0]-1):
                    s1.append(s1[-1] + np.linalg.norm(self.resampled_portions[ifield][ip+1,:] - self.resampled_portions[ifield][ip,:]))

                fig = plt.figure()
                ax = plt.axes()

                ax.scatter(s, sliced_field[ifield], color='black')
                ax.scatter(s1, proj_values, s = 0.5, color='red')

                plt.show()

        return proj_field, sliced_field

    def project_multiple(self, fields):
        proj_fields = []
        nportions = len(self.portions)
        sliced_fields = [[] for i in range(nportions)]
        indices = [t for t in fields]
        indices.sort()
        for t in indices:
            r_field = fields[t][self.resampled_indices]
            sliced_field = self.partition(r_field)

            for ipor in range(nportions):
                sliced_fields[ipor].append(sliced_field[ipor])

        for ipor in range(nportions):
            sliced_fields[ipor] = np.squeeze(np.array(sliced_fields[ipor])).transpose()
            weights = np.linalg.solve(self.interpolation_matrices[ipor],
                                      sliced_fields[ipor])
            if weights[0,0] != weights[0,0]:
                msg = "Interpolation weights became nan. Maybe points in " + \
                      "original geometry were too close. Try increasing the " + \
                      "resample_original parameter for this geometry."
                raise ValueError(msg)
            proj_values = np.matmul(self.projection_matrices[ipor], weights)
            proj_fields.append(proj_values)

        return proj_fields, sliced_fields, indices

    def resample(self):
        coarsening = random.randint(self.params['min_coarsening'],
                                    self.params['max_coarsening'])
        resampled_portions = []
        resampled_junctions = []
        resampled_tangents = []
        count = 0
        for portion in self.portions:
            # compute h of the portion
            alength = self.compute_portion_length(portion)
            N = int(np.floor(alength / (coarsening * self.h)))

            if N < 2:
                raise ValueError("Too few points in portion, \
                                  decrease resample frequency.")

            tck, u = scipy.interpolate.splprep([portion[:,0],
                                                portion[:,1],
                                                portion[:,2]], s=0, k = 3)
            u_fine = np.linspace(0, 1, N + 1)
            x, y, z = interpolate.splev(u_fine, tck)
            r_portion = np.vstack((x,y,z)).transpose()

            # we add the bifurcation point not at the extremum
            for junction in self.junctions:
                if junction['portion_1'] == count:
                    point = self.portions[count][junction['point_1'],:]
                    diff = np.linalg.norm(r_portion - point, axis = 1)
                    mindiff = np.min(diff)
                    minidx = np.where(np.abs(diff - mindiff) < 1e-12)[0][0]

                    if mindiff < 1e-5:
                        idx = minidx
                    elif minidx == 0:
                        if diff[1] > np.linalg.norm(r_portion[1,:] - \
                                                    r_portion[0,:]):
                            r_portion = np.insert(r_portion, 0, point, axis = 0)
                            idx = 0
                        else:
                            r_portion = np.insert(r_portion, 1, point, axis = 0)
                            idx = 1
                    elif minidx == r_portion.shape[0]-1:
                        if diff[-1] > np.linalg.norm(r_portion[-1,:] - \
                                                     r_portion[-2,:]):
                            r_portion = np.append(r_portion, point, axis = 0)
                            idx = r_portion.shape[0]-1
                        else:
                            r_portion = np.insert(r_portion,
                                                  r_portion.shape[0]-1,
                                                  point, axis = 0)
                            idx = r_portion.shape[0]-2
                    else:
                        if diff[minidx+1] < diff[minidx-1]:
                            r_portion = np.insert(r_portion, minidx+1,
                                                  point, axis = 0)
                            idx = minidx+1
                        else:
                            r_portion = np.insert(r_portion, minidx,
                                                  point, axis = 0)
                            idx = minidx
                    r_junction = {}
                    r_junction['portion_1'] = count
                    r_junction['point_1'] = idx
                    r_junction['portion_2'] = junction['portion_2']
                    r_junction['point_2'] = junction['point_2']
                    resampled_junctions.append(r_junction)

            # we reinterpolate after junction injection
            tck, u = scipy.interpolate.splprep([r_portion[:,0],
                                                r_portion[:,1],
                                                r_portion[:,2]], s=0, k = 3)

            x, y, z = interpolate.splev(u, tck, der = 1)
            tangent_portion = np.vstack((x,y,z)).transpose()
            tangent_portion = tangent_portion
            tnorms = np.linalg.norm(tangent_portion, axis = 1)
            for i in range(tnorms.size):
                tangent_portion[i,:] = tangent_portion[i,:] / tnorms[i]

            resampled_portions.append(r_portion)
            resampled_tangents.append(tangent_portion)
            count = count + 1

        self.resampled_portions = resampled_portions
        self.resampled_junctions = resampled_junctions
        self.tangents = resampled_tangents

    def construct_interpolation_matrices(self):
        p_matrices = []
        i_matrices = []
        stdevcoeff = 50

        def kernel(nnorm, h):
            # for some reason, norm^2 doesn't work
            # https://en.wikipedia.org/wiki/Radial_basis_function_kernel
            return np.exp(-(nnorm / (2 * (h * stdevcoeff)**2)))

        # def kernel(nnorm, h):
        #     # 99% of the gaussian distribution is within 3 stdev from the mean
        #     stdv = stdevcoeff * h
        #     return np.exp(-0.5*(nnorm / (2 * (stdv)))**2)

        for ipor in range(0,len(self.portions)):
            N = self.resampled_portions[ipor].shape[0]
            M = self.portions[ipor].shape[0]
            new_matrix = np.zeros((N,M))

            hs = []
            for j in range(0,M):
                h1 = -1
                h2 = -1
                if j != M-1:
                    h1 = np.linalg.norm(self.portions[ipor][j+1,:] - \
                                        self.portions[ipor][j,:])
                if j != 0:
                    h2 = np.linalg.norm(self.portions[ipor][j,:] - \
                                        self.portions[ipor][j-1,:])
                h = np.max((h1, h2))
                hs.append(h)
            hs = np.array(hs)
            for i in range(0,N):
                n = np.linalg.norm(self.portions[ipor] -
                                   self.resampled_portions[ipor][i,:], axis = 1)

                new_matrix[i,:] = kernel(n, hs)

            p_matrices.append(new_matrix)

            N = self.portions[ipor].shape[0]
            M = N
            new_matrix = np.zeros((N,M))
            for i in range(0,N):
                n = np.linalg.norm(self.portions[ipor] - \
                                   self.portions[ipor][i,:], axis = 1)
                    # if n < 4 * hs[j] * stdevcoeff:
                new_matrix[i,:] = kernel(n,hs)

            i_matrices.append(new_matrix)

        self.projection_matrices = p_matrices
        self.interpolation_matrices = i_matrices

    def stack(self, a_list):
        res = a_list[0]

        for subfield in a_list[1:]:
            res = np.concatenate((res, subfield), axis = 0)

        return res

    def compute_arclenght(self):
        arclength = []
        offset = 0
        for portion in self.portions:
            arclength.append(offset)
            for npoint in range(portion.shape[0]-1):
                arclength.append(np.linalg.norm(portion[npoint+1,:] - portion[npoint,:]) + \
                                 arclength[-1])
            offset = arclength[-1]

        return arclength

    def compute_resampled_arclenght(self):
        arclength = []
        offset = 0
        for portion in self.resampled_portions:
            arclength.append(offset)
            for npoint in range(portion.shape[0]-1):
                arclength.append(np.linalg.norm(portion[npoint+1,:] - portion[npoint,:]) + \
                                 arclength[-1])
            offset = arclength[-1]

        return arclength

    def partition_and_stack_field(self, field):
        p_field, s_field = self.project(field)
        res = self.stack(p_field)
        s_field_stacked = self.stack(s_field)
        return res, s_field_stacked

    def partition_and_stack_fields(self, fields):
        p_field, s_field, indices = self.project_multiple(fields)
        res = self.stack(p_field)
        s_field_stacked = self.stack(s_field)

        res_l = {}
        s_field_stacked_l = {}
        count = 0
        for ind in indices:
            res_l[ind] = np.expand_dims(res[:,count], 1)
            s_field_stacked_l[ind] = np.expand_dims(s_field_stacked[:,count],1)
            count = count + 1

        return res_l, s_field_stacked_l

    def create_homogeneous_graph(self):
        def seq_array(minv, maxv):
            return np.expand_dims(np.arange(minv,maxv+1), axis = 1)
        nodes = self.resampled_portions[0]
        edges = np.concatenate((seq_array(0,nodes.shape[0]-2), \
                                seq_array(1,nodes.shape[0]-1)), axis = 1)

        offsets = [0, nodes.shape[0]]
        for portion in self.resampled_portions[1:]:
            nodes = np.concatenate((nodes,portion), axis = 0)
            new_edges = np.concatenate((seq_array(0,portion.shape[0]-2), \
                                        seq_array(1,portion.shape[0]-1)), axis = 1)
            new_edges = new_edges + offsets[-1]
            edges = np.concatenate((edges, new_edges), axis = 0)
            offsets.append(offsets[-1] + portion.shape[0])

        # add bifurcations
        for junction in self.resampled_junctions:
            node1 = offsets[junction['portion_1']] + junction['point_1']
            node2 = offsets[junction['portion_2']] + junction['point_2']
            edges = np.concatenate((edges, np.array([[node1, node2]])), axis = 0)

        inlet_index = 0
        outlet_indices = [a - 1 for a in offsets[1:]]

        nodes_type = self.stack(self.nodes_type)
        bifurcation_id = self.stack(self.bifurcation_id)
        tangents = self.stack(self.tangents)

        return nodes, edges, inlet_index, outlet_indices, nodes_type, tangents, bifurcation_id

    def create_heterogeneous_graph(self):
        # Dijkstra's algorithm
        def dijkstra_algorithm(nodes, edges, index):
            # make edges bidirectional for simplicity
            nnodes = nodes.shape[0]
            tovisit = np.arange(0,nnodes)
            dists = np.ones((nnodes)) * np.infty
            prevs = np.ones((nnodes)) * (-1)
            b_edges = np.concatenate((edges, \
                      np.array([edges[:,1],edges[:,0]]).transpose()), axis = 0)

            dists[index] = 0
            while len(tovisit) != 0:
                minindex = -1
                minlen = np.infty
                for iinde in range(len(tovisit)):
                    if dists[tovisit[iinde]] < minlen:
                        minindex = iinde
                        minlen = dists[tovisit[iinde]]

                curindex = tovisit[minindex]
                tovisit = np.delete(tovisit, minindex)

                # find neighbors of curindex
                inb = b_edges[np.where(b_edges[:,0] == curindex)[0],1]

                for neib in inb:
                    if np.where(tovisit == neib)[0].size != 0:
                        alt = dists[curindex] + np.linalg.norm(nodes[curindex,:] - \
                              nodes[neib,:])
                        if alt < dists[neib]:
                            dists[neib] = alt
                            prevs[neib] = curindex
            if np.max(dists) == np.infty:
                fig = plt.figure()
                ax = plt.axes(projection='3d')
                ax.scatter(nodes[:,0], nodes[:,1], nodes[:,2], s = 0.5, c = 'black')
                idx = np.where(dists > 1e30)[0]
                ax.scatter(nodes[idx,0], nodes[idx,1], nodes[idx,2], c = 'red')
                plt.show()
                raise ValueError("Distance in Dijkstra is infinite for some reason. You can maybe adjust resample parameters.")

            return dists, prevs

        nodes, edges, inlet_index, \
        outlet_indices, nodes_type, tangents, \
        bifurcation_id = self.create_homogeneous_graph()

        nnodes = nodes.shape[0]
        branch_mask = np.where(nodes_type == 0)[0]
        junct_mask = np.where(nodes_type != 0)[0]

        nbranch_nodes = branch_mask.size
        njunct_nodes = junct_mask.size

        # duplicate inlet
        x = np.expand_dims(nodes[inlet_index,:], axis=0)
        distances_inlet, _ = dijkstra_algorithm(nodes, edges, inlet_index)
        distances_inlet_branch = distances_inlet[branch_mask]
        distances_inlet_junct = distances_inlet[junct_mask]

        inlet_edges_branch = np.zeros((nbranch_nodes,2))
        inlet_edges_branch[:,1] = np.arange(nbranch_nodes)
        inlet_edges_junct = np.zeros((njunct_nodes,2))
        inlet_edges_junct[:,1] = np.arange(njunct_nodes)

        inlet_physical_contiguous = np.zeros(nnodes)
        inlet_physical_contiguous[inlet_index] = 1
        inlet_physical_contiguous_branch = inlet_physical_contiguous[branch_mask]
        inlet_physical_contiguous_junct = inlet_physical_contiguous[junct_mask]

        map_inlet = {inlet_index: 0}

        inlet_dict = {'edges_branch': inlet_edges_branch.astype(int),
                      'edges_junct': inlet_edges_junct.astype(int), \
                      'distance_branch': distances_inlet_branch, \
                      'distance_junct': distances_inlet_junct, \
                      'x': np.expand_dims(nodes[inlet_index,:],axis=0), \
                      'mask': np.array([inlet_index]),
                      'physical_same_branch': inlet_physical_contiguous_branch.astype(int),
                      'physical_same_junct': inlet_physical_contiguous_junct.astype(int)}

        # duplicate outlets
        x = nodes[outlet_indices,:]
        outlet_edges = np.zeros((0,2))
        distances_outlets = np.zeros((0))
        outlet_physical_contiguous = np.zeros((0))
        map_outlet = {}
        for out_index in range(len(outlet_indices)):
            map_outlet[outlet_indices[out_index]] = out_index
            curoutedge = np.zeros((nnodes,2))
            curoutedge[:,1] = np.arange(0, nnodes)
            curoutedge[:,0] = out_index
            outlet_edges = np.concatenate((outlet_edges, curoutedge), axis = 0)
            curdistances, _ = dijkstra_algorithm(nodes, edges,\
                                                 outlet_indices[out_index])
            distances_outlets = np.concatenate((distances_outlets, curdistances))
            cur_opc = np.zeros(nnodes).astype(int)
            cur_opc[outlet_indices[out_index]] == 1
            outlet_physical_contiguous = np.concatenate((outlet_physical_contiguous,
                                                         cur_opc))

        # we select the edges and properties such that each inner node is
        # only connected to one outlet (based on smaller distance)
        single_connection_mask = []
        for inod in range(nnodes):
            mindist = np.amin(distances_outlets[inod::nnodes])
            indxs = np.where(np.abs(distances_outlets - mindist) < 1e-14)[0]
            added = False
            for idx in indxs.tolist():
                if (idx - inod) % nnodes == 0:
                    added = True
                    single_connection_mask.append(int(idx))

        outlet_edges = outlet_edges[single_connection_mask,:]
        outlet_edges_branch = outlet_edges[branch_mask, :]
        outlet_edges_branch[:,1] = np.arange(0, nbranch_nodes)
        outlet_edges_junct = outlet_edges[junct_mask, :]
        outlet_edges_junct[:,1] = np.arange(0, njunct_nodes)

        distances_outlets = distances_outlets[single_connection_mask]
        distances_outlets_branch = distances_outlets[branch_mask]
        distances_outlets_junct = distances_outlets[junct_mask]

        outlet_physical_contiguous = outlet_physical_contiguous[single_connection_mask]
        outlet_physical_contiguous_branch = outlet_physical_contiguous[branch_mask]
        outlet_physical_contiguous_junct = outlet_physical_contiguous[junct_mask]

        outlet_dict = {'edges_branch': outlet_edges_branch.astype(int), \
                       'edges_junct': outlet_edges_junct.astype(int), \
                       'distance_branch': distances_outlets_branch, \
                       'distance_junct': distances_outlets_junct, \
                       'x': nodes[outlet_indices,:], \
                       'mask': np.array(outlet_indices), \
                       'physical_same_branch': outlet_physical_contiguous_branch, \
                       'physical_same_junct': outlet_physical_contiguous_junct}

        edges_b2b = []
        edges_j2j = []
        edges_b2j = []
        edges_j2b = []

        branch_mask_list = branch_mask.tolist()
        junct_mask_list = junct_mask.tolist()

        branch_map = {}
        count = 0
        for ib in branch_mask_list:
            branch_map[ib] = count
            count = count + 1

        junct_map = {}
        count = 0
        for ij in junct_mask_list:
            junct_map[ij] = count
            count = count + 1

        for edge in edges:
            if edge[0] in branch_mask_list and edge[1] in branch_mask_list:
                edges_b2b.append([branch_map[edge[0]], branch_map[edge[1]]])
            elif edge[0] in junct_mask_list and edge[1] in junct_mask_list:
                edges_j2j.append([junct_map[edge[0]], junct_map[edge[1]]])
            else:
                if edge[0] in branch_mask_list:
                    e0 = edge[0]
                if edge[1] in branch_mask_list:
                    e0 = edge[1]
                if edge[0] in junct_mask_list:
                    e1 = edge[0]
                if edge[1] in junct_mask_list:
                    e1 = edge[1]

                edges_b2j.append([branch_map[e0], junct_map[e1]])

        edges_b2b = np.array(edges_b2b)
        # make it bidirectional
        edges_b2b = np.concatenate((edges_b2b,np.array([edges_b2b[:,1],edges_b2b[:,0]]).transpose()),axis = 0)

        edges_j2j = np.array(edges_j2j)
        # make it bidirectional
        edges_j2j = np.concatenate((edges_j2j,np.array([edges_j2j[:,1],edges_j2j[:,0]]).transpose()),axis = 0)

        edges_b2j = np.array(edges_b2j)
        edges_j2b = np.array([edges_b2j[:,1],edges_b2j[:,0]]).transpose()

        nodes_branch = nodes[branch_mask,:]
        nodes_junct = nodes[junct_mask,:]

        nedges_b2b = edges_b2b.shape[0]
        pos_b2b = np.zeros((nedges_b2b, 4))
        for iedg in range(nedges_b2b):
            pos_b2b[iedg,0:3] = nodes_branch[edges_b2b[iedg,1],:] - nodes_branch[edges_b2b[iedg,0],:]
            pos_b2b[iedg,3] = np.linalg.norm(pos_b2b[iedg,0:3])
            # we normalize the vector distance
            pos_b2b[iedg,0:3] = pos_b2b[iedg,0:3] / pos_b2b[iedg,3]

        nedges_j2j = edges_j2j.shape[0]
        pos_j2j = np.zeros((nedges_j2j, 4))
        for iedg in range(nedges_j2j):
            pos_j2j[iedg,0:3] = nodes_junct[edges_j2j[iedg,1],:] - nodes_junct[edges_j2j[iedg,0],:]
            pos_j2j[iedg,3] = np.linalg.norm(pos_j2j[iedg,0:3])
            pos_j2j[iedg,0:3] = pos_j2j[iedg,0:3] / pos_j2j[iedg,3]

        nedges_b2j = edges_b2j.shape[0]
        pos_b2j = np.zeros((nedges_b2j, 4))
        for iedg in range(nedges_b2j):
            pos_b2j[iedg,0:3] = nodes_junct[edges_b2j[iedg,1],:] - nodes_branch[edges_b2j[iedg,0],:]
            pos_b2j[iedg,3] = np.linalg.norm(pos_b2j[iedg,0:3])
            pos_b2j[iedg,0:3] = pos_b2j[iedg,0:3] / pos_b2j[iedg,3]

        pos_j2b = pos_b2j
        pos_j2b[:,0:3] = -pos_j2b[:,0:3]

        tangents_branch = tangents[branch_mask,:]

        branch_dict = {'edges': edges_b2b,
                       'pos': pos_b2b,
                       'x': nodes_branch,
                       'tangent': tangents_branch,
                       'branch_to_junct': edges_b2j,
                       'pos_branch_to_junct': pos_b2j,
                       'mask': branch_mask}

        tangents_junct = tangents[junct_mask,:]
        junct_dict = {'edges': edges_j2j,
                      'pos': pos_j2j,
                      'x': nodes_junct,
                      'tangent': tangents_junct,
                      'junct_to_branch': edges_j2b,
                      'pos_junct_to_branch': pos_j2b,
                      'node_type': nodes_type[junct_mask],
                      'mask': junct_mask}

        # let's find macro nodes and connectivity
        macro_nodes = np.ones(nnodes) * (-1)

        count_macro = 0
        in_0_portion = False
        for i in range(nnodes):
            if nodes_type[i] == 0:
                macro_nodes[i] = count_macro
                in_0_portion = True
            else:
                if in_0_portion:
                    in_0_portion = False
                    count_macro = count_macro + 1

        # find junctions and inflows - outflows
        bifs_ids = set()
        for i in range(nnodes):
            if bifurcation_id[i] != -1:
                bifs_ids.add(int(bifurcation_id[i]))

        inflows = []
        outflows = []
        for bif_id in bifs_ids:
            curinflows = []
            curoutflows = []
            for i in range(nnodes):
                if bifurcation_id[i] == bif_id:
                    if i == inlet_index:
                        curinflows.append(('inlet',i))
                    elif i in outlet_indices:
                        curoutflows.append(('outlet',i))
                    elif bifurcation_id[i-1] != bif_id and i-1 not in outlet_indices:
                        curinflows.append(('inner',i-1))
                    elif bifurcation_id[i+1] != bif_id and i+1 not in outlet_indices:
                        curoutflows.append(('inner',i+1))
            inflows.append(curinflows)
            outflows.append(curoutflows)

        inner_to_macro = []
        for i in range(macro_nodes.shape[0]):
            if macro_nodes[i] != -1:
                inner_to_macro.append([i, int(macro_nodes[i])])

        inlet_to_junction_positive = []
        macro_to_junction_positive = []
        for bif_id in range(len(inflows)):
            for infl in inflows[bif_id]:
                if infl[0] == 'inner':
                    macro_to_junction_positive.append([int(macro_nodes[infl[1]]), bif_id])
                elif infl[0] == 'inlet':
                    inlet_to_junction_positive.append([int(map_inlet[infl[1]]), bif_id])

        outlet_to_junction_negative = []
        macro_to_junction_negative = []
        for bif_id in range(len(inflows)):
            for outfl in outflows[bif_id]:
                if outfl[0] == 'inner':
                    macro_to_junction_negative.append([int(macro_nodes[outfl[1]]), bif_id])
                elif outfl[0] == 'outlet':
                    outlet_to_junction_negative.append([int(map_outlet[outfl[1]]), bif_id])

        branch_to_macro = np.array(inner_to_macro)
        # convert indices to global to braches
        for i in range(branch_to_macro.shape[0]):
            branch_to_macro[i,0] = branch_map[branch_to_macro[i,0]]

        macro_to_branch = np.array([branch_to_macro[:,1], branch_to_macro[:,0]]).transpose()
        inlet_to_junction_positive = np.array(inlet_to_junction_positive)
        macro_to_junction_positive = np.array(macro_to_junction_positive)
        outlet_to_junction_negative = np.array(outlet_to_junction_negative)
        macro_to_junction_negative = np.array(macro_to_junction_negative)

        macro_structure = {'branch_to_macro': branch_to_macro,
                           'macro_to_branch': macro_to_branch,
                           'macro_to_junction_positive': macro_to_junction_positive,
                           'macro_to_junction_negative': macro_to_junction_negative,
                           'inlet_to_junction_positive': inlet_to_junction_positive,
                           'outlet_to_junction_negative': outlet_to_junction_negative}

        return branch_dict, junct_dict, inlet_dict, outlet_dict, macro_structure

    def show(self):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        cmap = cm.get_cmap("tab20")

        m = np.infty
        M = np.NINF

        # colorby = self.nodes_type
        colorby = self.bifurcation_id

        for i in range(len(self.portions)):
            m = np.min([np.min(colorby[i]), M])
            M = np.max([np.max(colorby[i]), M])

        for i in range(len(self.portions)):
            portion = self.resampled_portions[i]
            nnodes = portion.shape[0]

            colors = np.zeros((nnodes,3))
            colors = cmap((colorby[i] - m) / (M - m))

            for j in range(self.nodes_type[i].shape[0]):
                if (colorby[i][j] - m) / (M - m) == 0:
                    ax.scatter(self.resampled_portions[i][j,0],
                               self.resampled_portions[i][j,1],
                               self.resampled_portions[i][j,2], color = 'black', depthshade=0)
                else:
                    ax.scatter(self.resampled_portions[i][j,0],
                               self.resampled_portions[i][j,1],
                               self.resampled_portions[i][j,2], color = colors[j,:], depthshade=0)

        plt.show()
