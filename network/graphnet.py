import torch
import dgl
from torch.nn.modules.module import Module
from torch.nn import LayerNorm
from torch.nn import Linear
from torch.nn import LeakyReLU
import torch.nn.functional as F
import dgl.function as fn
from torch.nn import Dropout
import numpy as np
import preprocessing as pp

class MLP(Module):
    def __init__(self, in_feats, latent_space, out_feats, n_h_layers, normalize = True):
        bound_init = 1
        super().__init__()
        self.encoder_in = Linear(in_feats, latent_space, bias = True).float()
        # torch.nn.init.uniform_(self.encoder_in.weight, -bound_init, bound_init)
        self.encoder_out = Linear(latent_space, out_feats, bias = True).float()
        # torch.nn.init.uniform_(self.encoder_out.weight, -bound_init, bound_init)

        self.n_h_layers = n_h_layers
        self.hidden_layers = torch.nn.ModuleList()
        for i in range(n_h_layers):
            self.hidden_layers.append(Linear(latent_space, latent_space, bias = True).float())
            # torch.nn.init.uniform_(self.hidden_layers[i].weight, -bound_init, bound_init)

        # self.dropout = torch.nn.Dropout(0.2)

        self.normalize = normalize
        if self.normalize: # Try GroupNorm
            self.norm = LayerNorm(out_feats).float()

    def forward(self, inp):
        enc_features = self.encoder_in(inp)
        enc_features = F.leaky_relu(enc_features) # try leaky relu (0.1) or elu

        for i in range(self.n_h_layers):
            enc_features = self.hidden_layers[i](enc_features)
            enc_features = F.leaky_relu(enc_features)

        # enc_features = self.dropout(enc_features)
        enc_features = self.encoder_out(enc_features)

        if self.normalize:
            enc_features = self.norm(enc_features)

        return enc_features

class GraphNet(Module):
    def __init__(self, params):
        super(GraphNet, self).__init__()

        out_bc_encoder = 8

        self.encoder_inlet_edge = MLP(7, 16, out_bc_encoder, 1, True)
        self.encoder_outlet_edge = MLP(7, 16, out_bc_encoder, 1, True)

        self.encoder_branch_nodes = MLP(6,
                                        params['latent_size_mlp'],
                                        params['latent_size_gnn'],
                                        params['hl_mlp'],
                                        params['normalize'])

        self.encoder_b2b_edges = MLP(4,
                                     params['latent_size_mlp'],
                                     params['latent_size_gnn'],
                                     params['hl_mlp'],
                                     params['normalize'])

        self.encoder_junction_nodes = MLP(6,
                                          params['latent_size_mlp'],
                                          params['latent_size_gnn'],
                                          params['hl_mlp'],
                                          params['normalize'])
        self.encoder_j2j_edges = MLP(4,
                                     params['latent_size_mlp'],
                                     params['latent_size_gnn'],
                                     params['hl_mlp'],
                                     params['normalize'])

        self.encoder_b2j_edges = MLP(4,
                                     params['latent_size_mlp'],
                                     params['latent_size_gnn'],
                                     params['hl_mlp'],
                                     params['normalize'])

        # we share the same weights between branch -> junction and junction -> branch
        # edges
        self.encoder_j2b_edges = self.encoder_b2j_edges

        self.processor_b2b_edges = torch.nn.ModuleList()
        self.processor_j2j_edges = torch.nn.ModuleList()
        self.processor_b2j_edges = torch.nn.ModuleList()
        self.processor_j2b_edges = torch.nn.ModuleList()
        self.processor_branch_nodes = torch.nn.ModuleList()
        self.processor_junction_nodes = torch.nn.ModuleList()
        self.process_iters = params['process_iterations']
        for i in range(self.process_iters):
            def generate_proc_MLP(in_feat):
                return MLP(in_feat,
                           params['latent_size_mlp'],
                           params['latent_size_gnn'],
                           params['hl_mlp'],
                           params['normalize'])
            self.processor_b2b_edges.append(generate_proc_MLP(params['latent_size_gnn'] * 3))
            self.processor_j2j_edges.append(generate_proc_MLP(params['latent_size_gnn'] * 3))
            self.processor_b2j_edges.append(generate_proc_MLP(params['latent_size_gnn'] * 3))
            self.processor_j2b_edges.append(generate_proc_MLP(params['latent_size_gnn'] * 3))
            self.processor_branch_nodes.append(generate_proc_MLP(params['latent_size_gnn'] * 2 + out_bc_encoder * 2))
            self.processor_junction_nodes.append(generate_proc_MLP(params['latent_size_gnn'] * 2 + out_bc_encoder * 2))

        self.output_branch = MLP(params['latent_size_gnn'],
                                 params['latent_size_mlp'],
                                 2,
                                 params['hl_mlp'],
                                 False)

        self.output_junction = MLP(params['latent_size_gnn'],
                                   params['latent_size_mlp'],
                                   2,
                                   params['hl_mlp'],
                                   False)

        pressure_selector = np.array([[1],[0]]).astype(np.float32)
        flowrate_selector = np.array([[0],[1]]).astype(np.float32)
        self.pressure_selector = torch.tensor(pressure_selector)
        self.flowrate_selector = torch.tensor(flowrate_selector)

        self.params = params

        # self.dropout = Dropout(0.5)

    def set_normalization_coefs(self, coefs_dict):
        self.normalization_coefs = coefs_dict

    def encode_inlet_edge(self, edges):
        f1 = edges.data['e_features']
        f2 = edges.src['n_features']
        enc_edge = self.encoder_inlet_edge(torch.cat((f1, f2),dim=1))
        return {'inlet_info' : enc_edge}

    def encode_outlet_edge(self, edges):
        f1 = edges.data['e_features']
        f2 = edges.src['n_features']
        enc_edge = self.encoder_outlet_edge(torch.cat((f1, f2),dim=1))
        return {'outlet_info' : enc_edge}

    def encode_branch_nodes(self, nodes):
        f = nodes.data['features_c']
        enc_features = self.encoder_branch_nodes(f)
        return {'proc_node': enc_features}

    def encode_junction_nodes(self, nodes):
        f = nodes.data['features_c']
        enc_features = self.encoder_junction_nodes(f)
        return {'proc_node': enc_features}

    def encode_b2b_edges(self, edges):
        f = edges.data['e_features']
        enc_features = self.encoder_b2b_edges(f)
        return {'proc_edge': enc_features}

    def encode_j2j_edges(self, edges):
        f = edges.data['e_features']
        enc_features = self.encoder_j2j_edges(f)
        return {'proc_edge': enc_features}

    def encode_b2j_edges(self, edges):
        f = edges.data['e_features']
        enc_features = self.encoder_b2j_edges(f)
        return {'proc_edge': enc_features}

    def encode_j2b_edges(self, edges):
        f = edges.data['e_features']
        enc_features = self.encoder_j2b_edges(f)
        return {'proc_edge': enc_features}

    def process_edges(self, edges, processor):
        f1 = edges.data['proc_edge']
        f2 = edges.src['proc_node']
        f3 = edges.dst['proc_node']
        proc_edge = processor(torch.cat((f1, f2, f3),dim=1))
        # add residual connection
        proc_edge = proc_edge + f1
        return {'proc_edge' : proc_edge}

    def process_nodes(self, nodes, processor):
        f1 = nodes.data['proc_node']
        f2 = nodes.data['pe_sum']
        fin = nodes.data['inlet_info']
        fout = nodes.data['outlet_info']
        proc_node = processor(torch.cat((f1, f2, fin, fout), dim=1))
        # add residual connection
        proc_node = proc_node + f1
        return {'proc_node' : proc_node}

    def decode_branch(self, nodes):
        f = nodes.data['proc_node']
        # f = self.dropout(f)
        h = self.output_branch(f)
        return {'h' : h}

    def decode_junction(self, nodes):
        f = nodes.data['proc_node']
        # f = self.dropout(f)
        h = self.output_junction(f)
        return {'h' : h}

    def compute_continuity_loss(self, g, pred_branch, pred_junction, label_coefs, coefs_dict):
        g.nodes['branch'].data['pred_q'] = torch.matmul(pred_branch,
                                           self.flowrate_selector)

        nnodes = g.nodes['branch'].data['x'].shape[0] + \
                 g.nodes['junction'].data['x'].shape[0]
        global_pred = torch.zeros((nnodes, 2))

        global_pred[g.nodes['branch'].data['mask'],:] = pred_branch
        global_pred[g.nodes['junction'].data['mask'],:] = pred_junction

        # we have to bring the flowrate into the original units to compute
        # mass loss
        if label_coefs['normalization_type'] == 'min_max':
            g.nodes['branch'].data['pred_q'] = label_coefs['min'][1] + \
                                               g.nodes['branch'].data['pred_q'] * \
                                               (label_coefs['max'][1] - label_coefs['min'][1])
        elif label_coefs['normalization_type'] == 'standard':
            g.nodes['branch'].data['pred_q'] = label_coefs['mean'][1] + \
                                               g.nodes['branch'].data['pred_q'] * \
                                               label_coefs['std'][1]
        if coefs_dict['type'] == 'min_max':
            g.nodes['branch'].data['pred_q'] = g.nodes['branch'].data['pred_q'] * \
                                               float(coefs_dict['flowrate']['max'] - coefs_dict['flowrate']['min'])
        elif coefs_dict['type'] == 'standard':
            g.nodes['branch'].data['pred_q'] = g.nodes['branch'].data['pred_q'] * \
                                               float(coefs_dict['flowrate']['std'])

        # this is kind of useless because pred should already be averaged
        g.update_all(fn.copy_src('pred_q', 'm'), fn.mean('m', 'average'),
                     etype='branch_to_macro')
        g.update_all(fn.copy_src('average', 'm'), fn.sum('m', 'positive_flow'),
                     etype='macro_to_junction_positive')
        g.update_all(fn.copy_src('average', 'm'), fn.sum('m', 'negative_flow'),
                     etype='macro_to_junction_negative')

        diff = g.nodes['junction'].data['positive_flow'] - \
               g.nodes['junction'].data['negative_flow']

        try:
            mask = g.nodes['inlet'].data['mask']
            g.nodes['inlet'].data['pred_q'] = global_pred[mask]

            g.update_all(fn.copy_src('pred_q', 'm'), fn.sum('m', 'positive_flow_inlet'),
                         etype='inlet_to_junction_positive')
            diff = diff + g.nodes['junction'].data['positive_flow_inlet']
        except dgl._ffi.base.DGLError:
            pass

        try:
            mask = g.nodes['outlet'].data['mask']
            g.nodes['outlet'].data['pred_q'] = global_pred[mask]
            g.update_all(fn.copy_src('pred_q', 'm'), fn.sum('m', 'negative_flow_outlet'),
                         etype='outlet_to_junction_positive')
            diff = diff + g.nodes['junction'].data['positive_flow_outlet']
        except dgl._ffi.base.DGLError:
            pass

        return torch.mean(torch.abs(diff))

    # g must have bcs set
    def compute_bc_loss(self, g,
                        branch_features,
                        junct_features,
                        inlet_features,
                        outlet_features,
                        pred_branch,
                        pred_junction,
                        label_coefs):

        # compute branch next p
        pred_p = torch.matmul(pred_branch, self.pressure_selector).squeeze()
        pred_q = torch.matmul(pred_branch, self.flowrate_selector).squeeze()

        # we have to bring the flowrate into the original units to compute
        # mass loss
        if label_coefs['normalization_type'] == 'min_max':
            pred_p = label_coefs['min'][0] + \
                     pred_p * \
                     (label_coefs['max'][0] - label_coefs['min'][0])
            pred_q = label_coefs['min'][1] + \
                     pred_q * \
                     (label_coefs['max'][1] - label_coefs['min'][1])
        elif label_coefs['normalization_type'] == 'standard':
            pred_p = label_coefs['mean'][0] + pred_p * label_coefs['std'][0]
            pred_q = label_coefs['mean'][1] + pred_q * label_coefs['std'][1]

        pred_p_branch = pred_p + branch_features[:,0]
        pred_q_branch = pred_q + branch_features[:,1]

        # compute branch next q
        pred_p = torch.matmul(pred_junction, self.pressure_selector).squeeze()
        pred_q = torch.matmul(pred_junction, self.flowrate_selector).squeeze()

        # we have to bring the flowrate into the original units to compute
        # mass loss
        if label_coefs['normalization_type'] == 'min_max':
            pred_p = label_coefs['min'][0] + \
                     pred_p * \
                     (label_coefs['max'][0] - label_coefs['min'][0])
            pred_q = label_coefs['min'][1] + \
                     pred_q * \
                     (label_coefs['max'][1] - label_coefs['min'][1])
        elif label_coefs['normalization_type'] == 'standard':
            pred_p = label_coefs['mean'][0] + pred_p * label_coefs['std'][0]
            pred_q = label_coefs['mean'][1] + pred_q * label_coefs['std'][1]

        pred_p_junction = pred_p + junct_features[:,0]
        pred_q_junction = pred_q + junct_features[:,1]

        nnodes = pred_p_branch.shape[0] + pred_p_junction.shape[0]

        global_sol = torch.zeros((nnodes,2))

        global_sol[g.nodes['branch'].data['mask'], 0] = pred_p_branch
        global_sol[g.nodes['branch'].data['mask'], 1] = pred_q_branch
        global_sol[g.nodes['junction'].data['mask'], 0] = pred_p_junction
        global_sol[g.nodes['junction'].data['mask'], 1] = pred_q_junction

        columns = [0, 2]
        n_inlet = inlet_features.shape[0]
        pred_inlet = global_sol[g.nodes['inlet'].data['mask'],:]
        loss_inlet = ((pred_inlet - inlet_features[:,columns]) ** 2).mean()

        n_outlets = outlet_features.shape[0]
        pred_outlets = global_sol[g.nodes['outlet'].data['mask'],:]
        loss_outlets = ((pred_outlets - outlet_features[:,columns]) ** 2).mean()

        return (loss_inlet * n_inlet + loss_outlets * n_outlets) / (n_inlet + n_outlets)

    def compute_flowrate_correction(self, edges):
        f1 = edges.src['average_q']
        f2 = edges.dst['pred_q']
        return {'correction': f1 - f2}

    def forward(self, g, in_feat_branch, in_feat_junction, average_flowrate = False):
        g.nodes['branch'].data['features_c'] = in_feat_branch
        g.nodes['junction'].data['features_c'] = in_feat_junction

        g.apply_nodes(self.encode_branch_nodes, ntype='branch')
        g.apply_nodes(self.encode_junction_nodes, ntype='junction')

        g.apply_edges(self.encode_b2b_edges, etype='branch_to_branch')
        g.apply_edges(self.encode_j2j_edges, etype='junction_to_junction')
        g.apply_edges(self.encode_b2j_edges, etype='branch_to_junction')
        g.apply_edges(self.encode_j2b_edges, etype='junction_to_branch')

        g.apply_edges(self.encode_inlet_edge, etype='in_to_branch')
        g.apply_edges(self.encode_inlet_edge, etype='in_to_junction')

        g.update_all(fn.copy_e('inlet_info', 'm'), fn.sum('m', 'inlet_info'),
                               etype='in_to_branch')
        g.update_all(fn.copy_e('inlet_info', 'm'), fn.sum('m', 'inlet_info'),
                               etype='in_to_junction')

        g.apply_edges(self.encode_outlet_edge, etype='out_to_branch')
        g.apply_edges(self.encode_outlet_edge, etype='out_to_junction')

        g.update_all(fn.copy_e('outlet_info', 'm'), fn.sum('m', 'outlet_info'),
                               etype='out_to_branch')
        g.update_all(fn.copy_e('outlet_info', 'm'), fn.sum('m', 'outlet_info'),
                               etype='out_to_junction')

        for i in range(self.process_iters):
            def pe_b2b(edges):
                return self.process_edges(edges, self.processor_b2b_edges[i])
            def pe_j2j(edges):
                return self.process_edges(edges, self.processor_j2j_edges[i])
            def pe_b2j(edges):
                return self.process_edges(edges, self.processor_b2j_edges[i])
            def pe_j2b(edges):
                return self.process_edges(edges, self.processor_j2b_edges[i])
            def pn_b(nodes):
                return self.process_nodes(nodes, self.processor_branch_nodes[i])
            def pn_j(nodes):
                return self.process_nodes(nodes, self.processor_junction_nodes[i])

            # compute junction-branch interactions
            g.apply_edges(pe_b2j, etype='branch_to_junction')
            g.update_all(fn.copy_e('proc_edge', 'm'), fn.sum('m', 'pe_sum'),
                                   etype='branch_to_junction')

            g.apply_edges(pe_j2b, etype='junction_to_branch')
            g.update_all(fn.copy_e('proc_edge', 'm'), fn.sum('m', 'pe_sum'),
                                   etype='junction_to_branch')

            # compute interactions in branches
            g.apply_edges(pe_b2b, etype='branch_to_branch')
            # aggregate new edge features in nodes
            g.update_all(fn.copy_e('proc_edge', 'm'), fn.sum('m', 'pe_sum'),
                                   etype='branch_to_branch')
            g.apply_nodes(pn_b, ntype='branch')

            # compute interactions in junctions
            g.apply_edges(pe_j2j, etype='junction_to_junction')
            # aggregate new edge features in nodes
            g.update_all(fn.copy_e('proc_edge', 'm'), fn.sum('m', 'pe_sum'),
                                   etype='branch_to_branch')
            g.apply_nodes(pn_j, ntype='junction')

        g.apply_nodes(self.decode_branch, ntype='branch')
        g.apply_nodes(self.decode_junction, ntype='junction')

        if average_flowrate:
            # adjust flowrate to be constant in portions without branches
            g.nodes['branch'].data['pred_p'] = torch.matmul(g.nodes['branch'].data['h'],
                                               self.pressure_selector)
            g.nodes['branch'].data['pred_q'] = torch.matmul(g.nodes['branch'].data['h'],
                                               self.flowrate_selector)
            if average_flowrate:
                g.update_all(fn.copy_src('pred_q', 'm'), fn.mean('m', 'average_q'),
                             etype='branch_to_macro')
                g.apply_edges(self.compute_flowrate_correction,
                              etype='macro_to_branch')
                g.update_all(fn.copy_e('correction', 'm'), fn.mean('m', 'correction'),
                             etype='macro_to_branch')
                g.nodes['branch'].data['pred_q'] = g.nodes['branch'].data['pred_q'] + \
                                                   g.nodes['branch'].data['correction']

            return torch.cat((g.nodes['branch'].data['pred_p'],
                              g.nodes['branch'].data['pred_q']), dim=1), \
                   g.nodes['junction'].data['h']
        else:
            return g.nodes['branch'].data['h'], g.nodes['junction'].data['h']
