import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from .layers import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GenNet_tanh(nn.Module):
    '''
    Class to define the generator
    Feature Extraction Module (FEM) + Embedding Derivation Module (EDM) + Embedding Aggregation Module (EAM)
    Embedding aggregation w/ tanh
    '''
    def __init__(self, FEM_dims, EDM_dims, EAM_dims, dropout_rate):
        super(GenNet_tanh, self).__init__()
        # ====================
        self.FEM_dims = FEM_dims # Layer configuration of FEM
        self.EDM_dims = EDM_dims # Layer configuration of EDM
        self.EAM_dims = EAM_dims # Layer configuration of EAM
        self.dropout_rate = dropout_rate # Dropout rate
        # ==========
        self.dropout_layer = nn.Dropout(p=self.dropout_rate)
        # ==========
        # Feature Extraction Module (FEM)
        self.num_FEM_layers = len(self.FEM_dims)-1
        self.FEM_layers = nn.ModuleList()
        for l in range(self.num_FEM_layers):
            self.FEM_layers.append(nn.Linear(in_features=self.FEM_dims[l], out_features=self.FEM_dims[l+1]))
        # ===========
        # Embedding Derivation Module (EDM), i.e., stacked GNN-RNN cell
        self.num_EDM_layers = len(self.EDM_dims)-1
        self.EDM_layers = nn.ModuleList()
        for l in range(self.num_EDM_layers):
            # GNN
            self.EDM_layers.append(GraphNeuralNetwork(input_dim=self.EDM_dims[l], output_dim=self.EDM_dims[l+1],
                                                          dropout_rate=self.dropout_rate))
            # (Inductive) RNN
            self.EDM_layers.append(IGRU(input_dim=self.EDM_dims[l+1], output_dim=self.EDM_dims[l+1],
                                            dropout_rate=self.dropout_rate))
            # Attentive Node Aligning Unit
            self.EDM_layers.append(AttNodeAlign(feat_dim=self.FEM_dims[-1], hid_dim=self.EDM_dims[l+1],
                                              dropout_rate=self.dropout_rate))
        # ==========
        # Embedding Aggregation Module (EAM)
        self.num_EAM_layers = len(self.EAM_dims)-1
        # Embedding mapping network
        self.emb_layers = nn.ModuleList()
        for l in range(self.num_EAM_layers):
            self.emb_layers.append(nn.Linear(in_features=self.EAM_dims[l], out_features=self.EAM_dims[l+1]))
        # Scaling network
        self.scal_layers = nn.ModuleList()
        for l in range(self.num_EAM_layers):
            self.scal_layers.append(nn.Linear(in_features=self.EAM_dims[l], out_features=self.EAM_dims[l+1]))

    def forward(self, sup_list, feat_list, noise_list, align_list, num_nodes_list, lambd, pred_flag=True):
        '''
        Rewrite the forward function
        :param sup_list: list of GNN supports (normalized adjacency matrices) w.r.t. each input snapshot (l)
        :param feat_list: list of node attributes w.r.t. each snapshot (input & output) (l+1)
        :param noise_list: list of noise input (l)
        :param align_list: list of align matrices (l)
        :param num_nodes_list: list of #nodes w.r.t. each snapshot (input & output) (l+1)
        :param lambd: parameter of attentive aligning unit
        :param pred_flag: boolean flag for the prediction mode (i.e., only derive the prediction result of next snapshot)
        :return: list of prediction results
        '''
        # ====================
        win_size = len(sup_list) # Window size (#historical snapshots)
        # ==========
        # Feature Extraction Module (FEM)
        FEM_input_list = feat_list
        FEM_output_list = None
        for l in range(self.num_FEM_layers):
            FEM_layer = self.FEM_layers[l]
            FEM_output_list = []
            for t in range(win_size+1):
                FEM_input = FEM_input_list[t]
                FEM_output = FEM_layer(FEM_input)
                FEM_output = torch.relu(FEM_output)
                FEM_output_list.append(FEM_output)
            FEM_input_list = FEM_output_list

        # ====================
        # Embedding Derivation Module (EDM)
        EDM_input_list = []
        align_output_list = None
        for t in range(win_size):
            EDM_input = FEM_output_list[t]
            noise = noise_list[t]
            EDM_input = torch.cat((EDM_input, noise), dim=1) # Concatenate GNN outputs with noise inputs
            EDM_input_list.append(EDM_input)
        for l in range(0, self.num_EDM_layers*3, 3):
            # ==========
            GNN_layer = self.EDM_layers[l]
            GNN_output_list = []
            for t in range(win_size):
                sup = sup_list[t] # GNN support w.r.t. current snapshot
                feat = EDM_input_list[t] # Feature input of current snapshot
                GNN_output = GNN_layer(feat, sup)
                GNN_output_list.append(GNN_output)
            # ==========
            RNN_layer = self.EDM_layers[l+1]
            align_unit = self.EDM_layers[l+2]
            hid_dim = self.EDM_dims[int(l/3)+1] # Dimension of the embedding in current layer
            pre_state = torch.zeros(num_nodes_list[0], hid_dim).to(device) # Previous state of (inductive) GRU
            RNN_output_list = []
            align_output_list = []
            for t in range(win_size):
                RNN_input = GNN_output_list[t]
                RNN_output = RNN_layer(pre_state, RNN_input)
                pre_state = align_unit(align_list[t], FEM_output_list[t], FEM_output_list[t+1], RNN_output, lambd=lambd)
                align_output_list.append(pre_state)
                RNN_output_list.append(RNN_output)
            RNN_output_list.append(pre_state)
            # ==========
            EDM_input_list = RNN_output_list
        EDM_output_list = align_output_list

        # ====================
        # Embedding Aggregation Module (EAM)
        if pred_flag==True: # Prediction mode, i.e., only derive the prediction result of next snapshot
            emb = EDM_output_list[-1]
            feat = FEM_output_list[-1]
            emb_cat = torch.cat((emb, feat), dim=1)
            # ==========
            emb_input = emb_cat
            emb_output = None
            for l in range(self.num_EAM_layers):
                emb_layer = self.emb_layers[l]
                emb_output = emb_layer(emb_input)
                emb_output = torch.tanh(emb_output)
                emb_input = emb_output
            emb = emb_output
            emb = F.normalize(emb, dim=0, p=2)
            # ==========
            scal_input = emb_cat
            scal_output = None
            for l in range(self.num_EAM_layers):
                scal_layer = self.scal_layers[l]
                scal_output = scal_layer(scal_input)
                scal_output = torch.sigmoid(scal_output)
                scal_input = scal_output
            scal = torch.mm(scal_output, scal_output.t())
            # ==========
            num_nodes = num_nodes_list[-1]
            emb_src = torch.reshape(emb, (1, num_nodes, self.EAM_dims[-1]))
            emb_dst = torch.reshape(emb, (num_nodes, 1, self.EAM_dims[-1]))
            adj_est = -torch.sum((emb_src-emb_dst)**2, dim=2)
            adj_est = 1+torch.tanh(torch.mul(adj_est, scal))

            return [adj_est]
        # ====================
        else: # pred_flag==False
            # ==========
            adj_est_list = [] # List of the prediction results (i.e., estimated adjacency matrices)
            for t in range(win_size):
                # ==========
                emb = EDM_output_list[t]
                feat = FEM_output_list[t+1]
                emb_cat = torch.cat((emb, feat), dim=1)
                # ==========
                emb_input = emb_cat
                emb_output = None
                for l in range(self.num_EAM_layers):
                    emb_layer = self.emb_layers[l]
                    emb_output = emb_layer(emb_input)
                    emb_output = torch.tanh(emb_output)
                    emb_input = emb_output
                emb = emb_output
                emb = F.normalize(emb, dim=0, p=2)
                # ==========
                scal_input = emb_cat
                scal_output = None
                for l in range(self.num_EAM_layers):
                    scal_layer = self.scal_layers[l]
                    scal_output = scal_layer(scal_input)
                    scal_output = torch.sigmoid(scal_output)
                    scal_input = scal_output
                scal = torch.mm(scal_output, scal_output.t())
                # ==========
                num_nodes = num_nodes_list[t+1]
                emb_src = torch.reshape(emb, (1, num_nodes, self.EAM_dims[-1]))
                emb_dst = torch.reshape(emb, (num_nodes, 1, self.EAM_dims[-1]))
                adj_est = -torch.sum((emb_src-emb_dst)**2, dim=2)
                adj_est = 1+torch.tanh(torch.mul(adj_est, scal))
                # ==========
                adj_est_list.append(adj_est)

            return adj_est_list

class GenNet_exp(nn.Module):
    '''
    Class to define the generator
    Feature Extraction Module (FEM) + Embedding Derivation Module (EDM) + Embedding Aggregation Module (EAM)
    Embedding aggregation w/ exp
    '''
    def __init__(self, FEM_dims, EDM_dims, EAM_dims, dropout_rate):
        super(GenNet_exp, self).__init__()
        # ====================
        self.FEM_dims = FEM_dims # Layer configuration of FEM
        self.EDM_dims = EDM_dims # Layer configuration of EDM
        self.EAM_dims = EAM_dims # Layer configuration of EAM
        self.dropout_rate = dropout_rate # Dropout rate
        # ==========
        self.dropout_layer = nn.Dropout(p=self.dropout_rate)
        # ==========
        # Feature Extraction Module (FEM)
        self.num_FEM_layers = len(self.FEM_dims)-1
        self.FEM_layers = nn.ModuleList()
        for l in range(self.num_FEM_layers):
            self.FEM_layers.append(nn.Linear(in_features=self.FEM_dims[l], out_features=self.FEM_dims[l+1]))
        # ===========
        # Embedding Derivation Module (EDM)
        self.num_EDM_layers = len(self.EDM_dims)-1
        self.EDM_layers = nn.ModuleList()
        for l in range(self.num_EDM_layers):
            # GNN
            self.EDM_layers.append(GraphNeuralNetwork(input_dim=self.EDM_dims[l], output_dim=self.EDM_dims[l+1],
                                                          dropout_rate=self.dropout_rate))
            # (Inductive) RNN
            self.EDM_layers.append(IGRU(input_dim=self.EDM_dims[l+1], output_dim=self.EDM_dims[l+1],
                                            dropout_rate=self.dropout_rate))
            # Attentive Node Aligning Unit
            self.EDM_layers.append(AttNodeAlign(feat_dim=self.FEM_dims[-1], hid_dim=self.EDM_dims[l+1],
                                              dropout_rate=self.dropout_rate))
        # ==========
        # Embedding Aggregation Module (EAM)
        self.num_EAM_layers = len(self.EAM_dims)-1
        # Embedding mapping network
        self.emb_layers = nn.ModuleList()
        for l in range(self.num_EAM_layers):
            self.emb_layers.append(nn.Linear(in_features=self.EAM_dims[l], out_features=self.EAM_dims[l+1]))
        # Scaling network
        self.scal_layers = nn.ModuleList()
        for l in range(self.num_EAM_layers):
            self.scal_layers.append(nn.Linear(in_features=self.EAM_dims[l], out_features=self.EAM_dims[l+1]))

    def forward(self, sup_list, feat_list, noise_list, align_list, num_nodes_list, lambd, pred_flag=True):
        '''
        Rewrite the forward function
        :param sup_list: list of GNN supports (normalized adjacency matrices) w.r.t. each input snapshot (l)
        :param feat_list: list of node attributes w.r.t. each snapshot (input & output) (l+1)
        :param noise_list: list of noise input (l)
        :param align_list: list of align matrices (l)
        :param num_nodes_list: list of #nodes w.r.t. each snapshot (input & output) (l+1)
        :param lambd: parameter of attentive aligning unit
        :param pred_flag: boolean flag for the prediction mode (i.e., only derive the prediction result of next snapshot)
        :return: list of prediction results
        '''
        # ====================
        win_size = len(sup_list) # Window size (#historical snapshots)
        # ==========
        # Feature Extraction Module (FEM)
        FEM_input_list = feat_list
        FEM_output_list = None
        for l in range(self.num_FEM_layers):
            FEM_layer = self.FEM_layers[l]
            FEM_output_list = []
            for t in range(win_size+1):
                FEM_input = FEM_input_list[t]
                FEM_output = FEM_layer(FEM_input)
                FEM_output = torch.relu(FEM_output)
                FEM_output_list.append(FEM_output)
            FEM_input_list = FEM_output_list

        # ====================
        # Embedding Derivation Module (EDM)
        EDM_input_list = []
        align_output_list = None
        for t in range(win_size):
            EDM_input = FEM_output_list[t]
            noise = noise_list[t]
            EDM_input = torch.cat((EDM_input, noise), dim=1) # Concatenate GNN outputs with noise inputs
            EDM_input_list.append(EDM_input)
        for l in range(0, self.num_EDM_layers*3, 3):
            # ==========
            GNN_layer = self.EDM_layers[l]
            GNN_output_list = []
            for t in range(win_size):
                sup = sup_list[t] # GNN support w.r.t. current snapshot
                feat = EDM_input_list[t] # Feature input of current snapshot
                GNN_output = GNN_layer(feat, sup)
                GNN_output_list.append(GNN_output)
            # ==========
            RNN_layer = self.EDM_layers[l+1]
            align_unit = self.EDM_layers[l+2]
            hid_dim = self.EDM_dims[int(l/3)+1] # Dimension of the embedding in current layer
            pre_state = torch.zeros(num_nodes_list[0], hid_dim).to(device) # Previous state of (inductive) GRU
            RNN_output_list = []
            align_output_list = []
            for t in range(win_size):
                RNN_input = GNN_output_list[t]
                RNN_output = RNN_layer(pre_state, RNN_input)
                pre_state = align_unit(align_list[t], FEM_output_list[t], FEM_output_list[t+1], RNN_output, lambd=lambd)
                align_output_list.append(pre_state)
                RNN_output_list.append(RNN_output)
            RNN_output_list.append(pre_state)
            # ==========
            EDM_input_list = RNN_output_list
        EDM_output_list = align_output_list

        # ====================
        # Embedding Aggregation Module (EAM)
        if pred_flag==True: # Prediction mode, i.e., only derive the prediction result of next snapshot
            emb = EDM_output_list[-1]
            feat = FEM_output_list[-1]
            emb_cat = torch.cat((emb, feat), dim=1)
            # ==========
            emb_input = emb_cat
            emb_output = None
            for l in range(self.num_EAM_layers):
                emb_layer = self.emb_layers[l]
                emb_output = emb_layer(emb_input)
                emb_output = torch.tanh(emb_output)
                emb_input = emb_output
            emb = emb_output
            emb = F.normalize(emb, dim=0, p=2)
            # ==========
            scal_input = emb_cat
            scal_output = None
            for l in range(self.num_EAM_layers):
                scal_layer = self.scal_layers[l]
                scal_output = scal_layer(scal_input)
                scal_output = torch.sigmoid(scal_output)
                scal_input = scal_output
            scal = torch.mm(scal_output, scal_output.t())
            # ==========
            num_nodes = num_nodes_list[-1]
            emb_src = torch.reshape(emb, (1, num_nodes, self.EAM_dims[-1]))
            emb_dst = torch.reshape(emb, (num_nodes, 1, self.EAM_dims[-1]))
            adj_est = -torch.sum((emb_src-emb_dst)**2, dim=2)
            adj_est = torch.exp(torch.mul(adj_est, scal))

            return [adj_est]
        # ====================
        else: # pred_flag==False
            # ==========
            adj_est_list = [] # List of the prediction results (i.e., estimated adjacency matrices)
            for t in range(win_size):
                # ==========
                emb = EDM_output_list[t]
                feat = FEM_output_list[t+1]
                emb_cat = torch.cat((emb, feat), dim=1)
                # ==========
                emb_input = emb_cat
                emb_output = None
                for l in range(self.num_EAM_layers):
                    emb_layer = self.emb_layers[l]
                    emb_output = emb_layer(emb_input)
                    emb_output = torch.tanh(emb_output)
                    emb_input = emb_output
                emb = emb_output
                emb = F.normalize(emb, dim=0, p=2)
                # ==========
                scal_input = emb_cat
                scal_output = None
                for l in range(self.num_EAM_layers):
                    scal_layer = self.scal_layers[l]
                    scal_output = scal_layer(scal_input)
                    scal_output = torch.sigmoid(scal_output)
                    scal_input = scal_output
                scal = torch.mm(scal_output, scal_output.t())
                # ==========
                num_nodes = num_nodes_list[t+1]
                emb_src = torch.reshape(emb, (1, num_nodes, self.EAM_dims[-1]))
                emb_dst = torch.reshape(emb, (num_nodes, 1, self.EAM_dims[-1]))
                adj_est = -torch.sum((emb_src-emb_dst)**2, dim=2)
                adj_est = torch.exp(torch.mul(adj_est, scal))
                # ==========
                adj_est_list.append(adj_est)

            return adj_est_list

class DiscNet(nn.Module):
    '''
    Class to define the discriminative network
    Feature Extraction Module (FEM) - Multi-layer GNN - FC Output Layer
    '''
    def __init__(self, FEM_dims, GNN_dims, dropout_rate):
        super(DiscNet, self).__init__()
        # ====================
        self.FEM_dims = FEM_dims # Layer configuration of FEM
        self.GNN_dims = GNN_dims # Layer configuration of GNN
        self.dropout_rate = dropout_rate  # Dropout rate
        # ==========
        self.num_FEM_layers = len(self.FEM_dims)-1
        self.FEM_layers = nn.ModuleList()
        for l in range(self.num_FEM_layers):
            self.FEM_layers.append(nn.Linear(in_features=self.FEM_dims[l], out_features=self.FEM_dims[l+1]))
        # ==========
        self.num_GNN_layers = len(self.GNN_dims)-1 # Number of GNN layers
        self.GNN_layers = nn.ModuleList()
        for l in range(self.num_GNN_layers):
            self.GNN_layers.append(GraphNeuralNetworkDense(self.GNN_dims[l], self.GNN_dims[l+1],
                                                               dropout_rate=self.dropout_rate))
        # ==========
        self.output_layer = nn.Linear(in_features=self.GNN_dims[-1], out_features=1)
        self.dropout_layer = nn.Dropout(p=self.dropout_rate)

    def forward(self, real_sup_list, real_feat_list, fake_sup_list, fake_feat_list):
        '''
        Rewrite the forward function
        :param real_sup_list: list of GNN support of real snapshots
        :param real_feat_list: list of feature input of real snapshots
        :param fake_sup_list: list of GNN support of generated snapshots
        :param fake_feat_list: list of feature input of generated snapshots
        :return: output w.r.t. the real & fake input
        '''
        # ====================
        win_size = len(real_sup_list)
        real_output_list = []
        fake_output_list = []
        for i in range(win_size):
            # ==========
            # Feature Extraction Module (FEM)
            real_feat_input = real_feat_list[i]
            real_feat_output = None
            fake_feat_input = fake_feat_list[i]
            fake_feat_output = None
            for l in range(self.num_FEM_layers):
                FEM_layer = self.FEM_layers[l]
                real_feat_output = FEM_layer(real_feat_input)
                #real_feat_output = self.dropout_layer(real_feat_output)
                real_feat_input = real_feat_output
                fake_feat_output = FEM_layer(fake_feat_input)
                #fake_feat_output = self.dropout_layer(fake_feat_output)
                fake_feat_input = fake_feat_output
            # ==========
            # GNN
            real_sup = real_sup_list[i]
            real_input = real_feat_output
            fake_sup = fake_sup_list[i]
            fake_input = fake_feat_output
            for l in range(self.num_GNN_layers):
                # ==========
                GNN_layer = self.GNN_layers[l]
                # ==========
                real_output = GNN_layer(real_input, real_sup)
                real_input = real_output
                # ==========
                fake_output = GNN_layer(fake_input, fake_sup)
                fake_input = fake_output
            # ==========
            # Output layer
            real_output = self.output_layer(real_input)
            real_output = self.dropout_layer(real_output)
            real_output = torch.sigmoid(real_output)
            fake_output = self.output_layer(fake_input)
            fake_output = self.dropout_layer(fake_output)
            fake_output = torch.sigmoid(fake_output)
            real_output_list.append(real_output)
            fake_output_list.append(fake_output)

        return real_output_list, fake_output_list
