import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from .layers import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GenNet(nn.Module):
    '''
    Class to define the generator
    '''
    def __init__(self, FRE_dims, NLU_dims, OD_dims, dropout_rate):
        super(GenNet, self).__init__()
        # ====================
        self.FRE_dims = FRE_dims # Layer configuration of FRE
        self.NLU_dims = NLU_dims # Layer configuration of NLU
        self.OD_dims = OD_dims # Layer configuration of OD
        self.dropout_rate = dropout_rate # Dropout rate
        # ==========
        self.dropout_layer = nn.Dropout(p=self.dropout_rate)
        # ==========
        # Define the FRE
        self.num_FRE_layers = len(self.FRE_dims)-1 # Number of FRE layers
        self.FRE_layer_list = nn.ModuleList()
        for l in range(self.num_FRE_layers):
            self.FRE_layer_list.append(nn.Linear(in_features=self.FRE_dims[l], out_features=self.FRE_dims[l+1]))
        # ===========
        # Define the NLU
        self.num_NLU_layers = len(self.NLU_dims)-1 # Number of NLU layers
        self.NLU_layer_list = nn.ModuleList()
        for l in range(self.num_NLU_layers):
            # GNN
            self.NLU_layer_list.append(GraphNeuralNetwork(input_dim=self.NLU_dims[l], output_dim=self.NLU_dims[l+1],
                                                          dropout_rate=self.dropout_rate))
            # RNN
            self.NLU_layer_list.append(IGRU(input_dim=self.NLU_dims[l+1], output_dim=self.NLU_dims[l+1],
                                            dropout_rate=self.dropout_rate))
            # Align Unit
            self.NLU_layer_list.append(BiGNNAlign(feat_dim=self.FRE_dims[-1], hid_dim=self.NLU_dims[l+1],
                                                  dropout_rate=self.dropout_rate))
        # ==========
        # Define the OD
        self.num_OD_layers = len(self.OD_dims)-1 # Number of OD layers
        # Embedding mapping network
        self.emb_layer_list = nn.ModuleList()
        for l in range(self.num_OD_layers):
            self.emb_layer_list.append(nn.Linear(in_features=self.OD_dims[l], out_features=self.OD_dims[l+1]))
        # Scaling network
        self.scal_layer_list = nn.ModuleList()
        for l in range(self.num_OD_layers):
            self.scal_layer_list.append(nn.Linear(in_features=self.OD_dims[l], out_features=self.OD_dims[l+1]))

    def forward(self, sup_list, feat_list, noise_list, align_list, num_nodes_list, lambd, pred_flag=True):
        '''
        Rewrite the forward function
        :param sup_list: list of GNN supports (normalized adjacency matrices) w.r.t. each input snapshot (l)
        :param feat_list: list of node attributes w.r.t. each snapshot (input & output) (l+1)
        :param noise_list: list of noise input (l)
        :param align_list: list of align matrices (l)
        :param num_nodes_list: list of #nodes w.r.t. each snapshot (input & output) (l+1)
        :return: list of prediction results (estimated adjacency matrices)
        '''
        # ====================
        win_size = len(sup_list) # l
        # ==========
        # FRE
        FRE_input_list = feat_list
        FRE_output_list = None
        for l in range(self.num_FRE_layers):
            FRE_layer = self.FRE_layer_list[l]
            FRE_output_list = []
            for i in range(win_size+1):
                FRE_input = FRE_input_list[i]
                FRE_output = FRE_layer(FRE_input)
                #FRE_output = self.dropout_layer(FRE_output)
                FRE_output = torch.relu(FRE_output)
                FRE_output_list.append(FRE_output)
            FRE_input_list = FRE_output_list
        # ====================
        # NLU
        # Concatenate GNN outputs with noise inputs
        NLU_input_list = []
        align_output_list = None
        for i in range(win_size):
            NLU_input = FRE_output_list[i]
            noise = noise_list[i]
            NLU_input = torch.cat((NLU_input, noise), dim=1)
            NLU_input_list.append(NLU_input)
        for l in range(0, self.num_NLU_layers*3, 3):
            # ==========
            GNN_layer = self.NLU_layer_list[l]
            GNN_output_list = []
            for i in range(win_size):
                sup = sup_list[i] # GNN support w.r.t. current snapshot
                feat = NLU_input_list[i] # Feature input of current snapshot
                GNN_output = GNN_layer(feat, sup)
                GNN_output_list.append(GNN_output)
            # ==========
            RNN_layer = self.NLU_layer_list[l+1]
            align_unit = self.NLU_layer_list[l+2]
            hid_dim = self.NLU_dims[int(l/3)+1] # Dimension of the embedding in current layer
            pre_state = torch.zeros(num_nodes_list[0], hid_dim).to(device)
            RNN_output_list = []
            align_output_list = []
            for i in range(win_size):
                RNN_input = GNN_output_list[i]
                RNN_output = RNN_layer(pre_state, RNN_input)
                pre_state = align_unit(align_list[i], FRE_output_list[i], FRE_output_list[i+1], RNN_output, fact=lambd)
                align_output_list.append(pre_state)
                RNN_output_list.append(RNN_output)
            RNN_output_list.append(pre_state)
            # ==========
            NLU_input_list = RNN_output_list
        NLU_output_list = align_output_list

        # ====================
        # OD
        if pred_flag==True:
            emb = NLU_output_list[-1]
            feat = FRE_output_list[-1]
            emb_cat = torch.cat((emb, feat), dim=1)
            # ==========
            emb_input = emb_cat
            emb_output = None
            for l in range(self.num_OD_layers):
                emb_layer = self.emb_layer_list[l]
                emb_output = emb_layer(emb_input)
                #emb_output = self.dropout_layer(emb_output)
                emb_output = torch.tanh(emb_output)
                emb_input = emb_output
            emb = emb_output
            emb = F.normalize(emb, dim=0, p=2)
            # ==========
            scal_input = emb_cat
            scal_output = None
            for l in range(self.num_OD_layers):
                scal_layer = self.scal_layer_list[l]
                scal_output = scal_layer(scal_input)
                # scal_output = self.dropout_layer(scal_output)
                scal_output = torch.sigmoid(scal_output)
                # scal_output = torch.relu(scal_output)
                scal_input = scal_output
            scal = torch.mm(scal_output, scal_output.t())
            # ==========
            num_nodes = num_nodes_list[-1]
            emb_src = torch.reshape(emb, (1, num_nodes, self.OD_dims[-1]))
            emb_dst = torch.reshape(emb, (num_nodes, 1, self.OD_dims[-1]))
            adj_est = -torch.sum((emb_src - emb_dst)**2, dim=2)
            #adj_est = torch.exp(torch.mul(adj_est, scal))
            adj_est = 1+torch.tanh(torch.mul(adj_est, scal))

            return [adj_est]
        # ====================
        else: # pred_flag==False
            # ==========
            adj_est_list = [] # List of the prediction results (i.e., estimated adjacency matrices)
            for i in range(win_size):
                # ==========
                emb = NLU_output_list[i]
                # emb = F.normalize(emb, dim=0, p=2)
                feat = FRE_output_list[i+1]
                # pred_feat = FRE_output_list[i+1]
                emb_cat = torch.cat((emb, feat), dim=1)
                # ==========
                emb_input = emb_cat
                emb_output = None
                for l in range(self.num_OD_layers):
                    emb_layer = self.emb_layer_list[l]
                    emb_output = emb_layer(emb_input)
                    # emb_output = self.dropout_layer(emb_output)
                    emb_output = torch.tanh(emb_output)
                    emb_input = emb_output
                emb = emb_output
                emb = F.normalize(emb, dim=0, p=2)
                # ==========
                scal_input = emb_cat
                scal_output = None
                for l in range(self.num_OD_layers):
                    scal_layer = self.scal_layer_list[l]
                    scal_output = scal_layer(scal_input)
                    # scal_output = self.dropout_layer(scal_output)
                    scal_output = torch.sigmoid(scal_output)
                    # scal_output = torch.relu(scal_output)
                    scal_input = scal_output
                scal = torch.mm(scal_output, scal_output.t())
                # ==========
                num_nodes = num_nodes_list[i + 1]
                emb_src = torch.reshape(emb, (1, num_nodes, self.OD_dims[-1]))
                emb_dst = torch.reshape(emb, (num_nodes, 1, self.OD_dims[-1]))
                adj_est = -torch.sum((emb_src - emb_dst)**2, dim=2)
                #adj_est = torch.exp(torch.mul(adj_est, scal))
                adj_est = 1+torch.tanh(torch.mul(adj_est, scal))
                # ==========
                adj_est_list.append(adj_est)

            return adj_est_list
