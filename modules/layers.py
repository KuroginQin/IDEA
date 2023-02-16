import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as Init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class GraphNeuralNetwork(Module):
    '''
    Class to define the GNN layer (w/ sparse matrix multiplication)
    '''

    def __init__(self, input_dim, output_dim, dropout_rate):
        super(GraphNeuralNetwork, self).__init__()
        # ====================
        self.input_dim = input_dim # Dimensionality of input features
        self.output_dim = output_dim # Dimensionality of output features
        self.dropout_rate = dropout_rate # Dropout rate
        # ====================
        # Initialize model parameters
        self.agg_wei = Init.xavier_uniform_(Parameter(torch.FloatTensor(input_dim, output_dim))) # Aggregation weight matrix
        # ==========
        self.param = nn.ParameterList()
        self.param.append(self.agg_wei)
        # ==========
        self.dropout_layer = nn.Dropout(p=self.dropout_rate)

    def forward(self, feat, sup):
        '''
        Rewrite the forward function
        :param feat: feature input of the GCN layer
        :param sup: GCN support (normalized adjacency matrix)
        :return: aggregated feature output of the GCN layer
        '''
        # ====================
        # Feature aggregation from immediate neighbors
        feat_agg = torch.spmm(sup, feat) # Aggregated feature
        agg_output = torch.relu(torch.mm(feat_agg, self.param[0]))
        agg_output = F.normalize(agg_output, dim=1, p=2) # l2-normalization
        agg_output = self.dropout_layer(agg_output)

        return agg_output

class GraphNeuralNetworkDense(Module):
    '''
    Class to define the GNN layer (w/ dense matrix multiplication)
    '''

    def __init__(self, input_dim, output_dim, dropout_rate):
        super(GraphNeuralNetworkDense, self).__init__()
        # ====================
        self.input_dim = input_dim # Dimensionality of input features
        self.output_dim = output_dim # Dimensionality of output features
        self.dropout_rate = dropout_rate # Dropout rate
        # ====================
        # Initialize model parameters
        self.agg_wei = Init.xavier_uniform_(Parameter(torch.FloatTensor(input_dim, output_dim))) # Aggregation weight matrix
        # ==========
        self.param = nn.ParameterList()
        self.param.append(self.agg_wei)
        # =========
        self.dropout_layer = nn.Dropout(p=self.dropout_rate)

    def forward(self, feat, sup):
        '''
        Rewrite the forward function
        :param feat: feature input of the GCN layer
        :param sup: GCN support (normalized adjacency matrix)
        :return: aggregated feature output of the GCN layer
        '''
        # ====================
        # Feature aggregation from immediate neighbors
        feat_agg = torch.mm(sup, feat) # Aggregated feature
        agg_output = torch.relu(torch.mm(feat_agg, self.param[0]))
        agg_output = F.normalize(agg_output, dim=1, p=2) # l2-normalization
        agg_output = self.dropout_layer(agg_output)

        return agg_output

class AttNodeAlign(Module):
    '''
    Class to define attentive node aligning unit
    '''

    def __init__(self, feat_dim, hid_dim, dropout_rate):
        super(AttNodeAlign, self).__init__()
        # ====================
        self.dropout_rate = dropout_rate
        # ====================
        self.feat_dim = feat_dim # Dimensionality of feature input
        self.hid_dim = hid_dim # Dimensionality of the hidden space
        # ====================
        self.from_map = nn.Linear(in_features=self.feat_dim, out_features=self.feat_dim) # FC feature mapping
        self.to_map = nn.Linear(in_features=self.feat_dim, out_features=self.feat_dim) # FC feature mapping
        # =========
        self.dropout_layer = nn.Dropout(p=self.dropout_rate)

    def forward(self, align, feat_from, feat_to, emb, lambd):
        '''
        Rewrite the forward function
        :param align: align matrix
        :param feat_from: (reduced) GNN feature of timeslice t
        :param feat_to: (reduced) GNN feature of timeslice (t+1)
        :param emb: hidden embedding
        :param lambd: factor of the attention module
        :return: aligned features
        '''
        # ====================
        feat_from_ = torch.tanh(self.from_map(feat_from))
        feat_to_ = torch.tanh(self.to_map(feat_to))
        att_align = torch.mm(feat_from_, feat_to_.t())
        hyd_align = align + lambd*att_align
        feat_align = torch.mm(hyd_align.t(), emb)

        return feat_align

class BiGNNAlign(Module):
    def __init__(self, feat_dim, hid_dim, dropout_rate):
        super(BiGNNAlign, self).__init__()
        # ====================
        self.dropout_rate = dropout_rate
        # ====================
        self.feat_dim = feat_dim # Dimensionality of feature input
        self.hid_dim = hid_dim # Dimensionality of the hidden space
        # ====================
        self.from_map = nn.Linear(in_features=self.feat_dim, out_features=self.feat_dim) # FC feature mapping
        self.to_map = nn.Linear(in_features=self.feat_dim, out_features=self.feat_dim) # FC feature mapping
        # =========
        self.dropout_layer = nn.Dropout(p=self.dropout_rate)

    def forward(self, align, feat_from, feat_to, emb, fact):
        # ====================
        feat_from_ = torch.tanh(self.from_map(feat_from)) # relu, sigmoid
        feat_to_ = torch.tanh(self.to_map(feat_to)) # relu, sigmoid
        att_align = torch.mm(feat_from_, feat_to_.t())
        hyd_align = align + fact*att_align
        feat_align = torch.mm(hyd_align.t(), emb)

        return feat_align

class IGRU(Module):
    '''
    Class to define inductive GRU
    '''

    def __init__(self, input_dim, output_dim, dropout_rate):
        super(IGRU, self).__init__()
        # ====================
        self.input_dim = input_dim # Dimensionality of input features
        self.output_dim = output_dim # Dimension of output features
        self.dropout_rate = dropout_rate # Dropout rate
        # ====================
        # Initialize model parameters
        self.reset_wei = Init.xavier_uniform_(Parameter(torch.FloatTensor(2*self.input_dim, self.output_dim)))
        self.reset_bias = Parameter(torch.zeros(self.output_dim))
        self.act_wei = Init.xavier_uniform_(Parameter(torch.FloatTensor(2*self.input_dim, self.output_dim)))
        self.act_bias = Parameter(torch.zeros(self.output_dim))
        self.update_wei = Init.xavier_uniform_(Parameter(torch.FloatTensor(2*self.input_dim, self.output_dim)))
        self.update_bias = Parameter(torch.zeros(self.output_dim))
        # ==========
        self.param = nn.ParameterList()
        self.param.append(self.reset_wei)
        self.param.append(self.reset_bias)
        self.param.append(self.act_wei)
        self.param.append(self.act_bias)
        self.param.append(self.update_wei)
        self.param.append(self.update_bias)
        # ==========
        self.dropout_layer = nn.Dropout(p=self.dropout_rate)

    def forward(self, pre_state, cur_state):
        '''
        Rewrite the forward function
        :param pre_state: previous state
        :param cur_state: current state
        :return: next state
        '''
        # ====================
        # Reset gate
        reset_input = torch.cat((cur_state, pre_state), dim=1)
        reset_output = torch.sigmoid(torch.mm(reset_input, self.param[0]) + self.param[1])
        # ==========
        # Input activation gate
        act_input = torch.cat((cur_state, torch.mul(reset_output, pre_state)), dim=1)
        act_output = torch.tanh(torch.mm(act_input, self.param[2]) + self.param[3])
        # ==========
        # Update gate
        update_input = torch.cat((cur_state, pre_state), dim=1)
        update_output = torch.sigmoid(torch.mm(update_input, self.param[4]) + self.param[5])
        # ==========
        # Next state
        next_state = torch.mul((1-update_output), pre_state) + torch.mul(update_output, act_output)
        next_state = self.dropout_layer(next_state)

        return next_state
