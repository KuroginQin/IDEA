import numpy as np
import scipy as sp
import torch

# ========================================
# Evaluation Metrics
def get_RMSE(adj_est, gnd, num_nodes):
    '''
    Function to get the RMSE (root mean square error) metric
    :param adj_est: prediction result
    :param gnd: ground-truth
    :param num_nodes: number of nodes
    :return: RMSE metric
    '''
    # =====================
    f_norm = np.linalg.norm(gnd-adj_est, ord='fro')**2
    #f_norm = np.sum((gnd - adj_est)**2)
    RMSE = np.sqrt(f_norm/(num_nodes*num_nodes))

    return RMSE

def get_MAE(adj_est, gnd, num_nodes):
    '''
    Funciton to get the MAE (mean absolute error) metric
    :param adj_est: prediction result
    :param gnd: ground-truth
    :param num_nodes: number of nodes
    :return: MAE metric
    '''
    # ====================
    MAE = np.sum(np.abs(gnd-adj_est))/(num_nodes*num_nodes)

    return MAE

def get_MLSD(adj_est, gnd, num_nodes):
    '''
    Function to get MLSD (mean logarithmic scale difference) metric
    :param adj_est: prediction result
    :param gnd: ground-truth
    :param num_nodes: number of nodes
    :return: MLSD metric
    '''
    # ====================
    epsilon = 1e-5
    adj_est_ = np.maximum(adj_est, epsilon)
    gnd_ = np.maximum(gnd, epsilon)
    MLSD = np.sum(np.abs(np.log10(adj_est_/gnd_)))
    MLSD /= (num_nodes*num_nodes)

    return MLSD

def get_MR(adj_est, gnd, num_nodes):
    '''
    Function to get MR (mismatch rate) metric
    :param adj_est: prediction result (i.e., the estimated adjacency matrix)
    :param gnd: ground-truth
    :param num_nodes: number of nodes
    :return: MR metric
    '''
    # ====================
    mis_sum = 0
    for r in range(num_nodes):
        for c in range(num_nodes):
            if (adj_est[r, c]>0 and gnd[r, c]==0) or (adj_est[r, c]==0 and gnd[r, c]>0):
                mis_sum += 1
    # ==========
    MR = mis_sum/(num_nodes*num_nodes)

    return MR

# ========================================
# Data processing
def get_adj_wei(edges, num_nodes, max_wei):
    '''
    Function to get (dense) weighted adjacency matrix according to edge list
    :param edges: edge list
    :param num_nodes: number of nodes
    :param max_wei: maximum edge weight
    :return: adj: adjacency matrix
    '''
    adj = np.zeros((num_nodes, num_nodes))
    num_edges = len(edges)
    for i in range(num_edges):
        src = int(edges[i][0])
        dst = int(edges[i][1])
        wei = float(edges[i][2])
        if wei>max_wei:
            wei = max_wei
        adj[src, dst] = wei
        adj[dst, src] = wei
    for i in range(num_nodes):
        adj[i, i] = 0

    return adj

def get_adj_wei_map(edges, node_map, num_nodes, max_thres):
    '''
    Function to get the (weighted) adjacency matrix according to the edge list
    :param edges: edge list
    :param node_num: number of nodes
    :param max_thres: threshold of the maximum edge weight
    :return: adj: adjacency matrix
    '''
    adj = np.zeros((num_nodes, num_nodes))
    num_edges = len(edges)
    for i in range(num_edges):
        src = int(edges[i][0])
        dst = int(edges[i][1])
        if (src not in node_map) or (dst not in node_map):
            continue
        src_idx = node_map[src]
        dst_idx = node_map[dst]
        wei = float(edges[i][2])
        if wei>max_thres:
            wei = max_thres
        adj[src_idx, dst_idx] = wei
        adj[dst_idx, src_idx] = wei
    for i in range(num_nodes):
        adj[i, i] = 0

    return adj

def get_node_map(node_set):
    node_idxs = sorted(list(node_set))
    node_map = {}
    node_cnt = 0
    for node_idx in node_idxs:
        node_map[node_idx] = node_cnt
        node_cnt += 1

    return node_map

def get_node_idxs_L2(pre_node_map_list, cur_node_map_L3):
    win_size = len(pre_node_map_list)
    pre_node_set_gbl = set()
    for t in range(win_size):
        pre_node_map = pre_node_map_list[t]
        for node in pre_node_map:
            if node not in pre_node_set_gbl:
                pre_node_set_gbl.add(node)
    # ==========
    node_idx_set_L2 = set()
    for node in pre_node_set_gbl:
        if node in cur_node_map_L3:
            node_idx = cur_node_map_L3[node]
            if node_idx not in node_idx_set_L2:
                node_idx_set_L2.add(node_idx)
    node_idx_list_L2 = sorted(list(node_idx_set_L2))

    return node_idx_list_L2

def gen_noise(m, n):
    '''
    Function to generate noise (feature) input
    :param m: number of rows
    :param n: number of columns
    :return: noise matrix
    '''
    # ====================
    return np.random.uniform(0, 1., size=[m, n])

def get_gnn_sup(adj):
    '''
    Function to get GNN support (normalized adjacency matrix with self-connected edges)
    :param adj: original adjacency matrix
    :return: GNN support
    '''
    # ====================
    num_nodes, _ = adj.shape
    adj = adj + np.eye(num_nodes)
    degs = np.sqrt(np.sum(adj, axis=1))
    sup = adj # GNN support
    for i in range(num_nodes):
        sup[i, :] /= degs[i]
    for j in range(num_nodes):
        sup[:, j] /= degs[j]

    return sup

def get_gnn_sup_woSE(adj):
    '''
    Function to get GNN support (normalized adjacency matrix w/o self-connected edges)
    :param adj: original adjacency matrix
    :return: GNN support
    '''
    # ====================
    num_nodes, _ = adj.shape
    degs = np.sqrt(np.sum(adj, axis=1))
    sup = adj # GNN support
    for i in range(num_nodes):
        sup[i, :] /= degs[i]
    for j in range(num_nodes):
        sup[:, j] /= degs[j]

    return sup

def sparse_to_tuple(sparse_mx):
    '''
    Function to transfer sparse matrix to tuple format
    :param sparse_mx: original sparse matrix
    :return: corresponding tuple format
    '''
    def to_tuple(mx):
        if not sp.sparse.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape
    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)
    return sparse_mx

def get_mod(adj):
    '''
    Function to get modularity matrix w.r.t. an adjacency matrix
    :param adj: adjacency matrix
    :return: corresponding modularity matrix
    '''
    wei_sum = np.sum(adj)
    degs = np.mat(np.sum(adj, axis=1))
    prop = np.matmul(degs.transpose(), degs)/wei_sum
    mod = adj - prop

    return mod

def get_mod_GPU(adj_tnr, node_num):
    '''
    Funtion to get modularity matrix w.r.t. an adjacency matrix (speeded up by GPU)
    :param adj_tnr: tensor of adjacency matrix
    :param node_num: number of nodes
    :return: modularity matrix
    '''
    degs = torch.sum(adj_tnr, dim=1)
    wei_sum = torch.sum(degs)
    degs = torch.reshape(degs, (1, node_num))
    prop = torch.mm(degs.t(), degs)/wei_sum

    return adj_tnr - prop

def rand_proj(num_nodes, hid_dim):
    '''
    Function to get random projection matrix
    num_nodes: number of nodes
    hid_dim: dimensionality of latent space
    :return: random projection matrix
    '''
    rand_mat = np.random.normal(0, 1.0/np.sqrt(hid_dim), (num_nodes, hid_dim))
    temp_l = np.linalg.norm(rand_mat, axis=1)
    for i in range(hid_dim):
        temp_row = rand_mat[:, i]
        for j in range(i-1):
            temp_j = rand_mat[:, j]
            temp_product = temp_row.T.dot(temp_j)/(temp_l[j]**2)
            temp_row -= temp_product*temp_j
        temp_row *= temp_l[i]/np.sqrt(temp_row.T.dot(temp_row))
        rand_mat[:, i] = temp_row

    return rand_mat

def get_pos_emb(pos, hid_dim):
    '''
    Funciton to get positional embedding
    :param pos: position index
    :param hid_dim: dimensionality of positional embedding
    :return: positional embedding
    '''
    pos_emb = np.zeros((1, hid_dim))
    for i in range(hid_dim):
        if i%2==0:
            pos_emb[0, i] = np.sin(pos/(10000**(i/hid_dim)))
        else:
            pos_emb[0, i] = np.cos(pos/(10000**((i-1)/hid_dim)))
    return pos_emb
