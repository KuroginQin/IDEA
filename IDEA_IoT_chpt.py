import torch
from utils import *
import scipy.sparse
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(100)

# ====================
data_name = 'IoT'
num_nodes_gbl = 668 # Number of accumulated nodes
num_snaps = 144 # Number of snapshots
max_thres = 1024 # Threshold for maximum edge weight
noise_dim = 48 # Dimensionality of noise input
feat_dim = 32 # Dimensionality of node feature
pos_dim = 32 # Dimensionality of position embedding

# ====================
lambd = 0.4 # Hyper-parameter of attentive aligning unit

# ====================
edge_seq = np.load('data/%s_edge_seq.npy' % (data_name), allow_pickle=True)
node_set_seq = np.load('data/%s_node_seq.npy' % (data_name), allow_pickle=True)
mod_seq = np.load('data/%s_mod_seq.npy' % (data_name), allow_pickle=True)
align_seq_gbl = np.load('data/%s_align_seq.npy' % (data_name), allow_pickle=True)
# =========
node_map_seq_gbl = []
num_nodes_seq_gbl = []
for t in range(num_snaps):
    node_set = node_set_seq[t]
    node_map = get_node_map(node_set)
    node_map_seq_gbl.append(node_map)
    num_nodes_seq_gbl.append(len(node_set))
# ==========
# Get global node features
feat_gbl = np.load('data/%s_feat.npy' % (data_name), allow_pickle=True)
feat_lcl_seq = []
for t in range(num_snaps):
    node_set = node_set_seq[t]
    node_idxs = sorted(list(node_set))
    feat_lcl = feat_gbl[node_idxs, :]
    feat_lcl_seq.append(feat_lcl)

# ====================
win_size = 10 # Window size of historical snapshots
epsilon = 1e-5 # Threshold of the zero-refining
num_test_snaps = 50 # Number of test snapshots
num_val_snaps = 10 # Number of validation snapshots
num_train_snaps = num_snaps-num_test_snaps-num_val_snaps # Number of training snapshots

# ====================
# Load check point
gen_net = torch.load('chpt/IDEA_%s.pkl' % (data_name)).to(device)

# ====================
# Evaluate the model on the test set
gen_net.eval()
# ==========
RMSE_list_L2 = []
MAE_list_L2 = []
MLSD_list_L2 = []
MR_list_L2 = []
# ==========
RMSE_list_L3 = []
MAE_list_L3 = []
MLSD_list_L3 = []
MR_list_L3 = []
for tau in range(num_snaps-num_test_snaps, num_snaps):
    # ====================
    sup_list = [] # List of GNN support (tensor)
    noise_list = [] # List of random noise inputs
    align_list = [] # List of align matrices
    feat_list = [] # List of feature input
    num_nodes_list = []
    pre_node_map_list = []
    for t in range(tau-win_size, tau):
        # ==========
        edges = edge_seq[t]
        num_nodes = num_nodes_seq_gbl[t]
        node_map = node_map_seq_gbl[t]
        adj = get_adj_wei_map(edges, node_map, num_nodes, max_thres)
        adj_norm = adj/max_thres # Normalize the edge weights to [0, 1]
        # ==========
        # Transfer the GNN support to a sparse tensor
        sup = get_gnn_sup(adj_norm)
        sup_sp = sp.sparse.coo_matrix(sup)
        sup_sp = sparse_to_tuple(sup_sp)
        idxs = torch.LongTensor(sup_sp[0].astype(float)).to(device)
        vals = torch.FloatTensor(sup_sp[1]).to(device)
        sup_tnr = torch.sparse.FloatTensor(idxs.t(), vals, sup_sp[2]).float().to(device)
        sup_list.append(sup_tnr)
        # =========
        # Generate the random noise via random projection
        mod_tnr = torch.FloatTensor(mod_seq[t]).to(device)
        rand_mat = rand_proj(num_nodes, noise_dim)
        rand_tnr = torch.FloatTensor(rand_mat).to(device)
        noise_tnr = torch.mm(mod_tnr, rand_tnr)
        noise_list.append(noise_tnr)
        # ==========
        align_tnr = torch.FloatTensor(align_seq_gbl[t]).to(device)
        align_list.append(align_tnr)
        feat_lcl = feat_lcl_seq[t]
        feat_tnr = torch.FloatTensor(feat_lcl).to(device)
        feat_list.append(feat_tnr)
        num_nodes_list.append(num_nodes)
        pre_node_map_list.append(node_map)
    # ==========
    # Get ground-truth
    edges = edge_seq[tau]
    # ==========
    # For L3
    num_nodes_L3 = num_nodes_seq_gbl[tau] # Number of nodes for L3
    node_map_L3 = node_map_seq_gbl[tau]
    gnd_L3 = get_adj_wei_map(edges, node_map_L3, num_nodes_L3, max_thres) # Ground-truth
    feat_lcl = feat_lcl_seq[tau]
    feat_tnr = torch.FloatTensor(feat_lcl).to(device)
    feat_list.append(feat_tnr)
    num_nodes_list.append(num_nodes_L3)
    # ==========
    # For L2
    node_idxs_L2 = get_node_idxs_L2(pre_node_map_list, node_map_L3)
    num_nodes_L2 = len(node_idxs_L2) # Number of nodes for L2
    gnd_L2 = gnd_L3[node_idxs_L2, :]
    gnd_L2 = gnd_L2[:, node_idxs_L2]

    # ==========
    # Get the prediction result
    adj_est_list = gen_net(sup_list, feat_list, noise_list, align_list, num_nodes_list, lambd, pred_flag=True)
    adj_est_L3 = adj_est_list[-1]
    adj_est_L2 = adj_est_L3[node_idxs_L2, :]
    adj_est_L2 = adj_est_L2[:, node_idxs_L2]
    if torch.cuda.is_available():
        adj_est_L2 = adj_est_L2.cpu().data.numpy()
        adj_est_L3 = adj_est_L3.cpu().data.numpy()
    else:
        adj_est_L2 = adj_est_L2.data.numpy()
        adj_est_L3 = adj_est_L3.data.numpy()
    # Rescale the edge weights to the original value range
    adj_est_L2 *= max_thres
    adj_est_L3 *= max_thres
    # ==========
    # Refine the prediction result
    for r in range(num_nodes_L3):
        if r<num_nodes_L2:
            adj_est_L2[r, r] = 0
        adj_est_L3[r, r] = 0
    for r in range(num_nodes_L3):
        for c in range(num_nodes_L3):
            if r<num_nodes_L2 and c<num_nodes_L2:
                if adj_est_L2[r, c]<=epsilon:
                    adj_est_L2[r, c] = 0
            if adj_est_L3[r, c]<=epsilon:
                adj_est_L3[r, c] = 0

    # ====================
    # Evaluate the prediction result
    RMSE_L2 = get_RMSE(adj_est_L2, gnd_L2, num_nodes_L2)
    MAE_L2 = get_MAE(adj_est_L2, gnd_L2, num_nodes_L2)
    MLSD_L2 = get_MLSD(adj_est_L2, gnd_L2, num_nodes_L2)
    MR_L2 = get_MR(adj_est_L2, gnd_L2, num_nodes_L2)
    # ==========
    RMSE_list_L2.append(RMSE_L2)
    MAE_list_L2.append(MAE_L2)
    MLSD_list_L2.append(MLSD_L2)
    MR_list_L2.append(MR_L2)
    # ==========
    RMSE_L3 = get_RMSE(adj_est_L3, gnd_L3, num_nodes_L3)
    MAE_L3 = get_MAE(adj_est_L3, gnd_L3, num_nodes_L3)
    MLSD_L3 = get_MLSD(adj_est_L3, gnd_L3, num_nodes_L3)
    MR_L3 = get_MR(adj_est_L3, gnd_L3, num_nodes_L3)
    # ==========
    RMSE_list_L3.append(RMSE_L3)
    MAE_list_L3.append(MAE_L3)
    MLSD_list_L3.append(MLSD_L3)
    MR_list_L3.append(MR_L3)

# ====================
RMSE_mean_L2 = np.mean(RMSE_list_L2)
RMSE_std_L2 = np.std(RMSE_list_L2, ddof=1)
MAE_mean_L2 = np.mean(MAE_list_L2)
MAE_std_L2 = np.std(MAE_list_L2, ddof=1)
MLSD_mean_L2 = np.mean(MLSD_list_L2)
MLSD_std_L2 = np.std(MLSD_list_L2, ddof=1)
MR_mean_L2 = np.mean(MR_list_L2)
MR_std_L2 = np.std(MR_list_L2, ddof=1)
print('(L2) Test RMSE %f %f MAE %f %f MLSD %f %f MR %f %f'
    % (RMSE_mean_L2, RMSE_std_L2, MAE_mean_L2, MAE_std_L2,
       MLSD_mean_L2, MLSD_std_L2, MR_mean_L2, MR_std_L2))
# ==========
RMSE_mean_L3 = np.mean(RMSE_list_L3)
RMSE_std_L3 = np.std(RMSE_list_L3, ddof=1)
MAE_mean_L3 = np.mean(MAE_list_L3)
MAE_std_L3 = np.std(MAE_list_L3, ddof=1)
MLSD_mean_L3 = np.mean(MLSD_list_L3)
MLSD_std_L3 = np.std(MLSD_list_L3, ddof=1)
MR_mean_L3 = np.mean(MR_list_L3)
MR_std_L3 = np.std(MR_list_L3, ddof=1)
print('(L3) Test RMSE %f %f MAE %f %f MLSD %f %f MR %f %f\n'
    % (RMSE_mean_L3, RMSE_std_L3, MAE_mean_L3, MAE_std_L3,
       MLSD_mean_L3, MLSD_std_L3, MR_mean_L3, MR_std_L3))
