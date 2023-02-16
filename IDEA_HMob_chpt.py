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
setup_seed(50)

# ====================
data_name = 'HMob'
num_nodes = 92 # Number of nodes
num_snaps = 500 # Number of snapshots
max_thres = 250 # Threshold for maximum edge weight
noise_dim = 64 # Dimensionality of noise input
pos_dim = 32 # Dimensionality of position embedding

# ====================
edge_seq = np.load('data/%s_edge_seq.npy' % (data_name), allow_pickle=True)
mod_seq = np.load('data/%s_mod_seq.npy' % (data_name), allow_pickle=True)
# ==========
# Get the positional embedding
pos_emb = None
for p in range(num_nodes):
    if p==0:
        pos_emb = get_pos_emb(p, pos_dim)
    else:
        pos_emb = np.concatenate((pos_emb, get_pos_emb(p, pos_dim)), axis=0)
feat_tnr = torch.FloatTensor(pos_emb).to(device)

# ====================
win_size = 5 # Window size of historical snapshots
lambd = 0.6 # Hyper-parameter of attentive aligning unit
epsilon = 0.01 # Threshold of the zero-refining
num_test_snaps = 50 # Number of test snapshots
num_val_snaps = 10 # Number of validation snapshots
num_train_snaps = num_snaps-num_test_snaps-num_val_snaps # Number of training snapshots

# ====================
# Get align matrices
align_mat = torch.eye(num_nodes).to(device)
align_list = []
for i in range(win_size):
    align_list.append(align_mat)
# ==========
feat_list = []
for i in range(win_size+1):
    feat_list.append(feat_tnr)
# ==========
num_nodes_list = []
for i in range(win_size+1):
    num_nodes_list.append(num_nodes)

# ====================
# Load check point
gen_net = torch.load('chpt/IDEA_%s.pkl' % (data_name)).to(device)

# ====================
# Test the model
gen_net.eval()
# ==========
RMSE_list = []
MAE_list = []
MLSD_list = []
MR_list = []
for tau in range(num_snaps-num_test_snaps, num_snaps):
    # ====================
    sup_list = [] # List of GNN support (tensor)
    noise_list = [] # List of random noise inputs
    for t in range(tau-win_size, tau):
        # ==========
        edges = edge_seq[t]
        adj = get_adj_wei(edges, num_nodes, max_thres)
        adj_norm = adj/max_thres  # Normalize the edge weights to [0, 1]
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
    # Get prediction result
    adj_est_list = gen_net(sup_list, feat_list, noise_list, align_list, num_nodes_list, lambd, pred_flag=True)
    adj_est = adj_est_list[-1]
    if torch.cuda.is_available():
        adj_est = adj_est.cpu().data.numpy()
    else:
        adj_est = adj_est.data.numpy()
    adj_est *= max_thres # Rescale the edge weights to the original value range
    # ==========
    # Refine the prediction result
    for r in range(num_nodes):
        adj_est[r, r] = 0
    for r in range(num_nodes):
        for c in range(num_nodes):
            if adj_est[r, c] <= epsilon:
                adj_est[r, c] = 0

    # ====================
    # Get ground-truth
    edges = edge_seq[tau]
    gnd = get_adj_wei(edges, num_nodes, max_thres)

    # ====================
    # Evaluate the prediction result
    RMSE = get_RMSE(adj_est, gnd, num_nodes)
    MAE = get_MAE(adj_est, gnd, num_nodes)
    MLSD = get_MLSD(adj_est, gnd, num_nodes)
    MR = get_MR(adj_est, gnd, num_nodes)
    # ==========
    RMSE_list.append(RMSE)
    MAE_list.append(MAE)
    MLSD_list.append(MLSD)
    MR_list.append(MR)

# ====================
RMSE_mean = np.mean(RMSE_list)
RMSE_std = np.std(RMSE_list, ddof=1)
MAE_mean = np.mean(MAE_list)
MAE_std = np.std(MAE_list, ddof=1)
MLSD_mean = np.mean(MLSD_list)
MLSD_std = np.std(MLSD_list, ddof=1)
MR_mean = np.mean(MR_list)
MR_std = np.std(MR_list, ddof=1)
print('Test RMSE %f %f MAE %f %f MLSD %f %f MR %f %f\n'
    % (RMSE_mean, RMSE_std, MAE_mean, MAE_std, MLSD_mean, MLSD_std, MR_mean, MR_std))
