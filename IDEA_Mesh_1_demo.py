import torch
import torch.optim as optim
from modules.IDEA import GenNet_tanh
from modules.IDEA import DiscNet
from modules.loss import get_pre_gen_loss
from modules.loss import get_gen_loss
from modules.loss import get_disc_loss
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
setup_seed(0)

# ====================
data_name = 'Mesh-1'
num_nodes = 38 # Number of nodes
num_snaps = 445 # Number of snapshots
max_thres = 2000 # Threshold for maximum edge weight
noise_dim = 32 # Dimensionality of noise input
feat_dim = 32 # Dimensionality of node feature
pos_dim = 16 # Dimensionality of positional embedding
GNN_feat_dim = feat_dim+pos_dim
FEM_dims = [GNN_feat_dim, 32, 16] # Layer configuration of feature extraction module (FEM)
EDM_dims = [(FEM_dims[-1]+noise_dim), 32, 32] # Layer configuration of embedding derivation module (EDM)
EAM_dims = [(EDM_dims[-1]+FEM_dims[-1]), 32, 16] # Layer configuration of embedding aggregation module (EAM)
disc_dims = [FEM_dims[-1], 32, 16, 8] # Layer configuration of discriminator
save_flag = False # Flag whether to save the trained model (w.r.t. each epoch)

# ====================
alpha = 10 # Parameter to balance the ER loss
beta = 0.1 # Parameter to balance the SDM loss
lambd = 0.1 # Parameter of attentive aligning unit
theta = 0.1 # Decaying factor

# ====================
edge_seq = np.load('data/%s_edge_seq.npy' % (data_name), allow_pickle=True)
mod_seq = np.load('data/%s_mod_seq.npy' % (data_name), allow_pickle=True)
# ==========
# Get the node features
feat = np.load('data/%s_feat.npy' % (data_name), allow_pickle=True)
feat_tnr = torch.FloatTensor(feat).to(device)
# Get positional embedding
pos_emb = None
for p in range(num_nodes):
    if p==0:
        pos_emb = get_pos_emb(p, pos_dim)
    else:
        pos_emb = np.concatenate((pos_emb, get_pos_emb(p, pos_dim)), axis=0)
pos_tnr = torch.FloatTensor(pos_emb).to(device)
feat_tnr = torch.cat((feat_tnr, pos_tnr), dim=1)

# ====================
dropout_rate = 0.0 # Dropout rate
win_size = 10 # Window size of historical snapshots
epsilon = 0.01 # Threshold of the zero-refining
num_pre_epochs = 30 # Number of pre-training epochs
num_epochs = 200 # Number of training epochs
num_test_snaps = 50 # Number of test snapshots
num_val_snaps = 10 # Number of validation snapshots
num_train_snaps = num_snaps-num_test_snaps-num_val_snaps # Number of training snapshots

# ====================
# Get align matrices (i.e., identity matrices for level-1)
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
# Define the model
gen_net = GenNet_tanh(FEM_dims, EDM_dims, EAM_dims, dropout_rate).to(device) # Generator
disc_net = DiscNet(FEM_dims, disc_dims, dropout_rate).to(device) # Discriminator
# ==========
# Define the optimizer
pre_gen_opt = optim.Adam(gen_net.parameters(), lr=5e-4, weight_decay=1e-5) # Optimizer for the pre-training of generator
gen_opt = optim.Adam(gen_net.parameters(), lr=5e-4, weight_decay=1e-5) # Optimizer for the formal optimization of generator
disc_opt = optim.Adam(disc_net.parameters(), lr=5e-4, weight_decay=1e-5) # Optimizer for the formal optimization of discrminator

# ====================
# Pre-training of generator
for epoch in range(num_pre_epochs):
    # ====================
    # Pre-train the model
    gen_net.train()
    disc_net.train()
    # ==========
    train_cnt = 0
    gen_loss_list = []
    # ==========
    for tau in range(win_size, num_train_snaps):
        # ====================
        sup_list = [] # List of GNN support (tensor)
        noise_list = [] # List of random noise inputs
        for t in range(tau-win_size, tau):
            # ==========
            edges = edge_seq[t]
            adj = get_adj_wei(edges, num_nodes, max_thres)
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
        gnd_list = []
        real_sup_list = []
        for t in range(tau-win_size+1, tau+1):
            # ==========
            edges = edge_seq[t]
            gnd = get_adj_wei(edges, num_nodes, max_thres)
            gnd_norm = gnd/max_thres  # Normalize the edge weights to [0, 1]
            gnd_norm += np.eye(num_nodes)
            gnd_tnr = torch.FloatTensor(gnd_norm).to(device)
            gnd_list.append(gnd_tnr)
            # ==========
            # Transfer the GNN support to a sparse tensor
            sup = get_gnn_sup_woSE(gnd_norm)
            sup_tnr = torch.FloatTensor(sup).to(device)
            real_sup_list.append(sup_tnr)

        # ====================
        # Train the generator
        adj_est_list = gen_net(sup_list, feat_list, noise_list, align_list, num_nodes_list, lambd, pred_flag=False)
        pre_gen_loss = get_pre_gen_loss(adj_est_list, gnd_list, theta)
        pre_gen_opt.zero_grad()
        pre_gen_loss.backward()
        pre_gen_opt.step()

        # ====================
        gen_loss_list.append(pre_gen_loss.item())
        train_cnt += 1
        if train_cnt % 100 == 0:
            print('-Train %d / %d' % (train_cnt, num_train_snaps))
    gen_loss_mean = np.mean(gen_loss_list)
    print('#%d Pre-Train G-Loss %f' % (epoch, gen_loss_mean))

    # ====================
    # Validate the model
    gen_net.eval()
    disc_net.eval()
    # ==========
    RMSE_list = []
    MAE_list = []
    MLSD_list = []
    MR_list = []
    for tau in range(num_snaps-num_test_snaps-num_val_snaps, num_snaps-num_test_snaps):
        # ====================
        sup_list = [] # List of GNN support (tensor)
        noise_list = [] # List of random noise inputs
        for t in range(tau-win_size, tau):
            # ==========
            edges = edge_seq[t]
            adj = get_adj_wei(edges, num_nodes, max_thres)
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
        # Get the prediction result
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
                if adj_est[r, c]<=epsilon:
                    adj_est[r, c] = 0

        # ====================
        # Get the ground-truth
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
    print('Val Pre-#%d RMSE %f %f MAE %f %f MLSD %f %f MR %f %f'
          % (epoch, RMSE_mean, RMSE_std, MAE_mean, MAE_std, MLSD_mean, MLSD_std, MR_mean, MR_std))
    # ==========
    f_input = open('res/%s_IDEA_rec.txt' % (data_name), 'a+')
    f_input.write('Val Pre #%d RMSE %f %f MAE %f %f MLSD %f %f MR %f %f\n'
            % (epoch, RMSE_mean, RMSE_std, MAE_mean, MAE_std, MLSD_mean, MLSD_std, MR_mean, MR_std))
    f_input.close()

    # ====================
    # Test the model
    gen_net.eval()
    disc_net.eval()
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
        # Get the prediction result
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
                if adj_est[r, c]<=epsilon:
                    adj_est[r, c] = 0

        # ====================
        # Get the ground-truth
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
    print('Test Pre-#%d RMSE %f %f MAE %f %f MLSD %f %f MR %f %f\n'
          % (epoch, RMSE_mean, RMSE_std, MAE_mean, MAE_std, MLSD_mean, MLSD_std, MR_mean, MR_std))
    # ==========
    f_input = open('res/%s_IDEA_rec.txt' % (data_name), 'a+')
    f_input.write('Test Pre #%d RMSE %f %f MAE %f %f MLSD %f %f MR %f %f\n'
            % (epoch, RMSE_mean, RMSE_std, MAE_mean, MAE_std, MLSD_mean, MLSD_std, MR_mean, MR_std))
    f_input.write('\n')
    f_input.close()

# ====================
# Joint optimization of the generator & discriminator
for epoch in range(num_epochs):
    # ====================
    # Train the model
    gen_net.train()
    disc_net.train()
    # ==========
    train_cnt = 0
    disc_loss_list = []
    gen_loss_list = []
    for tau in range(win_size, num_train_snaps):
        # ====================
        sup_list = [] # List of GNN support (tensor)
        noise_list = [] # List of random noise inputs
        for t in range(tau-win_size, tau):
            # ==========
            edges = edge_seq[t]
            adj = get_adj_wei(edges, num_nodes, max_thres)
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
        gnd_list = []
        real_sup_list = []
        for t in range(tau-win_size+1, tau+1):
            # ==========
            edges = edge_seq[t]
            gnd = get_adj_wei(edges, num_nodes, max_thres)
            gnd_norm = gnd/max_thres # Normalize the edge weights to [0, 1]
            gnd_norm += np.eye(num_nodes)
            gnd_tnr = torch.FloatTensor(gnd_norm).to(device)
            gnd_list.append(gnd_tnr)
            # ==========
            # Transfer the GNN support to a sparse tensor
            sup = get_gnn_sup_woSE(gnd_norm)
            sup_tnr = torch.FloatTensor(sup).to(device)
            real_sup_list.append(sup_tnr)

        # ====================
        # Train the discriminator
        adj_est_list = gen_net(sup_list, feat_list, noise_list, align_list, num_nodes_list, lambd, pred_flag=False)
        disc_real_list, disc_fake_list = disc_net(real_sup_list, feat_list, adj_est_list, feat_list)
        disc_loss = get_disc_loss(disc_real_list, disc_fake_list, theta)
        disc_opt.zero_grad()
        disc_loss.backward()
        disc_opt.step()
        # ==========
        # Train the generator
        adj_est_list = gen_net(sup_list, feat_list, noise_list, align_list, num_nodes_list, lambd, pred_flag=False)
        _, disc_fake_list = disc_net(real_sup_list, feat_list, adj_est_list, feat_list)
        gen_loss = get_gen_loss(adj_est_list, gnd_list, disc_fake_list, max_thres, alpha, beta, theta)
        gen_opt.zero_grad()
        gen_loss.backward()
        gen_opt.step()

        # ====================
        gen_loss_list.append(gen_loss.item())
        disc_loss_list.append(disc_loss.item())
        train_cnt += 1
        if train_cnt % 100 == 0:
            print('-Train %d / %d' % (train_cnt, num_train_snaps))
    gen_loss_mean = np.mean(gen_loss_list)
    disc_loss_mean = np.mean(disc_loss_list)
    print('#%d Train G-Loss %f D-Loss %f' % (epoch, gen_loss_mean, disc_loss_mean))
    # ====================
    # Save the trained model (w.r.t. current epoch)
    if save_flag:
        torch.save(gen_net, 'pt/IDEA_gen_%d.pkl' % (epoch))
        torch.save(disc_net, 'pt/IDEA_disc_%d.pkl' % (epoch))

    # ====================
    # Validate the model
    gen_net.eval()
    disc_net.eval()
    # ==========
    RMSE_list = []
    MAE_list = []
    MLSD_list = []
    MR_list = []
    for tau in range(num_snaps-num_test_snaps-num_val_snaps, num_snaps-num_test_snaps):
        # ====================
        sup_list = [] # List of GNN support (tensor)
        noise_list = [] # List of random noise inputs
        for t in range(tau-win_size, tau):
            # ==========
            edges = edge_seq[t]
            adj = get_adj_wei(edges, num_nodes, max_thres)
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
        # Get the prediction result
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
                if adj_est[r, c]<=epsilon:
                    adj_est[r, c] = 0

        # ====================
        # Get the ground-truth
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
    print('Val #%d RMSE %f %f MAE %f %f MLSD %f %f MR %f %f'
          % (epoch, RMSE_mean, RMSE_std, MAE_mean, MAE_std, MLSD_mean, MLSD_std, MR_mean, MR_std))
    # ==========
    f_input = open('res/%s_IDEA_rec.txt' % (data_name), 'a+')
    f_input.write('Val #%d RMSE %f %f MAE %f %f MLSD %f %f MR %f %f\n'
            % (epoch, RMSE_mean, RMSE_std, MAE_mean, MAE_std, MLSD_mean, MLSD_std, MR_mean, MR_std))
    f_input.close()

    # ====================
    # Test the model
    gen_net.eval()
    disc_net.eval()
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
        # Get the prediction result
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
        # Get the ground-truth
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
    print('Test #%d RMSE %f %f MAE %f %f MLSD %f %f MR %f %f\n'
          % (epoch, RMSE_mean, RMSE_std, MAE_mean, MAE_std, MLSD_mean, MLSD_std, MR_mean, MR_std))
    # ==========
    f_input = open('res/%s_IDEA_rec.txt' % (data_name), 'a+')
    f_input.write('Test #%d RMSE %f %f MAE %f %f MLSD %f %f MR %f %f\n'
            % (epoch, RMSE_mean, RMSE_std, MAE_mean, MAE_std, MLSD_mean, MLSD_std, MR_mean, MR_std))
    f_input.write('\n')
    f_input.close()
