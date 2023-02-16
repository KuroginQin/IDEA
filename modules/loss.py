import torch

def get_pre_gen_loss(adj_est_list, gnd_list, theta):
    '''
    Function to define the pre-training loss of generator
    :param adj_est_list: list of prediction results
    :param gnd_list: list of ground-truth
    :param theta: parameter of decaying factor
    :return: pre-training loss of generator
    '''
    # ====================
    loss = 0.0
    win_size = len(adj_est_list)
    for i in range(win_size):
        adj_est = adj_est_list[i]
        gnd = gnd_list[i]
        decay = (1-theta)**(win_size-i-1) # Decaying factor
        # ==========
        loss += decay*torch.norm((adj_est - gnd), p='fro')**2 # MSE loss
        loss += decay*torch.sum(torch.abs(adj_est - gnd)) # MAE loss

    return loss

def get_gen_loss(adj_est_list, gnd_list, disc_fake_list, max_thres, alpha, beta, theta):
    '''
    Function to define the loss of generator (in the formal optimization)
    :param adj_est_list: list of prediction results
    :param gnd_list: list of ground-truth
    :param disc_fake_list: list of discriminator's outputs (w.r.t. the fake inputs)
    :param alpha: parameter to control ME loss
    :param beta: parameter to control SDM loss
    :param theta: parameter of decaying factor
    :return: loss of generator
    '''
    # ====================
    loss = 0.0
    win_size = len(adj_est_list)
    for i in range(win_size):
        adj_est = adj_est_list[i]
        gnd = gnd_list[i]
        disc_fake = disc_fake_list[i]
        decay = (1-theta)**(win_size-i-1) # Decaying factor
        # ==========
        # AL (adversarial learning) loss
        loss += -decay*torch.mean(torch.log(disc_fake+1e-15))
        # EM (error minimization) loss
        loss += decay*alpha*torch.norm((adj_est - gnd), p='fro')**2
        loss += decay*alpha*torch.sum(torch.abs(adj_est - gnd))
        # SDM (scale difference minimization) loss
        epsilon = 1e-5/max_thres
        E = epsilon*torch.ones_like(adj_est)
        q = adj_est
        q = torch.where(q<epsilon, E, q)
        p = gnd
        p = torch.where(p<epsilon, E, p)
        loss += decay*beta*torch.sum(torch.abs(torch.log10(p/q)))

    return loss

def get_disc_loss(disc_real_list, disc_fake_list, theta):
    '''
    Function to define the loss discriminator (in the formal optimization)
    :param disc_real_list: list of discriminator's outputs w.r.t. real inputs
    :param disc_fake_list: list of discriminator's outputs w.r.t. fake inputs
    :return: loss discriminator
    '''
    # ====================
    loss = 0.0
    epsilon = 1e-15
    win_size = len(disc_real_list)
    for i in range(win_size):
        disc_real = disc_real_list[i]
        disc_fake = disc_fake_list[i]
        decay = (1-theta)**(win_size-i-1) # Decaying factor
        #loss += decay*(torch.mean(torch.log(disc_fake+epsilon)) - torch.mean(torch.log(disc_real+epsilon)))
        loss -= decay*(torch.mean(torch.log(1-disc_fake+epsilon)) + torch.mean(torch.log(disc_real+epsilon)))

    return loss