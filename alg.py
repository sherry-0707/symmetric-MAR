import torch
import numpy as np
# import gc
from tensorly.decomposition import tucker
import logging
from scipy.stats import ortho_group

from utils import tucker_product,vech,f_trans,unfold,generate_X,f_trans_linear
# from utils_glasso import fold,unfold,tucker_product

## Initialization
def spectral_initialize(y,X,r1,r2,r3):
    """
    Spectral initialization
    Transform MAR to VAR by vectorization
    y (T, N**2)
    X (T, N**2, P)
    A kron A = sum(outer(y[t:],X[t,:,:]))/T
    project A kron A to A, then HOOI
    return G, U1, U2, U3
    """
    T,N_square = y.shape
    # spectral initialization
    A_kron = torch.zeros((N_square,N_square*P))
    for t in range(T):
        A_kron = A_kron + torch.outer(y[t,:],X[t,:,:])
    A_kron = A_kron/(T-P)
    # fold A into a tensor
    A = kron_to_A(A_kron)

    # HOOI to get a low rank version
    G,U = tucker(A,rank=[r1,r2,r3])
    return G,U[0],U[1],U[2]

def GD_initialize(y,X,r1,r2,r3,a,b,step_size,iter,true_A,print_log):
    """
    Initialization of G cannot be zero as gradient_G has factor G, which will result in zero gradient
    """
    T,N_half_square,P = X.shape # (T, N*(N+1)/2, P)
    N = int(torch.sqrt(torch.tensor(2*N_half_square+0.25))-0.5)
    U1 = torch.tensor(ortho_group.rvs(N)[:,:r1])
    U2 = torch.tensor(ortho_group.rvs(N)[:,:r2])
    U3 = torch.tensor(ortho_group.rvs(P)[:,:r3])
    G = torch.rand((r1,r2,r3),dtype=torch.double)
    best_loss = torch.inf

    for i in range(iter):
        G, U1, U2, U3 = AGD(G, U1, U2, U3, y, X, a, b, step_size)
        A = tucker_product(G, U1, U2, U3)
        y_hat = torch.einsum('TNP,iNP->Ti',X,f_trans(A))
        loss = torch.sum(torch.norm(y_hat-y,dim=1)) / T
        if true_A is not None:
            err = torch.norm(unfold(A - true_A[:, :, :], 1), p='fro') ** 2
        if print_log:
            # print(err)
            print('init iter: {}, init loss: {}'.format(i, loss))

        if loss < best_loss:
            best_loss = loss
            G_best = G
            U1_best = U1
            U2_best = U2
            U3_best = U3
    # print("fro norm list: {}".format(torch.norm(A, p='fro', dim=(0, 1))))
    return G_best, U1_best, U2_best, U3_best

def GD_initialize_lasso(y,X,r1,r2,a,b,step_size,iter,true_A,print_log):
    T,N,P = X.shape
    gradient_A = 2*np.einsum('TNP,Ti->iNP', X,-y)/T # (N,N,P)
    if np.any(np.isnan(gradient_A)):
        raise ValueError
    U1 = ortho_group.rvs(N)[:,:r1]
    U2 = ortho_group.rvs(N)[:,:r2]
    G = np.zeros((r1,r2,P))
    best_loss = np.inf

    for i in range(iter):
    # gradient step for U1, U2, G
        U1 = U1 - step_size * (unfold(gradient_A,1) @ np.kron(np.identity(P),U2) @ unfold(G,1).T + a * (U1@(U1.T@U1-b**2*np.identity(r1))))
        U2 = U2 - step_size * (unfold(gradient_A,2) @ np.kron(U1,np.identity(P)) @ unfold(G,2).T + a * (U2@(U2.T@U2-b**2*np.identity(r2))))
        G = G - step_size * (tucker_product(gradient_A,U1.T,U2.T))
        # hard thresholding
        A = tucker_product(G,U1,U2)
        y_hat = np.einsum('TNP,iNP->Ti',X,A)
        gradient_A = 2*np.einsum('TNP,Ti->iNP', X,y_hat-y)/T # (N,N,P)
        loss = np.sum(np.linalg.norm(y_hat-y, ord=2, axis=1))/T
        if true_A is not None:
            err = np.linalg.norm(unfold(A-true_A[:,:,:P],1),ord='fro')**2
        if print_log:
            # print(err)
            print('init iter: {}, init loss: {}'.format(i,loss))

        if loss < best_loss:
            best_loss = loss
            G_best = G; U1_best = U1; U2_best = U2
    # print("fro norm list: {}".format(np.linalg.norm(A,ord='fro',axis=(0,1))))
    return G_best,U1_best,U2_best

## Algorithm componnets

def AGD(G_old,U1_old,U2_old,U3_old,y,P,a,b,step_size):
    """
    Alternating gradient descent algorithm
    Input:
        G (r_1, r_2, r_3)
        U1 (N, r_1)
        U2 (N, r_2)
        U3 (N, r_3)
        y [T, N(N+1)/2]
        a = 1
        b = 1
        step_size: learning rate
    Output:
        A (N, N, P)
    """
    T, _ = y.shape
    _, r1 = U1_old.shape
    _, r2 = U2_old.shape
    _, r3 = U3_old.shape
    
    G = G_old.detach().clone()
    U1 = U1_old.detach().clone()
    U2 = U2_old.detach().clone()
    U3 = U3_old.detach().clone()

    G.requires_grad_(True)
    U1.requires_grad_(True)
    U2.requires_grad_(True)
    U3.requires_grad_(True)
    
    # G = G_old.clone().detach().requires_grad_(True)
    # U1 = U1_old.clone().detach().requires_grad_(True)
    # U2 = U2_old.clone().detach().requires_grad_(True)
    # U3 = U3_old.clone().detach().requires_grad_(True)

    # calculate A and loss function L
    A = tucker_product(G,U1,U2,U3)
    y_hat = torch.einsum('TNP,iNP->Ti',generate_X(y,P),f_trans(A))
    loss = torch.sum(torch.norm(y_hat-y[P:,:],p=2,dim=1))/T
    loss.backward()
        
    # gradient step for U1, U2, G
    gradient_U1, gradient_U2, gradient_U3, gradient_G = U1.grad.detach(), U2.grad.detach(), U3.grad.detach(), G.grad.detach()
    # print('gradient: ',np.linalg.norm(unfold(gradient_A,1),ord='fro'))
    if torch.any(torch.isnan(gradient_G)):
        logging.critical('nan appeared in gradient_G, skip this replication')
        return (None, None, None, None, None)

    del U1.grad, U2.grad, U3.grad, G.grad  # Delete unnecessary variables
    # gc.collect()  # Perform garbage collection

    # gradient step for U1, U2, G
    with torch.no_grad():
        U1_new = U1 - step_size * (gradient_U1 + a * (U1 @ (U1.T @ U1 - b ** 2 * torch.eye(r1))))
        U2_new = U2 - step_size * (gradient_U2 + a * (U2 @ (U2.T @ U2 - b ** 2 * torch.eye(r2))))
        U3_new = U3 - step_size * (gradient_U3 + a * (U3 @ (U3.T @ U3 - b ** 2 * torch.eye(r3))))
        G_new = G - step_size * gradient_G
        A = tucker_product(G_new, U1_new, U2_new, U3_new)

    return G_new, U1_new, U2_new, U3_new #, loss

def train_epoch(y,P,r1,r2,r3,a,b,stop_method,schedule,max_iter=10000,stop_thres=1e-5,step_size=1e-3,init_step_size=5e-3,init_iter=1000,
                         true_G=None,true_U1=None,true_U2=None,true_U3=None,A_init_method=None,print_log=False):
    """
    The main train function
    Input:
        y [T, N(N+1)/2]
    Output:
        A_best,best_loss,err_path,est_path for stop_method == 'loss'
        A,loss,err_path,est_path for else stop_method
    """
    
    T, N_half_square = y.shape
    N = int(torch.sqrt(torch.tensor(2*N_half_square+0.25))-0.5)
    true_A = tucker_product(true_G,true_U1,true_U2,true_U3)
    # get initial values of G,U1,U2,U3
    if A_init_method == 'true':
        G,U1,U2,U3 = true_G,true_U1,true_U2,true_U3
    elif A_init_method == 'noisetrue_G':
        # noise = torch.einsum('ijk,k->ijk', torch.randn(size=true_G.shape) * 0.3,torch.tensor([1/t for t in range(1,P+1)]))
        G = true_G + torch.randn(size=true_G.shape) * 0.1
        U1,U2,U3 = true_U1,true_U2,true_U3
    elif A_init_method == 'noisetrue_U':
        G = true_G
        U1 = true_U1 + torch.randn(size=true_U1.shape) * 0.1
        U2 = true_U2 #+ torch.randn(size=true_U2.shape) * 0.1
        U3 = true_U3 #+ torch.randn(size=true_U3.shape) * 0.1
    elif A_init_method == 'noisetrue_A':
        # noise = torch.einsum('ijk,k->ijk', torch.randn(size=true_A.shape) * 0.3,torch.tensor([1/t for t in range(1,P+1)]))
        A_init = true_A + torch.randn(size=true_A.shape) * 0.1
        # print(torch.norm(A_init, p='fro', dim=(0, 1)))
        G,U = tucker(A_init, rank=[r1,r2,r3])
        U1 = U[0]; U2 = U[1]; U3 = U[2]
    elif A_init_method == 'random':
        U1 = torch.tensor(ortho_group.rvs(N)[:,:r1])
        U2 = torch.tensor(ortho_group.rvs(N)[:,:r2])
        U3 = torch.tensor(ortho_group.rvs(P)[:,:r3])
        G = torch.rand((r1,r2,r3),dtype=torch.double)
    # elif A_init_method == 'spec':
    #     G,U1,U2,U3 = spectral_initialize(y=y,X=X,r1=r1,r2=r2,r3=r3)
    # elif A_init_method == 'GD':
    #     G,U1,U2,U3 = GD_initialize(y=y,X=X,r1=r1,r2=r2,r3=r3,a=a,b=b,step_size=init_step_size,iter=init_iter,true_A=true_A,print_log=print_log)
        

    A_old = tucker_product(G,U1,U2,U3)
    best_loss = torch.tensor(float('inf')); A_diff = torch.tensor(float('inf')); loss_increase_count = 0
    err_path = torch.zeros(max_iter); est_path = torch.zeros(max_iter); f_A_est_path = torch.zeros(max_iter)
    for iter in range(max_iter):
        ################# lr schedule ###############
        if schedule == 'half':
            schedule_step = 500
            if iter % schedule_step == schedule_step -1:
                step_size *= 0.8

        ################# update G,U1,U2,U3 ###############
        G,U1,U2,U3 = AGD(G_old=G,U1_old=U1,U2_old=U2,U3_old=U3,y=y,P=P,a=a,b=b,step_size=step_size)   
        
        if G is None:
            return None,None,None,None,None
        A = tucker_product(G,U1,U2,U3)
        A_diff = torch.norm(unfold(A-A_old,1), p='fro')
        A_old = A

        # calculate loss and keep track of best loss        
        y_hat = torch.einsum('TNP,iNP->Ti',generate_X(y,P),f_trans(A))
        loss = torch.sum(torch.norm(y_hat-y[P:,:],p=2,dim=1))/T
        if loss > best_loss:
            loss_increase_count += 1
        else:
            best_loss = loss; loss_increase_count = 0
            A_best = A

        # for convergence log
        if true_A is not None:
            err_path[iter] = torch.norm(unfold(A-true_A,1), p='fro')**2 + torch.norm(unfold(true_A,1), p='fro')**2
            # true_A_copy = true_A.clone() # pay attension! true_A.copy()
            est_path[iter] = torch.norm(unfold(A-true_A,1), p='fro')**2
            f_A_est_path[iter] = torch.norm(f_trans(A)-f_trans(true_A), p='fro')**2
        if print_log and iter % 50 == 0:
            print('iter: {}, loss: {}, est err: {}'.format(iter, loss, est_path[iter]))
            # print('est err: {}, A fnorm: {}'.format(est_path[iter], torch.norm(unfold(A,1), p='fro')**2))
            # print('f_A est err: {}, f_A fnorm: {}'.format(f_A_est_path[iter], torch.norm(f_trans(A), p='fro')**2))
        
        ################## stop criteria ##################
        if ((stop_method == 'loss') and (A_diff < stop_thres or loss_increase_count > 500)):
            logging.info('training end, iter: {}, loss: {}'.format(iter,loss))
            # threshold the returned A
            return A_best,best_loss,err_path,est_path
        elif (stop_method == 'Adiff' and A_diff < stop_thres): 
            # mix two conditions:
            # 1. frobenius norm of (A_k - A_{k-1}) < epsilon
            # 2. loss does not decrease for 200 iterations
            logging.info('training end, iter: {}, loss: {}'.format(iter,loss))
            return A,loss,err_path,est_path

    logging.info('training end, iter: {}, loss: {}'.format(iter,loss))
    if stop_method == 'loss':
        return A_best,best_loss,err_path,est_path
    else:
        return A,loss,err_path,est_path

