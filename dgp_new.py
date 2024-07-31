"""
Generate symmetric bilinear MAR time series {Y} with dimension N x N given length T, coefficients {Aj} with j from 1 to P
"""
import torch
import logging
from scipy.stats import ortho_group

from utils import generate_G_new,f_trans,vech,check_stationarity,f_trans_linear

def DGP_MAR_new(N,T,burn,P,g):
    """
    Input:
        P: MAR order
        g: the diagonal vector of core tensor G
    Output:
        y[T,N(N+1)/2] with zero mean
    """
    # generate A
    G = generate_G_new(g)
    r1,r2,r3 = G.shape
    U1 = torch.tensor(ortho_group.rvs(N)[:,:r1], dtype=torch.float)
    U2 = torch.tensor(ortho_group.rvs(N)[:,:r2], dtype=torch.float)
    U3 = torch.tensor(ortho_group.rvs(P)[:,:r3], dtype=torch.float)
    A = torch.einsum('ij,klj->kli',U3,torch.einsum('ij,kjl->kil',U2,torch.einsum('ij,jkl->ikl',U1,G)))
    # A = A_0 * 2 / torch.sqrt(torch.tensor(r1))
    f_A = f_trans(A) # if non-stationary, A_j = A_j * torch.pow(\lambda,j)
    
    # generate symmetric Y
    y = torch.zeros((int(N*(N+1)/2),T+burn)) #[N(N+1)/2,T]
    eps = torch.zeros((N,N,T+burn))
    for t in range(P,T+burn):
        # eps[:,:,t] = torch.eye(N) @ (torch.diag(torch.pow(torch.randn(N), 2)) - torch.eye(N)) @ torch.eye(N) # with demean
        # if t < burn:# let eps[:,:,:burn] <> 0 and eps[:,:,burn:] = 0 to check the true loss
        eps[:,:,t] = torch.eye(N) @ torch.diag(torch.pow(torch.randn(N), 2)) @ torch.eye(N) # without demean
        y[:,t] = torch.einsum('ijk,jk->i',f_A,torch.flip(y[:,t-P:t],[1])) + vech(eps[:,:,t])
    
    # demean Y
    y_demeaned = y[:,burn:].T - torch.mean(y[:,burn:].T, dim=0)
    if not check_stationarity(y_demeaned):
        logging.info('Stationarity test is not passed!')
        print('Stationarity test is not passed!')
    # assert check_stationarity(y_demeaned), "stationary test not pass!"
    
    return (y_demeaned,A,G,U1,U2,U3,f_A)