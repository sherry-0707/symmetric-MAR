"""
Generate symmetric bilinear MAR time series {Y} with dimension N x N given length T, coefficients {Aj} with j from 1 to P
"""
import torch
import logging
from scipy.stats import ortho_group

from utils import generate_G,f_trans,vech,check_stationarity,f_trans_linear

def DGP_MAR(N,T,burn,P,g,seed):
    """
    Input:
        P: MAR order
        g: the diagonal vector of core tensor G
    Output:
        y[T,N(N+1)/2] with zero mean
    """
    torch.manual_seed(seed)
    # generate A
    G = generate_G(g)
    r1,r2,r3 = G.shape
    # U1 = torch.tensor(ortho_group.rvs(N)[:,:r1], dtype=torch.float)
    # U2 = torch.tensor(ortho_group.rvs(N)[:,:r2], dtype=torch.float)
    # U3 = torch.tensor(ortho_group.rvs(P)[:,:r3], dtype=torch.float)
    Matrix_1 = torch.randn(N, N)
    Matrix_2 = torch.randn(P, P)
    Q_1, _ = torch.linalg.qr(Matrix_1)
    Q_2, _ = torch.linalg.qr(Matrix_2)
    U1 = Q_1[:, :r1].clone().detach()
    U2 = Q_1[:, :r2].clone().detach()
    U3 = Q_2[:, :r3].clone().detach()
    A = torch.einsum('ij,klj->kli',U3,torch.einsum('ij,kjl->kil',U2,torch.einsum('ij,jkl->ikl',U1,G)))
    # A = A_0 * 2 / torch.sqrt(torch.tensor(r1))
    f_A = f_trans(A) # if non-stationary, A_j = A_j * torch.pow(\lambda,j)
    
    # generate symmetric \Sigma_e^{1/2}
    Sigma_half = torch.rand(N, N)
    Sigma_half_symmetric = (Sigma_half+Sigma_half.T)/2
    
    # generate symmetric Y
    y = torch.zeros((int(N*(N+1)/2),T+burn)) #[N(N+1)/2,T]
    eps = torch.zeros((N,N,T+burn))
    for t in range(P,T+burn):
        # eps[:,:,t] = torch.eye(N) @ (torch.diag(torch.pow(torch.randn(N), 2)) - torch.eye(N)) @ torch.eye(N) # with demean
        # if t < burn:# let eps[:,:,:burn] <> 0 and eps[:,:,burn:] = 0 to check the true loss
        eps[:,:,t] = Sigma_half_symmetric @ torch.diag(torch.pow(torch.randn(N), 2)) @ Sigma_half_symmetric # without demean
        y[:,t] = torch.einsum('ijk,jk->i',f_A,torch.flip(y[:,t-P:t],[1])) + vech(eps[:,:,t])
    
    # demean Y
    y_demeaned = y[:,burn:].T - torch.mean(y[:,burn:].T, dim=0)
    if not check_stationarity(y_demeaned):
        logging.info('Stationarity test is not passed!')
        print('Stationarity test is not passed!')
    # assert check_stationarity(y_demeaned), "stationary test not pass!"
    
    return (y_demeaned,A,G,U1,U2,U3,f_A)

def DGP_MAR_threshold(N,T,burn,P,g,seed,scale_e,delta_A=1e-5):
    """
    Input:
        P: MAR order
        g: the diagonal vector of core tensor G
    Output:
        y[T,N(N+1)/2] with zero mean
    """
    torch.manual_seed(seed)
    # generate A
    G = generate_G(g)
    r1,r2,r3 = G.shape
    # U1 = torch.tensor(ortho_group.rvs(N)[:,:r1], dtype=torch.float)
    # U2 = torch.tensor(ortho_group.rvs(N)[:,:r2], dtype=torch.float)
    # U3 = torch.tensor(ortho_group.rvs(P)[:,:r3], dtype=torch.float)
    Matrix_1 = torch.randn(N, N)
    Matrix_2 = torch.randn(P, P)
    Q_1, _ = torch.linalg.qr(Matrix_1)
    Q_2, _ = torch.linalg.qr(Matrix_2)
    U1 = Q_1[:, :r1].clone().detach()
    U2 = Q_1[:, :r2].clone().detach()
    U3 = Q_2[:, :r3].clone().detach()
    A = torch.einsum('ij,klj->kli',U3,torch.einsum('ij,kjl->kil',U2,torch.einsum('ij,jkl->ikl',U1,G)))
    # A = A_0 * 2 / torch.sqrt(torch.tensor(r1))
    
    # Make the minimum absolute value equal to delta_A
    min_value = torch.min(torch.abs(A))
    if min_value > delta_A:
        A[torch.abs(A) == min_value] = delta_A
    else:
        A[torch.abs(A) < delta_A] = delta_A
    f_A = f_trans(A) # if non-stationary, A_j = A_j * torch.pow(\lambda,j)
    
    Sigma_half_symmetric = scale_e * torch.eye(N)
    
    # generate symmetric Y
    y = torch.zeros((int(N*(N+1)/2),T+burn)) #[N(N+1)/2,T]
    eps = torch.zeros((N,N,T+burn))
    for t in range(P,T+burn):
        # eps[:,:,t] = torch.eye(N) @ (torch.diag(torch.pow(torch.randn(N), 2)) - torch.eye(N)) @ torch.eye(N) # with demean
        # if t < burn:# let eps[:,:,:burn] <> 0 and eps[:,:,burn:] = 0 to check the true loss
        eps[:,:,t] = Sigma_half_symmetric @ torch.diag(torch.pow(torch.randn(N), 2)) @ Sigma_half_symmetric # without demean
        y[:,t] = torch.einsum('ijk,jk->i',f_A,torch.flip(y[:,t-P:t],[1])) + vech(eps[:,:,t])
    
    # demean Y
    y_demeaned = y[:,burn:].T - torch.mean(y[:,burn:].T, dim=0)
    if not check_stationarity(y_demeaned):
        logging.info('Stationarity test is not passed!')
        print('Stationarity test is not passed!')
    # assert check_stationarity(y_demeaned), "stationary test not pass!"
    
    return (y_demeaned,A,G,U1,U2,U3,f_A)

