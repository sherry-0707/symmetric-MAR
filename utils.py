import torch
import numpy as np
from scipy.stats import ortho_group
from statsmodels.tsa.stattools import adfuller

########### use numpy ###########
def fold_np(matrix, shape, mode):
    """
    Fold a matrix into a three-way tensor
    """
    if mode == 1:
        shape = [shape[0], shape[2], shape[1]]
        tensor = np.transpose(matrix.reshape(shape),(0, 2, 1))
    elif mode == 2:
        shape = [shape[1], shape[2], shape[0]]
        tensor = np.transpose(matrix.reshape(shape),(2, 0, 1))
    elif mode == 3:
        shape = [shape[2], shape[1], shape[0]]
        tensor = np.transpose(matrix.reshape(shape),(2, 1, 0))
    return tensor

def unfold_np(tensor, mode):
    """
    Unfold a three-way tensor into a matrix
    """
    shape = tensor.shape
    if mode == 1:
        matrix = np.transpose(tensor,(0, 2, 1)).reshape(shape[0], -1)
    elif mode == 2:
        matrix = np.transpose(tensor,(1, 2, 0)).reshape(shape[1], -1)
    elif mode == 3:
        matrix = np.transpose(tensor,(2, 1, 0)).reshape(shape[2], -1)
    return matrix

########### use torch ###########
def generate_G(g):
    """
    Generate a diagonal core tensor G with diag(G) being g
    """
    r = g.shape[0]
    G = torch.zeros((r,r,r))
    for i in range(r):
        G[i,i,i] = g[i]
    return G

def generate_G_new(g):
    """
    Generate a diagonal core tensor G with diag(G) being g
    """
    r = g.shape[0]
    G = torch.zeros((r,r,r))
    for i in range(r):
        G[:,:,i] = torch.diag(g)
    return G

def generate_A(G,N,P):
    """
    Generate low-rank A by core tensor G and random orthogonal matrices U1, U2, U3
    Output:
        G times_1 U1 times_2 U2 times_3 U3
    """
    r1,r2,r3 = G.shape
    U1 = torch.tensor(ortho_group.rvs(N)[:,:r1], dtype=torch.float)
    U2 = torch.tensor(ortho_group.rvs(N)[:,:r2], dtype=torch.float)
    U3 = torch.tensor(ortho_group.rvs(P)[:,:r3], dtype=torch.float)
    return torch.einsum('ij,klj->kli',U3,torch.einsum('ij,kjl->kil',U2,torch.einsum('ij,jkl->ikl',U1,G)))

def generate_X(y,P):
    """
    generate X [T-P,int(N*(N+1)/2),P] from y[T,int(N*(N+1)/2)]
    """
    T,N = y.shape
    X = torch.zeros(T-P,N,P)
    for i in range(P):
        X[:,:,i] = y[(P-1-i):T-1-i,:]
    return X

def tucker_product(G,U1,U2,U3):
    """
    Compute tucker product in all three modes
    Output:
        G times_1 U1 times_2 U2 times_3 U3
    """
    if any(x is None for x in [G, U1, U2, U3]):
        return None
    
    return torch.einsum('ij,klj->kli',U3,torch.einsum('ij,kjl->kil',U2,torch.einsum('ij,jkl->ikl',U1,G)))

def vech(Y):
    """
    Return the half-vectorization of a symmetric Y
    Output:
        vech(Y) with length N(N+1)/2
    """
    N = Y.size(0)
    mask = torch.tril(torch.ones(N, N))  # Create a lower triangular mask
    return Y[mask.bool()]  # Apply the mask to extract the lower triangular elements

def kappa_operation(A,B):
    """
    Operation kappa
    Out put:
        K(A,B) = (A_1 kron B_1, A_2 kron B_2, ..., A_P kron B_P)
    """
    N, _, P = A.size()
    K = torch.zeros((N*N, N*N, P))
    for i in range(P):
        K[:,:,i] = torch.kron(A[:,:,i],B[:,:,i])
    return K

def select(N):
    """
    Select function to generate output based on given cases.
    """
    if N == 1:
        return torch.tensor([1])

    output = torch.tensor([1])
    for i in range(1,N):
        indices = torch.arange(start=i*N+1, end=i*N+i+2,step=1)
        output = torch.cat((output, indices))

    return output

def f_trans(A):
    """
    Corresponding transformation of kron(A,A) for vec(Y) to vech(Y)
    Input: A(N,N,P)
    Output: f(K(A,A))
    """
    N, _, P = A.size()
    kappa_A = kappa_operation(A,A)
    index_to_save = select(N)-torch.ones(int(N*(N+1)/2))
    f_trans = torch.zeros((int(N*(N+1)/2),int(N*(N+1)/2),P))
    for t in range(P):
        tensor1 = torch.index_select(kappa_A[:,:,t], 0, torch.tensor([i for i in range(N*N) if i in index_to_save])) # Delete rows
        f_trans[:,:,t] = torch.index_select(tensor1, 1, torch.tensor([j for j in range(N*N) if j in index_to_save])) # Delete columns
    return f_trans

def f_trans_linear(A):
    """
    Corresponding transformation of kron(A,A) for vec(Y) to vech(Y)
    Input: A(N,N,P)
    Output: f(K(A,A))
    """
    N, _, P = A.size()
    kappa_A = torch.zeros((N*N, N*N, P))
    for i in range(P):
        kappa_A[:,:,i] = torch.kron(torch.eye(N),A[:,:,i])

    index_to_save = select(N)-torch.ones(int(N*(N+1)/2))
    f_trans = torch.zeros((int(N*(N+1)/2),int(N*(N+1)/2),P))
    for t in range(P):
        tensor1 = torch.index_select(kappa_A[:,:,t], 0, torch.tensor([i for i in range(N*N) if i in index_to_save])) # Delete rows
        f_trans[:,:,t] = torch.index_select(tensor1, 1, torch.tensor([j for j in range(N*N) if j in index_to_save])) # Delete columns
    return f_trans

def reorder_matrix(matrix):
    num_columns = matrix.shape[1]
    reversed_indices = torch.arange(num_columns - 1, -1, -1)
    reordered_matrix = matrix[:, reversed_indices]
    return reordered_matrix

def unfold(tensor, mode):
    """
    Unfold a three-way tensor into a matrix
    """
    shape = tensor.shape
    if mode == 1:
        matrix = tensor.permute(0, 2, 1).reshape(shape[0], -1)
    elif mode == 2:
        matrix = tensor.permute(1, 2, 0).reshape(shape[1], -1)
    elif mode == 3:
        matrix = tensor.permute(2, 1, 0).reshape(shape[2], -1)
    return matrix

def check_stationarity(time_series_data):
    num_variables = time_series_data.shape[1]
    all_stationary = True

    for i in range(num_variables):
        variable_data = time_series_data[:,i]
        result = adfuller(variable_data)

        if result[1] > 0.1:
            all_stationary = False 

        # print(f"Variable {i+1}:")
        # print(f"ADF Statistic: {result[0]}")
        # print(f"p-value: {result[1]}")
        # print("Stationary" if result[1] <= 0.001 else "Non-stationary")
        # print()

    return all_stationary


