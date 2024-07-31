import torch
import numpy as np
from statsmodels.tsa.stattools import adfuller

from dgp import DGP_MAR
from alg import train_epoch
from utils import generate_X, unfold, f_trans

# Generate sample multivariate time series data
# N,T,dgp_p,dgp_r = 5,2700,4,4

N,T,dgp_p,dgp_r,dgp_core = 20,500,10,4,0.2
g = torch.tensor([dgp_core]*dgp_r)

# T_list = [500,590,710,910,1250,2000]
# for exp_T in T_list:
#     for n in range(50):
#         torch.manual_seed(n)
#         y,true_A,true_G,true_U1,true_U2,true_U3,f_A = DGP_MAR_new(N,exp_T,burn=600,P=dgp_p,g=g)
#         print('test {}: T = {}: stationary'.format(n, exp_T))

y,true_A,true_G,true_U1,true_U2,true_U3,f_A = DGP_MAR(N,T,burn=600,P=dgp_p,g=g)

# print(true_A.size())
# y_hat = torch.einsum('TNP,iNP->Ti',generate_X(y,dgp_p),f_A)
# true_loss = torch.sum(torch.norm(y_hat-y[dgp_p:,:],p=2,dim=1))/T

# A,loss,err_path,est_path = train_epoch(y=y,P=dgp_p,r1=dgp_r,r2=dgp_r,r3=dgp_r,a=1,b=1,stop_method='loss',schedule='half',max_iter=10000,stop_thres=1e-5,
#                                        step_size=2e-2,init_step_size=5e-3,init_iter=1000,true_G=true_G,true_U1=true_U1,true_U2=true_U2,true_U3=true_U3,
#                                        A_init_method='noisetrue_G',print_log=True)

# print("------------------------------------------")
# print("true loss is: {} | best loss is: {}".format(true_loss, loss))
# print("true_A fnorm: {} | estimation error: {}".format(torch.norm(unfold(true_A,1), p='fro')**2, torch.norm(unfold(A-true_A,1), p='fro')**2))
# print("f_trans(true_A) - f_trans(A) fnorm: {}".format(torch.norm(unfold(f_trans(A)-f_trans(true_A),1), p='fro')**2))



