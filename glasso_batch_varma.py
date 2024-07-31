import subprocess
import argparse
import os

## default VARMA setting ##
exp_name = f'varma'
r = 4
rho = 0.7
s = 10
N =200
T = 2000
T0 = 67
init = 'zero'
n_rep = 500
num_proc = 10
rep_per_proc = int(n_rep/num_proc)

## Exp 1: change N
exp_T0 = 20
for exp_N in [50,100,150,200]:
    for batch_no in range(num_proc):
        command = ['python','../train.py','--dgp','arma','--dgp_p','1','--dgp_r',str(r),'--rho',str(rho),'--n_rep',str(rep_per_proc),'--T',str(T),'--s',str(s),
                   '--T0',str(exp_T0),'--A_init',init,'--exp_name',exp_name,'--task','rate','--stop_method','Adiff','--schedule','half','--lr','1e-3',
                   '--N',str(exp_N),'--max_iter','100','--seed',str(rep_per_proc*batch_no)]
        subprocess.Popen(command)

exp_T0 = 37
for exp_N in [50,100,150,200]:
    for batch_no in range(num_proc):
        command = ['python','../train.py','--dgp','arma','--dgp_p','1','--dgp_r',str(r),'--rho',str(rho),'--n_rep',str(rep_per_proc),'--T',str(T),'--s',str(s),
                   '--T0',str(exp_T0),'--A_init',init,'--exp_name',exp_name,'--task','rate','--stop_method','Adiff','--schedule','half','--lr','1e-3',
                   '--N',str(exp_N),'--max_iter','100','--seed',str(rep_per_proc*batch_no)]
        subprocess.Popen(command)

exp_T0 = 67
for exp_N in [50,100,150,200]:
    for batch_no in range(num_proc):
        command = ['python','../train.py','--dgp','arma','--dgp_p','1','--dgp_r',str(r),'--rho',str(rho),'--n_rep',str(rep_per_proc),'--T',str(T),'--s',str(s),
                   '--T0',str(exp_T0),'--A_init',init,'--exp_name',exp_name,'--task','rate','--stop_method','Adiff','--schedule','half','--lr','1e-3',
                   '--N',str(exp_N),'--max_iter','100','--seed',str(rep_per_proc*batch_no)]
        subprocess.Popen(command)

# Exp 2: change T
for (exp_T,exp_T0) in [(840,16),(1050,17),(1390,18),(2070,20),(4100,24)]:
    for batch_no in range(num_proc):
        command = ['python','../train.py','--dgp','arma','--dgp_p','1','--dgp_r',str(r),'--rho',str(rho),'--n_rep',str(rep_per_proc),'--T',str(exp_T),'--s',str(s),
                   '--T0',str(T0),'--A_init',init,'--exp_name',exp_name,'--task','rate','--stop_method','Adiff','--schedule','half','--lr','1e-3','--N',str(N),
                   '--max_iter','100','--seed',str(rep_per_proc*batch_no)]
        subprocess.Popen(command)

for (exp_T,exp_T0) in [(840,28),(1050,30),(1390,33),(2070,38),(4100,48)]:
    for batch_no in range(num_proc):
        command = ['python','../train.py','--dgp','arma','--dgp_p','1','--dgp_r',str(r),'--rho',str(rho),'--n_rep',str(rep_per_proc),'--T',str(exp_T),'--s',str(s),
                   '--T0',str(T0),'--A_init',init,'--exp_name',exp_name,'--task','rate','--stop_method','Adiff','--schedule','half','--lr','1e-3','--N',str(N),
                   '--max_iter','100','--seed',str(rep_per_proc*batch_no)]
        subprocess.Popen(command)

for (exp_T,exp_T0) in [(840,43),(1050,48),(1390,55),(2070,68),(4100,96)]:
    for batch_no in range(num_proc):
        command = ['python','../train.py','--dgp','arma','--dgp_p','1','--dgp_r',str(r),'--rho',str(rho),'--n_rep',str(rep_per_proc),'--T',str(exp_T),'--s',str(s),
                   '--T0',str(T0),'--A_init',init,'--exp_name',exp_name,'--task','rate','--stop_method','Adiff','--schedule','half','--lr','1e-3','--N',str(N),
                   '--max_iter','100','--seed',str(rep_per_proc*batch_no)]
        subprocess.Popen(command)

## Exp 3: change r
exp_T0 = 20
for exp_r in [3,4,5,6]:
    for batch_no in range(num_proc):
        command = ['python','../train.py','--dgp','arma','--dgp_p','1','--dgp_r',str(exp_r),'--rho',str(rho),'--n_rep',str(rep_per_proc),'--T',str(T),'--s',str(s),
                   '--T0',str(exp_T0),'--A_init',init,'--exp_name',exp_name,'--task','rate','--stop_method','Adiff','--schedule','half','--lr','1e-3','--N',str(N),
                   '--max_iter','100','--seed',str(rep_per_proc*batch_no)]
        subprocess.Popen(command)

exp_T0 = 37
for exp_r in [3,4,5,6]:
    for batch_no in range(num_proc):
        command = ['python','../train.py','--dgp','arma','--dgp_p','1','--dgp_r',str(exp_r),'--rho',str(rho),'--n_rep',str(rep_per_proc),'--T',str(T),'--s',str(s),
                   '--T0',str(exp_T0),'--A_init',init,'--exp_name',exp_name,'--task','rate','--stop_method','Adiff','--schedule','half','--lr','1e-3','--N',str(N),
                   '--max_iter','100','--seed',str(rep_per_proc*batch_no)]
        subprocess.Popen(command)

exp_T0 = 67
for exp_r in [3,4,5,6]:
    for batch_no in range(num_proc):
        command = ['python','../train.py','--dgp','arma','--dgp_p','1','--dgp_r',str(exp_r),'--rho',str(rho),'--n_rep',str(rep_per_proc),'--T',str(T),'--s',str(s),
                   '--T0',str(exp_T0),'--A_init',init,'--exp_name',exp_name,'--task','rate','--stop_method','Adiff','--schedule','half','--lr','1e-3','--N',str(N),
                   '--max_iter','100','--seed',str(rep_per_proc*batch_no)]
        subprocess.Popen(command)

# exit(0)

# # uncomment the model if don't need to run
# model_list = [
#     # 'rw',
#     # 'ar',
#     # 'var',
#     'var_lasso',
#     'dfm',
#     'di',
#     # 'varma',
#     'zheng'
# ]
# data_path = f'/home/r6user2/Documents/KX/glasso/result/rate/{exp_name}'
# common_command = ['--exp_name',exp_name,'--data_path',data_path]

# wkdir = 'result/'+exp_name
# if not os.path.exists(wkdir):
#     os.makedirs(wkdir)

# if 'rw' in model_list:
# ##### Random walk #####
#     command = ['python','../sim_var.py','--random_walk'] + common_command
#     subprocess.run(command)

# if 'ar' in model_list:
# ##### AR1 #####
#     command = ['python','../sim_ar.py','--ar_p','1'] + common_command
#     subprocess.run(command)
# ##### AR2 #####
#     command = ['python','../sim_ar.py','--ar_p','2'] + common_command
#     subprocess.run(command)

# if 'var' in model_list:
# ##### VAR1 #####
#     command = ['python','../sim_var.py','--ar_p','1'] + common_command
#     subprocess.run(command)
# ##### VAR2 #####
#     command = ['python','../sim_var.py','--ar_p','2'] + common_command
#     subprocess.run(command)

# if 'var_lasso' in model_list:
# ##### VAR Lasso #####
#     command = ['python','../sim_basu.py','--VARpen','L1'] + common_command
#     subprocess.run(command)

# # if 'mlr_shorr' in model_list:
# # ##### MLR & SHORR #####
# #     command = ['julia','wd.jl',data_path,str(ar_p),str(r1),str(r2),str(r3),str(ttratio),exp_name]
# #     subprocess.run(command)

# if 'dfm' in model_list:
# ##### DFM #####
#     for k_factor in range(1,5):
#         command = ['python','../sim_dfm.py','--k_factor',str(r),'--factor_order',str(T0)] + common_command
#         subprocess.run(command)

# if 'di' in model_list:
# ##### Diffusion index #####
#     for k_factor in range(1,5):
#         command = ['python','../sim_di.py','--k_factor',str(r),'--ar_p',str(T0)] + common_command
#         subprocess.run(command)

# if 'varma' in model_list:
# ##### VARMA L1 ##### (~24 hr)
#     command = ['python','sim_basu.py','--VARpen','L1','--VARMApen','L1'] + common_command
#     subprocess.run(command)
# ##### VARMA Hlag ##### (~36 hr)
#     command = ['python','sim_basu.py','--VARpen','HLag','--VARMApen','HLag'] + common_command
#     subprocess.run(command)

# if 'ours' in model_list:
# ##### Ours ##### 
#     command = ['python','../train.py','--dgp','arma','--dgp_p','1','--dgp_r',str(r),'--rho',str(rho),'--n_rep','10','--T',str(T),'--s',str(s),'--T0',str(T0),'--A_init',init,'--exp_name',exp_name,'--task','rate','--stop_method','Adiff','--schedule','half','--lr','1e-3','--N',str(N),'--max_iter','100','--seed','0']
#     subprocess.run(command)

# if 'zheng' in model_list:
# ##### Ours ##### 
#     # for ar_p in [1,2,3,4]:
#     command = ['python','../sim_zheng.py','--ar_p',str(T0)] + common_command
#     subprocess.run(command)
# # print(' '.join(command))
# # subprocess.Popen(command)
