import subprocess

## default MAR setting ##
r = 4
N = 20
T = 2000
init = 'noisetrue_G'
n_rep = 300
num_proc = 20
rep_per_proc = int(n_rep / num_proc)


# ################ dgp_core = 0.7 ################
# exp_name = f'formal_N_seed500_{init}'

# ## Exp 1: change r
# for exp_r in [2,3,4,5,6]:
#     for batch_no in range(num_proc):
#         command = ['python','train.py','--dgp','mar','--dgp_p','10','--dgp_r',str(exp_r),'--n_rep',str(rep_per_proc),'--T',str(T),'--N',str(N),'--A_init',init,
#                    '--exp_name',exp_name,'--task','rate','--stop_method','Adiff','--schedule','half','--lr','1e-2','--seed',str(rep_per_proc*batch_no)]
#         print(' '.join(command))
#         subprocess.Popen(command)


# ## Exp 2: change T
# for exp_T in [500,590,710,910,1250,2000]:
#     for batch_no in range(num_proc):
#         command = ['python','train.py','--dgp','mar','--dgp_p','10','--dgp_r',str(r),'--n_rep',str(rep_per_proc),'--T',str(exp_T),'--N',str(N),'--A_init',init,
#                    '--exp_name',exp_name,'--task','rate','--stop_method','Adiff','--schedule','half','--lr','1e-2','--seed',str(rep_per_proc*batch_no)]
#         print(' '.join(command))
#         subprocess.Popen(command)


# ## Exp 3: change N
# N_list = [6,10,13,17,21,25]
# n_rep_list = [3,4,6,12,15,20]
# for exp_N in [25]:
#     for batch_no in range(num_proc):
#         command = ['python','train.py','--dgp','mar','--dgp_p','10','--dgp_r',str(r),'--n_rep',str(rep_per_proc),'--T',str(T),'--N',str(exp_N),'--A_init',init,
#                    '--exp_name',exp_name,'--task','rate','--stop_method','Adiff','--schedule','half','--lr','1e-2','--seed',str(rep_per_proc*batch_no)]
#         print(' '.join(command))
#         subprocess.Popen(command)


# ################ dgp_core = 0.8 ################
# exp_name = f'mar_T_seed200_{init}_dgp_core_0.8'

# ## Exp 2: change T
# for exp_T in [500,590,710,910,1250,2000]:
#     for batch_no in range(num_proc):
#         command = ['python','train.py','--dgp','mar','--dgp_p','10','--dgp_r',str(r),'--dgp_core','0.8','--n_rep',str(rep_per_proc),'--T',str(exp_T),'--N',str(N),'--A_init',init,
#                    '--exp_name',exp_name,'--task','rate','--stop_method','Adiff','--schedule','half','--lr','1e-2','--seed',str(rep_per_proc*batch_no)]
#         print(' '.join(command))
#         subprocess.Popen(command)


# ################ dgp_core = 0.75 ################
# exp_name = f'mar_T_seed200_{init}_dgp_core_0.75'

# ## Exp 2: change T
# T_list = [500,590,710,910,1250,2000]
# for exp_T in [910,1250,2000]:
#     for batch_no in range(num_proc):
#         command = ['python','train.py','--dgp','mar','--dgp_p','10','--dgp_r',str(r),'--dgp_core','0.75','--n_rep',str(rep_per_proc),'--T',str(exp_T),'--N',str(N),'--A_init',init,
#                    '--exp_name',exp_name,'--task','rate','--stop_method','Adiff','--schedule','half','--lr','1e-2','--seed',str(rep_per_proc*batch_no)]
#         print(' '.join(command))
#         subprocess.Popen(command)


################ dgp_core = 0.60 ################
exp_name = f'formal_T_seed300_dgp_core_0.60'

## Exp 2: change T
T_list = [500,590,710,910,1250,2000]
for exp_T in [500,590,710]:
    for batch_no in range(num_proc):
        command = ['python','train.py','--dgp','mar','--dgp_p','10','--dgp_r',str(r),'--dgp_core','0.60','--n_rep',str(rep_per_proc),'--T',str(exp_T),'--N',str(N),'--A_init',init,
                   '--exp_name',exp_name,'--task','rate','--stop_method','Adiff','--schedule','half','--lr','1e-2','--seed',str(rep_per_proc*batch_no+200)]
        print(' '.join(command))
        subprocess.Popen(command)


# ## Exp 3: change N
# N_list = [6,10,13,17,21,25]
# for exp_N in [25]:
#     for batch_no in range(num_proc):
#         command = ['python','train.py','--dgp','mar','--dgp_p','10','--dgp_r',str(r),'--dgp_core','0.60','--n_rep',str(rep_per_proc),'--T',str(T),'--N',str(exp_N),'--A_init',init,
#                    '--exp_name',exp_name,'--task','rate','--stop_method','Adiff','--schedule','half','--lr','1e-2','--seed',str(rep_per_proc*batch_no)]
#         print(' '.join(command))
#         subprocess.Popen(command)



# exp_name = f'test'

# command = ['python','train.py','--dgp','mar','--dgp_p','10','--dgp_r','2','--dgp_core','0.8','--n_rep','10','--T','2000','--N','6','--A_init',init,
#             '--exp_name',exp_name,'--task','debug','--stop_method','Adiff','--schedule','half','--lr','1e-2','--seed','0']
# print(' '.join(command))
# subprocess.run(command)

"""
export CUDA_VISIBLE_DEVICES=""
command lines:
conda activate myenv
python batch_mar.py
"""