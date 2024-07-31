import subprocess

## default MAR setting ##
r = 4
N = 20
T = 2000
init = 'noisetrue_G'
n_rep = 50
num_proc = 10
rep_per_proc = int(n_rep / num_proc)
exp_name = f'mar_T_seed50_{init}_0.8_new'


# ## Exp 1: change r
# for exp_r in [2,3,4,5,6]:
#     for batch_no in range(num_proc):
#         command = ['python','train.py','--dgp','mar','--dgp_p','10','--dgp_r',str(exp_r),'--n_rep',str(rep_per_proc),'--T',str(T),'--N',str(N),'--A_init',init,
#                    '--exp_name',exp_name,'--task','rate','--stop_method','Adiff','--schedule','half','--lr','1e-2','--seed',str(rep_per_proc*batch_no)]
#         print(' '.join(command))
#         subprocess.Popen(command)


## Exp 2: change T
for exp_T in [500,590,710,910,1250,2000]:
    for batch_no in range(num_proc):
        command = ['python','train_new.py','--dgp','mar','--dgp_p','10','--dgp_r',str(r),'--n_rep',str(rep_per_proc),'--T',str(exp_T),'--N',str(N),'--A_init',init,
                   '--exp_name',exp_name,'--task','rate','--stop_method','Adiff','--schedule','half','--lr','1e-2','--seed',str(rep_per_proc*batch_no)]
        print(' '.join(command))
        subprocess.Popen(command)


# ## Exp 3: change N
# N_list = [6,10,13,17,21,25]
# for exp_N in [6]:
#     for batch_no in range(num_proc):
#         command = ['python','train.py','--dgp','mar','--dgp_p','10','--dgp_r',str(r),'--n_rep',str(rep_per_proc),'--T',str(T),'--N',str(exp_N),'--A_init',init,
#                    '--exp_name',exp_name,'--task','rate','--stop_method','Adiff','--schedule','half','--lr','1e-2','--seed',str(rep_per_proc*batch_no)]
#         print(' '.join(command))
#         subprocess.Popen(command)



"""
export CUDA_VISIBLE_DEVICES=""
command lines:
conda activate myenv
python batch_mar_new.py
"""