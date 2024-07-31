import subprocess
import configargparse as argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dgp',type=str,choices=['mar'],default='mar')
parser.add_argument('--dgp_p',type=int,default=1)
parser.add_argument('--dgp_r',type=int,default=1)
parser.add_argument('--dgp_g',type=float,nargs='+',default=[0.7,0.7,0.7,0.7])

parser.add_argument('--N',type=int,default=10)
parser.add_argument('--task',type=str,choices=['rate','convergence'],required=True)
parser.add_argument('--changevar',type=str,choices=['T','rank','N'],default='T')
parser.add_argument('--seed',type=int)
args = parser.parse_args()


if args.changevar == 'T':
    T_list = [500,1000,1500,2000]
    r_list = [4]*4
    N_list = [20]*4
elif args.changevar == 'rank':
    T_list = [1000]*4
    r_list = [2,3,4,5]
    N_list = [20]*4
elif args.changevar == 'N':
    T_list = [2000]*4
    r_list = [4]*4
    N_list = [10,20,30,40]


# rate
if args.task == 'rate':
    exp_name = f'{args.dgp}_{args.changevar}_seed600'

    for T,r,N in zip(T_list,r_list,N_list):
        for init in ['noisetrue_G']:
            command = ['python','train.py','--dgp',args.dgp,'--dgp_p',str(args.dgp_p),'--dgp_r',str(r),'--n_rep','50','--T',str(T),'--N',str(N),'--A_init',init,
                       '--exp_name',exp_name,'--task',args.task,'--stop_method','loss','--schedule','half','--lr','1e-2','--seed','500']
            print(' '.join(command))
            subprocess.Popen(command,stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)







"""
command line: python batch_rate.py --dgp mar --dgp_p 10 --dgp_r 4 --N 10 --task rate --changevar T --seed 600
"""