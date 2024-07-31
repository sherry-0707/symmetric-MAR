import os
import sys
os.environ["OMP_NUM_THREADS"] = "1"
import platform
from packaging.version import Version

import torch
import numpy as np
import configargparse as argparse
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm

from dgp_new import DGP_MAR_new
from alg import train_epoch
from utils import unfold,generate_X,f_trans
# from utils.visualize import draw_acf

def main():
    """
    command line: python train.py --dgp mar --dgp_p 10 --dgp_r 4 --schedule half --stop_method loss --T 500 --N 10 --lr 0.01 --print_log --n_rep 10 --A_init noisetrue_G
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--config',is_config_file=True)
    parser.add_argument('--exp_name',type=str,default='test')
    
    # dgp
    parser.add_argument('--dgp',type=str,choices=['mar'],default='mar')
    parser.add_argument('--dgp_p',type=int,default=1)
    parser.add_argument('--dgp_r',type=int,default=1)
    parser.add_argument('--dgp_core',type=float,default=0.4,help='g=[core]*r')

    # experiment setting
    parser.add_argument('--n_rep',type=int,default=10,help='number of replications')
    parser.add_argument('--T', type=int,default=200, help='sample size for dgp')
    # parser.add_argument('--T0', type=int,default=10, help='T0')
    parser.add_argument('--N', type=int, default=10, help='dimension for dgp')
    parser.add_argument('--seed', type=int,default=0)

    # hyperparameter for training
    parser.add_argument('--a',type=float,default=1,help='hyperparameter in gradient descent')
    parser.add_argument('--b',type=float,default=1,help='hyperparameter in gradient descent')
    parser.add_argument('--lr',type=float,default=1e-3,help='step size')
    parser.add_argument('--stop_thres',type=float,default=1e-5,help='stopping threshold for the different of A in each iteration')
    parser.add_argument('--max_iter',type=int,default=10000)
    parser.add_argument('--stop_method',type=str,choices=['none','loss','Adiff'],default='loss',
                        help='none: no early stop; loss: loss does not decrease for some iterations; Adiff: diff of |A_t - A_{t-1}|_F less than some threshold')
    parser.add_argument('--schedule',type=str,choices=['none','half'],default='none')

    # task related
    parser.add_argument('--A_init',type=str,choices=['true','random','GD','noisetrue_A','noisetrue_G','noisetrue_U','noisezero'],default='true')
    parser.add_argument('--task',type=str,choices=['rate','convergence','hyperparameter','debug'],default='debug')

    # debug and visualize
    parser.add_argument('--visualize',action ='store_true')
    parser.add_argument('--print_log',action='store_true')


    args = parser.parse_args()
    # check and set other parameters based on args
    if args.visualize:
        args.stop_method = 'none'
    if args.task == 'convergence':
        args.max_iter = 10000; args.stop_method = 'none'

    # logging file
    dir_path = os.path.dirname(os.path.realpath(__file__))
    wkdir = f'{dir_path}/result/{args.task}/{args.exp_name}'
    if not os.path.exists(wkdir):
        os.makedirs(f'{wkdir}/log')
        os.makedirs(f'{wkdir}/csv')
        os.makedirs(f'{wkdir}/fig')
    
    filename = f'{wkdir}/log/N{args.N}_p{args.dgp_p}_r{args.dgp_r}_T{args.T}_init{args.A_init}.log'
    # filename = f'{wkdir}/log/p{args.dgp_p}r{args.dgp_r}T{args.T}_init{args.A_init}_schedule{args.schedule}_rank{args.dgp_r}_N{args.N}.log'
    print("make file")
    
    if Version(platform.python_version()) >= Version('3.9'):
        logging.basicConfig(filename=filename, encoding='utf-8', level=logging.INFO,
                            format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
    else:
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler(filename, 'a', 'utf-8')
        root_logger.addHandler(handler)

    logging.critical(f"""DGP setting:--dgp {args.dgp} --dgp_p {args.dgp_p} --dgp_r {args.dgp_r} --T {args.T} --N {args.N} \nExperiment setting:--n_rep {args.n_rep} --a {args.a} --b {args.b} --lr {args.lr} --A_init {args.A_init} --stop_method {args.stop_method} --schedule {args.schedule} --seed {args.seed}""")

    est_err = torch.zeros(args.n_rep)
    loss_list = torch.zeros(args.n_rep)
    mean_log_est_path = torch.zeros(args.max_iter)
    for rep in tqdm(range(args.n_rep)):
        torch.manual_seed(rep + args.seed) # set seed for CPU, if use GPU, torch.cuda.manual_seed(args.seed)
        y,true_A,true_G,true_U1,true_U2,true_U3,f_A = DGP_MAR_new(args.N,args.T,burn=600,P=args.dgp_p,g=torch.tensor([args.dgp_core]*args.dgp_r))
        # import pdb; pdb.set_trace()
        

        y_hat = torch.einsum('TNP,iNP->Ti',generate_X(y,args.dgp_p),f_A)
        true_loss = torch.sum(torch.norm(y_hat-y[args.dgp_p:,:],p=2,dim=1))/args.T
            
        if rep == 0:
            # A_fro_norm_list = torch.norm(true_A,p='fro',dim=(0, 1))
            logging.info('true A fro norm^2: {}, true f(A) fro norm^2: {}, true loss: {}'.format(torch.norm(unfold(true_A,1),p='fro')**2, torch.linalg.norm(unfold(f_A,1),ord='fro')**2, true_loss))
            if not os.path.exists('temp'):
                os.makedirs('temp')
            y_np = y[args.dgp_p:,:].numpy()
            np.savetxt('temp/y.csv', y_np, delimiter=',')
            
        A,loss,err_path,est_path = train_epoch(y=y,P=args.dgp_p,r1=args.dgp_r,r2=args.dgp_r,r3=args.dgp_r,a=args.a,b=args.b,stop_method=args.stop_method,
                                               schedule=args.schedule,max_iter=args.max_iter,stop_thres=args.stop_thres,step_size=args.lr,init_step_size=5e-3,
                                               init_iter=1000,true_G=true_G,true_U1=true_U1,true_U2=true_U2,true_U3=true_U3,A_init_method=args.A_init,print_log=args.print_log)
        
       
        # import pdb; pdb.set_trace()
        if loss is None:
            continue

        est_err[rep] = torch.norm(unfold(A-true_A,1),p='fro')**2
        loss_list[rep] = loss
        mean_log_est_path += torch.log(est_path/est_path[0])
        # print(A-true_A[:,:,:P])
        logging.info('replication no.{}: estimation error {} | f_trans error {}'.format(rep,est_err[rep],torch.norm(unfold(f_trans(A)-f_trans(true_A),1), p='fro')**2))
        

    mean_log_est_path /= args.n_rep
    if args.task == 'convergence':
        # plt.figure()
        # plt.plot(mean_err_path)
        # plt.savefig('{}/fig/p{}r{}T{}_cvg.png'.format(wkdir,args.dgp_p,args.dgp_r,args.T))
        plt.figure()
        plt.plot(mean_log_est_path)
        plt.savefig('{}/fig/p{}r{}T{}N{}_cvg_est.png'.format(wkdir,args.dgp_p,args.dgp_r,args.T,args.N))
        

    logging.critical("End replication, mean est err^2: {}, mean_loss: {}\n".format(torch.mean(est_err),torch.mean(loss_list)))

    # if args.task == 'rate':
    #     with open('{}/csv/p{}r{}N{}.csv'.format(wkdir,args.dgp_p,args.dgp_r,args.N),'a') as f:
    #         f.write(f'{args.n_rep},T{args.T}_N{args.N}_A_init_{args.A_init},{torch.mean(est_err)},{torch.mean(est_err)}\n')
    if args.task == 'convergence':
        with open('{}/csv/p{}r{}T{}N{}.csv'.format(wkdir,args.dgp_p,args.dgp_r,args.T,args.N),'a') as f:
            f.write(f'{args.n_rep},N{args.N}_A_init_{args.A_init},{mean_log_est_path}\n')


if __name__ == "__main__":
   main()
