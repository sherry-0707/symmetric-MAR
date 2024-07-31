import subprocess
from subprocess import STDOUT
import os

os.environ["OMP_NUM_THREADS"] = "1"

####### tmux 0 ########
# subprocess.run(['python', '/home/r8user2/Documents/Matrix-HAR-Ito/train.py',
#                 '--dgp', 'mar',
#                 '--dgp_p', '10',
#                 '--dgp_r', '4',
#                 # '--schedule', 'half',
#                 '--schedule', 'none',
#                 '--stop_method', 'loss',
#                 '--T', '1000',
#                 '--N', '10',
#                 '--lr', '0.001',
#                 '--n_rep', '20',
#                 '--A_init', 'random',
#                 '--standardize', 'demean',
#                 '--print_log',
#                ], stderr=STDOUT)

####### tmux 1 ########
# subprocess.run(['python', '/home/r8user2/Documents/Matrix-HAR-Ito/train.py',
#                 '--dgp', 'mar',
#                 '--dgp_p', '10',
#                 '--dgp_r', '4',
#                 # '--schedule', 'half',
#                 '--schedule', 'none',
#                 '--stop_method', 'Adiff',
#                 '--T', '1000',
#                 '--N', '20',
#                 '--lr', '0.001',
#                 '--n_rep', '10',
#                 '--A_init', 'noisetrue_G',
#                 '--standardize', 'demean',
#                 '--print_log',
#                ], stderr=STDOUT)
