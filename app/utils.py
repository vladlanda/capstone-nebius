import numpy as np
import os,sys
# Get the absolute path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the parent directory to sys.path
sys.path.append(parent_dir)

import config as conf

# RANDOM_SEED = 42

def set_seed():
    np.random.seed(conf.RANDOM_SEED)

if __name__ == '__main__':
    pass