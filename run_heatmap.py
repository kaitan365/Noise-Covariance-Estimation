import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
# set the subfolder EstimationResults as working directory.
os.chdir((dname+ '/EstimationResults')) 

import numpy as np
# from scipy import linalg as la
# import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
# import pandas as pd
# %% 
# n = 1000 # 1500, 2000
# S_type = 'full_rank' # 'low_rank'

# generate heatmap for a given n and S_type
def myheatmap(n, S_type):
    T = 20
    S  = np.zeros((T, T)) 
    if (S_type == 'full_rank'):
        for t in range(T):
            for tt in range(T):
                S[t, tt] = np.cos(t-tt) / (1+np.sqrt(abs(t-tt)))
    if (S_type == 'low_rank'):
        rng = np.random.RandomState(1234)
        V = rng.randn(T, 10)/np.sqrt(10)
        S = V @ V.T 
    # load estimation results
    EST = np.load(('EST_n'+str(n)+'_'+ S_type + '.npy'))
    EST_mean = np.mean(EST, axis=0) # bias
    EST_std = np.std(EST, axis=0) # standard deviation
    # heatmap for bias
    data = list(EST_mean - S) # bias
    fig, ax = plt.subplots(2, 2, figsize=(11,10))
    ax = ax.ravel()
    
    for i, d in enumerate(data):
        #print(i,d)
        sns.heatmap(d, 
                    xticklabels=False, 
                    yticklabels=False,  
                    vmin=-.3, vmax=.3,
                    center =0,
                    ax = ax[i],
                    cmap = 'bwr')
        # ax.set_title('some title', fontsize=15)
    ax[0].set_title('naive', fontsize=18)
    ax[1].set_title('mm', fontsize=18)
    ax[2].set_title('proposed', fontsize=18)
    ax[3].set_title('oracle', fontsize=18)

    plt.tight_layout()
    plt.savefig(f'heatmap_n{n}_{S_type}_bias.pdf')
    # heatmap for standard deviation
    data = list(EST_std) 
    fig, ax = plt.subplots(2, 2, figsize=(11,10))
    ax = ax.ravel()
    for i, d in enumerate(data):
        #print(i,d)
        sns.heatmap(d, 
                    xticklabels=False, 
                    yticklabels=False,  
                    vmin=0, vmax=0.3,
                    ax = ax[i],
                    cmap = 'OrRd')
    ax[0].set_title('naive',fontsize=18)
    ax[1].set_title('mm',fontsize=18)
    ax[2].set_title('proposed',fontsize=18)
    ax[3].set_title('oracle',fontsize=18)

    plt.tight_layout()
    plt.savefig(f'heatmap_n{n}_{S_type}_std.pdf')
    plt.close()
#   the end of this function
# %% 
for n in [1000, 1500, 2000]:
    for S_type in ['full_rank', 'low_rank']:
        myheatmap(n, S_type)
