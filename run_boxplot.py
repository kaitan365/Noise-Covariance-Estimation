import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
# set the subfolder EstimationResults as working directory.
os.chdir((dname+ '/EstimationResults')) 

import numpy as np
from scipy import linalg as la
# import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# %%
def myboxplot(S_type):
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
    # calculate Frobenius error 
    rep = 100
    EST1000 = np.load(('EST_n1000_' + S_type + '.npy'))
    Loss_fro1000 = np.zeros((4, rep))
    for i in range(4):
        for j in range(rep):
            est = EST1000[j,i,:,:]
            Loss_fro1000[i,j] = la.norm(est - S, 'fro')
    Loss_1000 = np.reshape(Loss_fro1000, (-1, 1)) # vectorize by row


    EST1500 = np.load(('EST_n1500_' + S_type + '.npy'))
    Loss_fro1500 = np.zeros((4, rep))
    for i in range(4):
        for j in range(rep):
            est = EST1500[j,i,:,:]
            Loss_fro1500[i,j] = la.norm(est - S, 'fro')
    Loss_1500 = np.reshape(Loss_fro1500, (-1, 1)) # vectorize by row

    EST2000 = np.load(('EST_n2000_' + S_type + '.npy'))
    Loss_fro2000 = np.zeros((4, rep))
    for i in range(4):
        for j in range(rep):
            est = EST2000[j,i,:,:]
            Loss_fro2000[i,j] = la.norm(est - S, 'fro')
    Loss_2000 = np.reshape(Loss_fro2000, (-1, 1)) # vectorize by row
    # construct dataframe
    a = np.concatenate((Loss_1000, Loss_1500, Loss_2000), axis =0)
    df = pd.DataFrame(a)
    b1 = ['naive']*100 + ['mm']*100 + ['proposed']*100 + ['oracle']*100
    df[1] = b1 * 3
    df[2] = np.repeat([1000,1500,2000], 400)
    df.columns = ['loss', 'method', 'sample size']
    # boxplot
    sns.set(style="darkgrid")
    sns.set_palette(sns.color_palette('muted'))
    sns_plot = sns.boxplot(x='method', y='loss', hue = 'sample size', data=df) 
    sns_plot.set(xlabel=None, ylabel ='Frobenius loss')
    plt.tight_layout()
    sns_plot.figure.savefig((('boxplot_' + S_type + '.pdf')))
    
    plt.close()
    return Loss_fro1000, Loss_fro1500, Loss_fro2000
#   the end of this function 
# %%
loss_full_1000, loss_full_1500, loss_full_2000 = myboxplot('full_rank')
print('Average and std loss for estimating full rank matrix (sample size = 1000) \n', 
      np.mean(loss_full_1000, axis = 1).round(3), np.std(loss_full_1000, axis = 1).round(3))
print('Average  and std loss for estimating full rank matrix (sample size = 1500) \n', 
      np.mean(loss_full_1500, axis = 1).round(3), np.std(loss_full_1500, axis = 1).round(3))
print('Average  and std loss for estimating full rank matrix (sample size = 2000) \n', 
      np.mean(loss_full_2000, axis = 1).round(3), np.std(loss_full_2000, axis = 1).round(3))

loss_low_1000, loss_low_1500, loss_low_2000 = myboxplot('low_rank')
print('Average and std loss for estimating low rank matrix (sample size = 1000) \n', 
      np.mean(loss_low_1000, axis = 1).round(3), np.std(loss_low_1000, axis = 1).round(3))
print('Average  and std loss for estimating low rank matrix (sample size = 1500) \n', 
      np.mean(loss_low_1500, axis = 1).round(3), np.std(loss_low_1500, axis = 1).round(3))
print('Average  and std loss for estimating low rank matrix (sample size = 2000) \n', 
      np.mean(loss_low_2000, axis = 1).round(3), np.std(loss_low_2000, axis = 1).round(3))