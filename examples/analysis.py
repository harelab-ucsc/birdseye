import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap

import sys
import csv
import pdb
import pickle
import yaml
import glob2
import os

dir_name = '/home/mwmaster/ros2_ws/src/birdseye/examples'
files = glob2.glob(os.path.join(dir_name, '*10.pkl'))
files.sort()
fig, ax = plt.subplots(4, 1, figsize=(12, 12)) #, sharex=True)
bins = 250
colors = get_cmap('viridis')
# Collect handles and labels for the global legend
handles, labels = [], []

sts_collect = []
# Plotting loop
for i,file in enumerate(files):
    print(file)
    tmp = file.split(os.sep)[-1]
    c = colors(i/len(files))

    with open(file, 'rb') as f:
        data = pickle.load(f)
        mu = np.array(data['mu'])
        Sigma = np.array(data['Sigma'])
        mu0 = data['mu0']

        # Plot data on respective subplots
        h0 = ax[0].hist(mu[:, 0], bins=bins, color=c, alpha=0.3, density=True, label=f'{tmp}')
        ax[0].vlines(mu0[0], -0.01, 0, colors=c)
        h1 = ax[1].hist(mu[:, 1], bins=bins, color=c, alpha=0.3, density=True, label=f'{tmp}')
        ax[1].vlines(mu0[1], -0.01, 0, colors=c)
        h2 = ax[2].hist(mu[:, 2], bins=bins, color=c, alpha=0.3, density=True, label=f'{tmp}')
        ax[2].vlines(mu0[2], -0.1, 0, colors=c)
        h3 = ax[3].hist(mu[:, 3], bins=bins, color=c, alpha=0.3, density=True, label=f'{tmp}')
        ax[3].vlines(mu0[3], -0.1, 0, colors=c)

        # Collect handles and labels for the legend
        handles.append(h0[2][0])  # Use one element from the histogram
        labels.append(f'{tmp[:-4]}')
    sts = []
    for i in range(mu.shape[-1]):
        sts.append([np.quantile(mu[:,i], 0.025), \
               np.quantile(mu[:,i], 0.975), \
               mu[:,i].mean(),\
               np.quantile(mu[:,i], 0.5)])
        print(f'mu_{i}: [{sts[i][0]:.4f}, {sts[i][1]:.4f}]')
        print(f'      mean: {sts[i][2]:.8f}, median: {sts[i][3]:.8f}')
        print(f'      std.dev.: {mu[:,i].std():.8f}\n')

        print(f'Sigma_{i}: [{np.quantile(Sigma[:,i,i], 0.025)**0.5:.4f}, {np.quantile(Sigma[:,i,i], 0.975)**0.5:.4f}]')
        print(f'      mean: {Sigma[:,i,i].mean()**0.5:.8f}, median: {np.quantile(Sigma[:,i,i], 0.5)**0.5:.8f}\n')
    sts_collect.append(sts)

sts_collect = np.array(sts_collect)
print(f'{sts_collect.shape}\n')
# print(f'{sts_collect.min(axis=0)}, \n\n{sts_collect.max(axis=0)}')
stat = np.array([[0,np.inf,0,0]]*sts_collect.shape[1])
print()
print()
for i in range(sts_collect.shape[0]):
    print('\nframe:')
    print(sts_collect[i,:,:])
    print('\nstat:')
    print(stat)
    print(stat.shape)

    for j in range(sts_collect.shape[1]):
        print('\nframe row:')
        print(sts_collect[i,j,:])
        print('\nstat row (pre):')
        print(stat[j,:])
        if sts_collect[i,j,0] > stat[j,0]:
            stat[j,0] = sts_collect[i,j,0]
        if sts_collect[i,j,1] < stat[j,1]:
            stat[j,1] = sts_collect[i,j,1]
        stat[j,2] += sts_collect[i,j,2]
        stat[j,3] += sts_collect[i,j,3]
        print('\nstat row (post):')
        print(stat[j,:])
stat[:,2] /= sts_collect.shape[0]
stat[:,3] /= sts_collect.shape[0]
print('\nstat:')
print(stat)

# Add a single legend to the figure
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=6, frameon=False)
# Adjust layout to fit the legend
plt.tight_layout(rect=[0, 0.1, 1, 1])
plt.show()
