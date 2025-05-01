#!/usr/bin/env python3

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from SLICAnnotator import offlineSLICAnnotator
import glob2
from dbConnector import dbConnector
from utilities import *
from AMI_ContourClassFamily import Contour
from birdsEye import birdsEye


# file_root = '/home/mwmaster/parsed_flights/2025_01_17/'
file_root = '/home/mwmaster/parsed_flights/2025_01_21/'
# file_root = '/home/mwmaster/parsed_flights/2025_01_20/'
# datasets = glob2.glob(os.path.join(file_root, 'acceptance_0*_rect/out_dict.pkl'))
datasets = glob2.glob(os.path.join(file_root, 'acceptance_01*_rect/out_dict.pkl'))
# datasets = glob2.glob(os.path.join(file_root, 'acceptance_02*_rect/out_dict.pkl'))
datasets += glob2.glob(os.path.join(file_root, 'acceptance_05*_rect/out_dict.pkl'))
datasets += glob2.glob(os.path.join(file_root, 'acceptance_06*_rect/out_dict.pkl'))

print(np.array(datasets))
data_raw = {}
data_rect = {}
for file in datasets:
    with open(file, 'rb') as f:
        tmp = pickle.load(f)
        if '_rect' in file:
            for key in tmp.keys():
                # print('rect: ', key)
                try:
                    data_rect[key] += tmp[key]
                except KeyError:
                    data_rect[key] = tmp[key]
        else:
            for key in tmp.keys():
                # print('raw: ', key)
                try:
                    data_raw[key] += tmp[key]
                except KeyError:
                    data_raw[key] = tmp[key]


fig, ax = plt.subplots(1, 2, figsize=(12,6))
ax[0].set_xlim(-2,2)
ax[0].set_ylim(-2,2)
ax[0].set_aspect('equal')
ax[0].set_xlabel('X (meters)', fontsize = 16)
ax[0].set_ylabel('Y (meters)', fontsize = 16)
ax[0].scatter(0.0, 0.0, c='r', s=100, marker='s', label='Truth')#0.0,

ax[1].set_xlim(-1920/2, 1920/2)
ax[1].set_ylim(-1080/2,1080/2)
ax[1].axhline(y=0, color='k', alpha=0.3)
ax[1].axvline(x=0, color='k', alpha=0.3)
ax[1].set_aspect('equal')
ax[1].set_xlabel('X (pixels)', fontsize = 16)
ax[1].set_ylabel('Y (pixels)', fontsize = 16)
ax[1].scatter(0.0, 0.0, c='k', s=20, label='Origin')

fig2, ax2 = plt.subplots(1, 2, figsize=(12,6))

for i, data in enumerate([data_raw, data_rect]):
    print(data.keys())  # dict_keys(['april_3D', 'bproj', 'clicks'])
    if len(data.keys()) == 0:
        continue
    if i == 0:
        colors = ['g', 'r', 'b']
        marker = '+'
        tmp = 'Raw'
    elif i == 1:
        colors = ['m', 'c', 'y']
        marker = 'x'
        tmp = 'Rect'
    print(data.keys())  # dict_keys(['april_3D', 'bproj', 'clicks'])
    bproj = np.array(data['bproj'])  # back-projection of clicks (to and from pixel space)
    reproj = np.array(data['reproj'])
    print('bproj:    ', type(bproj), len(bproj), len(bproj[0]))
    print('reproj:    ', type(reproj), len(reproj), len(reproj[0]))
    print()

    print('AprilTag 3D-projection accuracy (meters): \n', bproj.mean(axis=0), '+/- ', bproj.std(axis=0))
    print('Click projection accuracy (pixels): \n', reproj.mean(axis=0), '+/- ', reproj.std(axis=0))

    ax[0].scatter(bproj[:,0], bproj[:,1], c=colors[0], alpha=0.1, label=tmp+'BackProj')#bproj[:,2],
    ax[0].scatter(bproj[:,0].mean(), bproj[:,1].mean(), c=colors[2], marker=marker, label=tmp+'Mean Error')# bproj[:,2].mean,
    ax[1].scatter(reproj[:,0], reproj[:,1], c=colors[1], alpha=0.1, label=tmp+'Reproj')
    ax[0].legend(fontsize = 12)
    ax[1].legend(fontsize = 12)

    lbl = tmp+f'BackProj Norm: {np.linalg.norm(bproj, axis=1).mean():.3f} +/- {np.linalg.norm(bproj, axis=1).std():.3f} m'
    ax2[0].hist(np.linalg.norm(bproj, axis=1), bins=100, density=True, color=colors[0], label=lbl)
    lbl = tmp+f'ReProj Norm: {np.linalg.norm(reproj, axis=1).mean():.1f} +/- {np.linalg.norm(reproj, axis=1).std():.1f} px'
    ax2[1].hist(np.linalg.norm(reproj, axis=1), bins=100, density=True, color=colors[1], label=lbl)
    ax2[0].legend(fontsize=12)
    ax2[1].legend(fontsize=12)
    print()
plt.show()
