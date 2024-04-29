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


filename = '/home/mwmaster/parsed_flight/out_dict.pkl'

with open(filename, 'rb') as f:
    data = pickle.load(f)

print(data.keys())  # dict_keys(['april_3D', 'bproj', 'clicks'])
# print()

clicks = data['clicks'].tolist()
clicks = np.array([i + [0.0, 1.0] for i in clicks])  # 3D ground truth
april_2D = data['april_2D']
april_3D = data['april_3D']  # 3D visual detection projection
bproj = data['bproj']  # back-projection of clicks (to and from pixel space)
reproj = data['reproj']

# print('clicks:   ', type(clicks), len(clicks), len(clicks[0]))
# print('april_2D: ', type(april_2D), len(april_2D), len(april_2D[0]))
# print('april_3D: ', type(april_3D), len(april_3D), len(april_3D[0]))
# print('bproj:    ', type(bproj), len(bproj), len(bproj[0]))
# print()

res_apr = []
res_bpj = []
res_rpj = []
for click in clicks:
    tmp = []
    for det in april_3D:
        val = det - click
        tmp.append(val.tolist())
    res_apr.append(tmp)
    tmp = []
    for det in bproj:
        val = det - click
        tmp.append(val.tolist())
    res_bpj.append(tmp)
res_apr = np.array(res_apr)
res_bpj = np.array(res_bpj)
tmp_apr = np.linalg.norm(res_apr, axis=2)
tmp_bpj = np.linalg.norm(res_bpj, axis=2)
tmp_apr = np.where(tmp_apr<2.5, tmp_apr, 0)
tmp_bpj = np.where(tmp_bpj<2.5, tmp_bpj, 0)
masks_apr = np.nonzero(tmp_apr)
masks_bpj = np.nonzero(tmp_bpj)

# print('ping')
# print(masks_apr)
# print(masks_bpj)

# fig, ax = plt.subplots(3, 1)
fig, ax = plt.subplots(3, 3, figsize=(10,8))
plt.tight_layout()
valid_apr = []
valid_bpj = []
for i in range(tmp_apr.shape[0]):
    # print(i, '\n')
    if i:
        tmp = np.where(masks_apr[0] == i, masks_apr[0], 0)
    else:
        tmp = np.where(masks_apr[0] == i, masks_apr[0]+1, 0)
    tmp = np.nonzero(tmp)
    tmp = (masks_apr[0][tmp], masks_apr[1][tmp])
    # print(tmp, '\n')
    ax[i, 0].hist(tmp_apr[tmp], alpha=0.5, color='g', label=f'click_{i}')
    ax[i, 0].legend()
    ax[i, 2].hist(tmp_apr[tmp], color='g', alpha=0.5, label=f'april_click_{i}')
    valid_apr.append(res_apr[tmp].tolist())
    if i:
        tmp = np.where(masks_bpj[0] == i, masks_bpj[0], 0)
    else:
        tmp = np.where(masks_bpj[0] == i, masks_bpj[0]+1, 0)
    tmp = np.nonzero(tmp)
    tmp = (masks_bpj[0][tmp], masks_bpj[1][tmp])
    # print(tmp, '\n')
    ax[i, 1].hist(tmp_bpj[tmp], alpha=0.5, color='b', label=f'click_{i}')
    ax[i, 1].legend()
    ax[i, 2].hist(tmp_bpj[tmp], color='b', alpha=0.5, label=f'bproj_click_{i}')
    valid_bpj.append(res_bpj[tmp].tolist())
    ax[i, 2].legend()

ax[0,0].set_title('april_3D')
ax[0,1].set_title('bproj')
plt.show()

# print(len(valid_apr[0]))
# print(len(valid_apr[1]))
# print(len(valid_apr[2]))
# print()
# print(len(valid_bpj[0]))
# print(len(valid_bpj[1]))
# print(len(valid_bpj[2]))
# print()

apr = []
for val in valid_apr:
    apr += val
apr = np.array(apr)
bpj = []
for val in valid_bpj:
    bpj += val
bpj = np.array(bpj)

print('AprilTag 3D-projection accuracy (meters): \n', apr[:,:-1].mean(axis=0), '+/- ', apr[:,:-1].std(axis=0))
print('Click back-projection accuracy (meters): \n', bpj[:,:-1].mean(axis=0), '+/- ', bpj[:,:-1].std(axis=0))
fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(projection='3d')
ax.scatter(apr[:,0], apr[:,1], apr[:,2], c='g', label='april_3D')
ax.scatter(bpj[:,0], bpj[:,1], bpj[:,2], c='b', label='bproj')
ax.scatter(0.0, 0.0, 0.0, c='r', s=100, marker='s', label='truth')
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
ax.set_zlim(-0.5,0.5)
ax.set_xlabel('X (meters)')
ax.set_ylabel('Y (meters)')
ax.set_zlabel('Z (meters)')

ax.legend()
plt.show()


april_GT = reproj[:,0]
click_2D = reproj[:,1]
# print(april_GT.shape)
# print(click_2D.shape)
fig, ax = plt.subplots(2,1, sharex=False, figsize=(10,10))
ax[0].scatter(april_GT[:,0], april_GT[:,1], c='g', label='april_GT')
ax[0].scatter(click_2D[:,0], click_2D[:,1], c='b', label='click_2D')
for i in range(april_GT.shape[0]):
    # print(f'{i} of {april_GT.shape[0]}')
    ax[0].plot([april_GT[i,0], click_2D[i,0]],[april_GT[i,1], click_2D[i,1]],'k', alpha=0.3)

tmp_x = click_2D[:,0] - april_GT[:,0]
tmp_y = click_2D[:,1] - april_GT[:,1]
tmp = np.linalg.norm(click_2D - april_GT, axis=1)
print(f'Click Projection Error (pixels):\n {tmp.mean():.4f} +/- {tmp.std():.4f}')
ax[0].set_title(f'reproj: {tmp.mean():.4f} +/- {tmp.std():.4f}\n[min, max]: [{tmp.min():.4f}, {tmp.max():.4f}]')
ax[1].scatter(tmp_x, tmp_y, c='r', label='reproj (c2D-aGT)')
ax[0].set_xlim(0,1920)
ax[0].set_ylim(0,1080)
ax[0].set_aspect('equal')
ax[1].axhline(y=0, color='k', alpha=0.3)
ax[1].axvline(x=0, color='k', alpha=0.3)
ax[1].set_aspect('equal')
ax[1].set_xlabel('X (pixels)')
ax[0].set_ylabel('Y (pixels)')
ax[1].set_ylabel('Y (pixels)')
ax[1].scatter(0.0, 0.0, c='k', s=20, label='origin')

ax[0].legend()
ax[1].legend()
plt.show()
