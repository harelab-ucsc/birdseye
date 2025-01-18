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
bproj = np.array(data['bproj'])  # back-projection of clicks (to and from pixel space)
reproj = np.array(data['reproj'])
print()
print('bproj:    ', type(bproj), len(bproj), len(bproj[0]))
# print(bproj.mean(axis=0), bproj.std(axis=0))
print()
print('reproj:    ', type(reproj), len(reproj), len(reproj[0]))
# print(reproj.mean(axis=0), reproj.std(axis=0))
print()

print('AprilTag 3D-projection accuracy (meters): \n', bproj.mean(axis=0), '+/- ', bproj.std(axis=0))
print('Click projection accuracy (pixels): \n', reproj.mean(axis=0), '+/- ', reproj.std(axis=0))
fig, ax = plt.subplots(1, 2, figsize=(12,6))

ax[0].scatter(bproj[:,0], bproj[:,1], c='g', alpha=0.1, label='Back-Projection Error')#bproj[:,2],
ax[0].scatter(bproj[:,0].mean(), bproj[:,1].mean(), c='b', marker='x', label='Mean Error')# bproj[:,2].mean,
ax[0].scatter(0.0, 0.0, c='r', s=100, marker='s', label='Truth')#0.0,
ax[0].set_xlim(-2,2)
ax[0].set_ylim(-2,2)
# ax.set_zlim(-0.5,0.5)
ax[0].set_xlabel('X (meters)', fontsize = 16)
ax[0].set_ylabel('Y (meters)', fontsize = 16)
# ax.set_zlabel('Z (meters)')

ax[0].legend(fontsize = 16)

ax[1].scatter(reproj[:,0], reproj[:,1], c='r', alpha=0.1, label='Reprojection Error')
# ax.scatter(tmp_x, tmp_y, c='r', label='reproj (c2D-aGT)')
ax[1].set_xlim(-1920/2, 1920/2)
ax[1].set_ylim(-1080/2,1080/2)
# ax[0].set_aspect('equal')
ax[1].axhline(y=0, color='k', alpha=0.3)
ax[1].axvline(x=0, color='k', alpha=0.3)
ax[1].set_aspect('equal')
ax[1].set_xlabel('X (pixels)', fontsize = 16)
# ax[0].set_ylabel('Y (pixels)')
ax[1].set_ylabel('Y (pixels)', fontsize = 16)
ax[1].scatter(0.0, 0.0, c='k', s=20, label='Origin')
#
# ax[0].legend()
ax[1].legend(fontsize = 16)

fig2, ax2 = plt.subplots(1, 2, figsize=(12,6))
ax2[0].hist(np.linalg.norm(bproj, axis=1), bins=100, density=True, color='g')
ax2[0].set_title(f'BackProj Norm: {np.linalg.norm(bproj, axis=1).mean():.4f} +/- {np.linalg.norm(bproj, axis=1).std():.4f} meters')
ax2[1].hist(np.linalg.norm(reproj, axis=1), bins=100, density=True, color='r')
ax2[1].set_title(f'ReProj Norm: {np.linalg.norm(reproj, axis=1).mean():.4f} +/- {np.linalg.norm(reproj, axis=1).std():.4f} pixels')
plt.show()
