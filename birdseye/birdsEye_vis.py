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


def skip_check(file, skip_patterns):
<<<<<<< Updated upstream
    ret = True
    for pattern in skip_patterns:
        if pattern in file:
            ret = False
    return ret
=======
    for pattern in skip_patterns:
        if pattern in file:
            return False
    return True
>>>>>>> Stashed changes


file_roots = [
    # '/home/mwmaster/parsed_flights/2025_03_03/', \
    # '/home/mwmaster/parsed_flights/2025_03_10/', \
    '/home/mwmaster/parsed_flights/2025_03_18/'
]

skip_patterns = ['_20m_']

datasets = []
for file_root in file_roots:
    datasets += glob2.glob(os.path.join(file_root, 'acceptance_0*_rect/out_dict.pkl'))
print(np.array(datasets))

skip_patterns = ['20m', 'acceptance_01_10m_01']

data_06 = {}
data_05 = {}
data_02 = {}
data_01 = {}
for file in datasets:
    ret = skip_check(file, skip_patterns)
<<<<<<< Updated upstream
    if ret:
        with open(file, 'rb') as f:
            tmp = pickle.load(f)
            if 'acceptance_01' in file:
                for key in tmp.keys():
                    # print('rect: ', key)
                    try:
                        data_01[key] += tmp[key]
                    except KeyError:
                        try:
                            data_01[key] = tmp[key].tolist()
                        except:
                            data_01[key] = tmp[key]
                    except ValueError:
                        data_01[key] += tmp[key].tolist()
            elif 'acceptance_02' in file:
                for key in tmp.keys():
=======
    if ret:# and '_02_rect' in file:
        pass
    else:
        continue
    with open(file, 'rb') as f:
        tmp = pickle.load(f)
        if 'acceptance_01' in file:

            for key in tmp.keys():
                # print('rect: ', key)
                try:
                    data_01[key] += tmp[key]
                except KeyError:
                    try:
                        data_01[key] = tmp[key].tolist()
                    except:
                        data_01[key] = tmp[key]
                except ValueError:
                    # print(data_01[key].shape)
                    # print(tmp[key].shape)
                    data_01[key] += tmp[key].tolist()
        elif 'acceptance_02' in file:
            for key in tmp.keys():
                # print('raw: ', key)
                try:
                    data_02[key] += tmp[key]
                except KeyError:
>>>>>>> Stashed changes
                    try:
                        data_02[key] += tmp[key]
                    except KeyError:
                        try:
                            data_02[key] = tmp[key].tolist()
                        except:
                            data_02[key] = tmp[key]
                    except ValueError:
                        data_02[key] += tmp[key].tolist()
            elif 'acceptance_05' in file:
                for key in tmp.keys():
                    try:
                        data_05[key] += tmp[key]
                    except KeyError:
                        try:
                            data_05[key] = tmp[key].tolist()
                        except:
                            data_05[key] = tmp[key]
                    except ValueError:
                        data_05[key] += tmp[key].tolist()
            elif 'acceptance_06' in file:
                for key in tmp.keys():
                    try:
                        data_06[key] += tmp[key]
                    except KeyError:
                        try:
                            data_06[key] = tmp[key].tolist()
                        except:
                            data_06[key] = tmp[key]
                    except ValueError:
                        data_06[key] += tmp[key].tolist()

fig, ax = plt.subplots(1, 2, figsize=(12,6))
ax[0].set_xlim(-2,2)
ax[0].set_ylim(-2,2)
ax[0].set_aspect('equal')
ax[0].set_xlabel('X (meters)', fontsize = 16)
ax[0].set_ylabel('Y (meters)', fontsize = 16)
ax[0].scatter(0.0, 0.0, c='k', s=100, marker='s', label='Truth')#0.0,

ax[1].set_xlim(-1920/2, 1920/2)
ax[1].set_ylim(-1080/2,1080/2)
ax[1].axhline(y=0, color='k', alpha=0.3)
ax[1].axvline(x=0, color='k', alpha=0.3)
ax[1].set_aspect('equal')
ax[1].set_xlabel('X (pixels)', fontsize = 16)
ax[1].set_ylabel('Y (pixels)', fontsize = 16)
ax[1].scatter(0.0, 0.0, c='k', s=20, label='Origin')

fig2, ax2 = plt.subplots(1, 2, figsize=(12,6))
# fig3, ax3 = plt.subplots(1, 3, figsize=(18,6))

for i, data in enumerate([data_01, data_02, data_05, data_06]):
    if len(data.keys()) == 0:
        continue
    if i == 0:
        colors = ['xkcd:red', 'xkcd:magenta', 'xkcd:salmon']
        marker = '+'
        tmp = '01'
    elif i == 1:
        colors = ['xkcd:green', 'xkcd:lime green', 'xkcd:sea green']
        marker = 'x'
        tmp = '02'
    elif i == 2:
        colors = ['xkcd:purple', 'xkcd:lavender', 'xkcd:dark purple']
        marker = 'x'
        tmp = '05'
    elif i == 3:
        colors = ['xkcd:blue', 'xkcd:cyan', 'xkcd:turquoise']
        marker = 'x'
        tmp = '06'
    # print(data.keys())  # dict_keys(['april_3D', 'bproj', 'clicks'])
    print()
    bproj = np.array(data['bproj'])  # back-projection of clicks (to and from pixel space)
    reproj = np.array(data['reproj'])
    print('bproj:    ', type(bproj), len(bproj), len(bproj[0]))
    print('reproj:    ', type(reproj), len(reproj), len(reproj[0]))
    print()
    gt_c_bproj = np.array(data['gt_c_bproj'])
    gt_a_bproj = np.array(data['gt_a_bproj'])
    gt_a_reproj = np.array(data['gt_a_reproj'])

    print('AprilTag 3D-projection accuracy (wrt clicks_3D, meters): \n', bproj.mean(axis=0), '+/- ', bproj.std(axis=0))
    print('AprilTag 3D-projection accuracy (wrt clicks_gt, meters): \n', gt_a_bproj.mean(axis=0), '+/- ', gt_a_bproj.std(axis=0))
    print('Click projection accuracy (pixels): \n', reproj.mean(axis=0), '+/- ', reproj.std(axis=0))

    base = 1

    ax[0].scatter(bproj[:,0], bproj[:,1], c=colors[0], s=6*base, alpha=0.1, label=tmp+'BackProj')#bproj[:,2],
    ax[0].scatter(bproj[:,0].mean(), bproj[:,1].mean(), c=colors[0], marker=marker, label=tmp+'Mean Error')# bproj[:,2].mean,
    # ax[0].scatter(gt_c_bproj[:,0], gt_c_bproj[:,1], c=colors[1], s=4*base, alpha=0.1, label=tmp+'gtClickBackProj')
    ax[0].scatter(gt_a_bproj[:,0].mean(), gt_a_bproj[:,1].mean(), c=colors[0], marker=marker, label=tmp+'Mean Error')# bproj[:,2].mean,
    ax[0].scatter(gt_a_bproj[:,0], gt_a_bproj[:,1], c=colors[2], s=2*base, alpha=0.1, label=tmp+'gtAprilBackProj')
    ax[0].legend(fontsize = 12)

    ax[1].scatter(reproj[:,0], reproj[:,1], s=6*base, c=colors[0], alpha=0.1, label=tmp+'Reproj')
    # ax[1].scatter(gt_a_reproj[:,0], gt_a_reproj[:,1], c=colors[1], s=4*base, alpha=0.1, label=tmp+'aprilReproj')
    ax[1].legend(fontsize = 12)

    lbl = tmp+f'BackProj Norm: {np.linalg.norm(bproj, axis=1).mean():.3f} +/- {np.linalg.norm(bproj, axis=1).std():.3f} m'
    ax2[0].hist(np.linalg.norm(bproj, axis=1), bins=100, density=True, color=colors[0], label=lbl)
    lbl = tmp+f'ReProj Norm: {np.linalg.norm(reproj, axis=1).mean():.1f} +/- {np.linalg.norm(reproj, axis=1).std():.1f} px'
    ax2[1].hist(np.linalg.norm(reproj, axis=1), bins=100, density=True, color=colors[1], label=lbl)
    ax2[0].legend(fontsize=12)
    ax2[1].legend(fontsize=12)
    print()

    # ax3[0].plot(gt_c_bproj[:,0], gt_c_bproj[:,1], label='click_bproj')
    # ax3[0].plot(gt_a_bproj[:,0], gt_a_bproj[:,1], label='april_bproj')
    # ax3[1].plot(gt_c_bproj[:,0], label='click_bproj')
    # ax3[1].plot(gt_a_bproj[:,0], label='april_bproj')
    # ax3[1].plot(gt_c_bproj[:,1], label='click_bproj')
    # ax3[2].plot(gt_a_bproj[:,1], label='april_bproj')
    # ax3[2].legend(fontsize=12)

plt.show()
print()
