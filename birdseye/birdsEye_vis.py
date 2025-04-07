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
    for pattern in skip_patterns:
        if pattern in file:
            return False
    return True

file_roots = [
    # '/home/mwmaster/parsed_flights/2025_03_03/', \
    # '/home/mwmaster/parsed_flights/2025_03_10/', \
    '/home/mwmaster/parsed_flights/2025_03_18/', \
    '/home/mwmaster/parsed_flights/2025_04_03/'
]

skip_patterns = []

datasets = []
for file_root in file_roots:
    datasets += glob2.glob(os.path.join(file_root, 'acceptance_0*_rect/out_dict.pkl'))
print(np.array(datasets))

data_06_10m = {}
data_05_10m = {}
data_02_10m = {}
data_01_10m = {}
data_06_20m = {}
data_05_20m = {}
data_02_20m = {}
data_01_20m = {}
for file in datasets:
    ret = skip_check(file, skip_patterns)
    if ret:# and '_02_rect' in file:
        pass
    else:
        continue
    with open(file, 'rb') as f:
        tmp = pickle.load(f)
        if '2025_04_03/acceptance_01_10m' in file:
            for key in tmp.keys():
                # print('rect: ', key)
                try:
                    data_01_10m[key] += tmp[key]
                except KeyError:
                    try:
                        data_01_10m[key] = tmp[key].tolist()
                    except:
                        data_01_10m[key] = tmp[key]
                except ValueError:
                    # print(data_01[key].shape)
                    # print(tmp[key].shape)
                    data_01_10m[key] += tmp[key].tolist()
        elif 'acceptance_02_10m' in file:
            for key in tmp.keys():
                # print('raw: ', key)
                try:
                    data_02_10m[key] += tmp[key]
                except KeyError:
                    try:
                        data_02_10m[key] += tmp[key]
                    except KeyError:
                        try:
                            data_02_10m[key] = tmp[key].tolist()
                        except:
                            data_02_10m[key] = tmp[key]
                    except ValueError:
                        data_02_10m[key] += tmp[key].tolist()
        elif 'acceptance_05_10m' in file:
            for key in tmp.keys():
                try:
                    data_05_10m[key] += tmp[key]
                except KeyError:
                    try:
                        data_05_10m[key] = tmp[key].tolist()
                    except:
                        data_05_10m[key] = tmp[key]
                except ValueError:
                    data_05_10m[key] += tmp[key].tolist()
        elif 'acceptance_06_10m' in file:
            for key in tmp.keys():
                try:
                    data_06_10m[key] += tmp[key]
                except KeyError:
                    try:
                        data_06_10m[key] = tmp[key].tolist()
                    except:
                        data_06_10m[key] = tmp[key]
                except ValueError:
                    data_06_10m[key] += tmp[key].tolist()
        elif '2025_04_03/acceptance_01_20m' in file:
            for key in tmp.keys():
                # print('rect: ', key)
                try:
                    data_01_20m[key] += tmp[key]
                except KeyError:
                    try:
                        data_01_20m[key] = tmp[key].tolist()
                    except:
                        data_01_20m[key] = tmp[key]
                except ValueError:
                    # print(data_01[key].shape)
                    # print(tmp[key].shape)
                    data_01_20m[key] += tmp[key].tolist()
        elif 'acceptance_02_20m' in file:
            for key in tmp.keys():
                # print('raw: ', key)
                try:
                    data_02_20m[key] += tmp[key]
                except KeyError:
                    try:
                        data_02_20m[key] += tmp[key]
                    except KeyError:
                        try:
                            data_02_20m[key] = tmp[key].tolist()
                        except:
                            data_02_20m[key] = tmp[key]
                    except ValueError:
                        data_02_20m[key] += tmp[key].tolist()
        elif 'acceptance_05_20m' in file:
            for key in tmp.keys():
                try:
                    data_05_20m[key] += tmp[key]
                except KeyError:
                    try:
                        data_05_20m[key] = tmp[key].tolist()
                    except:
                        data_05_20m[key] = tmp[key]
                except ValueError:
                    data_05_20m[key] += tmp[key].tolist()
        elif 'acceptance_06_20m' in file:
            for key in tmp.keys():
                try:
                    data_06_20m[key] += tmp[key]
                except KeyError:
                    try:
                        data_06_20m[key] = tmp[key].tolist()
                    except:
                        data_06_20m[key] = tmp[key]
                except ValueError:
                    data_06_20m[key] += tmp[key].tolist()

fig0, ax0 = plt.subplots(1, 1, figsize=(8,8))
fig1, ax1 = plt.subplots(1, 1, figsize=(9.6,6))

ax0.set_xlim(-2,2)
ax0.set_ylim(-2,2)
ax0.set_aspect('equal')
ax0.set_xlabel('X (meters)', fontsize = 16)
ax0.set_ylabel('Y (meters)', fontsize = 16)
ax1.set_xlim(-1920/2, 1920/2)
ax1.set_ylim(-1200/2, 1200/2)
ax1.axhline(y=0, color='k', alpha=0.3)
ax1.axvline(x=0, color='k', alpha=0.3)
ax1.set_aspect('equal')
ax1.set_xlabel('X (pixels)', fontsize = 16)
ax1.set_ylabel('Y (pixels)', fontsize = 16)

inset_ax0 = ax0.inset_axes([0.08, 0.04, 0.35, 0.35])  # [x, y, width, height]
inset_ax0.set_xlim(-0.3,0.3)
inset_ax0.set_ylim(-0.3,0.3)
inset_ax0.set_aspect('equal')
ax0.indicate_inset_zoom(inset_ax0, alpha=0.5, edgecolor="black")
inset_ax1 = ax1.inset_axes([0.01, 0.06, 0.45, 0.45])  # [x, y, width, height]
inset_ax1.set_xlim(-125, 125)
inset_ax1.set_ylim(-125, 125)
inset_ax1.set_aspect('equal')
ax1.indicate_inset_zoom(inset_ax1, alpha=0.5, edgecolor="black")

fig2, ax2 = plt.subplots(1, 2, figsize=(12,6))
# fig3, ax3 = plt.subplots(1, 3, figsize=(18,6))

for i, data in enumerate([data_01_10m, data_02_10m, data_05_10m, data_06_10m, data_01_20m, data_02_20m, data_05_20m, data_06_20m]):
    if len(data.keys()) == 0:
        continue
    if i == 0:
        colors = ['xkcd:red', 'xkcd:magenta', 'xkcd:salmon']
        marker = '+'
        tmp = '01 @ 10m'
    elif i == 1:
        colors = ['xkcd:green', 'xkcd:lime green', 'xkcd:sea green']
        marker = 'x'
        tmp = '02 @ 10m'
    elif i == 2:
        colors = ['xkcd:purple', 'xkcd:lavender', 'xkcd:dark purple']
        marker = 'x'
        tmp = '05 @ 10m'
    elif i == 3:
        colors = ['xkcd:blue', 'xkcd:cyan', 'xkcd:turquoise']
        marker = 'x'
        tmp = '06 @ 10m'
    elif i == 4:
        colors = ['xkcd:brown', 'xkcd:light brown', 'xkcd:dark brown']
        marker = '+'
        tmp = '01 @ 20m'
    elif i == 5:
        colors = ['xkcd:mustard', 'xkcd:yellow', 'xkcd:light orange']
        marker = 'x'
        tmp = '02 @ 20m'
    elif i == 6:
        colors = ['xkcd:slate', 'xkcd:dark grey', 'xkcd:light grey']
        marker = 'x'
        tmp = '05 @ 20m'
    elif i == 7:
        colors = ['xkcd:terracotta', 'xkcd:mud', 'xkcd:clay']
        marker = 'x'
        tmp = '06 @ 20m'
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

    ax0.scatter(bproj[:,0], bproj[:,1], c=colors[0], s=6*base, alpha=0.1)#bproj[:,2],
    ax0.scatter(bproj[:,0].mean(), bproj[:,1].mean(), c=colors[0], marker='x', label=tmp)# bproj[:,2].mean,
    inset_ax0.scatter(bproj[:,0], bproj[:,1], c=colors[0], s=6*base, alpha=0.1)#bproj[:,2],
    inset_ax0.scatter(bproj[:,0].mean(), bproj[:,1].mean(), c=colors[0], marker='x')
    # ax[0].scatter(gt_c_bproj[:,0], gt_c_bproj[:,1], c=colors[1], s=4*base, alpha=0.1, label=tmp+'gtClickBackProj')
    # ax[0].scatter(gt_a_bproj[:,0].mean(), gt_a_bproj[:,1].mean(), c=colors[0], marker=marker, label=tmp+'Mean Error')# bproj[:,2].mean,
    # ax[0].scatter(gt_a_bproj[:,0], gt_a_bproj[:,1], c=colors[2], s=2*base, alpha=0.1, label=tmp+'gtAprilBackProj')

    ax1.scatter(reproj[:,0], reproj[:,1], s=6*base, c=colors[0], alpha=0.1)
    ax1.scatter(reproj[:,0].mean(), reproj[:,1].mean(), c=colors[0], marker='x', label=tmp)
    inset_ax1.scatter(reproj[:,0], reproj[:,1], s=6*base, c=colors[0], alpha=0.1)
    inset_ax1.scatter(reproj[:,0].mean(), reproj[:,1].mean(), c=colors[0], marker='x')
    # ax[1].scatter(gt_a_reproj[:,0], gt_a_reproj[:,1], c=colors[1], s=4*base, alpha=0.1, label=tmp+'aprilReproj')

    lbl = tmp+f' Norm: {np.linalg.norm(bproj, axis=1).mean():.3f} +/- {np.linalg.norm(bproj, axis=1).std():.3f} m'
    ax2[0].hist(np.linalg.norm(bproj, axis=1), bins=100, density=True, color=colors[0], label=lbl)
    lbl = tmp+f' Norm: {np.linalg.norm(reproj, axis=1).mean():.1f} +/- {np.linalg.norm(reproj, axis=1).std():.1f} px'
    ax2[1].hist(np.linalg.norm(reproj, axis=1), bins=100, density=True, color=colors[0], label=lbl)

    print()

    # ax3[0].plot(gt_c_bproj[:,0], gt_c_bproj[:,1], label='click_bproj')
    # ax3[0].plot(gt_a_bproj[:,0], gt_a_bproj[:,1], label='april_bproj')
    # ax3[1].plot(gt_c_bproj[:,0], label='click_bproj')
    # ax3[1].plot(gt_a_bproj[:,0], label='april_bproj')
    # ax3[1].plot(gt_c_bproj[:,1], label='click_bproj')
    # ax3[2].plot(gt_a_bproj[:,1], label='april_bproj')
    # ax3[2].legend(fontsize=12)
ax0.scatter(0.0, 0.0, c='k', s=100, marker='s', label='Truth')#0.0,
ax1.scatter(0.0, 0.0, c='k', s=20, label='Origin')
inset_ax0.scatter(0.0, 0.0, c='k', s=100, marker='s')#0.0,
inset_ax1.scatter(0.0, 0.0, c='k', s=100, marker='s')
ax0.legend(fontsize = 12)
ax1.legend(fontsize = 12)
ax2[0].legend(fontsize=12)
ax2[1].legend(fontsize=12)

plt.show()
print()
