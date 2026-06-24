import yaml

import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../birdseye"))
    )
import utilities

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from scipy.stats import binned_statistic_2d


ALTITUDES = [ 1.0, 10.0, 20.0 ]
NUM_DRAWS = 10000
THETAS = np.linspace(0,45,91)


def calibUptake(sensors_yaml, sensor):

    print(f'Reading sensor parameters YAML file: {sensors_yaml}...')

    with open(sensors_yaml, 'r') as f:
        params = yaml.safe_load(f)
        data = params[sensor]

        res = data["resolution"]
        K = data["intrinsics"]
        dist = data["distortion_coeffs"]
        extr = data["T_cam_imu"]  # extrinsics relative to imu base link
        extr = utilities.matrix_list_converter(extr, (4,4))

    return res, K, dist, extr


def _3Dto2D(List3D, K, T_WC):
    List2D = []

    List3D = np.array(List3D, dtype=np.float64)

    # Ensure points are homogeneous (Nx4)
    if List3D.shape[1] == 3:
        List3D = np.hstack((List3D, np.ones((List3D.shape[0], 1))))  # Add w=1

    cam_frame_points = np.linalg.inv(T_WC)@List3D.T  # 4xN result
    cam_frame_points = np.array([[0,-1,0,0],[-1,0,0,0],[0,0,1,0],[0,0,0,1]])@cam_frame_points
    projected = K@cam_frame_points[:-1, :]  # Remove homogeneous w
    projected /= projected[2]  # Normalize by depth (z)
    List2D = projected[:2].T.tolist()

    return List2D


def _2Dto3D(List2D, K, T_WC, h):
    List3D = []

    for p in List2D:

        p = np.array([p[0], p[1], 1]).T

        pc = np.linalg.inv(K) @ p
        pc = np.hstack((pc,1.0))
        pc = np.array([[0,-1,0,0],[-1,0,0,0],[0,0,1,0],[0,0,0,1]])@pc
        pw = T_WC @ pc

        # Find a ray from camera to 3d point, scale to depth
        vector = pw - T_WC[:,3]
        unit_vector = vector / np.linalg.norm(vector)
        up = np.array([0.0,0.0,1.0])
        corr = h/(unit_vector[:3]@up)
        p3D = T_WC[:,3] - corr * unit_vector

        List3D.append(p3D.tolist())

    return List3D


def compute_dh(points, theta, index):
    # points.shape == (N, 2)
    # r = np.linalg.norm(points, axis=1)   # sqrt(x^2 + y^2) for each row
    dh = points[:,index] * np.tan(np.deg2rad(theta))
    return dh.reshape(-1, 1)                 # shape: (N, 1)


def ned_to_enu_se3(pose_ned):
    R_ned_to_enu = np.array([[0, 1,  0],
                            [1, 0,  0],
                            [0, 0, -1]])

    T_ned_to_enu = np.eye(4)
    T_ned_to_enu[:3, :3] = R_ned_to_enu

    pose_enu = T_ned_to_enu @ pose_ned @ T_ned_to_enu.T
    return pose_enu


if __name__ == '__main__':
    sensors_yaml = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../config/birdsEyeSensorParams.yaml")
        )
    sensor = 'cam0'
    res, intr, dist, T_IC = calibUptake(sensors_yaml, sensor)
    K = np.array([[intr[0], 0.0, intr[2]], [0.0, intr[1], intr[3]], [0.0, 0.0, 1.0]])
    T_IC= utilities.matrix_list_converter(T_IC, (4,4))
    T_IC = ned_to_enu_se3(T_IC)
    res = np.array(res)
    # print(res, -res/2, res/2)

    for index in [0, 1]:
        print('Index: ', str(index))
        for h in ALTITUDES:
            print('    Altitude = ', h)
            all_theta = []
            all_radius = []
            all_error = []

            T_WI = np.eye(4)
            T_WI[2,3] = h  # Drone is {h}m above origin, axis-aligned ENU
            T_WC = T_WI@T_IC
            for th in THETAS:
                print('        Decline Angle = ', th, end='\r')
                xy = np.random.uniform([0.0, 0.0], res, size=(NUM_DRAWS, 2))

                xy_3d = _2Dto3D(xy.tolist(), K, T_WC, h)
                xy_3d = np.array(xy_3d)[:,:2] / h
                delta_hs = compute_dh(xy_3d, th, index) * h

                TAGS_POS = np.hstack((xy_3d * h, np.zeros_like(delta_hs)))
                TAGS_POS_DELTA = np.hstack((xy_3d * h, delta_hs))

                # plt.figure(figsize=(8, 6))
                # sc = plt.scatter(
                #     TAGS_POS_DELTA[:, 0],
                #     TAGS_POS_DELTA[:, 1],
                #     c=TAGS_POS_DELTA[:, 2],
                #     cmap='viridis'
                # )
                # plt.colorbar(sc, label="Tag Z position (m)")
                # plt.title(f"TAGS_POS_DELTA (h = {h})")
                # plt.tight_layout()
                # plt.show()
                # plt.close()

                PROJ = _3Dto2D(TAGS_POS, K, T_WC)
                PROJ = np.array(PROJ)
                PROJ_DELTA = _3Dto2D(TAGS_POS_DELTA, K, T_WC)
                PROJ_DELTA = np.array(PROJ_DELTA)

                # plt.figure(figsize=(8, 6))
                # plt.scatter(
                #     PROJ[:, 0],
                #     PROJ[:, 1],
                #     c='r',
                #     s=5
                # )
                # plt.scatter(
                #     PROJ_DELTA[:, 0],
                #     PROJ_DELTA[:, 1],
                #     c='b',
                #     s=5
                # )
                # plt.title(f"PROJ, PROJ_DELTA (h = {h}, theta = {th})")
                # plt.tight_layout()
                # plt.show()
                # plt.close()

                err = np.linalg.norm(PROJ - PROJ_DELTA, axis=1)
                err = err.tolist()

                all_theta.append([th]*len(err))
                all_radius.append(xy_3d[:,index])
                all_error.append(err)

            theta_vals = np.concatenate(all_theta)
            radius_vals = np.concatenate(all_radius)
            error_vals = np.concatenate(all_error)

            # Define grid resolution
            n_theta_bins = 45
            n_radius_bins = 45

            # Compute binned mean error
            stat, theta_edges, radius_edges, _ = binned_statistic_2d(
                theta_vals,
                radius_vals,
                error_vals,
                statistic="mean",
                bins=[n_theta_bins, n_radius_bins]
            )

            # Bin centers
            theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
            radius_centers = 0.5 * (radius_edges[:-1] + radius_edges[1:])

            Theta, Radius = np.meshgrid(theta_centers, radius_centers, indexing="ij")


            plt.figure(figsize=(8, 6))
            levels = np.geomspace(
                max(np.nanmin(stat), 1e-1),
                np.nanmax(stat),
                10)
            cf = plt.contour(
                Theta,
                Radius,
                stat,
                levels=levels,
                norm=mcolors.LogNorm(),
                colors='black')
            plt.clabel(cf, inline=True, fontsize=14, fmt='%1.1f')
            plt.xlabel("Terrain Average Slope (degrees)", fontsize=20)
            plt.ylabel("Distance from Optical Center [m/m]", fontsize=20)
            plt.title(f"{h}m", fontsize=20)
            plt.xticks(fontsize=12)
            plt.tight_layout()
            plt.savefig(f"/home/mwmaster/catch/flatworld_alt{int(h)}m_{index}_contour.png")

            #
            # eps = 1e-1  # small positive floor
            #
            # vmin = max(error_vals.min(), eps)
            # vmax = error_vals.max()
            #
            # norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
            #
            # plt.figure(figsize=(8, 6))
            # sc = plt.scatter(
            #     theta_vals,
            #     radius_vals,
            #     c=error_vals,
            #     cmap="viridis",
            #     norm=norm,
            #     s=8,
            #     alpha=0.7
            # )
            # plt.xlabel("Terrain Average Slope (degrees)", fontsize=16)
            # plt.ylabel("Normalized Distance from Optical Center [unitless]", fontsize=16)
            # plt.title(f"Altitude = {h}m", fontsize=16)
            # plt.xticks(fontsize=12)
            # cbar = plt.colorbar(sc)
            # cbar.ax.tick_params(labelsize=12)
            # cbar.set_label(label="Pixel error (px)", size=16)
            # plt.tight_layout()
            # plt.savefig(f'/home/mwmaster/catch/flatworld_alt{int(h)}m_{index}.png')
            print()
        plt.show()
