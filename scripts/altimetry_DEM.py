#!/usr/bin/env python3
import pdb
import rclpy
from rclpy.node import Node
from rosbag2_py import SequentialReader, SequentialWriter, StorageOptions, ConverterOptions
from inertial_sense_ros2.msg import DIDINS2
import argparse
import numpy as np
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message, serialize_message
from scipy.spatial.transform import Rotation as R
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from matplotlib.path import Path
from mpl_toolkits.mplot3d import Axes3D
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, RationalQuadratic, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from scipy.spatial import ConvexHull

import torch
import gpytorch
from torch.utils.data import TensorDataset, DataLoader

import joblib
import cv2
import os
import json
import yaml
import time
import copy
import utm
import math
from pyproj import Proj, Transformer


def quat2euler(qx, qy, qz, qw, degrees=False):
    euler = R.from_quat([qx, qy, qz, qw]).as_euler('xyz', degrees)
    return euler


class SVGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class BagProcessor:
    def __init__(self, input_bags, ds_dir, radalt_topic, ins_topic, tune_gp, load_gp, ds_rate=1):
        self.input_bags = input_bags
        self.bag_index = 0
        self.radalt_topic = radalt_topic
        self.ins_topic = ins_topic
        self.ds_dir = ds_dir
        if not os.path.isdir(self.ds_dir):
            os.makedirs(self.ds_dir, exist_ok=True)
        self.tune_gp = tune_gp
        self.load_gp = load_gp
        self.ds_rate = ds_rate
        self.count = 0
        self.radalt = None
        self.quat = None

        self.radalt_msgs = []
        self.ins_msgs = []

        self.east = 0.0
        self.north = 0.0
        self.ins_alt = 0.0
        self.center= None

        self.DEM = []
        self.pairs = []
        self.train_ds = []

        # self.kernel = C(1.0) * Matern(length_scale=0.5, length_scale_bounds=(1e-15, 1e5), nu=2.5) + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-15, 1e5))
        self.kernel = C(1.0) * RationalQuadratic(length_scale=0.001, alpha=2e-06, length_scale_bounds=(1e-5, 1e5), alpha_bounds=(1e-15, 1e5)) #+ WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-18, 1e5))
        # self.kernel = C(1.0) * RBF(length_scale=0.5, length_scale_bounds=(1e-15, 1e5)) #+ WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-15, 1e5))
        self.GP = GaussianProcessRegressor( \
            kernel=self.kernel, \
            n_restarts_optimizer=9, \
            optimizer=lambda obj_func, initial_theta, bounds: scipy.optimize.minimize(obj_func, initial_theta, method="L-BFGS-B", bounds=bounds, options={"maxiter": 100000}))  # Adjust max_iter here
        self.GP.optimizer = "fmin_l_bfgs_b"  # Ensure optimization is enabled
        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler()

        # Create a new figure and add a 3D subplot
        self.fig = plt.figure(figsize=(10, 7))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_box_aspect([1,1,1])

        self.halflength = 75
        self.density = 20
        self.boundary = int((self.halflength**2 * self.density)**0.5)

        # Set the viewing angle
        # self.ax.view_init(elev=elev, azim=azim)


    def load_intrinsics(self, intrinsics_path):
        """Load camera intrinsics from a YAML file."""
        with open(intrinsics_path, "r") as file:
            print('loading intrinsics')
            return yaml.safe_load(file)


    def load_trained_gp(self):
        print("Loading saved GP model and scalers...")
        self.GP = joblib.load(os.path.join(self.ds_dir, "trained_gp_model.pkl"))
        self.scaler_X = joblib.load(os.path.join(self.ds_dir, "scaler_X.pkl"))
        self.scaler_Y = joblib.load(os.path.join(self.ds_dir, "scaler_Y.pkl"))
        print("  GP model and scalers loaded.")


    def get_timestamp(self, msg):
        ts = msg.header.stamp
        ts = int(ts.sec * 1e9 + ts.nanosec)
        return ts


    def match_pairs(self, radalt_msgs, ins_msgs):
        """Find the closest image message to the given timestamp."""
        rad = [self.get_timestamp(msg) for msg in radalt_msgs]
        ins = [self.get_timestamp(msg) for msg in ins_msgs]

        cost = [[abs(i-j) for i in rad] for j in ins]
        pairs = zip(*linear_sum_assignment(cost))
        tmp = copy.deepcopy(pairs)
        tmp = [[rad[i[1]], \
                ins[i[0]]] for i in tmp]
        print(f'    time mismatch across {len(rad)} radalt messages is {sum([i[0]-i[1] for i in tmp])/1e9:.04f}s')
        return pairs


    def correct_altitude(self, ins, rad):
        self.quat = [ins.qn2b[1], ins.qn2b[2], ins.qn2b[3], ins.qn2b[0]]
        eulers = quat2euler(*self.quat)
        cos_theta = math.cos(eulers[0]) * math.cos(eulers[1])

        # Compute corrected altitude
        self.radalt = rad.altitude * cos_theta
        if self.radalt < 0:
            self.radalt *= -1


    def parse_INS(self, ins_msg):
        self.east, self.north, _, _ = utm.from_latlon(ins_msg.lla[0], ins_msg.lla[1])
        self.ins_alt = ins_msg.lla[2]


    def add_ground_point(self, ins_msg, rad_msg):
        self.correct_altitude(ins_msg, rad_msg)
        self.parse_INS(ins_msg=ins_msg)
        # ground_point = [self.east, self.north, self.ins_alt, rad_msg.altitude]
        self.DEM.append([self.east, self.north, self.ins_alt-rad_msg.altitude])
        self.train_ds.append([self.east, self.north])


    def tune_GP_hyperparameters(self):
        print("Starting GP hyperparameter tuning...")

        x_tmp = np.array(self.train_ds)
        y_tmp = np.array(self.DEM)
        y_tmp = y_tmp[:,-1].reshape(-1, 1)

        self.scaler_X.fit(x_tmp)
        self.scaler_Y.fit(y_tmp)
        x_tmp = self.scaler_X.transform(x_tmp)
        y_tmp = self.scaler_Y.transform(y_tmp)

        # Define kernel options for grid search
        kernel_options = [
            C(1.0) * RBF(length_scale=ls) for ls in [0.001, 0.01, 0.1, 0.5, 1.0, 10.0, 100.0]
        ] + [
            C(1.0) * RationalQuadratic(length_scale=ls, alpha=alpha)
            for ls in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0] for alpha in [1e-7, 1e-6, 1e-5, 1e-4]
        ] + [
            C(1.0) * Matern(length_scale=ls, nu=nu)
            for ls in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0] for nu in [0.5, 1.5, 2.5, 3.5]
        ]

        param_grid = {'kernel': kernel_options}

        gp = GaussianProcessRegressor(n_restarts_optimizer=3, optimizer='fmin_l_bfgs_b')
        search = GridSearchCV(gp, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=3)

        search.fit(x_tmp, y_tmp)

        print("Best kernel found:", search.best_params_['kernel'])
        print("Best score:", -search.best_score_)

        # Assign the best model to self.GP
        self.GP = search.best_estimator_


    def GP_train(self):
        x_tmp = np.array(self.train_ds)
        y_tmp = np.array(self.DEM)
        y_tmp = y_tmp[:,-1].reshape(-1,1)
        start = time.time()
        self.scaler_X.fit(x_tmp)
        self.scaler_Y.fit(y_tmp)
        x_tmp = self.scaler_X.transform(x_tmp)
        y_tmp = self.scaler_Y.transform(y_tmp)
        print('  training GP...')
        print(f'    initial performance: {self.GP.score(x_tmp, y_tmp)}, {self.GP.kernel}')
        start = time.time()
        self.GP.fit(x_tmp, y_tmp)
        print(f'    optimized performance: {self.GP.score(x_tmp, y_tmp)}, {self.GP.kernel_}')
        print(f'    took {time.time()-start:.04f}s')


    def train_svgp(self, epochs=100, batch_size=128):
        print("Training Sparse GP with GPyTorch...")

        x_tmp = torch.tensor(self.train_ds, dtype=torch.float32)
        y_tmp = torch.tensor(np.array(self.DEM)[:,-1], dtype=torch.float32).view(-1)

        # Normalize
        self.scaler_X.fit(x_tmp)
        x_tmp = torch.tensor(self.scaler_X.transform(x_tmp), dtype=torch.float32)

        # Setup model
        inducing_points = x_tmp[:500]  # Take first 500 as inducing
        model = SVGPModel(inducing_points)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam([
            {'params': model.parameters()},
            {'params': likelihood.parameters()}
        ], lr=0.01)

        mll = gpytorch.mlls.VariationalELBO(likelihood, model, y_tmp.numel())

        dataset = TensorDataset(x_tmp, y_tmp)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                output = model(x_batch)
                loss = -mll(output, y_batch)
                loss.backward()
                optimizer.step()
            print(f"[{epoch+1}/{epochs}] Loss: {loss.item():.4f}")

        self.GP = model
        self.likelihood = likelihood
        model.eval()
        likelihood.eval()


    def GP_save(self):
        joblib.dump(self.GP, os.path.join(self.ds_dir, "trained_gp_model.pkl"))
        joblib.dump(self.scaler_X, os.path.join(self.ds_dir, "scaler_X.pkl"))
        joblib.dump(self.scaler_Y, os.path.join(self.ds_dir, "scaler_Y.pkl"))
        print("  Saved GP model and scalers to:", self.ds_dir)


    def get_hull(self):
        if not hasattr(self, 'train_ds') or len(self.train_ds) == 0:
            raise ValueError("Training data not available to compute convex hull.")
        points_2d = np.array(self.train_ds)
        self.hull = ConvexHull(points_2d)
        self.hull_path = Path(points_2d[self.hull.vertices])
        print("  Convex hull computed.")


    def crop_to_hull(self, grid_points):
        """Filter grid points to those inside the convex hull."""
        return self.hull_path.contains_points(grid_points)


    def plot_dem_points(self, title='DEM', elev=30, azim=30):
        """
        Plots 3D DEM points provided as a list of [x: Easting, y: Northing, z: Up] coordinates.

        Parameters:
            points (list): A list where each element is a list or tuple of [x, y, z].
            title (str): Title for the plot.
            elev (float): Elevation angle (in degrees) for the 3D view.
            azim (float): Azimuth angle (in degrees) for the 3D view.
        """
        # Convert the list of points to a NumPy array for easier slicing
        points_np = np.array(self.DEM)

        # Validate that the input is of the expected shape (N x 3)
        if points_np.ndim != 2 or points_np.shape[1] != 3:
            raise ValueError("Input 'points' must be a list of [x, y, z] coordinates.")

        # Scatter plot: color the points based on their z (elevation) value
        scatter = self.ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2],
                            # c=points_np[:, 2], cmap='terrain', marker='o', s=20)
                            c='r', alpha=0.3, s=20)

        # Set the axis labels and plot title
        self.center = points_np.mean(axis=0)
        self.ax.set_xlabel('East (m)')
        self.ax.set_xlim(self.center[0]-self.halflength, self.center[0]+self.halflength)
        self.ax.set_ylabel('North (m)')
        self.ax.set_ylim(self.center[1]-self.halflength, self.center[1]+self.halflength)
        self.ax.set_zlabel('Z (m)')
        self.ax.set_zlim(self.center[2]-self.halflength, self.center[2]+self.halflength)
        self.ax.set_title(title)

        # Display the plot
        # plt.show()


    def plot_GP_surface(self):
        if self.center is None:
            self.center = np.array(self.DEM).mean(axis=0)
            self.ax.set_xlabel('East (m)')
            self.ax.set_xlim(self.center[0]-self.halflength, self.center[0]+self.halflength)
            self.ax.set_ylabel('North (m)')
            self.ax.set_ylim(self.center[1]-self.halflength, self.center[1]+self.halflength)
            self.ax.set_zlabel('Z (m)')
            self.ax.set_zlim(self.center[2]-self.halflength, self.center[2]+self.halflength)
            self.ax.set_title('DEM')

        src = []
        dst = []
        for i in range(self.boundary):
            for j in range(self.boundary):
                print(f'  progress: ({i+1}, {j+1}) of ({self.boundary}, {self.boundary})', end='\r')
                xoff = ((i+1)*2-self.boundary) * self.halflength / self.boundary
                yoff = ((j+1)*2-self.boundary) * self.halflength / self.boundary
                off = self.center + np.array([xoff, yoff, 0])
                off = np.expand_dims(off[:2], axis=0)
                src.append(off.tolist())

        # pdb.set_trace()

        src = np.array(src)
        src = np.squeeze(src)
        self.get_hull()  # Make sure hull is computed
        mask = self.crop_to_hull(src)
        src = src[mask]

        X = self.scaler_X.transform(src)
        Y = self.GP.predict(X)
        Y = Y.reshape(-1,1)
        Y = self.scaler_Y.inverse_transform(Y)
        dst = np.squeeze(Y)
        scatter = self.ax.scatter(src[:, 0], src[:, 1], dst,
                            c=dst, cmap='terrain', marker='o', s=10)
        # Add a colorbar to show elevation mapping
        cbar = plt.colorbar(scatter, ax=self.ax, pad=0.1)
        cbar.set_label('Elevation (Z)')
        print()


    def process_bag(self):
        # Initialize reader and writer
        rad_ct = 0
        ins_ct = 0
        reader = SequentialReader()
        try:
            storage_options = StorageOptions(uri=self.input_bags[self.bag_index], storage_id="mcap")
            converter_options = ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")
            reader.open(storage_options, converter_options)
        except RuntimeError:
            storage_options = StorageOptions(uri=self.input_bags[self.bag_index], storage_id="sqlite3")
            converter_options = ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")
            reader.open(storage_options, converter_options)
        topics_and_types = reader.get_all_topics_and_types()

        topic_type_map = {t.name:t.type for t in topics_and_types}

        print(f'reading bag {self.input_bags[self.bag_index]} at {self.ds_rate} downsampling rate')
        # Read and process messages
        while reader.has_next():
            topic, data, timestamp = reader.read_next()
            message_type = get_message(topic_type_map[topic])
            msg = deserialize_message(data, message_type)

            if topic == self.radalt_topic:
                rad_ct += 1
                rad_ct %= self.ds_rate
                if rad_ct == 0:
                    self.radalt_msgs.append(msg)
            elif topic == self.ins_topic:
                ins_ct += 1
                ins_ct %= self.ds_rate
                if ins_ct == 0:
                    self.ins_msgs.append(msg)
        print(f'    radalt_msgs length: {len(self.radalt_msgs)}')
        print(f'    ins_msgs length: {len(self.ins_msgs)}')
        print('  bag read done \n')

        print('starting timeseries alignment')
        start = time.time()
        pairs = self.match_pairs(self.radalt_msgs, self.ins_msgs)
        print(f'  took {time.time() - start:.04f}s')

        # Create DEM using altimeter-pose pairs
        print("developing digital elevation model...\n")
        for pair in pairs:
            self.add_ground_point(self.ins_msgs[pair[0]], self.radalt_msgs[pair[1]])
            self.pairs.append(pair)


    def process_bags(self):
        for i in range(len(self.input_bags)):
            self.bag_index = i
            self.process_bag()


        start = time.time()
        if self.load_gp:
            self.load_trained_gp()
        elif self.tune_gp:
            print('training GP regressor for surface interpolation...')
            self.tune_GP_hyperparameters()
        else:
            print('training GP regressor for surface interpolation...')
            print('  no hyperparameter search')
            # self.GP_train()
            self.train_svgp()
        print(f'  took {time.time() - start:.04f}s')
        self.GP_save()

        print('plotting...')
        self.plot_GP_surface()
        self.plot_dem_points()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Use radar altimetry and INS poses to make a Digital Elevation Map (DEM).")
    # parser.add_argument("input_bag", help="Path to the input ROS2 bag file")
    parser.add_argument('--input_bags', nargs='+', help='List of bag filepaths')
    parser.add_argument("--ds_dir",  help="Path to the directory to save images/ and poses.json to")
    parser.add_argument("--radalt_topic", help="Radar Altimeter topic name (e.g., /radalt/data)")
    parser.add_argument("--ins_topic", help="INS topic name (e.g., /ins/data)")
    parser.add_argument("--downsampling", type=int, help="rate to downsample data streams by (1 gives no downsampling))")
    parser.add_argument("--tune_gp", action="store_true", help="If set, perform GP hyperparameter tuning before training")
    parser.add_argument("--load_gp", action="store_true", help="Load previously saved GP model instead of training")

    args = parser.parse_args()

    rclpy.init()

    processor = BagProcessor(args.input_bags, args.ds_dir, args.radalt_topic, args.ins_topic, args.tune_gp, args.load_gp, ds_rate=args.downsampling)
    processor.process_bags()

    rclpy.shutdown()


if __name__ == "__main__":
    main()
