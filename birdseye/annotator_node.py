import queue
import utm
import csv
import yaml
import cv2
import os
import threading
import time

import numpy as np
import open3d as o3d

from functools import partial
from collections import deque
from scipy.spatial.transform import Rotation as R
from scipy.spatial import KDTree

import rclpy
import tf2_ros

from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    qos_profile_sensor_data,
)
from builtin_interfaces.msg import Time as BuiltinTime
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image

from hloc.localize_sfm import QueryLocalizer

from birdseye.camera.projection_models import ProjectionEngine, MeshBackend
from birdseye.camera.camera import SensorConfigLoader, PinholeCameraModel
from birdseye.geometry.se3 import SE3
from birdseye_msgs.msg import CaptureComplete, CameraCapture

# Tolerant imports — these message types live in repos that may not be
# installed in test/CI containers (inertial_sense_ros2, custom_msgs).
# Subscriptions are skipped when their msg types aren't importable, and
# all_caught() naturally only requires inputs we actually subscribed to.
try:
    from inertial_sense_ros2.msg import DIDINS2
except ImportError:
    DIDINS2 = None

# try:
#     from custom_msgs.msg import AltSNR
# except ImportError:
#     AltSNR = None

# RELIABLE QoS for navigation/sensor data — these topics use RELIABLE
# and must not be dropped (INS, radalt, spectrometer, PPS).
sns_qos = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)

# 1. PPS Trigger (The heartbeat of the state machine)
# depth=1: only the latest pulse matters. Prevents sync_node from
# receiving a burst of backlogged PPS messages on startup (which would
# create many simultaneous jobs and flood the drop log).
pps_qos = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

# BEST_EFFORT QoS for camera images — camera drivers publish with SensorDataQoS
# (BEST_EFFORT). A RELIABLE subscription against a BEST_EFFORT publisher is a
# DDS QoS incompatibility; no messages flow. For image data, BEST_EFFORT is
# correct: a missed frame is recovered on the next PPS cycle.
img_qos = qos_profile_sensor_data


class AnnotatorNode(Node):
    # TODO: registration against pre-existing mesh

    def __init__(self):
        super().__init__("projection_node")

        # --- PARAMETERS ---
        self.declare_parameter("yaml_filepath", "")   # payload calibration
        self.declare_parameter("mesh_filepath", "")   # 3D reconstruction data
        self.declare_parameter("click_filepath", "")  # RTK annotations and GCPs
        self.declare_parameter("save_dir", "")        #output path
        self.yaml_filepath = self.get_parameter("yaml_filepath").value
        self.mesh_filepath = self.get_parameter("mesh_filepath").value
        self.clicks_csv = self.get_parameter("click_filepath").value
        self.save_dir = self.get_parameter("save_dir").value
        self.label_save_name = os.path.join(self.save_dir, "labels.txt")
        os.close(os.open(self.label_save_name, os.O_CREAT | os.O_WRONLY))

        # --- State Machine Variables ---
        self.state_lock = threading.Lock()
        self.active_jobs = deque(maxlen=10)

        # --- CaptureComplete subscriber
        self.capture_sub = self.create_subscription(
            CaptureComplete,
            "/sync/capture_complete",
            self._capture_complete_callback,
            10,
        )

        # --- Camera setup ---
        self.cam_loader = SensorConfigLoader(self.yaml_filepath)
        self.pipelines = {}
        backend, mesh = self._load_mesh()  # backend: project, mesh: render
        self.backend = backend
        self._setup_cameras()

        # --- Geotag / click setup
        self.clicks = None
        self.labels = None
        self.clicks_read()

        # --- Producer/consumer save queue ---
        self.save_queue = queue.Queue()
        self._save_workers = []
        for _ in range(8):
            t = threading.Thread(target=self._save_worker, daemon=True)
            t.start()
            self._save_workers.append(t)

        # --- Watchdog thread ---
        threading.Thread(target=self._queue_watchdog, daemon=True).start()

        # --- Helper transform ---
        self.T_ned_enu = np.eye(4)
        self.T_ned_enu[:3, :3] = np.array(
            [[0, 1, 0], [1, 0, 0], [0, 0, -1]]
        )  # NED INS to ENU viz

    def clicks_read(self):
        self.get_logger().info(f"Reading clicks CSV: {self.clicks_csv}...")
        data = []
        with open(self.clicks_csv) as clicks:
            reader = csv.reader(clicks)
            for line in reader:
                # returns easting, northing, zone number, zone letter
                u = utm.from_latlon(float(line[0]), float(line[1]))
                tag = line[-1][-1]
                data.append(  # Easting, Northing, Number, Letter, EPS, MSL, tag
                    [u[0], u[1], u[2], u[3], float(line[2]), float(line[3]), tag]
                )
        data = np.array(data)
        labels = data[:, -1]
        data = data[:, [0,1,4]]  # east, north, wgs84_altitude
        data = self.backend.snap_points_along_direction(
            data,
            direction=[0, 0, 1],   # ENU "up" / NED "down"
            max_distance=20.0,
        )
        self.clicks = data
        self.labels = labels
        self.get_logger().info("    ...Done reading clicks.")


    def _setup_cameras(self):
        self.get_logger().info("Setting up cameras...")
        for cam_name in self.cam_loader.list_cameras():
            cam_cfg = self.cam_loader.get_camera(cam_name)
            cam = PinholeCameraModel.from_config(cam_cfg)
            backend = self.backend  # shared mesh
            proj = ProjectionEngine(cam, backend)
            self.pipelines[cam_name] = proj
        self.get_logger().info(
            f"    ...Set up cameras: {self.cam_loader.list_cameras()}."
        )

    def _load_mesh(self):
        self.get_logger().info("Loading mesh...")
        mesh = o3d.io.read_triangle_mesh(self.mesh_filepath)
        tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(tmesh)
        self.get_logger().info("    ...Done loading mesh.")
        return MeshBackend(scene), mesh

    def _get_msg_time(self, msg):
        try:
            return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        except AttributeError:
            return time.time()

    def _capture_complete_callback(self, msg: CaptureComplete):
        self.get_logger().debug(f"Received capture with {len(msg.cameras)} camera(s)")
        job = {
            "created_at": time.time(),
            "data": msg,
        }

        with self.state_lock:
            self.active_jobs.append(job)

        # Try to resolve older jobs
        self.process_jobs()

    def process_jobs(self):
        with self.state_lock:
            while self.active_jobs:
                job = self.active_jobs[0]
                self.active_jobs.popleft()
                self.save_queue.put(job["data"])

    def _save_worker(self):
        while True:
            item = self.save_queue.get()

            if item is None:
                self.get_logger().info("Worker exiting.")
                self.save_queue.task_done()
                break

            self.get_logger().debug(
                f"Worker starting job. Queue={self.save_queue.qsize()}"
            )

            start = time.perf_counter()

            try:
                msg = item
                self.post_process(msg)
            finally:
                elapsed = time.perf_counter() - start
                self.get_logger().info(
                    f"Finished save in {elapsed:.2f}s"
                )
                self.save_queue.task_done()

    def post_process(
        self,
        data,
        eps = 1e-6,
        min_contour_length=40,
    ):
        stamp = data.header.stamp
        real = cv2.imread(
            os.path.join(self.save_dir, cam.image_filename),
            cv2.IMREAD_GRAYSCALE
        )
        pose = data.ins_pose_ned
        T_ins_ned = self.pose_msg_to_matrix(pose)

        for cam in data.cameras:
            if cam.camera_name != 'rgb_1':
                continue

            proj = self.pipelines[cam.camera_name]
            T_cam_ins = proj.camera.T_cam_ins
            T_cam_world = self.T_ned_enu @ T_ins_ned @ T_cam_ins  # cam->world

            pose = self.localizer.localize(img, prior_pose)

            pixels, visible = proj.visible_world_points(self.clicks,T_cam_world)

            # TODO: all-black image filter

            # Apply flat-field correction
            gain = np.load("/home/mwmaster/catch/ffc_test/ffc_gain.npy")
            real = real.astype(np.float32)
            real *= gain

            # real *= 128.0 / real.mean()
            real = np.clip(real, 0, 255).astype(np.uint8)

            # TODO: hloc localization process

            # --- Annotate; diagnostic comparative for w/ and w/o register ---
            # TODO: annotate is not thread safe
            tags = self.labels[visible]
            pix = pixels[visible]
            labels = np.hstack((pix, tags[:, None]))
            self.annotate(cam.image_filename, labels)

            tags = self.labels[visible_r]
            pix = pixels_r[visible_r]
            labels = np.hstack((pix, tags[:, None]))
            self.annotate(cam.image_filename, labels, save_name='labelsReg.txt')

    def register_pose(
        self,
        proj,
        T_cam_world,
        dist,
        nearest_gx,
        nearest_gy,
        backend='gauss-newton',
        alpha=10,
    ):
        self.get_logger().info('Coarse search...')
        # Sample pose perturbations around INS estimate
        N = 200

        # xi ordering:
        # [tx, ty, tz, rx, ry, rz]
        # covariance values should be in matching units:
        # meters for translation, radians for rotation
        cov_diag = np.array([
            2,                  # sigma_x
            2,                  # sigma_y
            2 ,                   # sigma_z
            np.deg2rad(2),        # sigma_roll
            np.deg2rad(2),        # sigma_pitch
            np.deg2rad(15),      # sigma_yaw
        ]) ** 2

        Sigma = np.diag(cov_diag)

        # Draw perturbations in tangent space
        xis = np.random.multivariate_normal(
            mean=np.zeros(6),
            cov=Sigma,
            size=N
        )

        best_score = np.inf
        best_pose = T_cam_world
        T = np.linalg.inv(T_cam_world)

        for xi in xis:

            # SE3 perturbation around INS pose
            T_candidate = SE3.apply_se3_update(
                T,
                xi
            )

            synth_candidate = proj.render(T_candidate)
            H, W, D = synth_candidate.normals.shape

            u = synth_candidate.contour_pixels[:,0].astype(np.int32)
            v = synth_candidate.contour_pixels[:,1].astype(np.int32)
            syn_dirs = np.stack([
                synth_candidate.gx[v, u],
                synth_candidate.gy[v, u]
            ], axis=1)

            # Linearize around CURRENT accepted pose
            kps, valid_iter = proj.world_to_image(
                synth_candidate.contour_world,
                T
            )

            Pw = synth_candidate.contour_world[valid_iter]
            kps = kps[valid_iter]
            syn_iter = syn_dirs[valid_iter]

            u = np.round(kps[:,0]).astype(np.int32)
            v = np.round(kps[:,1]).astype(np.int32)

            inside = (
                (u >= 0) & (u < W) &
                (v >= 0) & (v < H)
            )

            Pw       = Pw[inside]
            kps      = kps[inside]
            syn_iter = syn_iter[inside]
            u        = u[inside]
            v        = v[inside]

            real_dirs = np.stack([
                nearest_gx[v, u],
                nearest_gy[v, u]
            ], axis=1)

            dot = np.sum(real_dirs * syn_iter, axis=1)
            dot = np.clip(dot, -1.0, 1.0)
            orientation_cost = 1.0 - np.abs(dot)

            r = self.residuals(dist, kps)
            r += alpha * orientation_cost

            score = np.dot(r,r)

            if score < best_score:
                best_score = score
                best_pose = T_candidate

        T_cam_world = np.linalg.inv(best_pose)
        self.get_logger().info('...coarse search done.')

        synth = proj.render(T_cam_world)
        if backend == 'gauss-newton':
            T_refined, residual, kps = self.gauss_newton_backend(
                proj,
                T_cam_world,
                dist,
                nearest_gx,
                nearest_gy,
                synth,
                alpha=alpha
            )
            return T_refined, residual, kps, synth
        elif backend == 'gradient-descent':
            return self.gradient_descent_backend(proj, T_cam_world, dist, synth), synth
        else:
            self.get_logger().info(
                f" Unrecognized backend choice. Got {backend};"
                " expected 'gauss-newton' or 'gradient-descent'"
            )

    def gauss_newton_backend(
        self,
        proj,
        T_cam_world,
        dist,
        nearest_gx,
        nearest_gy,
        synth,
        lamb = 1.0,
        alpha=10.0,
        beta=6.0,
        k_thresh = 5,
        max_iters=50
    ):

        gx = cv2.Sobel(dist, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(dist, cv2.CV_32F, 0, 1, ksize=3)

        T = np.linalg.inv(T_cam_world)

        H, W, D = synth.normals.shape
        landmarks = synth.contour_world

        u = synth.contour_pixels[:,0].astype(np.int32)
        v = synth.contour_pixels[:,1].astype(np.int32)
        syn_dirs = np.stack([
            synth.gx[v, u],
            synth.gy[v, u]
        ], axis=1)
        k = 0

        for iteration in range(max_iters):

            # Linearize around CURRENT accepted pose
            kps, valid_iter = proj.world_to_image(landmarks, T)

            Pw = landmarks[valid_iter]
            kps = kps[valid_iter]
            syn_iter = syn_dirs[valid_iter]

            u = np.round(kps[:,0]).astype(np.int32)
            v = np.round(kps[:,1]).astype(np.int32)

            inside = (
                (u >= 0) & (u < W) &
                (v >= 0) & (v < H)
            )

            Pw       = Pw[inside]
            kps      = kps[inside]
            syn_iter = syn_iter[inside]
            u        = u[inside]
            v        = v[inside]


            real_dirs = np.stack([
                nearest_gx[v, u],
                nearest_gy[v, u]
            ], axis=1)

            dot = np.sum(real_dirs * syn_iter, axis=1)
            dot = np.clip(dot, -1.0, 1.0)
            orientation_cost = 1.0 - np.abs(dot)

            J_proj = proj.analytic_projection_jacobian(T, Pw)
            J_image = self.analytic_metric_jacobian(dist, gx, gy, kps)[:, None, :]
            J = np.einsum("nij,njk->nik", J_image, J_proj).squeeze(1)

            r = self.residuals(dist, kps)
            r += alpha * orientation_cost

            # s = np.linalg.svd(J, compute_uv=False)

            # self.get_logger().info(f"Singular values: {s}")
            # self.get_logger().info(f"Condition number: {s[0] / s[-1]}")
            JTJ = J.T @ J
            g = J.T @ r

            current_cost = np.dot(r,r)
            if not iteration:
                self.get_logger().info(
                    f'Initial Residual: {current_cost:.4f}'
                    f'    (mean distance: {r.mean():.4f} +/- {r.std():.4f})'
                    # f'Initial T: \n{np.linalg.inv(T)}'
                )

            while True:
                if k == k_thresh:
                    self.get_logger().info(' Rendering new synthetic view.')
                    synth = proj.render(np.linalg.inv(T))
                    landmarks = synth.contour_world
                    u = synth.contour_pixels[:,0].astype(np.int32)
                    v = synth.contour_pixels[:,1].astype(np.int32)
                    syn_dirs = np.stack([
                        synth.gx[v, u],
                        synth.gy[v, u]
                    ], axis=1)
                    k = 0

                A = JTJ + lamb * np.diag(np.diag(JTJ))
                delta = np.linalg.solve(A, -g)

                T_candidate = SE3.apply_se3_update(T, delta)

                kps_new, valid = proj.world_to_image(landmarks, T_candidate)
                Pw_new = landmarks[valid]
                kps_new = kps_new[valid]
                syn_new = syn_dirs[valid]

                u = np.round(kps_new[:, 0]).astype(np.int32)
                v = np.round(kps_new[:, 1]).astype(np.int32)

                inside = (
                    (u >= 0) & (u < W) &
                    (v >= 0) & (v < H)
                )

                Pw_new   = Pw_new[inside]
                kps_new  = kps_new[inside]
                syn_new  = syn_new[inside]
                u        = u[inside]
                v        = v[inside]

                real_dirs = np.stack([
                    nearest_gx[v, u],
                    nearest_gy[v, u]
                ], axis=1)

                dot = np.sum(real_dirs * syn_new, axis=1)
                dot = np.clip(dot, -1.0, 1.0)
                orientation_cost = 1.0 - np.abs(dot)

                r_new = self.residuals(dist, kps_new)
                r_new += alpha * orientation_cost

                new_cost = np.dot(r_new, r_new)
                # self.get_logger().info(
                #     f"    lambda: {lamb}"
                    # f"    delta: {np.linalg.norm(delta)}\n"
                    # f"    predicted: {-g @ delta}\n"
                    # f"    current: {current_cost}\n"
                    # f"    candidate: {new_cost}"
                # )
                if new_cost < current_cost:
                    # accept
                    current_cost = new_cost
                    T = T_candidate
                    lamb *= 0.5
                    self.get_logger().info(
                        f'  Residual ({iteration+1}): {current_cost:.4f}'
                        f'    (mean distance: {r_new.mean():.4f})'
                    )
                    k += 1
                    break

                # reject
                lamb *= 2.0

                if lamb > 1e10:
                    self.get_logger().info(f'Early exit - LM lambda runaway: {lamb} > 1e10')
                    return T, r, kps

            if new_cost < 1e3:
                self.get_logger().info(
                    f'  Final Residual (Converged, {iteration+1}): '
                    f'{current_cost:.4f}'
                    f'    (mean distance: {r.mean():.4f} +/- {r.std():.4f})'
                    # f'\n  Final T: \n{np.linalg.inv(T)}'
                )
                return T, r, kps


        self.get_logger().info(
            f'  Final Residual (Not Converged): {current_cost:.4f}'
            f'    (mean distance: {r.mean():.4f} +/- {r.std():.4f})'
            # f'  Final T: {np.linalg.inv(T)}'
        )
        return T, r, kps

    def gradient_descent_backend(self, T_cam_world, dist):
        self.get_logger().info(
            f" Not implemented error: 'gradient-descent'"
            "backend is not implemented.\n"
            " Returning T_cam_world. "
        )
        return T_cam_world

    def residuals(self, distance_image, uv):
        u = np.clip(uv[:,0].astype(np.int32), 0, distance_image.shape[1]-1)
        v = np.clip(uv[:,1].astype(np.int32), 0, distance_image.shape[0]-1)
        return distance_image[v, u]  # opencv index is col/row vs numpy row/col

    def analytic_metric_jacobian(self, distance_image, gx, gy, kps):
        u = np.clip(kps[:,0].astype(np.int32), 0, distance_image.shape[1]-1)
        v = np.clip(kps[:,1].astype(np.int32), 0, distance_image.shape[0]-1)
        gx_i = gx[v, u]
        gy_i = gy[v, u]
        return np.stack((gx_i, gy_i), axis=-1)

    def pose_msg_to_matrix(self, pose):
        t = [pose.position.x, pose.position.y, pose.position.z]
        quat = [
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ]
        rot = R.from_quat(quat).as_matrix()

        Tf = np.eye(4)
        Tf[:3, :3] = rot
        Tf[:3, 3] = t
        return Tf

    def annotate(self, img_file, labels, save_name=None):
        if save_name is None:
            save_name = self.label_save_name

        with open(save_name, "a") as f:
            lines = []
            for label in labels:
                line = f"{img_file} "

                try:  # if label is an iterable
                    vals = ",".join([str(x) for x in label])
                    # print(vals)
                except TypeError:  # else
                    vals = str(label)

                line += vals
                line += "\n"
                f.write(line)
                # print(
                #     f"[PROC]    Annotation saved to {save_name}.txt:\n"
                #     f"[PROC]        {line}"
                # )

    def _queue_watchdog(self):
        while rclpy.ok():
            sq = self.save_queue.qsize()
            if sq > 0:
                self.get_logger().info(f"[PROC] [WATCH]    Save queue depth: {sq}")
            time.sleep(2.0)

    def destroy_node(self):
        # Snapshot depth before sentinels so we don't count them as pending work.
        remaining = self.save_queue.qsize()
        if remaining > 0:
            self.get_logger().info(
                f"Shutdown: waiting for {remaining} queued save(s) to finish..."
            )
            while not self.save_queue.empty():
                self.get_logger().info(
                    f"  save queue: {self.save_queue.qsize()} job(s) remaining"
                )
                time.sleep(2.0)
        for _ in self._save_workers:
            self.save_queue.put(None)
        self.save_queue.join()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = AnnotatorNode()
    try:
        while rclpy.ok():
            try:
                rclpy.spin_once(node, timeout_sec=1.0)
            except RuntimeError as e:
                # FastDDS SHM corruption (e.g. after a peer node SIGSEGV) can
                # cause take_message to throw; log and continue rather than
                # crashing the whole node.
                node.get_logger().error(f"Executor RuntimeError (continuing): {e}")
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
