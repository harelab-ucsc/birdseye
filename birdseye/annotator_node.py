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

from birdseye.camera.projection_models import ProjectionEngine, MeshBackend
from birdseye.camera.camera import SensorConfigLoader, PinholeCameraModel
from birdseye_msgs import CaptureComplete, CameraCapture

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
        self.declare_parameter("yaml_filepath", "")
        self.declare_parameter("mesh_filepath", "")
        self.declare_parameter("click_filepath", "")
        self.declare_parameter("save_dir", "")
        self.declare_parameter("img_format", ".png")
        self.yaml_filepath = self.get_parameter("yaml_filepath").value
        self.mesh_filepath = self.get_parameter("mesh_filepath").value
        self.clicks_csv = self.get_parameter("click_filepath").value
        self.save_dir = self.get_parameter("save_dir").value
        self.img_format = self.get_parameter("img_format").value
        self.label_save_name = os.path.join(self.save_dir, "labels.txt")
        self.clicks = None
        self.csv_read()

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
        self.backend = self._load_mesh()
        self._setup_cameras()

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
        self.T_ned_enu[:3,:3] = np.array([[0,1,0],[1,0,0],[0,0,-1]])  # NED INS to ENU viz

    def csv_read(self):
        self.get_logger().info(f"Reading clicks CSV: {self.clicks_csv}...")
        data = []
        with open(self.clicks_csv) as clicks:
            reader = csv.reader(clicks)
            for line in reader:
                # returns easting, northing, zone number, zone letter
                u = utm.from_latlon(float(line[0]), float(line[1]))
                tag = int(line[-1][-1])
                data.append(  # Eastingn Northing, Number, Letter, EPS, MSL, tag
                    [u[0], u[1], u[2], u[3], float(line[2]), float(line[3]), tag]
                )
        self.clicks = np.array(data)

    def _setup_cameras(self):
        for cam_name in self.cam_loader.list_cameras():
            cam_cfg = self.cam_loader.get_camera(cam_name)
            cam = PinholeCameraModel.from_config(cam_cfg)
            backend = self.backend  # shared mesh
            self.pipelines[cam_name] = ProjectionEngine(cam, backend)
            topic = f"/{cam_name}/camera/image_raw"
            self.subscribers[cam_name] = self.create_subscription(
                Image,
                topic,
                partial(self._image_callback, cam_name=cam_name),
                img_qos,
            )

    def _load_mesh(self):
        mesh = o3d.io.read_triangle_mesh(self.mesh_filepath)
        tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(tmesh)
        return MeshBackend(scene)

    def _get_msg_time(self, msg):
        try:
            return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        except AttributeError:
            return time.time()

    def _capture_complete_callback(self, msg: CaptureComplete):
        self.get_logger().info(
            f"Received capture with {len(msg.cameras)} camera(s)"
        )

        job = {
            "created_at": time.time(),
            "data": msg,
        }

        with self.state_lock:
            self.active_jobs.append(job)

        # Try to resolve older jobs
        self.process_jobs()

    def process_jobs(self):
        now = time.time()

        with self.state_lock:
            while self.active_jobs:
                job = self.active_jobs[0]
                self.active_jobs.popleft()
                self.save_queue.put(job["data"])

    def _save_worker(self):
        while True:
            item = self.save_queue.get()
            if item is None:
                self.save_queue.task_done()
                break
            data, stamp = item
            try:
                self.post_process(data, stamp)
            finally:
                self.save_queue.task_done()

    def post_process(self, data):
        stamp = data.header.stamp
        pose = data.ins_pose_ned
        T_ins_ned = self.pose_msg_to_matrix(pose)

        for cam in data.cameras:
            pipeline = self.pipelines[cam.camera_name]
            T_cam_ins = pipeline.camera.T_cam_ins
            assert T_cam_ins == self.pose_msg_to_matrix(cam.cam_pose_ins)

            T_cam_world = self.T_ned_enu @ T_ins_ned @ T_cam_ins
            pixels, visible = pipeline.visible_world_points(
                self.clicks[:, [0,1,4]],
                T_cam_world
            )
            img_file = os.path.join( self.save_dir, cam.image_filename )
            tags = self.clicks[visible][:, -1]
            pix = pixels[visible]
            labels = np.hstack((pix, tags[:, None]))
            self.annotate(img_file, labels)

    def pose_msg_to_matrix(self, pose):
        t = [
            pose.position.x,
            pose.position.y,
            pose.position.z
        ]
        quat = [
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w
        ]
        rot = R.from_quat(quat).as_matrix()

        Tf = np.eye(4)
        Tf[:3,:3] = rot
        Tf[:3,3] = t
        return Tf

    def annotate(self, img_file, labels, save_name=None):
        if save_name is None:
            save_name = self.label_save_name

        with open(save_name, "a")as f:
            for label in labels:
                line = f'{img_file} '

                try:  # if label is an iterable
                    vals = ','.join([str(x) for x in label])
                    # print(vals)
                except TypeError:  # else
                    vals = str(label)

                line += vals
                line += '\n'
                # print(line)
                f.write(line)
                print(
                    f'[PROC]    Annotation saved to {save_name}.txt:\n'
                    f'[PROC]        {line}'
                )

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
