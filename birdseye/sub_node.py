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

        # PARAMETERS (ROS2 style)
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
        self.max_latency = 0.2  # seconds (tune this)
        self.assignment_window = {  # Per-sensor assignment windows (AFTER PPS)
            "pose": self.max_latency,
        }

        self.cam_loader = SensorConfigLoader(self.yaml_filepath)
        self.pipelines = {}
        self.subscribers = {}
        self.backend = self._load_mesh()
        self.pretrigger_tolerance = 0.05
        self._setup_cameras()

        self.subscribers["pps"] = self.create_subscription(
            BuiltinTime, "/pps/time", self._pps_cb, qos_profile=pps_qos
        )
        if DIDINS2 is not None:
            self.subscribers["ins"] = self.create_subscription(
                DIDINS2, "/ins_quat_uvw_lla", self._ins_callback, qos_profile=sns_qos
            )

        # --- Producer/consumer save queue ---
        self.save_queue = queue.Queue()
        self._save_workers = []
        for _ in range(8):
            t = threading.Thread(target=self._save_worker, daemon=True)
            t.start()
            self._save_workers.append(t)

        threading.Thread(target=self._queue_watchdog, daemon=True).start()
        self.HDW_STROBE = 0x00000020

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
            self.assignment_window[cam_name] = self.max_latency

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

    def _pps_cb(self, msg: BuiltinTime):
        pps_time = msg.sec + msg.nanosec * 1e-9

        job = {
            "pps_time": pps_time,
            "stamp_msg": msg,
            "created_at": time.time(),
            "data": {
                "pose": None,
            },
            "dt": {},  # diagnostics
        }
        for cam_name in self.cam_loader.list_cameras():
            job["data"][cam_name] = None

        with self.state_lock:
            self.active_jobs.append(job)

        # Try to resolve older jobs
        self.process_jobs()

    def _ins_callback(self, msg):
        if msg.hdw_status & self.HDW_STROBE == self.HDW_STROBE:
            self.assign_to_job("pose", msg)

    def _image_callback(self, msg, cam_name):
        engine = self.pipelines[cam_name]
        self.assign_to_job(cam_name, msg)

    def assign_to_job(self, key, msg):
        ts = self.get_msg_time(msg)

        with self.state_lock:
            best_job = None
            best_dt = float("inf")

            for job in self.active_jobs:
                dt = ts - job["pps_time"]

                # Allow small pre-trigger (INS edge case)
                if dt < -self.pretrigger_tolerance:
                    continue

                if dt > self.assignment_window[key]:
                    continue

                abs_dt = abs(dt)

                if abs_dt < best_dt:
                    best_dt = abs_dt
                    best_job = job

            if best_job is None:
                return

            existing = best_job["data"][key]

            if existing is None:
                best_job["data"][key] = msg
                best_job["dt"][key] = best_dt
            else:
                # Replace if closer
                if best_dt < best_job["dt"][key]:
                    best_job["data"][key] = msg
                    best_job["dt"][key] = best_dt

    def process_jobs(self):
        now = time.time()

        with self.state_lock:
            while self.active_jobs:
                job = self.active_jobs[0]

                if now - job["created_at"] < self.max_latency:
                    break  # wait for more data

                self.active_jobs.popleft()

                if self.is_complete(job):
                    self.log_sync_diagnostics(job)
                    self.save_queue.put((job["data"], job["stamp_msg"]))
                else:
                    self.get_logger().warn(
                        f"PPS frame drop @ {job['pps_time']:.3f} (incomplete)"
                    )
                    for key in job["data"].keys():
                        if job["data"][key] is None:
                            self.get_logger().warn(f"    job['data'][{key}] is None")

    def is_complete(self, job):
        return all(v is not None for v in job["data"].values())

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

    def post_process(self, data, stamp):
        # ANNOTATIONS
        pose = data["pose"]
        u = utm.from_latlon(pose.lla[0], pose.lla[1])
        t = [  # UTM -> x:easting, y:northing, z:WGS84 altitude
            u[1],           # North
            u[0],           # East
            -pose.lla[2]    # Down
        ]
        quat = [  # quat is scalar-first NED -> convert to scalar-last NED
            pose.qn2b[1],
            pose.qn2b[2],
            pose.qn2b[3],
            pose.qn2b[0]
        ]
        rot = R.from_quat(quat).as_matrix()

        for cam_name in self.pipelines.keys():
            pipeline = self.pipelines[cam_name]
            T_cam_ins = pipeline.camera.T_cam_ins
            T_ins_ned = np.eye(4)
            T_ins_ned[:3,:3] = rot
            T_ins_ned[:3,3] = t
            T_cam_world = self.T_ned_enu @ T_ins_ned @ T_cam_ins
            pixels, visible = pipeline.visible_world_points(
                self.clicks[:, [0,1,4]],
                T_cam_world
            )
            img_file = os.path.join(
                self.save_dir,
                f'{cam_name}_{stamp.sec}.{stamp.nanosec:09}{self.img_format}'
            )
            tags = self.clicks[visible][:, -1]
            pix = pixels[visible]
            labels = np.hstack((pix, tags[:, None]))
            self.annotate(img_file, labels)

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

    def log_sync_diagnostics(self, job):
        dt_info = job["dt"]
        msg = ", ".join(
            f"{k}:{v * 1000:.1f}ms" for k, v in dt_info.items() if v is not None
        )
        self.get_logger().debug(f"[SYNC] {msg}")

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
