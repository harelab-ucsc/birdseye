import numpy as np
import open3d as o3d

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from functools import partial

import tf2_ros
from geometry_msgs.msg import TransformStamped

from birdseye.camera.projection_models import ProjectionEngine, MeshBackend
from birdseye.camera.camera import SensorConfigLoader, PinholeCameraModel


class PoseResolver:
    def __init__(self):
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, Node("pose_resolver"))

    def get_T_ins_world_tf2(self, timestamp):
        """
        map (world) ← base_link (INS)
        """
        tf = self.tf_buffer.lookup_transform(
            "map",
            "base_link",
            timestamp
        )
        t = tf.transform.translation
        q = tf.transform.rotation
        T = tf2_ros.transformations.quaternion_matrix([
            q.x, q.y, q.z, q.w
        ])
        T[:3, 3] = [t.x, t.y, t.z]
        return T

    def get_T_cam_world_tf2(self, T_cam_ins, timestamp):
        T_ins_world = self.get_T_ins_world_tf2(timestamp)
        return T_ins_world @ T_cam_ins


class ProjectionNode(Node):
    # TODO: registration against pre-existing mesh

    def __init__(self):
        super().__init__("projection_node")

        # PARAMETERS (ROS2 style)
        self.declare_parameter("yaml_filepath", "")
        self.declare_parameter("mesh_filepath", "")
        self.declare_parameter("use_ins_tf", True)
        self.yaml_filepath = self.get_parameter("yaml_filepath").value
        self.mesh_filename = self.get_parameter("mesh_filepath").value
        self.use_ins_tf = self.get_parameter("use_ins_tf").value

        self.cam_loader = SensorConfigLoader(self.yaml_filepath)
        self.pose = PoseResolver()
        self.pipelines = {}
        self.subscribers = {}
        self.backend = self._load_mesh()

        self._setup_cameras()

    def _setup_cameras(self):
        for cam_name in self.cam_loader.list_cameras():
            cam_cfg = self.cam_loader.raw[cam_name]
            cam = PinholeCameraModel.from_config(cam_cfg)
            backend = self.backend  # shared mesh
            self.pipelines[cam_name] = ProjectionEngine(cam, backend)
            topic = f"/{cam_name}/camera/image_raw"
            self.subscribers[cam_name] = self.create_subscription(
                Image,
                topic,
                partial(self.image_callback, cam_name=cam_name),
                qos_profile_sensor_data,
            )

    def _load_mesh(self):
        mesh = o3d.io.read_triangle_mesh(self.mesh_filename)
        tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(tmesh)
        return MeshBackend(scene)

    def ins_callback(self):
        if msg.hdw_status & self.HDW_STROBE == self.HDW_STROBE:
            # TODO: associate to an image
            # TODO: visibility check for clicks
            # TODO: make annotations
            pass

    def image_callback(self, msg, cam_name):
        engine = self.pipelines[cam_name]

        # POSE RESOLVE
        T_world_cam = self.pose.get_T_world_cam_tf2(
            engine.camera.T_cam_ins,
            msg.header.stamp
        )

        # DETECTIONS (detection as a service, dispatched by keyframer)
        # TODO: define a keyframer for detection triggering
        # TODO: define a CNN_Detection service/action
        # TODO: cast detection rays
        # TODO: manage ray casting outputs


def main(args=None):
    rclpy.init(args=args)
    sub_node = ProjectionNode()
    rclpy.spin(sub_node)
    sub_node.destroy_node()
    rclpy.shutdown()
