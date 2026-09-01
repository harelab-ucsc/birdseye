#!/usr/bin/env python3

import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy

from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
)

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header

from cv_bridge import CvBridge

from scipy.spatial.transform import Rotation

from hloc import (
    extract_features,
    match_features,
    pairs_from_retrieval,
    localize_sfm,
)


class HlocLocalizationNode(Node):

    def __init__(self):
        super().__init__("hloc_localizer")

        # ------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------

        self.declare_parameter(
            "dataset_dir",
            "",
        )

        self.declare_parameter(
            "reconstruction_dir",
            "",
        )

        self.declare_parameter(
            "work_dir",
            "/tmp/hloc_online",
        )

        self.declare_parameter(
            "query_topic",
            "/cam0/image_raw",
        )

        self.declare_parameter(
            "camera_info_topic",
            "/cam0/camera_info",
        )

        self.declare_parameter(
            "pps_topic",
            "/pps/time",
        )

        self.declare_parameter(
            "retrieval_count",
            20,
        )

        self.declare_parameter(
            "localization_rate",
            1.0,
        )

        self.declare_parameter(
            "jpeg_quality",
            95,
        )

        self.dataset_dir = Path(
            self.get_parameter("dataset_dir").value
        )

        self.reconstruction_dir = Path(
            self.get_parameter("reconstruction_dir").value
        )

        self.work_dir = Path(
            self.get_parameter("work_dir").value
        )

        self.query_topic = self.get_parameter(
            "query_topic"
        ).value

        self.camera_info_topic = self.get_parameter(
            "camera_info_topic"
        ).value

        self.pps_topic = self.get_parameter(
            "pps_topic"
        ).value

        self.retrieval_count = int(
            self.get_parameter("retrieval_count").value
        )

        self.localization_rate = float(
            self.get_parameter("localization_rate").value
        )

        jpeg_quality = int(
            self.get_parameter("jpeg_quality").value
        )

        # ------------------------------------------------------------
        # Paths
        # ------------------------------------------------------------

        self.reference_images = self.dataset_dir / "images"

        self.reference_descriptors = (
            self.work_dir / "global_descriptors.h5"
        )

        self.reference_features = (
            self.work_dir / "features.h5"
        )

        self.query_dir = self.work_dir / "queries"

        self.query_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ------------------------------------------------------------
        # ROS state
        # ------------------------------------------------------------

        self.bridge = CvBridge()

        self.last_frame = None
        self.last_camera_info = None
        self.last_pps = None

        self.processing = False
        self.last_localization_time = 0.0

        # ------------------------------------------------------------
        # QoS
        # ------------------------------------------------------------

        camera_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        # ------------------------------------------------------------
        # Subscribers
        # ------------------------------------------------------------

        self.image_sub = self.create_subscription(
            Image,
            self.query_topic,
            self.image_callback,
            camera_qos,
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            10,
        )

        # PPS message type depends on your pps_time_pub implementation.
        #
        # Do not subscribe to it blindly until its message type is known.
        #
        # self.pps_sub = self.create_subscription(
        #     ...
        # )

        # ------------------------------------------------------------
        # Publisher
        # ------------------------------------------------------------

        self.pose_pub = self.create_publisher(
            PoseStamped,
            "/hloc/pose",
            10,
        )

        self.get_logger().info(
            "HLOC localization node started"
        )

        self.get_logger().info(
            f"Dataset: {self.dataset_dir}"
        )

        self.get_logger().info(
            f"Reconstruction: {self.reconstruction_dir}"
        )

        self.get_logger().info(
            f"Query topic: {self.query_topic}"
        )

        # ------------------------------------------------------------
        # Prepare HLOC reference data
        # ------------------------------------------------------------

        self.prepare_reference_data()

    # ================================================================
    # Reference-side initialization
    # ================================================================

    def prepare_reference_data(self):

        self.get_logger().info(
            "Preparing reference-side HLOC data..."
        )

        retrieval_conf = extract_features.confs["netvlad"]
        feature_conf = extract_features.confs["superpoint_aachen"]

        # ------------------------------------------------------------
        # Reference NetVLAD descriptors
        #
        # These should ideally be generated once offline.
        # ------------------------------------------------------------

        if not self.reference_descriptors.exists():

            self.get_logger().info(
                "Extracting reference NetVLAD descriptors..."
            )

            extract_features.main(
                retrieval_conf,
                self.reference_images,
                self.work_dir,
            )

        # ------------------------------------------------------------
        # Reference SuperPoint features
        #
        # Also ideally generated once offline.
        # ------------------------------------------------------------

        if not self.reference_features.exists():

            self.get_logger().info(
                "Extracting reference SuperPoint features..."
            )

            extract_features.main(
                feature_conf,
                self.reference_images,
                self.work_dir,
            )

        self.get_logger().info(
            "Reference HLOC data ready."
        )

    # ================================================================
    # CameraInfo
    # ================================================================

    def camera_info_callback(
        self,
        msg: CameraInfo,
    ):

        self.last_camera_info = msg

    # ================================================================
    # Camera image
    # ================================================================

    def image_callback(
        self,
        msg: Image,
    ):

        now = time.monotonic()

        if (
            now - self.last_localization_time
            < 1.0 / self.localization_rate
        ):
            return

        if self.processing:
            return

        self.processing = True
        self.last_localization_time = now

        try:

            self.localize_frame(msg)

        except Exception as exc:

            self.get_logger().error(
                f"HLOC localization failed: {exc}"
            )

        finally:

            self.processing = False

    # ================================================================
    # Single-frame localization
    # ================================================================

    def localize_frame(
        self,
        msg: Image,
    ):

        self.get_logger().info(
            f"Localizing frame at "
            f"{msg.header.stamp.sec}."
            f"{msg.header.stamp.nanosec:09d}"
        )

        # ------------------------------------------------------------
        # Convert ROS image -> OpenCV
        # ------------------------------------------------------------

        image = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding="mono8",
        )

        # ------------------------------------------------------------
        # Save temporary query image
        # ------------------------------------------------------------

        query_name = (
            f"frame_"
            f"{msg.header.stamp.sec}_"
            f"{msg.header.stamp.nanosec:09d}.jpg"
        )

        query_path = self.query_dir / query_name

        cv2.imwrite(
            str(query_path),
            image,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                self.get_parameter("jpeg_quality").value,
            ],
        )

        # ------------------------------------------------------------
        # Local feature extraction
        # ------------------------------------------------------------

        feature_conf = extract_features.confs[
            "superpoint_aachen"
        ]

        query_features = extract_features.main(
            feature_conf,
            self.query_dir,
            self.work_dir,
        )

        # ------------------------------------------------------------
        # Global descriptor
        # ------------------------------------------------------------

        retrieval_conf = extract_features.confs["netvlad"]

        query_global = extract_features.main(
            retrieval_conf,
            self.query_dir,
            self.work_dir,
        )

        # ------------------------------------------------------------
        # Retrieval
        # ------------------------------------------------------------

        pairs_path = (
            self.work_dir /
            f"pairs_{query_name}.txt"
        )

        pairs_from_retrieval.main(
            query_global,
            pairs_path,
            num_matched=self.retrieval_count,
            db_prefix="",
            query_prefix="",
        )

        # ------------------------------------------------------------
        # Match query against retrieved references
        # ------------------------------------------------------------

        matcher_conf = match_features.confs["superglue"]

        matches = match_features.main(
            matcher_conf,
            pairs_path,
            query_features,
            self.work_dir,
        )

        # ------------------------------------------------------------
        # Localization
        # ------------------------------------------------------------

        result_path = (
            self.work_dir /
            f"pose_{query_name}.txt"
        )

        localize_sfm.main(
            self.reconstruction_dir,
            query_path,
            pairs_path,
            query_features,
            matches,
            result_path,
            covisibility_clustering=False,
        )

        # ------------------------------------------------------------
        # Parse pose
        # ------------------------------------------------------------

        pose = self.read_hloc_pose(
            result_path,
            query_name,
        )

        if pose is None:

            self.get_logger().warn(
                "Frame could not be localized."
            )

            return

        q_wxyz, t = pose

        # ------------------------------------------------------------
        # Convert HLOC/COLMAP world->camera pose to
        # camera position/orientation in reconstruction frame.
        # ------------------------------------------------------------

        q_xyzw = np.array(
            [
                q_wxyz[1],
                q_wxyz[2],
                q_wxyz[3],
                q_wxyz[0],
            ],
            dtype=float,
        )

        R_cw = Rotation.from_quat(
            q_xyzw
        ).as_matrix()

        R_wc = R_cw.T

        camera_position = -R_wc @ t

        camera_quat_xyzw = (
            Rotation.from_matrix(
                R_wc
            ).as_quat()
        )

        # ------------------------------------------------------------
        # Publish ROS pose
        # ------------------------------------------------------------

        pose_msg = PoseStamped()

        pose_msg.header = Header()
        pose_msg.header.stamp = msg.header.stamp

        # This is intentionally NOT "map" or "odom".
        #
        # It is the coordinate system of the HLOC/COLMAP
        # reconstruction.
        #
        pose_msg.header.frame_id = "reconstruction"

        pose_msg.pose.position.x = float(
            camera_position[0]
        )

        pose_msg.pose.position.y = float(
            camera_position[1]
        )

        pose_msg.pose.position.z = float(
            camera_position[2]
        )

        pose_msg.pose.orientation.x = float(
            camera_quat_xyzw[0]
        )

        pose_msg.pose.orientation.y = float(
            camera_quat_xyzw[1]
        )

        pose_msg.pose.orientation.z = float(
            camera_quat_xyzw[2]
        )

        pose_msg.pose.orientation.w = float(
            camera_quat_xyzw[3]
        )

        self.pose_pub.publish(
            pose_msg
        )

        self.get_logger().info(
            "Published localized pose."
        )

    # ================================================================
    # Parse HLOC pose file
    # ================================================================

    def read_hloc_pose(
        self,
        path: Path,
        query_name: str,
    ):

        if not path.exists():
            return None

        with open(path, "r") as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                fields = line.split()

                if len(fields) < 8:
                    continue

                image_name = fields[0]

                if (
                    Path(image_name).name
                    != query_name
                ):
                    continue

                qw = float(fields[1])
                qx = float(fields[2])
                qy = float(fields[3])
                qz = float(fields[4])

                tx = float(fields[5])
                ty = float(fields[6])
                tz = float(fields[7])

                return (
                    np.array(
                        [qw, qx, qy, qz],
                        dtype=float,
                    ),
                    np.array(
                        [tx, ty, tz],
                        dtype=float,
                    ),
                )

        return None


def main(args=None):

    rclpy.init(args=args)

    node = HlocLocalizationNode()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
