#!/usr/bin/env python3.11
import pdb
import rclpy
from rclpy.node import Node
from rosbag2_py import (
    SequentialReader,
    SequentialWriter,
    StorageOptions,
    ConverterOptions,
)
from sensor_msgs.msg import Image
from inertial_sense_ros2.msg import DIDINS2
import argparse
import numpy as np
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message, serialize_message
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from cv_bridge import CvBridge
import cv2
import os
import json
import yaml
import copy
from pyproj import Proj, Transformer
import matplotlib.pyplot as plt

# from rectify import rectify_image


class BagProcessor:
    def __init__(self, input_bag_path, ds_dir, image_topic, camera=0):
        self.input_bag_path = input_bag_path
        self.image_topic = image_topic
        self.ds_dir = ds_dir
        print(self.input_bag_path)
        print(self.image_topic)
        print(self.ds_dir)

        self.sum = None
        self.count = 0
        self.camera = camera

        self.bridge = CvBridge()

    def load_intrinsics(self, intrinsics_path):
        """Load camera intrinsics from a YAML file."""
        with open(intrinsics_path, "r") as file:
            print("loading intrinsics")
            return yaml.safe_load(file)

    def process_bag(self):
        # Initialize reader and writer
        reader = SequentialReader()
        try:
            storage_options = StorageOptions(uri=self.input_bag_path, storage_id="mcap")
            converter_options = ConverterOptions(
                input_serialization_format="cdr", output_serialization_format="cdr"
            )
            reader.open(storage_options, converter_options)
        except RuntimeError:
            storage_options = StorageOptions(
                uri=self.input_bag_path, storage_id="sqlite3"
            )
            converter_options = ConverterOptions(
                input_serialization_format="cdr", output_serialization_format="cdr"
            )
            reader.open(storage_options, converter_options)

        topics_and_types = reader.get_all_topics_and_types()

        topic_type_map = {t.name: t.type for t in topics_and_types}

        print("reading bag")
        # Read and process messages
        while reader.has_next():
            topic, data, timestamp = reader.read_next()
            message_type = get_message(topic_type_map[topic])
            msg = deserialize_message(data, message_type)

            if topic == self.image_topic:
                self.accumulate(msg)

        print(f"Processed {self.count} frames - bag read done.")
        self.save_ffc()
        print(f'Flat-Field Corrections saved to: {self.ds_dir}')

    def accumulate(self, image_msg, roi=None):
        img = self.bridge.imgmsg_to_cv2(
            image_msg,
            desired_encoding="passthrough"
        ).astype(np.float32)
        img = camera_roi(img, self.camera)
        img /= np.mean(img)

        if self.sum is None:
            self.sum = np.zeros_like(img, dtype=np.float32)

        self.sum += img
        self.count += 1

    def make_ffc(self, sigma=64):
        mean = self.sum / self.count

        # Split BGGR Bayer planes
        B  = mean[0::2, 0::2].copy()
        G1 = mean[0::2, 1::2].copy()
        G2 = mean[1::2, 0::2].copy()
        R  = mean[1::2, 1::2].copy()

        # Strong low-pass filter to estimate only illumination/vignetting
        B  = cv2.GaussianBlur(B,  (0, 0), sigma)
        G1 = cv2.GaussianBlur(G1, (0, 0), sigma)
        G2 = cv2.GaussianBlur(G2, (0, 0), sigma)
        R  = cv2.GaussianBlur(R,  (0, 0), sigma)

        # Normalize each plane independently
        B_gain  = np.mean(B)  / (B  + 1e-6)
        G1_gain = np.mean(G1) / (G1 + 1e-6)
        G2_gain = np.mean(G2) / (G2 + 1e-6)
        R_gain  = np.mean(R)  / (R  + 1e-6)

        # Prevent pathological gains
        for gain in (B_gain, G1_gain, G2_gain, R_gain):
            np.clip(gain, 0.5, 2.0, out=gain)

        # Reassemble Bayer gain image
        gain = np.empty_like(mean, dtype=np.float32)
        gain[0::2, 0::2] = B_gain
        gain[0::2, 1::2] = G1_gain
        gain[1::2, 0::2] = G2_gain
        gain[1::2, 1::2] = R_gain

        # Assemble smoothed flat field itself
        flat = np.empty_like(mean, dtype=np.float32)
        flat[0::2, 0::2] = B
        flat[0::2, 1::2] = G1
        flat[1::2, 0::2] = G2
        flat[1::2, 1::2] = R

        print(gain.shape)

        return gain, flat

    def save_ffc(self):
        gain, flat = self.make_ffc()
        os.makedirs(self.ds_dir, exist_ok=True)
        cv2.imwrite(
            os.path.join(self.ds_dir, f"ffc_mean_cam{self.camera}.png"),
            flat.astype(np.uint16)
        )
        np.save(
            os.path.join(self.ds_dir, f"ffc_gain_cam{self.camera}.npy"),
            gain.astype(np.float32)
        )
        vis = gain / gain.max()
        cv2.imwrite(
            os.path.join(self.ds_dir, f"ffc_gain_visualization_cam{self.camera}.png"),
            (255 * vis).astype(np.uint8)
        )


def camera_roi(img, camera):
    """
    Extract one camera from a 1x4 Arducam CamArray frame.

    camera:
        0 = leftmost
        1
        2
        3 = rightmost
    """
    if camera not in (0, 1, 2, 3):
        raise ValueError(f"camera must be 0-3, got {camera}")

    roi = (
        1280 * camera,  # x
        0,              # y
        1280,           # width
        800             # height
    )
    return extract_roi(img, roi)


def extract_roi(img, roi=None):
    """
    img : HxW or HxWxC image
    roi : (x, y, w, h) or None
    """
    if roi is None:
        return img

    x, y, w, h = roi
    return img[y:y+h, x:x+w]


def main():
    parser = argparse.ArgumentParser(
        description="Generate Flat-Field Correction parameters from a ROS2 bag."
        " For use on BGGR16-native imagery."
    )
    parser.add_argument(
        "input_bag",
        help="Path to the input ROS2 bag file"
    )
    parser.add_argument(
        "ds_dir", help="Path to the directory to save images/ and poses.json to"
    )
    parser.add_argument(
        "image_topic", help="Image topic name (e.g., /camera/image_raw)"
    )
    parser.add_argument(
        "camera", help="Index of the target camera frame from Arducam CamArray HAT (0, 1, 2, 3)"
    )

    args = parser.parse_args()
    rclpy.init()
    processor = BagProcessor(args.input_bag, args.ds_dir, args.image_topic, int(args.camera))
    processor.process_bag()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
