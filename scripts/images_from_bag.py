#!/usr/bin/env python3.11
import pdb
import rclpy
from rclpy.node import Node
from rosbag2_py import SequentialReader, SequentialWriter, StorageOptions, ConverterOptions
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

#from rectify import rectify_image


class BagProcessor:
    def __init__(self, input_bag_path, ds_dir, image_topic):
        self.input_bag_path = input_bag_path
        self.image_topic = image_topic
        self.ds_dir = ds_dir

        self.image_msgs = []

        print(self.input_bag_path)
        print(self.image_topic)
        print(self.ds_dir)


    def load_intrinsics(self, intrinsics_path):
        """Load camera intrinsics from a YAML file."""
        with open(intrinsics_path, "r") as file:
            print('loading intrinsics')
            return yaml.safe_load(file)


    def process_bag(self):
        # Initialize reader and writer
        reader = SequentialReader()
        try:
            storage_options = StorageOptions(uri=self.input_bag_path, storage_id="mcap")
            converter_options = ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")
            reader.open(storage_options, converter_options)
        except RuntimeError:
            storage_options = StorageOptions(uri=self.input_bag_path, storage_id="sqlite3")
            converter_options = ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")
            reader.open(storage_options, converter_options)

        topics_and_types = reader.get_all_topics_and_types()

        topic_type_map = {t.name:t.type for t in topics_and_types}

        print('reading bag')
        # Read and process messages
        while reader.has_next():
            topic, data, timestamp = reader.read_next()
            message_type = get_message(topic_type_map[topic])
            msg = deserialize_message(data, message_type)

            if topic == self.image_topic:
                self.image_msgs.append(msg)

        print(f'  image_msgs length: {len(self.image_msgs)}')
        print('bag read done \n')

        for img in self.image_msgs:
            timestamp_str = f"{img.header.stamp.sec}.{img.header.stamp.nanosec:09d}"
            # print(timestamp_str)
            self.save_image(img, timestamp_str)


    def save_image(self, image_msg, timestamp_str):
        """Save the image message as a PNG file."""
        img_data = np.frombuffer(image_msg.data, dtype=np.uint8).reshape(image_msg.height, image_msg.width, -1)
        savename = os.path.join(self.ds_dir, 'images')
        if not os.path.isdir(savename):
            print(f'  Making Save Directory: {savename}')
            os.makedirs(savename, exist_ok=True)

        savename = os.path.join(savename, f"{timestamp_str}.png")
#        print(f"  Saving Image To: {savename}")
        cv2.imwrite(savename, img_data)


def main():
    parser = argparse.ArgumentParser(description="Fix image timestamps in a ROS2 bag file using INS messages.")
    parser.add_argument("input_bag", help="Path to the input ROS2 bag file")
    parser.add_argument("ds_dir",  help="Path to the directory to save images/ and poses.json to")
    parser.add_argument("image_topic", help="Image topic name (e.g., /camera/image_raw)")

    args = parser.parse_args()

    rclpy.init()

    processor = BagProcessor(args.input_bag, args.ds_dir, args.image_topic)
    processor.process_bag()

    rclpy.shutdown()


if __name__ == "__main__":
    main()
