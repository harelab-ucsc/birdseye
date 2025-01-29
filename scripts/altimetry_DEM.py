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
from cv_bridge import CvBridge
import cv2
import os
import json
import yaml
import time
import copy
from pyproj import Proj, Transformer


class BagProcessor:
    def __init__(self, input_bag_path, ds_dir, radalt_topic, ins_topic):
        self.input_bag_path = input_bag_path
        self.radalt_topic = radalt_topic
        self.ins_topic = ins_topic
        self.ds_dir = ds_dir
        self.count = 0
        self.radalt = None
        self.quat = None


    def load_intrinsics(self, intrinsics_path):
        """Load camera intrinsics from a YAML file."""
        with open(intrinsics_path, "r") as file:
            print('loading intrinsics')
            return yaml.safe_load(file)


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
        print(f'  time mismatch across {len(rad)} radalt messages is {sum([i[0]-i[1] for i in tmp])/1e9}s')
        return pairs


    def correct_altitude(self, rad, ins):
        self.quat = [ins.qn2b[1], ins.qn2b[2], ins.qn2b[3], ins.qn2b[0]]
        eulers = quat2euler(self.quat)
        cos_theta = math.cos(eulers[0]) * math.cos(eulers[1])

        # Compute corrected altitude
        self.radalt = rad.altitude * cos_theta
        if self.radalt < 0:
            self.radalt *= -1


    def process_bag(self):
        # Initialize reader and writer
        reader = SequentialReader()
        storage_options = StorageOptions(uri=self.input_bag_path, storage_id="mcap")
        converter_options = ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")
        reader.open(storage_options, converter_options)
        topics_and_types = reader.get_all_topics_and_types()

        topic_type_map = {t.name:t.type for t in topics_and_types}

        radalt_msgs = []
        ins_msgs = []

        print('reading bag')
        # Read and process messages
        while reader.has_next():
            topic, data, timestamp = reader.read_next()
            message_type = get_message(topic_type_map[topic])
            msg = deserialize_message(data, message_type)

            if topic == self.radalt_topic:
                radalt_msgs.append(msg)
            elif topic == self.ins_topic:
                ins_msgs.append(msg)
        print(f'  radalt_msgs length: {len(radalt_msgs)}')
        print(f'  ins_msgs length: {len(ins_msgs)}')
        print('bag read done \n')

        print('starting timeseries alignment')
        start = time.time()
        pairs = self.match_pairs(radalt_msgs, ins_msgs)
        print(f'  took {time.time() - start}s')


def main():
    parser = argparse.ArgumentParser(description="Fix image timestamps in a ROS2 bag file using INS messages.")
    parser.add_argument("input_bag", help="Path to the input ROS2 bag file")
    parser.add_argument("ds_dir",  help="Path to the directory to save images/ and poses.json to")
    parser.add_argument("radalt_topic", help="Radar Altimeter topic name (e.g., /radalt/data)")
    parser.add_argument("ins_topic", help="INS topic name (e.g., /ins/data)")

    args = parser.parse_args()

    rclpy.init()

    processor = BagProcessor(args.input_bag, args.ds_dir, args.radalt_topic, args.ins_topic)
    processor.process_bag()

    rclpy.shutdown()


if __name__ == "__main__":
    main()
