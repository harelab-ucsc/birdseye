#!/usr/bin/env python3
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
    def __init__(self, input_bag_path, output_bag_path, ds_dir, image_topic, ins_topic, intrinsics_path, rectify, sync):
        self.input_bag_path = input_bag_path
        self.output_bag_path = output_bag_path
        self.image_topic = image_topic
        self.ins_topic = ins_topic
        self.ds_dir = ds_dir
        self.fps = 5

        self.deltas = []
        self.br = CvBridge()
        self.frames = []

        fig, ax = plt.subplots()
        self.fig = fig
        self.ax = ax
        self.ref = None

        self.intrinsics = self.load_intrinsics(intrinsics_path)
        self.K = np.array([[self.intrinsics["fx"], 0, self.intrinsics["cx"]],
                      [0, self.intrinsics["fy"], self.intrinsics["cy"]],
                      [0, 0, 1]])
        self.D = np.array([self.intrinsics["k1"], self.intrinsics["k2"], self.intrinsics["r1"], self.intrinsics["r2"]])
        self.width = self.intrinsics["resx"]
        self.height = self.intrinsics["resy"]
        self.map1, self.map2 = cv2.initUndistortRectifyMap(self.K, self.D, None, self.K, (self.width, self.height), cv2.CV_32FC1)

        self.rectify = rectify
        self.sync = sync

        self.count = 0
        self.strb = 0

        # Initialize UTM transformer
        self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:32610", always_xy=True)  # Replace EPSG:32633 with appropriate UTM zone

        self.image_msgs = []
        self.ins_msgs = []
        self.paired_flags = None


    def load_intrinsics(self, intrinsics_path):
        """Load camera intrinsics from a YAML file."""
        with open(intrinsics_path, "r") as file:
            print('loading intrinsics')
            return yaml.safe_load(file)


    def process_bag(self):
        # Initialize reader and writer
        reader = SequentialReader()
        writer = SequentialWriter()
        try:
            storage_options = StorageOptions(uri=self.input_bag_path, storage_id="mcap")
            converter_options = ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")
            reader.open(storage_options, converter_options)
        except RuntimeError:
            storage_options = StorageOptions(uri=self.input_bag_path, storage_id="sqlite3")
            converter_options = ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")
            reader.open(storage_options, converter_options)
        writer.open(StorageOptions(uri=self.output_bag_path, storage_id="mcap"), converter_options)

        topics_and_types = reader.get_all_topics_and_types()

        topic_type_map = {t.name:t.type for t in topics_and_types}
        # Register all topics with the writer
        for topic in topics_and_types:
            writer.create_topic(topic)

        # Ensure output directories exist
        os.makedirs("images", exist_ok=True)

        print('reading bag')
        # Read and process messages
        while reader.has_next():
            topic, data, timestamp = reader.read_next()
            message_type = get_message(topic_type_map[topic])
            msg = deserialize_message(data, message_type)

            if topic == self.image_topic:
                self.image_msgs.append(msg)
            elif topic == self.ins_topic:
                self.ins_msgs.append(msg)
                writer.write(topic, data, timestamp)
            else:
                # Copy all other topics as-is
                writer.write(topic, data, timestamp)
        self.paired_flags = [0]*len(self.image_msgs)
        print(f'image_msgs length: {len(self.image_msgs)}')
        print(f'ins_msgs length: {len(self.ins_msgs)} \n')
        print('bag read done \n')

        # Process INS messages and adjust image timestamps
        print('starting timeseries alignment')
        HDW_STATUS_STROBE_IN_EVENT = 0x00000020
        for ins_msg in self.ins_msgs:
            self.count += 1
            if ins_msg.hdw_status & HDW_STATUS_STROBE_IN_EVENT == HDW_STATUS_STROBE_IN_EVENT:
                self.strb += 1
                print(f'  found strobe-triggered INS2 {self.strb} (INS msg {self.count})', end='\r')
                ins_timestamp = ins_msg.header.stamp
                ins_timestamp_int = int(ins_timestamp.sec * 1e9 + ins_timestamp.nanosec)
                closest_image = self.find_closest_image(ins_timestamp)

                if closest_image:
                    if self.strb == 1:
                        self.ref = ins_timestamp_int/1e9
                    # Compute the time difference
                    old_time = closest_image.header.stamp.sec + closest_image.header.stamp.nanosec * 1e-9
                    new_time = ins_timestamp.sec + ins_timestamp.nanosec * 1e-9
                    delta = old_time - new_time
                    self.deltas.append(delta)

                    # Update image timestamp and save the image
                    if self.sync:
                        updated_image = self.update_image_timestamp(closest_image, ins_timestamp)
                    else:
                        updated_image = closest_image
                    if self.rectify:
                        updated_image = self.rectify_image(updated_image)

                    # TODO: updated_image = self.blur_check(updated_image)

                    timestamp_str = f"{ins_timestamp.sec}.{ins_timestamp.nanosec:09d}"
                    self.save_image(updated_image, timestamp_str)

                    # Append pose to JSON
                    self.append_pose_to_json(ins_msg, updated_image, timestamp_str)

                    new_image = serialize_message(updated_image)
                    writer.write(self.image_topic, new_image, ins_timestamp_int)
        deltas = np.array(self.deltas)
        mean = deltas.mean()
        std = deltas.std()
        fig, ax = plt.subplots()
        ax.hist(deltas, bins=150)
        min = plt.ylim()[0]
        max = plt.ylim()[1]
        ax.vlines(mean, ymin=min, ymax=max, colors='r')
        ax.vlines(mean-std, ymin=min, ymax=max, colors='k')
        ax.vlines(mean+std, ymin=min, ymax=max, colors='k')
        plt.savefig(os.path.join(self.ds_dir,'hist.png'))
        plt.show()

        self.save_json()
        writer.close()
        print(f'--> {sum(self.paired_flags)} images of {len(self.image_msgs)} matched')
        print(f'--> time correction mean: {mean} sec, std: {std} sec')


    def find_closest_image(self, target_timestamp):
        """Find the closest image message to the given timestamp."""
        closest_image = None
        min_diff = float("inf")
        ind = None
        tgt = target_timestamp.sec + target_timestamp.nanosec * 1e-9
        for i, image in enumerate(self.image_msgs):
            img = image.header.stamp.sec + image.header.stamp.nanosec * 1e-9
            diff = abs(img - tgt)
            if diff < 1/self.fps:
                if diff < min_diff and self.paired_flags[i] == 0:
                    closest_image = image
                    min_diff = diff
                    ind = i
                    val = img
        if ind is not None:
            self.paired_flags[ind] = 1 # remove image from list (only one pose for every image)
            if self.ref is not None:
                # self.ax.hlines(val, xmin=self.ref, xmax=val, color='xkcd:kelly green', linestyle='dotted')
                self.ax.scatter(tgt, val, s=2, c='xkcd:kelly green')
        else:
            if self.ref is not None:
                self.ax.scatter(tgt, tgt, s=10, c='k', alpha=0.2)
        return closest_image


    # def variance_of_laplacian(self, image):
    # 	# compute the Laplacian of the image and then return the focus
    # 	# measure, which is simply the variance of the Laplacian
    # 	return cv2.Laplacian(image, cv2.CV_64F).var()
    #

    # def blur_check(self, image_msg, thresh=200):
    #     print('    blur check')
    #     header = image_msg.header
    #     image = self.br.imgmsg_to_cv2(image_msg, desired_encoding='passthrough')
    #     image = cv2.cvtColor(image, cv2.COLOR_BAYER_RG2RGB)
    #     gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #     fm = self.variance_of_laplacian(gray)
    #     if fm < thresh:
    #         print(f'      variance of Laplacian is {fm}; deblurring')
    #         image = self.deblur(image)
    #     image = self.make_bayer(image)
    #     image_msg = self.br.cv2_to_imgmsg(image, encoding='bayer_rg8')
    #     image_msg.header = header
    #     return image_msg
    #
    #
    # def deblur(self, blurry_image):
    #     # Define a kernel for motion blur
    #     kernel_size = 15
    #     kernel = np.zeros((kernel_size, kernel_size))
    #     kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    #     kernel /= kernel_size
    #
    #     # Apply the inverse filter
    #     deblurred_image = cv2.filter2D(blurry_image, -1, kernel)
    #
    #     # kernel = kernel.T
    #     # deblurred_image = cv2.filter2D(deblurred_image, -1, kernel)
    #     return deblurred_image
    #
    #
    # def make_bayer(self, rgb_image):
    #     height, width, _ = rgb_image.shape
    #     # Create an empty Bayer image (single channel)
    #     bayer_image = np.zeros((height, width), dtype=np.uint8)
    #
    #     # Extract R, G, B channels
    #     R = rgb_image[:, :, 0]
    #     G = rgb_image[:, :, 1]
    #     B = rgb_image[:, :, 2]
    #
    #     # Simulate Bayer RG pattern
    #     bayer_image[0::2, 0::2] = R[0::2, 0::2]  # Red pixels
    #     bayer_image[0::2, 1::2] = G[0::2, 1::2]  # Green pixels (next to Red)
    #     bayer_image[1::2, 0::2] = G[1::2, 0::2]  # Green pixels (next to Blue)
    #     bayer_image[1::2, 1::2] = B[1::2, 1::2]  # Blue pixels
    #     return bayer_image


    def get_timestamp(self, msg):
        ts = msg.header.stamp
        ts = ts.sec + ts.nanosec*1e-9
        return ts


    def update_image_timestamp(self, image_msg, new_timestamp):
        """Create a new Image message with an updated timestamp."""
        new_image = Image()
        new_image.header = image_msg.header
        new_image.header.stamp = new_timestamp
        new_image.data = image_msg.data
        new_image.height = image_msg.height
        new_image.width = image_msg.width
        new_image.encoding = image_msg.encoding
        new_image.is_bigendian = image_msg.is_bigendian
        new_image.step = image_msg.step
        return new_image


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


    def append_pose_to_json(self, ins_msg, image_msg, timestamp_str):
        """Append the pose data from INS message to the JSON."""
        # Convert quaternion to R matrix
        quat = ins_msg.qn2b
        reordered_quat = [quat[1], quat[2], quat[3], quat[0]]
        rot = R.from_quat(reordered_quat)
        rot = rot.as_matrix()

        # Convert LLA to UTM
        utm_x, utm_y = self.transformer.transform(ins_msg.lla[1], ins_msg.lla[0])  # (longitude, latitude)
        altitude = ins_msg.lla[2]

        # make T vector
        trans = [utm_x, utm_y, altitude]

        # compose world pose of IMX-5
        transform_matrix = np.eye(4)
        transform_matrix[:3,:3] = rot
        transform_matrix[:3,3] = trans

        # compose world pose of BFLY
        # print(len(self.intrinsics["T_cam_imu"]), len(self.intrinsics["T_cam_imu"][0]))
        tf = np.array(self.intrinsics["T_cam_imu"])
        tf = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])@tf  # imu is in ENU, ins is in NED -> tf transforms between these frames
        transform_matrix = transform_matrix@tf
        transform_matrix = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])@transform_matrix  # imu is in ENU, ins is in NED -> tf transforms between these frames
        transform_matrix = transform_matrix.tolist()

        pose = {
            "w": image_msg.width,
            "h": image_msg.height,
            "fl_x": self.intrinsics["fx"],
            "fl_y": self.intrinsics["fy"],
            "cx": self.intrinsics["cx"],
            "cy": self.intrinsics["cy"],
            "timestamp": ins_msg.header.stamp.sec + ins_msg.header.stamp.nanosec * 1e-9,
            "file_path": f"{timestamp_str}.png",
            "transform_matrix": transform_matrix
        }
        self.frames.append(pose)


    def save_json(self):
        """Save all frames to a JSON file."""
        savename = os.path.join(self.ds_dir, "poses.json")
        with open(savename, "w") as json_file:
            print(f'  Saving JSON To: {savename}')
            json.dump({"frames": self.frames}, json_file, indent=4)


    def rectify_image(self, raw_image):
        # Convert raw image message to OpenCV image using rgb8 encoding
        cv_image = self.br.imgmsg_to_cv2(raw_image, desired_encoding='mono8')

        # Rectify the image using the maps
        rectified_image = cv2.remap(cv_image, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)

        # Convert the rectified image back to ROS Image message using rgb8 encoding
        rectified_img_msg = self.br.cv2_to_imgmsg(rectified_image, encoding='mono8')
        rectified_img_msg.header = raw_image.header

        return rectified_img_msg


def main():
    parser = argparse.ArgumentParser(description="Fix image timestamps in a ROS2 bag file using INS messages.")
    parser.add_argument("input_bag", help="Path to the input ROS2 bag file")
    parser.add_argument("output_bag", help="Path to the output ROS2 bag file")
    parser.add_argument("ds_dir",  help="Path to the directory to save images/ and poses.json to")
    parser.add_argument("image_topic", help="Image topic name (e.g., /camera/image_raw)")
    parser.add_argument("ins_topic", help="INS topic name (e.g., /ins/data)")
    parser.add_argument("intrinsics", help="Path to the YAML file with camera intrinsics")
    parser.add_argument("-r", "--rectify", action="store_false", help="Whether or not to rectify in vi_time_sync.py (default: True)")
    parser.add_argument("-s", "--sync", action="store_false", help="Whether or not to perform time synchronization (default: True)")

    args = parser.parse_args()

    rclpy.init()

    processor = BagProcessor(args.input_bag, args.output_bag, args.ds_dir, args.image_topic, args.ins_topic, args.intrinsics, args.rectify, args.sync)
    processor.process_bag()

    rclpy.shutdown()


if __name__ == "__main__":
    main()
