import numpy as np
import csv
import utm
import rclpy
import os
import shutil
import pdb
import cv2
import glob2
import stat
import time

# from . import simRotTools
# from . import fieldAI
from . import dbConnector
# from utilities import *
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Imu, Image, NavSatFix
from std_msgs.msg import String
from custom_msgs.msg import AltSNR
from ublox_msgs.msg import NavPVT

import rclpy.node
from rclpy.exceptions import ParameterNotDeclaredException
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult

#TODO: Get the subscriber working with correct time stamps for flight table
# db_name = 'flight_data'
clicks_csv = None


class subscriberNode(rclpy.node.Node):
    def __init__(self):
        # node init
        super().__init__('flight_data_sub')
        self.declare_parameter("sensorID", rclpy.Parameter.Type.STRING)
        self.sensor = self.get_parameter("sensorID").value

        self.declare_parameter("sensors_yaml", rclpy.Parameter.Type.STRING)
        self.sensors_yaml = self.get_parameter("sensors_yaml").value

        self.declare_parameter("dir_name", 'parsed_flight')
        self.dir_name = self.get_parameter("dir_name").value
        self.dir_name = os.path.join(os.path.expanduser('~'), self.dir_name)
        self.dirCheck()

        # db connector
        self.declare_parameter("db_name", 'flight_data')
        self.db_name = self.get_parameter("db_name").value
        self.dbc = dbConnector.dbConnector(os.path.join(self.dir_name, self.db_name))
        self.dbc.boot(self.db_name, self.sensor)

        os.chmod(os.path.join(self.dir_name, self.db_name+'.db'), stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        time.sleep(1)

        # fast data streams, relative to RTK
        self.alt = None
        self.att = None
        self.rtk_fix = 0
        self.alt_time = None
        self.imu_time = None

        # camera calibration and projection parameters
        self.K = None
        self.dist = None
        self.res = None
        self.extr = None
        self.calibUptake()

        self.br = CvBridge()

        # camera subscriber
        self.cam_sub = self.create_subscription(
            Image, '/image', self.cam_cb, 10)
        # ublox subscribers
        self.ublox_health_sub = self.create_subscription(
            NavPVT, '/gps_flag', self.ublox_health_cb, 10)
        self.ublox_sub = self.create_subscription(
            NavSatFix, '/gps', self.ublox_cb, 10)
        # microstrain subscriber
        self.imu_sub = self.create_subscription(
            Imu, '/imu', self.imu_cb, 10)
        # radalt subscriber
        self.radalt_sub = self.create_subscription(
            AltSNR, '/radalt', self.alt_cb, 10)


    def dirCheck(self):
        if not os.path.isdir(self.dir_name):
            self.get_logger().info(f"{self.dir_name} does not exist in home dir... Generating.")
            try:
                os.mkdir(self.dir_name)
            except FileExistsError:
                self.get_logger().info(f"{self.dir_name} exists now... Someone beat me to it.")
        else:
            self.get_logger().info(f"{self.dir_name} exists...")
            self.clear_dir()
        time.sleep(1)


    def calibUptake(self):
        devices = [f'{self.sensor}', 'imu', 'ublox', 'radalt']
        res = np.array([None])
        intr1 = np.array([None])
        intr2 = np.array([None])
        extr = np.array([None])
        with open(self.sensors_yaml, 'r') as f:
            params = self.load_yaml()
            for device in devices:
                data = params[device]
                if device == self.sensor:
                    self.res = data["resolution"]
                    self.K = data["intrinsics"]
                    self.dist = data["distortion_coeffs"]
                    self.extr = data["T_cam_imu"]  # extrinsics relative to imu base link
                    res = self.res
                    intr1 = self.K
                    intr2 = self.dist
                    extr = self.extr
                elif device == 'imu':
                    intr1 = np.array([data["accelerometer_noise_density"], \
                                    data["accelerometer_random_walk"]])
                    intr2 = np.array([data["gyroscope_noise_density"], \
                                    data["gyroscope_random_walk"]])
                    extr = np.eye(4)
                elif device is 'ublox':
                    extr = np.array(data["T_ubl_imu"])
                elif device is 'radalt':
                    extr = np.array(data["T_rad_imu"])
                self.putParameters(device, res, intr1, intr2, extr)
                res = np.array([None])
                intr1 = np.array([None])
                intr2 = np.array([None])
                extr = np.array([None])


    def putParameters(self, device_key, resolution, intrinsics1, intrinsics2, extrinsics):
        cols = "sensorID, resolution, intrinsics1, intrinsics2, extrinsics"
        valsList = [device_key, resolution, intrinsics1, intrinsics2, extrinsics]
        vals = ','.join([str(x) for x in valsList])
        self.dbc.insertIgnoreInto(f"parameters_{self.db_name}", cols, vals)


    # TODO: for a later day, add parameter set callback
    # def parameter_callback(self, params):
    #     for param in params:
    #         if param.name == 'my_str' and param.type_ == Parameter.Type.STRING:
    #             self.sensor = param.value
    #     return SetParametersResult(successful=True)


    def clear_dir(self):
        try:
            files = glob2.glob(os.path.join(self.dir_name, '*'))
            if len(files) > 1:
                for file in files:
                    if os.path.isfile(file):
                        os.remove(file)
                self.get_logger().info(f"All files in {self.dir_name} deleted successfully.")
            else:
                self.get_logger().info(f"No files in {self.dir_name}.")
        except Exception as e:
            self.get_logger().info(f"Error occurred while clearing {self.dir_name} files: {e}.")


    def cam_cb(self, msg: Image):
        # put data into db
        sec = str(msg.header.stamp.sec)
        nsec = str(msg.header.stamp.nanosec).rjust(9,str(0))
        time = f'{sec}.{nsec}'
        data_loc = self.dir_name + "/" + self.sensor + '_' + time + ".png"
        image = self.br.imgmsg_to_cv2(msg, desired_encoding='passthrough')

        #TODO: rectify the images before saving

        cv2.imwrite(data_loc, image)
        valsList = ['\"'+data_loc+'\"', self.rtk_fix, time]
        vals = ','.join([str(x) for x in valsList])
        self.dbc.insertInto(f"{self.sensor}_images_{self.db_name}", "save_loc, rtk_fix, time", vals)


    # the ublox subscribers in this structure is more error-prone,
    # vs having one message with std_sgs/Heading, NavPVT, and NavSatFix members
    # because the health flag and the rtk information are not currently
    # forced to be synchronized as 2 separate messages - MWM, 2024/02/02
    def ublox_health_cb(self, msg: NavPVT):
        self.rtk_fix = msg.flags


    def ublox_cb(self, msg: NavSatFix):
        if self.rtk_fix != 131:
            self.get_logger().info('ublox missed; no RTK fix')
        else:
            if self.att is None:
                self.get_logger().info('ublox missed; no IMU data')
            elif self.alt is None:
                self.get_logger().info('ublox missed; no radar altimeter data')
            else:
                sec = str(msg.header.stamp.sec)
                nsec = str(msg.header.stamp.nanosec).rjust(9,str(0))
                time = f'{sec}.{nsec}'
                u = utm.from_latlon(msg.latitude, msg.longitude)
                cols = "x, y, z, q, u, a, t, rtk_time, alt_time, imu_time"
                valsList = [u[0], u[1], self.alt, \
                            self.att.x, self.att.y, self.att.z, self.att.w, \
                            time, self.alt_time, self.imu_time]
                vals = ','.join([str(x) for x in valsList])
                self.dbc.insertInto(f"{self.sensor}_poses_{self.db_name}", cols, vals)
                self.att = None  # flush used attitude
                self.imu_time = None
                self.alt = None  # flush used altitude
                self.alt_time = None


    def imu_cb(self, msg: Imu):
        # microstrain -> class attribute;
        # 'catch freshest' approach is allowed because IMU is faster than RTK
        sec = str(msg.header.stamp.sec)
        nsec = str(msg.header.stamp.nanosec).rjust(9,'0')
        time = f'{sec}.{nsec}'
        self.att = msg.orientation  # may need to swap elements for unified coordinate system
        self.imu_time = time


    def alt_cb(self, msg: AltSNR):
        # rad alt -> flight z
        # 100Hz update rate, so 'catch freshest' approach is allowed again
        if msg.snr > 13:  # from device manual: "Altitude measurements associated with a SNR value of 13dB or lower are considered erroneous."
            self.alt = msg.altitude
            sec = str(msg.header.stamp.sec)
            nsec = str(msg.header.stamp.nanosec).rjust(9,'0')
            time = f'{sec}.{nsec}'
            self.alt_time = time
        else:
            self.get_logger().info(f'Radar altimeter data not recorded; SNR = {msg.snr}')


    def parametersToDB(self, sensorID, type):
        """
        sensorID: string referring to the sensor  whose intr/extrinsics are being written to database
        type: string, 'intrinsics' or 'extrinsics'
        """
        pass


def main(args=None):
    rclpy.init(args=args)
    sub_node = subscriberNode()
    rclpy.spin(sub_node)
    sub_node.destroy_node()
    rclpy.shutdown()
