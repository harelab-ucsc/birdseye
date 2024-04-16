#!/usr/bin/env python3

import csv
import yaml
import utm
import rclpy
import os
# import shutil
import pdb
import cv2
import glob2
import stat
import time
import numpy as np

# from . import simRotTools
# from . import fieldAI
from . import dbConnector
from . import utilities
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

        self.declare_parameter("sensors_yaml", "camera_yamls/birdsEyeSensorParams.yaml")
        self.sensors_yaml = self.get_parameter("sensors_yaml").value
        self.sensors_yaml = os.path.join(os.path.expanduser('~'), self.sensors_yaml)

        self.declare_parameter("dir_name", 'parsed_flight')
        self.dir_name = self.get_parameter("dir_name").value
        self.dir_name = os.path.join(os.path.expanduser('~'), self.dir_name)
        self.dirCheck()

        # db connector
        self.declare_parameter("db_name", 'flight_data')
        self.db_name = self.get_parameter("db_name").value
        self.dbc = dbConnector.dbConnector(os.path.join(self.dir_name, self.db_name))
        self.dbc.boot(self.db_name, self.sensor)

        self.declare_parameter("clicks_csv", "catch/data.csv")
        self.clicks_csv = self.get_parameter("clicks_csv").value
        self.clicks_csv = os.path.join(os.path.expanduser('~'), self.clicks_csv)
        self.csv_read()

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


    def clear_dir(self):
        try:
            files = glob2.glob(os.path.join(self.dir_name, '*'))
            if len(files) >= 1:
                for file in files:
                    if os.path.isfile(file):
                        os.remove(file)
                self.get_logger().info(f"All files in {self.dir_name} deleted successfully.")
            else:
                self.get_logger().info(f"No files in {self.dir_name}.")
        except Exception as e:
            self.get_logger().info(f"Error occurred while clearing {self.dir_name} files: {e}.")


    def csv_read(self):
        self.get_logger().info('Reading clicks CSV file...')
        data = []
        with open(self.clicks_csv) as clicks:
            reader = csv.reader(clicks)
            for line in reader:
                # breakdown line
                u = utm.from_latlon(float(line[0]), float(line[1]))
                tag = int(line[-1][-1])
                data.append([u[0], u[1], tag])
        self.dbc.insertClicks(f"clicks_{self.db_name}", data)


    def calibUptake(self):
        print('Reading sensor parameters YAML file...')
        devices = [f'{self.sensor}', 'imu', 'ublox', 'radalt']
        res = None
        intr1 = None
        intr2 = None
        extr = None
        with open(self.sensors_yaml, 'r') as f:
            params = yaml.safe_load(f)
            for device in devices:
                data = params[device]
                if device == self.sensor:
                    self.res = data["resolution"]
                    self.K = data["intrinsics"]
                    self.dist = data["distortion_coeffs"]
                    self.extr = data["T_cam_imu"]  # extrinsics relative to imu base link
                    self.extr = utilities.matrix_list_converter(self.extr, (4,4))
                    res = self.res
                    intr1 = self.K
                    intr2 = self.dist
                    extr = self.extr
                    # print(extr)
                    # print(type(data["T_cam_imu"]))
                    self.putParameters(device, res, intr1, intr2, extr)
                elif device == 'imu':
                    intr1 = [data["accelerometer_noise_density"], data["accelerometer_random_walk"]]
                    intr2 = [data["gyroscope_noise_density"],  data["gyroscope_random_walk"]]
                    self.putParameters(device, res, intr1, intr2, extr)
                elif device == 'ublox':
                    extr = data["T_ubl_imu"]
                    self.putParameters(device, res, intr1, intr2, utilities.matrix_list_converter(extr, (4,4)))
                elif device == 'radalt':
                    extr = data["T_rad_imu"]
                    self.putParameters(device, res, intr1, intr2, utilities.matrix_list_converter(extr, (4,4)))
                res = None
                intr1 = None
                intr2 = None
                extr = None


    def putParameters(self, device_key, resolution, intrinsics1, intrinsics2, extrinsics):
        vals = '"'
        cols = "sensorID, resolution, intrinsics1, intrinsics2, extrinsics"
        valsList = [device_key, resolution, intrinsics1, intrinsics2, extrinsics]
        vals += '","'.join([str(x) for x in valsList])
        vals += '"'
        self.dbc.insertIgnoreInto(f"parameters_{self.db_name}", cols, vals)


    def getParameters(self, device_key):
        params = []
        cols = "sensorID, resolution, intrinsics1, intrinsics2, extrinsics"
        table = f"parameters_{self.db_name}"
        ret = self.dbc.getFrom(cols, table, cond=f'WHERE sensorID = "{device_key}"')
        # print(ret)
        for elem in ret:
            # print(len(elem))
            for i, item in enumerate(elem):
                # print(i, item)
                if item == device_key:
                    params.append(item)
                elif item != 'None':
                    tmp = utilities.string_list_converter(item)
                    # print('tmp: ', tmp)
                    if item == elem[-1]:
                        tmp = utilities.matrix_list_converter(tmp, (4,4))
                    # print(tmp)
                    params.append(tmp)
        return params


    # TODO: for a later day, add parameter set callback
    # def parameter_callback(self, params):
    #     for param in params:
    #         if param.name == 'my_str' and param.type_ == Parameter.Type.STRING:
    #             self.sensor = param.value
    #     return SetParametersResult(successful=True)


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
        # else:
        if self.att is None:
            self.get_logger().info('ublox missed... no IMU data')
        elif self.alt is None:
            self.get_logger().info('ublox missed... no radar altimeter data')
        else:
            if self.rtk_fix != 131:
                self.get_logger().info(f'Bad pose recorded... no RTK fix: {self.rtk_fix} should be 131')
            sec = str(msg.header.stamp.sec)
            nsec = str(msg.header.stamp.nanosec).rjust(9,str(0))
            time = f'{sec}.{nsec}'
            u = utm.from_latlon(msg.latitude, msg.longitude)
            cols = "x, y, z, q, u, a, t, rtk_fix, rtk_time, alt_time, imu_time"
            tmp = [u[0], u[1], 0.0, 1.0]
            # self.get_logger().info(str(tmp))
            tmp = np.matmul(self.getParameters('ublox')[-1], tmp)
            # self.get_logger().info(str(tmp))
            valsList = [tmp[0], tmp[1], self.alt, \
                        self.att.x, self.att.y, self.att.z, self.att.w, \
                        self.rtk_fix, time, self.alt_time, self.imu_time]
            vals = ','.join([str(x) for x in valsList])
            self.dbc.insertInto(f"{self.sensor}_poses_{self.db_name}", cols, vals)
            self.att = None  # flush used attitude
            self.imu_time = None
            self.alt = None  # flush used altitude
            self.alt_time = None


    def imu_cb(self, msg: Imu):
        # 100Hz update rate, so 'catch freshest' approach is allowed; faster than RTK
        sec = str(msg.header.stamp.sec)
        nsec = str(msg.header.stamp.nanosec).rjust(9,'0')
        time = f'{sec}.{nsec}'
        self.att = msg.orientation  # may need to swap elements for unified coordinate system
        self.imu_time = time


    def alt_cb(self, msg: AltSNR):
        # 100Hz update rate, so 'catch freshest' approach is allowed again
        if msg.snr > 13:  # from device manual: "Altitude measurements associated with a SNR value of 13dB or lower are considered erroneous."
            self.alt = [0.0, 0.0, msg.altitude, 1.0]
            # next line transforms radalt measurements to camera frame
            self.alt = np.matmul(self.getParameters('radalt')[-1], self.alt)[2]
            sec = str(msg.header.stamp.sec)
            nsec = str(msg.header.stamp.nanosec).rjust(9,'0')
            time = f'{sec}.{nsec}'
            self.alt_time = time
        else:
            self.get_logger().info(f'Radar altimeter data not recorded; SNR = {msg.snr}')


def main(args=None):
    rclpy.init(args=args)
    sub_node = subscriberNode()
    rclpy.spin(sub_node)
    sub_node.destroy_node()
    rclpy.shutdown()
