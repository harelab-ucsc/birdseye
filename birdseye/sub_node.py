#!/usr/bin/env python3

import csv
import yaml
import utm
import rclpy
import os
import time
import pdb
import cv2
import glob2
import stat
import time
import sqlite3
import numpy as np

from . import dbConnector
from . import utilities

from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu, NavSatFix
from std_msgs.msg import String
from inertial_sense_ros2.msg import DIDINS2
from custom_msgs.msg import AltSNR

import rclpy.node
from rclpy.exceptions import ParameterNotDeclaredException
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


clicks_csv = None

# Create a custom QoSProfile to prevent message drops
qos_profile = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,  # Reliable delivery
    history=HistoryPolicy.KEEP_ALL           # Keep all messages
)


class subscriberNode(rclpy.node.Node):
    def __init__(self):
        time.sleep(1)
        # node init
        super().__init__('flight_data_sub')
        self.declare_parameter("sensorID", "cam0")
        self.sensor = self.get_parameter("sensorID").value

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
        self.db_bool = True
        time.sleep(1)

        # sensor calibration parameters
        self.declare_parameter("sensors_yaml", "sensor_params/birdsEyeSensorParams.yaml")
        self.sensors_yaml = self.get_parameter("sensors_yaml").value
        self.sensors_yaml = os.path.join(os.path.expanduser('~'), self.sensors_yaml)
        self.calibUptake()

        self.declare_parameter("clicks_csv", "catch/data__2025_01_09.csv")
        # self.declare_parameter("clicks_csv", "catch/data__2025_01_10.csv")
        self.clicks_csv = self.get_parameter("clicks_csv").value
        self.clicks_csv = os.path.join(os.path.expanduser('~'), self.clicks_csv)
        self.csv_read()

        self.br = CvBridge()

        # camera subscriber
        self.cam_sub = self.create_subscription(
            Image, '/image', self.cam_cb, qos_profile=qos_profile)
        self.data_loc = None
        self.image = None
        self.cam_times = None

        # ins subscriber and relevant attributes
        self.ins_sub = self.create_subscription(
            DIDINS2, '/ins', self.ins_cb, 100)
        self.HDW_STATUS_STROBE_IN_EVENT = 0x00000020
        self.INS_STATUS_SOLUTION_MASK = 0x000F0000
        self.INS_STATUS_SOLUTION_OFFSET = 16
        self.INS_STATUS_GPS_NAV_FIX_MASK = 0x03000000
        self.INS_STATUS_GPS_NAV_FIX_OFFSET = 24
        self.pos = None
        self.quat = None
        self.ins_times = None
        self.RTK_STATUS = None
        self.INS_STATUS = None
        self.STROBE = None

        # radalt subscriber
        self.rad_sub = self.create_subscription(
            AltSNR, '/rad_altitude', self.radalt_cb, 100)
        self.radalt = None

        self.check_list = [self.image, \
                           self.pos, \
                           self.quat, \
                           self.RTK_STATUS, \
                           self.INS_STATUS, \
                           self.STROBE, \
                           self.data_loc, \
                           self.ins_times, \
                           self.cam_times, \
                           self.radalt]


    def dirCheck(self):
        if not os.path.isdir(self.dir_name):
            self.get_logger().info(f"{self.dir_name} does not exist in home dir... Generating.")
            try:
                os.makedirs(self.dir_name, exist_ok=True)
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
                self.get_logger().info(f"All files in {self.dir_name} deleted successfully.\n")
            else:
                self.get_logger().info(f"No files in {self.dir_name}.\n")
        except Exception as e:
            self.get_logger().info(f"Error occurred while clearing {self.dir_name} files: {e}.\n")


    def csv_read(self):
        self.get_logger().info(f'Reading clicks CSV file: {self.clicks_csv}...')
        data = []
        with open(self.clicks_csv) as clicks:
            reader = csv.reader(clicks)
            for line in reader:
                # breakdown line
                # self.get_logger().info(f'{line}')
                u = utm.from_latlon(float(line[0]), float(line[1]))
                tag = int(line[-1][-1])
                data.append([u[0], u[1], float(line[2]), float(line[3]), tag])
        self.dbc.insertClicks(f"clicks_{self.db_name}", data)
        self.get_logger().info('...Done reading clicks CSV file.\n')


    def calibUptake(self):
        self.get_logger().info(f'Reading sensor parameters YAML file: {self.sensors_yaml}...')
        devices = [f'{self.sensor}', 'ins', 'radalt']
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
                    self.putParameters(device, res, intr1, intr2, extr)
                elif device == 'ins':
                    intr1 = [data["accelerometer_noise_density"], data["accelerometer_random_walk"]]
                    intr2 = [data["gyroscope_noise_density"],  data["gyroscope_random_walk"]]
                    self.putParameters(device, res, intr1, intr2, extr)
                elif device == 'radalt':
                    extr = data["T_rad_imu"]
                    self.putParameters(device, res, intr1, intr2, utilities.matrix_list_converter(extr, (4,4)))
                res = None
                intr1 = None
                intr2 = None
                extr = None
        self.get_logger().info('...Done reading sensor parameters YAML file.\n')


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
        for elem in ret:
            for i, item in enumerate(elem):
                if item == device_key:
                    params.append(item)
                elif item != 'None':
                    tmp = utilities.string_list_converter(item)
                    if item == elem[-1]:
                        tmp = utilities.matrix_list_converter(tmp, (4,4))
                    params.append(tmp)
        return params


    # TODO: for a later day, add parameter set callback
    # def parameter_callback(self, params):
    #     for param in params:
    #         if param.name == 'my_str' and param.type_ == Parameter.Type.STRING:
    #             self.sensor = param.value
    #     return SetParametersResult(successful=True)


    def update_check_list(self):
        self.check_list = [self.image, \
                           self.pos, \
                           self.quat, \
                           self.RTK_STATUS, \
                           self.INS_STATUS, \
                           self.STROBE, \
                           self.data_loc, \
                           self.ins_times, \
                           self.cam_times, \
                           self.radalt]


    def status_check(self):
        tst = [0 if i is None else 1 for i in self.check_list]
#        self.get_logger().info(f'      status_check: {tst}, {sum(tst)}')
        if sum(tst) == len(self.check_list):
            return True
        else:
            return False


    def save_image_pose(self):
        self.get_logger().info(f'  Recording image and pose... (RTK_STATUS, INS_STATUS): ({self.RTK_STATUS}, {self.INS_STATUS})')
        self.get_logger().info(f'                              (cam_time1, cam_time2): ({self.cam_times[0]}, {self.cam_times[1]})')
        self.get_logger().info(f'                              (ins_time1, ins_time2): ({self.ins_times[0]}, {self.ins_times[1]})')

        try:
            # cv2.resize(image, (0, 0), fx = 0.25, fy = 0.25)
            cv2.imwrite(self.data_loc, self.image)
            valsList = self.pos + self.quat + [self.RTK_STATUS, self.INS_STATUS, self.radalt, '\"'+self.data_loc+'\"', self.cam_times[0], self.cam_times[1], self.ins_times[0], self.ins_times[1]]
            vals = ','.join([str(x) for x in valsList])
            self.dbc.insertIgnoreInto(f"{self.sensor}_images_{self.db_name}", \
                           "x, y, z, q, u, a, t, rtk_status, ins_status, radalt, save_loc, cam_time1, cam_time2, ins_time1, ins_time2", vals)
        except sqlite3.OperationalError as ex:
            self.get_logger().info(f'    Attempted to insert bad pose: {ex}')

        self.image = None
        self.pos = None
        self.quat = None
        self.RTK_STATUS = None
        self.INS_STATUS = None
        self.STROBE = None
        self.data_loc = None
        self.ins_times = None
        self.cam_times = None

        self.update_check_list()


    def cam_cb(self, msg: Image):
        self.get_logger().info('  Image received.')
 #       start = time.time()

        tmp = self.get_clock().now().to_msg()
        sec1 = str(tmp.sec)
        nsec1 = str(tmp.nanosec).rjust(9,str(0))
        time1 = f'{sec1}.{nsec1}'

        sec2 = str(msg.header.stamp.sec)
        nsec2 = str(msg.header.stamp.nanosec).rjust(9,str(0))
        time2 = f'{sec2}.{nsec2}'

        self.cam_times = [time1, time2]

        self.data_loc = self.dir_name + "/" + self.sensor + '_' + time2 + ".png"
        self.image = self.br.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BAYER_RG2RGB)

        self.update_check_list()

        if self.STROBE is not None:
            if self.status_check():
                self.save_image_pose()
            elif self.radalt is None:
                self.get_logger().info(f'    skipping image and pose; self.radalt is still unset')
            else:
                self.get_logger().info(f'    *** BONK ***')
        else:
            self.get_logger().info(f'    holding image; STROBE is unset')

#        self.get_logger().info(f'      cam_cb runtime: {time.time()-start}')


    def radalt_cb(self, msg: AltSNR):
        if msg.snr > 13:
            self.radalt = msg.altitude
        else:
            print('radalt measurement discarded; SNR too small')


    def ins_cb(self, msg: DIDINS2):
        self.get_logger().info('  Pose received.')
#        start = time.time()

        tmp = self.get_clock().now().to_msg()
        sec1 = str(tmp.sec)
        nsec1 = str(tmp.nanosec).rjust(9,str(0))
        time1 = f'{sec1}.{nsec1}'

        sec2 = str(msg.header.stamp.sec)
        nsec2 = str(msg.header.stamp.nanosec).rjust(9,str(0))
        time2 = f'{sec2}.{nsec2}'

        u = utm.from_latlon(msg.lla[0], msg.lla[1])
        self.pos = [u[0], u[1], msg.lla[2]]
        self.quat = [msg.qn2b[1], msg.qn2b[2], msg.qn2b[3], msg.qn2b[0]]

        self.ins_times = [time1, time2]
        self.RTK_STATUS = ((msg.ins_status)&self.INS_STATUS_GPS_NAV_FIX_MASK)>>self.INS_STATUS_GPS_NAV_FIX_OFFSET
        self.INS_STATUS = ((msg.ins_status)&self.INS_STATUS_SOLUTION_MASK)>>self.INS_STATUS_SOLUTION_OFFSET

        if msg.hdw_status & self.HDW_STATUS_STROBE_IN_EVENT == self.HDW_STATUS_STROBE_IN_EVENT:
            self.get_logger().info('    Strobed.')
            self.STROBE = 1
            self.ins_times = [time1, time2]
            self.RTK_STATUS = ((msg.ins_status)&self.INS_STATUS_GPS_NAV_FIX_MASK)>>self.INS_STATUS_GPS_NAV_FIX_OFFSET
            self.update_check_list()

        if self.status_check() and self.radalt is not None:
            self.save_image_pose()
        elif self.radalt is None:
            self.get_logger().info(f'    Skipping image and pose; radalt is still unset')
            # else:
            #    self.get_logger().info(f'    *** PING ***')

#        valsList = pos + quat + [msg.ins_status, msg.hdw_status, time1, time2]
#        vals = ','.join([str(x) for x in valsList])
#        self.dbc.insertIgnoreInto(f"ins_data_{self.db_name}", \
#            "x, y, z, q, u, a, t, insStatus, hdwStatus, time1, time2", vals)
#        self.get_logger().info(f'      ins_cb runtime: {time.time()-start}')


def main(args=None):
    rclpy.init(args=args)
    sub_node = subscriberNode()
    rclpy.spin(sub_node)
    sub_node.destroy_node()
    rclpy.shutdown()
