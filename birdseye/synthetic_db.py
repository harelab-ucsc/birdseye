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

from . import dbConnector
from . import utilities
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

import rclpy.node
from rclpy.exceptions import ParameterNotDeclaredException
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult

#TODO: Get the subscriber working with correct time stamps for flight table
db_name = 'sim'
clicks_csv = None

def generatePose(f_trans = False, s_trans = False, y_rot = False, p_rot = True, r_rot = False, y_trans = False, 
                 p_trans = False, r_trans = False, yp_trans = False, yr_trans = False, rp_trans = False, ypr_trans = False):
    # rule for movig UAV system in space. 3 steps for each hard coded scenario
    pose = []
    x = 0.0
    t = 1.0

    # forward translation
    if f_trans == True:
        for i in range(3):
            ang1 = euler2quat(0,0,0)
            pose.append((583758+x, 4090492, 10, ang1[0], ang1[1], ang1[2], ang1[3], t))
            x+=5.0
            t+=1.0
            
    # sideways translation
    if s_trans == True:
        for i in range(3):
            ang1 = euler2quat(0,0,0)
            pose.append((583758, 4090492+x, 10, ang1[0], ang1[1], ang1[2], ang1[3], t))
            x+=5.0
            t+=1.0

    # yaw
    if y_rot == True:
        for i in range(3):
            ang1 = euler2quat(0,0,x) 
            pose.append((583758, 4090492, 10, ang1[0], ang1[1], ang1[2], ang1[3], t))
            x+=10
            t+=1.0

    # pitch
    if p_rot == True:
        for i in range(3):
            ang1 = euler2quat(x,0,0)
            pose.append((583758, 4090492, 10, ang1[0], ang1[1], ang1[2], ang1[3], t))
            x+=5
            t+=1.0

    # roll
    if r_rot == True:
        for i in range(3):
            ang1 = euler2quat(0,x,0)
            pose.append((583758, 4090492, 10, ang1[0], ang1[1], ang1[2], ang1[3], t))
            x+=5
            t+=1.0

    # yaw and translation 
    if y_trans == True:
        for i in range(3):
            ang1 = euler2quat(0,x,0)
            pose.append((583758+x, 4090492, 10, ang1[0], ang1[1], ang1[2], ang1[3], t))
            x+=5
            t+=1.0

    # pitch and translation
    if p_trans == True:
        for i in range(3):
            ang1 = euler2quat(x,0,0)
            pose.append((583758, 4090492+x, 10, ang1[0], ang1[1], ang1[2], ang1[3], t))
            x+=5.0
            t+=1.0 

    # roll and translation
    if r_trans == True:
        for i in range(3):
            ang1 = euler2quat(0,0,x)
            pose.append((583758+x, 4090492, 10, ang1[0], ang1[1], ang1[2], ang1[3], t))
            x+=5.0
            t+=1.0

    # yaw pitch translation
    if yp_trans == True:
        for i in range(3):
            ang1 = euler2quat(x,x,0)
            pose.append((583758+x, 4090492, 10, ang1[0], ang1[1], ang1[2], ang1[3], t))
            x+=5.0
            t+=1.0

    # yaw roll translation
    if yr_trans == True:
        for i in range(3):
            ang1 = euler2quat(0,x,x)
            pose.append((583758, 4090492+x, 10, ang1[0], ang1[1], ang1[2], ang1[3], t))
            x+=5.0
            t+=1.0

    # roll pitch translation
    if rp_trans == True:
        for i in range(3):
            ang1 = euler2quat(x,0,x)
            pose.append((583758+x, 4090492, 10, ang1[0], ang1[1], ang1[2], ang1[3], t))
            x+=5
            t+=1.0

    # yaw pitch roll translation
    if ypr_trans == True:
        for i in range(3):
            ang1 = euler2quat(x,x,x)
            pose.append((583758+x, 4090492+x, 10, ang1[0], ang1[1], ang1[2], ang1[3], t))
            x+=10.0
            t+=1.0

    return pose 


def generateATpos():
    # given a fixed point (nonzero x,y) = some GPS of apriltag
        # project this coordinate into pixel space (astr0)

    # Set apriltag center in pixel space
        # undistort the image
        # project a ray to ground plane @ zero altitude
        # repeat click location of april tag (sim GPS)
        # later add noise
    
    # sets random point between (583758, 4090492) and (583785, 4090516) in utm
    AT_pos = [(random.randint(583758, 583785), random.randint(4090492, 4090516), 1)]
    return AT_pos




class subscriberNode(rclpy.node.Node):
    def __init__(self):
        time.sleep(1)
        # node init
        super().__init__('flight_data_sub')
        self.declare_parameter("sensorID", rclpy.Parameter.Type.STRING)
        self.sensor = self.get_parameter("sensorID").value

        self.declare_parameter("sensors_yaml", "sensor_params/birdsEyeSensorParams.yaml")
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

        # camera calibration and projection parameters
        self.calibUptake()

        os.chmod(os.path.join(self.dir_name, self.db_name+'.db'), stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        time.sleep(1)

        # tf2 piping
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.declare_parameter('source_frame', 'cam0')
        self.source_frame = self.get_parameter('source_frame').value
        self.declare_parameter('target_frame', 'utm')
        self.target_frame = self.get_parameter('target_frame').value

        self.br = CvBridge()

        # camera subscriber
        self.cam_sub = self.create_subscription(
            Image, '/image', self.cam_cb, 100)
        # ublox subscribers
        self.ublox_health_sub = self.create_subscription(
            NavPVT, '/gps_flag', self.navpvt_cb, 100)
        self.RTK_STATUS = None


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
        self.get_logger().info(f'Reading clicks CSV file: {self.clicks_csv}...')
        data = []
        with open(self.clicks_csv) as clicks:
            reader = csv.reader(clicks)
            for line in reader:
                # breakdown line
                u = utm.from_latlon(float(line[0]), float(line[1]))
                tag = int(line[-1][-1])
                data.append([u[0], u[1], tag])
        self.dbc.insertClicks(f"clicks_{self.db_name}", data)
        self.get_logger().info('...Done reading clicks CSV file.')


    def calibUptake(self):
        self.get_logger().info(f'Reading sensor parameters YAML file: {self.sensors_yaml}...')
        devices = [f'{self.sensor}', 'imu', 'ublox'] #, 'radalt']
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
                # elif device == 'radalt':
                #     extr = data["T_rad_imu"]
                #     self.putParameters(device, res, intr1, intr2, utilities.matrix_list_converter(extr, (4,4)))
                res = None
                intr1 = None
                intr2 = None
                extr = None
        self.get_logger().info('...Done reading sensor parameters YAML file.')


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

        if self.RTK_STATUS == 131:
        # if self.RTK_STATUS == 3 or self.RTK_STATUS == 67 or self.RTK_STATUS == 131:
            try:
                t = self.tf_buffer.lookup_transform(
                    self.target_frame,
                    self.source_frame,
                    rclpy.time.Time())
                t = t.transform
                # self.get_logger().info(f'[{t.translation.x}, {t.translation.y}, {t.translation.z}]')
                pos = [t.translation.x, t.translation.y, t.translation.z]
                quat = [t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w]
                cv2.imwrite(data_loc, image)
                valsList = pos + quat + [self.RTK_STATUS, '\"'+data_loc+'\"', time]
                vals = ','.join([str(x) for x in valsList])
                self.dbc.insertIgnoreInto(f"{self.sensor}_images_{self.db_name}", \
                                            "x, y, z, q, u, a, t, rtk_fix, save_loc, time", vals)
                # if self.RTK_STATUS != 131:
                #     self.get_logger().info(f'Bad pose recorded... no RTK fix: {self.RTK_STATUS} should be 131')
            except TransformException as ex:
                self.get_logger().info(
                    f'Could not transform {self.source_frame} to {self.target_frame}: {ex}')
                pass
        else:
            # self.get_logger().info(f'bad RTK_STATUS {self.RTK_STATUS}; should be one of 3, 67, 131')
            self.get_logger().info(f'bad RTK_STATUS {self.RTK_STATUS}; should be 131')


    def navpvt_cb(self, msg: NavPVT):
        self.RTK_STATUS = msg.flags  # in {3:GPS, 67:RTK_FLOAT, 131:RTK_FIX}


def main(args=None):
    rclpy.init(args=args)
    sub_node = subscriberNode()
    rclpy.spin(sub_node)
    sub_node.destroy_node()
    rclpy.shutdown()
