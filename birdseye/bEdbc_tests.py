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
from dbConnector import dbConnector
from copy import deepcopy
from db_utilities import *


class tester:
    def __init__(self):
        self.sensor = 'cam0'
        self.db_name = 'flight_data'
        self.dir_name = 'parsed_flight'
        self.dir_name = os.path.join(os.path.expanduser('~'), self.dir_name)
        self.dbc = dbConnector(os.path.join(self.dir_name, self.db_name))
        self.dbc.boot(self.db_name, self.sensor)

        self.sensors_yaml = "camera_yamls/birdsEyeSensorParams.yaml"
        self.sensors_yaml = os.path.join(os.path.expanduser('~'), self.sensors_yaml)

        self.K = None
        self.dist = None
        self.res = None
        self.extr = None
        self.calibUptake()


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
                    self.extr = matrix_list_converter(self.extr, (4,4))
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
                    self.putParameters(device, res, intr1, intr2, matrix_list_converter(extr, (4,4)))
                elif device == 'radalt':
                    extr = data["T_rad_imu"]
                    self.putParameters(device, res, intr1, intr2, matrix_list_converter(extr, (4,4)))
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
                    tmp = string_list_converter(item)
                    # print('tmp: ', tmp)
                    if item == elem[-1]:
                        tmp = matrix_list_converter(tmp, (4,4))
                    # print(tmp)
                    params.append(tmp)
        return params


if __name__ == '__main__':
    tst = tester()
    print('loaded \n')
    tmp = tst.getParameters('ublox')
    # print('tmp: ', tmp)
    for param in tmp:
        print('param: ', param, '\n', type(param), end=' ')
        if param is not None and not isinstance(param, str):
            print(len(param), end=' ')
            if len(param) == 4:
                try:
                    print(len(param[0]), end=' ')
                except TypeError:
                    pass
            print(type(param), type(param[0]), end=' ')
            try:
                print(type(param[0][0]), end=' ')
            except TypeError:
                pass
        print('\n')
