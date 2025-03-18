#!/usr/bin/env python3

import numpy as np
import csv
import utm
import os
import pickle
from cf_triad import *
import rclpy
import tflite_runtime.interpreter as tflite
from rclpy.node import Node
from sensor_msgs.msg import Imu, Image, NavSatFix
from std_msgs.msg import String
from SLICAnnotator import offlineSLICAnnotator
import glob2
from dbConnector import dbConnector
from utilities import *
from AMI_ContourClassFamily import Contour
from pupil_apriltags import Detector
import time
import math
import pdb

memory = 25


def apriltag_detect(img):
    print('  apriltag_detect')
    ret = None
    state = 0

    at_detector = Detector(
        families="tag36h11",
        nthreads=2,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0
        )

    results = at_detector.detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    if results:
        ret = []
        state = 1
        for r in results:
            # extract the bounding box (x, y)-coordinates for the AprilTag
            # and convert each of the (x, y)-coordinate pairs to integers
            (ptA, ptB, ptC, ptD) = r.corners
            ptB = (int(ptB[0]), int(ptB[1]))
            ptC = (int(ptC[0]), int(ptC[1]))
            ptD = (int(ptD[0]), int(ptD[1]))
            ptA = (int(ptA[0]), int(ptA[1]))
            # draw the bounding box of the AprilTag detection
            cv2.line(img, ptA, ptB, (0, 255, 0), 2)
            cv2.line(img, ptB, ptC, (0, 255, 0), 2)
            cv2.line(img, ptC, ptD, (0, 255, 0), 2)
            cv2.line(img, ptD, ptA, (0, 255, 0), 2)
            # draw the center (x, y)-coordinates of the AprilTag
            (cX, cY) = (int(r.center[0]), int(r.center[1]))
            cv2.circle(img, (cX, cY), 5, (0, 0, 255), -1)
            ret.append([cX,cY])
    print(f'    returning {state}, {ret}')
    return state, ret, img


class birdsEye():
    def __init__(self, **kwargs):
        plt.ion()
        self.img_dir = kwargs.pop('img_dir', None)
        self.save_name = os.path.join(self.img_dir, 'labels')
        self.det_name = os.path.join(self.img_dir, 'detections')
        self.apriltags = kwargs.pop('apriltags', False)
        self.db_name = kwargs.pop('db_name', None)
        self.sensor = kwargs.pop('sensor', 'cam0')
        self.dbc = dbConnector(os.path.join(self.img_dir, self.db_name))
        self.dbc.boot(self.db_name, self.sensor)

        self.model_path = kwargs.pop('model_path', os.path.join(os.path.expanduser('~'),'ucsc_512_384_13.tflite'))
        self.model = tflite.Interpreter(model_path=self.model_path, num_threads=4)
        self.model.allocate_tensors()
        self.input_details = self.model.get_input_details()
        self.output_details = self.model.get_output_details()

        self.radalt = None
        self.data = None
        self.frame_index = None

        # camera specs
        tmp = self.getParameters(self.sensor)
        self.res = [int(tmp[1][0]), int(tmp[1][1])]
        self.K = np.array([[tmp[2][0],0.0,tmp[2][2]], \
                           [0.0,tmp[2][1],tmp[2][3]], \
                           [0.0,0.0,1.0]])
        self.D = np.array(tmp[3])
        self.T_WC = None
        self.T_IC = np.array(tmp[4])


        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        plt.show(block=False)

        self.RTK_watchdog = None

        self.april_2D = []  # list of apriltag detection pixel coordinates
        self.april_3D = []  # list of apriltag detection world coordinates
        self.clicks_2D = []  # list of framewise click pixel coordinates
        self.clicks_3D = []  # list of framewise click world coordinates
        self.bproj = []  # list of framewise (april_3D - clicks_3D)
        self.reproj = []  # list of framewise (clicks_2D - april_2D)
        # self.contour = Contour(res=(self.res[1], self.res[0]))


    def getParameters(self, device_key):
        print(f'getting sensor parameters: {device_key}')
        params = []
        cols = "sensorID, resolution, intrinsics1, intrinsics2, extrinsics"
        table = f"parameters_{self.db_name}"
        ret = self.dbc.getFrom(cols, table, cond=f'WHERE sensorID = "{device_key}"')
        for elem in ret:
            for i, item in enumerate(elem):
                if item == device_key:
                    params.append(item)
                elif item != 'None':
                    tmp = string_list_converter(item)
                    if item == elem[-1]:
                        tmp = matrix_list_converter(tmp, (4,4))
                    params.append(tmp)
        return params


    def correct_altitude(self, frame):
        eulers = quat2euler(*frame[3:7])
        cos_theta = math.cos(eulers[0]) * math.cos(eulers[1])

        # Compute corrected altitude
        self.radalt = self.radalt * cos_theta
        if self.radalt < 0:
            self.radalt *= -1


    def frameProcessSetup(self, frame, clks):
        # make homogeneous coordinates for clicks wrt drone pose and radalt
        self.radalt = frame[8]
        self.correct_altitude(frame)
        clicks = np.hstack((clks, np.ones_like(clks[:,0]).reshape(-1,1)*(frame[2]-self.radalt)))

        # convert pose to 4x4 homogeneous transform
        T_WI = poseRowToTransform(frame[:7])  # our base link maps from the world origin to the base link
        T_WI = T_WI@np.array([[0,1,0,0],[1,0,0,0],[0,0,-1,0],[0,0,0,1]])

        if frame[7] == 3:
            self.RTK_watchdog = 1
        else:
            self.RTK_watchdog = 0

        return T_WI, clicks


    def parseFlightDatabase(self):
        clks = self.dbc.getFrom('x, y', f"clicks_{self.db_name}")
        clks = np.array(clks)
        # print("clicks: \n", clks, "\n clicks.shape:", clks.shape)

        # load every pose entry saved by `sub_node.py`; each row is a pose
        self.data = self.dbc.getFrom('x, y, z, q, u, a, t, rtk_status, radalt, save_loc, cam_time1, cam_time2, ins_time1, ins_time2', f'{self.sensor}_images_{self.db_name}')

        # Create rectification and projection maps
        map1, map2 = cv2.initUndistortRectifyMap(self.K, self.D, None, self.K, (self.res[0], self.res[1]), cv2.CV_32FC1)

        out_dict = {}

        cv2.namedWindow("Window", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Window", int(self.res[0]/2), int(self.res[1]/2))

        for i, frame in enumerate(self.data):
            print(f'frame: {i+1} of {len(self.data)}')
            self.frame_index = i

            T_WI, clicks = self.frameProcessSetup(frame, clks)
            img = None
            april_2D = None

            if self.radalt > 3.0:
                img = cv2.imread(frame[-5])
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                rect = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)

                print('  detecting, projecting apriltags')
                state, april_2D, rect = apriltag_detect(rect)
                print('  done')

                cv2.putText(rect, f'{frame[-1]}', (30,80), \
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3)

                cv2.imshow("Window", rect)
                cv2.waitKey(30)

        with open(os.path.join(self.img_dir, 'out_dict.pkl'), 'wb') as f:
            pickle.dump(out_dict, f)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src_dir", help="path to source directory (default: parsed_flight)")
    args = vars(parser.parse_args())

    if args['src_dir'] is not None:
        dir_path = os.path.join(os.path.expanduser('~'), args['src_dir'])
    else:
        dir_path = os.path.join(os.path.expanduser('~'), 'parsed_flight')

    print(f'Processing data from {dir_path}')

    db_name = 'flight_data'
    # dbc = dbConnector(os.path.join(dir_path,db_name))
    tst = birdsEye(db_name=db_name, img_dir=dir_path, apriltags=True)

    tst.parseFlightDatabase()
    # tst.detectionProcess()
