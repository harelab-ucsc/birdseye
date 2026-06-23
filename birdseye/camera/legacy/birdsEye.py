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
    print('    returning')
    return state, ret, img


def projection_stats(april_2D, april_3D, clicks_2D, clicks_3D):
    bproj = None
    reproj = None

    try:
        if april_2D is not None:
            april_2D = np.array(april_2D)
        if clicks_2D is not None:
            clicks_2D = np.array(clicks_2D)
        if april_2D.shape == clicks_2D.shape:
            reproj = clicks_2D - april_2D
        else:
            print('  inconsistent arrays for reprojection calculation:')
            print(f'    clicks_2D: {clicks_2D.shape}')
            print(f'    april_2D: {april_2D.shape}')
    except AttributeError as e:
        print(e)
        print('  leaving reproj as None')

    try:
        if april_3D is not None:
            april_3D = np.array(april_3D)
        if clicks_3D is not None:
            clicks_3D = np.array(clicks_3D)
        if april_3D.shape == clicks_3D.shape:
            bproj = april_3D - clicks_3D
        else:
            print('  inconsistent arrays for backprojection calculation:')
            print(f'    clicks_3D: {clicks_3D.shape}')
            print(f'    april_3D: {april_3D.shape}')
    except AttributeError as e:
        print(e)
        print('  leaving bproj as None')

    # print(f'  bproj: {bproj}')
    # print(f'  reproj: {reproj}')

    return bproj, reproj


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
        # self.T_IC = self.T_IC@np.array([[0,1,0,0],[1,0,0,0],[0,0,-1,0],[0,0,0,1]])
        self._2DFrameVertices = ((0,0), \
                                 (self.res[0] - 1, 0), \
                                 (self.res[0] - 1, self.res[1] - 1), \
                                 (0, self.res[1] - 1))
        self.noise = (125.65394801,90.87022367)  # projection sigma, in the order (sigma_x, sigma_y)
        self.scale = 2  # parameter to tune buffer width; self.scale*self.noise
        self._2DInnerBound = ((self.scale*self.noise[0],self.scale*self.noise[1]), \
                              (self.res[0]-self.scale*self.noise[0]-1, self.scale*self.noise[1]), \
                              (self.res[0]-self.scale*self.noise[0]-1, self.res[1]-self.scale*self.noise[1]-1), \
                              (self.scale*self.noise[0], self.res[1]-self.scale*self.noise[1]-1))
        self._2DOuterBound = ((-self.scale*self.noise[0],-self.scale*self.noise[1]), \
                              (self.res[0]+self.scale*self.noise[0]-1, -self.scale*self.noise[1]), \
                              (self.res[0]+self.scale*self.noise[0]-1, self.res[1]+self.scale*self.noise[1]-1), \
                              (-self.scale*self.noise[0], self.res[1]+self.scale*self.noise[1]-1))
        self._3DFrameVertices = None
        self._3DInnerBound = None
        self._3DOuterBound = None

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        plt.show(block=False)

        self.rtk_tracker = [0]*4
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
        # print(f'{device_key} params read: \n    resolution: {params[1]} \
        #                                   \n    intrinsics1: {params[2]} \
        #                                   \n    intrinsics2: {params[3]} \
        #                                   \n    extrinsics: {params[4]}')
        return params


    def correct_altitude(self, frame):
        eulers = quat2euler(*frame[3:7])
        cos_theta = math.cos(eulers[0]) * math.cos(eulers[1])

        # Compute corrected altitude
        self.radalt = self.radalt * cos_theta
        if self.radalt < 0:
            self.radalt *= -1


    def _2Dto3D(self, List2D):
        # From https://math.stackexchange.com/questions/4382437/back-projecting-a-2d-pixel-from-an-image-to-its-corresponding-3d-point
        # T_WB : World to base link transform, 4x4 nonsingular matrix
        # T_BC : Base link to camera transform, 4x4 nonsingular matrix
        # K : Intrinsic matrix for camera
        # List2D : n x 2 array of pixel locations in an image

        List3D = []

        if List2D is not None:
            List2D = np.array(List2D)

            for p in List2D:
                # Homogeneous pixel coordinate
                p = np.array([p[0], p[1], 1]).T;
                # Transform pixel in Camera coordinate frame
                pc = np.linalg.inv(self.K) @ p
                # print(pc, pc.shape)
                pc = np.hstack((pc,1.0))
                # print(pc, pc.shape)

                # Transform pixel in World coordinate frame
                pw = self.T_WC @ pc

                # Transform camera origin in World coordinate frame
                cam = np.array([0,0,0,1]).T
                cam_world = self.T_WC @ cam

                # Find a ray from camera to 3d point
                vector = pw - cam_world
                unit_vector = vector / np.linalg.norm(vector)

                # Point scaled along this ray
                p3D = cam_world + self.radalt*vector
                List3D.append(p3D.tolist())

        return List3D


    def _3Dto2D(self, pins):
        #Given world->base and base->camera 4x4 matrices, as well as a K matrix, and a 2d array of pins, project each pin into image space
        #Return a 2d array of image coordinates corresponding to each pin
        #First, compose the image space. That is, take the world pose from the DB, and use a rigid transform to get the sensor frame

        # changing the dimensions of the click to be a 1x4
        pins = np.concatenate((pins,np.ones((len(pins), 1))), axis=1)
        # print(pins.shape)

        # projecting the image by x_image = K * T_{wc} * X_utm
        sensor_frame_pins = np.linalg.inv(self.T_WC)@pins.T

        #in the sensor frame, z points down, x backward, and y right
        projected = self.K@sensor_frame_pins[:-1,:]

        #The projected coordinates are now the columns of the array 'projected'
        #Normalize the homogeneous coordinate to 1
        projected_normalized = projected / projected[2]

        #These are the pins as projected into pixel space for this pose
        return projected_normalized[:2].T.tolist()


    def _2DBoxCheck(self, pts, box='frame', stats=False):
        if stats:
            backproj = []

        if box == 'frame':
            valid_mask =  self.inQuadrilateralCheck(self._2DFrameVertices, pts)
        elif box == 'inner':
            valid_mask =  self.inQuadrilateralCheck(self._2DInnerBound, pts)
        elif box == 'outer':
            valid_mask =  self.inQuadrilateralCheck(self._2DOuterBound, pts)

        valid_pts = np.array(pts)[valid_mask]
        valid_pts = valid_pts.tolist()
        if stats:
            backproj = self._2Dto3D(valid_pts)
            return valid_pts, backproj
        else:
            return valid_pts, None


    def inQuadrilateralCheck(self, frame, pts):
        n = len(frame)
        frame = np.array(frame)
        # print('    inQuadCheck frame: \n', frame)
        val = []
        ring = lambda y: [ (x + 1) % y for x in range(y)]
        det = lambda x,y: x[0]*y[1] - [x[1]*y[0]]
        tmp1 = frame[ring(n)] - frame

        for v in pts:
            v = np.array(v)
            tmp2 = v - frame
            tmp = [int(det(tmp1[i], tmp2[i])) >= 0 for i in range(n)]
            val.append(sum(tmp))
        val = [True if i == 4.0 else False for i in val]
        return val


    def annotate(self, img_file, label):
        f = open(f"{self.save_name}.txt", "a")
        line = f'{self.frame_index} {img_file} {label} \n'
        f.write(line)
        f.close()
        print(f'    Annotation saved: frame index {self.frame_index}, {self.save_name}.txt')


    def detect(self, img, img_file):
        start = time.time()
        img.resize((384, 512, img.shape[-1]), refcheck=False)
        # print(type(img), img.shape)
        img = img.astype(np.float32)
        self.model.set_tensor(self.input_details[0]['index'], np.expand_dims(img, axis=0))
        self.model.invoke()
        self.pred = self.model.get_tensor(self.output_details[0]['index'])[0]
        f = open(f"{self.det_name}.txt", "a")
        if self.pred[0] < 0.5:
            print(f'    False: {self.pred[0]}')
            line = f'{self.frame_index} {img_file} 0.0 {self.pred[0]} \n'
        else:
            print(f'    True: {self.pred[0]}')
            line = f'{self.frame_index} {img_file} 1.0 {self.pred[0]} \n'
        f.write(line)
        f.close()
        print(f'        Detection complete: took {time.time()-start}s.')


    def frameProcessSetup(self, frame, clks):
        # make homogeneous coordinates for clicks wrt drone pose and radalt
        #    plot in 3D UTM coords
        self.radalt = frame[8]
        self.correct_altitude(frame)
        clicks = np.hstack((clks, np.ones_like(clks[:,0]).reshape(-1,1)*(frame[2]-self.radalt)))
        self.ax.scatter(clicks[:,0], \
                        clicks[:,1], \
                        clicks[:,2], \
                        marker='s', alpha=0.5, c='m', s=32, label='Click')

        # convert pose to 4x4 homogeneous transform
        #    plot in 3D UTM coords
        T_WI = poseRowToTransform(frame[:7])  # our base link maps from the world origin to the base link
        T_WI = T_WI@np.array([[0,1,0,0],[1,0,0,0],[0,0,-1,0],[0,0,0,1]])
        self.T_WC = T_WI@self.T_IC
        plotTransform(self.ax, T_WI, colors=['m','y','c'], labels=['INS x-axis','INS y-axis', 'INS z-axis'])
        plotTransform(self.ax, self.T_WC)

        self.ax.set_xlim(frame[0]-15, frame[0]+15)
        self.ax.set_xlabel('East (m)')
        self.ax.set_ylim(frame[1]-15, frame[1]+15)
        self.ax.set_ylabel('North (m)')
        self.ax.set_zlim(frame[2]-20, frame[2]+1)
        self.ax.set_zlabel('Z (m)')

        self.ax.set_title(f'Time: {frame[-1]}')
        self.ax.set_box_aspect([1,1,1])
        self.ax.set_proj_type('ortho')

        # if self.frame_index == 1:
        #     pdb.set_trace()

        self._3DFrameVertices = self._2Dto3D(self._2DFrameVertices)
        self.ax.scatter(np.array(self._3DFrameVertices)[:,0], \
                        np.array(self._3DFrameVertices)[:,1], \
                        np.array(self._3DFrameVertices)[:,2], \
                        marker='s', color='k', label='Frame')

        self._3DInnerBound = self._2Dto3D(self._2DInnerBound)
        self.ax.scatter(np.array(self._3DInnerBound)[:,0], \
                        np.array(self._3DInnerBound)[:,1], \
                        np.array(self._3DInnerBound)[:,2], \
                        marker='s', color='g', label='InnerBound')

        self._3DOuterBound = self._2Dto3D(self._2DOuterBound)
        self.ax.scatter(np.array(self._3DOuterBound)[:,0], \
                        np.array(self._3DOuterBound)[:,1], \
                        np.array(self._3DOuterBound)[:,2], \
                        marker='s', color='c', label='OuterBound')

        if frame[7] == 3:
            color = 'g'
            self.rtk_tracker[0] += 1
            self.RTK_watchdog = 1
        elif frame[7] == 2:
            color = 'y'
            self.rtk_tracker[1] += 1
            self.RTK_watchdog = 0
        elif frame[7] == 1:
            color = 'r'
            self.rtk_tracker[2] += 1
            self.RTK_watchdog = 0
        else:
            color = 'k'
            self.rtk_tracker[3] += 1
            self.RTK_watchdog = 0

        if self.frame_index >= memory:  # `memory` is defined at the top of the file
            for tmp in self.data[(self.frame_index-memory):self.frame_index]:
                if tmp[7] == 3:
                    color = 'g'
                elif tmp[7] == 2:
                    color = 'y'
                elif tmp[7] == 1:
                    color = 'r'
                else:
                    color = 'k'
                self.ax.scatter(tmp[0], tmp[1], tmp[2], c=color, alpha=0.1, s=32)
        else:
            for tmp in self.data[:self.frame_index]:
                if tmp[7] == 3:
                    color = 'g'
                elif tmp[7] == 2:
                    color = 'y'
                elif tmp[7] == 1:
                    color = 'r'
                else:
                    color = 'k'
                self.ax.scatter(tmp[0], tmp[1], tmp[2], c=color, alpha=0.1, s=32)
        return clicks


    def parseFlightDatabase(self):
        clks = self.dbc.getFrom('x, y', f"clicks_{self.db_name}")
        clks = np.array(clks)
        # print("clicks: \n", clks, "\n clicks.shape:", clks.shape)

        # load every pose entry saved by `sub_node.py`; each row is a pose
        self.data = self.dbc.getFrom('x, y, z, q, u, a, t, rtk_status, radalt, save_loc, cam_time1, cam_time2, ins_time1, ins_time2', f'{self.sensor}_images_{self.db_name}')

        # Create rectification and projection maps
        map1, map2 = cv2.initUndistortRectifyMap(self.K, self.D, None, self.K, (self.res[0], self.res[1]), cv2.CV_32FC1)
        self._2DFrameVertices = cv2.undistortPointsIter(np.array(self._2DFrameVertices,dtype = np.float64), self.K, self.D, None, self.K, (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 0.003))
        self._2DFrameVertices = np.squeeze(self._2DFrameVertices).tolist()
        self._2DInnerBound = cv2.undistortPointsIter(np.array(self._2DInnerBound,dtype = np.float64), self.K, self.D, None, self.K, (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 0.003))
        self._2DInnerBound = np.squeeze(self._2DInnerBound).tolist()
        self._2DOuterBound = cv2.undistortPointsIter(np.array(self._2DOuterBound,dtype = np.float64), self.K, self.D, None, self.K, (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 0.003))
        self._2DOuterBound = np.squeeze(self._2DOuterBound).tolist()

        out_dict = {}

        cv2.namedWindow("Window", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Window", 1920, 1200)

        for i, frame in enumerate(self.data):
            print(f'frame: {i+1} of {len(self.data)}')
            self.frame_index = i

            clicks = self.frameProcessSetup(frame, clks)
            img = None
            bproj = None
            reproj = None
            state = None
            april_2D = None
            april_3D = None
            clicks_2D = None
            clicks_3D = None

            if self.radalt > 3.0: # and frame[-5] == 131:
                # detect apriltags (`reproj`) as GT for `clicks_2D`
                # back-project clicks (`bproj`), compare to `clicks` as GT
                img = cv2.imread(frame[-5])
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                rect = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)

                print('  detecting, projecting apriltags')
                state, april_2D, rect = apriltag_detect(rect)
                april_3D = self._2Dto3D(april_2D)
                print('  detecting, projecting clicks')
                clicks_2D = self._3Dto2D(clicks)
                clicks_2D, clicks_3D = self._2DBoxCheck(clicks_2D, stats=True)
                inner, _ = self._2DBoxCheck(clicks_2D, box='inner')
                outer, _ = self._2DBoxCheck(clicks_2D, box='outer')
                rect = cv2.rectangle(rect, \
                     [int(i) for i in self._2DInnerBound[0]], \
                     [int(i) for i in self._2DInnerBound[2]], \
                     (0,255,255), \
                     2)
                print('  getting stats')
                bproj, reproj = projection_stats(april_2D, april_3D, clicks_2D, clicks_3D)
                if self.RTK_watchdog:
                    print('    healthy RTK; saving projection stats')
                    if bproj is not None:
                        self.bproj += bproj.tolist()
                    if reproj is not None:
                        self.reproj += reproj.tolist()

                    out_dict['bproj'] = self.bproj
                    out_dict['reproj'] = self.reproj

                    with open(os.path.join(self.img_dir, 'out_dict.pkl'), 'wb') as f:
                        pickle.dump(out_dict, f)

                cv2.putText(rect, f'{frame[-1]}', (30,80), \
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3)

                if len(outer) > 0:
                    if frame[7] == 3:
                        color = (0,255,0)
                        if len(inner) > 0:
                            print('    clicks within InnerBound; annotating frame True')
                            self.annotate(frame[-5], 1.0)
                            cv2.putText(rect, 'True', (1600,80), \
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3)
                        elif len(inner) == 0:
                            print('    click detected in frame buffer region; annotating frame "Test"')
                            self.annotate(frame[-5], "Test")
                            cv2.putText(rect, 'Test', (1600,80), \
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 3)
                    elif frame[7] == 2:
                        color = (0,255,255)
                    elif frame[7] == 1:
                        color = (0,0,255)
                    else:
                        color = (0,0,0)

                    if not self.RTK_watchdog:
                        print(f'    skipping annotation: bad RTK_STATUS, {frame[-5]}')

                    for click in clicks_2D:
                        cv2.circle(rect, [int(click[0]), int(click[1])], 15, color, -1)
                    if clicks_3D is not None:
                        self.clicks_3D += clicks_3D
                        bp = np.array(self.clicks_3D)
                        bp = np.squeeze(bp)
                    if april_3D is not None:
                        self.april_3D += april_3D
                        ap = np.array(self.april_3D)
                        ap = np.squeeze(ap)
                else:
                    cv2.putText(rect, 'False', (1600,80), \
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
                    if self.RTK_watchdog:
                        self.annotate(frame[-5], 0.0)
                    else:
                        print(f'    skipping annotation: bad RTK_STATUS, {frame[7]}')

                cv2.imshow("Window", rect)
                cv2.waitKey(30)
                # p = os.path.expanduser('~')
                # p = os.path.join(p, 'catch', 'tmp', f'2d_{str(self.frame_index).rjust(3,str(0))}.png')
                # cv2.imwrite(p, rect)

            if len(self.clicks_3D) > 1:
                self.ax.scatter(bp[:,0], \
                                bp[:,1], \
                                bp[:,2], \
                                c='b', alpha=0.1, s=32, label='ClickBackProj')
            elif len(self.clicks_3D) == 1:
                # first click
                self.ax.scatter(bp[0], \
                                bp[1], \
                                bp[2], \
                                c='b', alpha=0.1, s=32)
            else:
                pass  # nothing yet

            if len(self.april_3D) > 1:
                self.ax.scatter(ap[:,0], \
                                ap[:,1], \
                                ap[:,2], \
                                c='r', alpha=0.3, s=16, label='AprilBackProj')
            elif len(self.april_3D) == 1:
                # first click
                self.ax.scatter(ap[0], \
                                ap[1], \
                                ap[2], \
                                c='r', alpha=0.3, s=16)
            else:
                pass  # nothing yet

            self.ax.legend()
            self.fig.canvas.draw_idle()
            plt.pause(0.01)
            # p = os.path.expanduser('~')
            # p = os.path.join(p, 'catch', 'tmp', f'3d_{str(self.frame_index).rjust(3,str(0))}.png')
            # self.fig.savefig(p)
            self.ax.cla()

        # self.grab_plots()

        print(self.rtk_tracker)
        print(f'RTK Service Stats:')
        print(f'    Status 3 (Fix): {self.rtk_tracker[0]} of {sum(self.rtk_tracker)} ({self.rtk_tracker[0]/sum(self.rtk_tracker)})')
        print(f'    Status 2 (Float): {self.rtk_tracker[1]} of {sum(self.rtk_tracker)} ({self.rtk_tracker[1]/sum(self.rtk_tracker)})')
        print(f'    Status 1 (None): {self.rtk_tracker[2]} of {sum(self.rtk_tracker)} ({self.rtk_tracker[2]/sum(self.rtk_tracker)})')
        print(f'    Rare Statuses: {self.rtk_tracker[3]} of {sum(self.rtk_tracker)} ({self.rtk_tracker[3]/sum(self.rtk_tracker)})')

        with open(os.path.join(self.img_dir, 'out_dict.pkl'), 'wb') as f:
            pickle.dump(out_dict, f)


    # def detectionProcess(self):
    #     clks = self.dbc.getFrom('x, y', f"clicks_{self.db_name}")
    #     clks = np.array(clks)
    #     print("clicks: \n", clks, "\n clicks.shape:", clks.shape)
    #
    #     self.data = self.dbc.getFrom('x, y, z, q, u, a, t, rtk_fix, radalt, save_loc, time', f'{self.sensor}_images_{self.db_name}')
    #     # print(self.data)
    #
    #     # Create rectification and projection maps
    #     map1, map2 = cv2.initUndistortRectifyMap(self.K, self.D, None, self.K, (self.res[0], self.res[1]), cv2.CV_32FC1)
    #     self._2DFrameVertices = cv2.undistortPointsIter(np.array(self._2DFrameVertices,dtype = np.float64), self.K, self.D, None, self.K, (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 0.003))
    #     self._2DFrameVertices = np.squeeze(self._2DFrameVertices).tolist()
    #
    #     cv2.namedWindow("Window", cv2.WINDOW_NORMAL)
    #     cv2.resizeWindow("Window", 1920, 1200)
    #
    #     for i, frame in enumerate(self.data):
    #         print(f'frame: {i+1} of {len(self.data)}')
    #         self.frame_index = i
    #
    #         clicks = np.hstack((clks, np.ones_like(clks[:,0]).reshape(-1,1)*(frame[2]-frame[-3])))
    #         # print("clicks: \n", clicks, "\n clicks.shape:", clicks.shape)
    #
    #         self.T_WC = poseRowToTransform(frame[:7])  # our base link maps from the world origin to the base link
    #         self.radalt = frame[8]
    #         plotTransform(self.ax, self.T_WC)
    #         clicks_2D = self._3Dto2D(clicks)
    #         self.ax.scatter(clicks[:,0], \
    #                         clicks[:,1], \
    #                         clicks[:,2], \
    #                         marker='s', alpha=0.5, c='m', s=32, label='Click')
    #         self._3DFrameVertices = self._2Dto3D(self._2DFrameVertices)
    #         self.ax.scatter(np.array(self._3DFrameVertices)[:,0], \
    #                         np.array(self._3DFrameVertices)[:,1], \
    #                         np.array(self._3DFrameVertices)[:,2], \
    #                         marker='s', color='k', label='Frame')
    #         clicks_2D, bproj = self._2DBoxCheck(clicks_2D, stats=True)
    #
    #         if self.radalt > 3.0:  # and frame[7] == 3:
    #             # print('  cv2.imread')
    #             img = cv2.imread(frame[-3])
    #             # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #             rect = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
    #             cv2.putText(rect, f'{frame[-1]}', (50,100), \
    #                 cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 4)
    #             self.detect(img, frame[-3])
    #
    #             if self.pred[0] >= 0.5:
    #                 if frame[7] == 3:
    #                     color = (0,255,0)
    #                 elif frame[7] == 2:
    #                     color = (0,255,255)
    #                 elif frame[7] == 1:
    #                     color = (0,0,255)
    #                 else:
    #                     color == (0,0,0)
    #                 cv2.putText(rect, 'True', (1600,100), \
    #                     cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 4)
    #
    #                 for click in clicks_2D:
    #                     cv2.circle(rect, [int(click[0]), int(click[1])], 15, color, -1)
    #                 if bproj is not None:
    #                     self.bproj += bproj
    #                     bp = np.array(self.bproj)
    #                     bp = np.squeeze(bp)
    #             else:
    #                 cv2.putText(rect, 'False', (1600,100), \
    #                     cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 4)
    #
    #             cv2.imshow("Window", rect)
    #             cv2.waitKey(30)
    #
    #             if len(self.bproj) > 1:
    #                 # print(f'    new clicks: \n    {bp}')
    #                 self.ax.scatter(bp[:,0], \
    #                                 bp[:,1], \
    #                                 bp[:,2], \
    #                                 c='b', alpha=0.3, s=64, label='BackProj')
    #             elif len(self.bproj) == 1:
    #                 # print(f'    first click: \n    {bp}')
    #                 self.ax.scatter(bp[0], \
    #                                 bp[1], \
    #                                 bp[2], \
    #                                 c='b', alpha=0.3, s=64)
    #             else:
    #                 pass  # nothing yet
    #
    #             if i >= memory :
    #                 for tmp in self.data[(i-memory):i]:
    #                     if self.pred[0] >= 0.5:
    #                         color = 'g'
    #                     else:
    #                         color = 'k'
    #                     self.ax.scatter(tmp[0], tmp[1], tmp[2], c=color, alpha=0.1, s=32)
    #             else:
    #                 for tmp in self.data[:i]:
    #                     if self.pred[0] >= 0.5:
    #                         color = 'g'
    #                     else:
    #                         color = 'k'
    #                     self.ax.scatter(tmp[0], tmp[1], tmp[2], c=color, alpha=0.1, s=32)
    #
    #         self.ax.set_xlim(frame[0]-15, frame[0]+15)
    #         self.ax.set_xlabel('X')
    #         self.ax.set_ylim(frame[1]-15, frame[1]+15)
    #         self.ax.set_ylabel('Y')
    #         self.ax.set_zlim(frame[2]-20, frame[2]+1)
    #         self.ax.set_zlabel('Z')
    #         self.ax.legend()
    #
    #         self.ax.set_title(f'Time: {frame[-1]}')
    #         self.ax.set_box_aspect([1,1,1])
    #         self.ax.set_proj_type('ortho')
    #         self.fig.canvas.draw_idle()
    #         plt.pause(0.05)
    #         p = os.path.expanduser('~')
    #         p = os.path.join(p, 'catch', 'tmp', f'3d_{str(self.frame_index).rjust(3,str(0))}.png')
    #         self.fig.savefig(p)
    #         self.ax.cla()
    #
    #     self.grab_plots()


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
