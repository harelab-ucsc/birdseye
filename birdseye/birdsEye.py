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


memory = 25


class birdsEye():
    def __init__(self, **kwargs):
        plt.ion()
        self.img_dir = kwargs.pop('img_dir', None)
        self.save_name = os.path.join(self.img_dir, 'labels')
        self.det_name = os.path.join(self.img_dir, 'detections')
        self.apriltags = kwargs.pop('apriltgs', False)
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
        self._2DFrameVertices = ((0,0), \
                                 (self.res[0] - 1, 0), \
                                 (self.res[0] - 1, self.res[1] - 1), \
                                 (0, self.res[1] - 1))
        self._3DFrameVertices = None
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        plt.show(block=False)

        self.rtk_tracker = [0]*4

        # self.tgts = None
        # self.april_2D = []
        # self.april_3D = []
        self.bproj = []
        # self.bproj_tgt = []
        # self.reproj = []
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
        print(f'{device_key} params read: \n    resolution: {params[1]} \
                                          \n    intrinsics1: {params[2]} \
                                          \n    intrinsics2: {params[3]} \
                                          \n    extrinsics: {params[4]}')
        return params


    def _2Dto3D(self, List2D):
        # From https://math.stackexchange.com/questions/4382437/back-projecting-a-2d-pixel-from-an-image-to-its-corresponding-3d-point
        # T_WB : World to base link transform, 4x4 nonsingular matrix
        # T_BC : Base link to camera transform, 4x4 nonsingular matrix
        # K : Intrinsic matrix for camera
        # List2D : n x 2 array of pixel locations in an image

        List2D = np.array(List2D)
        List3D = []

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


    def _2DFrameCheck(self, pts, stats=False):
        if stats:
            backproj = []
        valid_mask =  self.inQuadrilateralCheck(self._2DFrameVertices, pts)

        valid_pts = np.array(pts)[valid_mask]
        valid_pts = valid_pts.tolist()
        if stats:
            backproj = self._2Dto3D(valid_pts)
        if stats:
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
        print(f'    Mask saved: frame index {self.frame_index}, {self.save_name}.txt')


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


    def apriltag_detect(self, img):
        # print('apriltag_detect')
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

        # cv2.namedWindow("Detect", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("Detect", 512, 384)

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
                # draw the tag family on the image
                tagFamily = r.tag_family.decode("utf-8")
                cv2.putText(img, tagFamily, (ptA[0], ptA[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # print("[INFO] tag family: {}".format(tagFamily))
                ret.append([cX,cY])
                # ret.append([[cX,cY], [ptA,ptB,ptC,ptD]])
        return state, ret, img


    def frameProcessSetup(self, frame, clks):
        # make homogeneous coordinates for clicks wrt drone pose and radalt
        #    plot in 3D UTM coords
        clicks = np.hstack((clks, np.ones_like(clks[:,0]).reshape(-1,1)*(frame[2]-frame[-4])))
        self.ax.scatter(clicks[:,0], \
                        clicks[:,1], \
                        clicks[:,2], \
                        marker='s', alpha=0.5, c='m', s=32, label='Click')

        # convert pose to 4x4 homogeneous transform
        #    plot in 3D UTM coords
        self.T_WC = poseRowToTransform(frame[:7])  # our base link maps from the world origin to the base link
        self.radalt = frame[-4]
        plotTransform(self.ax, self.T_WC)

        self._3DFrameVertices = self._2Dto3D(self._2DFrameVertices)
        self.ax.scatter(np.array(self._3DFrameVertices)[:,0], \
                        np.array(self._3DFrameVertices)[:,1], \
                        np.array(self._3DFrameVertices)[:,2], \
                        marker='s', color='k', label='Frame')

        if frame[-5] == 131:
            color = 'g'
            self.rtk_tracker[0] += 1
        elif frame[-5] == 67:
            color = 'y'
            self.rtk_tracker[1] += 1
        elif frame[-5] == 3:
            color = 'r'
            self.rtk_tracker[2] += 1
        else:
            color = 'k'
            self.rtk_tracker[3] += 1

        if self.frame_index >= memory:  # `memory` is defined at the top of the file
            for tmp in self.data[(self.frame_index-memory):self.frame_index]:
                if tmp[-5] == 131:
                    color = 'g'
                elif tmp[-5] == 67:
                    color = 'y'
                elif tmp[-5] == 3:
                    color = 'r'
                else:
                    color = 'k'
                self.ax.scatter(tmp[0], tmp[1], tmp[2], c=color, alpha=0.1, s=32)
        else:
            for tmp in self.data[:self.frame_index]:
                if tmp[-5] == 131:
                    color = 'g'
                elif tmp[-5] == 67:
                    color = 'y'
                elif tmp[-5] == 3:
                    color = 'r'
                else:
                    color = 'k'
                self.ax.scatter(tmp[0], tmp[1], tmp[2], c=color, alpha=0.1, s=32)
        return clicks


    def parseFlightDatabase(self):
        clks = self.dbc.getFrom('x, y', f"clicks_{self.db_name}")
        clks = np.array(clks)
        print("clicks: \n", clks, "\n clicks.shape:", clks.shape)

        # load every pose entry saved by `sub_node.py`; each row is a pose
        self.data = self.dbc.getFrom('x, y, z, q, u, a, t, rtk_fix, radalt, save_loc, time1, time2', f'{self.sensor}_images_{self.db_name}')

        # Create rectification and projection maps
        map1, map2 = cv2.initUndistortRectifyMap(self.K, self.D, None, self.K, (self.res[0], self.res[1]), cv2.CV_32FC1)
        self._2DFrameVertices = cv2.undistortPointsIter(np.array(self._2DFrameVertices,dtype = np.float64), self.K, self.D, None, self.K, (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 0.003))
        self._2DFrameVertices = np.squeeze(self._2DFrameVertices).tolist()

        cv2.namedWindow("Window", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Window", 1920, 1200)

        for i, frame in enumerate(self.data):
            print(f'frame: {i+1} of {len(self.data)}')
            self.frame_index = i
            clicks = self.frameProcessSetup(frame, clks)
            bproj = None
            reproj = None

            if self.radalt > 3.0: # and frame[-5] == 131:
                # detect apriltags (`reproj`) as GT for `clicks_2D`
                # back-project clicks (`bproj`), compare to `clicks` as GT
                img = cv2.imread(frame[-3])
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                rect = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
                state, reproj, rect = self.apriltag_detect(rect)

                clicks_2D = self._3Dto2D(clicks)
                clicks_2D, bproj = self._2DFrameCheck(clicks_2D, stats=True)

                cv2.putText(rect, f'{frame[-1]}', (50,100), \
                    cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 4)

                if len(clicks_2D) > 0:
                    if frame[-5] == 131:
                        color = (0,255,0)
                        self.annotate(frame[-3], 1.0)
                    elif frame[-5] == 67:
                        color = (0,255,255)
                    elif frame[-5] == 3:
                        color = (0,0,255)
                    else:
                        color = (0,0,0)

                    if color != (0,255,0):
                        print(f'    skipping annotation: bad RTK_STATUS, {frame[-5]}')

                    cv2.putText(rect, 'True', (1600,100), \
                        cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 4)

                    for click in clicks_2D:
                        cv2.circle(rect, [int(click[0]), int(click[1])], 15, color, -1)
                    # if reproj is not None:
                    #     for tgt in reproj:
                    #         cv2.circle(rect, [int(tgt[0]), int(tgt[1])], 30, color, 5)
                    if bproj is not None:
                        self.bproj += bproj
                        bp = np.array(self.bproj)
                        bp = np.squeeze(bp)
                else:
                    cv2.putText(rect, 'False', (1600,100), \
                        cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 4)
                    if frame[-5] == 131:
                        self.annotate(frame[-3], 0.0)
                    else:
                        print(f'    skipping annotation: bad RTK_STATUS, {frame[-5]}')

                cv2.imshow("Window", rect)
                cv2.waitKey(30)
                p = os.path.expanduser('~')
                p = os.path.join(p, 'catch', 'tmp', f'2d_{str(self.frame_index).rjust(3,str(0))}.png')
                cv2.imwrite(p, rect)

            if len(self.bproj) > 1:
                self.ax.scatter(bp[:,0], \
                                bp[:,1], \
                                bp[:,2], \
                                c='b', alpha=0.3, s=64, label='BackProj')
            elif len(self.bproj) == 1:
                # first click
                self.ax.scatter(bp[0], \
                                bp[1], \
                                bp[2], \
                                c='b', alpha=0.3, s=64)
            else:
                pass  # nothing yet

            self.ax.set_xlim(frame[0]-15, frame[0]+15)
            self.ax.set_xlabel('X')
            self.ax.set_ylim(frame[1]-15, frame[1]+15)
            self.ax.set_ylabel('Y')
            self.ax.set_zlim(frame[2]-20, frame[2]+1)
            self.ax.set_zlabel('Z')
            self.ax.legend()

            self.ax.set_title(f'Time: {frame[-1]}')
            self.ax.set_box_aspect([1,1,1])
            self.ax.set_proj_type('ortho')
            self.fig.canvas.draw_idle()
            plt.pause(0.05)
            p = os.path.expanduser('~')
            p = os.path.join(p, 'catch', 'tmp', f'3d_{str(self.frame_index).rjust(3,str(0))}.png')
            self.fig.savefig(p)
            self.ax.cla()

        self.grab_plots()

        print(self.rtk_tracker)
        print(f'RTK Service Stats:')
        print(f'    Status 131: {self.rtk_tracker[0]} of {sum(self.rtk_tracker)} ({self.rtk_tracker[0]/sum(self.rtk_tracker)})')
        print(f'    Status 67: {self.rtk_tracker[1]} of {sum(self.rtk_tracker)} ({self.rtk_tracker[1]/sum(self.rtk_tracker)})')
        print(f'    Status 3: {self.rtk_tracker[2]} of {sum(self.rtk_tracker)} ({self.rtk_tracker[2]/sum(self.rtk_tracker)})')
        print(f'    Rare Statuses: {self.rtk_tracker[3]} of {sum(self.rtk_tracker)} ({self.rtk_tracker[3]/sum(self.rtk_tracker)})')

        out_dict = {}
        tmp = np.squeeze(np.array(self.bproj))
        out_dict['bproj'] = tmp
        out_dict['clicks'] = clicks

        with open(os.path.join(self.img_dir, 'out_dict.pkl'), 'wb') as f:
            pickle.dump(out_dict, f)


    def detectionProcess(self):
        clks = self.dbc.getFrom('x, y', f"clicks_{self.db_name}")
        clks = np.array(clks)
        print("clicks: \n", clks, "\n clicks.shape:", clks.shape)

        self.data = self.dbc.getFrom('x, y, z, q, u, a, t, rtk_fix, radalt, save_loc, time', f'{self.sensor}_images_{self.db_name}')
        # print(self.data)

        # Create rectification and projection maps
        map1, map2 = cv2.initUndistortRectifyMap(self.K, self.D, None, self.K, (self.res[0], self.res[1]), cv2.CV_32FC1)
        self._2DFrameVertices = cv2.undistortPointsIter(np.array(self._2DFrameVertices,dtype = np.float64), self.K, self.D, None, self.K, (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 0.003))
        self._2DFrameVertices = np.squeeze(self._2DFrameVertices).tolist()

        cv2.namedWindow("Window", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Window", 1920, 1200)

        for i, frame in enumerate(self.data):
            print(f'frame: {i+1} of {len(self.data)}')
            self.frame_index = i

            clicks = np.hstack((clks, np.ones_like(clks[:,0]).reshape(-1,1)*(frame[2]-frame[-3])))
            # print("clicks: \n", clicks, "\n clicks.shape:", clicks.shape)

            self.T_WC = poseRowToTransform(frame[:7])  # our base link maps from the world origin to the base link
            self.radalt = frame[-4]
            plotTransform(self.ax, self.T_WC)
            clicks_2D = self._3Dto2D(clicks)
            self.ax.scatter(clicks[:,0], \
                            clicks[:,1], \
                            clicks[:,2], \
                            marker='s', alpha=0.5, c='m', s=32, label='Click')
            self._3DFrameVertices = self._2Dto3D(self._2DFrameVertices)
            self.ax.scatter(np.array(self._3DFrameVertices)[:,0], \
                            np.array(self._3DFrameVertices)[:,1], \
                            np.array(self._3DFrameVertices)[:,2], \
                            marker='s', color='k', label='Frame')
            clicks_2D, bproj = self._2DFrameCheck(clicks_2D, stats=True)


            if self.radalt > 3.0 and frame[-5] == 131:
                # print('  cv2.imread')
                img = cv2.imread(frame[-3])
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                rect = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
                cv2.putText(rect, f'{frame[-1]}', (50,100), \
                    cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 4)
                self.detect(img, frame[-3])

                if self.pred[0] >= 0.5:
                    if frame[-5] == 131:
                        color = (0,255,0)
                    elif frame[-5] == 67:
                        color = (0,255,255)
                    elif frame[-5] == 3:
                        color = (0,0,255)
                    else:
                        color == (0,0,0)
                    cv2.putText(rect, 'True', (1600,100), \
                        cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 4)

                    for click in clicks_2D:
                        cv2.circle(rect, [int(click[0]), int(click[1])], 15, color, -1)
                    if bproj is not None:
                        self.bproj += bproj
                        bp = np.array(self.bproj)
                        bp = np.squeeze(bp)
                else:
                    cv2.putText(rect, 'False', (1600,100), \
                        cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 4)

                cv2.imshow("Window", rect)
                cv2.waitKey(30)

                if len(self.bproj) > 1:
                    # print(f'    new clicks: \n    {bp}')
                    self.ax.scatter(bp[:,0], \
                                    bp[:,1], \
                                    bp[:,2], \
                                    c='b', alpha=0.3, s=64, label='BackProj')
                elif len(self.bproj) == 1:
                    # print(f'    first click: \n    {bp}')
                    self.ax.scatter(bp[0], \
                                    bp[1], \
                                    bp[2], \
                                    c='b', alpha=0.3, s=64)
                else:
                    pass  # nothing yet

                if i >= memory :
                    for tmp in self.data[(i-memory):i]:
                        if self.pred[0] >= 0.5:
                            color = 'g'
                        else:
                            color = 'k'
                        self.ax.scatter(tmp[0], tmp[1], tmp[2], c=color, alpha=0.1, s=32)
                else:
                    for tmp in self.data[:i]:
                        if self.pred[0] >= 0.5:
                            color = 'g'
                        else:
                            color = 'k'
                        self.ax.scatter(tmp[0], tmp[1], tmp[2], c=color, alpha=0.1, s=32)

            self.ax.set_xlim(frame[0]-15, frame[0]+15)
            self.ax.set_xlabel('X')
            self.ax.set_ylim(frame[1]-15, frame[1]+15)
            self.ax.set_ylabel('Y')
            self.ax.set_zlim(frame[2]-20, frame[2]+1)
            self.ax.set_zlabel('Z')
            self.ax.legend()

            self.ax.set_title(f'Time: {frame[-1]}')
            self.ax.set_box_aspect([1,1,1])
            self.ax.set_proj_type('ortho')
            self.fig.canvas.draw_idle()
            plt.pause(0.05)
            p = os.path.expanduser('~')
            p = os.path.join(p, 'catch', 'tmp', f'3d_{str(self.frame_index).rjust(3,str(0))}.png')
            self.fig.savefig(p)
            self.ax.cla()
            # self.tgts = None

        self.grab_plots()


    def grab_plots(self):
        fig, ax = plt.subplots(1, 1, figsize=(15, 15))
        rtk_data = self.dbc.getFrom('lat, lon, altitude, rtk_fix, time', f'rtk_data_{self.db_name}')
        rtk_tmp = [[*utm.from_latlon(rtk_data[i][0], rtk_data[i][1])[:2], rtk_data[i][2], rtk_data[i][-1]] for i in range(len(rtk_data))]
        xs = [rtk_tmp[i][0] for i in range(len(rtk_tmp))]
        ys = [rtk_tmp[i][1] for i in range(len(rtk_tmp))]
        zs = [rtk_tmp[i][2] for i in range(len(rtk_tmp))]
        # ts = [rtk_tmp[i][-1] for i in range(len(rtk_tmp))]
        # ax[0].scatter(xs, ys, c='k', s=1, label='RTK')
        # ax[1].plot(zs, 'k', label='RTK')
        o_xs = [self.data[i][0]for i in range(len(self.data))]
        o_ys = [self.data[i][1]for i in range(len(self.data))]
        o_zs = [self.data[i][2]for i in range(len(self.data))]
        # o_ts = [self.data[i][-1]for i in range(len(self.data))]
        o_t = [i*len(rtk_tmp)/len(self.data) for i in range(len(self.data))]
        # ax[0].scatter(o_xs, o_ys, c='r', s=20, label='EKF')
        # ax[1].scatter(o_t, o_zs, c='r', s=10, label='EKF')
        # ax[0].legend(fontsize=16)

        ax.scatter(xs, ys, c='k', s=1, label='RTK')
        ax.scatter(o_xs, o_ys, c='r', s=20, label='EKF')
        ax.legend(fontsize=16)

        # ax.plot(zs, 'k', label='RTK')
        # ax.scatter(o_t, o_zs, c='r', s=10, label='EKF')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        #
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.set_box_aspect(1)
        plt.savefig('rtk_odom_sanity.png', transparent=True)
        # plt.savefig('rtk_alt_sanity.png', transparent=True)

        fig, ax = plt.subplots(1, 3, figsize=(15,6))
        ahrs_data = self.dbc.getFrom('q, u, a, t, v_a, v_b, v_g, a_x, a_y, a_z, time', f'ahrs_data_{self.db_name}')
        ahrs_tmp = [quat2euler(ahrs_data[i][0],ahrs_data[i][1],ahrs_data[i][2],ahrs_data[i][3]) for i in range(len(ahrs_data))]
        als = [ahrs_tmp[i][0] for i in range(len(ahrs_tmp))]
        bes = [ahrs_tmp[i][1] for i in range(len(ahrs_tmp))]
        gas = [ahrs_tmp[i][2] for i in range(len(ahrs_tmp))]
        ax[0].scatter([i for i in range(len(als))], als, c='k', s=1, label='AHRS')
        ax[1].scatter([i for i in range(len(bes))], bes, c='k', s=1, label='AHRS')
        ax[2].scatter([i for i in range(len(gas))], gas, c='k', s=1, label='AHRS')
        o_tmp = [quat2euler(self.data[i][3], self.data[i][4], self.data[i][5], self.data[i][6]) for i in range(len(self.data))]
        o_as = [o_tmp[i][0]for i in range(len(o_tmp))]
        o_bs = [o_tmp[i][1]for i in range(len(o_tmp))]
        o_gs = [o_tmp[i][2]for i in range(len(o_tmp))]
        o_t = [i*len(ahrs_tmp)/len(self.data) for i in range(len(self.data))]
        ax[0].scatter(o_t, o_as, c='r', s=1, label='EKF')
        ax[1].scatter(o_t, o_bs, c='r', s=1, label='EKF')
        ax[2].scatter(o_t, o_gs, c='r', s=1, label='EKF')
        plt.savefig('ahrs_ekf_sanity.png')

        # tmp = [[self.data[i][-2], self.data[i][0], self.data[i][1], self.data[i][2], o_gs[i], o_bs[i], o_as[i]] for i in range(len(self.data))]
        # if os.path.isfile('imageData.txt'):
        #     os.remove('imageData.txt')
        # for line in tmp:
        #     with open('imageData.txt', 'a') as f:
        #         vals = ','.join([str(x) for x in line])
        #         # print(vals)
        #         f.write(vals+"\n")
        print('ping')


if __name__ == '__main__':
    dir_path = os.path.join(os.path.expanduser('~'), 'parsed_flight')
    db_name = 'flight_data'
    # dbc = dbConnector(os.path.join(dir_path,db_name))
    tst = birdsEye(db_name=db_name, img_dir=dir_path, apriltags=True)

    tst.parseFlightDatabase()
    # tst.detectionProcess()
