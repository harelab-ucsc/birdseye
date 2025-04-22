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
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2


memory = 25


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
        # print(e)
        print('  leaving reproj as None: ', e)

    try:
        if april_3D is not None:
            april_3D = np.array(april_3D)
        if clicks_3D is not None:
            clicks_3D = np.array(clicks_3D)
        if april_3D.shape == clicks_3D.shape:
            bproj = april_3D - clicks_3D
        else:
            print(f'  inconsistent arrays for backprojection calculation:  clicks_3D: {clicks_3D.shape},  april_3D: {april_3D.shape}')
    except AttributeError as e:
        # print(e)
        print('  leaving bproj as None: ', e)

    return bproj, reproj


# GUI button click tracking - define outside the class
button_regions = {
    "prev": ((30, 30), (180, 100)),
    "next": ((200, 30), (350, 100)),
    "quit": ((370, 30), (520, 100))
}
button_clicked = None

def mouse_click(event, x, y, flags, param):
    global button_clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        for name, ((x1, y1), (x2, y2)) in button_regions.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                button_clicked = name


class birdsEye():
    def __init__(self, **kwargs):
        plt.ion()
        self.img_dir = kwargs.pop('img_dir', None)
        self.save_name = os.path.join(self.img_dir, 'labels')
        self.det_name = os.path.join(self.img_dir, 'detections')
        self.db_name = kwargs.pop('db_name', None)
        self.sensor = kwargs.pop('sensor', 'cam0')
        self.dbc = dbConnector(os.path.join(self.img_dir, self.db_name))
        self.dbc.boot(self.db_name, self.sensor)

        self.apriltags = kwargs.pop('apriltags', None)
        self.stats = kwargs.pop('stats', None)
        self.plot = kwargs.pop('plot', None)
        self.detect = kwargs.pop('detect', None)
        self.manual = kwargs.pop('manual', None)
        print(self.detect)

        # self.model_path = kwargs.pop('model_path', os.path.join(os.path.expanduser('~'),'ucsc_512_384_13.tflite'))
        # self.model = tflite.Interpreter(model_path=self.model_path, num_threads=4)
        # self.model.allocate_tensors()
        # self.input_details = self.model.get_input_details()
        # self.output_details = self.model.get_output_details()

        self.radalt = None
        self.rtk_tracker = [0]*4
        self.RTK_watchdog = None
        self.data = None
        self.frame_index = None
        self.clear_flag = None

        # camera specs
        tmp = self.getParameters(self.sensor)
        self.res = [int(tmp[1][0]), int(tmp[1][1])]
        self.K = np.array([[tmp[2][0],0.0,tmp[2][2]], \
                           [0.0,tmp[2][1],tmp[2][3]], \
                           [0.0,0.0,1.0]])
        self.K += np.array([[  0.0,   0.0,   0.0],
                            [  0.0,   0.0,   0.0],
                            [  0.0,   0.0,   0.0]])
        self.D = np.array(tmp[3])
        self.T_WC = None
        self.T_IC = np.array(tmp[4])
        self.T_IC = self.ned_to_enu_se3(self.T_IC)

        self.tx = 0
        self.ty = 0
        self.tz = 0
        self.rr = 4
        self.rp = -5
        self.ry = 0
        self.mod = 0.125
        r_adj = R.from_euler('xyz', \
                              [self.rr*self.mod, self.rp*self.mod, self.ry*self.mod], \
                              degrees=True).as_matrix()
        t_adj = np.array([self.tx,
                          self.ty,
                          self.tz])
        self.T_IC[:3,3] = r_adj@self.T_IC[:3,3]
        self.T_IC[:3,3] = t_adj + self.T_IC[:3,3]
        self.T_IC[:3,:3] = r_adj@self.T_IC[:3,:3]

        self._2DFrameVertices = ((0,0), \
                                 (self.res[0] - 1, 0), \
                                 (self.res[0] - 1, self.res[1] - 1), \
                                 (0, self.res[1] - 1))
        self.noise = (81.5, 81.5)  # projection sigma, in the order (sigma_x, sigma_y)
        self.scale = 1  # parameter to tune buffer width; self.scale*self.noise
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

        if self.plot:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d')
            plt.show(block=False)

        if self. apriltags:
            self.at_detector = Detector(
                families="tag36h11",
                nthreads=2,
                quad_decimate=1.0,
                quad_sigma=0.0,
                refine_edges=1,
                decode_sharpening=0.25,
                debug=0
                )
            self.april_2D = []  # list of apriltag detection pixel coordinates
            self.april_3D = []  # list of apriltag detection world coordinates

        if self.detect:
            root = os.path.join(os.path.expanduser('~'),'ros2_ws/src/birdseye/models')
            default = os.path.join(root, 'birdseye_960_600_009.weights.h5')
            tmp = kwargs.pop('model_path', None)
            if tmp is not None:
                 self.model_path = tmp
            else:
                self.model_path = default
            self.IMG_HEIGHT = int(self.model_path.split('_')[-2])
            self.IMG_WIDTH = int(self.model_path.split('_')[-3])
            self.IMG_CHANNELS = 3
            self.model = self.generator()
            self.model.load_weights(self.model_path)
        self.clicks_2D = []  # list of framewise click pixel coordinates
        self.clicks_3D = []  # list of framewise click world coordinates

        if self.stats:
            self.gt_c_bproj = []
            self.gt_a_bproj = []
            self.gt_a_reproj = []
            self.bproj = []  # list of framewise (april_3D - clicks_3D)
            self.reproj = []  # list of framewise (clicks_2D - april_2D)
        # self.contour = Contour(res=(self.res[1], self.res[0]))


    # Define a custom detection head (binary classification: object present or not)
    def detection_head(self, inputs):
        x = tf.keras.layers.GlobalAveragePooling2D()(inputs)  # Convert feature map to vector
        x = tf.keras.layers.Dense(256, activation="relu")(x)  # Fully connected layer
        x = tf.keras.layers.Dropout(0.5)(x)  # Regularization
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)  # Binary detection (0 or 1)
        return outputs


    def pretrained_backbone(self, inp):
        x = tf.keras.ops.cast(inp, "float32")
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
        backbone = MobileNetV2(input_shape=(self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS), include_top=False, weights="imagenet")
        x = backbone(x)  # Extract features without updating backbone weights
        outputs = self.detection_head(x)  # Apply binary detection head
        return outputs


    def generator(self):
        inp = tf.keras.Input(shape=(self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS), name='inp_layer')
        out = self.pretrained_backbone(inp)
        return tf.keras.Model(inputs=inp, outputs=out)


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
        eulers = R.from_quat(frame[3:7]).as_euler('xyz')
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
                p = np.array([p[0], p[1], 1]).T

                # Transform pixel in Camera coordinate frame
                pc = np.linalg.inv(self.K) @ p
                pc = np.hstack((pc,1.0))
                pc = np.array([[0,-1,0,0],[-1,0,0,0],[0,0,1,0],[0,0,0,1]])@pc
                # print(pc)

                # Transform pixel in World coordinate frame
                pw = self.T_WC @ pc

                # Find a ray from camera to 3d point
                vector = pw - self.T_WC[:,3]
                unit_vector = vector / np.linalg.norm(vector)

                # Point scaled along this ray
                p3D = self.T_WC[:,3] - self.radalt*unit_vector
                List3D.append(p3D.tolist())

        return List3D


    def _3Dto2D(self, List3D):
        """
        Projects 3D points from world coordinates into 2D image coordinates.

        - Assumes points are given in NED coordinates.
        - Uses the camera's intrinsic matrix (K) and world-to-camera transform (T_WC).

        :param List3D: List of Nx3 or Nx4 3D points in the world frame.
        :return: Nx2 list of projected 2D image coordinates.
        """
        List2D = []

        try:
            if not List3D:
                return List2D  # Return empty list if no points
        except ValueError:
            pass

        List3D = np.array(List3D, dtype=np.float64)

        # Ensure points are homogeneous (Nx4)
        if List3D.shape[1] == 3:
            List3D = np.hstack((List3D, np.ones((List3D.shape[0], 1))))  # Add w=1

        # Transform world points into the camera frame
        cam_frame_points = np.linalg.inv(self.T_WC)@List3D.T  # 4xN result
        cam_frame_points = np.array([[0,-1,0,0],[-1,0,0,0],[0,0,1,0],[0,0,0,1]])@cam_frame_points

        # Apply intrinsic matrix to project into image plane
        projected = self.K@cam_frame_points[:-1, :]  # Remove homogeneous w

        # Normalize homogeneous coordinates
        projected /= projected[2]  # Normalize by depth (z)

        # Collect results as Nx2 pixel coordinates
        List2D = projected[:2].T.tolist()

        return List2D


    def _2DBoxCheck(self, pts, box='frame', stats=False):
        if stats:
            backproj = []

        if box == 'frame':
            valid_mask =  self.inQuadrilateralCheck(self._2DFrameVertices, pts)
        elif box == 'inner':
            valid_mask =  self.inQuadrilateralCheck(self._2DInnerBound, pts)
        elif box == 'outer':
            valid_mask =  self.inQuadrilateralCheck(self._2DOuterBound, pts)

        valid_ind = np.nonzero(valid_mask)
        valid_pts = np.array(pts)[valid_mask]
        valid_pts = valid_pts.tolist()
        # print('r', pts)
        # print('v', valid_pts)
        if stats:
            backproj = self._2Dto3D(valid_pts)
            return valid_pts, valid_ind, backproj
        else:
            return valid_pts, valid_ind, None


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


    def apriltag_detect(self, img):
        # print('  apriltag_detect')
        px_ret = None
        world_ret = None
        state = 0

        params = [self.K[0,0], self.K[1,1], self.K[0,2], self.K[1,2]]

        results = self.at_detector.detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                                     estimate_tag_pose=True,
                                     camera_params=params,
                                     tag_size=0.5)

        if results:
            px_ret = []
            world_ret = []
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

                # extract the pose estimate of the tag in the camera reference frame
                pose = np.eye(4)
                pose[:3,:3] = r.pose_R
                pose[:3,3] = np.squeeze(r.pose_t)

                # catch data
                px_ret.append([cX,cY])
                world_ret.append([pose, r.pose_err])
            # print('    returning results')
            return state, px_ret, world_ret, img
        else:
            # print('    returning empty')
            return state, px_ret, world_ret, img


    def detector(self, img):
        # print('  apriltag_detect')
        img = tf.convert_to_tensor(img)
        img = tf.image.resize(img, size=(self.IMG_HEIGHT,self.IMG_WIDTH))
        ret_raw = self.model.predict(tf.expand_dims(img, axis=0), verbose=0)
        ret_raw = ret_raw[0][0]**1.1
        print(f'  detection results: {ret_raw}     -->     ', end=' ')
        if ret_raw < 0.5:
            ret = 0.0
        else:
            ret = 1.0
        print(ret)
        return ret, ret_raw


    def get_stats(self, clicks, click_ind, april_2D, april_3D, clicks_2D, clicks_3D):
        april_reproj = np.array(self._3Dto2D(april_3D)) - np.array(april_2D)
        april_reproj = april_reproj.tolist()

        bproj, reproj = projection_stats(april_2D, april_3D, clicks_2D, clicks_3D)

        try:
            gt_click_bproj = np.array(clicks_3D)[:,:3] - np.array(clicks)[click_ind,:]
            gt_april_bproj = np.array(april_3D)[:,:3] - np.array(clicks)[click_ind,:]
        except IndexError:
            # print('    lost one of the test points... clearing variables')
            gt_click_bproj = None
            gt_april_bproj = None
            # april_reproj = None
            pass

        if self.RTK_watchdog:
            # print('    healthy RTK; saving projection stats')
            if bproj is not None:
                self.bproj += bproj.tolist()
            if gt_click_bproj is not None:
                self.gt_c_bproj += gt_click_bproj.tolist()
            if gt_april_bproj is not None:
                self.gt_a_bproj += gt_april_bproj.tolist()
            if reproj is not None:
                self.reproj += reproj.tolist()
            if april_reproj is not None:
                self.gt_a_reproj += april_reproj


            with open(os.path.join(self.img_dir, 'out_dict.pkl'), 'wb') as f:
                out_dict = {}

                out_dict['bproj'] = self.bproj
                out_dict['reproj'] = self.reproj
                tmp_c = np.squeeze(np.array(self.gt_c_bproj))
                tmp_a = np.squeeze(np.array(self.gt_a_bproj))
                out_dict['gt_c_bproj'] = tmp_c.tolist()
                out_dict['gt_a_bproj'] = tmp_a.tolist()
                tmp_a = np.squeeze(np.array(self.gt_a_reproj))
                out_dict['gt_a_reproj'] = tmp_a.tolist()

                pickle.dump(out_dict, f)


    def annotator(self, outer, inner, rect, frame, ret):
        if self.RTK_watchdog:
            if len(outer) > 0:
                if len(inner) > 0:
                    # print('    clicks within InnerBound; annotating frame True')
                    for pt in inner:
                        self.annotate(frame[-5], [pt[0], pt[1], 1.0])
                    if self.plot:
                        cv2.putText(rect, 'Label:', (1300,80), \
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 3)
                        cv2.putText(rect, '1.0', (1750,80), \
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3)
                elif len(inner) == 0:
                    # print('    click detected in frame buffer region; "Test"')
                    # self.annotate(frame[-5], "Test")
                    if self.plot:
                        cv2.putText(rect, 'Label:', (1300,80), \
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 3)
                        cv2.putText(rect, 'Test', (1600,80), \
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 3)
            else:
                # self.annotate(frame[-5], 0.0)
                if self.plot:
                    cv2.putText(rect, 'Label:', (1400,80), \
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 3)
                    cv2.putText(rect, '0.0', (1750,80), \
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)

            if self.apriltags:
                if april_2D is not None:
                    self.annotate(frame[-5], 1.0, save_name=os.path.join(self.img_dir,'april_labels'))
                else:
                    self.annotate(frame[-5], 0.0, save_name=os.path.join(self.img_dir,'april_labels'))

            if self.detect:
                self.annotate(frame[-5], ret, save_name=os.path.join(self.img_dir,'results'))

        else:
            print(f'    skipping annotation: bad RTK_STATUS, {frame[-5]}')


    def annotate(self, img_file, label, save_name=None):
        if save_name is None:
            save_name = self.save_name

        if self.clear_flag is None:
            f = open(f"{save_name}.txt", "w")
            f.close()
            self.clear_flag = True

        f = open(f"{save_name}.txt", "a")
        line = f'{self.frame_index} {img_file} '

        try:  # if label is an iterable
            vals = ','.join([str(x) for x in label])
            # print(vals)
        except TypeError:  # else
            vals = str(label)

        line += vals
        line += '\n'
        # print(line)
        f.write(line)
        f.close()
        print(f'    Annotation saved: {self.frame_index}, {save_name}.txt, {label}')


    def ned_to_enu_se3(self, pose_ned):
        R_ned_to_enu = np.array([[0, 1,  0],
                                [1, 0,  0],
                                [0, 0, -1]])

        T_ned_to_enu = np.eye(4)
        T_ned_to_enu[:3, :3] = R_ned_to_enu

        pose_enu = T_ned_to_enu @ pose_ned @ T_ned_to_enu.T
        return pose_enu


    def frameProcessSetup(self, frame, clks):
        # make homogeneous coordinates for clicks wrt drone pose and radalt
        self.radalt = frame[8]
        self.correct_altitude(frame)

        # convert pose to 4x4 homogeneous transform
        T_WI_ENU = poseRowToTransform(frame[:7])  # our base link maps from the world origin to the base link

        self.T_WC = T_WI_ENU@self.T_IC

        clicks = np.hstack((clks, np.ones_like(clks[:,0]).reshape(-1,1)*(self.T_WC[2,3]-self.radalt)))

        if frame[7] == 3:
            self.rtk_tracker[0] += 1
            self.RTK_watchdog = 1
        elif frame[7] == 2:
            self.rtk_tracker[1] += 1
            self.RTK_watchdog = 0
        elif frame[7] == 1:
            self.rtk_tracker[2] += 1
            self.RTK_watchdog = 0
        else:
            self.rtk_tracker[3] += 1
            self.RTK_watchdog = 0

        if self.plot:
            self.framePlotterSetup(frame, clicks, T_WI_ENU)

        return clicks


    def framePlotterSetup(self, frame, clicks, T_WI):
        _ = plotTransform(self.ax, T_WI, colors=['m','y','c'], labels=['INS x-axis','INS y-axis', 'INS z-axis'])
        _ = plotTransform(self.ax, self.T_WC)

        self.ax.set_xlim(T_WI[0,3]-15, T_WI[0,3]+15)
        self.ax.set_xlabel('East (m)')
        self.ax.set_ylim(T_WI[1,3]-15, T_WI[1,3]+15)
        self.ax.set_ylabel('North (m)')
        self.ax.set_zlim(T_WI[2,3]-20, T_WI[2,3]+1)
        self.ax.set_zlabel('Z (m)')

        self.ax.set_title(f'Time: {frame[-1]}')
        self.ax.set_box_aspect([1,1,1])
        self.ax.set_proj_type('ortho')

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

        self.ax.scatter(clicks[:,0], \
                        clicks[:,1], \
                        clicks[:,2], \
                        marker='s', alpha=0.5, c='m', s=32, label='Click')

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
                self.ax.scatter(tmp[1], tmp[0], tmp[2], c=color, alpha=0.1, s=32)
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
                self.ax.scatter(tmp[1], tmp[0], tmp[2], c=color, alpha=0.1, s=32)


    def draw_buttons(self, img, frame_index, total_frames):
        font = cv2.FONT_HERSHEY_SIMPLEX
        for name, ((x1, y1), (x2, y2)) in button_regions.items():
            cv2.rectangle(img, (x1, y1), (x2, y2), (200, 200, 200), -1)
            cv2.putText(img, name.upper(), (x1 + 10, y1 + 50), font, 1.5, (0, 0, 0), 2)

        # Add frame index counter
        cv2.putText(img, f'Frame {frame_index+1}/{total_frames}', (600, 80), font, 1.5, (255, 255, 255), 2)


    def frameProcessPlotter(self, frame, rect, clicks_2D, april_3D):
        rect = cv2.rectangle(rect, \
             [int(i) for i in self._2DInnerBound[0]], \
             [int(i) for i in self._2DInnerBound[2]], \
             (0,255,255), \
             2)
        cv2.putText(rect, f'{frame[-1]}', (30,80), \
            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3)
        if frame[7] == 3:
            color = (0,255,0)
        elif frame[7] == 2:
            color = (0,255,255)
        elif frame[7] == 1:
            color = (0,0,255)
        else:
            color = (0,0,0)
        for click in clicks_2D:
            cv2.circle(rect, [int(click[0]), int(click[1])], 15, color, 3)

        for apr in self._3Dto2D(april_3D):
            cv2.circle(rect, [int(apr[0]), int(apr[1])], 15, color, 3)


        bp = np.array(self.clicks_3D)
        bp = np.squeeze(bp)
        if self.apriltags:
            ap = np.array(self.april_3D)
            ap = np.squeeze(ap)

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

        if self.apriltags:
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


    def parseFlightDatabase(self):
        clks = self.dbc.getFrom('x, y', f"clicks_{self.db_name}")
        clks = np.array(clks)

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

        if self.plot:
            cv2.namedWindow("Window", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Window", self.res[0], self.res[1])

        if self.manual:
            self.manualProcess()
        else:
            self.autoProcess()

        print('\nRTK Service Stats:')
        print(f'    Status 3 (Fix): {self.rtk_tracker[0]} of {sum(self.rtk_tracker)} ({self.rtk_tracker[0]/sum(self.rtk_tracker)})')
        print(f'    Status 2 (Float): {self.rtk_tracker[1]} of {sum(self.rtk_tracker)} ({self.rtk_tracker[1]/sum(self.rtk_tracker)})')
        print(f'    Status 1 (None): {self.rtk_tracker[2]} of {sum(self.rtk_tracker)} ({self.rtk_tracker[2]/sum(self.rtk_tracker)})')
        print(f'    Rare Statuses: {self.rtk_tracker[3]} of {sum(self.rtk_tracker)} ({self.rtk_tracker[3]/sum(self.rtk_tracker)})\n')

        print(f'roll, pitch, yaw adjustments: {self.rr}, {self.rp}, {self.ry} (mod: {self.mod})')


    def manualProcess(self):
        i = 0
        total_frames = len(self.data)
        while True:
            frame = self.data[i]
            self.frame_index = i
            print(f'\nFrame: {i+1} of {total_frames}')

            clicks = self.frameProcessSetup(frame, clks)
            img = None
            bproj = None
            reproj = None
            state = None
            april_2D = None
            april_3D = None
            clicks_2D = None
            clicks_3D = None

            if self.radalt > 3.0:
                # changing to my filepath
                # modified_img_path = frame[-5].replace('/home/mwmaster/', '/media/akorycki/Data/')
                modified_img_path = frame[-5]
                img = cv2.imread(modified_img_path)

                # rectify image distortion
                rect = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)

                if self.apriltags:
                    state, april_2D, tag_pose, rect = self.apriltag_detect(rect)
                    april_3D = self._2Dto3D(april_2D)
                    if april_3D is not None:
                        self.april_3D += april_3D

                if self.detect:
                    ret, ret_raw = self.detector(rect)
                    self.annotate(frame[-5], ret, save_name=os.path.join(self.img_dir,'results'))

                clicks_2D = self._3Dto2D(clicks)
                inner, i_ind, _ = self._2DBoxCheck(clicks_2D, box='inner')
                outer, o_ind, _ = self._2DBoxCheck(clicks_2D, box='outer')
                clicks_2D, click_ind, clicks_3D = self._2DBoxCheck(clicks_2D, stats=self.stats)
                if clicks_3D is not None:
                    self.clicks_3D += clicks_3D

                # optional statistics generation step
                if self.stats:
                    self.get_stats(clicks, click_ind, april_2D, april_3D, clicks_2D, clicks_3D)

                # annotation step
                self.annotator(outer, inner, rect, frame, ret)

                # optional plotting step
                if self.plot:
                    self.frameProcessPlotter(frame, rect, clicks_2D, april_3D)

                    if self.detect:
                        bot = (0, 0, 255)
                        vec = (0, 2.55, -2.55)
                        tmp = int(ret_raw*100)
                        c1 = (0, tmp*vec[1] + bot[1], tmp*vec[2] + bot[2])
                        c2 = (0, ret*100*vec[1] + bot[1], ret*100*vec[2] + bot[2])
                        cv2.putText(rect, f'CNN: ', (950,1190), \
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 3)
                        cv2.putText(rect, f' {ret_raw:.04f} -> ', (1150,1190), \
                            cv2.FONT_HERSHEY_SIMPLEX, 3, c1, 3)
                        cv2.putText(rect, f'{ret}', (1750,1190), \
                            cv2.FONT_HERSHEY_SIMPLEX, 3, c2, 3)

                    self.fig.canvas.draw_idle()
                    plt.pause(0.01)
                    self.ax.cla()

                    self.draw_buttons(rect, i, total_frames)
                    cv2.imshow("Window", rect)

                    global button_clicked
                    button_clicked = None
                    cv2.setMouseCallback("Window", mouse_click)

                    timeout = 0
                    while button_clicked is None and timeout < 100:
                        key = cv2.waitKey(50)
                        timeout += 1

                        if key == ord('d'):
                            button_clicked = "next"
                        elif key == ord('a'):
                            button_clicked = "prev"
                        elif key == ord('q'):
                            button_clicked = "quit"

                    if button_clicked == "next":
                        i = min(i + 1, total_frames - 1)
                    elif button_clicked == "prev":
                        i = max(i - 1, 0)
                    elif button_clicked == "quit":
                        break


    def autoProcess(self):
        for i, frame in enumerate(self.data):
            print(f'\nframe: {i+1} of {len(self.data)}')
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

            if self.radalt > 3.0:
                # changing to my filepath
                # modified_img_path = frame[-5].replace('/home/mwmaster/', '/media/akorycki/Data/')
                modified_img_path = frame[-5]
                img = cv2.imread(modified_img_path)

                # rectify image distortion
                rect = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)

                if self.apriltags:
                    state, april_2D, tag_pose, rect = self.apriltag_detect(rect)
                    april_3D = self._2Dto3D(april_2D)
                    if april_3D is not None:
                        self.april_3D += april_3D

                if self.detect:
                    ret, ret_raw = self.detector(rect)
                    self.annotate(frame[-5], ret, save_name=os.path.join(self.img_dir,'results'))
                    if self.plot:
                        bot = (0, 0, 255)
                        vec = (0, 2.55, -2.55)
                        tmp = int(ret_raw*100)
                        c1 = (0, tmp*vec[1] + bot[1], tmp*vec[2] + bot[2])
                        c2 = (0, ret*100*vec[1] + bot[1], ret*100*vec[2] + bot[2])
                        cv2.putText(rect, f'CNN: ', (950,1190), \
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 3)
                        cv2.putText(rect, f' {ret_raw:.04f} -> ', (1150,1190), \
                            cv2.FONT_HERSHEY_SIMPLEX, 3, c1, 3)
                        cv2.putText(rect, f'{ret}', (1750,1190), \
                            cv2.FONT_HERSHEY_SIMPLEX, 3, c2, 3)

                clicks_2D = self._3Dto2D(clicks)
                inner, i_ind, _ = self._2DBoxCheck(clicks_2D, box='inner')
                outer, o_ind, _ = self._2DBoxCheck(clicks_2D, box='outer')
                clicks_2D, click_ind, clicks_3D = self._2DBoxCheck(clicks_2D, stats=self.stats)
                if clicks_3D is not None:
                    self.clicks_3D += clicks_3D

                # optional statistics generation step
                if self.stats:
                    self.get_stats(clicks, click_ind, april_2D, april_3D, clicks_2D, clicks_3D)

                # annotation step
                self.annotator(outer, inner, rect, frame, ret)

                # optional plotting step
                if self.plot:
                    self.frameProcessPlotter(frame, rect, clicks_2D, april_3D)

                    self.fig.canvas.draw_idle()
                    plt.pause(0.01)
                    p = os.path.expanduser('~')
                    p = os.path.join(p, 'catch', 'tmp', f'2d_{str(self.frame_index).rjust(5,str(0))}.png')
                    # self.fig.savefig(p)
                    self.ax.cla()

                    cv2.imshow("Window", rect)
                    cv2.waitKey(200)
                    cv2.imwrite(p, rect)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-S", "--src_dir", help="path to source directory (default: parsed_flight)")
    parser.add_argument("-s", "--stats", action='store_true', help="Boolean, whether or not to derive projection stats (default: False)")
    parser.add_argument("-p", "--plot", action='store_true', help="Boolean, whether or not to plot visualizations (default: False)")
    parser.add_argument("-a", "--apriltags", action='store_true', help="Boolean, whether or not to detect apriltags (default: False)")
    parser.add_argument("-d", "--detect", action='store_true', help="Boolean, whether or not to run a loaded AI detector (default: False)")
    parser.add_argument("-m", "--manual", action='store_true', help="Boolean, whether or not to manually advance frames (default: False)")
    parser.add_argument("-M", "--model_path", help="path to trained detection model (default: birdseye/models/birdseye_960_600_009.weights.h5)")
    # parser.add_argument("-pr", "--playback-rate", help="Float, whether or not to detect apriltags (default: False)")

    args = vars(parser.parse_args())

    if args['src_dir'] is not None:
        dir_path = os.path.join(os.path.expanduser('~'), args['src_dir'])
    else:
        dir_path = os.path.join(os.path.expanduser('~'), 'parsed_flight')

    print(f'Processing data from {dir_path}')

    db_name = 'flight_data'
    tst = birdsEye(
        db_name=db_name,
        img_dir=dir_path,
        apriltags=args['apriltags'],
        plot=args['plot'],
        detect=args['detect'],
        manual=args['manual'])

    tst.parseFlightDatabase()
