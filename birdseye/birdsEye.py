#!/usr/bin/env python3

import numpy as np
import csv
import utm
import os
from cf_triad import *
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, Image, NavSatFix
from std_msgs.msg import String
from SLICAnnotator import offlineSLICAnnotator
import glob2
from dbConnector import dbConnector
from utilities import *


memory = 150


def apriltag_detect(img, gray):
    print('apriltag_detect')
    ret = []

    options = apriltag.DetectorOptions(families="tag36h11")
    detector = apriltag.Detector(options)
    results = detector.detect(gray)  # returns empty list if no targets

    if results:
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
            print("[INFO] tag family: {}".format(tagFamily))
            ret.append([int(r.center[0]), int(r.center[1])])
        # show the output image after AprilTag detection
        cv2.imshow("Image", img)
        cv2.waitKey(0)
    return ret


class birdsEye():
    def __init__(self, dbc, **kwargs):
        plt.ion()
        self.img_dir = kwargs.pop('img_dir', None)
        self.db_name = kwargs.pop('db_name', None)
        self.sensor = kwargs.pop('sensor', 'cam0')
        self.dbc = dbConnector(os.path.join(self.img_dir, self.db_name))
        self.dbc.boot(self.db_name, self.sensor)

        self.poses = None
        self.images = None
        self.pose_time = None
        self.tgts = None

        # camera specs
        tmp = self.getParameters(self.sensor)
        self.res = tmp[1]
        self.K = np.array([[tmp[2][0],0.0,tmp[2][2]],[0.0, tmp[2][1], tmp[2][3]],[0.0,0.0,1.0]])
        # print(self.K)
        self.T_BC = np.linalg.inv(np.array(tmp[4]))  # tmp[4] = T_cam_imu
        self.T_WB = None
        self._2DFrameVertices = ((0,0), \
                                 (self.res[0] - 1, 0), \
                                 (self.res[0] - 1, self.res[1] - 1), \
                                 (0, self.res[1] - 1))
        self._3DFrameVertices = None
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        plt.show(block=False)


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
                    tmp = string_list_converter(item)
                    if item == elem[-1]:
                        tmp = matrix_list_converter(tmp, (4,4))
                    params.append(tmp)
        return params


    def _2Dto3D(self, List2D):
        # From https://math.stackexchange.com/questions/4382437/back-projecting-a-2d-pixel-from-an-image-to-its-corresponding-3d-point
        # T_WB : World to base link transform, 4x4 nonsingular matrix
        # T_BC : Base link to camera transform, 4x4 nonsingular matrix
        # K : Intrinsic matrix for camera
        # List2D : n x 2 array of pixel locations in an image

        List2D = np.array(List2D)
        List3D = []
        T_WC = self.T_WB @ self.T_BC

        for p in List2D:
            # Homogeneous pixel coordinate
            p = np.array([p[0], p[1], 1]).T;
            # Transform pixel in Camera coordinate frame
            pc = np.linalg.inv(self.K) @ p
            # print(pc, pc.shape)
            pc = np.hstack((pc,1.0))
            # print(pc, pc.shape)

            # Transform pixel in World coordinate frame
            pw = T_WC @ pc

            # Transform camera origin in World coordinate frame
            cam = np.array([0,0,0,1]).T
            cam_world = T_WC @ cam

            # Find a ray from camera to 3d point
            vector = pw - cam_world
            unit_vector = vector / np.linalg.norm(vector)

            # Point scaled along this ray
            p3D = cam_world + T_WC[2,3]*vector
            List3D.append(p3D.tolist())

        return List3D


    def _3Dto2D(self, pins):
        #Given world->base and base->camera 4x4 matrices, as well as a K matrix, and a 2d array of pins, project each pin into image space
        #Return a 2d array of image coordinates corresponding to each pin
        #First, compose the image space. That is, take the world pose from the DB, and use a rigid transform to get the sensor frame

        # changing the dimensions of the click to be a 1x4
        pins = np.concatenate((pins, np.zeros((len(pins), 1)),np.ones((len(pins), 1))), axis=1)

        # projecting the image by x_image = K * T_{wc} * X_utm
        base_frame_pins = np.linalg.inv(self.T_WB)@pins.T
        sensor_frame_pins = np.linalg.inv(self.T_BC)@base_frame_pins

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


    ### This is not so simple
    # def _3DFrameCheck(self, T_WB, T_BC, K, pts, stats=False):
    #     valid_pts = []
    #     if stats:
    #         reproj = []
    #     frame = self._2Dto3D(self._2DFrameVertices)
    #     for pt in pts:
    #         if self.inQuadrilateralCheck(frame, pt):
    #             valid_pts.append(pt)
    #         if stats:
    #             reproj.append(self._3Dto2D(T_WB, T_BC, K, pts))
    #     if stats:
    #         return valid_pts, reproj
    #     else:
    #         return valid_pts, None


    def inQuadrilateralCheck(self, frame, pts):
        n = len(frame)
        frame = np.array(frame)
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


    def imgCheck(self, i):
        flag = False
        tmp = [0,0]
        if i == len(self.poses) - 1:
            return flag
        tmp[0] = self.poses[i][-1] - self.img[-1] <= 0
        tmp[1] = self.poses[i+1][-1] - self.img[-1] > 0
        if sum(tmp) == 2:
            print(' frame found')
            # print('last, now, next: ', self.poses[i][-1], self.img[-1], self.poses[i+1][-1])
            flag = True
            self.img = self.images.pop(0)
        elif not tmp[0]:
            print(' img pop ')
            # print('last, now, next: ', self.poses[i][-1], self.img[-1], self.poses[i+1][-1])
            self.img = self.images.pop(0)
            flag = self.imgCheck(i)
        else:
            print('pose is catching up to self.img')
        return flag


    def annotate(self, pts, encoding):
        images = glob2.glob(self.img_dir + f"*all.{encoding}")
        # above line can read direct from db as well
        save_name = os.path.join(self.img_dir, self.img_dir.split(os.sep)[-2])
        sA = offlineSLICAnnotator(images=images, save_name=save_name)
        for i, image in enumerate(images):
            sA.frameProcess(pts)
            sA.frame_index = i


    def parseFlightDatabase(self):
        clicks = self.dbc.getFrom('x, y', f"clicks_{self.db_name}")
        clicks = np.array(clicks[-3:])
        print("clicks: \n", clicks)

        self.poses = self.dbc.getFrom('x, y, z, q, u, a, t, rtk_fix, rtk_time, alt_time, imu_time', f'{self.sensor}_poses_{self.db_name}')
        params = self.dbc.getFrom(f"sensorID, resolution, intrinsics1, intrinsics2, extrinsics", f"parameters_{self.db_name}")
        self.images = self.dbc.getFrom('save_loc, rtk_fix, time', f'{self.sensor}_images_{self.db_name}')
        print(len(self.poses), len(self.images))
        self.img = self.images.pop(0)

        for i, pose in enumerate(self.poses):
            print(f'frame: {i+1} of {len(self.poses)}')
            # ret = self.imgCheck(i)
            # if ret:
            #     img = cv2.imread(self.img[0])
            #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #     self.tgts = apriltag_detect(img, gray)

            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlim(0,15)
            self.ax.set_zlabel('Z')

            self.T_WB = poseRowToTransform(pose)  # our base link maps from the world origin to the base link
            T = self.T_WB@self.T_BC
            plotTransform(self.ax, T)
            clicks_2D = self._3Dto2D(clicks)
            # self.ax.scatter(clicks[:,0], clicks[:,1], np.zeros_like(clicks[:,1]), marker='x', c='g', s=150)
            self._3DFrameVertices = self._2Dto3D(self._2DFrameVertices)
            self.ax.scatter(np.array(self._3DFrameVertices)[:,0], \
                            np.array(self._3DFrameVertices)[:,1], \
                            np.array(self._3DFrameVertices)[:,2], \
                            marker='s', color='k')
            clicks_2D = self._2DFrameCheck(clicks_2D)

            if i >= memory :
                for tmp in self.poses[(i-memory):i]:
                    if tmp[-4] == 131:
                        color = 'g'
                    elif tmp[-4] == 67:
                        color = 'y'
                    else:
                        color = 'r'
                    self.ax.scatter(tmp[0], tmp[1], tmp[2], c=color, alpha=0.8, s=8)
            else:
                for tmp in self.poses[:i]:
                    if tmp[-4] == 131:
                        color = 'g'
                    elif tmp[-4] == 67:
                        color = 'y'
                    elif tmp[-4] == 3:
                        color = 'r'
                    else:
                        color = 'k'
                    self.ax.scatter(tmp[0], tmp[1], tmp[2], c=color, alpha=0.8, s=8)
            self.ax.set_title(f'RTK time: {pose[-3]} \n RadAlt time: {pose[-2]} \n AHRS time: {pose[-1]}')
            self.ax.set_box_aspect([1,1,1])
            self.ax.set_proj_type('ortho')
            self.fig.canvas.draw_idle()
            plt.pause(0.05)
            self.ax.cla()
            self.tgts = None
