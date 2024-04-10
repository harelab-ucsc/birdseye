import numpy as np
import csv
import utm
import os
# from simRotTools import *
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, Image, NavSatFix
from std_msgs.msg import String
from SLICAnnotator import offlineSLICAnnotator
import glob2
from db_utilities import *


#TODO: Get the subscriber working with correct time stamps for flight table

class birdsEye():
    def __init__(self, dbc, **kwargs):
        self.img_dir = kwargs.pop('img_dir', None)
        # camera specs defined
        self.K = kwargs.pop('K', None)
        self.res = kwargs.pop('res', None)
        self.dbc = dbc
        self.db_name = kwargs.pop('db_name', None)
        self.sensor = kwargs.pop('sensor', 'cam0')
        self.dbc.boot(self.db_name, self.sensor)
        self._2DFrameVertices = ((0,0), \
                                 (self.res[0] - 1, 0), \
                                 (self.res[0] - 1, self.res[1] - 1), \
                                 (0, self.res[1] - 1))


    def _2Dto3D(self, T_WB, T_BC, K, List2D):
        # From https://math.stackexchange.com/questions/4382437/back-projecting-a-2d-pixel-from-an-image-to-its-corresponding-3d-point
        # T_WB : World to base link transform, 4x4 nonsingular matrix
        # T_BC : Base link to camera transform, 4x4 nonsingular matrix
        # K : Intrinsic matrix for camera
        # List2D : n x 2 array of pixel locations in an image

        List2D = np.array(List2D)
        List3D = []
        T_WC = T_WB @ T_BC

        for p in List2D:
            # Homogeneous pixel coordinate
            p = np.array([p[0], p[1], 1]).T;

            # Transform pixel in Camera coordinate frame
            pc = np.linalg.inv(K) @ p

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
            List3D.append(p3D)

        return List3D


    def _3Dto2D(self, T_WB, T_BC, K, pins):
        #Given world->base and base->camera 4x4 matrices, as well as a K matrix, and a 2d array of pins, project each pin into image space
        #Return a 2d array of image coordinates corresponding to each pin
        #First, compose the image space. That is, take the world pose from the DB, and use a rigid transform to get the sensor frame

        # changing the dimensions of the click to be a 1x4
        pins = np.concatenate((pins, np.zeros((len(pins), 1)),np.ones((len(pins), 1))), axis=1)

        # projecting the image by x_image = K * T_{wc} * X_utm
        base_frame_pins = np.linalg.inv(T_WB)@pins.T
        sensor_frame_pins = np.linalg.inv(T_BC)@base_frame_pins

        #in the sensor frame, z points down, x backward, and y right
        projected = K@sensor_frame_pins

        #The projected coordinates are now the columns of the array 'projected'
        #Normalize the homogeneous coordinate to 1
        projected_normalized = projected / projected[2]

        #These are the pins as projected into pixel space for this pose
        return projected_normalized[:2].T


    def _2DFrameCheck(self, T_WB, T_BC, K, pts, stats=False):
        if stats:
            backproj = []
        valid_mask =  self.inQuadrilateralCheck(self._2DFrameVertices, pts)
        valid_pts = np.array(pts)[valid_mask]
        valid_pts = valid_pts.tolist()
        if stats:
            backproj = self._2Dto3D(T_WB, T_BC, K, valid_pts)
        if stats:
            return valid_pts, backproj
        else:
            return valid_pts, None


    ### This is not so simple
    # def _3DFrameCheck(self, T_WB, T_BC, K, pts, stats=False):
    #     valid_pts = []
    #     if stats:
    #         reproj = []
    #     frame = self._2Dto3D(T_WB, T_BC, K, self._2DFrameVertices)
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
        clicks = np.array(clicks)
        print('clicks: ', clicks)
        poses = self.dbc.getFrom('x, y, z, q, u, a, t, rtk_time, alt_time, imu_time', f'{self.sensor}_poses_{self.db_name}')
        print('poses: ', poses)
        params = self.dbc.getFrom(f"sensorID, resolution, intrinsics1, intrinsics2, extrinsics", f"parameters_{self.db_name}")
        print('params: ', params)
        images = self.dbc.getFrom('save_loc, rtk_fix, time', f'{self.sensor}_images_{self.db_name}')
        print('images: ', images)
        T_UI= makeAPose(-0.0351, 0.0, 0.28, 180, 0, 0)[0]  # ruler+eye measurements
        T_IC = makeAPose(0.02545, -0.02465, -0.077336, 0, 0, 0)[0]
        pass
