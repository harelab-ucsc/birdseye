import numpy as np
import csv
import utm
import os
from simRotTools import *
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, Image, NavSatFix
from std_msgs.msg import String


#TODO: Get the subscriber working with correct time stamps for flight table

class birdsEye():
    def __init__(self, dbc, **kwargs):
        # camera specs defined
        self.K = kwargs.pop('intrinsics', None)
        self.res = kwargs.pop('resolution', None)
        self.dist = kwargs.pop('distortion', None)
        # self.extrinsics = kwargs.pop('extrinsics', None)
        self.dbc = dbc
        self.db_name = kwargs.pop('db_name', None)
        self.sensor = kwargs.pop('sensor', 'cam0')
        self.dbc.boot(self.db_name, self.sensor)
        # self.frame = None
        # self.setupFrame()


    # def setupFrame(self):
    #     frame = [(x, -(self.res[0]-self.K[0,2])) for x in range(0, self.res[1], 32)]
    #     frame += [(x, self.K[0,2]-1) for x in range(0, self.res[1], 32)]
    #     frame += [(0, x-(self.res[0]-self.K[0,2])) for x in range(0, self.res[0], 32)]
    #     frame += [(self.res[1]-1, x-(self.res[0]-self.K[0,2])) for x in range(0, self.res[0], 32)]
    #     self.frame = np.array(frame, dtype=np.float32)
    #     # print(frame.shape)
    #     # self.frame = cv2.undistortPointsIter(frame, self.K, self.dist, None, self.K, (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 0.03))
    #     # self.frame = np.squeeze(self.frame).T
    #     # self.frame = np.concatenate((self.frame, np.ones((1, self.frame.shape[1]))), axis=0)
    #     self.frame = self.frame.astype(int)


    def get3Dfrom2D(self, List2D, K, R, t):
        # From https://math.stackexchange.com/questions/4382437/back-projecting-a-2d-pixel-from-an-image-to-its-corresponding-3d-point
        # List2D : n x 2 array of pixel locations in an image
        # K : Intrinsic matrix for camera
        # R : Rotation matrix describing rotation of camera frame
        #     w.r.t world frame.
        # t : translation vector describing the translation of camera frame
        #     w.r.t world frame
        # [R t] combined is known as the Camera Pose.

        List2D = np.array(List2D)
        List3D = []

        for p in List2D:
            # Homogeneous pixel coordinate
            p = np.array([p[0], p[1], 1]).T;

            # Transform pixel in Camera coordinate frame
            pc = np.linalg.inv(K) @ p

            # Transform pixel in World coordinate frame
            pw = t + (R@pc)

            # Transform camera origin in World coordinate frame
            cam = np.array([0,0,0]).T
            cam_world = t + R @ cam

            # Find a ray from camera to 3d point
            vector = pw - cam_world
            unit_vector = vector / np.linalg.norm(vector)

            # Point scaled along this ray
            p3D = cam_world + t[-1]*vector
            List3D.append(p3D)

        return List3D


    def projectPins2Pix(self, T_WB, T_BC, K, pins):
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


    def findVisiblePins(self, pins):
        #Given a camera resolution in x,y; and a set of projected pins in pixel space, return a bitmap if the
        #pin is visible
        visiblePins = list()
        for pin in pins:
            # if (0 <= pin[0] < self.res[1]) and (-(self.res[0]-self.K[0,2]) <= pin[1] < self.K[0,2]):
            if (0 <= pin[0] < self.res[1]) and (0 <= pin[1] < self.res[0]):
                visiblePins.append(1)
            else:
                visiblePins.append(0)

        return visiblePins


    def poseRowToTransform(pose, rpy=None):
        #Given a row from the db, produce a 4x4 homogeneous transform
        #Return as a 4x4 nparray
        if rpy is None:
            rpy = quat2euler(pose[3], pose[4], pose[5], pose[6])

        rot = euler2dcm(rpy[0], rpy[1], rpy[2])
        translate = [[pose[0]],[pose[1]],[pose[2]]]
        T = np.hstack([rot, translate])
        T = np.vstack([T, [0,0,0,1]])

        return T


    def csv_read(self, csv_file, dbc):
        data = []
        with open(csv_file) as clicks:
            reader = csv.reader(clicks)
            for line in reader:
                # breakdown line
                u = utm.from_latlon(float(line[0]), float(line[1]))
                # health = int(line[-1])
                data.append((u[0], u[1]))#, health))
        self.dbc.insertClicks(f"clicks_{self.db_name}", data)
        self.csv_loaded = True


    def convertAndSave(self, msg, sensor, time):

        """ bender code """

        # vcimg = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        location = os.path.join(self.datadir.path, 'stacks/')
        location = os.path.join(location, sensor)
        # location = os.path.join(location, "img_"+str(time.secs)+".png")
        location = os.path.join(location, "img_"+str(time.secs) + '.' + str(time.nsecs)+".png")
        # cv2.imwrite(location, vcimg)
        # cv2.imshow('frame', vcimg)
        return location
