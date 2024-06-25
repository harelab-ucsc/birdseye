#!/usr/bin/env python3

import numpy as np
import csv
import utm
import os
import pickle
from cf_triad import *
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, Image, NavSatFix
from std_msgs.msg import String
from SLICAnnotator import offlineSLICAnnotator
import glob2
from dbConnector import dbConnector
from utilities import *
from AMI_ContourClassFamily import Contour
from pupil_apriltags import Detector


memory = 25


def apriltag_detect(img, gray):
    # print('apriltag_detect')
    ret = []
    state = 0

    # options = apriltag.DetectorOptions(families="tag36h11")
    # detector = apriltag.Detector(options)
    # results = detector.detect(gray)  # returns empty list if no targets

    # Changed april tag dector to pupil-labs. Increasing nthreads seem to help not have the segmentation fault issue.
    at_detector = Detector(
        families="tag36h11",
        nthreads=2,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0
        )

    results = at_detector.detect(gray)

    cv2.namedWindow("Window", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Window", 512, 384)

    if results:
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
            ret.append([[cX,cY], [ptA,ptB,ptC,ptD]])
        # show the output image after AprilTag detection
    cv2.imshow("Window", img)
    cv2.waitKey(30)
    return state, ret


class birdsEye():
    def __init__(self, dbc, **kwargs):
        plt.ion()
        self.img_dir = kwargs.pop('img_dir', None)
        self.db_name = kwargs.pop('db_name', None)
        self.sensor = kwargs.pop('sensor', 'cam0')
        self.dbc = dbConnector(os.path.join(self.img_dir, self.db_name))
        self.dbc.boot(self.db_name, self.sensor)

        self.data = None
        # self.images = None
        self.frame_index = None
        # self.pose_time = None

        # camera specs
        tmp = self.getParameters(self.sensor)
        self.res = [int(tmp[1][0]), int(tmp[1][1])]
        self.K = np.array([[tmp[2][0],0.0,tmp[2][2]],[0.0, tmp[2][1], tmp[2][3]],[0.0,0.0,1.0]])
        self.T_WC = None
        self._2DFrameVertices = ((0,0), \
                                 (self.res[0] - 1, 0), \
                                 (self.res[0] - 1, self.res[1] - 1), \
                                 (0, self.res[1] - 1))
        self._3DFrameVertices = None
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        plt.show(block=False)

        self.tgts = None
        self.april_2D = []
        self.april_3D = []
        self.bproj = []
        self.reproj = []
        self.contour = Contour(res=(self.res[1], self.res[0]))


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


    # def imgCheck(self, i):
    #     flag = False
    #     img_time = None
    #     tmp = [0,0]
    #     if i == len(self.poses) - 1:
    #         return flag, img_time
    #     tmp[0] = self.poses[i][-2] - self.img[-1] <= 0
    #     tmp[1] = self.poses[i+1][-2] - self.img[-1] > 0
    #     try:
    #         if sum(tmp) == 2:
    #             # print('here')
    #             flag = True
    #             img_time = self.img[-1]
    #             self.img = self.images.pop(0)
    #             self.frame_index +=1
    #         elif not tmp[0]:
    #             # print('catch up to imu')
    #             self.img = self.images.pop(0)
    #             self.frame_index += 1
    #             flag, img_time = self.imgCheck(i)
    #         else:
    #             # print('img time ahead of pose time')
    #             pass
    #     except IndexError as e:
    #         print(e)
    #         print('    INFO: list of image filenames is depleted. Passing.')
    #     return flag, img_time


    def annotate(self, pts, encoding, save_name=None):
        if not save_name:
            save_name = os.path.join(self.img_dir, self.img_dir.split(os.sep)[-2])
        images = glob2.glob(self.img_dir + f"*.{encoding}")
        # above line can read direct from db as well
        sA = offlineSLICAnnotator(images=images, save_name=save_name)
        for i, image in enumerate(images):
            sA.frameProcess(pts)
            sA.frame_index = i


    def parseFlightDatabase(self):
        clicks = self.dbc.getFrom('x, y', f"clicks_{self.db_name}")
        clicks = np.array(clicks[-3:])
        print("clicks: \n", clicks)

        self.data = self.dbc.getFrom('x, y, z, q, u, a, t, rtk_fix, save_loc, time', f'{self.sensor}_images_{self.db_name}')
        # params = self.dbc.getFrom(f"sensorID, resolution, intrinsics1, intrinsics2, extrinsics", f"parameters_{self.db_name}")
        # self.images = self.dbc.getFrom('save_loc, rtk_fix, time', f'{self.sensor}_images_{self.db_name}')
        save_name = os.path.join(self.img_dir, self.img_dir.split(os.sep)[-2])
        sA = offlineSLICAnnotator(images=[i[-2] for i in self.data], save_name=save_name) #, mask_res=self.res[::-1])

        # self.img = self.images.pop(0)
        # self.frame_index = 0

        rtk_tracker = [0,0,0]
        for i, frame in enumerate(self.data):
            print(f'frame: {i+1} of {len(self.data)}')
            # ret, img_time = self.imgCheck(i)
            # print(f'    image frame index {self.frame_index} of {l}')
            sA.frame_index = i
            sA.load_frame()

            self.T_WC = poseRowToTransform(frame[:7])  # our base link maps from the world origin to the base link
            # T = self.T_WB@self.T_BC
            plotTransform(self.ax, self.T_WC)
            clicks_2D = self._3Dto2D(clicks)
            self.ax.scatter(clicks[:,0], clicks[:,1], np.zeros_like(clicks[:,1]),marker='s', alpha=0.5, c='m', s=64, label='Click')
            self._3DFrameVertices = self._2Dto3D(self._2DFrameVertices)
            self.ax.scatter(np.array(self._3DFrameVertices)[:,0], \
                            np.array(self._3DFrameVertices)[:,1], \
                            np.array(self._3DFrameVertices)[:,2], \
                            marker='s', color='k', label='Frame')
            clicks_2D, bproj = self._2DFrameCheck(clicks_2D, stats=True)

            if frame[-3] == 131:
                color = 'g'
                rtk_tracker[0] += 1
            elif tmp[-3] == 67:
                color = 'y'
                rtk_tracker[1] += 1
            else:
                color = 'r'
                rtk_tracker[2] += 1

            if i >= memory :
                for tmp in self.data[(i-memory):i]:
                    if tmp[-3] == 131:
                        color = 'g'
                    elif tmp[-3] == 67:
                        color = 'y'
                    elif tmp[-3] == 3:
                        color = 'r'
                    else:
                        color = 'k'
                    self.ax.scatter(tmp[0], tmp[1], tmp[2], c=color, alpha=0.1, s=32)
            else:
                for tmp in self.data[:i]:
                    if tmp[-3] == 131:
                        color = 'g'
                    elif tmp[-3] == 67:
                        color = 'y'
                    elif tmp[-3] == 3:
                        color = 'r'
                    else:
                        color = 'k'
                    self.ax.scatter(tmp[0], tmp[1], tmp[2], c=color, alpha=0.1, s=32)

            # if ret:
            if frame[2] > 3.0:
                img = cv2.imread(frame[-2])
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                cv2.putText(img, f'{img_time}', (50,100), \
                    cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 2)
                ret, self.tgts = apriltag_detect(img, gray)
                if clicks_2D:
                    save_name = os.path.join(self.img_dir, 'click_masks')
                    sA.frameProcess(clicks_2D, self.frame_index, save_name=save_name)
                    for click in clicks_2D:
                        cv2.circle(img, [int(click[0]), int(click[1])], 25, (255, 0, 0), -1)
                if bproj:
                    self.bproj.append(bproj)

                if self.tgts:
                    cent = [i[0] for i in self.tgts]
                    self.april_2D.append(cent)
                    if clicks_2D:
                        self.reproj.append(cent + clicks_2D)
                    cnts = np.array([i[1] for i in self.tgts])
                    for i, cnt in enumerate(cnts):
                        self.contour.update(cnt)
                        sA._mask.channels[:,:,i] = cv2.resize(self.contour.img, (512,384), cv2.INTER_CUBIC)
                        sA._mask.update([['next',]]) # need to add class adjustment capability
                        sA.mask = sA._mask.channels[:,:,sA._mask.index]
                    sA._mask.update([['save', self.frame_index, None]])
                else:
                    cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("Mask", 512, 384)
                    cv2.imshow('Mask', np.zeros(self.contour.res))
                    cv2.waitKey(30)
                    cv2.namedWindow("Window", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("Window", 512, 384)
                    cv2.imshow("Window", img)
                    cv2.waitKey(30)

            if self.tgts:
                self.april_3D.append(self._2Dto3D([i[0] for i in self.tgts]))
            if len(self.bproj) > 1:
                # print('    new click')
                self.ax.scatter(np.array(self.bproj)[:,:,0], \
                                np.array(self.bproj)[:,:,1], \
                                np.array(self.bproj)[:,:,2], \
                                c='b', alpha=0.3, s=64, label='BackProj')
            elif len(self.bproj) == 1:
                # print('    first click')
                self.ax.scatter(self.bproj[0][0][0], \
                                self.bproj[0][0][1], \
                                self.bproj[0][0][2], \
                                c='b', alpha=0.3, s=64)
            else:
                pass  # nothing yet
            if len(self.april_3D) > 1:
                self.ax.scatter(np.array(self.april_3D)[:,:,0], \
                                np.array(self.april_3D)[:,:,1], \
                                np.array(self.april_3D)[:,:,2], \
                                c='g', alpha=0.3, s=64, label='TagDet')
            elif len(self.april_3D) == 1:
                # print('    first aprilTag')
                self.ax.scatter(self.april_3D[0][0][0], \
                                self.april_3D[0][0][1], \
                                self.april_3D[0][0][2], \
                                c='g', alpha=0.3, s=64)
            self.ax.set_xlim(frame[0]-15, frame[0]+15)
            self.ax.set_xlabel('X')
            self.ax.set_ylim(frame[1]-15, frame[1]+15)
            self.ax.set_ylabel('Y')
            self.ax.set_zlim(0, 15)
            self.ax.set_zlabel('Z')
            self.ax.legend()

            self.ax.set_title(f'Time: {frame[-1]}')
            self.ax.set_box_aspect([1,1,1])
            self.ax.set_proj_type('ortho')
            self.fig.canvas.draw_idle()
            plt.pause(0.05)
            self.ax.cla()
            self.tgts = None

        print()
        print(f'RTK Service Stats:')
        print(f'    Status 131: {rtk_tracker[0]} of {sum(rtk_tracker)} ({rtk_tracker[0]/sum(rtk_tracker)})')
        print(f'    Status 67: {rtk_tracker[1]} of {sum(rtk_tracker)} ({rtk_tracker[1]/sum(rtk_tracker)})')
        print(f'    Status 3: {rtk_tracker[2]} of {sum(rtk_tracker)} ({rtk_tracker[2]/sum(rtk_tracker)})')

        out_dict = {}

        print()
        tmp = np.squeeze(np.array(self.april_2D))
        print('Visual Targeting Stats (2D):')
        print(tmp.shape)
        # print(f'    Tag UTM: {tmp.mean(axis=0)} +/- {tmp.std(axis=0)}')
        out_dict['april_2D'] = tmp

        print()
        tmp = np.squeeze(np.array(self.april_3D))
        print('Visual Targeting Stats (3D):')
        print(tmp.shape)
        # print(f'    Tag UTM: {tmp.mean(axis=0)} +/- {tmp.std(axis=0)}')
        out_dict['april_3D'] = tmp

        print()
        tmp = np.squeeze(np.array(self.bproj))
        print('GPS Targeting Stats:')
        print(tmp.shape)
        # print(f'    Click Back-Projection UTM: {tmp.mean(axis=0)} +/- {tmp.std(axis=0)}')
        out_dict['bproj'] = tmp

        print()
        tmp=np.squeeze(np.array(self.reproj))
        print('Reprojection Error Stats:')
        print(tmp.shape)
        out_dict['reproj'] = tmp

        out_dict['clicks'] = clicks

        with open(os.path.join(self.img_dir, 'out_dict.pkl'), 'wb') as f:
            pickle.dump(out_dict, f)


if __name__ == '__main__':
    dir_path = os.path.join(os.path.expanduser('~'), 'parsed_flight')
    save_name = os.path.join(dir_path, 'data')
    db_name = 'flight_data'
    dbc = dbConnector(os.path.join(dir_path,db_name))
    tst = birdsEye(dbc, db_name=db_name, img_dir=dir_path, save_name=save_name)

    tst.parseFlightDatabase()
