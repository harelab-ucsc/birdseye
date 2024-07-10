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
    return state, ret, img


class birdsEye():
    def __init__(self, dbc, **kwargs):
        plt.ion()
        self.img_dir = kwargs.pop('img_dir', None)
        self.click_save_name = os.path.join(self.img_dir, 'click_masks')
        self.visual_save_name = os.path.join(self.img_dir, 'apriltag')
        self.db_name = kwargs.pop('db_name', None)
        self.sensor = kwargs.pop('sensor', 'cam0')
        self.dbc = dbConnector(os.path.join(self.img_dir, self.db_name))
        self.dbc.boot(self.db_name, self.sensor)

        self.init_alt = kwargs.pop('init_alt', None)
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

        self.tgts = None
        self.april_2D = []
        self.april_3D = []
        self.bproj = []
        self.bproj_tgt = []
        self.reproj = []
        self.contour = Contour(res=(self.res[1], self.res[0]))


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
            p3D = cam_world + self.T_WC[2,3]*vector
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


    # def annotate(self, pts, encoding, save_name=None):
    #     if not save_name:
    #         save_name = os.path.join(self.img_dir, self.img_dir.split(os.sep)[-2])
    #     images = glob2.glob(self.img_dir + f"*.{encoding}")
    #     # above line can read direct from db as well
    #     sA = offlineSLICAnnotator(images=images, save_name=save_name)
    #     for i, image in enumerate(images):
    #         sA.frameProcess(pts)
    #         sA.frame_index = i


    def parseFlightDatabase(self):
        clicks = self.dbc.getFrom('x, y', f"clicks_{self.db_name}")
        clicks = np.array(clicks)
        print("clicks: \n", clicks)

        self.data = self.dbc.getFrom('x, y, z, q, u, a, t, rtk_fix, save_loc, time', f'{self.sensor}_images_{self.db_name}')
        save_name = os.path.join(self.img_dir, self.img_dir.split(os.sep)[-2])
        sA = offlineSLICAnnotator(images=[i[-2] for i in self.data], save_name=save_name) #, mask_res=self.res[::-1])

        # Create rectification and projection maps
        map1, map2 = cv2.initUndistortRectifyMap(self.K, self.D, None, self.K, (self.res[0], self.res[1]), cv2.CV_32FC1)

        cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Mask", 512, 384)
        cv2.namedWindow("Window", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Window", 512, 384)

        rtk_tracker = [0]*4
        for i, frame in enumerate(self.data):
            print(f'frame: {i+1} of {len(self.data)}')
            self.frame_index = i
            sA.frame_index = i

            if self.init_alt is None:
                self.init_alt = frame[2]
                print(f'    initial altitude: {self.init_alt}')

            self.T_WC = poseRowToTransform(frame[:7])  # our base link maps from the world origin to the base link
            self.T_WC[2,3] -= self.init_alt
            plotTransform(self.ax, self.T_WC)
            clicks_2D = self._3Dto2D(clicks)
            self.ax.scatter(clicks[:,0], clicks[:,1], np.zeros_like(clicks[:,1]),marker='s', alpha=0.5, c='m', s=64, label='Click')
            self._3DFrameVertices = self._2Dto3D(self._2DFrameVertices)
            # if i == 100: print(self._3DFrameVertices)
            self.ax.scatter(np.array(self._3DFrameVertices)[:,0], \
                            np.array(self._3DFrameVertices)[:,1], \
                            np.array(self._3DFrameVertices)[:,2], \
                            marker='s', color='k', label='Frame')
            clicks_2D, bproj = self._2DFrameCheck(clicks_2D, stats=True)

            if frame[-3] == 131:
                color = 'g'
                rtk_tracker[0] += 1
            elif frame[-3] == 67:
                color = 'y'
                rtk_tracker[1] += 1
            elif frame[-3] == 3:
                color = 'r'
                rtk_tracker[2] += 1
            else:
                color = 'k'
                rtk_tracker[3] += 1

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
                    self.ax.scatter(tmp[0], tmp[1], tmp[2] - self.init_alt, c=color, alpha=0.1, s=32)
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
                    self.ax.scatter(tmp[0], tmp[1], tmp[2] - self.init_alt, c=color, alpha=0.1, s=32)

            if frame[2]-self.init_alt > 3.0:
                img = cv2.imread(frame[-2])
                rect = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
                gray = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY)
                cv2.putText(rect, f'{frame[-1]}', (50,100), \
                    cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 2)
                ret, self.tgts, rect = apriltag_detect(rect, gray)
                if clicks_2D:
                    sA.frameProcess(clicks_2D, frame_index=self.frame_index, save_name=self.click_save_name)
                    for click in clicks_2D:
                        cv2.circle(rect, [int(click[0]), int(click[1])], 15, (0, 0, 255), -1)
                    if bproj:
                        self.bproj.append(bproj)

                cv2.imshow("Window", rect)
                cv2.waitKey(30)

                if self.tgts:
                    cent = [i[0] for i in self.tgts]
                    self.april_2D.append(cent)
                    april_3D = self._2Dto3D(cent)
                    self.april_3D.append(april_3D)
                    if clicks_2D:
                        val = [[a[0] - b[0], a[1] - b[1]] for a,b in zip(clicks_2D, cent)]
                        print('    targeting reprojection pixel error:')
                        for v in val:
                            print(f'        {v}, 2-norm: {np.linalg.norm(v)}')
                            self.reproj.append(v)
                    if bproj:
                        val = [[a[0] - b[0], a[1] - b[1]] for a,b in zip(april_3D, bproj)]
                        print('    targeting reprojection metric error:')
                        for v in val:
                            print(f'        {v}, 2-norm: {np.linalg.norm(v)}')
                            self.bproj_tgt.append(v)
                    cnts = np.array([i[1] for i in self.tgts])
                    for i, cnt in enumerate(cnts):
                        self.contour.update(cnt)
                        sA._mask.channels[:,:,i] = cv2.resize(self.contour.img, (512,384), cv2.INTER_CUBIC)
                        sA._mask.update([['next',]]) # need to add class adjustment capability
                        sA.mask = sA._mask.channels[:,:,sA._mask.index]
                    # print(f'    second check: bE.frame_index: {self.frame_index}')
                    sA._mask.update([['save', sA.frame_index, self.visual_save_name]])
                else:
                    cv2.imshow('Mask', np.zeros(self.contour.res))
                    cv2.waitKey(30)
                    # cv2.imshow("Window", img)
                    # cv2.waitKey(30)

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
        print(f'    Rare Statuses: {rtk_tracker[3]} of {sum(rtk_tracker)} ({rtk_tracker[3]/sum(rtk_tracker)})')

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
        print('GPS Targeting Stats (click_bproj vs click):')
        print(tmp.shape)
        # print(f'    Click Back-Projection UTM: {tmp.mean(axis=0)} +/- {tmp.std(axis=0)}')
        out_dict['bproj'] = tmp

        print()
        tmp = np.squeeze(np.array(self.bproj_tgt))
        print('GPS Targeting Stats (tgt vs click_bproj):')
        print(tmp.shape)
        # print(f'    Click Back-Projection UTM: {tmp.mean(axis=0)} +/- {tmp.std(axis=0)}')
        out_dict['bproj_tgt'] = tmp

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
    tst = birdsEye(dbc, db_name=db_name, img_dir=dir_path)

    tst.parseFlightDatabase()
