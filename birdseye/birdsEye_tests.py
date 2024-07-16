#!/usr/bin/env python3

import cv2
import time
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
# from AMI_ContourClassFamily import Contour
from birdsEye import *
from dbConnector import dbConnector
from utilities import *

XRES = 1920
YRES = 1080

tgts = [[0,0], \
        [1/4*(XRES-1),0], [-1/4*(XRES-1),0], \
        [0,-1/4*(YRES-1)], [0,1/4*(YRES-1)], \
        [1/4*(XRES-1),1/4*(YRES-1)], [-1/4*(XRES-1),-1/4*(YRES-1)], \
        [1/4*(XRES-1),-1/4*(YRES-1)], [-1/4*(XRES-1),1/4*(YRES-1)], \
        [1/2*(XRES-1),0], [-1/2*(XRES-1),0], \
        [0,-1/2*(YRES-1)], [0,1/2*(YRES-1)], \
        [1/2*(XRES-1),1/2*(YRES-1)], [-1/2*(XRES-1),-1/2*(YRES-1)], \
        [1/2*(XRES-1),-1/2*(YRES-1)], [-1/2*(XRES-1),1/2*(YRES-1)], \
        [-3/2*(XRES-1),-3/2*(YRES-1)], [3/2*(XRES-1),3/2*(YRES-1)], \
        [3/2*(XRES-1),-3/2*(YRES-1)], [-3/2*(XRES-1),3/2*(YRES-1)], \
        [3/2*(XRES-1),0], [-3/2*(XRES-1),0], \
        [0,-3/2*(YRES-1)], [0,3/2*(YRES-1)]]

K = [[3686.410575174854, 0.0, 1100.2760525469466], \
     [0.0, 3705.5188854027347, 745.8037372198796], \
     [0.0, 0.0, 1.0]]
CH = 3

offFrame = 0
annotate = 0
database = 1

shift_point = lambda vertex, res, scalex, scaley: [int(vertex[0] + scalex*res[0]), int(vertex[1] + scaley*res[1])]

def targetGenerate(img, pos, tgt_radius, test_object):
    return shift_point(pos, (XRES,YRES), 2, 2)


def offFrameCheck_Test(test_object, tgts):
    print('birdsEye._2DFrameCheck test')
    print(len(tgts), 'test_points')
    test_object._2DFrameVertices = [shift_point(vertex, (XRES,YRES), 1.5, 1.5) for vertex in test_object._2DFrameVertices]
    # img = np.zeros((4*YRES, 4*XRES))
    # cv2.rectangle(img, test_object._2DFrameVertices[0], test_object._2DFrameVertices[2], 31, -1)
    for i,tgt in enumerate(tgts):
        if i == 0:
            print('test points 1-17 are inside frame; pass if point is returned as valid')
            print('index test_point    pass/fail \n')
        elif i == 17:
            print('test points 18+ are outside frame; pass if point is not returned as valid')
            print('index test_point    pass/fail \n')
        print(i+1, shift_point(tgt, (XRES, YRES), 1/2, 1/2), end='')
        img = np.zeros((4*YRES, 4*XRES))
        tgt = targetGenerate(img, tgt, 20, test_object)
        ret, reproj = test_object._2DFrameCheck([[1 if i == j else 0 for i in range(4)] for j in range(4)], \
                                        [[1 if i == j else 0 for i in range(4)] for j in range(4)], \
                                        [[1 if (i == j or i == 3 ) else 0 for i in range(4)]for j in range(4)], \
                                        [tgt])
        cv2.rectangle(img, test_object._2DFrameVertices[0], test_object._2DFrameVertices[2], 0.125, -1)
        cv2.circle(img, tgt, 20, 1, -1)
        # img = img[test_object._2DFrameVertices[1][1]:test_object._2DFrameVertices[2][1], test_object._2DFrameVertices[0][0]:test_object._2DFrameVertices[1][0]]
        # print('here: ', img.shape)
        img = cv2.resize(img, (XRES//2, YRES//2))
        # cv2.imwrite(f'/home/mwmaster/catch/test_point_all.jpg', img)
        cv2.imshow('sample', img)
        cv2.waitKey(0)

        if i <= 16:
            assert ret == [tgt]
            print('    pass')
            # assert test_object.edge == -1
        else:
            assert ret == []
            print('    pass')
            # assert test_object.edge == 0
        if i ==16:
            print()
    # img = img[test_object._2DFrameVertices[1][1]:test_object._2DFrameVertices[2][1], test_object._2DFrameVertices[0][0]:test_object._2DFrameVertices[1][0]]
    # img = cv2.resize(img, (XRES//2, YRES//2))
    # cv2.imwrite(f'/home/mwmaster/catch/test_point_all.jpg', img)
    ret, reproj = test_object._2DFrameCheck(np.eye(4), \
                        [shift_point(tgt, (XRES,YRES), 2, 2) for tgt in tgts])
    print('\n batch test; pass if only test points 1-17 are returned \n')
    assert ret == [shift_point(tgt, (XRES,YRES), 2, 2) for tgt in tgts][:17]
    print('pass: \n', np.array([shift_point(i, (XRES,YRES), -3/2, -3/2)for i in ret]))
    return ret


def annotate_Test(test_object, tgts):
    print('birdsEye.annotate test')
    test_object.annotate(tgts, 'jpg')


if __name__ == '__main__':
    dir_path = '/home/mwmaster/parsed_flight/'
    save_name = os.path.join(dir_path, 'data')
    db_name = 'flight_data'
    dbc = dbConnector(os.path.join(dir_path,db_name))
    tst = birdsEye(dbc, db_name=db_name, img_dir=dir_path, save_name=save_name)
    # tst.K = np.array([[1 if (i == j or i == 3 ) else 0 for i in range(4)]for j in range(4)])

    if offFrame:
        tst.T_WB = np.eye(4)
        tst.T_BC = np.eye(4)
        offFrameCheck_Test(tst, tgts)

    if annotate:
        tst.T_WB = np.eye(4)
        tst.T_BC = np.eye(4)
        annotate_Test(tst, [shift_point(tgt, (XRES,YRES), 1/2, 1/2) for tgt in tgts[:17]])

    if database:
        # tst.T_BC = np.eye(4)
        tst.parseFlightDatabase()
