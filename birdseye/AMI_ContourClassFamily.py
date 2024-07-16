import cv2
import time
import pickle
import opensimplex as ox
import matplotlib.pyplot as plt
import numpy as np

from scipy.spatial.transform import Rotation as R
from scipy.stats import invwishart, lognorm, special_ortho_group
from scipy.stats import multivariate_normal as mvn
from math import copysign
from copy import deepcopy


# from scipy.stats import multivariate_normal as mvn

# generate a shape (rectangle) or arbitrary dimensions and rotation
# for N iterations:
    # add a draw of simplex noise to the edge of the shape
    # find countours and contour Hu moments
    # log transform and untransformed tests

# XRES = 1024
# YRES = 1024
XRES = 512
YRES = 384

SCALE = 10
FEATURE_SIZE = 10^(2)

M = 10
N = 1000

MAX_ITER = 100

s = 4

filename = f'{time.time()}_XRES{XRES}_YRES{YRES}_M{M}_N{N}.pkl'


def ccw(A,B,C):
    # https://stackoverflow.com/a/61882574
    # used in Contour.contourIntersect
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])


class Contour:
    def __init__(self, iterable=(), **kwargs):
        if isinstance(iterable, dict):
            self.__dict__.update(iterable, **kwargs)
        else:
            # give resolution as (height, width) for numpy, which natively uses (row, column))
            self.res = kwargs.pop('res', (YRES, XRES))
            self.img = kwargs.pop('img', None)

            self.cnt = kwargs.pop('cnt', None)
            self.hry = kwargs.pop('hry', None)
            self.moments = kwargs.pop('moments', None)
            self.hu = kwargs.pop('hu', None)

            self.pos = kwargs.pop('postion', [0,0])  # position
            self.edge = kwargs.pop('edge', 0)  # bool for "edge of frame"
            self.att = kwargs.pop('attitude', [[1,0],[0,1]])
            self.rtn = kwargs.pop('rotation', [[1,0],[0,1]])
            self.lvel = kwargs.pop('velocity', [0,0])  # linear velocity
            self.avel = kwargs.pop('ang_velocity', [0,0])  # angular velocity

            self.feat = kwargs.pop('feature_vector', None)

        self.intersect = False  # no True behavior developed, circa 08/23/2023
        self.CNT0ERROR = 0
        self.tlost = 5


    def update(self, cnt):
        self.img = np.zeros(self.res)
        poly = [np.reshape(cnt, (cnt.shape[0], 1, cnt.shape[1])).astype(np.int32)]
        cv2.drawContours(self.img, poly, -1, 255, 1)
        cv2.fillPoly(self.img, poly, color=255)
        cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Mask", 512, 384)
        cv2.imshow('Mask', self.img)
        cv2.waitKey(150)
        self.getContourData(self.img)


    def getContourData(self, img):
        cnt, hry = cv2.findContours(img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # if len(cnt) == 0:
        #     self.CNT0ERROR = 1
        #     # cv2.imshow('bonk window: failure to find a contour', img)
        #     # cv2.waitKey(0)

        if len(cnt) > 1:  # if cnt is a list of contours
            l = 0
            tmp = None
            val = 0
            for c in cnt:
                if len(c) > val:
                    tmp = c
                    val = len(c)
            cnt = tmp
        self.cnt = np.squeeze(cnt)
        self.hry = hry

        self.moments = cv2.moments(cnt[0])
        self.hu = cv2.HuMoments(self.moments)

        # if not self.CNT0ERROR:
        #     self.moments = cv2.moments(cnt[0])
        #     self.hu = cv2.HuMoments(self.moments)
        # else:
        #     self.moments = None
        #     self.hu = None
        #     # print(f'        getContourData: CNT0ERROR: self.moments = None, self.hu = None')


    def get_featureVector(self):
        feat = []
        try:
            feat.append(self.moments['m10']/self.moments['m00'])  # centroid x position
            feat.append(self.moments['m01']/self.moments['m00'])  # centroid y position
            feat.append(self.moments['m00'])  # area/scale
        except ZeroDivisionError as ze:
            # thrown when the contour is a linear set of pixels at the frame edge
            # print(ze)
            # print(self.cnt.shape, end=' ')
            # print(len(self.cnt))
            if len(self.cnt.shape) == 1:  # contour is single pixel at the edge
                feat.append(self.cnt[0])  # x position
                feat.append(self.cnt[1])  # y position
                feat.append(len(self.cnt))  # area/scale
            else:  # contour is a line at the edge
                # print(self.cnt[:,0])
                # print(self.cnt[:,1])
                feat.append(self.cnt[:,0].mean())  # centroid x position
                feat.append(self.cnt[:,1].mean())  # centroid y position
                feat.append(len(self.cnt))  # area/scale
        except TypeError as te:
            print(te)

        for h in self.hu.flatten():
            # htmp = float(-1* np.copysign(1.0, h) * np.log10(abs(h+1e-12)))
            feat.append(h)  # hu moments 1-7

        # feat.append(max(self.edge, 0))
        # print('    get_featureVector: ', len(feat), feat)
        self.feat = np.array(feat)


    def offFrameCheck(self, cnt):
        val = []
        yres, xres = self.res
        yres -= 1
        xres -= 1
        for v in cnt:
            tmp = np.zeros(4)
            tmp[0] = v[1]*xres >= 0
            tmp[1] = (xres - v[0])*yres >= 0
            tmp[2] = (yres - v[1])*xres >= 0
            tmp[3] = v[0]*yres >= 0
            # print('tmp', tmp, 'tmp.sum()', tmp.sum())
            val.append(tmp.sum())
        val = set(val)

        if 4.0 not in val:  # totally off frame
            # print('        off frame')
            self.edge = 0
            return True
        elif val == {4.0}:
            # print('        in frame')
            self.edge = -1
            # print('            offFrameCheck pass')
        elif any(ele == 4.0 for ele in val):  # partially in frame
            # print('        on edge')
            self.edge = 1
        else:
            print('unexpected behavior in offFrameCheck')
        return False


class simContour(Contour):
    """ for generic auto-generation, like in VertRect subclass,
    we can use a line segment of simplex noise, wrapped to be a
    contour + compute normal to surface at each contour pixel,
    then spline interpolation on the fanned-out normals?
    -> need to define the map/template from segment to contour
    -> could draw 1 noise sample for source contour, then 1
       to add noise to the source
    -> could also seed with simple geometry template, per
       VertRectangle, but for Ellipse, Quadrilateral, Triangle,
       etc. """


    def __init__(self, iterable=(), **kwargs):
        super().__init__(iterable, **kwargs)
        self.cnt_refs = kwargs.pop('cnt_refs', None) # reference contours for intersection check
        self.srcCnt = None
        self.noiseCnt = None
        self.rotCnt = None


    def reset(self):
        if self.rotCnt is not None:
            cnt = deepcopy(self.rotCnt)
            tgt = 'rotCnt.'
        elif self.noiseCnt is not None:
            cnt = deepcopy(self.noiseCnt)
            tgt = 'noiseCnt.'
        else:
            cnt = deepcopy(self.srcCnt)
            tgt = 'srcCnt.'
        return cnt, tgt


    def randomRotation(self):
        print('        randomRotation')
        for i in range(MAX_ITER):
            cnt, _ = self.reset()
            # print(cnt)
            tmp = np.tile(self.pos.astype(np.int32), (cnt.shape[0],1))  # probable inefficiency
            rtn = special_ortho_group.rvs(2)

            cnt -= tmp
            cnt = (rtn@cnt.T).T
            cnt += tmp
            cnt = cnt.astype(np.int32)

            # check for off frame
            ret = self.offFrameCheck(cnt)
            if ret:
                print("            rot check fail")
                if i == MAX_ITER - 1:
                    cnt, tgt = self.reset()
                    print(f'            randomRotation MAX_ITER_TERM; reset to '+tgt)
                continue
            else:  # print('            roll accepted')
                print("            rot check pass")
                self.rotCnt = cnt
                break
        self.update(cnt)


    def randomTranslation(self, cov=400):
        print('        randomTranslation')
        for i in range(MAX_ITER):
            cnt, _ = self.reset()
            tns = mvn.rvs(mean=[0,0], cov=cov)
            tmp1 = np.tile(self.pos.astype(np.int32), (cnt.shape[0],1))  # probable inefficiency
            cnt -= tmp1
            tmp2 = np.tile(tns.astype(np.int32), (cnt.shape[0],1))
            cnt += tmp2
            cnt += tmp1
            # print(cnt)
            # check for off frame
            ret = self.offFrameCheck(cnt)
            if ret:
                print("            tns check fail")
                if i == MAX_ITER - 1:
                    cnt, tgt = self.reset()
                    print(f'            randomTranslation MAX_ITER_TERM; reset to '+tgt)
                continue
            else:  # print('            roll accepted')
                print("            tns check pass")
                self.pos += tns
                break
        self.update(cnt)


    def clear(self):
        self.noiseCnt = None
        self.rotCnt = None


class VertRectangle(simContour):

    def __init__(self, iterable=(), **kwargs):
        super().__init__(iterable, **kwargs)
        self.ul = None
        self.br = None
        self.lx = None
        self.ly = None

        if self.cnt_refs is not None:
            self.intersect = True
            while self.intersect:
                img = self.generate()
                self.intersect = self.contourIntersect()
                if not self.intersect:
                    break
        else:
            self.generate()
        self.srcCnt = deepcopy(self.cnt)


    def generate(self):
        img = np.zeros(self.res)
        boxx = np.random.choice(self.res[1], size=2)
        boxy = np.random.choice(self.res[0], size=2)
        boxx = np.sort(boxx)
        boxy = np.sort(boxy)
        self.pos = np.array([boxx.mean(), boxy.mean()])
        self.ul = (min(boxx), min(boxy))
        self.br = (max(boxx), max(boxy))
        self.ly = max(boxy) - min(boxy)
        self.lx = max(boxx) - min(boxx)

        img = np.zeros(self.res)
        cv2.rectangle(img, self.ul, self.br, 1, -1)
        self.getContourData(img)
        return img


    def contourIntersect(self):
        """ bs is list with VertRectangle objects as elements
        this only workes for vertically-aligned rectangles """
        ret = False  # no intersection
        for cnt in self.cnt_refs:
            if self.ul[0] > cnt.br[0]:  # we are entirely to the right of cnt
                ret = False
            elif self.ul[1] > cnt.br[1]:  # we are entirely below cnt
                ret = False
            elif self.br[0] < cnt.ul[0]:  # we are entirely to the left of cnt
                ret = False
            elif self.br[1] < cnt.ul[1]:  # we are entirely above cnt
                ret = False
            else:
                ret = True
            if ret:
                break
        return ret


    def simplexNoised(self, FEATURE_SIZE, scale=SCALE):
        print('        simplexNoised')
        cnt = deepcopy(self.srcCnt)
        tmp = np.tile(self.pos.astype(np.int32), (cnt.shape[0],1))
        cnt -= tmp

        noisegen = ox.OpenSimplex(time.time_ns())

        for k, val in enumerate(cnt):
            # print(val)
            i, j = val
            # noise = sumOctave(16, i, j, .5, FEATURE_SIZE, 0, 255)
            noise = noisegen.noise2(i/FEATURE_SIZE, j/FEATURE_SIZE)
            # print(f'noise: {noise}')
            if k < self.ly - 1:
                # print('before:', cnt[k, 0])
                cnt[k, 0] += scale * noise
                # print('after:', cnt[k, 0])
            elif k < self.lx + self.ly - 1:
                # print('before:', cnt[k, 1])
                cnt[k, 1] += scale * noise
                # print('after:', cnt[k, 1])
            elif k < self.lx + 2*self.ly - 1:
                # print('before:', cnt[k, 0])
                cnt[k, 0] += scale * noise
                # print('after:', cnt[k, 0])
            else:
                # print('before:', cnt[k, 1])
                cnt[k, 1] += scale * noise
                # print('after:', cnt[k, 1])
        cnt += tmp
        self.noiseCnt = cnt
        # print('res', self.res)
        # print('src', self.srcCnt.max(axis=0), self.srcCnt.min(axis=0))
        # print('cnt', cnt.max(axis=0), cnt.min(axis=0))
        self.offFrameCheck(cnt)
        self.update(cnt)


if __name__ == '__main__':
    print('AMI_simplexNoise ran as main')
    filename = f'{time.time()}_XRES{XRES}_YRES{YRES}_M{M}_N{N}.pkl'

    res = (YRES, XRES)
    data_dict = {}
    data_dict['M'] = M
    data_dict['N'] = N
    # rects = []
    # old = None
    for m in range(M):
        print(f'source {m+1} of {M}')
        compile_dict = {}
        img = np.zeros(res)
        rect = VertRectangle(img, res=res)
        tmp1 = vars(rect)
        keys = list(tmp1.keys())
        keys.remove('img')
        keys.remove('cnt_refs')
        keys.remove('res')
        tmp2 = {}
        for key in keys:
            tmp2[key] = tmp1[key]
        compile_dict['src'] = tmp2

        for n in range(N):
            print(f'    target {n+1} of {N}')
            rect.simplexNoised(FEATURE_SIZE)
            tmp1 = vars(rect)
            tmp2 = {}
            for key in keys:
                tmp2[key] = tmp1[key]
            compile_dict[f'tgt{n}'] = tmp2
        data_dict['src' + str(m)] = compile_dict

    with open(filename, 'wb') as f:
        pickle.dump(data_dict, f)
