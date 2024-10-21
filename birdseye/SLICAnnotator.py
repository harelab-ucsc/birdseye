import cv2
import numpy as np
from copy import deepcopy
import functools
import sys
import glob2
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
# from skimage.exposure import rescale_intensity
from skimage.util import img_as_float
import matplotlib.pyplot as plt
import argparse
import os
from mask import Mask
import matplotlib.pyplot as plt

import pdb


class SLICAnnotator:

    def __init__(self, **kwargs):
        print('SLIC Annotator booting...')
        self.images = kwargs.pop('images', None)
        self.save_name = kwargs.pop('save_name', None)
        self.frame_index = 0
        self.image = None
        self.refPt = []
        self.rmPt = []
        self.segments = None
        self.segvals = None
        self.dist = kwargs.pop('distortion_coefs', np.zeros(4))
        self.K = kwargs.pop('K_matrix', np.eye(3))
        self.rle = kwargs.pop('rle', True)
        self.src_res = kwargs.pop('src_res', (1200,1920))
        self.mask_res = kwargs.pop('mask_res', (384, 512))
        self.mask = np.zeros(self.mask_res, dtype='uint8')
        self.draw = False

        self._mask = Mask(save_name=self.save_name, \
                        resolution=self.mask_res, \
                        rle_encoding=self.rle)


    def monitor_frame_index(self):
        self.frame_index %= len(self.images)


    def click_mask(self, coord):
        mask = np.zeros(self.image.shape[:2])
        mask[coord] = 255
        return mask


    def check_segments(self):
        for coord in self.refPt:
            # print('coord: ', coord)
            for segval in self.segvals:
                tmp = np.zeros(self.mask_res, dtype='uint8')
                tmp[self.segments == segval] = 255
                try:
                    # pdb.set_trace()
                    if tmp[coord] == 255:
                        self.mask = cv2.bitwise_or(tmp, self.mask)
                except IndexError:
                    print('    off-frame')
                    pass

    def rm_cells(self):
        for coord in self.rmPt:
            for segval in self.segvals:
                tmp = np.zeros(self.mask_res, dtype='uint8')
                tmp[self.segments == segval] = 255
                try:
                    if tmp[coord] == 255:
                        tmp = cv2.bitwise_and(self.mask, tmp)
                        self.mask = self.mask - tmp
                except IndexError:
                    pass

    def click_drag_callback(self, event, x, y, flags, param):
        # if the left mouse button was clicked, record the (x, y) coordinates until released
        coord = (y, x)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.draw = True
            self.refPt.append(coord)
        elif event == cv2.EVENT_MOUSEMOVE and self.draw:
            self.refPt.append(coord)
        elif event == cv2.EVENT_LBUTTONUP and self.draw:
            self.draw = False
            self.refPt.append(coord)
        elif event == cv2.EVENT_RBUTTONDOWN:  # unmask a cell if right click
            self.rmPt.append(coord)


    def load_frame(self, frame_index=None):
        if frame_index is None:
            frame_index = self.frame_index
        # print('frame_index: ', frame_index)
        # print('prep')

        self.image = None
        self.image = cv2.imread(self.images[frame_index])
        # print('ping')
        # self.image = cv2.undistort(self.image, self.K, self.dist)
        self.image = cv2.resize(self.image, self.mask_res[::-1])
        # save_name = self.images[frame_index][:-4]
        # self._mask = Mask(save_name=self.save_name, \
        #                 resolution=self.image.shape[:2], \
        #                 rle_encoding=self.rle)
        self._mask.load(frame_index)
        # print('sA.load_frame: bonk')
        self.mask = self._mask.channels[:,:,self._mask.index]
        self.segments = slic(img_as_float(self.image), n_segments=800, \
                            sigma=5, slic_zero=True, start_label=0)
        self.segvals = np.unique(self.segments)
        # self.reset_mask()
        # print('made it')


    def reset_mask(self):
        self.mask = np.zeros(self.image.shape[:2], dtype='uint8')


    def frameProcess(self):
        self.load_frame()
        clone = self.image.copy()
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", self.click_drag_callback)
        cv2.namedWindow("mask")

        while True:
            # display the image and wait for a keypress
            if len(self.refPt) > 0:
                self.check_segments()
                self.refPt = []
            if len(self.rmPt) > 0:
                self.rm_cells()
                self.rmPt = []
            cmds = []
            cv2.imshow("image", mark_boundaries(img_as_float(self.image), self.segments, color=(0.5,0.5,0.5)))
            cv2.imshow("mask", self.mask)

            key = cv2.waitKey(1) & 0xFF

            # client-side commands
            if key == ord('1'):
                self.frame_index -= 1
                self.monitor_frame_index()
                self.load_frame()
                clone = self.image.copy()
                print('previous frame: ', self.images[self.frame_index])

            elif key == ord('2'):
                self.frame_index += 1
                self.monitor_frame_index()
                self.load_frame()
                clone = self.image.copy()
                print('next frame: ', self.images[self.frame_index])

            elif key == ord("e"):
                self.image = clone.copy()
                self.reset_mask()
                print('reset channel')

            # mask commands
            elif key == ord("w"):
                cmds.append(['write', self.mask])
                print('write to channel')

            elif key == ord('d'):
                cmds.append(['next',])
                # print('next channel')

            elif key == ord('a'):
                cmds.append(['prev',])
                # print('previous channel')

            elif key == ord('='):  # '=' because neighbor to '-'
                cmds.append(['class_up',])
                # print('up a class')

            elif key == ord('-'):
                cmds.append(['class_dn',])
                # print('down a class')

            elif key == ord('s'):
                cmds.append(['save', self.frame_index, None])
                # print('save mask')

            elif key == ord('l'):
                cmds.append(['load', self.frame_index])
                # print('load mask')

            # Quit command
            elif key == ord("q"):
                print('quit')
                sys.exit()

            self._mask.update(cmds)

            if key in [ord('a'), ord('d'), ord('l')]:
                self.mask = self._mask.channels[:,:,self._mask.index]


class offlineSLICAnnotator(SLICAnnotator):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def check_segments(self, pts):
        for coord in pts:
            # print(coord)
            x, y = coord
            x = int(x/self.src_res[1]*self.mask_res[1])
            y = int(y/self.src_res[0]*self.mask_res[0])
            # print(x,y)

            for segval in self.segvals:
                tmp = np.zeros(self.mask_res, dtype='uint8')
                tmp[self.segments == segval] = 255
                try:
                    # pdb.set_trace()
                    if tmp[y,x] == 255:
                        self.mask = cv2.bitwise_or(tmp, self.mask)
                except IndexError:
                    print('sA.check_segments: bonk')
                    pass


    def frameProcess(self, pts, frame_index=None, save_name=None):
        if frame_index is None:
            frame_index = self.frame_index
        if save_name is None:
            save_name = self.save_name
        # print('offline frame_index: ', frame_index)
        # print('offline save_name: ', save_name)
        self.load_frame()


        for pt in pts:
            # print(pt)
            self.check_segments([pt])
            self._mask.update([['write', self.mask]])
            # cv2.imshow("image", mark_boundaries(img_as_float(self.image), self.segments, color=(0.5,0.5,0.5)))
            cv2.namedWindow("Click", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Click", 512, 384)
            cv2.imshow("Click", self.mask)
            key = cv2.waitKey(30)
            self._mask.update([['next',]]) # need to add class adjustment capability
            self.mask = self._mask.channels[:,:,self._mask.index]
        self._mask.update([['save', frame_index, save_name]]) # this save will duplicate frame_ids, rather than overwriting old entries


class SLICTranslator(SLICAnnotator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def load_frame(self, frame_index=None):
        if frame_index is None:
            frame_index = self.frame_index
        # print('frame_index: ', frame_index)
        # print('prep')
        # print(self.images[frame_index])

        tmp = None
        tmp = cv2.imread(self.images[frame_index])

        w = tmp.shape[1]
        w = w // 2

        msk = tmp[:, w:, :]
        msk = cv2.resize(msk, self.mask_res[::-1])
        self.image = tmp[:, :w, :]
        p = os.path.join('/', *self.save_name.split('/')[:-1])
        p = os.path.join(p, 'frame_'+str(frame_index).rjust(3,str(0))+'.png')
        cv2.imwrite(p, self.image)
        self.image = cv2.resize(self.image, self.mask_res[::-1])

        tmp = None
        self._mask.channels[:,:,0] = msk[:,:,0]
        self._mask.channels[:,:,1] = msk[:,:,1]
        tmp = cv2.bitwise_or(msk[:,:,0], msk[:,:,1])
        self._mask.channels[:,:,2] = msk[:,:,2]
        tmp = cv2.bitwise_or(tmp, msk[:,:,2])
        self._mask.channels[:,:,3] = cv2.bitwise_not(tmp)

        self._mask.classes[0] = 1
        self._mask.classes[1] = 2
        self._mask.classes[2] = 0
        self._mask.classes[3] = 3

        self.mask = self._mask.channels[:,:,0]
        self.segments = slic(img_as_float(self.image), n_segments=800, \
                            sigma=5, slic_zero=True, start_label=0)
        self.segvals = np.unique(self.segments)


    def frameProcess(self, frame_index=None, save_name=None):
        if frame_index is None:
            frame_index = self.frame_index
        if save_name is None:
            save_name = self.save_name
        # print('offline frame_index: ', frame_index)
        # print('offline save_name: ', save_name)
        self.load_frame()
        self._mask.update([['save', frame_index, save_name]])
        self.frame_index += 1
        self.monitor_frame_index()


    def translate_dataset(self):
        for i in range(len(self.images)):
            self.frameProcess()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dirname', type=str, required=True,
        help='required directory name, path to parent directory of images to annotate')
    parser.add_argument('-f', '--format', nargs='?', const=1, type=str, default='png',
        help='image file format; allowed args are "png" and "jpg". Default: png.')
    args = vars(parser.parse_args())
    print()
    images = glob2.glob(args["dirname"] + "*." + args["format"])
    images.sort()
    print(f'{len(images)} {args["format"]} images found in {args["dirname"]}')

    save_name = os.path.join(args["dirname"], args["dirname"].split(os.sep)[-2])
    print(f'annotations will be saved to: {save_name} \n')
    # obj = SLICAnnotator(images=images, save_name=save_name)
    # obj.frameProcess()

    obj = SLICTranslator(images=images, save_name=save_name)
    obj.translate_dataset()
