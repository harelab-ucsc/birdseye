import cv2
import argparse
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

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-b", "--bags", required=True, help="Path(s) to the bag(s)")
# args = vars(ap.parse_args())

c0K = np.array([[3735.2983834164324, 0.0, 848.6860906986433],
                  [0.0, 3729.9647614383052, 631.3744387583913],
                  [0, 0, 1]])
c0dist = np.array([-0.4558699204912608, 0.08050367904783852, -0.004116759227388499, 0.0036714155730349692])
c1K = np.array([[3761.72887708, 0.0, 935.8625819],
                  [0.0, 3760.03527496, 513.33656615],
                  [0, 0, 1]])
c1dist = np.array([-0.44721163,  0.26811723,  0.0015342,   0.00293593])


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
        self._mask = None
        self.mask_res = (384, 512)
        self.mask = np.zeros(self.mask_res, dtype='uint8')
        self.draw = False


    def monitor_frame_index(self):
        self.frame_index %= len(self.images)


    def click_mask(self, coord):
        mask = np.zeros(self.image.shape[:2])
        mask[coord] = 255
        return mask


    def check_segments(self):
        for coord in self.refPt:
            for segval in self.segvals:
                tmp = np.zeros(self.mask_res, dtype='uint8')
                tmp[self.segments == segval] = 255
                try:
                    if tmp[coord] == 255:
                        self.mask = cv2.bitwise_or(tmp, self.mask)
                except IndexError:
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


    def load_frame(self):
        self.image = cv2.imread(self.images[self.frame_index])
        self.image = cv2.undistort(self.image, self.K, self.dist)
        self.image = cv2.resize(self.image, self.mask_res[::-1])
        save_name = self.images[self.frame_index][:-4]
        self._mask = Mask(save_name=self.save_name, \
                        resolution=self.image.shape[:2], \
                        rle_encoding=self.rle)
        self._mask.load(self.frame_index)
        self.mask = self._mask.channels[:,:,self._mask.index]
        self.segments = slic(img_as_float(self.image), n_segments=800, \
                            sigma=5, slic_zero=True, start_label=0)
        self.segvals = np.unique(self.segments)
        # self.reset_mask()


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
                print('next channel')

            elif key == ord('a'):
                cmds.append(['prev',])
                print('previous channel')

            elif key == ord('='):  # '=' because neighbor to '-'
                cmds.append(['class_up',])
                print('up a class')

            elif key == ord('-'):
                cmds.append(['class_dn',])
                print('down a class')

            elif key == ord('s'):
                cmds.append(['save', self.frame_index])
                print('save mask')

            elif key == ord('l'):
                cmds.append(['load', self.frame_index])
                print('load mask')

            # Quit command
            elif key == ord("q"):
                print('quit')
                sys.exit()

            self._mask.update(cmds)

            if key in [ord('a'), ord('d'), ord('l')]:
                self.mask = self._mask.channels[:,:,self._mask.index]


if __name__ == '__main__':
    dir_path = "C:\\Users\\mwmasters\\Documents\\APL-subVision-Shapiro\\preUCSC\\waterline_20210609_release\\data\\data\\GOPRO_FremontCut_1\\"
    images = glob2.glob(dir_path + "*.jpg")
    save_name = os.path.join(dir_path, dir_path.split(os.sep)[-2])
    obj = SLICAnnotator(images=images, save_name=save_name)
    obj.frameProcess()
