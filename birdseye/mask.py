import numpy as np
import pickle as pkl
from pycocotools.mask import *
import matplotlib.pyplot as plt
import pdb

class Mask:
    def __init__(self, *args, **kwargs):
        self.save_name = kwargs.pop('save_name', None)
        self.load_name = kwargs.pop('load_name', self.save_name)
        self.num_channels = kwargs.pop('num_channels', 200)
        self.num_classes = kwargs.pop('num_classes', 5)  # boat, land, water, sky, other
        self.res = kwargs.pop('resolution', (384, 512))

        self.index = 0
        self.cl = 0
        self.union = np.zeros(self.res, dtype='uint8')
        self.channels = np.zeros(self.res+(self.num_channels,), dtype='uint8')
        self.classes = np.zeros(self.num_channels, dtype='uint8')
        # self.classes[self.index] = 0
        self.rle = kwargs.pop('rle_encoding', False)


    def monitor_index(self):
        self.index %= self.num_channels


    def monitor_class(self):
        self.cl %= self.num_classes


    def update(self, cmds):
        for cmd in cmds:
            if cmd[0] == 'next':
                self.index += 1
                self.monitor_index()
                self.cl = self.classes[self.index]
                print(f'    mask: next (index {self.index}, class {self.classes[self.index]})')

            elif cmd[0] == 'prev':
                self.index -= 1
                self.monitor_index()
                self.cl = self.classes[self.index]
                print(f'    mask: prev (index {self.index}, class {self.classes[self.index]})')

            elif cmd[0] == 'write':
                print('    mask: write')
                self.channels[:,:,self.index] = cmd[1]
                self.classes[self.index] = self.cl

            elif cmd[0] == 'save':
                print('    mask: save')
                self.save(cmd[1], cmd[2])

            elif cmd[0] == 'load':
                print('    mask: load')
                self.load(cmd[1])

            elif cmd[0] == 'class_up':
                self.cl += 1
                self.monitor_class()
                print(f'    mask: class up ({self.cl})')

            elif cmd[0] == 'class_dn':
                self.cl -= 1
                self.monitor_class()
                print(f'    mask: class down ({self.cl})')


    def save(self, frame_index, save_name):
        if save_name is None:
            save_name = self.save_name
        # print(f'    Mask: save: frame_index: {frame_index}, save_name: {save_name}')
        if not self.rle:
            save_dict = {}
            for i in range(self.num_channels):
                if len(np.argwhere(self.channels[:,:,i])):
                    save_dict[f'{i}'] = (self.channels[:,:,i], self.classes[:,i])
            with open(self.save_name+'.pkl', 'ab') as f:
                pkl.dump(save_dict, f)
                print(f'        mask saved: {self.save_name}.pkl')
        else:
            # MOT defines a segmentation annotation entry as
            #  "time_frame id class_id img_height img_width rle"
            f = open(f"{save_name}.txt", "a")
            for i in range(self.num_channels):
                if len(np.argwhere(self.channels[:,:,i])):
                    save_dict = encode(np.asfortranarray(self.channels[:,:,i]))
                    tmp1 = save_dict['size'][0]
                    tmp2 = save_dict['size'][1]
                    tmp3 = str(save_dict['counts'], encoding='utf-8')
                    line = f'{frame_index} <object_id> {self.classes[i]} {tmp1} {tmp2} {tmp3}\n'
                    f.write(line)
            f.close()
            print(f'        mask saved: frame index {frame_index} {save_name}.txt')


    def export_masks(self):
        return self.channels


    def export_classes(self):
        onehot = [[0.0]*self.num_classes]*self.num_channels
        for i, cl in enumerate(self.classes):
            onehot[i][cl] = 1.0
        onehot = np.array(onehot).T
        return onehot


    def load(self, frame_index):
        if not self.rle:
            with open(self.load_name+'.pkl', 'rb') as f:
                ret = pkl.load(f)
                print(f'        mask loaded: {self.load_name}.pkl')
                for i, key in enumerate(ret.keys()):
                    self.channels[:,:,i] = ret[key][0]
                    # self.classes[self.index] = np.zeros_like(self.classes[:,self.index], dtype='uint8')
                    self.classes[self.index] = ret[key][2]

        else:
            try:
                f = open(f"{self.load_name}.txt", "r")
                print(f'        mask loaded: frame index {frame_index}, {self.load_name}.txt')
                i = 0
                while True:
                    # print(i)
                    mask = f.readline()
                    tmp = mask.split()
                    if len(tmp) == 0:
                        break
                    # print('tmp: ', tmp)
                    if int(tmp[0]) == frame_index:
                        self.cl = int(tmp[2])
                        self.classes[i] = int(tmp[2])
                        rle = {'size': (int(tmp[3]), int(tmp[4])), 'counts': tmp[5]}
                        self.channels[:,:,i] = decode(rle).astype('uint8')*255
                        # plt.imshow(self.channels[:,:,i])
                        # plt.show()
                        i += 1
                f.close()
            except FileNotFoundError:
                pass
