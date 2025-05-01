import numpy as np
import pickle as pkl
from pycocotools.mask import *
import matplotlib.pyplot as plt
from PIL import Image
import os

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
        onehot = np.zeros((self.num_channels, self.num_classes))
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


class PointMask(Mask):
    def __init__(self, annotation_txt, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.annotation_txt = annotation_txt
        self.annotations = self._parse_annotations()

    def _parse_annotations(self):
        annotations = {}
        with open(self.annotation_txt, 'r') as f:
            for line in f:
                # Each line: frame_index, /path/to/image/file, x, y, class
                parts = line.strip().split(',')
                if len(parts) != 5:
                    continue  # skip malformed lines
                frame_index, img_path, data = parts
                x, y, class_str = data.split(',')
                frame_index = int(frame_index)
                x, y = int(float(x)), int(float(y))  # allow float coordinates
                class_ = int(class_str)
                if frame_index not in annotations:
                    annotations[frame_index] = []
                annotations[frame_index].append({
                    'x': x,
                    'y': y,
                    'class': class_str,
                    'img_path': img_path.strip()
                })
        return annotations

    def load(self, frame_index):
        """Populate mask for the given frame index based on point annotations."""
        # Clear previous masks
        self.channels.fill(0)
        self.classes.fill(0)

        if frame_index not in self.annotations:
            print(f"No annotations found for frame {frame_index}")
            return

        point_data = self.annotations[frame_index]

        # Use class IDs to select channels, or spread across sequential channels
        channel_map = {}  # Map class_id to channel index
        channel_counter = 0

        for point in point_data:
            x, y, cls = point['x'], point['y'], point['class']
            if cls not in channel_map:
                if channel_counter >= self.num_channels:
                    print(f"Warning: exceeded max number of channels ({self.num_channels})")
                    continue
                channel_map[cls] = channel_counter
                self.classes[channel_counter] = cls
                channel_counter += 1

            ch = channel_map[cls]

            if 0 <= x < self.res[1] and 0 <= y < self.res[0]:
                self.channels[y, x, ch] = 255  # Use 255 for visualization compatibility
            else:
                print(f"Point ({x}, {y}) out of bounds for resolution {self.res}")

        print(f"Loaded {len(point_data)} points for frame {frame_index}")

    def get_image_path(self, frame_index):
        """Get image path associated with a frame index (optional helper)."""
        if frame_index in self.annotations and self.annotations[frame_index]:
            return self.annotations[frame_index][0]['img_path']
        return None

    def show_overlay(self, frame_index, image=None):
        """Visualize the overlaid masks on the image."""
        img_path = self.get_image_path(frame_index)

        if image is None and img_path and os.path.exists(img_path):
            image = np.array(Image.open(img_path).resize((self.res[1], self.res[0])))

        if image is None:
            image = np.zeros((*self.res, 3), dtype=np.uint8)

        mask_sum = np.sum(self.channels, axis=2)
        mask_overlay = np.clip(mask_sum, 0, 255).astype(np.uint8)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(image)
        plt.imshow(mask_overlay, cmap='jet', alpha=0.5)
        plt.title("Overlay with Mask Points")
        plt.axis('off')
        plt.show()
