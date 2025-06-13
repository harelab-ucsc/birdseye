"Frame Loader for YOLO-style data generation"

import os
import glob2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import keras_cv

class FrameLoader:
    def __init__(self, image_size=(416, 416), num_classes=2, grid_size=13, anchors=5, augment=False):
        self.image_size = image_size
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.anchors = anchors
        self.augment = augment
        self.flip_augmenter = keras_cv.layers.RandomFlip(mode="horizontal_and_vertical")

    def load_labels(self, label_file, image_filename):
        labels = []
        image_path_str = image_filename if isinstance(image_filename, str) else image_filename.numpy().decode("utf-8")

        tmp = os.path.join(os.path.split(image_path_str)[0], label_file)
        try:
            load_name = glob2.glob(tmp)[0]
            with open(load_name, "r") as f:
                for line in f:
                    try:
                        _, path, data = line.strip().split()
                        x_str, y_str, class_str = data.split(',')
                        if os.path.basename(path) == os.path.basename(image_path_str):
                            labels.append((float(x_str), float(y_str), int(float(class_str))))
                    except Exception as sub_e:
                        pass
                       # print(f"[Label Parse Error] {sub_e} in line: {line}")
        except Exception as e:
            pass
            # print(f"[Label Load Error] {e} — from file: {tmp}")

        if not labels:
            pass
            # print(f"[WARNING] No labels found for {os.path.basename(image_path_str)}")

        
    # print(f"Loaded labels for {os.path.basename(image_path_str)}: {labels}")
        return labels

    # def encode_labels_to_grid(self, labels):
    #     target = np.zeros((self.grid_size, self.grid_size, self.anchors, 5 + self.num_classes), dtype=np.float32)
    #     img_h, img_w = self.image_size
    #     cell_w = img_w / self.grid_size
    #     cell_h = img_h / self.grid_size

    #     for x, y, cls in labels:
    #         gx = int(x / cell_w)
    #         gy = int(y / cell_h)
    #         if gx >= self.grid_size or gy >= self.grid_size:
    #             continue
    #         for a in range(self.anchors):
    #             if target[gy, gx, a, 4] == 0:
    #                 x_rel = (x % cell_w) / cell_w
    #                 y_rel = (y % cell_h) / cell_h
    #                 box_w = 0.1
    #                 box_h = 0.1
    #                 cls = 0
    #                 target[gy, gx, a, 0:4] = [x_rel, y_rel, box_w, box_h]
    #                 target[gy, gx, a, 4] = 1.0
    #                 target[gy, gx, a, 5 + cls] = 1.0
    #                 break
    #     print("Target sum:", np.sum(target[..., 4]))
    #     return target
    
    def encode_labels_to_grid(self, labels, original_image_size=(1920, 1200)):
        target = np.zeros((self.grid_size, self.grid_size, self.anchors, 5 + self.num_classes), dtype=np.float32)
        model_w, model_h = self.image_size
        orig_w, orig_h = original_image_size
        cell_w = model_w / self.grid_size
        cell_h = model_h / self.grid_size

        for x, y, cls in labels:
            # Scale (x, y) from original image size to model input size
            x = x * model_w / orig_w
            y = y * model_h / orig_h

            gx = int(x / cell_w)
            gy = int(y / cell_h)

            if gx >= self.grid_size or gy >= self.grid_size:
                continue

            for a in range(self.anchors):
                if target[gy, gx, a, 4] == 0:
                    x_rel = (x % cell_w) / cell_w
                    y_rel = (y % cell_h) / cell_h
                    box_w = 0.1  # Placeholder size
                    box_h = 0.1

                    target[gy, gx, a, 0:4] = [x_rel, y_rel, box_w, box_h]
                    target[gy, gx, a, 4] = 1.0  # objectness
                    if self.num_classes > 1:
                        target[gy, gx, a, 5 + cls] = 1.0  # one-hot class
                    break  # only one anchor per object

        # print("Target sum:", np.sum(target[..., 4]))  # debug: number of positive anchors
        return target


    def tf_frame_fn(self, image_path, label_file):
        def pyfunc(image_path_py):
            image_path_str = image_path_py.numpy().decode("utf-8")
            image = tf.io.read_file(image_path_str)
            image = tf.io.decode_png(image, channels=3).numpy()
            image = tf.image.resize(image, self.image_size).numpy().astype(np.uint8)

            labels = self.load_labels(label_file, image_path_str)

            if self.augment:
                image = self.flip_augmenter(tf.convert_to_tensor(image, dtype=tf.float32)).numpy().astype(np.uint8)

            target = self.encode_labels_to_grid(labels)
            return image, target
            

        image, label = tf.py_function(
            pyfunc,
            [image_path],
            [tf.uint8, tf.float32]
        )
        image.set_shape([self.image_size[0], self.image_size[1], 3])
        label.set_shape([self.grid_size, self.grid_size, self.anchors, 5 + self.num_classes])
        return image, label

    def build_dataset(self, file_list, label_file, batch_size, buffer_size=64, repeat=True, augment=False):
        self.augment = augment
        paths = tf.convert_to_tensor(file_list, dtype=tf.string)
        ds = tf.data.Dataset.from_tensor_slices(paths)

        def has_positive_labels(image_path):
            def _check(path):
                path_str = path.numpy().decode("utf-8")
                labels = self.load_labels(label_file, path_str)
                return len(labels) > 0
            return tf.py_function(_check, [image_path], Tout=tf.bool)

        ds = ds.filter(has_positive_labels)

        ds = ds.map(lambda path: self.tf_frame_fn(path, label_file), num_parallel_calls=tf.data.AUTOTUNE)

        if repeat:
            ds = ds.repeat()

        ds = ds.shuffle(buffer_size).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        return ds


    # def build_dataset(self, file_list, label_file, batch_size, buffer_size=64, repeat=True, augment=False):
    #     self.augment = augment
    #     ds = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(file_list, dtype=tf.string))
    #     ds = ds.map(lambda path: self.tf_frame_fn(path, label_file), num_parallel_calls=tf.data.AUTOTUNE)

    #     if repeat:
    #         ds = ds.repeat()

    #     ds = ds.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    #     return ds
