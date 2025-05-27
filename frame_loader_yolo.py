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
        tmp = os.path.join(os.path.split(image_filename)[0], label_file)
        load_name = glob2.glob(tmp)[0]
        try:
            with open(load_name, "r") as f:
                for line in f:
                    _, path, data = line.strip().split()
                    x_str, y_str, class_str = data.split(',')
                    if os.path.basename(path) == os.path.basename(image_filename):
                        labels.append((float(x_str), float(y_str), int(float(class_str))))
        except Exception as e:
            print(f"      [Label Load Error] {e}")
        return labels

    def encode_labels_to_grid(self, labels):
        target = np.zeros((self.grid_size, self.grid_size, self.anchors, 5 + self.num_classes), dtype=np.float32)
        img_h, img_w = self.image_size
        cell_w = img_w / self.grid_size
        cell_h = img_h / self.grid_size

        for x, y, cls in labels:
            gx = int(x / cell_w)
            gy = int(y / cell_h)
            if gx >= self.grid_size or gy >= self.grid_size:
                continue
            for a in range(self.anchors):
                if target[gy, gx, a, 4] == 0:
                    x_rel = (x % cell_w) / cell_w
                    y_rel = (y % cell_h) / cell_h
                    box_w = 0.1
                    box_h = 0.1
                    cls = 0
                    target[gy, gx, a, 0:4] = [x_rel, y_rel, box_w, box_h]
                    target[gy, gx, a, 4] = 1.0
                    target[gy, gx, a, 5 + cls] = 1.0
                    break
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
        ds = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(file_list, dtype=tf.string))
        ds = ds.map(lambda path: self.tf_frame_fn(path, label_file), num_parallel_calls=tf.data.AUTOTUNE)

        if repeat:
            ds = ds.repeat()

        ds = ds.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds


# import os
# import glob2
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import layers, models, regularizers

# class FrameLoader:
#     def __init__(self, image_size=(416, 416), num_classes=2):
#         self.image_size = image_size
#         self.num_classes = num_classes

#     def load_labels(self, label_file, image_filename):
#         labels = []
#         tmp = os.path.join(os.path.split(image_filename)[0], label_file)
#         load_name = glob2.glob(tmp)[0]
#         try:
#             with open(load_name, "r") as f:
#                 for line in f:
#                     _, path, data = line.strip().split()
#                     x_str, y_str, class_str = data.split(',')
#                     if os.path.basename(path) == os.path.basename(image_filename):
#                         labels.append((float(x_str), float(y_str), int(float(class_str))))
#         except Exception as e:
#             print(f"      [Label Load Error] {e}")
#         return labels

#     def encode_labels_to_grid(self, labels, image_shape=(416, 416), grid_size=13):
#         target = np.zeros((grid_size, grid_size, 5 + self.num_classes), dtype=np.float32)
#         img_h, img_w = image_shape
#         cell_w = img_w / grid_size
#         cell_h = img_h / grid_size

#         for x, y, cls in labels:
#             gx = int(x / cell_w)
#             gy = int(y / cell_h)
#             if gx >= grid_size or gy >= grid_size:
#                 continue

#             if target[gy, gx, 4] == 1.0:
#                 continue  # Skip if already assigned

#             x_rel = (x % cell_w) / cell_w
#             y_rel = (y % cell_h) / cell_h
#             box_w = 0.1
#             box_h = 0.1

#             target[gy, gx, 0:4] = [x_rel, y_rel, box_w, box_h]
#             target[gy, gx, 4] = 1.0
#             target[gy, gx, 5 + int(cls)] = 1.0

#         return target

#     def tf_frame_fn(self, image_path, label_file):
#         def pyfunc(image_path_py):
#             image_path_str = image_path_py.numpy().decode("utf-8")
#             image = tf.io.read_file(image_path_str)
#             image = tf.io.decode_png(image, channels=3).numpy()
#             image = tf.image.resize(image, self.image_size).numpy().astype(np.uint8)

#             labels = self.load_labels(label_file, image_path_str)
#             target = self.encode_labels_to_grid(labels, image_shape=self.image_size, grid_size=13)
#             return image, target

#         image, label = tf.py_function(
#             pyfunc,
#             [image_path],
#             [tf.uint8, tf.float32]
#         )
#         image.set_shape([self.image_size[0], self.image_size[1], 3])
#         label.set_shape([13, 13, 5 + self.num_classes])
#         return image, label

#     def build_dataset(self, file_list, label_file, batch_size, buffer_size=64, repeat=True, augment=False):
#         ds = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(file_list, dtype=tf.string))
#         ds = ds.map(lambda path: self.tf_frame_fn(path, label_file), num_parallel_calls=tf.data.AUTOTUNE)

#         if repeat:
#             ds = ds.repeat()

#         ds = ds.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
#         return ds

# def yolo_model(input_shape=(416, 416, 3), num_classes=2, l2_reg=0.01, dropout_rate=0.5):
#     base_model = tf.keras.applications.MobileNetV2(
#         input_shape=input_shape,
#         include_top=False,
#         weights='imagenet'
#     )
#     base_model.trainable = True

#     x = base_model.output
#     x = layers.Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Dropout(dropout_rate)(x)

#     output = layers.Conv2D(
#         filters=(5 + num_classes),
#         kernel_size=1,
#         padding='same',
#         activation='sigmoid'
#     )(x)

#     return models.Model(inputs=base_model.input, outputs=output)
