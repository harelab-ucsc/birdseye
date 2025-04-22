import os
import glob2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from collections import defaultdict
import random


class TileLoader:
    
    def __init__(self, 
        label_file='labels.txt', 
        tile_size=(224, 224), 
        edge_buffer=81, 
        use_heatmaps=False, 
        include_negatives=True, 
        balance_ratio=1.0
        ):

        self.tile_height, self.tile_width = tile_size
        self.label_file = label_file
        self.edge_buffer = edge_buffer
        self.use_heatmaps = use_heatmaps
        self.include_negatives = include_negatives
        self.balance_ratio = balance_ratio  # ratio of negatives to retain


    def load_labels(self, label_file, image_filename):
        """Parse labels from a text file. Each line: idx, filepath, [x,y,class]"""
        labels = []
        tmp = os.path.join(os.path.split(image_filename)[0], label_file)
        load_name = glob2.glob(tmp)[0]
        try:
            with open(load_name, "r") as f:
                for line in f:
                    _, path, data = line.strip().split()
                    x_str, y_str, class_str = data.split(',')
                    if os.path.basename(path) == os.path.basename(image_filename):
                        labels.append((float(x_str), float(y_str), float(class_str)))
        except Exception as e:
            print(f"      [Label Load Error] {e}")
        return labels


    def search_labels(self, x, y, labels):
        tile_labels = []
        th, tw = self.tile_height, self.tile_width

        for entry in labels:
            lx, ly, cls_str = entry
            if x <= lx < x+tw and y <= ly < y+th:
                tile_labels.append(cls_str)
        return tile_labels


    def tile_image_and_label(self, image_np, labels):
        h, w = image_np.shape[:2]
        th, tw = self.tile_height, self.tile_width
        buffer = self.edge_buffer

        tiles = []
        label_list = []

        for y in range(buffer, h - buffer - th + 1, th):
            for x in range(buffer, w - buffer - tw + 1, tw):
                tile = image_np[y:y+th, x:x+tw]
                tile_labels = self.search_labels(x, y, labels)

                if len(tile_labels) > 0:
                    tiles.append(tile)
                    if self.use_heatmaps:
                        heatmap = np.zeros((th, tw, 1), dtype=np.float32)
                        for lx, ly, cls_str in labels:
                            if x <= lx < x+tw and y <= ly < y+th:
                                px, py = int(lx - x), int(ly - y)
                                heatmap[py, px, 0] = cls_str
                        label_list.append(heatmap)
                    else:
                        label_list.append(1.0)
                elif self.include_negatives:
                    if random.random() < self.balance_ratio:
                        tiles.append(tile)
                        if self.use_heatmaps:
                            label_list.append(np.zeros((th, tw, 1), dtype=np.float32))
                        else:
                            label_list.append(0.0)
                    else:
                        continue
        return tiles, label_list


    def tf_tile_fn(self, image_path):

        def pyfunc(image_path):
            image_path_str = image_path.numpy().decode("utf-8")
            image = tf.io.read_file(image_path_str)
            image = tf.io.decode_png(image, channels=3).numpy()

            labels = self.load_labels(self.label_file, image_path_str)
            tiles, classes = self.tile_image_and_label(image, labels)

            if len(tiles) == 0: # add spacer to filter away later
                return (
                    np.zeros((0, self.tile_height, self.tile_width, 3), dtype=np.uint8),
                    np.zeros((0, self.tile_height, self.tile_width, 1), dtype=np.float32) if self.use_heatmaps
                    else np.zeros((0,), dtype=np.float32)
                )

            tiles = np.array(tiles, dtype=np.uint8)

            if self.use_heatmaps:
                labels = np.array(classes, dtype=np.float32)
            else:
                labels = np.array(classes, dtype=np.float32).reshape(-1)

            return tiles, labels

        tiles, labels = tf.py_function(
            pyfunc,
            [image_path],
            [tf.uint8, tf.float32]
        )

        tiles.set_shape([None, self.tile_height, self.tile_width, 3])
        if self.use_heatmaps:
            labels.set_shape([None, None, None, 1])
        else:
            labels.set_shape([None])

        dataset = tf.data.Dataset.from_tensor_slices((tiles, labels))

        # Filter out empty tile sets
        dataset = dataset.filter(lambda x, y: tf.shape(x)[0] > 0)

        return dataset


    def augment(self, image, label):
        image = tf.image.convert_image_dtype(image, tf.float32)

        m = tf.random.uniform([5])
        n = tf.random.uniform([], minval=-1.0, maxval=1.0)

        if m[0] < 0.5:
            image = tf.image.adjust_brightness(image, n * m[0] / 2)
        if m[1] < 0.5:
            image = tf.image.adjust_contrast(image, 1 + n * m[1])
        if m[2] < 0.5:
            image = tf.image.random_hue(image, 0.02)
            image = tf.image.random_saturation(image, 0.95, 1.05)
        if m[3] < 0.5:
            image = tf.image.flip_left_right(image)
        if m[4] < 0.5:
            image = tf.image.flip_up_down(image)

        image = tf.image.convert_image_dtype(image, tf.uint8)
        return image, tf.expand_dims(label, -1)


    def build_dataset(self, file_list, label_file, batch_size, buffer_size=64, repeat=True, augment=False):
        ds = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(file_list, dtype=tf.string))
        ds = ds.flat_map(self.tf_tile_fn)

        if augment:
            ds = ds.map(self.augment, num_parallel_calls=tf.data.AUTOTUNE)
        else:
            ds = ds.map(lambda x, y: (x, tf.expand_dims(y, -1)), num_parallel_calls=tf.data.AUTOTUNE)

        if repeat:
            ds = ds.repeat()

        ds = ds.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

# ========== Class Weights Helper ========== #

def get_class_weights(file_list, tile_loader):
    """
    Computes class weights accounting for:
    - all tiled frames (positive or negative)
    - balance_ratio governing negative sampling
    """
    total_pos = 0
    total_neg = 0

    for i, image_path in enumerate(file_list):
        print(f'  Progress: {i+1}/{len(file_list)} images', end='\r')
        try:
            image_path_str = str(image_path)
            image = tf.io.decode_png(tf.io.read_file(image_path_str), channels=3).numpy()
            labels = tile_loader.load_labels(tile_loader.label_file, image_path_str)
            tiles, label_list = tile_loader.tile_image_and_label(image, labels)

            for lbl in label_list:
                if tile_loader.use_heatmaps:
                    if np.any(lbl > 0):
                        total_pos += 1
                    else:
                        total_neg += 1
                else:
                    if lbl == 1.0:
                        total_pos += 1
                    else:
                        total_neg += 1
        except Exception as e:
            print(f"[Weight Estimation Skipped] {image_path}: {e}")
            continue

    print(f"\n\n📏 Final class sample counts: pos={total_pos}, neg={total_neg}")
    y_true = [0] * total_neg + [1] * total_pos
    weights = compute_class_weight('balanced', classes=np.unique(y_true), y=y_true)
    return {int(cl): float(w) for cl, w in zip(np.unique(y_true), weights)}


def visualize_tiles(dataset, heatmap=False, num_tiles=16):
    count = 0
    for images, labels in dataset.unbatch():
        if count >= num_tiles:
            break

        try:
            img = tf.cast(images, tf.uint8).numpy()

            plt.figure(figsize=(4, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.title("Tile")
            plt.axis('off')

            plt.subplot(1, 2, 2)
            if heatmap:
                label = tf.squeeze(labels).numpy()
                plt.imshow(label, cmap='hot')
                plt.title("Heatmap")
            else:
                label_val = labels.numpy() if tf.rank(labels) == 0 else labels.numpy()[0]
                plt.text(0.5, 0.5, f"Class: {int(label_val)}", ha='center', va='center', fontsize=16)
                plt.title("Label")
                plt.axis('off')

            plt.tight_layout()
            plt.show()
            count += 1
        except Exception as e:
            print(f"Skipped tile due to error: {e}")
