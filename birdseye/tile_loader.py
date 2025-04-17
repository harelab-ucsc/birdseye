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


    def tile_image_and_label(self, image_np, labels):
        h, w = image_np.shape[:2]
        th, tw = self.tile_height, self.tile_width
        buffer = self.edge_buffer

        tiles = []
        label_list = []

        for y in range(buffer, h - buffer - th + 1, th):
            for x in range(buffer, w - buffer - tw + 1, tw):
                tile = image_np[y:y+th, x:x+tw]
                tile_labels = [cls for lx, ly, cls in labels if x <= lx < x+tw and y <= ly < y+th]

                if tile_labels:
                    tiles.append(tile)
                    if self.use_heatmaps:
                        heatmap = np.zeros((th, tw, 1), dtype=np.float32)
                        for lx, ly, cls in labels:
                            if x <= lx < x+tw and y <= ly < y+th:
                                px, py = int(lx - x), int(ly - y)
                                heatmap[py, px, 0] = cls
                        label_list.append(heatmap)
                    else:
                        label_list.append(tile_labels[0])
                elif self.include_negatives:
                    if random.random() < self.balance_ratio:
                        tiles.append(tile)
                        label_list.append(np.zeros((th, tw, 1), dtype=np.float32) if self.use_heatmaps else 0.0)

        return tiles, label_list


    def tf_tile_fn(self, image_path):

        def pyfunc(image_path):
            image_path_str = image_path.numpy().decode("utf-8")
            image = tf.io.read_file(image_path_str)
            image = tf.io.decode_png(image, channels=3).numpy()

            labels = self.load_labels(self.label_file, image_path_str)
            tiles, classes = self.tile_image_and_label(image, labels)

            tiles = np.array(tiles, dtype=np.uint8)
            labels = np.array(classes, dtype=np.float32).reshape(-1)

            return tiles, labels

        tiles, labels = tf.py_function(
            pyfunc,
            [image_path],
            [tf.uint8, tf.float32]
        )

        tiles.set_shape([None, self.tile_height, self.tile_width, 3])
        labels.set_shape([None])

        return tf.data.Dataset.from_tensor_slices((tiles, labels))


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

def get_tile_level_class_weights(
    paths, label_file,
    image_shape=(1200, 1920),  # (H, W) — approximate if unknown
    tile_size=(224, 224),
    edge_buffer=0
):
    label_txt_paths = []
    for path in paths:
        tmp = os.path.join(path, label_file)
        print(f'      adding: {tmp}')
        label_txt_paths.append(tmp)
    print()
    tile_h, tile_w = tile_size
    img_h, img_w = image_shape

    tiles_x = (img_w - 2 * edge_buffer) // tile_w
    tiles_y = (img_h - 2 * edge_buffer) // tile_h
    tiles_per_image = tiles_x * tiles_y

    # Count positive tiles per image
    positive_tile_map = defaultdict(set)
    total_images = set()

    for path in label_txt_paths:
        with open(path, 'r') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    _, img_path, data = line.strip().split(' ')
                    x_str, y_str, class_str = data.split(',')
                    x, y = float(x_str), float(y_str)
                    if edge_buffer < x < img_w - edge_buffer and edge_buffer < y < img_h - edge_buffer:
                        tile_x = int((x - edge_buffer) // tile_w)
                        tile_y = int((y - edge_buffer) // tile_h)
                        tile_id = (tile_x, tile_y)
                        positive_tile_map[img_path].add(tile_id)
                        total_images.add(img_path)
                except ValueError as e:
                    print(e)
                    continue

    num_pos_tiles = sum(len(tiles) for tiles in positive_tile_map.values())
    num_images = len(total_images)
    total_tiles = num_images * tiles_per_image
    num_neg_tiles = total_tiles - num_pos_tiles

    print(f"📦 Total images:        {num_images}")
    print(f"🧩 Tiles per image:     {tiles_per_image}")
    print(f"✅ Positive tiles:       {num_pos_tiles}")
    print(f"🚫 Estimated negatives:  {num_neg_tiles}")

    class_labels = [0] * num_neg_tiles + [1] * num_pos_tiles
    weights = compute_class_weight('balanced', classes=np.unique(class_labels), y=class_labels)
    return {int(cls): float(w) for cls, w in zip(np.unique(class_labels), weights)}, num_images, tiles_per_image


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
