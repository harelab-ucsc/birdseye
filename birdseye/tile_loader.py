import os
import glob2
import random
import pickle

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight
from collections import defaultdict


# -- Elastic transformation helper --
def elastic_transform(image, alpha=34, sigma=4):
    random_state = np.random.RandomState(None)
    h, w = image.shape[:2]

    dx = nd.gaussian_filter((random_state.rand(h, w) * 2 - 1), sigma) * alpha
    dy = nd.gaussian_filter((random_state.rand(h, w) * 2 - 1), sigma) * alpha

    x, y = np.meshgrid(np.arange(w), np.arange(h))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    if image.ndim == 2:
        warped = nd.map_coordinates(image, indices, order=1, mode='reflect')
        return warped.reshape((h, w)).astype(np.float32)

    elif image.ndim == 3:
        channels = []
        for c in range(image.shape[-1]):
            warped = nd.map_coordinates(image[..., c], indices, order=1, mode='reflect')
            warped = warped.reshape((h, w))
            channels.append(warped)
        return np.stack(channels, axis=-1).astype(np.float32)


class TileLoader:
    
    def __init__(self, 
        label_file='labels.txt', 
        tile_size=(224, 224), 
        edge_buffer=81, 
        use_heatmaps=False, 
        include_negatives=True, 
        balance_ratio=1.0,
        negative_mode="random",
        # semisupervized=False
        ):

        self.tile_height, self.tile_width = tile_size
        self.label_file = label_file
        self.edge_buffer = edge_buffer
        self.use_heatmaps = use_heatmaps
        self.include_negatives = include_negatives
        self.balance_ratio = balance_ratio  # ratio of negatives to retain
        self.negative_mode = negative_mode  # options: 'random', 'once_per_image', 'none'
        # self.semisupervised = semisupervised


    def load_labels(self, label_file, image_filename):
        labels = []
        tmp = os.path.join(os.path.split(image_filename)[0], label_file)
        load_name = glob2.glob(tmp)[0]
        try:
            with open(load_name, "r") as f:
                for line in f:
                    _, path, data = line.strip().split()
                    x_str, y_str, cls_str = data.split(',')
                    if os.path.basename(path) == os.path.basename(image_filename):
                        source = "cnn" if label_file == "results.txt" else "human"
                        labels.append((float(x_str), float(y_str), float(cls_str), source))
        except Exception as e:
            print(f"      [Label Load Error] {e}")
        return labels


    def search_labels(self, x, y, labels):
        th, tw = self.tile_height, self.tile_width
        return [
            (lx, ly, cls_str, source)
            for lx, ly, cls_str, source in labels
            if x <= lx < x + tw and y <= ly < y + th
        ]


    def draw_gaussian(self, heatmap, weightmap, px, py, sigma=10, weight=1.0):
        radius = int(3 * sigma)
        size = 2 * radius + 1
        x_coords = np.arange(0, size, 1, float)
        y_coords = x_coords[:, np.newaxis]
        x0 = y0 = radius
        g = np.exp(-((x_coords - x0)**2 + (y_coords - y0)**2) / (2 * sigma**2))

        x1, x2 = px - radius, px + radius + 1
        y1, y2 = py - radius, py + radius + 1

        g_x1, g_x2 = 0, size
        g_y1, g_y2 = 0, size

        if x1 < 0:
            g_x1 = -x1
            x1 = 0
        if y1 < 0:
            g_y1 = -y1
            y1 = 0
        if x2 > self.tile_width:
            g_x2 = size - (x2 - self.tile_width)
            x2 = self.tile_width
        if y2 > self.tile_height:
            g_y2 = size - (y2 - self.tile_height)
            y2 = self.tile_height

        heatmap[y1:y2, x1:x2, 0] = np.maximum(
            heatmap[y1:y2, x1:x2, 0], g[g_y1:g_y2, g_x1:g_x2]
        )
        # if self.semisupervised:
        weightmap[y1:y2, x1:x2, 0] = np.maximum(
            weightmap[y1:y2, x1:x2, 0], weight
        )


    def tile_image_and_label(self, image_np, labels):
        h, w = image_np.shape[:2]
        th, tw = self.tile_height, self.tile_width
        buffer = self.edge_buffer

        tiles = []
        label_list = []

        try:
            for y in range(buffer, h - buffer - th + 1, th):
                for x in range(buffer, w - buffer - tw + 1, tw):
                    tile = image_np[y:y+th, x:x+tw]
                    tile_labels = self.search_labels(x, y, labels)

                    if len(tile_labels) > 0:
                        tiles.append(tile)
                        if self.use_heatmaps:
                            heatmap = np.zeros((th, tw, 1), dtype=np.float32)
                            weightmap = np.zeros((th, tw, 1), dtype=np.float32)
                            for lx, ly, cls_str, source in tile_labels:
                                if x <= lx < x+tw and y <= ly < y+th:
                                    px, py = int(lx - x), int(ly - y)
                                    sigma = 3 if source == "cnn" else 10
                                    weight = 0.5 if source == "cnn" else 1.0
                                    self.draw_gaussian(heatmap, weightmap, px, py, sigma, weight)
                            weightmap = np.clip(weightmap, 1e-2, 1.0)
                            label_list.append((heatmap, weightmap))
                        else:
                            label_list.append(1.0)

                    elif self.include_negatives:
                        keep = False

                        if self.negative_mode == "random":
                            keep = random.random() < self.balance_ratio
                        elif self.negative_mode == "once_per_image":
                            keep = (len(label_list) == 0)  # keep the first empty tile only
                        elif self.negative_mode == "none":
                            keep = False

                        if keep:
                            tiles.append(tile)
                            if self.use_heatmaps:
                                empty_heat = np.zeros((th, tw, 1), dtype=np.float32)
                                empty_weight = np.zeros((th, tw, 1), dtype=np.float32)
                                label_list.append((empty_heat, empty_weight))
                            else:
                                label_list.append(0.0)
        except Exception as e:
            print(f'[Tiler Error] {e}')
        return tiles, label_list


    def tf_tile_fn(self, image_path):

        def pyfunc(image_path):
            image_path_str = image_path.numpy().decode("utf-8")
            image = tf.io.read_file(image_path_str)
            image = tf.io.decode_png(image, channels=3).numpy()

            labels = self.load_labels(self.label_file, image_path_str)

            # # if there is a spatially-denoised set of CNN detections, load them
            # if os.path.exists(os.path.join(os.path.split(image_path_str)[0], 'results.txt')):
            #     labels += self.load_labels('results.txt', image_path_str)
            tiles, classes = self.tile_image_and_label(image, labels)

            if len(tiles) == 0: # add spacer to filter away later
                if self.use_heatmaps:
                    return (
                        np.zeros((0, self.tile_height, self.tile_width, 3), dtype=np.uint8),
                        np.zeros((0, self.tile_height, self.tile_width, 2), dtype=np.float32)
                    )
                else:
                    return (
                        np.zeros((0, self.tile_height, self.tile_width, 3), dtype=np.uint8),
                        np.zeros((0,), dtype=np.float32)
                    )

            tiles = np.array(tiles, dtype=np.uint8)

            if self.use_heatmaps:
                # classes is a list of (heatmap, weightmap) pairs, each [H, W, 1]
                combined = [np.concatenate([h, w], axis=-1) for h, w in classes]  # each becomes [H, W, 2]
                labels = np.stack(combined, axis=0).astype(np.float32) 
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
            labels.set_shape([None, self.tile_height, self.tile_width, 2])  # <- must be 2 channels
        else:
            labels.set_shape([None])

        dataset = tf.data.Dataset.from_tensor_slices((tiles, labels))

        # Filter out empty tile sets
        dataset = dataset.filter(lambda x, y: tf.shape(x)[0] > 0)

        return dataset


    def add_noise(self, image):
        noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=8.0, dtype=tf.float32)
        image = image + noise
        return tf.clip_by_value(image, 0.0, 255.0)


    def augment(self, image, label):
        image = tf.image.convert_image_dtype(image, tf.float32)

        # --- Random transform selection mask ---
        m = tf.random.uniform([7])
        n = tf.random.uniform([], minval=-1.0, maxval=1.0)

        # --- Color transforms ---
        if m[0] < 0.5:
            image = tf.image.adjust_brightness(image, n * m[0] / 2)
        if m[1] < 0.5:
            image = tf.image.adjust_contrast(image, 1 + n * m[1])
        if m[2] < 0.5:
            image = tf.image.random_hue(image, 0.02)
            image = tf.image.random_saturation(image, 0.95, 1.05)

        # --- Geometric transforms ---
        if m[3] < 0.5:
            image = tf.image.flip_left_right(image)
            if self.use_heatmaps:
                label = tf.image.flip_left_right(label)

        if m[4] < 0.5:
            image = tf.image.flip_up_down(image)
            if self.use_heatmaps:
                label = tf.image.flip_up_down(label)

        # --- Elastic deformation (only apply when heatmaps used) ---
        def apply_elastic(image_np, label_np):
            return elastic_transform(image_np), elastic_transform(label_np)

        if self.use_heatmaps and m[5] < 0.3:
            image, label = tf.numpy_function(
                func=apply_elastic,
                inp=[image, label],
                Tout=[tf.float32, tf.float32]
            )
            image.set_shape([self.tile_height, self.tile_width, 3])
            label.set_shape([self.tile_height, self.tile_width, 2])

        # # --- Rotation (0, 90, 180, 270) ---
        # k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
        # image = tf.image.rot90(image, k)
        # if self.use_heatmaps:
        #     label = tf.image.rot90(label, k)

        --- Additive Gaussian noise ---
        if m[6] < 0.5:
            image = self.add_noise(image)

        # --- Postprocess ---
        image = tf.cast(image, tf.uint8)  # ✅ safe and shape-preserving
        if self.use_heatmaps:
            heatmap = label[..., 0:1]
            weightmap = label[..., 1:2]
            label = tf.concat([heatmap, weightmap], axis=-1)
            tf.ensure_shape(label, [self.tile_height, self.tile_width, 2])  # <- must be 2 channels
            label = tf.clip_by_value(label, 0.0, 1.0)
        else:
            label = tf.expand_dims(label, -1)

        return image, label


    def build_dataset(self, file_list, label_file, batch_size, buffer_size=64, repeat=True, augment=False):
        ds = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(file_list, dtype=tf.string))
        ds = ds.flat_map(self.tf_tile_fn)

        if augment:
            ds = ds.map(self.augment, num_parallel_calls=tf.data.AUTOTUNE)

        if repeat:
            ds = ds.repeat()

        ds = ds.shuffle(buffer_size)
        ds = ds.batch(batch_size)  
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

# ========== Class Weights Helper ========== #

def get_cache_key(file_list, tile_loader, semisupervised):
    """
    Returns a hashable cache key that accounts for:
    - list of image files
    - label file name
    - tile loader config
    - timestamps of any results.txt files (CNN detections)
    """
    result_file_times = []
    cnn_path = None
    for img_path in file_list:
        dir_path = os.path.dirname(img_path)
        if semisupervised:
            cnn_path = os.path.join(dir_path, "results.txt")
        if cnn_path is not None and os.path.exists(cnn_path):
            result_file_times.append(os.path.getmtime(cnn_path))

    summary = {
        "files": sorted(file_list),
        "label_file": tile_loader.label_file,
        "use_heatmaps": tile_loader.use_heatmaps,
        "tile_size": (tile_loader.tile_height, tile_loader.tile_width),
        "buffer": tile_loader.edge_buffer,
        "results_txt_times": sorted(result_file_times)
    }

    import json, hashlib
    summary_str = json.dumps(summary, sort_keys=True)
    return hashlib.md5(summary_str.encode()).hexdigest()


def get_class_weights(file_list, tile_loader, semisupervised=False, cache_dir=None, bypass_cache=False):
    if cache_dir is None or bypass_cache:
        use_cache = False
    else:
        use_cache = True
        os.makedirs(cache_dir, exist_ok=True)

    cache_key = get_cache_key(file_list, tile_loader, semisupervised)
    cache_path = os.path.join(cache_dir, f"{cache_key}.pkl") if use_cache else None

    if use_cache and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                print(f"⚡ Loaded cached class weights from: {cache_path}")
                return pickle.load(f)
        except Exception as e:
            print(f"⚠️ Failed to load cached weights at {cache_path}: {e}")
            print("🧹 Deleting corrupted cache and recomputing...")
            os.remove(cache_path)
    else:
        print('No cached weights found... computing...')

    total_pos = 0
    total_neg = 0

    for i, image_path in enumerate(file_list):
        print(f'  Progress: {i+1}/{len(file_list)} images', end='\r')
        try:
            image_path_str = str(image_path)
            image = tf.io.decode_png(tf.io.read_file(image_path_str), channels=3).numpy()
            labels = tile_loader.load_labels(tile_loader.label_file, image_path_str)
            if semisupervised:
                if os.path.exists(os.path.join(os.path.split(image_path_str)[0], 'results.txt')):
                    labels += tile_loader.load_labels('results.txt', image_path_str)
            tiles, label_list = tile_loader.tile_image_and_label(image, labels)

            for lbl in label_list:
                if tile_loader.use_heatmaps:
                    heatmap, _ = lbl  # unpack tuple
                    if np.any(heatmap > 0):
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

    print(f"\n📏 Final class sample counts: pos={total_pos}, neg={total_neg}\n")
    y_true = [0] * total_neg + [1] * total_pos
    weights = compute_class_weight('balanced', classes=np.unique(y_true), y=y_true)
    result = {int(cl): float(w) for cl, w in zip(np.unique(y_true), weights)}

    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
        print(f"💾 Cached class weights to: {cache_path}")

    return result


def visualize_tiles(dataset, heatmap=False, num_tiles=16):
    count = 0
    for images, labels in dataset.unbatch():
        if count >= num_tiles:
            break

        try:
            img = tf.cast(images, tf.uint8).numpy()

            plt.figure(figsize=(12, 4))

            # Show RGB tile
            plt.subplot(1, 3, 1)
            plt.imshow(img)
            plt.title("Tile")
            plt.axis('off')

            if heatmap:
                heatmap_arr, weightmap_arr = tf.unstack(labels, axis=-1)
                heatmap_arr = heatmap_arr.numpy()
                weightmap_arr = weightmap_arr.numpy()

                plt.subplot(1, 3, 2)
                plt.imshow(heatmap_arr, cmap='hot', vmin=0.0, vmax=1.0)
                plt.title("Label Heatmap")
                plt.axis('off')

                plt.subplot(1, 3, 3)
                plt.imshow(weightmap_arr, cmap='Blues', vmin=0.0, vmax=1.0)
                plt.title("Weight Map")
                plt.axis('off')

            else:
                label_val = labels.numpy() if tf.rank(labels) == 0 else labels.numpy()[0]
                plt.subplot(1, 3, 2)
                plt.text(0.5, 0.5, f"Class: {int(label_val)}", ha='center', va='center', fontsize=16)
                plt.title("Label")
                plt.axis('off')

            plt.tight_layout()
            plt.show()
            count += 1

        except Exception as e:
            print(f"Skipped tile due to error: {e}")
