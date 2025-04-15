import os
import glob2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from collections import defaultdict

# Default config (can override)
TILE_HEIGHT = 224
TILE_WIDTH = 224
TILE_CHANNELS = 3
LABEL_FILENAME = 'labels.txt'


# ========== Label Reading ========== #

def read_label_file(load_name, image_file):
    labels = []
    try:
        with open(load_name, "r") as f:
            for line in f:
                tmp = line.strip().split()
                if len(tmp) < 5:
                    continue
                frame_name = os.path.split(tmp[1])[1]
                if frame_name == os.path.split(image_file.numpy())[1]:
                    x = float(tmp[2])
                    y = float(tmp[3])
                    cl = int(tmp[4])
                    labels.append((x, y, cl))
    except FileNotFoundError:
        pass
    return labels


def load_rle(image_file, label_file=LABEL_FILENAME):
    tmp = tf.keras.backend.get_value(image_file).decode('utf-8')
    load_name = os.path.join(os.path.split(tmp)[0], label_file)
    load_name = glob2.glob(load_name)[0]
    labels = read_label_file(load_name, image_file)
    return labels


# ========== Optional: Gaussian Heatmap ========== #

def generate_heatmap(tile_labels, tile_x, tile_y, sigma=5):
    heatmap = np.zeros((tile_size, tile_size), dtype=np.float32)
    for x, y, _ in tile_labels:
        if 0 <= x < tile_x and 0 <= y < tile_y:
            xx, yy = int(x), int(y)
            heatmap = add_gaussian(heatmap, xx, yy, sigma)
    return np.expand_dims(heatmap, axis=-1)

def add_gaussian(heatmap, x, y, sigma=5):
    h, w = heatmap.shape
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    gaussian = np.exp(-((X - x)**2 + (Y - y)**2) / (2 * sigma**2))
    return np.maximum(heatmap, gaussian)


# ========== Image + Label Tiling ========== #

def tile_image_and_label(
    image, labels,
    tile_x=TILE_WIDTH, tile_y=TILE_HEIGHT,
    use_heatmaps=False,
    edge_buffer=0
):
    tiles = []
    h, w, _ = image.shape

    for y in range(0, h, tile_y):
        for x in range(0, w, tile_x):
            # Tile must be fully inside bounds
            if y + tile_y > h or x + tile_x > w:
                continue

            # Tile must NOT be inside the edge buffer
            if x < edge_buffer or y < edge_buffer:
                continue
            if x + tile_x > w - edge_buffer or y + tile_y > h - edge_buffer:
                continue

            tile = image[y:y+tile_y, x:x+tile_x]
            tile_labels = [
                (lx - x, ly - y, cl)
                for lx, ly, cl in labels
                if x <= lx < x + tile_x and y <= ly < y + tile_y
            ]

            if use_heatmaps:
                label = generate_heatmap(tile_labels, tile_x, tile_y)
            else:
                label = 1.0 if len(tile_labels) > 0 else 0.0

            tiles.append((tile, label))

    return tiles



# ========== Loader Function ========== #

def tile_loader_single(image_file, use_heatmaps=False, edge_buffer=0):
    image = tf.io.read_file(image_file)
    image = tf.io.decode_png(image, channels=TILE_CHANNELS)

    def tile_and_label(img_path, img_tensor):
        labels = load_rle(img_path)
        img_np = img_tensor.numpy()
        tile_label_pairs = tile_image_and_label(
            img_np, labels,
            tile_x=TILE_WIDTH, tile_y=TILE_HEIGHT,
            use_heatmaps=use_heatmaps,
            edge_buffer=edge_buffer  # buffer in pixels
        )

        tiles, labels = zip(*tile_label_pairs)
        tiles = np.stack(tiles)  # shape [N, H, W, 3] - uint8

        if use_heatmaps:
            labels = np.stack(labels)
        else:
            labels = np.array(labels, dtype=np.float32).reshape(-1)

        return tiles, labels

    tiles, labels = tf.py_function(
        tile_and_label,
        [image_file, image],
        [tf.uint8, tf.float32]
    )

    tiles.set_shape([None, TILE_HEIGHT, TILE_WIDTH, TILE_CHANNELS])
    if use_heatmaps:
        labels.set_shape([None, TILE_HEIGHT, TILE_WIDTH, 1])
    else:
        labels.set_shape([None])

    return tf.data.Dataset.from_tensor_slices((tiles, labels))


# ========== Data Augmentation ========== #

def random_jitter(input_image, thresh=0.5):
    m = tf.random.uniform([5])

    def maybe(fn, cond):
        return tf.cond(cond, lambda: fn(input_image), lambda: input_image)

    def adjust_brightness(img):
        return tf.image.adjust_brightness(img, tf.random.uniform([], -0.1, 0.1))

    def adjust_contrast(img):
        return tf.image.adjust_contrast(img, tf.random.uniform([], 0.9, 1.1))

    def adjust_color(img):
        img = tf.image.random_hue(img, 0.02)
        img = tf.image.random_saturation(img, 0.95, 1.05)
        return img

    def flip_lr(img): return tf.image.flip_left_right(img)
    def flip_ud(img): return tf.image.flip_up_down(img)

    input_image = maybe(adjust_brightness, m[0] < thresh)
    input_image = maybe(adjust_contrast,   m[1] < thresh)
    input_image = maybe(adjust_color,      m[2] < thresh)
    input_image = maybe(flip_lr,           m[3] < thresh)
    input_image = maybe(flip_ud,           m[4] < thresh)

    return input_image


# ========== Build Dataset ========== #

def build_dataset(file_list, batch_size, buffer_size=64, use_heatmaps=False, augment=False, edge_buffer=0, repeat=True):
    ds = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(file_list, dtype=tf.string))

    def apply_loader(img_path):
        return tile_loader_single(img_path, use_heatmaps)

    def apply_aug(image_tile, label):
        if augment:
            image_tile = tf.image.convert_image_dtype(image_tile, tf.float32)  # [0, 1]
            image_tile = random_jitter(image_tile)
            image_tile = tf.image.convert_image_dtype(image_tile, tf.uint8)
        return image_tile, tf.expand_dims(label, axis=-1)

    ds = ds.flat_map(apply_loader)
    ds = ds.map(apply_aug, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    if repeat:
        ds = ds.repeat()
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


