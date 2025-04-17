
"Adjusting original code using CNN for point detection"

import os
import cv2
import numpy as np
import tensorflow as tf
import scipy.ndimage as ndimage
import glob
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

BUFFER_SIZE = 64
BATCH_SIZE = 16
IMG_WIDTH = 512
IMG_HEIGHT = 384
IMAGE_CHANNELS = 3
EPOCHS = 1000

# Paths to images
dirnames = ['farm_0_20240725', 'farm_1_20240725', 'farm_2_20240729', 'farm_3_20240729']
paths = [os.path.join(os.path.expanduser('~'), dirname) for dirname in dirnames]

tf.data.experimental.enable_debug_mode()

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

class reduce_sum(tf.keras.layers.Layer):
    def call(self, x):
        return tf.math.reduce_sum(x, axis=1, keepdims=False)

def read_label_file(load_name, image_file):
    try:
        with open(f"{load_name}", "r") as f:
            for line in f:
                tmp = line.split()
                if len(tmp) >= 4 and os.path.split(tmp[1])[1] == os.path.split(image_file.numpy())[1]:
                    x_coord = float(tmp[2])
                    y_coord = float(tmp[3])
                    return np.array([x_coord, y_coord], dtype=np.float32)
    except FileNotFoundError:
        pass
    return np.array([0.0, 0.0], dtype=np.float32)

def load_rle(image_file):
    tmp = tf.keras.backend.get_value(image_file).decode('utf-8')
    txt_path = os.path.join(os.path.split(tmp)[0], '*.txt')
    label_files = glob.glob(txt_path)
    if label_files:
        cl = read_label_file(label_files[0], image_file)
    else:
        cl = np.array([0.0, 0.0], dtype=np.float32)
    return cl

@tf.function
def load_from_rle(image_file):
    image = tf.io.read_file(image_file)
    image = tf.io.decode_png(image, channels=IMAGE_CHANNELS)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH], method=tf.image.ResizeMethod.BICUBIC)
    cl = tf.py_function(load_rle, [image_file], tf.float32)
    cl.set_shape([2])
    return image, cl

def normalize(input_image):
    return input_image / 255.0

# Improve generalization
def random_jitter(input_image):
    input_image = tf.image.resize(input_image, [int(IMG_HEIGHT * 1.2), int(IMG_WIDTH * 1.2)])
    m = np.random.rand(5)
    n = np.random.choice((-1,1))
    if m[0] < 0.5:
        input_image = tf.image.adjust_brightness(input_image, n*m[0]/2)
    if m[1] < 0.5:
        input_image = tf.image.adjust_contrast(input_image, n*m[1]/2)
    if m[3] < 0.5:
        input_image = tf.image.flip_left_right(input_image)
    if m[4] < 0.5:
        input_image = tf.image.flip_up_down(input_image)
    return input_image

@tf.function
def load_rle_train(image_file):
    input_image, real_class = load_from_rle(image_file)
    input_image = normalize(input_image)
    input_image = random_jitter(input_image)
    input_image = tf.image.resize(input_image, [IMG_HEIGHT, IMG_WIDTH])
    return input_image, real_class

@tf.function
def load_rle_test(image_file):
    input_image, real_class = load_from_rle(image_file)
    input_image = normalize(input_image)
    input_image = tf.image.resize(input_image, [IMG_HEIGHT, IMG_WIDTH])
    return input_image, real_class

def in_block(x, filters, size):
    x = tf.keras.layers.Conv2D(filters[0], size, padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters[1], size, padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    return x

def down_block(x, filters, size):
    res = tf.keras.layers.Conv2D(filters, 1, strides=2)(x)
    x = tf.keras.layers.SeparableConv2D(filters, size, padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.SeparableConv2D(filters, size, padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    return tf.keras.layers.add([res, x])

# Output block that predicts the (x, y) location as normalized coordinates
def out_block(x):
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(2, activation='sigmoid')(x)  # x, y output
    return x

def point_detector_model():
    inp = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMAGE_CHANNELS))
    x = in_block(inp, [32, 64], 3)
    x = down_block(x, 128, 3)
    x = down_block(x, 256, 3)
    x = down_block(x, 512, 3)
    out = out_block(x)
    return tf.keras.Model(inputs=inp, outputs=out)

if __name__ == '__main__':
    filename = f'gopher_point_{IMG_WIDTH}_{IMG_HEIGHT}'

    all_files = []
    for path in paths:
        all_files += glob.glob(os.path.join(path, '*.png'))
    random.shuffle(all_files)
    preglob_tr, preglob_te = train_test_split(all_files, shuffle=False, test_size=0.3)

    train_ds = tf.data.Dataset.from_tensor_slices(preglob_tr)
    train_ds = train_ds.map(load_rle_train, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices(preglob_te)
    test_ds = test_ds.map(load_rle_test).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    model = point_detector_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-6),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanSquaredError(name='mse')]
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_mse', factor=0.5, patience=3, min_lr=0),
        tf.keras.callbacks.EarlyStopping(monitor='val_mse', patience=9),
        tf.keras.callbacks.ModelCheckpoint(filepath=filename+'.weights.h5', save_weights_only=True, save_best_only=True, monitor='val_mse', verbose=2)
    ]

    model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS, callbacks=callbacks, verbose=1)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(filename+'.tflite', 'wb') as f:
        f.write(tflite_model)
else:
    print('train.py ran as import')
