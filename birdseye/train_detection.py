#!/usr/bin/env python
"""
Created on Thursday May 28 2020 at 11:52

@author: mwmasters
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import scipy.ndimage as ndimage
import glob2
import copy
import random
import itertools
import pickle
# import pdb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE # Function to extract penultimate layer embeddings
from tensorflow.keras import mixed_precision
from keras.applications import EfficientNetV2B3, MobileNetV2, MobileNetV3Small, MobileNetV3Large, Xception
# from tile_loader import visualize_tiles

# mixed_precision.set_global_policy('mixed_float16')


# define , filepaths, and model savenames
BUFFER_SIZE = 36
BATCH_SIZE = 9
# IMG_WIDTH = 1920
# IMG_HEIGHT = 1200
IMG_WIDTH = 960
IMG_HEIGHT = 600
# IMG_WIDTH = 720
# IMG_HEIGHT = 450
# IMG_WIDTH = 224
# IMG_HEIGHT = 224
IMAGE_CHANNELS = 3
epochs = 1000

models_dir = os.path.join(os.path.expanduser('~'), 'birdseye', 'models')
filename_prefix = os.path.join(models_dir, f'birdseye_{IMG_WIDTH}_{IMG_HEIGHT}')
val = str(len(glob2.glob(os.path.join(models_dir, filename_prefix+'*.weights.h5')))+1).rjust(3,'0')
filename = filename_prefix + f'_{val}'

# print(f'\n\n{filename}\n\n')

paths = []

label_file = 'labels.txt'

train_dates = [
    '2025_03_25', 
    '2025_04_04',
    '2025_04_09',
]

test_dates = [
    '2025_04_16'
]

train_dirlists = [
    ['haybarn_original_01_01_rect', 'haybarn_eviltwin_01_01_rect'],
    ['original_01_rect', 'original_02_rect', 'eviltwin_01_rect', 'eviltwin_02_rect', 'eviltwin_03_rect'],
    ['original_01_rect', 'original_02_rect', 'eviltwin_01_rect', 'eviltwin_02_rect'],
]

test_dirlists = [
    ['pieranch_rect']
]


def get_filepaths(paths, date, dirnames):
    paths += [os.path.join(os.path.expanduser('~'), 'birdseye_CNN_data', date, dirname) for dirname in dirnames]
    return paths 


def build_file_lists(dates, dirlists):
    paths = []
    for i, date in enumerate(dates):
        paths = get_filepaths(paths, date, dirlists[i])

    preglob = []
    for path in paths:
        imgs = glob2.glob(os.path.join(path, '*.png'))
        lbl = os.path.join(path, label_file)
        if os.path.exists(lbl):
            pass
        else:
            print(f'  the label file does not exist: {lbl}')
            continue
        preglob += imgs
    return paths, preglob


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~ # data augmentation pipeline to add randomness/volume to dataset # ~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def read_label_file(label_file, image_filename):
        """Parse labels from a text file. Each line: idx, filepath, x, y, class"""
        cl = 0.0
        tmp = tf.keras.backend.get_value(image_filename).decode('utf-8')
        load_name = os.path.join(os.path.split(tmp)[0], label_file)
        load_name = glob2.glob(load_name)[0]
        try:
            with open(load_name, "r") as f:
                for line in f:
                    _, path, data = line.strip().split()
                    x_str, y_str, class_str = data.split(',')
                    if os.path.basename(path) == os.path.basename(tmp):
                        cl = float(class_str)
                        break
        except Exception as e:
            print(f"      [Label Load Error] {e}")
        return cl


def load_rle(image_file, label_file=label_file):
    load_name = None
    
    cl = read_label_file(label_file, image_file)
    cl = np.array([cl])
    load_name = None
    return cl


@tf.function
def load_from_rle(image_file):
    image = tf.io.read_file(image_file)
    image = tf.io.decode_png(image, channels=IMAGE_CHANNELS, name='image')
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH], \
                                  method=tf.image.ResizeMethod.BICUBIC)
    cl = tf.py_function(load_rle, [image_file], tf.float32)
    cl.set_shape([1])
    return image, cl


def resize(input_image, height, width):
    """ resize input image from native resolution to designated resolution

        use bicubic resizing to maintain good image fidelity
    """
    input_image = tf.image.resize(input_image, [height, width],
                                  method=tf.image.ResizeMethod.BICUBIC)
    return input_image


def random_crop(input_image):
    """ take a random crop from resized images, to strech the dataset

    crop from resized images (output of resize(.)) to fixed model input size
    """
    cropped_image = tf.image.random_crop(
        input_image, size=[IMG_HEIGHT, IMG_WIDTH, IMAGE_CHANNELS])
    return cropped_image


def normalize(input_image):
    """ normalizing the images to [0, 1] """
    input_image = input_image / 255
    return input_image


def random_jitter(input_image, thresh=0.5):
    m = np.random.rand(5)
    n = np.random.choice((-1,1))
    if m[0] < thresh:
        input_image = tf.image.adjust_brightness(input_image, n*m[0]/2)
    if m[1] < thresh:
        input_image = tf.image.adjust_contrast(input_image, n*m[1]/2)
    if m[2] < thresh:
        input_image = tf.image.random_hue(input_image, 0.02)
        input_image = tf.image.random_saturation(input_image, 0.95, 1.05)
    if m[3] < thresh:
        input_image = tf.image.flip_left_right(input_image)
    if m[4] < thresh:
        input_image = tf.image.flip_up_down(input_image)
    return input_image


@tf.function
def load_rle_train(image_file):
    # pdb.set_trace()
    input_image, real_class = load_from_rle(image_file)
    # input_image = normalize(input_image)
    input_image = random_jitter(input_image)
    input_image = resize(input_image, IMG_HEIGHT, IMG_WIDTH)
    return input_image, real_class


@tf.function
def load_rle_test(image_file):
    input_image, real_class = load_from_rle(image_file)
    # input_image = normalize(input_image)
    input_image = resize(input_image, IMG_HEIGHT, IMG_WIDTH)
    return input_image, real_class


# Define a custom detection head (binary classification: object present or not)
def detection_head(inputs, dim=256, dropout=0.5):
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)  # Convert feature map to vector
    x = tf.keras.layers.Dense(dim, activation="relu")(x)  # Fully connected layer
    x = tf.keras.layers.Dropout(dropout)(x)  # Regularization
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)  # Binary detection (0 or 1)
    return outputs


def pretrained_backbone(inp, unfreeze_frac=0.3, h=IMG_HEIGHT, w=IMG_WIDTH, c=IMAGE_CHANNELS, trainable=False):
    x = tf.keras.ops.cast(inp, "float32")
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    # x = tf.keras.applications.mobilenet_v3.preprocess_input(x)
    # x = tf.keras.applications.efficientnet_v2.preprocess_input(x)
    # x = tf.keras.applications.xception.preprocess_input(x)
    backbone = MobileNetV2(input_shape=(h, w, c), include_top=False, weights="imagenet")
    # backbone = MobileNetV3Small(input_shape=(h, w, c), include_top=False, weights="imagenet")
    # backbone = MobileNetV3Large(input_shape=(h, w, c), include_top=False, weights="imagenet")
    # backbone = Xception(input_shape=(h, w, c), include_top=False, weights="imagenet")
    # backbone = EfficientNetV2B3(input_shape=(h, w, c), include_top=False, weights="imagenet")

    # Freeze the backbone (optional for transfer learning)
    backbone.trainable = False  

    if trainable:
        unfreeze_fraction = unfreeze_frac
        n_total = len(backbone.layers)
        n_unfreeze = int(n_total * unfreeze_fraction)

        # 3. Unfreeze top layers
        for layer in backbone.layers[-n_unfreeze:]:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True
            else:
                # Optionally keep BatchNorm frozen (recommended for stability)
                layer.trainable = False

    # Build the final model
    x = backbone(x, training=False)  # Extract features without updating backbone weights
    outputs = detection_head(x)  # Apply binary detection head

    return outputs


def generator(unfreeze_frac=0.3, h=IMG_HEIGHT, w=IMG_WIDTH, c=IMAGE_CHANNELS, trainable=False):
    inp = tf.keras.Input(shape=(h, w, c), name='inp_layer')
    out = pretrained_backbone(inp, unfreeze_frac=unfreeze_frac, trainable=trainable)
    return tf.keras.Model(inputs=inp, outputs=out)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # tf.data.experimental.enable_debug_mode()

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    thresh = 0.5
    unfreeze_frac = 0.3
    dropout = 0.2
    monitor = 'val_loss'
    # monitor = 'val_bce'

    preview = True
    train = True
    tflite_conv = False
    logits = False
    finetune = False
    if finetune:
        # finetune_source = '/home/harey/birdseye/models/birdseye_720_450_027.weights.h5' # mobilenetv2
        # finetune_source = '/home/harey/birdseye/models/birdseye_960_600_006.weights.h5' # mobilenetv2
        # finetune_source = '/home/harey/birdseye/models/birdseye_960_600_019.weights.h5'  # mobilenetv3large
        finetune_source = '/home/harey/birdseye/models/birdseye_960_600_021.weights.h5' # mobilenetv2
        # finetune_source = '/home/harey/birdseye/models/birdseye_1920_1200_001.weights.h5' # mobilenetv2
    
    global preglob_tr
    global preglob_va
    global preglob_te

    
    print('\n\n    loading training and validation sets...')
    paths_tr, preglob_tr = build_file_lists(train_dates, train_dirlists)
    print(f'      ... training set found: {len(preglob_tr)} images from {paths_tr}')
    paths_va, preglob_va = build_file_lists(test_dates, test_dirlists)
    print(f'      ... validation set found: {len(preglob_va)} images from {paths_va} \n')

    train_ds = tf.data.Dataset.from_tensor_slices(preglob_tr)
    train_ds = train_ds.shuffle(BUFFER_SIZE)  # Shuffle early for better randomness
    train_ds = train_ds.map(load_rle_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds = train_ds.batch(BATCH_SIZE)  # Batch after transformation
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)  # Prefetch to optimize pipeline

    test_ds = tf.data.Dataset.from_tensor_slices(preglob_va)
    test_ds = test_ds.shuffle(BUFFER_SIZE)  # Shuffle early for better randomness
    test_ds = test_ds.map(load_rle_test, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE)  # Batch after transformation
    test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)  # Prefetch to optimize pipeline

    # if preview:
    #     print('\n    sanity check visualization...')
    #     ds = tf.data.Dataset.from_tensor_slices(preglob_va)
    #     ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    #     visualize_tiles(ds, heatmap=False)  

    generator = generator(unfreeze_frac=unfreeze_frac, trainable=finetune)
    generator.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-5, beta_1=0.9, beta_2=0.99),
                    # loss=tf.keras.losses.BinaryFocalCrossentropy(alpha=0.4, gamma=2.0, from_logits=logits),
                    loss=tf.keras.losses.BinaryCrossentropy(from_logits=logits),
                    metrics=[tf.keras.metrics.BinaryCrossentropy(from_logits=logits, name='bce'), \
                            tf.keras.metrics.BinaryAccuracy(threshold=thresh, name='bin_acc'), \
                            tf.keras.metrics.F1Score(threshold=thresh, name='f1'), \
                            tf.keras.metrics.Precision(name='prec'), \
                            tf.keras.metrics.Recall(name='rec'), \
                            # tf.keras.metrics.TruePositives(name='TP'), \
                            # tf.keras.metrics.FalsePositives(name='FP'), \
                            # tf.keras.metrics.TrueNegatives(name='TN'), \
                            # tf.keras.metrics.FalseNegatives(name='FN'), \
                            tf.keras.metrics.AUC()])
    generator.summary()
    
    print(f'\n\n {len(preglob_tr)+len(preglob_va)} samples: {len(preglob_tr)} training, {len(preglob_va)} validation')

    if finetune:
        print(f'\n\nloading model {finetune_source}\n\n')
        generator.load_weights(finetune_source)

    if train:

        # print("\nRunning warm-up steps...\n")
        # for _ in train_ds.take(5):
        #     pass
        # for _ in test_ds.take(5):
        #     pass
        # print("Warm-up complete.\n")

        print(f'\n\ntraining {filename}\n\n')
        # Define TensorBoard callback
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir="logs",  # Directory to save logs
            histogram_freq=1,  # Record weight histograms every epoch
            write_graph=True,  # Save the model graph
            write_images=True  # Log model weights as images
        )
        callbacks = [tensorboard_callback, 
                tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=3, min_lr=0),
                tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=15),
                tf.keras.callbacks.ModelCheckpoint(filepath=filename+'.weights.h5', save_weights_only=True, save_best_only=True, monitor=monitor, verbose=2)]
        history = generator.fit(train_ds, 
            validation_data=test_ds, 
            epochs=epochs, 
            callbacks=callbacks, 
            verbose=1)
        print(history.history.keys())
        with open(filename+'_history.pkl', 'wb') as f:
            pickle.dump(history, f)
        
        fig, ax = plt.subplots(3,2)

        ax[0,0].plot(history.history['loss'], c='g')
        ax[0,0].plot(history.history['val_loss'], c='b')
        ax[0,0].set_title('Loss')

        ax[0,1].plot(history.history['bce'], c='g')
        ax[0,1].plot(history.history['val_bce'], c='b')
        ax[0,1].set_title('BCE')

        ax[1,0].plot(history.history['auc'], c='g')
        ax[1,0].plot(history.history['val_auc'], c='b')
        ax[1,0].set_title('AUROC')

        ax[1,1].plot(history.history['f1'], c='g', label='Train')
        ax[1,1].plot(history.history['val_f1'], c='b', label='Valid')
        ax[1,1].set_title('F1 Score')

        ax[2,0].plot(history.history['prec'], c='g', label='Train')
        ax[2,0].plot(history.history['val_rec'], c='b', label='Valid')
        ax[2,0].set_title('Precision')

        ax[2,1].plot(history.history['rec'], c='g', label='Train')
        ax[2,1].plot(history.history['val_rec'], c='b', label='Valid')
        ax[2,1].set_title('Recall')
        ax[2,1].legend()

        plt.show()

    if tflite_conv:
        print(f'\n\nconverting {filename} to tflite\n\n')
        converter = tf.lite.TFLiteConverter.from_keras_model(generator)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        with open(filename+'.tflite', 'wb') as f:
            f.write(tflite_model)
else:
    print('train.py ran as import')
