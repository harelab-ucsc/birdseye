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
from keras.applications import EfficientNetV2B3, MobileNetV2, MobileNetV3Small, MobileNetV3Large, Xception

from tile_loader import TileLoader, visualize_tiles, get_tile_level_class_weights


# define , filepaths, and model savenames
BUFFER_SIZE = 320
BATCH_SIZE = 64
IMG_WIDTH = 224
IMG_HEIGHT = 224
IMAGE_CHANNELS = 3
epochs = 1000

models_dir = os.path.join(os.path.expanduser('~'), 'birdseye', 'models')
filename_prefix = os.path.join(models_dir, f'birdseye_{IMG_WIDTH}_{IMG_HEIGHT}')
val = str(len(glob2.glob(os.path.join(models_dir, filename_prefix+'*.weights.h5')))+1).rjust(3,'0')
filename = filename_prefix + f'_{val}'

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


# Define a custom detection head (binary classification: object present or not)
def detection_head(inputs, dim=256):
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)  # Convert feature map to vector
    x = tf.keras.layers.Dense(dim, activation="relu")(x)  # Fully connected layer
    x = tf.keras.layers.Dropout(0.5)(x)  # Regularization
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)  # Binary detection (0 or 1)
    return outputs


def heatmap_head(inputs):
    x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(1, 1, padding='same', activation='sigmoid')(x)  # Output: heatmap
    x = tf.image.resize(x, [IMG_HEIGHT, IMG_WIDTH], method='bicubic')  # ensure full resolution
    return x


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
    monitor = 'val_loss'
    # monitor = 'val_bce'
    
    preview = True
    train = True
    tflite_conv = False
    logits = False
    finetune = False

    if finetune:
        finetune_source = '/home/harey/birdseye/models/birdseye_tiles_224_224_001.weights.h5' # mobilenetv2
    
    print('\n\n    loading training and validation sets...')
    paths_tr, preglob_tr = build_file_lists(train_dates, train_dirlists)
    print(f'      ... training set found: {len(preglob_tr)} images from {paths_tr}')
    paths_va, preglob_va = build_file_lists(test_dates, test_dirlists)
    print(f'      ... validation set found: {len(preglob_va)} images from {paths_va} \n')
    
    tile_loader = TileLoader(tile_size=(224, 224), edge_buffer=81, use_heatmaps=False)

    train_ds = tile_loader.build_dataset(
        file_list=preglob_tr,
        label_file=label_file,
        batch_size=BATCH_SIZE,
        buffer_size=BUFFER_SIZE,
        repeat=True,
        augment=True
    )

    test_ds = tile_loader.build_dataset(
        file_list=preglob_va,
        label_file=label_file,
        batch_size=BATCH_SIZE,
        buffer_size=BUFFER_SIZE,
        repeat=True,
        augment=False
    )

    if preview:
        print('\n    sanity check visualization...')
        sample_ds = tile_loader.build_dataset(
            file_list=preglob_va,
            label_file=label_file,
            batch_size=BATCH_SIZE,
            buffer_size=BUFFER_SIZE,
            repeat=False,
            augment=True
        )
        visualize_tiles(sample_ds, heatmap=False)  # use_heatmaps=True if you're using spatial labels

    print('\n    computing training class_weights...')
    class_weights, num_images, tiles_per_image = get_tile_level_class_weights(paths_tr, label_file, edge_buffer=81)
    for cl, w in class_weights.items():
        print(f"      Class {cl}: {w:.4f}")
    print('\n\n    generating validation set stats')
    _ = get_tile_level_class_weights(paths_tr, label_file, edge_buffer=81)
    print('\n\n')

    generator = generator(unfreeze_frac=unfreeze_frac, trainable=finetune)
    generator.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-5, beta_1=0.9, beta_2=0.99),
                    loss=tf.keras.losses.BinaryFocalCrossentropy(alpha=0.35, gamma=4.0, from_logits=logits),
                    # loss=tf.keras.losses.BinaryCrossentropy(from_logits=logits),
                    metrics=[tf.keras.metrics.BinaryCrossentropy(from_logits=logits, name='bce'), \
                            tf.keras.metrics.BinaryAccuracy(threshold=thresh, name='bin_acc'), \
                            tf.keras.metrics.F1Score(threshold=thresh, name='f1'), \
                            tf.keras.metrics.Precision(name='prec'), \
                            tf.keras.metrics.Recall(name='rec'), \
                            tf.keras.metrics.AUC()])
    generator.summary()
    
    if finetune:
        print(f'\n\nloading model {finetune_source}\n\n')
        generator.load_weights(finetune_source)

    if train:
        print(f'\n\ntraining {filename}\n\n')

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir="logs",  # Directory to save logs
            histogram_freq=1,  # Record weight histograms every epoch
            write_graph=True,  # Save the model graph
            write_images=True  # Log model weights as images
        )

        callbacks = [tensorboard_callback, 
            tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=3, min_lr=0),
            tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=15),
            tf.keras.callbacks.ModelCheckpoint(filepath=filename+'.weights.h5', save_weights_only=True, save_best_only=True, monitor=monitor, verbose=2)
        ]

        history = generator.fit(
            train_ds,
            validation_data=test_ds,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1,
            steps_per_epoch=(num_images * tiles_per_image) // BATCH_SIZE
        )
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
