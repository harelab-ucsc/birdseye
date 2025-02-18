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
# import pdb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE # Function to extract penultimate layer embeddings
from tensorflow.keras import mixed_precision


# mixed_precision.set_global_policy('mixed_float16')

train = True
tSNE = False

# define , filepaths, and model savenames
BUFFER_SIZE = 64
BATCH_SIZE = 4
# IMG_WIDTH = 960
# IMG_HEIGHT = 600
IMG_WIDTH = 720
IMG_HEIGHT = 450
IMAGE_CHANNELS = 3
epochs = 1000

models_dir = os.path.join(os.path.expanduser('~'), 'birdseye', 'models')
filename_prefix = os.path.join(models_dir, f'birdseye_{IMG_WIDTH}_{IMG_HEIGHT}')
val = str(len(glob2.glob(os.path.join(models_dir, filename_prefix+'*')))).rjust(3,'0')
filename = filename_prefix + f'_{val}'

print(f'\n\n{filename}\n\n')

paths = []

label_file = 'labels_checked.txt'
dates = ['2024_07_XX', '2025_01_29', '2025_02_05']
dirlists = [['farm_0__2024_07_25', 'farm_1__2024_07_25', 'farm_2__2024_07_29'],['haybarn_01_rect', 'haybarn_02_rect', 'haybarn_05_rect'], ['haybarn_01_rect']]


def load_trained_model(model, model_path):
    """Loads a trained Keras model from a given file path."""
    model.load_weights(model_path)
    return model


def get_embeddings(model, dataset, embedding_layer_name="global_max_pooling2d"):
    """
    Extracts embeddings and labels from a trained model.
    
    Args:
        model: A trained Keras model.
        dataset: A tf.data.Dataset of (image, label) pairs.
        embedding_layer_name: Name of the layer to extract embeddings from.
    
    Returns:
        Tuple (embeddings, labels) as NumPy arrays.
    """
    # Create an embedding model
    embedding_model = tf.keras.Model(
        inputs=model.input, 
        outputs=model.get_layer(embedding_layer_name).output
    )

    embeddings = []
    labels = []
    cnt = 0
    for images, lbls in dataset:  # Extract images & labels
        cnt += 1
        emb = embedding_model.predict(images, verbose=0)  # Get embeddings
        print(cnt, end='\r')
        # # Flatten the 4D embeddings into 2D (N, height * width * channels)
        # emb = emb.reshape(emb.shape[0], -1)  # Flatten to (N, features)
        
        embeddings.append(emb)
        labels.append(lbls.numpy())

    embeddings = np.vstack(embeddings)  # Stack all embeddings into (N, features)
    labels = np.concatenate(labels)

    print(f"\nExtracted embeddings shape: {embeddings.shape}")  
    return embeddings, labels


def visualize_embeddings(embeddings, labels):
    """Applies t-SNE to embeddings and visualizes them."""
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(label="Class Labels")
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.title("t-SNE Visualization of Learned Embeddings")
    plt.show()


def get_filepaths(paths, date, dirnames, label_file):
    paths += [os.path.join(os.path.expanduser('~'), 'birdseye_CNN_data', date, dirname) for dirname in dirnames]
    return paths 

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~ # data augmentation pipeline to add randomness/volume to dataset # ~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def read_label_file(load_name, image_file):
    try:
        f = open(load_name, "rb")
        while True:
            mask = f.readline()
            tmp = mask.split()
            if os.path.split(tmp[1])[1] == os.path.split(image_file.numpy())[1]:
                cl = float(tmp[2])
                break
            elif len(tmp) == 0:
                break
        f.close()
    except FileNotFoundError:
        # print('bonk')
        pass
    return cl


def load_rle(image_file, label_file=label_file):
    load_name = None
    tmp = tf.keras.backend.get_value(image_file).decode('utf-8')
    tmp = os.path.join(os.path.split(tmp)[0], label_file)
    load_name = glob2.glob(tmp)[0]
    cl = read_label_file(load_name, image_file)
    cl = np.array([cl])
    load_name = None
    return cl


@tf.function
def load_from_rle(image_file):
    # print('image_file: ', image_file)
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


# next 5 functions generate randomness within dataset to help regularize training
# can be thought of as synthesizing extra data from real data or
# "adding water to the shampoo bottle to get every bit out"


@tf.function
def random_rotate(input_image, range=30):
    rot = np.random.uniform(-range, range)
    input_image = tf.py_function(lambda img: ndimage.rotate(img, rot, reshape=False, mode='nearest'),
                                 inp=[input_image], Tout=tf.float32)
    return input_image


def tf_random_rotate(input_image):
    im_shape = input_image.shape
    input_image = random_rotate(input_image)
    input_image.set_shape(im_shape)
    return input_image


def random_jitter(input_image, thresh=0.5):
    input_image = resize(input_image, int(IMG_HEIGHT * 1.2), int(IMG_WIDTH * 1.2))
    # input_image, real_image = random_crop(input_image, real_image)
    m = np.random.rand(5)
    n = np.random.choice((-1,1))
    if m[0] < thresh:
        input_image = tf.image.adjust_brightness(input_image, n*m[0]/2)
    if m[1] < thresh:
        input_image = tf.image.adjust_contrast(input_image, n*m[1]/2)
    # if m[2] < thresh:
    #     input_image = tf_random_rotate(input_image)
    if m[3] < thresh:
        input_image = tf.image.flip_left_right(input_image)
    if m[4] < thresh:
        input_image = tf.image.flip_up_down(input_image)
    return input_image


@tf.function
def load_rle_train(image_file):
    # pdb.set_trace()
    input_image, real_class = load_from_rle(image_file)
    input_image = normalize(input_image)
    input_image = random_jitter(input_image)
    input_image = resize(input_image, IMG_HEIGHT, IMG_WIDTH)
    return input_image, real_class


@tf.function
def load_rle_test(image_file):
    input_image, real_class = load_from_rle(image_file)
    input_image = normalize(input_image)
    input_image = resize(input_image, IMG_HEIGHT, IMG_WIDTH)
    return input_image, real_class


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~ # define model architecture as blocks of layers for ease of experiment # ~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# stuff in play that I read papers/textbooks about:
# separable convolutions (parameter-efficient convolutions, Xception, DeepLab architectures),
# U-Net architecture (mainstay archetype of segmentation networks),
# Batch normalization,
# atrous/dilated convolutions (these seem very data-hungry, DeepLab architecture),
# residual connections (ultra-effective way to improve network fidelity, ResNet architecture)
# parameter regularization (weight decay)
# parameter constraints (recast problem as constrained optimization)

# following *_blocks are general building blocks themed off of Xception network
# entry and main flow blocks (see paper)
def in_block(x, filters, size, dr, use_bias, ker_reg, ker_con, bias_reg, bias_con, act_reg):
    x = tf.keras.layers.Conv2D(filters[0],
                               size,
                               strides=1,
                               dilation_rate=dr[0],
                               padding='same',
                               use_bias=use_bias,
                               kernel_regularizer=ker_reg,
                               kernel_constraint=ker_con,
                               bias_regularizer=bias_reg,
                               bias_constraint=bias_con,
                               activity_regularizer=act_reg)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(filters[1], size,
                               strides=1,
                               dilation_rate=dr[1],
                               padding='same',
                               use_bias=use_bias,
                               kernel_regularizer=ker_reg,
                               kernel_constraint=ker_con,
                               bias_regularizer=bias_reg,
                               bias_constraint=bias_con,
                               activity_regularizer=act_reg)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    return x


def down_block_v2(x, filters, size, use_bias, ker_reg, ker_con, bias_reg, bias_con, act_reg, dropout, pool=True):
    if pool:
        res = tf.keras.layers.Conv2D(filters, 1,
                                    strides=2,
                                    use_bias=use_bias,
                                    kernel_regularizer=ker_reg,
                                    kernel_constraint=ker_con,
                                    bias_regularizer=bias_reg,
                                    bias_constraint=bias_con,
                                    activity_regularizer=act_reg)(x)
    else:
        res = tf.keras.layers.Conv2D(filters, 1,
                                    strides=1,
                                    use_bias=use_bias,
                                    kernel_regularizer=ker_reg,
                                    kernel_constraint=ker_con,
                                    bias_regularizer=bias_reg,
                                    bias_constraint=bias_con,
                                    activity_regularizer=act_reg)(x)

    x = tf.keras.layers.SeparableConv2D(filters, size,
                                        strides=1,
                                        use_bias=use_bias,
                                        padding='same',
                                        depthwise_regularizer=ker_reg,
                                        depthwise_constraint=ker_con,
                                        pointwise_regularizer=ker_reg,
                                        pointwise_constraint=ker_con,
                                        bias_regularizer=bias_reg,
                                        bias_constraint=bias_con,
                                        activity_regularizer=act_reg)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.SeparableConv2D(filters, size,
                                        strides=1,
                                        use_bias=use_bias,
                                        padding='same',
                                        depthwise_regularizer=ker_reg,
                                        depthwise_constraint=ker_con,
                                        pointwise_regularizer=ker_reg,
                                        pointwise_constraint=ker_con,
                                        bias_regularizer=bias_reg,
                                        bias_constraint=bias_con,
                                        activity_regularizer=act_reg)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)

    if pool:
        x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    else:
        pass

    add = tf.keras.layers.add([res, x])
    return add



def out_block(x, filters, size, use_bias, ker_reg, ker_con, bias_reg, bias_con, act_reg):
    x = tf.keras.layers.Dense(filters[0],
                            use_bias=use_bias,
                            kernel_regularizer=ker_reg,
                            kernel_constraint=ker_con,
                            bias_regularizer=bias_reg,
                            bias_constraint=bias_con,
                            activity_regularizer=act_reg)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(int(filters[0]),
                                use_bias=use_bias,
                                kernel_regularizer=ker_reg,
                                kernel_constraint=ker_con,
                                bias_regularizer=bias_reg,
                                bias_constraint=bias_con,
                                activity_regularizer=act_reg)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(int(filters[0]),
                                use_bias=use_bias,
                                kernel_regularizer=ker_reg,
                                kernel_constraint=ker_con,
                                bias_regularizer=bias_reg,
                                bias_constraint=bias_con,
                                activity_regularizer=act_reg)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(int(filters[1]),
                                use_bias=use_bias,
                                kernel_regularizer=ker_reg,
                                kernel_constraint=ker_con,
                                bias_regularizer=bias_reg,
                                bias_constraint=bias_con,
                                activity_regularizer=act_reg)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(1,
                                use_bias=use_bias,
                                kernel_regularizer=ker_reg,
                                kernel_constraint=ker_con,
                                bias_regularizer=bias_reg,
                                bias_constraint=bias_con,
                                activity_regularizer=act_reg)(x)
    x = tf.keras.layers.GlobalMaxPool2D()(x)
    x = tf.keras.activations.sigmoid(x)
    return x


def baseline_net(inputs, use_bias, ker_reg, ker_con, bias_reg, bias_con, act_reg, ker=3):
    """ this model trains reliably """
    x = in_block(inputs, [32, 64], ker, [1, 1], use_bias, ker_reg, ker_con, bias_reg, bias_con, act_reg)  # 32, 64
    x = down_block(x, 64, ker, use_bias, ker_reg, ker_con, bias_reg, bias_con, act_reg)  # 128
    x = down_block(x, 64, ker, use_bias, ker_reg, ker_con, bias_reg, bias_con, act_reg)  # 256
    # x = down_block(x, 512, ker, use_bias, ker_reg, ker_con, bias_reg, bias_con, act_reg)  # 512
    # x = down_block(x, 1024, ker, use_bias, ker_reg, ker_con, bias_reg, bias_con, act_reg)  # 1024
    # x = down_block(x, 1024, ker, use_bias, ker_reg, ker_con, bias_reg, bias_con, act_reg)  # 1024
    out = out_block(x, [64, 32], ker, use_bias, ker_reg, ker_con, bias_reg, bias_con, act_reg)
    return out


def testing_net(inputs, use_bias, ker_reg, ker_con, bias_reg, bias_con, act_reg, dropout, ker=3):
    """ this model trains reliably """
    x = in_block(inputs, [32, 64], ker, [1, 1], use_bias, ker_reg, ker_con, bias_reg, bias_con, act_reg)  # 32, 64
    x = down_block_v2(x, 128, ker, use_bias, ker_reg, ker_con, bias_reg, bias_con, act_reg, dropout)  # 128
    x = down_block_v2(x, 256, ker, use_bias, ker_reg, ker_con, bias_reg, bias_con, act_reg, dropout)  # 256
    x = down_block_v2(x, 512, ker, use_bias, ker_reg, ker_con, bias_reg, bias_con, act_reg, dropout)  # 512
    x = down_block_v2(x, 1024, ker, use_bias, ker_reg, ker_con, bias_reg, bias_con, act_reg, dropout, pool=False)  # 1024
    x = down_block_v2(x, 1024, ker, use_bias, ker_reg, ker_con, bias_reg, bias_con, act_reg, dropout, pool=False)  # 1024
    x = down_block_v2(x, 1024, ker, use_bias, ker_reg, ker_con, bias_reg, bias_con, act_reg, dropout, pool=False)  # 1024
    x = down_block_v2(x, 1024, ker, use_bias, ker_reg, ker_con, bias_reg, bias_con, act_reg, dropout, pool=False)  # 1024
    # x = down_block_v2(x, 1024, ker, use_bias, ker_reg, ker_con, bias_reg, bias_con, act_reg, dropout, pool=False)  # 1024
    # x = down_block_v2(x, 1024, ker, use_bias, ker_reg, ker_con, bias_reg, bias_con, act_reg, dropout, pool=False)  # 1024
    out = out_block(x, [512, 256], ker, use_bias, ker_reg, ker_con, bias_reg, bias_con, act_reg)
    return out


def generator(use_bias, ker_reg, ker_con, bias_reg, bias_con, act_reg, dropout, h=IMG_HEIGHT, w=IMG_WIDTH, c=IMAGE_CHANNELS, ker=3):
    inp = tf.keras.Input(shape=(h, w, c), name='inp_layer')
    out = testing_net(inp, use_bias, ker_reg, ker_con, bias_reg, bias_con, act_reg, dropout, ker=ker)
    return tf.keras.Model(inputs=inp, outputs=out)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    for i, date in enumerate(dates):
        paths = get_filepaths(paths, date, dirlists[i], label_file)

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
    dropout = 0.2

    tflite_conv = False
    use_bias = True
    use_regularizers = True
    use_constraints = True
    logits = False
    if use_regularizers:
        ker_reg = tf.keras.regularizers.L1L2(l1=1e-7, l2=1e-4)  # 1e-6, 1e-3
        act_reg = tf.keras.regularizers.L1L2(l1=1e-12, l2=1e-9)  # 1e-11, 1e-8
    else:
        ker_reg = None
        act_reg = None
    if use_constraints:
        ker_con = tf.keras.constraints.MinMaxNorm(max_value=1.0, rate=1)
    else:
        ker_con = None
    if use_bias:
        if use_regularizers:
            bias_reg = tf.keras.regularizers.L1L2(l1=1e-7, l2=1e-4)  # 1e-6, 1e-3
        else:
            bias_reg = None
        if use_constraints:
            bias_con = tf.keras.constraints.MinMaxNorm(max_value=1.0, rate=1)
        else:
            bias_con = None
    else:
        bias_reg = None
        bias_con = None

    global preglob_tr
    global preglob_va
    global preglob_te

    preglob = []
    for path in paths:
        imgs = glob2.glob(os.path.join(path, '*.png'))
        lbl = os.path.join(path, label_file)
        # print(f'cleaning {lbl}')
        if os.path.exists(lbl):
            # print('  ', lbl)
            pass
        else:
            # print(f'  the given label file does not exist: {lbl}')
            continue
        # cnt = 0
        f = open(lbl, "rb")
        while True:
            line = f.readline()
            line = line.split()

            if len(line) == 0:
                # print('end')
                break
            else:
                for image_file in imgs:
                    if os.path.split(line[1])[1].decode('utf-8') == os.path.split(image_file)[1]:
                        try:
                            float(line[2])
                            preglob.append(image_file)
                        except ValueError as e:
                            # print(f'  {e},   pruning entry')
                            pass

        f.close()

    preglob_tr, preglob_va = train_test_split(preglob, shuffle=False, test_size=0.2)

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


    # strategy = tf.distribute.MirroredStrategy()
    # print("Number of devices: {}".format(strategy.num_replicas_in_sync))
    # with strategy.scope():
    generator = generator(use_bias, ker_reg, ker_con, bias_reg, bias_con, act_reg, dropout, ker=3)
    generator.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-5, beta_1=0.9, beta_2=0.99),
                    loss=tf.keras.losses.BinaryFocalCrossentropy(alpha=0.5, gamma=2.0, from_logits=logits),
                    metrics=[tf.keras.metrics.BinaryCrossentropy(from_logits=logits), \
                            tf.keras.metrics.BinaryAccuracy(threshold=thresh), \
                            tf.keras.metrics.F1Score(threshold=thresh), \
                            tf.keras.metrics.AUC()])
    generator.summary()

    print(f'\n\n {len(preglob)} samples: {len(preglob_tr)} training, {len(preglob_va)} validation\n\n')

    if train:
        print(f'\n\ntraining {filename}\n\n')
        callbacks = [tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0),
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=9),
                tf.keras.callbacks.ModelCheckpoint(filepath=filename+'.weights.h5', save_weights_only=True, save_best_only=True, monitor='val_loss', verbose=2)]
        history = generator.fit(train_ds, validation_data=test_ds, epochs=epochs, callbacks=callbacks, verbose=1)
        print(history.history.keys())
        fig, ax = plt.subplots(2,2)

        ax[0,0].plot(history.history['loss'], c='g')
        ax[0,0].plot(history.history['val_loss'], c='b')

        ax[0,1].plot(history.history['binary_crossentropy'], c='g')
        ax[0,1].plot(history.history['val_binary_crossentropy'], c='b')

        ax[1,0].plot(history.history['auc'], c='g')
        ax[1,0].plot(history.history['val_auc'], c='b')

        ax[1,1].plot(history.history['f1_core'], c='g')
        ax[1,1].plot(history.history['val_f1_score'], c='b')

        plt.show()
    if tSNE:
        # filename = '/home/harey/birdseye/models/birdseye_960_600_006.weights.h5'
        print(f'\n\ncomputing t-SNE for {filename}\n\n')
        model = load_trained_model(generator, filename)
        
        embeddings, labels = get_embeddings(model, test_ds)
        visualize_embeddings(embeddings, labels)
    if tflite_conv:
        print(f'\n\nconverting {filename} to tflite\n\n')
        converter = tf.lite.TFLiteConverter.from_keras_model(generator)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        with open(filename+'.tflite', 'wb') as f:
            f.write(tflite_model)
else:
    print('train.py ran as import')
