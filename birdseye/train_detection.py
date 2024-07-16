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
from mask import Mask
# from AMI_ContourClassFamily import Contour
import glob
import pdb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~ # data augmentation pipeline to add randomness/volume to dataset # ~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# define , filepaths, and model savenames
BUFFER_SIZE = 32
BATCH_SIZE = 8
IMG_WIDTH = 512
IMG_HEIGHT = 384
epochs = 1000

# PATH = os.getcwd()
PATH = os.path.join(os.path.expanduser('~'), 'parsed_flight')
# PATH = 'C:\\Users\\mwmasters\\Documents\\APL-subVision-Shapiro\\preUCSC\\waterline_20210609_release\\data\\data\\GOPRO_FremontCut_1'
IMAGE_CHANNELS = 3

# tf.data.experimental.enable_debug_mode()


class reduce_sum(tf.keras.layers.Layer):
    def call(self, x):
        return tf.math.reduce_sum(x, axis=1, keepdims=False, name=None)


def read_label_file(load_name, frame_index):
    try:
        f = open(f"{load_name}", "r")
        # print(f'    Mask loaded: frame index {frame_index}, {load_name}')
        i = 0
        while True:
            # print(i)
            mask = f.readline()
            tmp = mask.split()
            if len(tmp) == 0:
                break
            # print('tmp: ', tmp)
            if int(tmp[0]) == frame_index:
                cl = tmp[2]
        f.close()
    except FileNotFoundError:
        pass
    return cl


def load_rle(image_file):
    load_name = None
    tmp = tf.keras.backend.get_value(image_file).decode('utf-8')
    try:
        frame_index = preglob_tr.index(image_file)
    except ValueError:
        frame_index = preglob_te.index(image_file)
    tmp = os.path.join(os.path.split(tmp)[0], '*.txt')
    load_name = glob.glob(tmp)[0]
    # right here, read labels.txt for the frame_index and return the label value
    cl = read_label_file(load_name, frame_index)
    cl = float(cl)
    load_name = None
    return cl


@tf.function
def load_from_rle(image_file):
    image = tf.io.read_file(image_file)
    image = tf.io.decode_png(image, channels=IMAGE_CHANNELS, name='image')
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH], \
                                  method=tf.image.ResizeMethod.BICUBIC)
    cl = tf.py_function(load_rle, [image_file], [tf.float32])
    return image, cl


def resize(input_image, height, width):
    """ resize input image from native resolution to designated resolution

        use bicubic resizing to maintain good image fidelity
    """
    input_image = tf.image.resize(input_image, [height, width],
                                  method=tf.image.ResizeMethod.BICUBIC)
    return input_image


def random_crop(input_image, real_image):
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
    input_image, real_class = load_from_rle(image_file)
    # real_class.set_shape([NUM_CLASSES, MASK_CHANNELS])
    input_image = normalize(input_image)

    input_image = random_jitter(input_image)
    input_image = resize(input_image, IMG_HEIGHT, IMG_WIDTH)

    # pdb.set_trace()
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
    # print('filters ', filters, 'size ', size)
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


def down_block(x, filters, size, use_bias, ker_reg, ker_con, bias_reg, bias_con, act_reg):
    res = tf.keras.layers.Conv2D(filters, 1,
                                 strides=2,
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

    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    add = tf.keras.layers.add([res, x])
    return add


def out_block(x, filters, size, use_bias, ker_reg, ker_con, bias_reg, bias_con, act_reg):
    x = reduce_sum()(x)
    x = tf.keras.layers.Dense(filters[0],
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
    # x = tf.keras.layers.Softmax()(x)
    return x


def baseline_UNet(inputs, use_bias, ker_reg, ker_con, bias_reg, bias_con, act_reg, ker=3):
    """ this model trains reliably """
    d1 = in_block(inputs, [32, 64], ker, [1, 1], use_bias, ker_reg, ker_con, bias_reg, bias_con, act_reg)  # 32, 64
    d2 = down_block(d1, 128, ker, use_bias, ker_reg, ker_con, bias_reg, bias_con, act_reg)  # 128
    d3 = down_block(d2, 256, ker, use_bias, ker_reg, ker_con, bias_reg, bias_con, act_reg)  # 256
    # d4 = down_block(d3, 512, ker, use_bias, ker_reg, ker_con, bias_reg, bias_con, act_reg)  # 512
    # d5 = down_block(d4, 1024, ker, use_bias, ker_reg, ker_con, bias_reg, bias_con, act_reg)  # 1024
    # d6 = down_block(d5, 1024, ker, use_bias, ker_reg, ker_con, bias_reg, bias_con, act_reg)  # 1024
    out = out_block(d2, [64, 32], ker, use_bias, ker_reg, ker_con, bias_reg, bias_con, act_reg)
    return out


def generator(use_bias, ker_reg, ker_con, bias_reg, bias_con, act_reg, h=IMG_HEIGHT, w=IMG_WIDTH, c=IMAGE_CHANNELS, ker=3):
    inp = tf.keras.Input(shape=(h, w, c), name='inp_layer')
    out = baseline_UNet(inp, use_bias, ker_reg, ker_con, bias_reg, bias_con, act_reg, ker=ker)
    return tf.keras.Model(inputs=inp, outputs=out)


if __name__ == '__main__':
    filename = f'ucsc_subvision_{IMG_WIDTH}_{IMG_HEIGHT}_01'.format(IMG_WIDTH, IMG_HEIGHT)

    use_bias = True
    use_regularizers = True
    use_constraints = False
    logits = False
    if use_regularizers:
        ker_reg = tf.keras.regularizers.L1L2(l1=1e-6, l2=1e-4)  # 1e-6, 1e-3
        act_reg = tf.keras.regularizers.L1L2(l1=1e-11, l2=1e-8)  # 1e-11, 1e-8
    else:
        ker_reg = None
        act_reg = None
    if use_constraints:
        ker_con = tf.keras.constraints.MinMaxNorm(max_value=1.0, rate=0.5)
    else:
        ker_con = None
    if use_bias:
        if use_regularizers:
            bias_reg = tf.keras.regularizers.L1L2(l1=1e-6, l2=1e-4)  # 1e-6, 1e-3
        else:
            bias_reg = None
        if use_constraints:
            bias_con = tf.keras.constraints.MinMaxNorm(max_value=1.0, rate=0.5)
        else:
            bias_con = None
    else:
        bias_reg = None
        bias_con = None

    global preglob_tr
    global preglob_te

    preglob = glob.glob(os.path.join(PATH, '*.png'))

    # print(type(preglob))
    preglob_tr, preglob_te = train_test_split(preglob, shuffle=True, test_size=0.4)

    train_ds = tf.data.Dataset.from_tensor_slices(preglob_tr)
    train_ds = train_ds.map(load_rle_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds = train_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    # print('train_ds.element_spec', train_ds.element_spec)
    # print(type(train_ds))
    test_ds = tf.data.Dataset.from_tensor_slices(preglob_te)
    test_ds = test_ds.map(load_rle_test)
    test_ds = test_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    # print('test_ds.element_spec', test_ds.element_spec)
    # print(type(test_ds))

    # lossFunc = {'output1': tf.keras.losses.BinaryCrossentropy(from_logits=logits),
    #             'output2': tf.keras.losses.BinaryCrossentropy(from_logits=logits)}
    # lossWeights = {'output1': 0.5, 'output2': 0.5}

    generator = generator(use_bias, ker_reg, ker_con, bias_reg, bias_con, act_reg, ker=3)
    generator.summary()
    generator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5,
                            beta_1=0.9, beta_2=0.99, clipnorm=1.0),
                      # loss=lossFunc,
                      # loss_weights = lossWeights,
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=logits),
                      metrics=[tf.keras.metrics.BinaryCrossentropy(from_logits=logits)])
    callbacks = [tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0),
                 tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15),
                 tf.keras.callbacks.ModelCheckpoint(filepath=filename+'.weights.h5', save_weights_only=True, save_best_only=True, monitor='val_loss', verbose=2)]
    # if os.path.isfile(filename+'.weights.h5'):
    #     generator.load_weights(filename+'.weights.h5')
    # else:
    #     pass
    generator.fit(train_ds, validation_data=test_ds, epochs=epochs, callbacks=callbacks, verbose=1)

    converter = tf.lite.TFLiteConverter.from_keras_model(generator)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(filename+'.tflite', 'wb') as f:
        f.write(tflite_model)
else:
    print('train.py ran as import')
