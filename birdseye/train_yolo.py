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
from tensorflow.keras.metrics import Metric
from tensorflow.keras import backend as K

mixed_precision.set_global_policy('mixed_float16')

train = True
tSNE = False

# define , filepaths, and model savenames
BUFFER_SIZE = 32
BATCH_SIZE = 2
# IMG_WIDTH = 960
# IMG_HEIGHT = 600
IMG_WIDTH = 720
IMG_HEIGHT = 450
IMAGE_CHANNELS = 3
epochs = 1000
PCK_THRESH = 50

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

# Define YOLO parameters
S = 7  # Grid size
B = 2  # Number of points per cell
C = 3  # Number of classes

# Function to load image
def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))  # Resize for YOLO
    image = image / 255.0  # Normalize
    return image


def preprocess_label(image_path):
    label = tf.zeros((S, S, B * (3 + C)))  # (x, y, confidence, class_one_hot...)

    # Retrieve annotations (list of objects with 'x', 'y', and 'class')
    # anns = annotations.get(image_path.numpy().decode(), [])
    anns = read_label_file()

    for obj in anns:
        x, y = obj["loc"]  # Extract point coordinates
        class_idx = obj["class"]    # Class index

        grid_x, grid_y = int(S * x), int(S * y)  # Find grid cell location
        
        # Normalize x, y relative to grid cell
        x_offset, y_offset = S * x - grid_x, S * y - grid_y  

        # Assign values to the label tensor
        label = tf.tensor_scatter_nd_update(label,
                                            [[grid_y, grid_x, 0]],
                                            [x_offset])  # x_offset
        label = tf.tensor_scatter_nd_update(label,
                                            [[grid_y, grid_x, 1]],
                                            [y_offset])  # y_offset
        label = tf.tensor_scatter_nd_update(label,
                                            [[grid_y, grid_x, 2]],
                                            [1.0])  # Confidence (1 = point exists)
        label = tf.tensor_scatter_nd_update(label,
                                            [[grid_y, grid_x, 3 + class_idx]],
                                            [1.0])  # One-hot class encoding

    return label

# Function to load image and label together
def load_data(image_path):
    image = load_image(image_path)
    label = tf.py_function(preprocess_label, [image_path], tf.float32)
    label.set_shape((S, S, B * (5 + C)))
    return image, label


def read_label_file(load_name, image_file):
    try:
        f = open(load_name, "rb")
        while True:
            mask = f.readline()
            tmp = mask.split()
            if os.path.split(tmp[1])[1] == os.path.split(image_file.numpy())[1]:
                cl = float(tmp[2])
                ## more here
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
    input_image = input_image / 255.0
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

# Define a custom detection head (binary classification: object present or not)
def detection_head(inputs):
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)  # Convert feature map to vector
    x = tf.keras.layers.Dense(256, activation="relu")(x)  # Fully connected layer
    x = tf.keras.layers.Dropout(0.5)(x)  # Regularization
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)  # Binary detection (0 or 1)
    return outputs


def pretrained_backbone(inp, use_bias, ker_reg, ker_con, bias_reg, bias_con, act_reg, dropout, h=IMG_HEIGHT, w=IMG_WIDTH, c=IMAGE_CHANNELS, ker=3, trainable=False):
    x = tf.keras.ops.cast(inp, "float32")
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    backbone = MobileNetV2(input_shape=(h, w, c), include_top=False, weights="imagenet")
    # backbone = MobileNetV3Small(input_shape=(h, w, c), include_top=False, weights="imagenet")
    # backbone = MobileNetV3Large(input_shape=(h, w, c), include_top=False, weights="imagenet")
    # backbone = Xception(input_shape=(h, w, c), include_top=False, weights="imagenet")

    # Freeze the backbone (optional for transfer learning)
    backbone.trainable = False  

    if trainable:
        unfreeze_fraction = 0.3
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


def generator(use_bias, ker_reg, ker_con, bias_reg, bias_con, act_reg, dropout, h=IMG_HEIGHT, w=IMG_WIDTH, c=IMAGE_CHANNELS, ker=3, trainable=False):
    inp = tf.keras.Input(shape=(h, w, c), name='inp_layer')
    # out = testing_net(inp, use_bias, ker_reg, ker_con, bias_reg, bias_con, act_reg, dropout, ker=ker)
    out = pretrained_backbone(inp, use_bias, ker_reg, ker_con, bias_reg, bias_con, act_reg, dropout, ker=ker, trainable=trainable)
    return tf.keras.Model(inputs=inp, outputs=out)


def point_yolo_loss(y_true, y_pred, S=7, B=2, C=3, lambda_coord=5, lambda_noobj=0.5):
    """
    Modified YOLO loss for point-based confidence rather than bounding box IoU.

    Args:
        y_true: (batch, S, S, B*(3+C)) - Ground truth tensor (x, y, conf, classes...).
        y_pred: (batch, S, S, B*(3+C)) - Predicted tensor.
        S: Grid size.
        B: Number of points per grid cell.
        C: Number of classes.
        lambda_coord: Weighting factor for coordinate loss.
        lambda_noobj: Weighting factor for confidence loss of no-object cells.

    Returns:
        Total loss (sum of localization, confidence, and classification loss).
    """

    # Reshape tensors
    y_pred = tf.reshape(y_pred, (-1, S, S, B, 3 + C))  # (x, y, conf, classes...)
    y_true = tf.reshape(y_true, (-1, S, S, B, 3 + C))  

    # Extract predictions and ground truths
    pred_xy = y_pred[..., 0:2]    # (x, y) predicted points
    pred_conf = y_pred[..., 2]    # Predicted confidence
    pred_class = y_pred[..., 3:]  # Predicted class probabilities

    true_xy = y_true[..., 0:2]    # Ground truth (x, y)
    true_conf = y_true[..., 2]    # Ground truth confidence
    true_class = y_true[..., 3:]  # Ground truth class (one-hot)

    # **1. Localization Loss (Point Regression)**
    loc_loss = lambda_coord * tf.reduce_sum(true_conf * tf.square(true_xy - pred_xy))

    # **2. Confidence Loss (Using Inverse Distance)**
    # Compute Euclidean distance between predicted and ground truth points
    distance = tf.norm(true_xy - pred_xy, axis=-1)  # Euclidean distance
    
    # Define confidence as an inverse function of distance (lower distance → higher confidence)
    pred_conf_new = tf.exp(-distance)  # Confidence = exp(-gamma * distance) ensures smooth decay
    
    # Compute confidence loss
    obj_loss = tf.reduce_sum(true_conf * tf.square(pred_conf_new - pred_conf))  # Object exists
    noobj_loss = lambda_noobj * tf.reduce_sum((1 - true_conf) * tf.square(0 - pred_conf))  # No object

    # **3. Classification Loss**
    class_loss = tf.reduce_sum(true_conf * tf.square(true_class - pred_class))  # Only for object points

    # **Total Loss**
    total_loss = loc_loss + obj_loss + noobj_loss + class_loss

    return total_loss


class MeanEuclideanDistance(Metric):
    """Mean Euclidean Distance (MED) between predicted and true keypoints."""
    def __init__(self, name="mean_euclidean_distance", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total_distance = self.add_weight(name="total_distance", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        true_xy = y_true[..., :2]
        pred_xy = y_pred[..., :2]
        distance = tf.norm(true_xy - pred_xy, axis=-1)  # Euclidean distance
        self.total_distance.assign_add(tf.reduce_sum(distance))
        self.count.assign_add(tf.cast(tf.size(distance), tf.float32))

    def result(self):
        return self.total_distance / (self.count + K.epsilon())  # Avoid division by zero

class PCK(Metric):
    """Percentage of Correct Keypoints (PCK) within a threshold."""
    def __init__(self, threshold=PCK_THRESH, name="pck", **kwargs):  
        super().__init__(name=name, **kwargs)
        self.threshold = threshold # Threshold distance in pixels
        self.correct_count = self.add_weight(name="correct_count", initializer="zeros")
        self.total_count = self.add_weight(name="total_count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        true_xy = y_true[..., :2]
        pred_xy = y_pred[..., :2]
        distance = tf.norm(true_xy - pred_xy, axis=-1)  # Euclidean distance
        correct = tf.cast(distance < self.threshold, tf.float32)
        self.correct_count.assign_add(tf.reduce_sum(correct))
        self.total_count.assign_add(tf.cast(tf.size(correct), tf.float32))

    def result(self):
        return self.correct_count / (self.total_count + K.epsilon())  # Avoid division by zero

class ConfidenceAveragePrecision(Metric):
    """Average Precision (AP) for confidence scores (Precision-Recall Curve)."""
    def __init__(self, name="confidence_ap", **kwargs):
        super().__init__(name=name, **kwargs)
        self.auc = tf.keras.metrics.AUC(curve="PR")

    def update_state(self, y_true, y_pred, sample_weight=None):
        true_conf = y_true[..., 2]  # Assuming confidence is at index 2
        pred_conf = y_pred[..., 2]
        self.auc.update_state(true_conf, pred_conf)

    def result(self):
        return self.auc.result()

class ClassAccuracy(Metric):
    """Categorical Accuracy for multi-class classification at detected keypoints."""
    def __init__(self, name="class_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.accuracy = tf.keras.metrics.CategoricalAccuracy()

    def update_state(self, y_true, y_pred, sample_weight=None):
        true_class = y_true[..., 3:]  # Assuming class labels start at index 3
        pred_class = y_pred[..., 3:]
        self.accuracy.update_state(true_class, pred_class)

    def result(self):
        return self.accuracy.result()



if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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

    # Create dataset
    preglob_tr, preglob_va = train_test_split(preglob, shuffle=False, test_size=0.2)  

    train_ds = tf.data.Dataset.from_tensor_slices(preglob_tr)
    train_ds = train_ds.map(load_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds = train_ds.batch(BATCH_SIZE).shuffle(BUFFER_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices(preglob_va)
    test_ds = test_ds.map(load_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE).shuffle(BUFFER_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

    # strategy = tf.distribute.MirroredStrategy()
    # print("Number of devices: {}".format(strategy.num_replicas_in_sync))
    # with strategy.scope():
    generator = generator(use_bias, ker_reg, ker_con, bias_reg, bias_con, act_reg, dropout, ker=3)
    generator.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-5, beta_1=0.9, beta_2=0.99),
                    loss=point_yolo_loss(),
                    metrics=[
                        MeanEuclideanDistance(),
                        PCK(),  
                        ConfidenceAveragePrecision(),
                        ClassAccuracy()])
    generator.summary()

    print(f'\n\n {len(preglob)} samples: {len(preglob_tr)} training, {len(preglob_va)} validation\n\n')

    if train:
        print(f'\n\ntraining {filename}\n\n')
        callbacks = [
                tf.keras.callbacks.TensorBoard(log_dir="logs", histogram_freq=1),
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0),
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=9),
                tf.keras.callbacks.ModelCheckpoint(filepath=filename+'.weights.h5', save_weights_only=True, save_best_only=True, monitor='val_loss', verbose=2)]
        history = generator.fit(train_ds, validation_data=test_ds, epochs=epochs, callbacks=callbacks, verbose=1)
        print(history.history.keys())
        
        fig, ax = plt.subplots(2,2)

        ax[0,0].plot(history.history['loss'], c='g')
        ax[0,0].plot(history.history['val_loss'], c='b')
        ax[0,0].set_title('Loss')

        ax[0,1].plot(history.history['binary_crossentropy'], c='g')
        ax[0,1].plot(history.history['val_binary_crossentropy'], c='b')
        ax[0,1].set_title('BCE')

        ax[1,0].plot(history.history['auc'], c='g')
        ax[1,0].plot(history.history['val_auc'], c='b')
        ax[1,0].set_title('AUROC')

        ax[1,1].plot(history.history['f1_score'], c='g', label='Train')
        ax[1,1].plot(history.history['val_f1_score'], c='b', label='Valid')
        ax[1,1].set_title('F1 Score')
        ax[1,1].legend()

        plt.show()
