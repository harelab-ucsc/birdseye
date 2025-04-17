
"YOLO-style point detection using MobileNetV2 in TensorFlow"

import os
import tensorflow as tf
from keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image

IMG_WIDTH = 512
IMG_HEIGHT = 384
BATCH_SIZE = 16
EPOCHS = 50
CHANNELS = 3


class GopherDataset(tf.keras.utils.Sequence):
    def __init__(self, image_dir, label_file, batch_size, img_height, img_width, shuffle=True):
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.shuffle = shuffle
        self.labels = []

        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    fname, x, y = parts
                    self.labels.append((fname, float(x), float(y)))

        self.indexes = np.arange(len(self.labels))
        if shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.labels) / self.batch_size))

    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_imgs = []
        batch_labels = []

        for i in batch_indexes:
            fname, x, y = self.labels[i]
            img_path = os.path.join(self.image_dir, fname)
            img = Image.open(img_path).convert('RGB').resize((self.img_width, self.img_height))
            img = np.array(img) / 255.0
            label = [x / self.img_width, y / self.img_height]  # normalize

            batch_imgs.append(img)
            batch_labels.append(label)

        return np.array(batch_imgs), np.array(batch_labels)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

# ===================================================
# YOLO-STYLE HEAD ON MOBILENETV2
# ===================================================
def create_yolo_point_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS)):
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                                   include_top=False,
                                                   weights='imagenet')
    base_model.trainable = False  # or True if fine-tuning

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(2, activation='sigmoid')(x)  # Output: normalized (x, y)

    model = models.Model(inputs=base_model.input, outputs=output)
    return model


if __name__ == '__main__':
    image_dir = os.path.expanduser("~/farm_0_20240725")  
    label_file = os.path.join(image_dir, "labels.txt")  

    dataset = GopherDataset(image_dir, label_file, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH)

    # Train/test split
    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    train_data = tf.data.Dataset.from_generator(
        lambda: (x for x in dataset[:train_size]),
        output_signature=(
            tf.TensorSpec(shape=(None, IMG_HEIGHT, IMG_WIDTH, CHANNELS), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 2), dtype=tf.float32))
    ).unbatch().batch(BATCH_SIZE)

    test_data = tf.data.Dataset.from_generator(
        lambda: (x for x in dataset[train_size:]),
        output_signature=(
            tf.TensorSpec(shape=(None, IMG_HEIGHT, IMG_WIDTH, CHANNELS), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 2), dtype=tf.float32))
    ).unbatch().batch(BATCH_SIZE)

    # Build model
    model = create_yolo_point_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss='mse',
                  metrics=['mae'])

    model.summary()

    # Train model
    model.fit(train_data,
              validation_data=test_data,
              epochs=EPOCHS)

    # Save model
    model.save("gopher_yolo_mobilenetv2.h5")
