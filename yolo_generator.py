"Functions to generate YOLO-style data for training a model using Keras and TensorFlow"
"MobileNetV2 as the backbone"

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.metrics import (
    BinaryCrossentropy, BinaryAccuracy, Precision, Recall, AUC
)

def yolo_model(input_shape=(416, 416, 3), num_classes=1, l2_reg=0.01, dropout_rate=0.5):
    # Base model
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = True  # Set to False if you want to freeze base layers

    x = base_model.output

    # Convolutional head with L2 regularization
    x = layers.Conv2D(
        256, (3, 3), padding='same', activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)

    # Fully connected head
    x = layers.Dense(
        128, activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg)
    )(x)
    x = layers.Dropout(dropout_rate)(x)

    output = layers.Dense(1, activation='sigmoid')(x)


    model = models.Model(inputs=base_model.input, outputs=output)

    return model
