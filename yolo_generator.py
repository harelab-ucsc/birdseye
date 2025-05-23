"Functions to generate YOLO-style data for training a model using Keras and TensorFlow"
"MobileNetV2 as the backbone"

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def yolo_model(input_shape=(416, 416, 3), num_classes=1, l2_reg=0.01, dropout_rate=0.5, anchors=5):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = True

    x = base_model.output  # e.g., (None, 13, 13, 1280) or (None, 7, 7, ...)
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    num_outputs = 5 + num_classes  # (x, y, w, h, obj) + class
    x = layers.Conv2D(filters=anchors * num_outputs, kernel_size=1, padding='same', activation='sigmoid')(x)

    # Dynamically compute spatial dims
    h, w = x.shape[1], x.shape[2]
    x = layers.Reshape((h, w, anchors, num_outputs))(x)

    return models.Model(inputs=base_model.input, outputs=x)


# def yolo_model(input_shape=(416, 416, 3), num_classes=1, l2_reg=0.01, dropout_rate=0.5):
#     # Base model
#     base_model = tf.keras.applications.MobileNetV2(
#         input_shape=input_shape,
#         include_top=False,
#         weights='imagenet'
#     )
#     base_model.trainable = True  # Set to False if you want to freeze base layers

#     x = base_model.output

#     # Convolutional head with L2 regularization
#     x = layers.Conv2D(
#         256, (3, 3), padding='same', activation='relu',
#         kernel_regularizer=regularizers.l2(l2_reg)
#     )(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.GlobalAveragePooling2D()(x)
#     x = layers.Dropout(dropout_rate)(x)

#     # Fully connected head
#     x = layers.Dense(
#         128, activation='relu',
#         kernel_regularizer=regularizers.l2(l2_reg)
#     )(x)
#     x = layers.Dropout(dropout_rate)(x)

#     output = layers.Dense(1, activation='sigmoid')(x)


#     model = models.Model(inputs=base_model.input, outputs=output)

#     return model


