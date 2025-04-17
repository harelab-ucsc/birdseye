"Functions to generate YOLO-style data for training a model using Keras and TensorFlow"
"MobileNetV2 as the backbone"

import tensorflow as tf
from keras import layers, models

def yolo_model(input_shape=(416, 416, 3), num_classes=1):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )

    base_model.trainable = True  # You can freeze layers if needed

    x = base_model.output
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    # YOLO-style output head (simplified):
    # [x_center, y_center, width, height, objectness, class1, class2, ...]
    num_outputs = 5 + num_classes  # 4 bbox coords + objectness + class probs

    x = layers.Dense(128, activation='relu')(x)
    output = layers.Dense(num_outputs, activation='sigmoid')(x)

    model = models.Model(inputs=base_model.input, outputs=output)
    return model
