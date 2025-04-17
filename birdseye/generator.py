import tensorflow as tf
from keras.applications import EfficientNetV2B3, MobileNetV2, MobileNetV3Small, MobileNetV3Large, Xception


# Define a custom detection head (binary classification: object present or not)
def detection_head(inputs, dim=256):
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)  # Convert feature map to vector
    x = tf.keras.layers.Dense(dim, activation="relu")(x)  # Fully connected layer
    x = tf.keras.layers.Dropout(0.5)(x)  # Regularization
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)  # Binary detection (0 or 1)
    return outputs


def heatmap_head(inputs, h, w):
    x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(1, 1, padding='same', activation='sigmoid')(x)  # Output: heatmap
    x = tf.image.resize(x, [h, w], method='bicubic')  # ensure full resolution
    return x


def pretrained_backbone(inp, unfreeze_frac=0.3, h=None, w=None, c=None, trainable=False):
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


def generator(h, w, c, unfreeze_frac=0.3, trainable=False):
    inp = tf.keras.Input(shape=(h, w, c), name='inp_layer')
    out = pretrained_backbone(inp, h=h, w=w, c=c, unfreeze_frac=unfreeze_frac, trainable=trainable)
    return tf.keras.Model(inputs=inp, outputs=out)