import tensorflow as tf
from keras.applications import EfficientNetV2B3, MobileNetV2, MobileNetV3Small, MobileNetV3Large, Xception


# Define a custom detection head (binary classification: object present or not)
def detection_head(inputs, dim=256):
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)  # Convert feature map to vector
    x = tf.keras.layers.Dense(dim, activation="relu")(x)  # Fully connected layer
    x = tf.keras.layers.Dropout(0.5)(x)  # Regularization
    x = tf.keras.layers.Dense(int(dim/2), activation="relu")(x)  # Fully connected layer
    x = tf.keras.layers.Dropout(0.5)(x)  # Regularization
    x = tf.keras.layers.Dense(int(dim/4), activation="relu")(x)  # Fully connected layer
    x = tf.keras.layers.Dropout(0.5)(x)  # Regularization
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)  # Binary detection (0 or 1)
    return outputs


def heatmap_head_deep(inputs):
    # assumes efficientNet backbone and 224x224 tiles
    x = tf.keras.layers.Conv2DTranspose(512, 3, strides=2, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.Conv2DTranspose(256, 3, strides=2,  padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(128, 3, strides=2,  padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2,  padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(32, 3, strides=2,  padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(1, 1, padding='same', activation='sigmoid')(x)  # Output: heatmap
    return x

def heatmap_head_hybrid(inputs, h, w):
    x = tf.keras.layers.Conv2DTranspose(128, 3, strides=2,  padding='same', activation='relu')(inputs)
    x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2,  padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(1, 1, strides=2,  padding='same', activation='sigmoid')(x)  # Output: heatmap
    x = tf.keras.layers.Resizing(h, w, interpolation='bilinear')(x)  # ensure full resolution
    return x


def heatmap_head_interp(inputs, h, w):
    x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(1, 1, padding='same', activation='sigmoid')(x)  # Output: heatmap
    x = tf.keras.layers.Resizing(h, w, interpolation='bilinear')(x)  # ensure full resolution
    return x


def pretrained_backbone(inp, unfreeze_frac=0.3, h=None, w=None, c=None, trainable=False):
    x = tf.keras.ops.cast(inp, "float32")
    x = tf.keras.applications.efficientnet_v2.preprocess_input(x)
    backbone = EfficientNetV2B3(input_shape=(h, w, c), include_top=False, weights="imagenet")

    # Freeze the backbone (optional for transfer learning)
    backbone.trainable = False  

    if trainable:
        unfreeze_fraction = unfreeze_frac
        n_total = len(backbone.layers)
        n_unfreeze = int(n_total * unfreeze_fraction)

        # Unfreeze top layers
        for layer in backbone.layers[-n_unfreeze:]:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True
            else:
                # Optionally keep BatchNorm frozen (recommended for stability)
                layer.trainable = False

    # Build the final model
    x = backbone(x, training=False)  # Extract features without updating backbone weights

    return x


def generator(h, w, c, unfreeze_frac=0.3, trainable=False, use_heatmap=False):
    inp = tf.keras.Input(shape=(h, w, c), name='inp_layer')
    features = pretrained_backbone(inp, h=h, w=w, c=c, unfreeze_frac=unfreeze_frac, trainable=trainable)

    if use_heatmap:
        out = heatmap_head_deep(features)
    else:
        out = detection_head(features)

    return tf.keras.Model(inputs=inp, outputs=out)


# Xception-style residual block with depthwise separable convolutions
def xception_block(inputs, filters, strides=1):
    x = layers.SeparableConv2D(filters, 3, padding='same', strides=strides, use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.SeparableConv2D(filters, 3, padding='same', strides=1, use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    residual = inputs
    if strides > 1 or inputs.shape[-1] != filters:
        residual = layers.Conv2D(filters, 1, strides=strides, padding='same', use_bias=False)(inputs)
        residual = layers.BatchNormalization()(residual)

    x = layers.Add()([x, residual])
    x = layers.LeakyReLU()(x)
    return x

# Atrous Spatial Pyramid Pooling block (DeepLabV3-style)
def aspp_block(inputs, filters=256):
    shape = tf.shape(inputs)

    conv_1x1 = layers.Conv2D(filters, 1, padding='same', use_bias=False)(inputs)
    conv_1x1 = layers.BatchNormalization()(conv_1x1)
    conv_1x1 = layers.LeakyReLU()(conv_1x1)

    conv_3x3_r6 = layers.SeparableConv2D(filters, 3, padding='same', dilation_rate=6, use_bias=False)(inputs)
    conv_3x3_r6 = layers.BatchNormalization()(conv_3x3_r6)
    conv_3x3_r6 = layers.LeakyReLU()(conv_3x3_r6)

    conv_3x3_r12 = layers.SeparableConv2D(filters, 3, padding='same', dilation_rate=12, use_bias=False)(inputs)
    conv_3x3_r12 = layers.BatchNormalization()(conv_3x3_r12)
    conv_3x3_r12 = layers.LeakyReLU()(conv_3x3_r12)

    conv_3x3_r18 = layers.SeparableConv2D(filters, 3, padding='same', dilation_rate=18, use_bias=False)(inputs)
    conv_3x3_r18 = layers.BatchNormalization()(conv_3x3_r18)
    conv_3x3_r18 = layers.LeakyReLU()(conv_3x3_r18)

    # Image-level features
    pooled = layers.GlobalAveragePooling2D()(inputs)
    pooled = layers.Reshape((1, 1, inputs.shape[-1]))(pooled)
    pooled = layers.Conv2D(filters, 1, padding='same', use_bias=False)(pooled)
    pooled = layers.UpSampling2D(size=(shape[1], shape[2]), interpolation='bilinear')(pooled)

    x = layers.Concatenate()([conv_1x1, conv_3x3_r6, conv_3x3_r12, conv_3x3_r18, pooled])
    x = layers.Conv2D(filters, 1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    return x

# Decoder upsampling block
def upsample_block(inputs, skip, filters):
    x = layers.UpSampling2D()(inputs)
    x = layers.Concatenate()([x, skip])
    x = layers.SeparableConv2D(filters, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.SeparableConv2D(filters, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    return x

# Final model builder
def build_modified_unet_aspp(input_shape=(224, 224, 3), num_classes=1):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    e1 = xception_block(inputs, 32)          # 224 → 224
    e2 = xception_block(e1, 64, strides=2)   # 224 → 112
    e3 = xception_block(e2, 128, strides=2)  # 112 → 56
    e4 = xception_block(e3, 256, strides=2)  # 56 → 28
    e5 = xception_block(e4, 512, strides=2)  # 28 → 14

    # Bridge: ASPP
    b = aspp_block(e5, filters=256)          # 14 × 14 context-rich features

    # Decoder
    d4 = upsample_block(b, e4, 256)          # 14 → 28
    d3 = upsample_block(d4, e3, 128)         # 28 → 56
    d2 = upsample_block(d3, e2, 64)          # 56 → 112
    d1 = upsample_block(d2, e1, 32)          # 112 → 224

    outputs = layers.Conv2D(num_classes, 1, padding='same', activation='sigmoid')(d1)

    return Model(inputs, outputs)
