import tensorflow as tf

# Binary focal loss computed from logits (not probabilities) 
def binary_focal_loss_from_logits(gamma=2.0, alpha=0.25):
    def loss_fn(y_true, y_pred):
        # Convert logits to probabilities for modulation terms
        prob = tf.sigmoid(y_pred)
        ce_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)

        # Focal modulation
        p_t = y_true * prob + (1 - y_true) * (1 - prob)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        modulating = tf.pow(1.0 - p_t, gamma)

        loss = alpha_t * modulating * ce_loss
        return tf.reduce_mean(loss)
    return loss_fn


# YOLO-style point loss: focal loss for objectness, MSE for coordinates 
def yolo_point_loss(obj_weight=5.0, coord_weight=1.0, gamma=2.0, alpha=0.25):
    focal_loss = binary_focal_loss_from_logits(gamma=gamma, alpha=alpha)

    def loss_fn(y_true, y_pred):
        obj_true = y_true[..., 4]                    # Ground truth objectness
        obj_pred = y_pred[..., 4]                    # Predicted logits (not sigmoid)

        # Objectness loss (focal loss with logits)
        bce_loss = focal_loss(obj_true, obj_pred)

        # Coordinate loss (apply sigmoid to predictions to constrain to [0,1])
        obj_mask = tf.cast(obj_true > 0.5, tf.float32)
        xy_diff = tf.square(y_true[..., 0:2] - tf.sigmoid(y_pred[..., 0:2]))
        xy_loss = tf.reduce_sum(obj_mask[..., tf.newaxis] * xy_diff)
        xy_loss = xy_loss / (tf.reduce_sum(obj_mask) + 1e-6)

        return obj_weight * bce_loss + coord_weight * xy_loss

    return loss_fn

# BCE LOSS

# def yolo_point_loss(obj_weight=5.0, coord_weight=1.0):
#     def loss_fn(y_true, y_pred):
#         obj_true = y_true[..., 4]
#         obj_pred = y_pred[..., 4]

#         # Objectness loss
#         bce = tf.keras.losses.binary_crossentropy(obj_true, obj_pred)
#         bce_loss = tf.reduce_mean(bce)

#         # Coordinate loss (only where object exists)
#         obj_mask = tf.cast(obj_true > 0.5, tf.float32)
#         xy_diff = tf.square(y_true[..., 0:2] - y_pred[..., 0:2])
#         xy_loss = tf.reduce_sum(obj_mask[..., tf.newaxis] * xy_diff)
#         xy_loss = xy_loss / (tf.reduce_sum(obj_mask) + 1e-6)

#         return obj_weight * bce_loss + coord_weight * xy_loss
#     return loss_fn

