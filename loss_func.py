import tensorflow as tf

def yolo_point_loss(obj_weight=5.0, coord_weight=1.0):
    def loss_fn(y_true, y_pred):
        obj_true = y_true[..., 4]
        obj_pred = y_pred[..., 4]

        # Objectness loss
        bce = tf.keras.losses.binary_crossentropy(obj_true, obj_pred)
        bce_loss = tf.reduce_mean(bce)

        # Coordinate loss (only where object exists)
        obj_mask = tf.cast(obj_true > 0.5, tf.float32)
        xy_diff = tf.square(y_true[..., 0:2] - y_pred[..., 0:2])
        xy_loss = tf.reduce_sum(obj_mask[..., tf.newaxis] * xy_diff)
        xy_loss = xy_loss / (tf.reduce_sum(obj_mask) + 1e-6)

        return obj_weight * bce_loss + coord_weight * xy_loss
    return loss_fn

