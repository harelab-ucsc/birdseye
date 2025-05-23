import tensorflow as tf

def yolo_point_loss():
    def loss_fn(y_true, y_pred):
        obj_true = y_true[..., 4]
        obj_pred = y_pred[..., 4]

        # BCE loss on objectness
        bce = tf.keras.losses.binary_crossentropy(obj_true, obj_pred)
        bce_loss = tf.reduce_mean(bce)

        # MSE loss on (x, y) coords, only where obj=1
        obj_mask = tf.cast(obj_true > 0.5, tf.float32)
        xy_diff = tf.square(y_true[..., 0:2] - y_pred[..., 0:2])
        xy_loss = tf.reduce_sum(obj_mask[..., tf.newaxis] * xy_diff)
        xy_loss = xy_loss / (tf.reduce_sum(obj_mask) + 1e-6)

        return bce_loss + xy_loss

    return loss_fn

