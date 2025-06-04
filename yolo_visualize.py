import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import glob2
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2

from frame_loader_yolo import FrameLoader
from yolo_generator import yolo_model

class YOLOFrameInference:
    def __init__(self, model_weights, num_classes=2):
        # Inferred input shape from weights
        dummy = yolo_model(input_shape=(416, 416, 3), num_classes=num_classes)
        dummy.load_weights(model_weights)
        self.input_shape = dummy.input_shape[1:]
        print(f"Inferred model input shape: {self.input_shape}")

        self.model = yolo_model(input_shape=self.input_shape, num_classes=num_classes)
        self.model.load_weights(model_weights)
        self.model.compile()

        self.loader = FrameLoader(image_size=self.input_shape[:2], num_classes=num_classes)

    def predict_points(self, image_path, conf_threshold=0.05):
        image_raw = tf.io.decode_png(tf.io.read_file(image_path), channels=3)
        image_resized = tf.image.resize(image_raw, self.loader.image_size)
        image_input = tf.expand_dims(image_resized, axis=0)

        preds = self.model.predict(image_input, verbose=0)[0]
        grid_h, grid_w, num_anchors, _ = preds.shape

        points = []
        for y in range(grid_h):
            for x in range(grid_w):
                for a in range(num_anchors):
                    cell = preds[y, x, a]
                    obj_score = cell[4]
                    if obj_score < conf_threshold:
                        continue
                    x_rel, y_rel = cell[0], cell[1]
                    px = int((x + x_rel) * (self.loader.image_size[1] / grid_w))
                    py = int((y + y_rel) * (self.loader.image_size[0] / grid_h))
                    points.append((px, py))
        return points

    def predict_and_visualize(self, image_path, threshold=0.3):
        image_raw = tf.io.decode_png(tf.io.read_file(image_path), channels=3)
        image_resized = tf.image.resize(image_raw, self.loader.image_size)
        image_input = tf.expand_dims(image_resized, axis=0)

        preds = self.model.predict(image_input, verbose=0)[0]
        grid_h, grid_w, num_anchors, _ = preds.shape
        vis = image_resized.numpy().astype(np.uint8).copy()

        count = 0
        for y in range(grid_h):
            for x in range(grid_w):
                for a in range(num_anchors):
                    cell = preds[y, x, a]
                    obj_score = cell[4]
                    if obj_score < threshold:
                        continue

                    x_rel, y_rel = cell[0], cell[1]
                    px = int((x + x_rel) * (self.loader.image_size[1] / grid_w))
                    py = int((y + y_rel) * (self.loader.image_size[0] / grid_h))

                    cv2.circle(vis, (px, py), 5, (0, 255, 0), -1)
                    cv2.putText(vis, f"{obj_score:.2f}", (px + 5, py - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    count += 1

        print(f"{count} detections in {os.path.basename(image_path)}")
        plt.figure(figsize=(8, 6))
        plt.imshow(vis)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    frames_dirs = [
        os.path.join(os.path.expanduser('~'), 'birdseye_CNN_data', '2025_04_16', 'pieranch_rect'),
    ]
    models_dir = os.path.join(os.path.expanduser('~'), 'birdseye', 'models')
    weight_file = 'birdseye_416_416_013yolo.weights.h5'
    model_path = os.path.join(models_dir, weight_file)

    infer = YOLOFrameInference(model_weights=model_path, num_classes=1)

    for _dir in frames_dirs:
        image_paths = sorted(glob2.glob(os.path.join(_dir, '*.png')))
        for image_path in image_paths[:10]:
            print(f" Predicting {image_path}")
            infer.predict_and_visualize(image_path)


# import os
# import glob2
# import tensorflow as tf
# import matplotlib.pyplot as plt
# import numpy as np
# import cv2

# from frame_loader_yolo import FrameLoader
# from yolo_generator import yolo_model


# class YOLOFrameInference:
#     def __init__(self, model_weights, input_shape=(416, 416, 3), num_classes=2):
#         self.model = yolo_model(input_shape=input_shape, num_classes=num_classes)
#         self.model.load_weights(model_weights)
#         self.model.compile()  # No-op for inference, but avoids Keras warnings
#         self.loader = FrameLoader(image_size=input_shape[:2], num_classes=num_classes)

#     def predict_and_visualize(self, image_path, label_file=None, threshold=0.3):
#         image_raw = tf.io.decode_png(tf.io.read_file(image_path), channels=3)
#         image_resized = tf.image.resize(image_raw, self.loader.image_size)

#         image_input = tf.expand_dims(image_resized, axis=0)
#         preds = self.model.predict(image_input, verbose=0)[0]

#         # Extract prediction values
#         x_center, y_center, w, h, obj_score = preds[:5]
#         class_scores = preds[5:]

#         if obj_score < threshold:
#             print("No confident object detected.")
#             return

#         img_np = image_resized.numpy().astype(np.uint8)
#         h_img, w_img = self.loader.image_size

#         # Convert normalized x_center, y_center to pixel coordinates
#         cx = int(x_center * w_img)
#         cy = int(y_center * h_img)

#         # Draw point and label
#         img_with_point = img_np.copy()
#         cv2.circle(img_with_point, (cx, cy), radius=6, color=(0, 255, 0), thickness=-1)
#         cv2.putText(img_with_point, f"obj: {obj_score:.2f}", (cx + 5, cy - 5),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

#         # Show result
#         plt.figure(figsize=(6, 6))
#         plt.imshow(img_with_point)
#         plt.title("YOLO Point Prediction")
#         plt.axis("off")
#         plt.tight_layout()
#         plt.show()

#     def predict_points(self, image_path, conf_threshold=0.3):
#         image_raw = tf.io.decode_png(tf.io.read_file(image_path), channels=3)
#         image_resized = tf.image.resize(image_raw, self.loader.image_size)
#         image_input = tf.expand_dims(image_resized, axis=0)

#         preds = self.model.predict(image_input, verbose=0)[0]
#         grid_h, grid_w = preds.shape[:2]
#         pred = preds.reshape((grid_h, grid_w, self.anchors, 5 + self.num_classes))

#         points = []
#         for y in range(grid_h):
#             for x in range(grid_w):
#                 for b in range(self.anchors):
#                     bx = pred[y, x, b]
#                     obj_score = 1 / (1 + np.exp(-bx[4]))  # sigmoid
#                     class_probs = 1 / (1 + np.exp(-bx[5:]))
#                     conf = obj_score * np.max(class_probs)

#                     if conf < conf_threshold:
#                         continue

#                     cx = (x + 1 / (1 + np.exp(-bx[0]))) / grid_w
#                     cy = (y + 1 / (1 + np.exp(-bx[1]))) / grid_h

#                     px = int(cx * self.loader.image_size[1])
#                     py = int(cy * self.loader.image_size[0])
#                     points.append((px, py))

#         return points



# if __name__ == '__main__':
#     frames_dirs = [
#         os.path.join(os.path.expanduser('~'), 'birdseye_CNN_data', '2025_02_05', 'haybarn_01_rect'),
#     ]

#     models_dir = os.path.join(os.path.expanduser('~'), 'birdseye', 'models')
#     weight_file = 'birdseye_1920_1200_013_yolo.weights.h5'  # updated to match config
#     model_path = os.path.join(models_dir, weight_file)

#     infer = YOLOFrameInference(model_weights=model_path, input_shape=(416, 416, 3), num_classes=2)

#     for _dir in frames_dirs:
#         image_paths = sorted(glob2.glob(os.path.join(_dir, '*.png')))
#         for image_path in image_paths[:10]:  # limit to 10 for quick view
#             print(f" Predicting {image_path}")
#             infer.predict_and_visualize(image_path, label_file="labels.txt")