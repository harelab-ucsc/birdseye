import os
import glob2
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2

from frame_loader_yolo import FrameLoader
from yolo_generator import yolo_model


class YOLOFrameInference:
    def __init__(self, model_weights, input_shape=(416, 416, 3), num_classes=1):
        self.model = yolo_model(input_shape=input_shape, num_classes=num_classes)
        self.model.load_weights(model_weights)
        self.model.compile()  # No-op for inference, but avoids Keras warnings
        self.loader = FrameLoader(image_size=input_shape[:2], num_classes=num_classes)

    def predict_and_visualize(self, image_path, label_file=None, threshold=0.3):
        image_raw = tf.io.decode_png(tf.io.read_file(image_path), channels=3)
        image_resized = tf.image.resize(image_raw, self.loader.image_size)

        image_input = tf.expand_dims(image_resized, axis=0)
        preds = self.model.predict(image_input, verbose=0)[0]

        # Extract prediction values
        x_center, y_center, w, h, obj_score = preds[:5]
        class_scores = preds[5:]

        if obj_score < threshold:
            print("No confident object detected.")
            return

        img_np = image_resized.numpy().astype(np.uint8)
        h_img, w_img = self.loader.image_size

        # Convert normalized x_center, y_center to pixel coordinates
        cx = int(x_center * w_img)
        cy = int(y_center * h_img)

        # Draw point and label
        img_with_point = img_np.copy()
        cv2.circle(img_with_point, (cx, cy), radius=6, color=(0, 255, 0), thickness=-1)
        cv2.putText(img_with_point, f"obj: {obj_score:.2f}", (cx + 5, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Show result
        plt.figure(figsize=(6, 6))
        plt.imshow(img_with_point)
        plt.title("YOLO Point Prediction")
        plt.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    frames_dirs = [
        os.path.join(os.path.expanduser('~'), 'birdseye_CNN_data', '2025_04_16', 'pieranch_rect'),
    ]

    models_dir = os.path.join(os.path.expanduser('~'), 'birdseye', 'models')
    weight_file = 'birdseye_1920_1200_013_yolo.weights.h5'  # updated to match config
    model_path = os.path.join(models_dir, weight_file)

    infer = YOLOFrameInference(model_weights=model_path, input_shape=(416, 416, 3), num_classes=2)

    for _dir in frames_dirs:
        image_paths = sorted(glob2.glob(os.path.join(_dir, '*.png')))
        for image_path in image_paths[:10]:  # limit to 10 for quick view
            print(f" Predicting {image_path}")
            infer.predict_and_visualize(image_path, label_file="labels.txt")
