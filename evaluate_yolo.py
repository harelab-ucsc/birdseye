import os
import glob2
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

from frame_loader_yolo import FrameLoader
from yolo_generator import yolo_model  # or wherever yolo_model is defined

def decode_predictions(pred_tensor, threshold=0.3, grid_size=13, img_size=(416, 416)):
    """
    Convert YOLO-style prediction tensor to list of (x, y, conf, class) detections.
    """
    pred_tensor = np.squeeze(pred_tensor)  # shape: (13, 13, 5 + num_classes)
    detections = []

    cell_h = img_size[0] / grid_size
    cell_w = img_size[1] / grid_size

    for gy in range(grid_size):
        for gx in range(grid_size):
            cell = pred_tensor[gy, gx]
            conf = cell[4]
            if conf < threshold:
                continue

            x_rel, y_rel, w, h = cell[0:4]
            abs_x = (gx + x_rel) * cell_w
            abs_y = (gy + y_rel) * cell_h
            class_id = int(np.argmax(cell[5:]))

            detections.append((abs_x, abs_y, conf, class_id))

    return detections

def visualize_predictions(image, predictions, labels=None):
    vis = image.copy()

    # Draw predictions
    for x, y, conf, cls in predictions:
        color = (0, 255, 0) if cls == 0 else (0, 0, 255)
        cv2.circle(vis, (int(x), int(y)), 5, color, 2)
        cv2.putText(vis, f"{conf:.2f}", (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Draw ground-truth labels if provided
    if labels:
        for x, y, cls in labels:
            color = (255, 255, 0) if cls == 0 else (255, 0, 255)
            cv2.circle(vis, (int(x), int(y)), 4, color, 1)

    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

def evaluate(image_path, weights_path, label_file=None):
    image_size = (416, 416)
    num_classes = 2
    grid_size = 13

    # Load and preprocess image
    raw_image = cv2.imread(image_path)
    raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(raw_image, image_size)
    input_img = resized.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(input_img, axis=0)

    # Load model
    model = yolo_model(input_shape=(416, 416, 3), num_classes=num_classes)
    model.load_weights(weights_path)

    # Predict
    pred = model.predict(input_tensor)[0]  # shape: (13, 13, 5 + num_classes)

    # Decode predictions
    detections = decode_predictions(pred, threshold=0.3, grid_size=grid_size, img_size=image_size)

    # Optional: Load ground-truth labels
    labels = None
    if label_file:
        loader = FrameLoader(image_size=image_size, num_classes=num_classes)
        labels = loader.load_labels(label_file, image_path)

    # Visualize
    visualize_predictions(cv2.cvtColor(resized, cv2.COLOR_RGB2BGR), detections, labels)

if __name__ == "__main__":
    image_path = "test_images/example.png"
    weights_path = "models/birdseye_416_416_yolo.weights.h5"
    label_file = "labels.txt"  # optional

    evaluate(image_path, weights_path, label_file)
