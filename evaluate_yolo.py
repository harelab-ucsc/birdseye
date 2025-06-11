import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

from frame_loader_yolo import FrameLoader
from yolo_generator import yolo_model


def decode_predictions(pred_tensor, threshold=0.3, grid_size=13, img_size=(416, 416), anchors=5, verbose=True):

    detections = []
    cell_h = img_size[0] / grid_size
    cell_w = img_size[1] / grid_size

    # Debug: objectness stats
    obj_scores = pred_tensor[..., 4]
    if verbose:
        print("Max objectness score:", np.max(obj_scores))
        print("Mean objectness score:", np.mean(obj_scores))
        print(f"Total cells above threshold {threshold}:", np.sum(obj_scores > threshold))

    for gy in range(grid_size):
        for gx in range(grid_size):
            for a in range(anchors):
                cell = pred_tensor[gy, gx, a]
                conf = cell[4]
                if conf < threshold:
                    continue

                x_rel, y_rel, w, h = cell[0:4]
                abs_x = (gx + x_rel) * cell_w
                abs_y = (gy + y_rel) * cell_h

                if cell.shape[-1] > 6:
                    class_id = int(np.argmax(cell[5:]))
                else:
                    class_id = 0

                detections.append((abs_x, abs_y, conf, class_id))

    if verbose:
        print(f"Total detections returned: {len(detections)}")
    return detections

# def decode_predictions(pred_tensor, threshold=0.3, grid_size=13, img_size=(416, 416), anchors=5):
#     detections = []
#     cell_h = img_size[0] / grid_size
#     cell_w = img_size[1] / grid_size

#     for gy in range(grid_size):
#         for gx in range(grid_size):
#             for a in range(anchors):
#                 cell = pred_tensor[gy, gx, a]
#                 conf = cell[4]
#                 if conf < threshold:
#                     continue

#                 x_rel, y_rel, w, h = cell[0:4]
#                 abs_x = (gx + x_rel) * cell_w
#                 abs_y = (gy + y_rel) * cell_h
#                 class_id = 0 if cell.shape[-1] == 6 else int(np.argmax(cell[5:]))
#                 detections.append((abs_x, abs_y, conf, class_id))
#     return detections

def visualize_predictions(image, predictions, labels=None):
    vis = image.copy()

    for x, y, conf, cls in predictions:
        color = (0, 255, 0) if cls == 0 else (0, 0, 255)
        cv2.circle(vis, (int(x), int(y)), 5, color, 2)
        cv2.putText(vis, f"{conf:.2f}", (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    if labels:
        for x, y, cls in labels:
            color = (255, 255, 0) if cls == 0 else (255, 0, 255)
            cv2.circle(vis, (int(x), int(y)), 4, color, 1)

    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Predictions and Ground Truth")
    plt.show()

def visualize_objectness_map(pred_tensor):
    obj_map = pred_tensor[..., 4]  # (13, 13, 5)
    heatmap = np.mean(obj_map, axis=-1)  # average across anchors

    plt.figure(figsize=(6, 5))
    plt.imshow(heatmap, cmap="hot", interpolation="nearest")
    plt.title("Average Objectness Heatmap")
    plt.colorbar()
    plt.show()

def evaluate(image_path, weights_path, label_file=None):
    image_size = (416, 416)
    num_classes = 1
    grid_size = 13
    anchors = 5

    # Load and preprocess image
    raw_image = cv2.imread(image_path)
    raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(raw_image, image_size)
    input_img = resized.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(input_img, axis=0)

    # Load model
    model = yolo_model(input_shape=(416, 416, 3), num_classes=num_classes, anchors=anchors)
    model.load_weights(weights_path)

    # Predict
    pred = model.predict(input_tensor)[0]  # shape: (13, 13, 5, 6)
    visualize_objectness_map(pred)  # confirm if model is firing

    # Decode detections
    detections = decode_predictions(pred, threshold=0.3, grid_size=grid_size, img_size=image_size, anchors=anchors)

    # Load ground-truth labels
    labels = None
    if label_file:
        loader = FrameLoader(image_size=image_size, num_classes=num_classes)
        labels = loader.load_labels(label_file, image_path)

    # Visualize prediction vs label
    visualize_predictions(cv2.cvtColor(resized, cv2.COLOR_RGB2BGR), detections, labels)

if __name__ == "__main__":
    image_path = "/home/harey/birdseye_CNN_data/2025_04_16/pieranch_rect/cam0_1744818192.323051000.png"
    weights_path = "models/birdseye_416_416_009_yolo.weights.h5"
    label_file = "labels.txt"

    evaluate(image_path, weights_path, label_file)


# import os
# import numpy as np
# import tensorflow as tf
# import cv2
# import matplotlib.pyplot as plt

# from frame_loader_yolo import FrameLoader
# from yolo_generator import yolo_model

# def decode_predictions(pred_tensor, threshold=0.3, grid_size=13, img_size=(416, 416), anchors=5):
#     """
#     Decode YOLO-style tensor into a list of (x, y, conf, class_id) predictions.
#     """
#     detections = []
#     cell_h = img_size[0] / grid_size
#     cell_w = img_size[1] / grid_size

#     for gy in range(grid_size):
#         for gx in range(grid_size):
#             for a in range(anchors):
#                 cell = pred_tensor[gy, gx, a]
#                 conf = cell[4]
#                 if conf < threshold:
#                     continue

#                 x_rel, y_rel, w, h = cell[0:4]
#                 abs_x = (gx + x_rel) * cell_w
#                 abs_y = (gy + y_rel) * cell_h
#                 class_id = 0 if cell.shape[-1] == 6 else int(np.argmax(cell[5:]))
#                 detections.append((abs_x, abs_y, conf, class_id))
#     return detections

# def visualize_predictions(image, predictions, labels=None):
#     """
#     Overlay model predictions and ground-truth labels on the input image.
#     """
#     vis = image.copy()

#     for x, y, conf, cls in predictions:
#         color = (0, 255, 0) if cls == 0 else (0, 0, 255)
#         cv2.circle(vis, (int(x), int(y)), 5, color, 2)
#         cv2.putText(vis, f"{conf:.2f}", (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

#     if labels:
#         for x, y, cls in labels:
#             color = (255, 255, 0) if cls == 0 else (255, 0, 255)
#             cv2.circle(vis, (int(x), int(y)), 4, color, 1)

#     plt.figure(figsize=(10, 8))
#     plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
#     plt.axis("off")
#     plt.title("Predictions and Ground Truth")
#     plt.show()

# def evaluate(image_path, weights_path, label_file=None):
#     image_size = (416, 416)
#     num_classes = 1
#     grid_size = 13
#     anchors = 5

#     # Load image
#     raw_image = cv2.imread(image_path)
#     raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
#     resized = cv2.resize(raw_image, image_size)
#     input_img = resized.astype(np.float32) / 255.0
#     input_tensor = np.expand_dims(input_img, axis=0)

#     # Load model
#     model = yolo_model(input_shape=(416, 416, 3), num_classes=num_classes, anchors=anchors)
#     model.load_weights(weights_path)

#     # Run prediction
#     pred = model.predict(input_tensor)[0]  # shape: (13, 13, anchors, 5 + num_classes)

#     # Decode detections
#     detections = decode_predictions(pred, threshold=0.3, grid_size=grid_size, img_size=image_size, anchors=anchors)

#     # Load ground truth labels if provided
#     labels = None
#     if label_file:
#         loader = FrameLoader(image_size=image_size, num_classes=num_classes)
#         labels = loader.load_labels(label_file, image_path)

#     # Visualize result
#     visualize_predictions(cv2.cvtColor(resized, cv2.COLOR_RGB2BGR), detections, labels)

# if __name__ == "__main__":
#     image_path = "test_images/example.png"  # <-- update as needed
#     weights_path = "models/birdseye_416_416_yolo.weights.h5"  # <-- update as needed
#     label_file = "labels.txt"  # optional

#     evaluate(image_path, weights_path, label_file)