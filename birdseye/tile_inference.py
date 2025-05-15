import os
import glob2
import cv2

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from tile_loader import TileLoader
from generator import generator

class TileInference:
    def __init__(self, model_weights, tile_size=(224, 224), edge_buffer=81, use_heatmaps=False):
        self.tile_loader = TileLoader(
            tile_size=tile_size,
            edge_buffer=edge_buffer,
            use_heatmaps=use_heatmaps,
            include_negatives=True
        )
        self.model = generator()
        self.model.load_weights(model_weights)
        self.model.compile()
        self.use_heatmaps = use_heatmaps

    def predict_on_image(self, image_path, label_file=None):
        dataset = self.tile_loader.build_dataset(
            file_list=[image_path],
            label_file=label_file or "",  # dummy if unused
            batch_size=1,
            buffer_size=1,
            repeat=False,
            augment=False
        )

        predictions = []
        for images, _ in dataset:
            preds = self.model.predict(images, verbose=0)
            predictions.extend(preds)
        return predictions

    def visualize_predictions(self, image_path, label_file=None):
        import matplotlib.pyplot as plt

        dataset = self.tile_loader.build_dataset(
            file_list=[image_path],
            label_file=label_file or "",  # dummy if unused
            batch_size=1,
            buffer_size=1,
            repeat=False,
            augment=False
        )

        for images, labels in dataset.take(1):
            preds = self.model.predict(images, verbose=0)

            for i in range(min(6, images.shape[0])):
                plt.figure(figsize=(4, 4))
                plt.subplot(1, 2, 1)
                plt.imshow(images[i].numpy().astype(np.uint8))
                plt.title("Tile")
                plt.axis("off")

                plt.subplot(1, 2, 2)
                if self.use_heatmaps:
                    plt.imshow(labels[i].numpy().squeeze(), cmap='hot')
                    plt.title("Heatmap")
                else:
                    plt.text(0.5, 0.5, f"Pred: {preds[i][0]:.2f}", ha='center', va='center', fontsize=16)
                    plt.title("Prediction")
                    plt.axis("off")
                plt.tight_layout()
                plt.show()


def predict_frame_heatmap(frame, model, tile_size=(224, 224), tile_overlap=(30,40)):
    h, w = frame.shape[:2]
    th, tw = tile_size
    stride_y = th - tile_overlap[0]
    stride_x = tw - tile_overlap[1]

    heatmap_full = np.zeros((h, w), dtype=np.float32)
    weight_mask = np.zeros((h, w), dtype=np.float32)

    tiles = []
    positions = []

    # Step 1: Extract tiles
    for y in range(0, h - th + 1, stride_y):
        for x in range(0, w - tw + 1, stride_x):
            tile = frame[y:y+th, x:x+tw]
            tiles.append(tile)
            positions.append((y, x))

    tiles = np.array(tiles, dtype=np.uint8)
    tiles_tf = tf.convert_to_tensor(tiles)

    # Step 2: Run prediction in batches
    preds = model.predict(tiles_tf, batch_size=32)

    # Step 3: Reassemble into heatmap
    for i, (y, x) in enumerate(positions):
        pred_tile = preds[i].squeeze()
        heatmap_full[y:y+th, x:x+tw] += pred_tile
        weight_mask[y:y+th, x:x+tw] += 1.0

    # Step 4: Normalize overlapping regions
    heatmap_full = np.divide(
        heatmap_full,
        weight_mask,
        out=np.zeros_like(heatmap_full),
        where=weight_mask > 0
    )

    return heatmap_full


def postprocess_heatmap(heatmap, blur=True, thresh=0.5, min_area=10):
    if blur:
        heatmap = cv2.GaussianBlur(heatmap, (5, 5), sigmaX=1)

    _, binary = cv2.threshold(heatmap, thresh, 1.0, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours((binary * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                detections.append((cx, cy))
    return detections, binary


if __name__ == '__main__':
    # # Example usage
    # infer = TileInference(model_weights="birds_eye_model.weights.h5", use_heatmaps=False)
    # predictions = infer.predict_on_image("/path/to/image.png", label_file="labels.txt")
    # print("Predictions:", predictions)
    # infer.visualize_predictions("/path/to/image.png", label_file="labels.txt")

    frames_dirs = [
        os.path.join(os.path.expanduser('~'), 'birdseye_CNN_data', '2025_04_16', 'pieranch_rect'),
        # os.path.join(os.path.expanduser('~'), 'parsed_flights', '2025_04_16', 'pieranch_rect'),
    ]

    # models_dir = os.path.join(os.path.expanduser('~'), 'ros2_ws', 'src', 'birdseye', 'models')
    models_dir = os.path.join(os.path.expanduser('~'), 'birdseye', 'models')
    # weight_file = 'birdseye_224_224_016.weights.h5'
    # weight_file = 'birdseye_224_224_013.weights.h5'
    weight_file = 'birdseye_224_224_019.weights.h5'

    model = generator(224, 224, 3, use_heatmap=True)
    model.load_weights(os.path.join(models_dir, weight_file))
    model.compile()

    plt.ion()
    plt.tight_layout()
    plt.show(block=False)

    for _dir in frames_dirs:
        frames = sorted(glob2.glob(os.path.join(_dir, '*.png')))

        fig, ax = plt.subplots(1,2, figsize=(18,8))
        print()
        for i, frame in enumerate(frames):
            print(f'predicting on {frame}')

            image = tf.io.decode_png(tf.io.read_file(frame), channels=3)
            ax[0].imshow(image.numpy())
            pred = predict_frame_heatmap(image, model)
            pred = tf.squeeze(pred).numpy()
            # pred /= pred.max()

            det, pred = postprocess_heatmap(pred, thresh=0.3)
            ax[1].imshow(pred, cmap='hot')

            fig.canvas.draw_idle()
            plt.pause(0.2)
            p = os.path.expanduser('~')
            p = os.path.join(p, 'catch', 'tmp', f'2d_{str(i).rjust(3,str(0))}.png')
            fig.savefig(p)
            ax[0].cla()
            ax[1].cla()
