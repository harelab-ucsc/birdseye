import os
import glob2
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math

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


def predict_frame_heatmap(frame, model, edge_buffer=(81, 81), tile_size=(224, 224)):
    h, w = frame.shape[:2]
    # print('frame height, width', h, w)
    th, tw = tile_size
    tile_rows = math.ceil(h/th)
    tile_cols = math.ceil(w/tw)
    # print('rows, cols: ', tile_rows, tile_cols)
    delta_y = math.ceil(th - h/tile_rows)
    delta_x = math.ceil(tw - w/tile_cols)
    stride_y = th - delta_y
    stride_x = tw - delta_x
    # print('strides: ', stride_y, stride_x)
    # stride_y = th - tile_overlap
    # stride_x = tw - tile_overlap

    heatmap_full = np.zeros((h, w), dtype=np.float32)
    weight_mask = np.zeros((h, w), dtype=np.float32)

    tiles = []
    positions = []

    # Step 1: Extract tiles
    for y in range(edge_buffer[0], h - edge_buffer[0] - th + delta_y+ 1, stride_y):
        for x in range(edge_buffer[1], w - edge_buffer[1] - tw + delta_x + 1, stride_x):
            # print(f'y, x: {y},{x} ({stride_y}, {stride_x})')
            tile = frame[y:y+th, x:x+tw]
            if tile.shape != (224, 224, 3):
                pad_y = 224 - tile.shape[0]
                pad_x = 224 - tile.shape[1]
                tile = np.pad(
                    tile,
                    ((0, pad_y), (0, pad_x), (0, 0)),
                    mode='constant',
                    constant_values=0
                )
            tiles.append(tile)
            positions.append((y, x))

    tiles = np.array(tiles, dtype=np.uint8)
    tiles_tf = tf.convert_to_tensor(tiles)

    # Step 2: Run prediction in batches
    preds = model.predict(tiles_tf, batch_size=32)

    # Step 3: Reassemble into heatmap
    for i, (y, x) in enumerate(positions):
        pred_tile = preds[i].squeeze()
        # TODO: strip the pads from tiles which received pads
                # Determine original tile height and width (before padding)
        tile_h = min(th, h - y)
        tile_w = min(tw, w - x)

        heatmap_full[y:y+tile_h, x:x+tile_w] += pred_tile[:tile_h, :tile_w]
        weight_mask[y:y+tile_h, x:x+tile_w] += 1.0


    # Step 4: Normalize overlapping regions
    heatmap_full = np.divide(
        heatmap_full,
        weight_mask,
        out=np.zeros_like(heatmap_full),
        where=weight_mask > 0
    )

    return heatmap_full


def postprocess_heatmap(heatmap, blur=True, blurs=10, thresh=0.5, min_area=10):
    if blur:
        for blur in range(blurs):
            _, binary = cv2.threshold(heatmap, thresh, 1.0, cv2.THRESH_BINARY)
            heatmap = cv2.GaussianBlur(heatmap, (11, 11), sigmaX=1)

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
        # os.path.join(os.path.expanduser('~'), 'birdseye_CNN_data', '2025_04_16', 'pieranch_rect'),
        os.path.join(os.path.expanduser('~'), 'parsed_flights', '2025_04_16', 'pieranch_rect'),
    ]

    # models_dir = os.path.join(os.path.expanduser('~'), 'birdseye', 'models')
    models_dir = os.path.join(os.path.expanduser('~'), 'ros2_ws', 'src', 'birdseye', 'models')
    weight_file = 'birdseye_224_224_008.weights.h5'

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
        for frame in frames:
            print(f'predicting on {frame}')

            image = tf.io.decode_png(tf.io.read_file(frame), channels=3)
            ax[0].imshow(image.numpy())
            pred = predict_frame_heatmap(image, model)
            pred = tf.squeeze(pred).numpy()
            pred /= pred.max()

            # ax[2].hist(pred.flatten())

            det, pred = postprocess_heatmap(pred, thresh=0.25)
            ax[1].imshow(pred, cmap='hot')

            # ax[2].hist(pred.flatten())

            fig.canvas.draw_idle()
            plt.pause(0.2)
            # p = os.path.expanduser('~')
            # p = os.path.join(p, 'catch', 'tmp', f'3d_{str(self.frame_index).rjust(3,str(0))}.png')
            # self.fig.savefig(p)
            ax[0].cla()
            ax[1].cla()
            # ax[2].cla()
