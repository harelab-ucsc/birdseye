import os
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


if __name__ == '__main__':
    # Example usage
    infer = TileInference(model_weights="birds_eye_model.weights.h5", use_heatmaps=False)
    predictions = infer.predict_on_image("/path/to/image.png", label_file="labels.txt")
    print("Predictions:", predictions)
    infer.visualize_predictions("/path/to/image.png", label_file="labels.txt")
