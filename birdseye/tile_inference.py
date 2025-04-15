import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def tile_image(image, tile_x, tile_y, stride_x=None, stride_y=None, edge_buffer=0):
    """Split image into tiles, skipping those too close to edges."""
    stride_x = stride_x or tile_x
    stride_y = stride_y or tile_y

    tiles = []
    positions = []
    h, w, _ = image.shape

    for y in range(0, h, stride_y):
        for x in range(0, w, stride_x):
            if y + tile_y > h or x + tile_x > w:
                continue
            if x < edge_buffer or y < edge_buffer:
                continue
            if x + tile_x > w - edge_buffer or y + tile_y > h - edge_buffer:
                continue

            tile = image[y:y+tile_y, x:x+tile_x]
            tiles.append(tile)
            positions.append((y, x))

    return tiles, positions, (h, w)


def reassemble_tiles(preds, positions, full_shape, tile_size, output_type='scalar', mode='avg'):
    """
    Reassemble predicted tiles into a full-size result (e.g., heatmap or scalar mask).
    """
    tile_y, tile_x = tile_size
    h, w = full_shape
    result = np.zeros((h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)

    for (y, x), pred in zip(positions, preds):
        pred = np.squeeze(pred)

        if output_type == 'scalar':
            val = pred if np.isscalar(pred) else pred[0]
            pred_tile = np.full((tile_y, tile_x), val, dtype=np.float32)
        elif output_type == 'heatmap':
            if pred.shape != (tile_y, tile_x):
                pred = tf.image.resize(pred[..., np.newaxis], [tile_y, tile_x]).numpy().squeeze()
            pred_tile = pred
        else:
            raise ValueError(f"Unsupported output_type: {output_type}")

        result[y:y+tile_y, x:x+tile_x] += pred_tile
        weight[y:y+tile_y, x:x+tile_x] += 1.0

    return result / np.maximum(weight, 1e-6) if mode == 'avg' else result


def predict_full_image(
    model,
    image,
    tile_size=(224, 224),
    stride=None,
    return_prob=False,
    output_type='scalar',
    edge_buffer=0
):
    """
    Predict over an entire image by tiling, model inference, and reassembling output.
    """
    if isinstance(image, tf.Tensor):
        image = image.numpy()
    if image.max() > 1.0:
        image = (image * 255.0).astype(np.uint8)

    tile_y, tile_x = tile_size
    stride_y = stride_x = stride or tile_x  # default: non-overlapping

    tiles, positions, full_shape = tile_image(image, tile_x, tile_y, stride_x, stride_y, edge_buffer)
    if len(tiles) == 0:
        raise ValueError("No tiles produced — try reducing edge_buffer or image size.")

    tile_array = np.array(tiles, dtype=np.uint8)

    preds = model.predict(tile_array, verbose=0)

    if output_type == 'scalar' and not return_prob:
        preds = (preds > 0.5).astype(np.float32)

    heatmap = reassemble_tiles(preds, positions, full_shape, (tile_y, tile_x), output_type=output_type)
    return heatmap


def visualize_prediction_overlay(
    image,
    prediction,
    cmap='jet',
    alpha=0.5,
    threshold=None,
    figsize=(8, 8),
    title=None,
    show=True,
    ground_truth=None
):
    """
    Display the original image overlaid with a prediction heatmap or binary mask.
    """
    if isinstance(image, tf.Tensor):
        image = image.numpy()
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    img_rgb = image[..., :3] if image.shape[-1] > 1 else np.repeat(image, 3, axis=-1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img_rgb)

    if threshold is not None:
        overlay = (prediction > threshold).astype(np.float32)
    else:
        overlay = prediction

    ax.imshow(overlay, cmap=cmap, alpha=alpha)
    ax.set_title(title or "Prediction Overlay")
    ax.axis('off')

    if ground_truth is not None:
        plt.figure(figsize=figsize)
        plt.imshow(ground_truth, cmap='gray')
        plt.title("Ground Truth")
        plt.axis("off")

    if show:
        plt.show()

