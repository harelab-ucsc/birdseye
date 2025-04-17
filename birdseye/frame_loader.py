import tensorflow as tf
import numpy as np
import os
import glob2
import matplotlib.pyplot as plt

class FrameLoader:
    def __init__(self, image_size=(224, 224), use_heatmaps=False):
        self.image_size = image_size
        self.use_heatmaps = use_heatmaps

    def load_labels(self, label_file, image_filename):
        """Parse labels from a text file. Each line: idx, filepath, [x,y,class]"""
        labels = []
        tmp = os.path.join(os.path.split(image_filename)[0], label_file)
        load_name = glob2.glob(tmp)[0]
        try:
            with open(load_name, "r") as f:
                for line in f:
                    _, path, data = line.strip().split()
                    x_str, y_str, class_str = data.split(',')
                    if os.path.basename(path) == os.path.basename(image_filename):
                        labels.append((float(x_str), float(y_str), float(class_str)))
        except Exception as e:
            print(f"      [Label Load Error] {e}")
        return labels

    def frame_label(self, image_np, labels):
        h, w = image_np.shape[:2]
        if self.use_heatmaps:
            heatmap = np.zeros((h, w, 1), dtype=np.float32)
            for x, y, cls in labels:
                px, py = int(x), int(y)
                if 0 <= px < w and 0 <= py < h:
                    heatmap[py, px, 0] = cls
            return heatmap
        else:
            return 1.0 if labels else 0.0

    def tf_frame_fn(self, image_path, label_file):
        def pyfunc(image_path_py):
            image_path_str = image_path_py.numpy().decode("utf-8")
            image = tf.io.read_file(image_path_str)
            image = tf.io.decode_png(image, channels=3).numpy()

            labels = self.load_labels(label_file, image_path_str)
            label = self.frame_label(image, labels)

            image = tf.image.resize(image, self.image_size).numpy().astype(np.uint8)

            return image, label

        image, label = tf.py_function(
            pyfunc,
            [image_path],
            [tf.uint8, tf.float32 if not self.use_heatmaps else tf.float32]
        )

        image.set_shape([self.image_size[0], self.image_size[1], 3])
        if self.use_heatmaps:
            label.set_shape([self.image_size[0], self.image_size[1], 1])
        else:
            label.set_shape([])

        return image, label

    def augment(self, image, label):
        image = tf.image.convert_image_dtype(image, tf.float32)
        m = tf.random.uniform([5])
        n = tf.random.uniform([], minval=-1.0, maxval=1.0)

        if m[0] < 0.5:
            image = tf.image.adjust_brightness(image, n * m[0] / 2)
        if m[1] < 0.5:
            image = tf.image.adjust_contrast(image, 1 + n * m[1])
        if m[3] < 0.5:
            image = tf.image.flip_left_right(image)
        if m[4] < 0.5:
            image = tf.image.flip_up_down(image)

        image = tf.image.convert_image_dtype(image, tf.uint8)
        return image, tf.expand_dims(label, -1) if not self.use_heatmaps else label

    def build_dataset(self, file_list, label_file, batch_size, buffer_size=64, repeat=True, augment=False):
        ds = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(file_list, dtype=tf.string))
        ds = ds.map(lambda path: self.tf_frame_fn(path, label_file), num_parallel_calls=tf.data.AUTOTUNE)

        if augment:
            ds = ds.map(self.augment, num_parallel_calls=tf.data.AUTOTUNE)
        elif not self.use_heatmaps:
            ds = ds.map(lambda x, y: (x, tf.expand_dims(y, -1)), num_parallel_calls=tf.data.AUTOTUNE)

        if repeat:
            ds = ds.repeat()

        ds = ds.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    def show_frame_batch(self, dataset, num_samples=6):
        for images, labels in dataset.take(1):
            for i in range(min(num_samples, images.shape[0])):
                plt.figure(figsize=(4, 4))
                plt.subplot(1, 2, 1)
                plt.imshow(images[i].numpy().astype(np.uint8))
                plt.title("Frame")
                plt.axis('off')

                plt.subplot(1, 2, 2)
                if self.use_heatmaps:
                    plt.imshow(labels[i].numpy().squeeze(), cmap='hot')
                    plt.title("Heatmap")
                else:
                    lbl = int(labels[i].numpy()) if tf.rank(labels[i]) == 0 else int(labels[i].numpy()[0])
                    plt.text(0.5, 0.5, f"Class: {lbl}", ha='center', va='center', fontsize=16)
                    plt.title("Label")
                    plt.axis('off')

                plt.tight_layout()
                plt.show()
