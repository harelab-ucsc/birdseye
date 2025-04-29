import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from frame_loader_yolo import FrameLoader

# --- Configuration --- #
data_root = os.path.join(os.path.expanduser('~'), 'birdseye_CNN_data')
label_file = 'labels.txt'

train_dates = [
    '2025_03_25', 
    '2025_04_04',
    '2025_04_09',
]

train_dirlists = [
    ['haybarn_original_01_01_rect', 'haybarn_eviltwin_01_01_rect'],
    ['original_01_rect', 'original_02_rect', 'eviltwin_01_rect', 'eviltwin_02_rect', 'eviltwin_03_rect'],
    ['original_01_rect', 'original_02_rect', 'eviltwin_01_rect', 'eviltwin_02_rect'],
]

# --- Build File List --- #
def collect_image_paths(dates, dirlists):
    files = []
    for d, dirs in zip(dates, dirlists):
        for subdir in dirs:
            full_path = os.path.join(data_root, d, subdir)
            if os.path.isdir(full_path):
                files.extend(tf.io.gfile.glob(os.path.join(full_path, '*.png')))
    return sorted(files)

file_list = collect_image_paths(train_dates, train_dirlists)

# --- Load Dataset --- #
loader = FrameLoader(image_size=(416, 416), num_classes=2)
train_ds = loader.build_dataset(
    file_list=file_list,
    label_file=label_file,
    batch_size=1,
    buffer_size=16,
    repeat=False,
    augment=False
)

# # --- Visualize a Sample --- #
# def visualize_yolo_sample(dataset, num_classes):
#     for images, labels in dataset.take(1):
#         image = images[0].numpy()
#         label = labels[0].numpy()

#         x_center, y_center, box_w, box_h = label[:4]
#         class_id = int(np.argmax(label[5:])) if num_classes > 1 else 0

#         h, w = image.shape[:2]
#         x1 = int((x_center - box_w / 2) * w)
#         y1 = int((y_center - box_h / 2) * h)
#         x2 = int((x_center + box_w / 2) * w)
#         y2 = int((y_center + box_h / 2) * h)

#         plt.imshow(image)
#         plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
#                                           edgecolor='lime', facecolor='none', lw=2))
#         plt.text(x1, y1 - 10, f"Class: {class_id}", color='lime', fontsize=12)
#         plt.title("YOLO Label Visualization")
#         plt.axis('off')
#         plt.show()
#         break


if __name__ == "__main__":
    # visualize_yolo_sample(train_ds, num_classes=2)

    # 💡 Label inspection
    for img, label in train_ds.take(1):
        print("Label shape:", label.shape)
        print("Raw values:", label.numpy())

