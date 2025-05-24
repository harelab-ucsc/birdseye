import os
import glob2
from frame_loader_yolo import FrameLoader

def check_label_density(data_root, dates, dirlists, label_file='labels.txt'):
    total_images = 0
    images_with_labels = 0
    label_counts = []

    loader = FrameLoader()

    for d, dirs in zip(dates, dirlists):
        for subdir in dirs:
            folder = os.path.join(data_root, d, subdir)
            image_paths = sorted(glob2.glob(os.path.join(folder, '*.png')))
            for img_path in image_paths:
                labels = loader.load_labels(label_file, img_path)
                total_images += 1
                label_counts.append(len(labels))
                if len(labels) > 0:
                    images_with_labels += 1

    ratio = images_with_labels / total_images
    print(f"\n Total images:            {total_images}")
    print(f" Images with labels:      {images_with_labels}")
    print(f" Label coverage:          {ratio * 100:.2f}%")
    print(f" Avg points/image:        {sum(label_counts) / total_images:.2f}")
    print(f" Avg points/labeled image:{sum(label_counts) / (images_with_labels + 1e-6):.2f}")
    print()

if __name__ == "__main__":
    from config_birdseye import get_config
    config = get_config("yolo")
    check_label_density(
        config["data_root"],
        config["train_dates"],
        config["train_dirlists"],
        label_file=config["label_file"]
    )
