import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
from frame_loader_yolo import FrameLoader
from yolo_generator import yolo_model
from loss_func import yolo_point_loss

class SanityTrainer:
    def __init__(self):
        self.data_root = os.path.expanduser("~/birdseye_CNN_data")
        self.label_file = "labels.txt"
        self.train_files = sorted(tf.io.gfile.glob(os.path.join(
            self.data_root, "2025_04_04/original_01_rect", "*.png")))[:20]
        self.val_files = sorted(tf.io.gfile.glob(os.path.join(
            self.data_root, "2025_04_16/pieranch_rect", "*.png")))[:5]

        self.image_size = (416, 416)
        self.batch_size = 4
        self.epochs = 20
        self.lr = 1e-4

        self.frame_loader = FrameLoader(image_size=self.image_size, num_classes=1)
        self.train_ds = self.frame_loader.build_dataset(
            file_list=self.train_files,
            label_file=self.label_file,
            batch_size=self.batch_size,
            buffer_size=16,
            repeat=True,
            augment=False
        )

        self.val_ds = self.frame_loader.build_dataset(
            file_list=self.val_files,
            label_file=self.label_file,
            batch_size=self.batch_size,
            buffer_size=16,
            repeat=True,
            augment=False
        )

    def build_model(self):
        input_shape = self.image_size + (3,)
        self.model = yolo_model(input_shape=input_shape, num_classes=1)
        loss_fn = yolo_point_loss(obj_weight=1.0, coord_weight=1.0)

        metrics = [
            tf.keras.metrics.BinaryCrossentropy(name="obj_bce"),
            tf.keras.metrics.BinaryAccuracy(threshold=0.2, name="obj_acc"),
            tf.keras.metrics.AUC(name="obj_auc", curve="ROC"),
            tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
        ]

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
            loss=loss_fn,
            metrics=metrics
        )

        self.model.summary()

    def train(self):
        print("\nStarting sanity check training on 20 images with positive samples only sn\n")
        self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.epochs,
            steps_per_epoch=len(self.train_files) // self.batch_size,
            validation_steps=len(self.val_files) // self.batch_size,
            verbose=1
        )

if __name__ == "__main__":
    trainer = SanityTrainer()
    trainer.build_model()
    trainer.train()
