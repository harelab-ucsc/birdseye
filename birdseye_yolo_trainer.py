import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import glob2
import tensorflow as tf
from frame_loader_yolo import FrameLoader
from tile_loader_yolo import TileLoader, get_tile_level_class_weights
from yolo_generator import yolo_model

def yolo_point_loss(y_true, y_pred):
    """
    Minimal YOLO-style loss: only supervises xy and objectness.
    Assumes y_pred[..., 4] is a raw logit (no sigmoid in model).
    """
    # Extract ground truth and predictions
    xy_true = tf.cast(y_true[..., 0:2], tf.float32)
    obj_true = tf.cast(y_true[..., 4], tf.float32)

    xy_pred = tf.cast(y_pred[..., 0:2], tf.float32)
    obj_logit = tf.cast(y_pred[..., 4], tf.float32)

    obj_mask = obj_true
    num_pos = tf.reduce_sum(obj_mask) + 1e-6

    # 1. XY loss (mean squared error for positives only)
    xy_loss = tf.reduce_sum(obj_mask[..., tf.newaxis] * tf.square(xy_true - xy_pred)) / num_pos

    # 2. Objectness loss (sigmoid BCE from logits)
    obj_loss_raw = tf.nn.sigmoid_cross_entropy_with_logits(labels=obj_true, logits=obj_logit)
    obj_loss = tf.reduce_sum(obj_loss_raw) / tf.cast(tf.size(obj_loss_raw), tf.float32)

    # Combine
    total_loss = 5.0 * xy_loss + 1.0 * obj_loss

    tf.debugging.assert_all_finite(total_loss, "Loss became NaN or Inf!")
    return total_loss


class BirdsEyeTrainer:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.train_ds = None
        self.val_ds = None
        self.class_weights = None

        self.IMG_HEIGHT = config.get("img_height", 416)
        self.IMG_WIDTH = config.get("img_width", 416)
        self.IMG_CHANNELS = config.get("img_channels", 3)
        self.TILE_HEIGHT, self.TILE_WIDTH = config.get("tile_size", (224, 224))

        self.tile_loader = TileLoader(
            tile_size=config.get("tile_size", (224, 224)),
            edge_buffer=config.get("edge_buffer", 81),
            use_heatmaps=config.get("use_heatmaps", False),
            include_negatives=True,
            balance_ratio=config.get("balance_ratio", 1.0)
        )

        self.frame_loader = FrameLoader(
            image_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
            num_classes=config.get("num_classes", 2)
        )

    def build_file_lists(self):
        def collect_paths(dates, dirlists):
            files = []
            paths = []
            for d, dirs in zip(dates, dirlists):
                for subdir in dirs:
                    full_path = os.path.join(self.config["data_root"], d, subdir)
                    if os.path.isdir(full_path):
                        paths.append(full_path)
                        files.extend(tf.io.gfile.glob(os.path.join(full_path, "*.png")))
            return sorted(files), paths

        self.train_files, train_paths = collect_paths(self.config["train_dates"], self.config["train_dirlists"])
        self.val_files, val_paths = collect_paths(self.config["val_dates"], self.config["val_dirlists"])

        if self.config.get("mode") == "tile":
            print("\n\nComputing class weights... \n\nTraining weights:")
            self.class_weights, _, _ = get_tile_level_class_weights(
                train_paths, 
                self.config["label_file"], 
                edge_buffer=self.config.get("edge_buffer", 81)
            )
            print("\n\nValidation stats:")
            _ = get_tile_level_class_weights(val_paths, 
                self.config["label_file"],
                edge_buffer=self.config.get("edge_buffer", 81)
            )
            print("\n\n")

    def setup_data(self):
        self.build_file_lists()

        if self.config.get("mode") == "tile":
            loader = self.tile_loader
        else:
            loader = self.frame_loader

        self.train_ds = loader.build_dataset(
            file_list=self.train_files,
            label_file=self.config["label_file"],
            batch_size=self.config["batch_size"],
            buffer_size=self.config["buffer_size"],
            repeat=False,
            augment=True
        )

        self.val_ds = loader.build_dataset(
            file_list=self.val_files,
            label_file=self.config["label_file"],
            batch_size=self.config["batch_size"],
            buffer_size=self.config["buffer_size"],
            repeat=False,
            augment=False
        )
        self.config["steps_per_epoch"] = len(self.train_files) // self.config["batch_size"]

    def build_model(self):
        mode = self.config.get("mode")

        if mode == "tile":
            input_shape = (self.TILE_HEIGHT, self.TILE_WIDTH, self.IMG_CHANNELS)
        else:
            input_shape = (self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS)

        print(f" Building model for mode: {mode.upper()}...")

        self.model = yolo_model(
            input_shape=input_shape,
            num_classes=self.config["num_classes"]
        )

        if mode == "yolo":
            loss_fn = yolo_point_loss
            metrics = [
                tf.keras.metrics.BinaryAccuracy(threshold=0.3, name="obj_acc"),
                tf.keras.metrics.AUC(name="obj_auc")
            ]
        else:
            loss_fn = tf.keras.losses.BinaryFocalCrossentropy(
                alpha=self.config.get("alpha", 0.25),
                gamma=self.config.get("gamma", 2.0),
                from_logits=False
            )
            metrics = [
                tf.keras.metrics.BinaryCrossentropy(from_logits=False, name="bce"),
                tf.keras.metrics.BinaryAccuracy(threshold=0.5, name="bin_acc"),
                tf.keras.metrics.Precision(thresholds=0.5, name="prec"),
                tf.keras.metrics.Recall(thresholds=0.5, name="rec"),
                tf.keras.metrics.AUC(name="roc_auc"),
                tf.keras.metrics.AUC(curve="PR", name="pr_auc")
            ]

        self.model.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=self.config["lr"]),
            loss=loss_fn,
            metrics=metrics
        )

        self.model.summary()

        if self.config.get("finetune"):
            print(f"\n Loading weights from: {self.config['finetune_source']}\n")
            self.model.load_weights(self.config["finetune_source"])

    def train(self):
        callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir="logs", histogram_freq=1, write_graph=True, write_images=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor=self.config["monitor"], factor=0.5, patience=3, min_lr=0),
            tf.keras.callbacks.EarlyStopping(monitor=self.config["monitor"], patience=50),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=self.config["filename"] + ".weights.h5",
                save_weights_only=True,
                save_best_only=True,
                monitor=self.config["monitor"],
                verbose=2
            )
        ]

        print("\n\n Beginning Training...\n\n")
        self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.config["epochs"],
            callbacks=callbacks,
            class_weight=self.class_weights if self.config.get("mode") == "tile" else None,
            verbose=1,
            steps_per_epoch=self.config.get("steps_per_epoch")
        )

    def run(self):
        self.setup_data()
        self.build_model()
        self.train()

if __name__ == '__main__':
    from config_birdseye import get_config
    config = get_config(mode="yolo")
    trainer = BirdsEyeTrainer(config)
    trainer.run()


# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# import glob2
# import tensorflow as tf
# from frame_loader_yolo import FrameLoader
# from tile_loader_yolo import TileLoader, get_tile_level_class_weights
# from yolo_generator import yolo_model

# class BirdsEyeTrainer:
#     def __init__(self, config):
#         self.config = config
#         self.model = None
#         self.train_ds = None
#         self.val_ds = None
#         self.class_weights = None

#         self.IMG_HEIGHT = config.get("img_height", 1200)
#         self.IMG_WIDTH = config.get("img_width", 1920)
#         self.IMG_CHANNELS = config.get("img_channels", 3)
#         self.TILE_HEIGHT, self.TILE_WIDTH = config.get("tile_size", (224, 224))

#         self.tile_loader = TileLoader(
#             tile_size=config.get("tile_size", (224, 224)),
#             edge_buffer=config.get("edge_buffer", 81),
#             use_heatmaps=config.get("use_heatmaps", False),
#             include_negatives=True,
#             balance_ratio=config.get("balance_ratio", 1.0)
#         )

#         self.frame_loader = FrameLoader(
#             image_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
#             num_classes=config.get("num_classes", 1)
#         )

#     def build_file_lists(self):
#         def collect_paths(dates, dirlists):
#             files = []
#             paths = []
#             for d, dirs in zip(dates, dirlists):
#                 for subdir in dirs:
#                     full_path = os.path.join(self.config["data_root"], d, subdir)
#                     if os.path.isdir(full_path):
#                         paths.append(full_path)
#                         files.extend(tf.io.gfile.glob(os.path.join(full_path, "*.png")))
#             return sorted(files), paths

#         self.train_files, train_paths = collect_paths(self.config["train_dates"], self.config["train_dirlists"])
#         self.val_files, val_paths = collect_paths(self.config["val_dates"], self.config["val_dirlists"])

#         if self.config.get("mode") == "tile":
#             print("\n\nComputing class weights... \n\nTraining weights:")
#             self.class_weights, _, _ = get_tile_level_class_weights(
#                 train_paths, 
#                 self.config["label_file"], 
#                 edge_buffer=self.config.get("edge_buffer", 81)
#             )
#             print("\n\nValidation stats:")
#             _ = get_tile_level_class_weights(val_paths, 
#                 self.config["label_file"],
#                 edge_buffer=self.config.get("edge_buffer", 81)
#             )
#             print("\n\n")

#     def setup_data(self):
#         self.build_file_lists()

#         if self.config.get("mode") == "tile":
#             loader = self.tile_loader
#         else:
#             loader = self.frame_loader

#         self.train_ds = loader.build_dataset(
#             file_list=self.train_files,
#             label_file=self.config["label_file"],
#             batch_size=self.config["batch_size"],
#             buffer_size=self.config["buffer_size"],
#             repeat=False,
#             augment=True
#         )

#         self.val_ds = loader.build_dataset(
#             file_list=self.val_files,
#             label_file=self.config["label_file"],
#             batch_size=self.config["batch_size"],
#             buffer_size=self.config["buffer_size"],
#             repeat=False,
#             augment=False
#         )
#         # Calculate steps_per_epoch after the dataset is built
#         self.config["steps_per_epoch"] = len(self.train_files) // self.config["batch_size"]
    
#     def build_model(self):
#         mode = self.config.get("mode")

#         if mode == "tile":
#             input_shape = (self.TILE_HEIGHT, self.TILE_WIDTH, self.IMG_CHANNELS)
#         else:
#             input_shape = (self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS)

#         print(f" Building model for mode: {mode.upper()}...")

#         self.model = yolo_model(
#             input_shape=input_shape,
#             num_classes=self.config["num_classes"]
#         )

#         # Choose loss and metrics based on mode
#         if mode == "yolo":
#             loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
#             metrics = [
#                 tf.keras.metrics.BinaryCrossentropy(from_logits=False, name="bce"),
#                 tf.keras.metrics.BinaryAccuracy(threshold=0.3, name="bin_acc"),
#                 tf.keras.metrics.Precision(thresholds=0.3, name="prec"),
#                 tf.keras.metrics.Recall(thresholds=0.3, name="rec"),
#                 tf.keras.metrics.AUC(name="roc_auc"),
#                 tf.keras.metrics.AUC(curve="PR", name="pr_auc")
#             ]
#         else:
#             loss_fn = tf.keras.losses.BinaryFocalCrossentropy(
#                 alpha=self.config.get("alpha", 0.25),
#                 gamma=self.config.get("gamma", 2.0),
#                 from_logits=False
#             )
#             metrics = [
#                 tf.keras.metrics.BinaryCrossentropy(from_logits=False, name="bce"),
#                 tf.keras.metrics.BinaryAccuracy(threshold=0.5, name="bin_acc"),
#                 tf.keras.metrics.Precision(thresholds=0.5, name="prec"),
#                 tf.keras.metrics.Recall(thresholds=0.5, name="rec"),
#                 tf.keras.metrics.AUC(name="roc_auc"),
#                 tf.keras.metrics.AUC(curve="PR", name="pr_auc")
#             ]

#         self.model.compile(
#             optimizer=tf.keras.optimizers.AdamW(learning_rate=self.config["lr"]),
#             loss=loss_fn,
#             metrics=metrics
#         )

#         self.model.summary()

#         if self.config.get("finetune"):
#             print(f"\n Loading weights from: {self.config['finetune_source']}\n")
#             self.model.load_weights(self.config["finetune_source"])


#     def train(self):
#         callbacks = [
#             tf.keras.callbacks.TensorBoard(log_dir="logs", histogram_freq=1, write_graph=True, write_images=True),
#             tf.keras.callbacks.ReduceLROnPlateau(monitor=self.config["monitor"], factor=0.5, patience=3, min_lr=0),
#             tf.keras.callbacks.EarlyStopping(monitor=self.config["monitor"], patience=15),
#             tf.keras.callbacks.ModelCheckpoint(
#                 filepath=self.config["filename"] + ".weights.h5",
#                 save_weights_only=True,
#                 save_best_only=True,
#                 monitor=self.config["monitor"],
#                 verbose=2
#             )
#         ]

#         print("\n\n Beginning Training...\n\n")
#         self.model.fit(
#             self.train_ds,
#             validation_data=self.val_ds,
#             epochs=self.config["epochs"],
#             callbacks=callbacks,
#             class_weight=self.class_weights if self.config.get("mode") == "tile" else None,
#             verbose=1,
#             steps_per_epoch=self.config.get("steps_per_epoch")
#         )

#     def run(self):
#         self.setup_data()
#         self.build_model()
#         self.train()

# if __name__ == '__main__':
#     from config_birdseye import get_config
#     config = get_config(mode="tile")
#     trainer = BirdsEyeTrainer(config)
#     trainer.run()
