import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import glob2
import tensorflow as tf
from frame_loader_yolo import FrameLoader
from tile_loader_yolo import TileLoader, get_tile_level_class_weights
from yolo_generator import yolo_model

class BirdsEyeTrainer:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.train_ds = None
        self.val_ds = None
        self.class_weights = None

        self.IMG_HEIGHT = config.get("img_height", 1200)
        self.IMG_WIDTH = config.get("img_width", 1920)
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
            num_classes=config.get("num_classes", 1)
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
            repeat=True,
            augment=True
        )

        self.val_ds = loader.build_dataset(
            file_list=self.val_files,
            label_file=self.config["label_file"],
            batch_size=self.config["batch_size"],
            buffer_size=self.config["buffer_size"],
            repeat=True,
            augment=False
        )
        # Calculate steps_per_epoch after the dataset is built
        self.config["steps_per_epoch"] = len(self.train_files) // self.config["batch_size"]

    def build_model(self):
        from loss_func import yolo_point_loss  # make sure this path is correct

        mode = self.config.get("mode")

        if mode == "tile":
            input_shape = (self.TILE_HEIGHT, self.TILE_WIDTH, self.IMG_CHANNELS)
        else:
            input_shape = (self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS)

        print(f" Building model for mode: {mode.upper()}...")

        self.model = yolo_model(
            input_shape=input_shape,
            num_classes=1  # assuming binary: hole or no hole
        )

        # Select loss + metrics
        if mode == "yolo":
            loss_fn = yolo_point_loss(obj_weight=1.0, coord_weight=1.0)
            metrics = [
                tf.keras.metrics.BinaryCrossentropy(name="obj_bce"),
                tf.keras.metrics.BinaryAccuracy(threshold=0.2, name="obj_acc"),
                tf.keras.metrics.AUC(name="obj_auc", curve="ROC"),
                tf.keras.metrics.AUC(name="pr_auc", curve="PR")
            ]
        else:
            loss_fn = tf.keras.losses.BinaryFocalCrossentropy(
                alpha=self.config.get("alpha", 0.25),
                gamma=self.config.get("gamma", 2.0),
                from_logits=False
            )
            metrics = [
                tf.keras.metrics.BinaryCrossentropy(name="bce"),
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
            tf.keras.callbacks.EarlyStopping(monitor=self.config["monitor"], patience=15),
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

    def debug_gradients(self):
        import tensorflow as tf

        # Get one batch
        for x_batch, y_batch in self.train_ds.take(1):
            with tf.GradientTape() as tape:
                y_pred = self.model(x_batch, training=True)
                loss = self.model.compute_loss(x_batch, y_batch, y_pred)

            gradients = tape.gradient(loss, self.model.trainable_variables)

            print(f"\n[Gradient Debug]")
            print(f"Loss: {loss.numpy():.6f}")
            for var, grad in zip(self.model.trainable_variables, gradients):
                if grad is None:
                    print(f" - {var.name:45s}: None")
                else:
                    norm = tf.norm(grad).numpy()
                    print(f" - {var.name:45s}: grad norm = {norm:.6f}")
            break

    def run(self):
        self.setup_data()
        self.build_model()
        self.train()

    
if __name__ == '__main__':
    from config_birdseye import get_config
    config = get_config(mode="yolo")
    trainer = BirdsEyeTrainer(config)
    trainer.run()

    # # Setup data and model (but don't train yet)
    # trainer.setup_data()
    # trainer.build_model()

    # # Inspect gradients for a single batch
    trainer.debug_gradients()

    # # Then continue training
    # trainer.train()

