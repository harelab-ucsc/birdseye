import os
import glob2
import tensorflow as tf
from frame_loader_yolo import FrameLoader
from tile_loader_yolo import TileLoader, get_tile_level_class_weights
from yolo_generator import yolo_model  # new: custom YOLO model for TensorFlow

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

        # self.tile_loader = TileLoader(
        #     tile_size=config.get("tile_size", (224, 224)),
        #     edge_buffer=config.get("edge_buffer", 81),
        #     use_heatmaps=config.get("use_heatmaps", False),
        #     include_negatives=True,
        #     balance_ratio=config.get("balance_ratio", 1.0)
        # )

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

        if self.config["mode"] == "tile":
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

        loader = self.frame_loader if self.config.get("yolo", False) else (
            self.tile_loader if self.config["tiled"] else self.frame_loader
        )

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

    def build_model(self):
        if self.config.get("yolo", False):
            print("🔧 Building YOLO-style model...")
            self.model = yolo_model(
                input_shape=(self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS),
                num_classes=self.config["num_classes"]
            )
            loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)  # or a custom YOLO loss
        else:
            input_height = self.TILE_HEIGHT if self.config["tiled"] else self.IMG_HEIGHT
            input_width = self.TILE_WIDTH if self.config["tiled"] else self.IMG_WIDTH
            self.model = yolo_model(
                input_height,
                input_width,
                self.IMG_CHANNELS,
                unfreeze_frac=self.config.get("unfreeze_frac", 1.0),
                trainable=self.config.get("finetune", False)
            )
            loss_fn = tf.keras.losses.BinaryFocalCrossentropy(
                alpha=self.config.get("alpha", 0.25),
                gamma=self.config.get("gamma", 2.0),
                from_logits=self.config.get("logits", False)
            )

        self.model.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=self.config["lr"]),
            loss=loss_fn,
            metrics=[
                tf.keras.metrics.BinaryCrossentropy(from_logits=self.config.get("logits", True), name='bce'),
                tf.keras.metrics.BinaryAccuracy(threshold=self.config.get("thresh", 0.5), name='bin_acc'),
                tf.keras.metrics.Precision(name='prec'),
                tf.keras.metrics.Recall(name='rec'),
                tf.keras.metrics.AUC()
            ]
        )

        self.model.summary()

        if self.config.get("finetune"):
            print(f"\nLoading weights from: {self.config['finetune_source']}\n")
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

        print("\n\n🚀 Beginning Training...\n\n")
        self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.config["epochs"],
            callbacks=callbacks,
            class_weight=self.class_weights if not self.config.get("yolo", False) else None,
            verbose=1,
            steps_per_epoch=self.config.get("steps_per_epoch")
        )

    def run(self):
        self.setup_data()
        self.build_model()
        self.train()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    data_root = os.path.join(os.path.expanduser('~'), 'birdseye_CNN_data')

    label_file = 'labels.txt'

    train_dates = [
        '2025_03_25', 
        '2025_04_04',
        '2025_04_09',
    ]

    val_dates = [
        '2025_04_16'
    ]

    train_dirlists = [
        ['haybarn_original_01_01_rect', 'haybarn_eviltwin_01_01_rect'],
        ['original_01_rect', 'original_02_rect', 'eviltwin_01_rect', 'eviltwin_02_rect', 'eviltwin_03_rect'],
        ['original_01_rect', 'original_02_rect', 'eviltwin_01_rect', 'eviltwin_02_rect'],
    ]

    val_dirlists = [
        ['pieranch_rect']
    ]

    finetune_source = '/home/harey/birdseye/models/birdseye_960_600_021.weights.h5' # mobilenetv2

    IMG_HEIGHT = 1200
    IMG_WIDTH = 1920

    models_dir = os.path.join(os.path.expanduser('~'), 'birdseye', 'models')
    filename_prefix = os.path.join(models_dir, f'birdseye_{IMG_WIDTH}_{IMG_HEIGHT}')
    val = str(len(glob2.glob(os.path.join(models_dir, filename_prefix+'*.weights.h5')))+1).rjust(3,'0')
    filename = filename_prefix + f'_{val}'

# Choose training mode: "tile", "frame", or "yolo"
TRAINING_MODE = "yolo"

config = {
    "mode": TRAINING_MODE,  

    # Dataset
    "data_root": data_root,
    "train_dates": train_dates,
    "train_dirlists": train_dirlists,
    "val_dates": val_dates,
    "val_dirlists": val_dirlists,
    "label_file": label_file,

    # Image properties
    "img_height": 416 if TRAINING_MODE == "yolo" else 1200,
    "img_width": 416 if TRAINING_MODE == "yolo" else 1920,
    "img_channels": 3,
    "tile_size": (224, 224),
    "edge_buffer": 81,

    # Model & training
    "batch_size": 9,
    "buffer_size": 36,
    "unfreeze_frac": 0.3,
    "finetune": False,
    "finetune_source": finetune_source,
    "lr": 1e-4 if TRAINING_MODE == "yolo" else 1e-5,
    "epochs": 50,

    # Loss settings
    "alpha": 0.3,
    "gamma": 2.0,
    "thresh": 0.5,
    "logits": False,

    # Monitoring & output
    "monitor": "val_loss",
    "num_classes": 2,  # adjust as needed
    "filename": filename + f"_{TRAINING_MODE}"
}
trainer = BirdsEyeTrainer(config)
trainer.run()