import os
import glob2
import tensorflow as tf
from tile_loader import TileLoader
from tile_loader import get_class_weights as tile_weights
from frame_loader import FrameLoader
from frame_loader import get_class_weights as frame_weights
from generator import generator  # assumes you have a generator() model builder defined
import pickle


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
            image_size=(self.IMG_HEIGHT, self.IMG_WIDTH))


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
            return sorted(files)

        self.train_files = collect_paths(self.config["train_dates"], self.config["train_dirlists"])
        self.val_files = collect_paths(self.config["val_dates"], self.config["val_dirlists"])
        
        print('\n\nComputing class weights... \n\n')
        if self.config["tiled"]:
            self.class_weights = tile_weights(self.train_files, self.tile_loader)
            tmp = tile_weights(self.val_files, self.tile_loader)
        else:
            self.class_weights = frame_weights(self.train_files, self.frame_loader)
            tmp = frame_weights(self.val_files, self.frame_loader)
        print(f'Training weights:\n{self.class_weights}\n')
        print(f'Validation stats:\n{tmp}\n\n')


    def setup_data(self):
        self.build_file_lists()
        loader = self.tile_loader if self.config["tiled"] else self.frame_loader

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
        self.model = generator(
            self.TILE_HEIGHT if self.config["tiled"] else self.IMG_HEIGHT,
            self.TILE_WIDTH if self.config["tiled"] else self.IMG_WIDTH,
            self.IMG_CHANNELS,
            unfreeze_frac=self.config.get("unfreeze_frac", 1.0),
            trainable=self.config.get("finetune", False),
            use_heatmap=self.config.get("use_heatmaps", False)
        )

        if self.config.get("use_heatmaps", False):
            loss_fn = tf.keras.losses.BinaryCrossentropy(self.config.get("logits", False))
            metrics=[
                tf.keras.metrics.BinaryCrossentropy(from_logits=self.config.get("logits", False), name='bce'),
                tf.keras.metrics.MeanSquaredError(name='mse'),
                tf.keras.metrics.MeanAbsoluteError(name='mae'),
            ]
        else:
            loss_fn = tf.keras.losses.BinaryFocalCrossentropy(
                alpha=self.config.get("alpha", 0.25),
                gamma=self.config.get("gamma", 2.0),
                from_logits=self.config.get("logits", False)
            )
            metrics=[
                tf.keras.metrics.BinaryCrossentropy(from_logits=self.config.get("logits", True), name='bce'),
                tf.keras.metrics.BinaryAccuracy(threshold=self.config.get("thresh", 0.5), name='bin_acc'),
                tf.keras.metrics.F1Score(threshold=self.config.get("thresh", 0.5), name='f1'),
                tf.keras.metrics.Precision(name='prec'),
                tf.keras.metrics.Recall(name='rec'),
                tf.keras.metrics.AUC()
            ]
            
        self.model.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=self.config["lr"]),
                loss=loss_fn,
            metrics=metrics
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

        tmp = self.config["filename"]
        print(f"\n\nBeginning training of model {tmp}...\n\n")
        history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.config["epochs"],
            callbacks=callbacks,
            class_weight=self.class_weights,
            verbose=1,
            steps_per_epoch=self.config.get("steps_per_epoch")
        )
        with open(self.config["filename"]+'_history.pkl', 'wb') as f:
            pickle.dump(history, f)


    def show_heatmap_prediction(model, dataset, num_samples=4):
        for batch_images, batch_labels in dataset.take(1):
            preds = model.predict(batch_images)
            for i in range(min(num_samples, len(batch_images))):
                image = batch_images[i].numpy().astype(np.uint8)
                true_heatmap = batch_labels[i].numpy().squeeze()
                pred_heatmap = preds[i].squeeze()

                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                axs[0].imshow(image)
                axs[0].set_title("Tile")
                axs[0].axis('off')

                axs[1].imshow(true_heatmap, cmap='hot')
                axs[1].set_title("Ground Truth Heatmap")
                axs[1].axis('off')

                axs[2].imshow(pred_heatmap, cmap='hot')
                axs[2].set_title("Predicted Heatmap")
                axs[2].axis('off')

                plt.tight_layout()
                plt.show()


    def run(self):
        self.setup_data()
        self.build_model()
        self.train()
        # if self.config.get("use_heatmaps", False):
        #     self.show_heatmap_prediction(self.model, self.val_ds)



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    data_root = os.path.join(os.path.expanduser('~'), 'birdseye_CNN_data')

    label_file = 'labels.txt'

    train_dates = [
        '2025_03_25', 
        '2025_04_04',
        '2025_04_09',
        '2025_04_23',
        '2025_05_02'
    ]

    val_dates = [
        '2025_04_16',
        '2025_04_21',
        '2025_04_23',
    ]

    train_dirlists = [
        ['haybarn_original_01_01_rect', 'haybarn_eviltwin_01_01_rect'],
        ['original_01_rect', 'original_02_rect', 'eviltwin_01_rect', 'eviltwin_02_rect', 'eviltwin_03_rect'],
        ['original_01_rect', 'original_02_rect', 'eviltwin_01_rect', 'eviltwin_02_rect'],
        ['rosemary_rect'],  # first jacobs farm sample
        ['rosemary_02_rect', 'jacobs_01_rect']  # different jacobs farm rosemary block, roadside holing
    ]

    val_dirlists = [
        ['pieranch_rect'],  # first pie ranch sample
        ['original_02_rect', 'original_03_rect'],
        ['casfs_original', 'casfs_eviltwin'],
    ]

    finetune = False
    finetune_source = '/home/harey/birdseye/models/birdseye_224_224_007.weights.h5' 

    tiled = True

    if tiled:
        IMG_HEIGHT = 224
        IMG_WIDTH = 224
        BATCH_SIZE = 64
        BUFFER_SIZE = 128
    else:
        IMG_HEIGHT = 600
        IMG_WIDTH = 960
        BATCH_SIZE = 9
        BUFFER_SIZE = 36

    models_dir = os.path.join(os.path.expanduser('~'), 'birdseye', 'models')
    filename_prefix = os.path.join(models_dir, f'birdseye_{IMG_WIDTH}_{IMG_HEIGHT}')
    val = str(len(glob2.glob(os.path.join(models_dir, filename_prefix+'*.weights.h5')))+1).rjust(3,'0')
    filename = filename_prefix + f'_{val}'

    # Example usage
    config = {
        "tiled": tiled,
        "data_root": data_root,
        "train_dates": train_dates,
        "train_dirlists": train_dirlists,
        "val_dates": val_dates,
        "val_dirlists": val_dirlists,
        "label_file": label_file,
        "img_height": IMG_HEIGHT,
        "img_width": IMG_WIDTH,
        "img_channels": 3,
        "tile_size": (224, 224),
        "edge_buffer": 81,
        "batch_size": BATCH_SIZE,
        "buffer_size": BUFFER_SIZE,
        "balance_ratio": 0.2,
        "unfreeze_frac": 0.3,
        "use_heatmaps": True,
        "finetune": finetune,
        "finetune_source": finetune_source,
        "lr": 1e-5,
        "epochs": 500,
        "alpha": 0.3,
        "gamma": 2.0,
        "thresh": 0.5,
        "logits": False,
        "monitor": "val_loss",
        "filename": filename
    }

    trainer = BirdsEyeTrainer(config)
    trainer.run()
