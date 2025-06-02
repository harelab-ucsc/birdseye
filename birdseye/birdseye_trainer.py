import os
import glob2
import pickle
import json
import tensorflow as tf
import tensorflow.keras.backend as K
from tile_loader import TileLoader
from tile_loader import get_class_weights as tile_weights
from frame_loader import FrameLoader
from frame_loader import get_class_weights as frame_weights
from generator import generator  # assumes you have a generator() model builder defined


class SlicedMetric(tf.keras.metrics.Metric):
    def __init__(self, base_metric, name=None):
        super().__init__(name=name or base_metric.name, dtype=base_metric.dtype)
        self.base_metric = base_metric

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_sliced = y_true[..., 0:1]
        return self.base_metric.update_state(y_true_sliced, y_pred, sample_weight)

    def result(self):
        return self.base_metric.result()

    def reset_state(self):
        return self.base_metric.reset_state()


class HeatmapLogger(tf.keras.callbacks.Callback):
    def __init__(self, model, val_ds, log_dir, log_freq=5, num_samples=4):
        super().__init__()
        self.model = model
        self.val_ds = val_ds
        self.writer = tf.summary.create_file_writer(log_dir)
        self.log_freq = log_freq
        self.num_samples = num_samples

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.log_freq != 0:
            return

        val_batch = next(iter(self.val_ds.take(1)))
        images, labels = val_batch
        preds = self.model.predict(images)

        with self.writer.as_default():
            for i in range(min(self.num_samples, images.shape[0])):
                img = tf.cast(images[i], tf.uint8)
                true = tf.squeeze(labels[i][..., 0:1])
                pred = tf.squeeze(preds[i])

                # Normalize for display
                true = tf.clip_by_value(true, 0.0, 1.0)
                pred = tf.clip_by_value(pred, 0.0, 1.0)

                tf.summary.image(f"input/image_{i}", tf.expand_dims(img, 0), step=epoch)
                tf.summary.image(f"label/true_heatmap_{i}", tf.expand_dims(tf.expand_dims(true, -1), 0), step=epoch)
                tf.summary.image(f"prediction/pred_heatmap_{i}", tf.expand_dims(tf.expand_dims(pred, -1), 0), step=epoch)


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
            include_negatives=self.config.get("include_negatives", True),
            balance_ratio=config.get("balance_ratio", 1.0),
            negative_mode=self.config.get("negative_mode", "random"),
        )
        self.frame_loader = FrameLoader(
            image_size=(self.IMG_HEIGHT, self.IMG_WIDTH))
        self.strategy = tf.distribute.MirroredStrategy()


    def build_file_lists(self):
        def collect_paths(dates, dirlists):
            files = []
            paths = []
            for d, dirs in zip(dates, dirlists):
                for subdir in dirs:
                    full_path = os.path.join(self.config["data_root"], d, subdir)
                    print(f'  {full_path}')
                    if os.path.isdir(full_path):
                        paths.append(full_path)
                        files.extend(tf.io.gfile.glob(os.path.join(full_path, "*.png")))
            print(f'    {len(files)} images')
            return sorted(files)

        print('\n\nTraining dataset:')
        self.train_files = collect_paths(self.config["train_dates"], self.config["train_dirlists"])
        print('\nValidation dataset:')
        self.val_files = collect_paths(self.config["val_dates"], self.config["val_dirlists"])
        print('\n\n')
        if self.config["negative_mode"] != 'none':
            if self.config["tiled"]:
                # self.class_weights = tile_weights(
                #     self.train_files,
                #     self.tile_loader,
                #     semisupervised = self.config["semisupervised"],
                #     cache_dir=self.config["train_cache_dir"],
                #     bypass_cache=self.config.get("bypass_cache", False)
                # )
                # tmp = tile_weights(
                #     self.val_files,
                #     self.tile_loader,
                #     cache_dir=self.config["val_cache_dir"],
                #     bypass_cache=self.config.get("bypass_cache", False)
                # )
                # print(f'Validation stats:\n{tmp}\n\n')
                pass
            else:
                print('\n\nComputing class weights... \n\n')
                self.class_weights = frame_weights(self.train_files, self.frame_loader)
                tmp = frame_weights(self.val_files, self.frame_loader)

                print(f'\n\nTraining weights:\n{self.class_weights}\n')
                print(f'Validation stats:\n{tmp}\n\n')
        else:
            print('  not using all-negative tiles... skipping.')


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
        # with self.strategy.scope():
        self.model = generator(
            self.TILE_HEIGHT if self.config["tiled"] else self.IMG_HEIGHT,
            self.TILE_WIDTH if self.config["tiled"] else self.IMG_WIDTH,
            self.IMG_CHANNELS,
            unfreeze_frac=self.config.get("unfreeze_frac", 1.0),
            trainable=self.config.get("finetune", False),
            use_heatmap=self.config.get("use_heatmaps", False)
        )

        if self.config.get("use_heatmaps", False):
            def weighted_heatmap_loss(from_logits=False, debug=False):
                def loss_fn(y_true, y_pred):
                    # Extract channels
                    y = y_true[..., 0:1]  # heatmap
                    w = y_true[..., 1:2]  # weightmap

                    # ✅ Shape sanity checks
                    tf.debugging.assert_equal(tf.shape(y), tf.shape(y_pred), message="Mismatch between y and y_pred shapes")
                    tf.debugging.assert_equal(tf.shape(y), tf.shape(w), message="Mismatch between y and weightmap shapes")

                    # ✅ Check for sparse or empty weightmaps
                    total_weight = tf.reduce_sum(w)
                    tf.debugging.assert_greater(total_weight, 1.0, message="Weightmap too sparse — likely no labels in batch")

                    # Loss computation
                    if from_logits:
                        bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_pred)
                    else:
                        bce = tf.keras.backend.binary_crossentropy(y, y_pred)

                    weighted = bce * w
                    loss = tf.reduce_sum(weighted) / (total_weight + 1e-6)

                    # ✅ Optional debug logging
                    if debug:
                        tf.print("Total weight:", total_weight)
                        tf.print("Mean weighted loss:", loss)
                        tf.print("Batch BCE mean:", tf.reduce_mean(bce))
                        tf.print("Heatmap mean (y):", tf.reduce_mean(y))
                        tf.print("Prediction mean (y_pred):", tf.reduce_mean(y_pred))

                    return loss
                return loss_fn


            def safe_heatmap_loss(from_logits=False, debug=False):
                def loss_fn(y_true, y_pred):
                    y = y_true[..., 0:1]
                    bce = tf.keras.backend.binary_crossentropy(y, y_pred, from_logits=from_logits)  # [B, H, W, 1]

                    if debug:
                        tf.debugging.assert_equal(tf.shape(y_true)[-1], 1, message="y_true has >1 channel but only one is used")
                        tf.print("Heatmap mean:", tf.reduce_mean(y))
                        tf.debugging.assert_all_finite(tf.reduce_sum(y), message="Heatmap contains NaNs/Infs")

                    return tf.reduce_mean(bce)  
                return loss_fn

            if self.config.get("safe", True):
                loss_fn = safe_heatmap_loss(from_logits=self.config.get("logits", False), debug=False)      
                print('\n\n  Using safe_heatmap_loss (unweighted BCE)')     
            else:
                loss_fn = weighted_heatmap_loss(from_logits=self.config.get("logits", False), debug=False)                     
                print('\n\n  Using weighted_heatmap_loss (weighted BCE)')     
            metrics=[
                SlicedMetric(tf.keras.metrics.BinaryCrossentropy(from_logits=self.config.get("logits", False), name='bce')),
                SlicedMetric(tf.keras.metrics.MeanSquaredError(name='mse')),
                SlicedMetric(tf.keras.metrics.MeanAbsoluteError(name='mae')),
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
            tf.keras.callbacks.ModelCheckpoint(filepath=self.config["filename"] + ".weights.h5", save_weights_only=True, save_best_only=True, monitor=self.config["monitor"], verbose=2),
        ]

        if self.config.get("log_images", False):
            callbacks.append(HeatmapLogger(
                self.model,
                self.val_ds,
                log_dir=os.path.join("logs", "images"),
                log_freq=self.config.get("log_images_every", 5)
            ))

        tmp = self.config["filename"]
        print(f"\n\nBeginning training of model {tmp}...\n\n")

        # Save run config
        with open(tmp + "_config.json", "w") as f:
            json.dump(self.config, f, indent=2)

        history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.config["epochs"],
            callbacks=callbacks,
            class_weight=self.class_weights,
            verbose=1,
            # steps_per_epoch=self.config.get("steps_per_epoch")
        )
        metrics = self.model.evaluate(self.val_ds, verbose=1, return_dict=True)

        with open(self.config["filename"] + "_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n📊 Final Validation Metrics:\n{json.dumps(metrics, indent=2)}\n")

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
        if self.config.get("use_heatmaps", False):
            
            self.show_heatmap_prediction(self.model, self.val_ds)



if __name__ == '__main__':
    assert 'subvision' in os.environ.get('CONDA_DEFAULT_ENV', ''), "Not in subvision environment!"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # gpus = tf.config.list_physical_devices('GPU')
    # for gpu in gpus:
    #     tf.config.experimental.set_memory_growth(gpu, True)

    data_root = os.path.join(os.path.expanduser('~'), 'birdseye_CNN_data')

    label_file = 'labels.txt'

    train_dates = [
        # '2025_03_25', 
        '2025_04_04',
        '2025_04_09',
        '2025_04_23',
        '2025_05_02',
        '2025_05_21',
        '2025_05_26',
    ]

    val_dates = [
        # '2025_04_16',
        '2025_04_21',
        '2025_04_23',
        '2025_05_23',
    ]

    train_dirlists = [
        # ['haybarn_original_01_01_rect', 'haybarn_eviltwin_01_01_rect'],
        ['original_01_rect', 'original_02_rect', 'eviltwin_01_rect', 'eviltwin_02_rect', 'eviltwin_03_rect'],
        ['original_01_rect', 'original_02_rect', 'eviltwin_01_rect', 'eviltwin_02_rect'],
        ['rosemary_rect'],  # first jacobs farm sample
        ['rosemary_02_rect', 'jacobs_01_rect'],  # different jacobs farm rosemary block, roadside holing
        ['haybarn_rect'],
        ['main_rect']
    ]

    val_dirlists = [
        # ['pieranch_rect'],  # first pie ranch sample
        ['original_02_rect', 'original_03_rect'],
        ['casfs_original', 'casfs_eviltwin'],
        ['oceanview_rect']
    ]

    safe = True
    semisup = False
    finetune = False
    # finetune_source = '/home/harey/birdseye/models/birdseye_224_224_019.weights.h5' 
    finetune_source = '/home/harey/birdseye/models/birdseye_224_224_013.weights.h5' 

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
        "balance_ratio": 0.05,
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
        "filename": filename,
        "train_cache_dir": os.path.expanduser("~/.cache/birdseye/class_weights"),
        "val_cache_dir": os.path.expanduser("~/.cache/birdseye/val_weights"),
        "bypass_cache": False,
        "negative_mode": "random",  # options: 'random', 'once_per_image', 'none'
        "semisupervised": semisup,
        "safe": safe,
    }
    print('\n\n')
    for key in config.keys():
        print(f'{key}: {config[key]}')
    print('\n\n')
    trainer = BirdsEyeTrainer(config)
    trainer.run()
