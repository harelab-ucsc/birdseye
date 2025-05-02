import os
import glob2

def get_config(mode="yolo"):  # default to "yolo" to match the example
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

    val_dates = ['2025_04_16']
    val_dirlists = [['pieranch_rect']]

    finetune_source = os.path.join(os.path.expanduser('~'), 'birdseye', 'models', 'birdseye_960_600_021.weights.h5')

    IMG_HEIGHT = 416 if mode == "yolo" else 1200
    IMG_WIDTH = 416 if mode == "yolo" else 1920
    lr = 1e-4 if mode == "yolo" else 1e-5

    models_dir = os.path.join(os.path.expanduser('~'), 'birdseye', 'models')
    filename_prefix = os.path.join(models_dir, f'birdseye_{IMG_WIDTH}_{IMG_HEIGHT}')
    val = str(len(glob2.glob(os.path.join(models_dir, filename_prefix + '*.weights.h5'))) + 1).rjust(3, '0')
    filename = filename_prefix + f'_{val}_{mode}'

    return {
        "mode": mode,
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
        "batch_size": 4,
        "buffer_size": 16,
        "unfreeze_frac": 0.3,
        "finetune": False,
        "finetune_source": finetune_source,
        "lr": lr,
        "epochs": 50,
        "alpha": 0.3,
        "gamma": 2.0,
        "thresh": 0.5,
        "logits": False,
        "monitor": "val_loss",
        "num_classes": 2,
        "filename": filename
    }
