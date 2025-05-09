import os
import json
import glob
import pickle
import argparse
import matplotlib.pyplot as plt
from tabulate import tabulate

def load_model_info(base_path, sort_metric):
    config_path = base_path + "_config.json"
    metrics_path = base_path + "_metrics.json"
    weights_path = base_path + ".weights.h5"

    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    except Exception as e:
        print(f"⚠️ Skipping {base_path}: {e}")
        return None

    if not config.get("use_heatmaps", False):
        return None  # Only include heatmap models

    return {
        "Model": os.path.basename(weights_path),
        sort_metric: metrics.get(sort_metric, float("inf")),
        "BCE": metrics.get("bce"),
        "MSE": metrics.get("mse"),
        "MAE": metrics.get("mae"),
        "Path": base_path
    }

def plot_history(base_path):
    pkl_path = base_path + "_history.pkl"
    try:
        with open(pkl_path, "rb") as f:
            history = pickle.load(f)
        if hasattr(history, 'history'):
            history = history.history  # unwrap Keras History object
    except Exception as e:
        print(f"⚠️ Could not load history for {base_path}: {e}")
        return

    plt.figure(figsize=(10, 5))
    for key in history:
        if "val" not in key and "loss" not in key:
            continue
        plt.plot(history[key], label=key)
    for key in history:
        if "val_" in key or key == "val_loss":
            plt.plot(history[key], linestyle='--', label=key)

    plt.title(os.path.basename(base_path))
    plt.xlabel("Epoch")
    plt.ylabel("Loss / Metric")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Compare heatmap model performance")
    parser.add_argument("--top", type=int, default=3, help="How many top models to plot")
    parser.add_argument("--metric", type=str, default="val_loss", help="Metric to sort by (e.g. val_loss, mse, mae)")
    parser.add_argument("--model_dir", type=str, default=os.path.expanduser("~/birdseye/models"), help="Directory with model files")

    args = parser.parse_args()
    top_n = args.top
    sort_metric = args.metric
    model_dir = args.model_dir

    models = []
    for metrics_file in glob.glob(os.path.join(model_dir, "*.metrics.json")):
        base_path = metrics_file.replace("_metrics.json", "")
        info = load_model_info(base_path, sort_metric)
        if info:
            models.append(info)

    if not models:
        print("❌ No valid heatmap models with metrics found.")
        return

    models.sort(key=lambda x: x[sort_metric])
    print(f"\n📊 Sorted by: {sort_metric}")
    print(tabulate(models, headers=["Model", sort_metric, "BCE", "MSE", "MAE"], floatfmt=".5f"))

    print(f"\n📈 Plotting training curves for top {top_n} models:")
    for i, model in enumerate(models[:top_n]):
        print(f"  ↳ Model {i}: {model['Model']}")
        plot_history(model["Path"])

if __name__ == "__main__":
    main()
