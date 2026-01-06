"""
Lightweight training visualization (English-only).
Differences vs original visualize_training.py:
- No best-point annotations on any curves.
- Adds a dedicated plot for Train F1 vs. epoch.
- All labels/titles are in English.
"""

import argparse
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_training_history(history_file: str) -> Dict:
    with open(history_file, "r") as f:
        return json.load(f)


def plot_loss_curve(history: Dict, save_path: str):
    epochs = history["epochs"]
    train_losses = history["train_losses"]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, "b-", linewidth=2, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[saved] {save_path}")


def plot_f1_curves(history: Dict, save_path: str):
    epochs = history["epochs"]
    train_f1s = history["train_f1s"]
    dev_f1s = history["dev_f1s"]
    test_f1s = history["test_f1s"]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_f1s, "b-", linewidth=2, label="Train F1")
    plt.plot(epochs, dev_f1s, "g-", linewidth=2, label="Dev F1")
    plt.plot(epochs, test_f1s, "r-", linewidth=2, label="Test F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("F1 Curves (Train/Dev/Test)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[saved] {save_path}")


def plot_train_f1_only(history: Dict, save_path: str):
    epochs = history["epochs"]
    train_f1s = history["train_f1s"]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_f1s, "b-", linewidth=2, label="Train F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("Train F1 over Epochs")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[saved] {save_path}")


def plot_class_f1_curves(history: Dict, save_path: str, dataset: str = "iemocap_4"):
    epochs = history["epochs"]
    class_f1s: Dict[str, List[float]] = history.get("class_f1s", {})
    if not class_f1s:
        print("[warn] No class F1 data found, skip class plot.")
        return

    plt.figure(figsize=(8, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    markers = ["o", "s", "^", "v", "D", "p"]

    if dataset == "iemocap_4":
        class_order = ["hap", "sad", "neu", "ang"]
        class_labels = ["Happy", "Sad", "Neutral", "Angry"]
    elif dataset == "iemocap":
        class_order = ["hap", "sad", "neu", "ang", "exc", "fru"]
        class_labels = ["Happy", "Sad", "Neutral", "Angry", "Excited", "Frustrated"]
    else:
        class_order = list(class_f1s.keys())
        class_labels = [name.capitalize() for name in class_order]

    for i, (cname, clabel) in enumerate(zip(class_order, class_labels)):
        if cname in class_f1s:
            plt.plot(
                epochs,
                class_f1s[cname],
                color=colors[i % len(colors)],
                linewidth=2,
                label=clabel,
                marker=markers[i % len(markers)],
                markersize=4,
            )

    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("Class-wise F1 over Epochs")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[saved] {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Simple training visualization (English)")
    parser.add_argument("--history_file", type=str, default=None, help="Path to training history JSON")
    parser.add_argument("--dataset", type=str, default="iemocap_4", choices=["iemocap", "iemocap_4", "mosei", "meld"])
    parser.add_argument("--modalities", type=str, default="atv")
    parser.add_argument("--label_set", type=str, default=None,
                        choices=["iemocap4", "iemocap6"],
                        help="可选：指定 IEMOCAP 标签集合（4/6 类）；若提供将覆盖 dataset")
    # 4 类 / 6 类输出目录默认分开，避免覆盖
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    if args.label_set:
        dataset_name = "iemocap" if args.label_set == "iemocap6" else "iemocap_4"
    else:
        dataset_name = args.dataset

    if args.history_file is None:
        if dataset_name in ("iemocap", "iemocap_4"):
            base = "iemocap" if dataset_name == "iemocap" else "iemocap_4"
            history_file = os.path.join("training_history", f"{base}_{args.modalities}_history.json")
        else:
            history_file = os.path.join("training_history", f"{dataset_name}_{args.modalities}_history.json")
    else:
        history_file = args.history_file

    if not os.path.exists(history_file):
        print(f"[error] history file not found: {history_file}")
        return

    history = load_training_history(history_file)

    default_output_dir_map = {
        "iemocap": "training_plots_simple_6cls",
        "iemocap_4": "training_plots_simple_4cls",
    }
    output_dir = args.output_dir or default_output_dir_map.get(
        dataset_name, f"training_plots_simple_{dataset_name}"
    )
    os.makedirs(output_dir, exist_ok=True)

    loss_path = os.path.join(output_dir, f"{dataset_name}_{args.modalities}_loss.png")
    f1_path = os.path.join(output_dir, f"{dataset_name}_{args.modalities}_f1.png")
    train_f1_only_path = os.path.join(output_dir, f"{dataset_name}_{args.modalities}_train_f1.png")
    class_f1_path = os.path.join(output_dir, f"{dataset_name}_{args.modalities}_class_f1.png")

    plot_loss_curve(history, loss_path)
    plot_f1_curves(history, f1_path)
    plot_train_f1_only(history, train_f1_only_path)
    plot_class_f1_curves(history, class_f1_path, dataset=dataset_name)

    print("[done] all plots saved to:", output_dir)
    print("dataset:", dataset_name, "| label_set:", args.label_set or "auto-from-dataset")


if __name__ == "__main__":
    main()

