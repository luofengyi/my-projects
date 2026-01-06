"""
可视化 IEMOCAP 6 类 (hap/sad/neu/ang/exc/fru) 训练结果：
- 读取 training_history/iemocap_atv_history.json
- 统计每类 F1 的最高值并打印
- 绘制各类 F1 随 epoch 变化曲线
- 绘制训练 F1 随 epoch 变化曲线
- 绘制训练 loss 随 epoch 变化曲线
- 基于 test_preds / test_golds 绘制混淆矩阵

输出目录默认：iemocap6_outputs （不含 "plots" 字样）
不改动现有代码，只新增本文件。
"""

import argparse
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_history(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_best_class_f1(epochs: List[int], class_f1s: Dict[str, List[float]], class_order: List[str]):
    best_info = []
    for cname in class_order:
        scores = class_f1s.get(cname, [])
        if not scores:
            best_info.append((cname, None, None))
            continue
        best_idx = int(np.argmax(scores))
        best_info.append((cname, epochs[best_idx], scores[best_idx]))
    return best_info


def plot_loss_curve(epochs: List[int], train_losses: List[float], output_path: str):
    if not train_losses:
        print("[warn] train_losses empty; skip loss curve.")
        return
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, "b-", linewidth=2, marker="o", markersize=4, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve (6-class)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[saved] loss curve -> {output_path}")


def plot_train_f1_curve(epochs: List[int], train_f1s: List[float], output_path: str):
    if not train_f1s:
        print("[warn] train_f1s empty; skip train F1 curve.")
        return
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_f1s, "g-", linewidth=2, marker="s", markersize=4, label="Train F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("Training F1 Curve (6-class)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[saved] train F1 curve -> {output_path}")


def plot_class_f1_curves(epochs: List[int], class_f1s: Dict[str, List[float]], class_order: List[str], output_path: str):
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    markers = ["o", "s", "^", "v", "D", "p"]

    plt.figure(figsize=(10, 6))
    for i, cname in enumerate(class_order):
        scores = class_f1s.get(cname, [])
        if not scores:
            continue
        plt.plot(
            epochs,
            scores,
            color=colors[i % len(colors)],
            linewidth=2,
            label=cname,
            marker=markers[i % len(markers)],
            markersize=4,
        )
        best_idx = int(np.argmax(scores))
        plt.plot(
            epochs[best_idx],
            scores[best_idx],
            marker="*",
            color=colors[i % len(colors)],
            markersize=10,
            markeredgecolor="black",
            zorder=5,
        )

    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("IEMOCAP 6-class F1 by Epoch")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[saved] class F1 curves -> {output_path}")


def plot_confusion_matrix(golds: List[int], preds: List[int], labels: List[int], label_names: List[str], output_path: str, normalize: bool = True):
    cm = confusion_matrix(golds, preds, labels=labels)
    if normalize:
        cm_show = cm.astype(float)
        row_sum = cm_show.sum(axis=1, keepdims=True)
        cm_show = np.divide(cm_show, row_sum, out=np.zeros_like(cm_show), where=row_sum != 0)
        fmt = ".2f"
        title_suffix = " (normalized)"
    else:
        cm_show = cm
        fmt = "d"
        title_suffix = ""

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_show,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        cbar=True,
        square=True,
        linewidths=0.5,
        linecolor="white",
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("IEMOCAP 6-class Confusion Matrix" + title_suffix)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[saved] confusion matrix -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize IEMOCAP 6-class results")
    parser.add_argument(
        "--history_file",
        type=str,
        default=os.path.join("training_history", "iemocap_atv_history.json"),
        help="路径到 iemocap 6 类训练历史 JSON（默认 training_history/iemocap_atv_history.json）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="iemocap6_outputs",
        help="输出目录（默认 iemocap6_outputs，不含 plots 字样）",
    )
    parser.add_argument("--no_normalize_cm", action="store_true", help="不对混淆矩阵按行归一化")
    args = parser.parse_args()

    if not os.path.exists(args.history_file):
        print(f"[error] history file not found: {args.history_file}")
        return

    history = load_history(args.history_file)
    epochs = history.get("epochs", [])
    class_f1s = history.get("class_f1s", {})
    train_f1s = history.get("train_f1s", [])
    train_losses = history.get("train_losses", [])
    test_preds = history.get("test_preds", [])
    test_golds = history.get("test_golds", [])

    class_order = ["hap", "sad", "neu", "ang", "exc", "fru"]
    label_id_map = {name: idx for idx, name in enumerate(class_order)}
    labels = list(label_id_map.values())

    os.makedirs(args.output_dir, exist_ok=True)

    # 绘制训练 Loss
    loss_curve_path = os.path.join(args.output_dir, "iemocap6_loss_curve.png")
    plot_loss_curve(epochs, train_losses, loss_curve_path)

    # 绘制训练 F1
    train_f1_curve_path = os.path.join(args.output_dir, "iemocap6_train_f1_curve.png")
    plot_train_f1_curve(epochs, train_f1s, train_f1_curve_path)

    # 统计每类最高 F1
    best_info = summarize_best_class_f1(epochs, class_f1s, class_order)
    print("=== Best F1 per class (epoch, F1) ===")
    for cname, epoch, f1 in best_info:
        if f1 is None:
            print(f"{cname}: no data")
        else:
            print(f"{cname}: epoch {epoch}, F1 {f1:.4f}")

    # 绘制类 F1 曲线
    f1_curve_path = os.path.join(args.output_dir, "iemocap6_class_f1_curves.png")
    plot_class_f1_curves(epochs, class_f1s, class_order, f1_curve_path)

    # 绘制混淆矩阵
    if test_preds and test_golds:
        preds_ids = [label_id_map[p] if isinstance(p, str) else int(p) for p in test_preds]
        golds_ids = [label_id_map[g] if isinstance(g, str) else int(g) for g in test_golds]
        cm_path = os.path.join(args.output_dir, "iemocap6_confusion_matrix.png")
        plot_confusion_matrix(
            golds_ids,
            preds_ids,
            labels,
            class_order,
            cm_path,
            normalize=not args.no_normalize_cm,
        )
    else:
        print("[warn] test_preds/test_golds not found; skip confusion matrix.")

    print(f"[done] outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()

