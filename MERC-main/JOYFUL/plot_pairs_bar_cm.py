"""
读取 best_train_pairs.csv（或任意 true/pred 两列的配对表），输出：
1) 真实值 vs 预测值的样本数对比柱状图
2) 混淆矩阵热力图
整体布局类似用户示例图，默认情感映射：{"hap":0,"sad":1,"neu":2,"ang":3}

使用示例（仓库根目录）：
  python JOYFUL/plot_pairs_bar_cm.py \
    --pairs_csv ./plots/best_train_pairs.csv \
    --out_png ./plots/best_train_pairs_vis.png \
    --normalize_cm
"""

import argparse
import os
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


def load_pairs(pairs_csv: str) -> pd.DataFrame:
    df = pd.read_csv(pairs_csv)
    if df.shape[1] < 2:
        raise ValueError("CSV 至少需要两列，列名为 true,pred 或前两列即为真实/预测。")
    # 兼容列名
    if "true" in df.columns and "pred" in df.columns:
        return df[["true", "pred"]]
    else:
        cols = df.columns[:2]
        df = df[list(cols)]
        df.columns = ["true", "pred"]
        return df


def plot_bar_and_cm(df: pd.DataFrame,
                    id2label: Dict[int, str],
                    normalize_cm: bool,
                    out_png: str = None):
    labels = [id2label[i] for i in sorted(id2label.keys())]
    true_counts = df["true"].map(id2label).value_counts().reindex(labels, fill_value=0)
    pred_counts = df["pred"].map(id2label).value_counts().reindex(labels, fill_value=0)

    # 混淆矩阵
    cm = confusion_matrix(df["true"], df["pred"], labels=sorted(id2label.keys()))
    cm_show = cm.astype(float)
    if normalize_cm:
        row_sum = cm_show.sum(axis=1, keepdims=True)
        cm_show = np.divide(cm_show, row_sum, out=np.zeros_like(cm_show), where=row_sum != 0)

    fig = plt.figure(figsize=(14, 6))

    # 子图1：柱状图
    ax1 = plt.subplot(1, 2, 1)
    x = np.arange(len(labels))
    width = 0.35
    ax1.bar(x - width/2, true_counts.values, width, label="真实值")
    ax1.bar(x + width/2, pred_counts.values, width, label="预测值")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("样本数量")
    ax1.set_title("真实值与预测值分布对比")
    ax1.legend()

    # 子图2：混淆矩阵
    ax2 = plt.subplot(1, 2, 2)
    sns.heatmap(
        cm_show,
        annot=True,
        fmt=".2f" if normalize_cm else "d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        square=True,
        linewidths=0.5,
        linecolor="white",
        ax=ax2,
    )
    ax2.set_xlabel("预测标签")
    ax2.set_ylabel("真实标签")
    ax2.set_title("混淆矩阵")

    plt.tight_layout()
    if out_png:
        out_dir = os.path.dirname(out_png)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        print(f"Saved plot to: {out_png}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="基于 true/pred 配对表绘制柱状分布与混淆矩阵")
    parser.add_argument("--pairs_csv", type=str, required=True, help="配对表 CSV 路径（含 true,pred 两列）")
    parser.add_argument("--out_png", type=str, default=None, help="可选：输出图片路径")
    parser.add_argument("--normalize_cm", action="store_true", help="混淆矩阵按行归一化显示")
    args = parser.parse_args()

    # 固定映射：{"hap":0,"sad":1,"neu":2,"ang":3}
    id2label = {0: "hap", 1: "sad", 2: "neu", 3: "ang"}

    df = load_pairs(args.pairs_csv)
    plot_bar_and_cm(df, id2label, args.normalize_cm, args.out_png)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[Error] {e}")
        sys.exit(1)

