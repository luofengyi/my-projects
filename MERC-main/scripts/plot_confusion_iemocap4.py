"""
绘制 IEMOCAP (4-way) 情感分类的混淆矩阵。

默认：
- 数据：./JOYFUL/data/iemocap_4/data_iemocap_4.pkl
- 模型：./model_checkpoints/iemocap_4_best_dev_f1_model_atv.pt

运行示例（仓库根目录）：
  python scripts/plot_confusion_iemocap4.py --normalize --save_png ./plots/cm_iemocap4.png --save_csv ./plots/cm_iemocap4.csv
"""

import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

import joyful


def load_pkl(file_path: str):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def plot_confusion_matrix(cm: np.ndarray, class_names, normalize: bool, cmap: str):
    cm_show = cm.astype(float)
    if normalize:
        row_sum = cm_show.sum(axis=1, keepdims=True)
        cm_show = np.divide(cm_show, row_sum, out=np.zeros_like(cm_show), where=row_sum != 0)

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm_show,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        square=True,
        linewidths=0.5,
        linecolor="white",
    )
    plt.xlabel("Predicted label", fontsize=11)
    plt.ylabel("True label", fontsize=11)
    plt.title(
        "IEMOCAP (4-way) Confusion Matrix" + (" (normalized)" if normalize else ""),
        fontsize=12,
    )
    plt.tight_layout()


def main():
    parser = argparse.ArgumentParser(description="Plot IEMOCAP-4 confusion matrix")
    parser.add_argument(
        "--data",
        type=str,
        default="./JOYFUL/data/iemocap_4/data_iemocap_4.pkl",
        help="pkl 数据路径（含 train/dev/test 字典）。",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./model_checkpoints/iemocap_4_best_dev_f1_model_atv.pt",
        help="训练好的模型 checkpoint 路径。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="计算设备，例如 cuda:0 / cpu。",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="评估 batch 大小（会写回到存储的 args 中）。",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="是否对混淆矩阵按行归一化显示比例。",
    )
    parser.add_argument(
        "--save_png",
        type=str,
        default=None,
        help="可选：保存热力图到指定路径（例如 ./plots/cm_iemocap4.png）。",
    )
    parser.add_argument(
        "--save_csv",
        type=str,
        default=None,
        help="可选：保存混淆矩阵原始计数到 CSV。",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="Blues",
        help="seaborn / matplotlib 颜色映射。",
    )
    args = parser.parse_args()

    # 加载数据与模型
    data = load_pkl(args.data)
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    stored_args = ckpt["args"]

    # 覆盖运行时配置
    stored_args.device = args.device
    stored_args.batch_size = args.batch_size

    model = ckpt["modelN_state_dict"]
    modelF = ckpt["modelF_state_dict"]

    # 构建测试集
    testset = joyful.Dataset(data["test"], modelF, False, stored_args)

    model.eval()
    modelF.eval()

    golds = []
    preds = []
    with torch.no_grad():
        for idx in tqdm(range(len(testset)), desc="test"):
            batch = testset[idx]
            golds.append(batch["label_tensor"])
            for k, v in batch.items():
                if k != "utterance_texts":
                    batch[k] = v.to(stored_args.device)
            y_hat = model(batch, False)
            preds.append(y_hat.detach().to("cpu"))

    golds = torch.cat(golds, dim=-1).cpu().numpy()
    preds = torch.cat(preds, dim=-1).cpu().numpy()

    # 生成混淆矩阵（标签顺序：Happy, Sad, Neutral, Angry）
    class_names = ["Happy", "Sad", "Neutral", "Angry"]
    cm = confusion_matrix(golds, preds, labels=range(len(class_names)))

    print("Confusion Matrix (rows=true, cols=pred):")
    print(cm)

    # 可选保存 CSV
    if args.save_csv:
        csv_dir = os.path.dirname(args.save_csv)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
        with open(args.save_csv, "w", encoding="utf-8") as f:
            f.write("," + ",".join(class_names) + "\n")
            for i, row in enumerate(cm):
                f.write(class_names[i] + "," + ",".join(str(int(x)) for x in row) + "\n")
        print(f"Confusion matrix saved to CSV: {args.save_csv}")

    # 绘图
    plot_confusion_matrix(cm, class_names, normalize=args.normalize, cmap=args.cmap)

    if args.save_png:
        png_dir = os.path.dirname(args.save_png)
        if png_dir:
            os.makedirs(png_dir, exist_ok=True)
        plt.savefig(args.save_png, dpi=300, bbox_inches="tight")
        print(f"Confusion matrix plot saved: {args.save_png}")

    plt.show()


if __name__ == "__main__":
    main()

