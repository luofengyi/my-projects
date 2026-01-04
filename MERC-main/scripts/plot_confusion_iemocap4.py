"""
绘制 IEMOCAP (4-way) 情感分类的混淆矩阵。

默认：
- 数据：./JOYFUL/data/iemocap_4/data_iemocap_4.pkl
- 模型：./model_checkpoints/iemocap_4_best_dev_f1_model_atv.pt

运行示例（仓库根目录）：
  python scripts/plot_confusion_iemocap4.py --normalize --save_png ./plots/cm_iemocap4.png --save_csv ./plots/cm_iemocap4.csv
"""

import argparse
import json
import os
import pickle
import sys
import os

# 添加上级目录到Python路径
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from JOYFUL import joyful


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
        help="训练好的模型 checkpoint 路径（若不提供 --history_json，将用该模型重新推理得到预测标签）。",
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
        "--history_json",
        type=str,
        default=None,
        help="可选：训练保存的 history JSON（如 training_history/iemocap_4_atv_history.json）。若包含预测标签，可直接用于绘制，无需推理。",
    )
    parser.add_argument(
        "--pred_key",
        type=str,
        default=None,
        help="history JSON 中预测标签的键名（如 test_preds/preds/pred_labels）。为空则自动尝试常见键。",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="Blues",
        help="seaborn / matplotlib 颜色映射。",
    )
    args = parser.parse_args()

    class_names = ["Happy", "Sad", "Neutral", "Angry"]

    def load_golds_from_data(data_path: str):
        data_obj = load_pkl(data_path)
        test_split = data_obj.get("test", [])
        gold_list = []
        for sample in test_split:
            try:
                gold_list.extend(list(sample.label))
            except Exception:
                continue
        return np.array(gold_list, dtype=np.int64)

    def load_preds_from_history(hist_path: str, pred_key: str | None):
        with open(hist_path, "r", encoding="utf-8") as f:
            hist = json.load(f)
        keys_to_try = [pred_key] if pred_key else []
        keys_to_try += ["test_preds", "preds", "pred_labels", "y_pred", "yhat"]
        for k in keys_to_try:
            if k and k in hist:
                preds_arr = np.array(hist[k], dtype=np.int64)
                golds_arr = None
                if "test_golds" in hist:
                    golds_arr = np.array(hist["test_golds"], dtype=np.int64)
                return preds_arr, golds_arr
        return None, None

    golds = None
    preds = None

    # 方案 A：使用 history JSON 中的预测标签
    if args.history_json:
        preds, golds_hist = load_preds_from_history(args.history_json, args.pred_key)
        if preds is None:
            print(f"warning: {args.history_json} 未找到预测标签字段，改用模型推理获得预测。")
            golds = None  # 重新用推理路径获取 golds，以保持一致
        else:
            golds = golds_hist if golds_hist is not None else load_golds_from_data(args.data)

    # 方案 B：用模型重新推理得到预测标签
    if preds is None or golds is None:
        data = load_pkl(args.data)
        ckpt = torch.load(args.checkpoint, map_location=args.device)
        stored_args = ckpt["args"]
        stored_args.device = args.device
        stored_args.batch_size = args.batch_size
        model = ckpt["modelN_state_dict"]
        modelF = ckpt["modelF_state_dict"]
        testset = joyful.Dataset(data["test"], modelF, False, stored_args)
        model.eval()
        modelF.eval()

        golds_list = []
        preds_list = []
        with torch.no_grad():
            for idx in tqdm(range(len(testset)), desc="test"):
                batch = testset[idx]
                golds_list.append(batch["label_tensor"])
                for k, v in batch.items():
                    if k != "utterance_texts":
                        batch[k] = v.to(stored_args.device)
                y_hat = model(batch, False)
                preds_list.append(y_hat.detach().to("cpu"))

        golds = torch.cat(golds_list, dim=-1).cpu().numpy()
        preds = torch.cat(preds_list, dim=-1).cpu().numpy()

    if len(golds) != len(preds):
        raise ValueError(f"标签数量不一致: golds={len(golds)}, preds={len(preds)}")

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

