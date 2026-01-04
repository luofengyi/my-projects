"""
从训练历史 JSON 中提取“训练集最佳F1所在轮次”的真实/预测标签，生成逐条配对表格。

假设训练时已在 history 中写入：
- best_train_preds: 训练集最佳F1轮的预测标签（列表）
- best_train_golds: 对应的真实标签（列表）

使用示例（仓库根目录执行）：
  python JOYFUL/generate_best_train_pairs.py \
    --history ./JOYFUL/training_history/iemocap_4_atv_history.json \
    --out_csv ./plots/best_train_pairs.csv
"""

import argparse
import json
import os
import sys
from typing import List


def load_best_train_pairs(history_path: str):
    with open(history_path, "r", encoding="utf-8") as f:
        hist = json.load(f)

    preds = hist.get("best_train_preds")
    golds = hist.get("best_train_golds")
    epoch = hist.get("best_train_epoch")
    f1 = hist.get("best_train_f1")

    if preds is None or golds is None:
        raise ValueError("history 中未找到 best_train_preds / best_train_golds 字段，请确认已使用最新训练脚本。")
    if len(preds) != len(golds):
        raise ValueError(f"pred/gold 数量不一致: preds={len(preds)}, golds={len(golds)}")

    return preds, golds, epoch, f1


def write_pairs_csv(preds: List[int], golds: List[int], out_csv: str):
    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("true,pred\n")
        for t, p in zip(golds, preds):
            f.write(f"{int(t)},{int(p)}\n")


def main():
    parser = argparse.ArgumentParser(description="导出最佳训练F1轮次的真实/预测配对表")
    parser.add_argument(
        "--history",
        type=str,
        required=True,
        help="训练历史 JSON 路径，例如 JOYFUL/training_history/iemocap_4_atv_history.json",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="./plots/best_train_pairs.csv",
        help="输出配对表 CSV 路径（含目录）。默认 ./plots/best_train_pairs.csv",
    )
    args = parser.parse_args()

    preds, golds, epoch, f1 = load_best_train_pairs(args.history)
    write_pairs_csv(preds, golds, args.out_csv)

    print(f"Saved true/pred pairs to: {args.out_csv}")
    if epoch is not None:
        print(f"Best train epoch: {epoch}")
    if f1 is not None:
        print(f"Best train F1: {f1}")


if __name__ == "__main__":
    # 兼容作为模块调用
    try:
        main()
    except Exception as e:
        print(f"[Error] {e}")
        sys.exit(1)

