import pickle
import argparse
import torch
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import joyful
import os

log = joyful.utils.get_logger()


def load_pkl(file):
    with open(file, "rb") as f:
        return pickle.load(f)


def main(args):
    data = load_pkl(f"./data/iemocap_4/data_iemocap_4.pkl")
    #data = load_pkl(f"./data/iemocap/data_iemocap.pkl")
    model_dict = torch.load('./model_checkpoints/iemocap_4_best_dev_f1_model_atv.pt')
    #model_dict = torch.load('model_checkpoints/iemocap_best_dev_f1_model_atv.pt')
    stored_args = model_dict["args"]
    model = model_dict["modelN_state_dict"]
    modelF = model_dict["modelF_state_dict"]
    testset = joyful.Dataset(data["test"], modelF, False, stored_args)
    test = True
    with torch.no_grad():
        golds = []
        preds = []
        for idx in tqdm(range(len(testset)), desc="test" if test else "dev"):
            data = testset[idx]
            golds.append(data["label_tensor"])
            for k, v in data.items():
                if not k == "utterance_texts":
                    data[k] = v.to(stored_args.device)
            y_hat = model(data, False)
            preds.append(y_hat.detach().to("cpu"))

        if stored_args.dataset == "mosei" and stored_args.emotion == "multilabel":
            golds = torch.cat(golds, dim=0).numpy()
            preds = torch.cat(preds, dim=0).numpy()
            f1 = metrics.f1_score(golds, preds, average="weighted")
            acc = metrics.accuracy_score(golds, preds)
        else:
            golds = torch.cat(golds, dim=-1).numpy()
            preds = torch.cat(preds, dim=-1).numpy()
            f1 = metrics.f1_score(golds, preds, average="weighted")

        if test:
            print(metrics.classification_report(golds, preds, digits=4))

            # ========== Confusion Matrix（仅适用于单标签多分类）==========
            # iemocap_4: 4类；preds/golds 为整型标签时可直接计算
            if args.confusion_matrix:
                cm = confusion_matrix(golds, preds)
                print("\nConfusion Matrix (rows=true, cols=pred):")
                print(cm)

                if args.cm_out is not None and args.cm_out.strip():
                    out_path = args.cm_out
                    out_dir = os.path.dirname(out_path)
                    if out_dir:
                        os.makedirs(out_dir, exist_ok=True)

                    # 保存 CSV
                    csv_path = out_path if out_path.lower().endswith(".csv") else out_path + ".csv"
                    with open(csv_path, "w", encoding="utf-8") as f:
                        f.write("," + ",".join([str(i) for i in range(cm.shape[1])]) + "\n")
                        for i in range(cm.shape[0]):
                            f.write(str(i) + "," + ",".join([str(int(x)) for x in cm[i]]) + "\n")
                    print(f"Confusion Matrix saved to: {csv_path}")

                    # 可选保存 PNG（若 matplotlib 可用）
                    if args.cm_plot:
                        try:
                            import matplotlib.pyplot as plt
                            import numpy as np

                            plt.figure(figsize=(6, 5))
                            cm_show = cm.astype(float)
                            if args.cm_normalize:
                                row_sum = cm_show.sum(axis=1, keepdims=True)
                                cm_show = np.divide(cm_show, row_sum, out=np.zeros_like(cm_show), where=row_sum != 0)

                            plt.imshow(cm_show, interpolation="nearest", cmap=plt.cm.Blues)
                            plt.title("Confusion Matrix" + (" (normalized)" if args.cm_normalize else ""))
                            plt.colorbar()
                            plt.xlabel("Predicted")
                            plt.ylabel("True")

                            # 标注数值
                            for i in range(cm.shape[0]):
                                for j in range(cm.shape[1]):
                                    val = cm_show[i, j]
                                    text = f"{val:.2f}" if args.cm_normalize else str(int(cm[i, j]))
                                    plt.text(j, i, text, ha="center", va="center",
                                             color="white" if val > (cm_show.max() / 2.0 + 1e-12) else "black")

                            plt.tight_layout()
                            png_path = os.path.splitext(csv_path)[0] + ".png"
                            plt.savefig(png_path, dpi=200, bbox_inches="tight")
                            plt.close()
                            print(f"Confusion Matrix plot saved to: {png_path}")
                        except Exception as e:
                            print(f"Warning: failed to plot confusion matrix (matplotlib missing or error): {e}")

            if stored_args.dataset == "mosei" and stored_args.emotion == "multilabel":
                happy = metrics.f1_score(golds[:, 0], preds[:, 0], average="weighted")
                sad = metrics.f1_score(golds[:, 1], preds[:, 1], average="weighted")
                anger = metrics.f1_score(golds[:, 2], preds[:, 2], average="weighted")
                surprise = metrics.f1_score(
                    golds[:, 3], preds[:, 3], average="weighted"
                )
                disgust = metrics.f1_score(golds[:, 4], preds[:, 4], average="weighted")
                fear = metrics.f1_score(golds[:, 5], preds[:, 5], average="weighted")

                f1 = {
                    "happy": happy,
                    "sad": sad,
                    "anger": anger,
                    "surprise": surprise,
                    "disgust": disgust,
                    "fear": fear,
                }

            print(f"F1 Score: {f1}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="eval.py")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        default="iemocap_4",
        choices=["iemocap", "iemocap_4", "mosei", "meld"],
        help="Dataset name.",
    )

    parser.add_argument(
        "--data_dir_path", type=str, help="Dataset directory path", default="./data"
    )

    parser.add_argument("--device", type=str, default="cuda:0", help="Computing device.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")

    # Confusion matrix options
    parser.add_argument("--confusion_matrix", action="store_true", default=False,
                        help="Print confusion matrix on test set (single-label classification only).")
    parser.add_argument("--cm_out", type=str, default=None,
                        help="Output path prefix for confusion matrix (CSV; PNG optional). Example: training_plots/cm_iemocap4_atv")
    parser.add_argument("--cm_plot", action="store_true", default=False,
                        help="If set, also save confusion matrix heatmap as PNG (requires matplotlib).")
    parser.add_argument("--cm_normalize", action="store_true", default=False,
                        help="If set, normalize confusion matrix by row (true label).")
    # Modalities
    """ Modalities effects:
        -> dimentions of input vectors in dataset.py
        -> number of heads in transformer_conv in seqcontext.py"""
    parser.add_argument(
        "--modalities",
        type=str,
        default="atv",
        # required=True,
        choices=["a", "at", "atv", "t", "v", "av"],
        help="Modalities",
    )

    # emotion
    parser.add_argument(
        "--emotion", type=str, default=None, help="emotion class for mosei"
    )

    args = parser.parse_args()
    main(args)