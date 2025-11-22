"""
JOYFUL模型可视化脚本
借鉴MMSA的可视化方法，对JOYFUL模型进行特征可视化
"""
import argparse
import torch
import pickle
import joyful
from pathlib import Path

log = joyful.utils.get_logger()


def load_pkl(file):
    with open(file, "rb") as f:
        return pickle.load(f)


def main(args):
    """主函数：加载模型和数据，执行可视化"""
    
    # 加载数据
    if args.emotion:
        data_path = Path(args.data_dir_path) / args.dataset / f"data_{args.dataset}_{args.emotion}.pkl"
    else:
        if args.transformers:
            data_path = Path(args.data_dir_path) / args.dataset / "transformers" / f"data_{args.dataset}.pkl"
        else:
            data_path = Path(args.data_dir_path) / args.dataset / f"data_{args.dataset}.pkl"
    
    log.info(f"加载数据: {data_path}")
    data = load_pkl(str(data_path))
    
    # 加载模型
    if args.checkpoint_path:
        checkpoint_path = args.checkpoint_path
    else:
        # 使用默认路径
        if args.dataset == "mosei" and args.emotion:
            checkpoint_path = f"model_checkpoints/mosei_best_dev_f1_model_{args.modalities}_{args.emotion}.pt"
        else:
            checkpoint_path = f"model_checkpoints/{args.dataset}_best_dev_f1_model_{args.modalities}.pt"
    
    log.info(f"加载模型: {checkpoint_path}")
    model_dict = torch.load(checkpoint_path, map_location=args.device)
    stored_args = model_dict.get("args")
    if stored_args is None:
        stored_args = args
    else:
        # 更新一些关键参数
        stored_args.device = args.device
        stored_args.modalities = args.modalities
    
    model = model_dict.get("modelN_state_dict") or model_dict.get("state_dict")
    modelF = model_dict.get("modelF_state_dict")
    
    # 创建数据集
    dataset_mode = args.mode if hasattr(args, 'mode') else 'test'
    dataset = joyful.Dataset(data[dataset_mode], modelF, False, stored_args)
    
    # 创建可视化器
    from joyful.visualization import JOYFULVisualizer
    visualizer = JOYFULVisualizer(stored_args, save_dir=args.save_dir)
    
    # 执行可视化
    log.info("开始特征提取和可视化...")
    visualizer.visualize_all(
        model=model,
        modelF=modelF,
        dataset=dataset,
        labels=None,
        device=args.device,
        save_prefix=f"{args.dataset}_{args.modalities}_{dataset_mode}"
    )
    
    log.info(f"可视化完成！结果保存在: {args.save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JOYFUL模型可视化工具")
    
    # 数据集参数
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        default="iemocap_4",
        choices=["iemocap", "iemocap_4", "mosei", "meld"],
        help="数据集名称",
    )
    parser.add_argument(
        "--data_dir_path",
        type=str,
        default="./data",
        help="数据集目录路径",
    )
    parser.add_argument(
        "--modalities",
        type=str,
        default="atv",
        choices=["a", "t", "v", "at", "tv", "av", "atv"],
        help="使用的模态",
    )
    parser.add_argument(
        "--emotion",
        type=str,
        default=None,
        help="MOSEI数据集的情感类别（multilabel等）",
    )
    parser.add_argument(
        "--transformers",
        action="store_true",
        default=False,
        help="是否使用transformers特征",
    )
    
    # 模型参数
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="模型检查点路径（如果不指定，将使用默认路径）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="计算设备",
    )
    
    # 可视化参数
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./visualizations",
        help="可视化结果保存目录",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="test",
        choices=["train", "dev", "test"],
        help="要可视化的数据集模式",
    )
    
    args = parser.parse_args()
    main(args)

