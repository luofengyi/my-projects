"""
JOYFUL可视化模块
借鉴MMSA的特征可视化方法，适配JOYFUL模型的特点
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import seaborn as sns
from pathlib import Path
import torch

import joyful

log = joyful.utils.get_logger()


class JOYFULVisualizer:
    """JOYFUL模型可视化工具类"""
    
    def __init__(self, args, save_dir="./visualizations"):
        """
        初始化可视化器
        
        Args:
            args: 模型参数
            save_dir: 可视化结果保存目录
        """
        self.args = args
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据集标签映射
        self.dataset_label_dict = {
            "iemocap": {"hap": 0, "sad": 1, "neu": 2, "ang": 3, "exc": 4, "fru": 5},
            "iemocap_4": {"hap": 0, "sad": 1, "neu": 2, "ang": 3},
            "mosei": {"Negative": 0, "Positive": 1},
            "meld": {"Neutral": 0, "Surprise": 1, "Fear": 2, "Sadness": 3, "Joy": 4, "Disgust": 5, "Angry": 6}
        }
        
        if args.dataset and args.emotion == "multilabel":
            self.dataset_label_dict["mosei"] = {
                "happiness": 0, "sadness": 1, "anger": 2,
                "surprise": 3, "disgust": 4, "fear": 5
            }
        
        self.label_to_idx = self.dataset_label_dict.get(args.dataset, {})
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        
    def extract_features(self, model, modelF, dataset, device):
        """
        从模型中提取特征
        
        Args:
            model: JOYFUL模型
            modelF: 融合模型
            dataset: 数据集
            device: 设备
            
        Returns:
            features_dict: 包含各种特征的字典
            labels: 标签列表
        """
        model.eval()
        modelF.eval()
        
        all_features = {
            'input_fused': [],      # 融合后的输入特征
            'seq_context': [],      # 序列上下文特征（RNN输出）
            'graph_input': [],      # 图输入特征（graphify之前）
            'graph_output': [],     # 图输出特征（GNN输出）
            'final': []             # 最终分类器输入特征
        }
        all_labels = []
        
        with torch.no_grad():
            for idx in range(len(dataset)):
                data = dataset[idx]
                labels = data["label_tensor"]  # [batch_size]
                
                # 移动到设备
                for k, v in data.items():
                    if k != "utterance_texts":
                        data[k] = v.to(device)
                
                batch_size = labels.shape[0]
                text_len = data["text_len_tensor"]  # [batch_size]
                
                # 提取融合后的输入特征（按样本聚合）
                input_tensor = data["input_tensor"]  # [batch_size, max_seq_len, feature_dim]
                input_fused_list = []
                for i in range(batch_size):
                    length = text_len[i].item()
                    # 取该样本的平均特征
                    seq_feat = input_tensor[i, :length, :].mean(dim=0)
                    input_fused_list.append(seq_feat)
                if len(input_fused_list) > 0:
                    input_fused = torch.stack(input_fused_list)
                    all_features['input_fused'].append(input_fused.cpu().numpy())
                
                # 提取序列上下文特征
                node_features = model.rnn(data["text_len_tensor"], data["input_tensor"])
                # node_features形状: [total_nodes, feature_dim]，需要按样本聚合
                seq_context_list = []
                start_idx = 0
                for i in range(batch_size):
                    length = text_len[i].item()
                    end_idx = start_idx + length
                    if end_idx <= node_features.shape[0]:
                        seq_feat = node_features[start_idx:end_idx].mean(dim=0)
                        seq_context_list.append(seq_feat)
                    start_idx = end_idx
                if len(seq_context_list) > 0:
                    seq_context = torch.stack(seq_context_list)
                    all_features['seq_context'].append(seq_context.cpu().numpy())
                
                # 提取图输入和输出特征
                graph_out, features, _ = model.get_rep(data, whetherT=False)
                # features和graph_out形状: [total_nodes, feature_dim]
                graph_input_list = []
                graph_output_list = []
                start_idx = 0
                for i in range(batch_size):
                    length = text_len[i].item()
                    end_idx = start_idx + length
                    if end_idx <= features.shape[0]:
                        graph_in_feat = features[start_idx:end_idx].mean(dim=0)
                        graph_out_feat = graph_out[start_idx:end_idx].mean(dim=0)
                        graph_input_list.append(graph_in_feat)
                        graph_output_list.append(graph_out_feat)
                    start_idx = end_idx
                if len(graph_input_list) > 0:
                    graph_input = torch.stack(graph_input_list)
                    graph_output = torch.stack(graph_output_list)
                    all_features['graph_input'].append(graph_input.cpu().numpy())
                    all_features['graph_output'].append(graph_output.cpu().numpy())
                
                # 提取最终特征（分类器输入）
                if model.concat_gin_gout and len(graph_input_list) > 0:
                    final_features = torch.cat([graph_input, graph_output], dim=-1)
                elif len(graph_output_list) > 0:
                    final_features = graph_output
                else:
                    continue
                all_features['final'].append(final_features.cpu().numpy())
                
                # 收集标签（每个样本一个标签）
                all_labels.extend(labels.cpu().numpy().tolist())
        
        # 合并所有特征
        for key in all_features:
            if len(all_features[key]) > 0:
                all_features[key] = np.concatenate(all_features[key], axis=0)
        
        all_labels = np.array(all_labels)
        
        return all_features, all_labels
    
    def visualize_features_pca(self, features_dict, labels, save_prefix="features"):
        """
        使用PCA降维可视化特征（借鉴MMSA方法）
        
        Args:
            features_dict: 特征字典
            labels: 标签数组
            save_prefix: 保存文件前缀
        """
        log.info("开始PCA特征可视化...")
        
        # 为不同情感类别分配颜色
        num_classes = len(self.label_to_idx)
        colors = cm.get_cmap('tab10' if num_classes <= 10 else 'tab20')(np.linspace(0, 1, num_classes))
        color_map = {idx: colors[i] for i, idx in enumerate(sorted(self.label_to_idx.values()))}
        
        all_features_vis = {}
        
        for feature_name, features in features_dict.items():
            if len(features) == 0:
                continue
                
            log.info(f"处理特征: {feature_name}, 形状: {features.shape}")
            
            # 确保特征数量与标签数量匹配
            if len(features) != len(labels):
                # 如果特征数量多于标签，可能是由于序列长度导致的
                # 需要将特征聚合到样本级别
                if len(features) > len(labels):
                    # 简单处理：取前N个特征（N=标签数量）
                    features = features[:len(labels)]
                else:
                    continue
            
            # 2D PCA
            if features.shape[1] > 2:
                pca_2d = PCA(n_components=2, whiten=True)
                features_2d = pca_2d.fit_transform(features)
                explained_var_2d = pca_2d.explained_variance_ratio_
            else:
                features_2d = features[:, :2]
                explained_var_2d = [1.0, 0.0]
            
            # 3D PCA
            if features.shape[1] > 3:
                pca_3d = PCA(n_components=3, whiten=True)
                features_3d = pca_3d.fit_transform(features)
                explained_var_3d = pca_3d.explained_variance_ratio_
            else:
                features_3d = np.pad(features, ((0, 0), (0, 3 - features.shape[1])), mode='constant')
                explained_var_3d = [1.0, 0.0, 0.0]
            
            # 按标签分组
            features_by_label = {}
            for label_idx in self.label_to_idx.values():
                mask = labels == label_idx
                if mask.sum() > 0:
                    features_by_label[self.idx_to_label[label_idx]] = {
                        '2d': features_2d[mask],
                        '3d': features_3d[mask]
                    }
            
            all_features_vis[feature_name] = {
                '2D': features_by_label,
                '3D': features_by_label,
                'explained_var_2d': explained_var_2d,
                'explained_var_3d': explained_var_3d
            }
            
            # 绘制2D图
            plt.figure(figsize=(10, 8))
            for label_name, label_idx in self.label_to_idx.items():
                if label_name in features_by_label:
                    plt.scatter(
                        features_by_label[label_name]['2d'][:, 0],
                        features_by_label[label_name]['2d'][:, 1],
                        c=[color_map[label_idx]],
                        label=label_name,
                        alpha=0.6,
                        s=20
                    )
            plt.xlabel(f'PC1 ({explained_var_2d[0]:.2%} variance)')
            plt.ylabel(f'PC2 ({explained_var_2d[1]:.2%} variance)')
            plt.title(f'PCA 2D Visualization: {feature_name}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.save_dir / f"{save_prefix}_{feature_name}_2d.png", dpi=300, bbox_inches='tight')
            plt.close()
            log.info(f"已保存2D可视化: {save_prefix}_{feature_name}_2d.png")
            
            # 绘制3D图
            if features.shape[1] >= 3:
                fig = plt.figure(figsize=(12, 10))
                ax = fig.add_subplot(111, projection='3d')
                for label_name, label_idx in self.label_to_idx.items():
                    if label_name in features_by_label:
                        ax.scatter(
                            features_by_label[label_name]['3d'][:, 0],
                            features_by_label[label_name]['3d'][:, 1],
                            features_by_label[label_name]['3d'][:, 2],
                            c=[color_map[label_idx]],
                            label=label_name,
                            alpha=0.6,
                            s=20
                        )
                ax.set_xlabel(f'PC1 ({explained_var_3d[0]:.2%} variance)')
                ax.set_ylabel(f'PC2 ({explained_var_3d[1]:.2%} variance)')
                ax.set_zlabel(f'PC3 ({explained_var_3d[2]:.2%} variance)')
                ax.set_title(f'PCA 3D Visualization: {feature_name}')
                ax.legend()
                plt.tight_layout()
                plt.savefig(self.save_dir / f"{save_prefix}_{feature_name}_3d.png", dpi=300, bbox_inches='tight')
                plt.close()
                log.info(f"已保存3D可视化: {save_prefix}_{feature_name}_3d.png")
        
        # 保存特征数据（类似MMSA）
        save_path = self.save_dir / f"{save_prefix}_features.pkl"
        with open(save_path, 'wb') as fp:
            pickle.dump(all_features_vis, fp, protocol=4)
        log.info(f'特征数据已保存到: {save_path}')
        
        return all_features_vis
    
    def visualize_training_curves(self, train_losses, dev_f1s, test_f1s, save_name="training_curves.png"):
        """
        可视化训练曲线
        
        Args:
            train_losses: 训练损失列表
            dev_f1s: 验证集F1分数列表
            test_f1s: 测试集F1分数列表
            save_name: 保存文件名
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        axes[0].plot(train_losses, label='Train Loss', color='blue', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # F1分数曲线
        axes[1].plot(dev_f1s, label='Dev F1', color='green', linewidth=2, marker='o')
        axes[1].plot(test_f1s, label='Test F1', color='red', linewidth=2, marker='s')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('F1 Score')
        axes[1].set_title('F1 Score')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        log.info(f"训练曲线已保存: {save_name}")
    
    def visualize_confusion_matrix(self, y_true, y_pred, save_name="confusion_matrix.png"):
        """
        可视化混淆矩阵
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            save_name: 保存文件名
        """
        cm_matrix = confusion_matrix(y_true, y_pred)
        
        # 获取标签名称
        label_names = [self.idx_to_label.get(i, f"Class {i}") for i in sorted(self.label_to_idx.values())]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=label_names,
            yticklabels=label_names
        )
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        log.info(f"混淆矩阵已保存: {save_name}")
    
    def visualize_all(self, model, modelF, dataset, labels, device, save_prefix="joyful"):
        """
        执行所有可视化
        
        Args:
            model: JOYFUL模型
            modelF: 融合模型
            dataset: 数据集
            labels: 标签（可选，如果不提供则从数据集中提取）
            device: 设备
            save_prefix: 保存文件前缀
        """
        log.info("开始完整可视化流程...")
        
        # 提取特征
        features_dict, extracted_labels = self.extract_features(model, modelF, dataset, device)
        
        # 如果提供了标签，使用提供的标签；否则使用提取的标签
        if labels is not None:
            # 需要将标签对齐到特征数量
            if len(labels) != len(extracted_labels):
                log.warning(f"标签数量({len(labels)})与特征数量({len(extracted_labels)})不匹配，使用提取的标签")
                labels = extracted_labels
        else:
            labels = extracted_labels
        
        # PCA可视化
        self.visualize_features_pca(features_dict, labels, save_prefix=save_prefix)
        
        log.info("可视化完成！")


def visualize_from_checkpoint(checkpoint_path, data_path, args, device="cuda:0", save_dir="./visualizations"):
    """
    从检查点文件加载模型并进行可视化
    
    Args:
        checkpoint_path: 模型检查点路径
        data_path: 数据文件路径
        args: 模型参数（可选，如果不提供则从checkpoint加载）
        device: 设备
        save_dir: 可视化结果保存目录
    """
    import joyful
    
    # 加载模型
    model_dict = torch.load(checkpoint_path, map_location=device)
    if args is None:
        args = model_dict.get("args")
    
    model = model_dict.get("modelN_state_dict") or model_dict.get("state_dict")
    modelF = model_dict.get("modelF_state_dict")
    
    # 加载数据
    data = joyful.utils.load_pkl(data_path)
    testset = joyful.Dataset(data["test"], modelF, False, args)
    
    # 创建可视化器
    visualizer = JOYFULVisualizer(args, save_dir=save_dir)
    
    # 执行可视化
    visualizer.visualize_all(model, modelF, testset, None, device, save_prefix="checkpoint")

