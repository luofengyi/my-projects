"""
训练过程可视化脚本
绘制三个图表：
1. 损失结果随epoch变化的曲线
2. 训练时平均F1、DEV F1、Test F1随epoch变化的曲线
3. 四个情感类别随epoch变化的曲线
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse

# 设置中文字体（如果需要显示中文）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_training_history(history_file):
    """加载训练历史数据"""
    with open(history_file, 'r') as f:
        history = json.load(f)
    return history


def plot_loss_curve(history, save_path):
    """
    绘制图表1：损失结果随epoch变化的曲线
    标注最佳轮次（损失最低的点）
    """
    epochs = history['epochs']
    train_losses = history['train_losses']
    
    # 找到最佳点（损失最低）
    best_idx = np.argmin(train_losses)
    best_epoch = epochs[best_idx]
    best_loss = train_losses[best_idx]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss', marker='o', markersize=4)
    
    # 标注最佳点
    plt.plot(best_epoch, best_loss, 'ro', markersize=10, zorder=5)
    plt.annotate(f'Best: Epoch {best_epoch}\nLoss: {best_loss:.4f}',
                xy=(best_epoch, best_loss),
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Chart 1 saved: {save_path} (Best: Epoch {best_epoch}, Loss: {best_loss:.4f})")


def plot_f1_curves(history, save_path):
    """
    绘制图表2：训练时平均F1、DEV F1、Test F1随epoch变化的曲线
    标注最佳轮次（Dev F1最高的点，因为通常用Dev F1选择最佳模型）
    """
    epochs = history['epochs']
    train_f1s = history['train_f1s']
    dev_f1s = history['dev_f1s']
    test_f1s = history['test_f1s']
    
    # 找到最佳点（Dev F1最高，因为通常用Dev F1选择最佳模型）
    best_dev_idx = np.argmax(dev_f1s)
    best_dev_epoch = epochs[best_dev_idx]
    best_dev_f1 = dev_f1s[best_dev_idx]
    best_test_f1 = test_f1s[best_dev_idx]  # 对应最佳Dev F1的Test F1
    
    # 找到Test F1最高的点（可能不同）
    best_test_idx = np.argmax(test_f1s)
    best_test_epoch = epochs[best_test_idx]
    best_test_f1_max = test_f1s[best_test_idx]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_f1s, 'b-', linewidth=2, label='Train F1', marker='o', markersize=4)
    plt.plot(epochs, dev_f1s, 'g-', linewidth=2, label='Dev F1', marker='s', markersize=4)
    plt.plot(epochs, test_f1s, 'r-', linewidth=2, label='Test F1', marker='^', markersize=4)
    
    # 标注最佳Dev F1点（主要标注）
    plt.plot(best_dev_epoch, best_dev_f1, 'go', markersize=12, zorder=5, label='Best Dev F1')
    plt.annotate(f'Best Dev: Epoch {best_dev_epoch}\nDev F1: {best_dev_f1:.4f}\nTest F1: {best_test_f1:.4f}',
                xy=(best_dev_epoch, best_dev_f1),
                xytext=(15, 15),
                textcoords='offset points',
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # 如果Test F1最高的点与Dev F1最佳点不同，也标注
    if best_test_epoch != best_dev_epoch:
        plt.plot(best_test_epoch, best_test_f1_max, 'r*', markersize=12, zorder=5, label='Best Test F1')
        plt.annotate(f'Best Test: Epoch {best_test_epoch}\nTest F1: {best_test_f1_max:.4f}',
                    xy=(best_test_epoch, best_test_f1_max),
                    xytext=(15, -25),
                    textcoords='offset points',
                    fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('F1 Score Curves (Train/Dev/Test)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Chart 2 saved: {save_path} (Best Dev: Epoch {best_dev_epoch}, Dev F1: {best_dev_f1:.4f}, Test F1: {best_test_f1:.4f})")


def plot_class_f1_curves(history, save_path, dataset='iemocap_4'):
    """
    绘制图表3：四个情感类别随epoch变化的曲线
    """
    epochs = history['epochs']
    class_f1s = history.get('class_f1s', {})
    
    if not class_f1s:
        print("⚠ Warning: No class F1 data found. Skipping chart 3.")
        return
    
    plt.figure(figsize=(10, 6))
    
    # 定义颜色和标记
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'v', 'D', 'p']
    
    # 对于IEMOCAP_4，有4个类别：hap, sad, neu, ang
    if dataset == 'iemocap_4':
        class_order = ['hap', 'sad', 'neu', 'ang']
        class_labels = ['Happy', 'Sad', 'Neutral', 'Angry']
    elif dataset == 'iemocap':
        class_order = ['hap', 'sad', 'neu', 'ang', 'exc', 'fru']
        class_labels = ['Happy', 'Sad', 'Neutral', 'Angry', 'Excited', 'Frustrated']
    else:
        # 使用所有可用的类别
        class_order = list(class_f1s.keys())
        class_labels = [name.capitalize() for name in class_order]
    
    # 绘制每个类别的F1曲线并标注最佳点
    best_points_info = []
    for i, (class_name, class_label) in enumerate(zip(class_order, class_labels)):
        if class_name in class_f1s:
            f1_scores = class_f1s[class_name]
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            plt.plot(epochs, f1_scores, 
                    color=color, linewidth=2, 
                    label=class_label, 
                    marker=marker, markersize=4)
            
            # 找到该类别的最佳点
            best_idx = np.argmax(f1_scores)
            best_epoch = epochs[best_idx]
            best_f1 = f1_scores[best_idx]
            
            # 标注最佳点
            plt.plot(best_epoch, best_f1, 
                    color=color, marker='*', markersize=12, 
                    markeredgecolor='black', markeredgewidth=1, zorder=5)
            
            # 添加文本标注（错开位置避免重叠）
            offset_x = 15 if i % 2 == 0 else -15
            offset_y = 15 if i < len(class_order) // 2 else -25
            plt.annotate(f'{class_label}\nEpoch {best_epoch}\nF1: {best_f1:.4f}',
                        xy=(best_epoch, best_f1),
                        xytext=(offset_x, offset_y),
                        textcoords='offset points',
                        fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.4', facecolor=color, alpha=0.3, edgecolor=color),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color=color, lw=1.5))
            
            best_points_info.append(f"{class_label}: Epoch {best_epoch}, F1: {best_f1:.4f}")
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('F1 Score by Emotion Class', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11, loc='best')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Chart 3 saved: {save_path}")
    for info in best_points_info:
        print(f"   {info}")


def main():
    parser = argparse.ArgumentParser(description='Visualize training results')
    parser.add_argument('--history_file', type=str, 
                       default=None,
                       help='Path to training history JSON file')
    parser.add_argument('--dataset', type=str, default='iemocap_4',
                       choices=['iemocap', 'iemocap_4', 'mosei', 'meld'],
                       help='Dataset name')
    parser.add_argument('--modalities', type=str, default='atv',
                       help='Modalities used')
    parser.add_argument('--output_dir', type=str, default='training_plots',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # 确定历史文件路径
    if args.history_file is None:
        history_file = os.path.join(
            'training_history',
            f"{args.dataset}_{args.modalities}_history.json"
        )
    else:
        history_file = args.history_file
    
    # 检查文件是否存在
    if not os.path.exists(history_file):
        print(f"❌ Error: History file not found: {history_file}")
        print("Please run training first to generate the history file.")
        return
    
    # 加载历史数据
    print(f"Loading training history from: {history_file}")
    history = load_training_history(history_file)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 绘制三个图表
    print("\nGenerating visualization charts...")
    
    # 图表1：损失曲线
    loss_path = os.path.join(args.output_dir, f"{args.dataset}_{args.modalities}_loss_curve.png")
    plot_loss_curve(history, loss_path)
    
    # 图表2：F1曲线
    f1_path = os.path.join(args.output_dir, f"{args.dataset}_{args.modalities}_f1_curves.png")
    plot_f1_curves(history, f1_path)
    
    # 图表3：各类别F1曲线
    class_f1_path = os.path.join(args.output_dir, f"{args.dataset}_{args.modalities}_class_f1_curves.png")
    plot_class_f1_curves(history, class_f1_path, dataset=args.dataset)
    
    print(f"\n✅ All charts saved to: {args.output_dir}")
    print(f"   - Chart 1 (Loss): {loss_path}")
    print(f"   - Chart 2 (F1 Scores): {f1_path}")
    print(f"   - Chart 3 (Class F1): {class_f1_path}")


if __name__ == "__main__":
    main()

