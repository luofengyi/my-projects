# JOYFUL 可视化功能说明

本可视化模块借鉴了MMSA模型的可视化方法，为JOYFUL模型提供了丰富的可视化功能。

## 功能特性

### 1. 特征PCA可视化
- **2D PCA可视化**：将高维特征降维到2维，按情感类别着色显示
- **3D PCA可视化**：将高维特征降维到3维，提供更丰富的空间信息
- **多层级特征可视化**：
  - `input_fused`: 融合后的输入特征
  - `seq_context`: 序列上下文特征（RNN输出）
  - `graph_input`: 图输入特征
  - `graph_output`: 图输出特征（GNN输出）
  - `final`: 最终分类器输入特征

### 2. 训练曲线可视化
- 训练损失曲线
- 验证集和测试集F1分数曲线

### 3. 混淆矩阵可视化
- 按情感类别显示分类结果的混淆矩阵

## 使用方法

### 方法1：使用独立可视化脚本

```bash
python visualize.py \
    --dataset iemocap_4 \
    --modalities atv \
    --data_dir_path ./data \
    --checkpoint_path model_checkpoints/iemocap_4_best_dev_f1_model_atv.pt \
    --device cuda:0 \
    --save_dir ./visualizations \
    --mode test
```

参数说明：
- `--dataset`: 数据集名称（iemocap, iemocap_4, mosei, meld）
- `--modalities`: 使用的模态（a, t, v, at, tv, av, atv）
- `--data_dir_path`: 数据目录路径
- `--checkpoint_path`: 模型检查点路径（可选，默认使用标准路径）
- `--device`: 计算设备（cuda:0 或 cpu）
- `--save_dir`: 可视化结果保存目录
- `--mode`: 要可视化的数据集模式（train, dev, test）

### 方法2：在训练过程中自动可视化

在训练脚本中添加可视化参数：

```python
args.enable_visualization = True
args.visualization_dir = "./visualizations"
```

训练完成后会自动生成：
- 训练曲线图
- 测试集混淆矩阵

### 方法3：在代码中使用可视化类

```python
from joyful.visualization import JOYFULVisualizer

# 创建可视化器
visualizer = JOYFULVisualizer(args, save_dir="./visualizations")

# 提取特征并可视化
features_dict, labels = visualizer.extract_features(model, modelF, dataset, device)
visualizer.visualize_features_pca(features_dict, labels, save_prefix="my_features")

# 可视化训练曲线
visualizer.visualize_training_curves(train_losses, dev_f1s, test_f1s)

# 可视化混淆矩阵
visualizer.visualize_confusion_matrix(y_true, y_pred)
```

## 输出文件

可视化结果保存在指定的`save_dir`目录下：

```
visualizations/
├── features_input_fused_2d.png      # 输入特征2D PCA
├── features_input_fused_3d.png      # 输入特征3D PCA
├── features_seq_context_2d.png     # 序列上下文特征2D PCA
├── features_seq_context_3d.png     # 序列上下文特征3D PCA
├── features_graph_input_2d.png     # 图输入特征2D PCA
├── features_graph_input_3d.png     # 图输入特征3D PCA
├── features_graph_output_2d.png    # 图输出特征2D PCA
├── features_graph_output_3d.png    # 图输出特征3D PCA
├── features_final_2d.png           # 最终特征2D PCA
├── features_final_3d.png           # 最终特征3D PCA
├── features_features.pkl           # 特征数据（用于后续分析）
├── training_curves.png             # 训练曲线
└── confusion_matrix.png             # 混淆矩阵
```

## 依赖包

确保安装以下Python包：

```bash
pip install matplotlib seaborn scikit-learn numpy torch
```

## 与MMSA可视化的对比

| 特性 | MMSA | JOYFUL |
|------|------|--------|
| PCA降维 | ✅ | ✅ |
| 2D/3D可视化 | ✅ | ✅ |
| 按标签着色 | ✅ | ✅ |
| 多模态特征 | ✅ | ✅ |
| 训练曲线 | ❌ | ✅ |
| 混淆矩阵 | ❌ | ✅ |
| 图结构特征 | ❌ | ✅ |

## 注意事项

1. **特征对齐**：JOYFUL模型使用图结构，特征提取时会自动将节点级特征聚合到样本级
2. **内存使用**：对于大型数据集，建议使用较小的batch size或分批处理
3. **标签映射**：不同数据集的标签映射已自动处理，支持IEMOCAP、MOSEI、MELD等数据集

## 示例输出

### PCA特征可视化示例
- 不同情感类别的特征在PCA空间中应该形成可分离的簇
- 如果特征分离良好，说明模型学习到了有效的表示

### 训练曲线示例
- 训练损失应该逐渐下降
- 验证集和测试集F1分数应该逐渐上升并趋于稳定

### 混淆矩阵示例
- 对角线元素应该较大，表示分类准确
- 可以识别模型容易混淆的情感类别

## 故障排除

1. **特征数量不匹配**：如果出现特征数量与标签数量不匹配的错误，检查数据集是否正确加载
2. **内存不足**：减少batch size或使用CPU模式
3. **可视化图片空白**：检查matplotlib后端设置，可能需要安装GUI后端

## 参考

- MMSA可视化方法：`1721816976181_SelfMM/MMSA/src/MMSA/run.py` (第504-554行)
- JOYFUL模型结构：`joyful/model/JOYFUL.py`

