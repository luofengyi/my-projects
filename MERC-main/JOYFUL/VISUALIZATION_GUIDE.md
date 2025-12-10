# 训练过程可视化指南

## 一、功能概述

训练完成后，会自动生成三个可视化图表：

1. **图表1**：损失结果随epoch变化的曲线
2. **图表2**：训练时平均F1、DEV F1、Test F1随epoch变化的曲线
3. **图表3**：四个情感类别随epoch变化的曲线

## 二、自动生成

### 2.1 训练时自动保存

训练完成后，会自动：
1. 保存训练历史到 `training_history/{dataset}_{modalities}_history.json`
2. 包含所有epoch的损失和F1分数数据

### 2.2 数据记录内容

训练历史包含：
- `epochs`: 所有epoch编号
- `train_losses`: 每个epoch的训练损失
- `train_f1s`: 每个epoch的训练F1分数
- `dev_f1s`: 每个epoch的开发集F1分数
- `test_f1s`: 每个epoch的测试集F1分数
- `class_f1s`: 每个类别的F1分数（字典）

## 三、生成可视化图表

### 3.1 训练后自动生成

训练完成后，运行可视化脚本：

```bash
# 基础使用（自动检测历史文件）
python visualize_training.py --dataset="iemocap_4" --modalities="atv"

# 指定历史文件
python visualize_training.py --history_file="training_history/iemocap_4_atv_history.json"

# 指定输出目录
python visualize_training.py --dataset="iemocap_4" --modalities="atv" --output_dir="my_plots"
```

### 3.2 输出文件

生成的图表保存在 `training_plots/` 目录（或指定的输出目录）：

1. **`{dataset}_{modalities}_loss_curve.png`** - 损失曲线（图表1）
2. **`{dataset}_{modalities}_f1_curves.png`** - F1曲线（图表2）
3. **`{dataset}_{modalities}_class_f1_curves.png`** - 类别F1曲线（图表3）

## 四、图表说明

### 4.1 图表1：损失曲线

**内容**：
- X轴：Epoch
- Y轴：Loss
- 曲线：训练损失

**特点**：
- 蓝色实线
- 带标记点
- 网格背景

### 4.2 图表2：F1分数曲线

**内容**：
- X轴：Epoch
- Y轴：F1 Score
- 曲线：
  - 蓝色：Train F1
  - 绿色：Dev F1
  - 红色：Test F1

**特点**：
- 三条曲线对比
- 不同颜色和标记
- 网格背景

### 4.3 图表3：各类别F1曲线

**内容**：
- X轴：Epoch
- Y轴：F1 Score
- 曲线：每个情感类别的F1分数

**IEMOCAP_4的四个类别**：
- Happy (hap)
- Sad (sad)
- Neutral (neu)
- Angry (ang)

**特点**：
- 不同颜色区分类别
- 图例显示类别名称
- 网格背景

## 五、使用示例

### 5.1 完整流程

```bash
# 步骤1：训练模型
python train.py \
    --dataset="iemocap_4" \
    --modalities="atv" \
    --encoder_loss_weight=0.03 \
    --use_hierarchical_fusion \
    --use_smooth_l1 \
    --epochs=50

# 步骤2：生成可视化图表
python visualize_training.py \
    --dataset="iemocap_4" \
    --modalities="atv"
```

### 5.2 查看结果

训练完成后，检查：

1. **训练历史文件**：
   ```
   training_history/iemocap_4_atv_history.json
   ```

2. **可视化图表**：
   ```
   training_plots/iemocap_4_atv_loss_curve.png
   training_plots/iemocap_4_atv_f1_curves.png
   training_plots/iemocap_4_atv_class_f1_curves.png
   ```

## 六、参数说明

### 6.1 visualize_training.py参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--history_file` | 训练历史JSON文件路径 | 自动检测 |
| `--dataset` | 数据集名称 | iemocap_4 |
| `--modalities` | 模态组合 | atv |
| `--output_dir` | 输出目录 | training_plots |

### 6.2 自动检测规则

如果不指定`--history_file`，脚本会自动查找：
```
training_history/{dataset}_{modalities}_history.json
```

## 七、数据格式

### 7.1 JSON格式示例

```json
{
  "epochs": [1, 2, 3, ...],
  "train_losses": [5.2, 3.1, 2.0, ...],
  "train_f1s": [0.45, 0.58, 0.65, ...],
  "dev_f1s": [0.42, 0.55, 0.62, ...],
  "test_f1s": [0.40, 0.53, 0.60, ...],
  "class_f1s": {
    "hap": [0.50, 0.60, 0.65, ...],
    "sad": [0.45, 0.55, 0.60, ...],
    "neu": [0.40, 0.50, 0.55, ...],
    "ang": [0.35, 0.45, 0.50, ...]
  }
}
```

## 八、注意事项

1. **依赖库**：需要安装matplotlib
   ```bash
   pip install matplotlib
   ```

2. **训练历史**：只有在训练完成后才会生成历史文件

3. **类别数量**：图表3会根据数据集自动调整类别数量

4. **中文支持**：如果类别名称是中文，可能需要配置字体

## 九、故障排除

### 问题1：找不到历史文件

**错误**：`History file not found`

**解决**：
- 确保已经完成训练
- 检查文件路径是否正确
- 使用`--history_file`指定完整路径

### 问题2：图表3没有数据

**原因**：可能不是IEMOCAP_4数据集，或类别F1数据未记录

**解决**：
- 检查数据集类型
- 确认训练历史中包含`class_f1s`字段

### 问题3：图表显示异常

**原因**：可能是matplotlib配置问题

**解决**：
- 更新matplotlib版本
- 检查字体配置

## 十、总结

### 功能特点

- ✅ 自动记录训练过程
- ✅ 自动保存历史数据
- ✅ 一键生成可视化图表
- ✅ 支持多个数据集

### 使用流程

1. 训练模型（自动保存历史）
2. 运行可视化脚本
3. 查看生成的图表

### 输出文件

- 训练历史：`training_history/*.json`
- 可视化图表：`training_plots/*.png`

**推荐**：训练完成后立即运行可视化脚本，查看训练效果！





