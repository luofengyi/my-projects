# 快速可视化使用指南

## 一、快速开始

### 步骤1：训练模型

```bash
python train.py \
    --dataset="iemocap_4" \
    --modalities="atv" \
    --encoder_loss_weight=0.03 \
    --epochs=50
```

训练完成后，会自动生成历史文件：
```
training_history/iemocap_4_atv_history.json
```

### 步骤2：生成可视化图表

```bash
python visualize_training.py \
    --dataset="iemocap_4" \
    --modalities="atv"
```

## 二、生成的三个图表

### 图表1：损失曲线
**文件**：`training_plots/iemocap_4_atv_loss_curve.png`
- 显示训练损失随epoch的变化

### 图表2：F1分数曲线
**文件**：`training_plots/iemocap_4_atv_f1_curves.png`
- 显示Train F1、Dev F1、Test F1随epoch的变化

### 图表3：各类别F1曲线
**文件**：`training_plots/iemocap_4_atv_class_f1_curves.png`
- 显示四个情感类别（hap, sad, neu, ang）的F1分数随epoch的变化

## 三、一键生成所有图表

```bash
# 自动检测并生成
python visualize_training.py --dataset="iemocap_4" --modalities="atv"
```

输出：
```
✓ Chart 1 saved: training_plots/iemocap_4_atv_loss_curve.png
✓ Chart 2 saved: training_plots/iemocap_4_atv_f1_curves.png
✓ Chart 3 saved: training_plots/iemocap_4_atv_class_f1_curves.png
```

## 四、注意事项

1. **需要matplotlib**：`pip install matplotlib`
2. **训练完成后**：只有训练完成后才会生成历史文件
3. **自动保存**：训练历史在训练结束时自动保存






