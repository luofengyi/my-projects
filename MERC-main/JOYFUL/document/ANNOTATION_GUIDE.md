# 曲线标注说明

## 一、标注功能

所有绘制的曲线都会自动标注最佳轮次（epoch）对应的结果数值。

## 二、三个图表的标注说明

### 2.1 图表1：损失曲线

**标注内容**：
- **最佳点**：损失最低的epoch
- **标注信息**：
  - Epoch编号
  - 对应的Loss数值

**标注样式**：
- 红色圆点标记最佳点
- 黄色背景的文本框显示详细信息
- 箭头指向最佳点

**示例**：
```
Best: Epoch 25
Loss: 0.1234
```

### 2.2 图表2：F1分数曲线

**标注内容**：
- **主要标注**：Dev F1最高的epoch（因为通常用Dev F1选择最佳模型）
  - Dev F1数值
  - 对应epoch的Test F1数值
- **次要标注**：如果Test F1最高的epoch与Dev F1最佳epoch不同，也会标注
  - Test F1最高值

**标注样式**：
- 绿色圆点标记最佳Dev F1点
- 浅绿色背景文本框显示Dev F1信息
- 红色星号标记最佳Test F1点（如果不同）
- 浅红色背景文本框显示Test F1信息

**示例**：
```
主要标注：
Best Dev: Epoch 28
Dev F1: 0.6543
Test F1: 0.6421

次要标注（如果不同）：
Best Test: Epoch 30
Test F1: 0.6489
```

### 2.3 图表3：各类别F1曲线

**标注内容**：
- **每个类别**：该类别F1最高的epoch
- **标注信息**：
  - 类别名称
  - Epoch编号
  - F1数值

**标注样式**：
- 星号标记每个类别的最佳点
- 与类别曲线相同颜色的文本框
- 错开位置避免重叠

**示例**（IEMOCAP_4的四个类别）：
```
Happy
Epoch 25
F1: 0.7123

Sad
Epoch 28
F1: 0.6543

Neutral
Epoch 30
F1: 0.6890

Angry
Epoch 27
F1: 0.6234
```

## 三、标注特点

### 3.1 视觉特点

1. **清晰可见**：
   - 使用较大的标记点（星号、圆点）
   - 高对比度的颜色
   - 明显的文本框背景

2. **信息完整**：
   - 显示Epoch编号
   - 显示对应的数值
   - 包含必要的上下文信息

3. **布局合理**：
   - 标注位置错开，避免重叠
   - 箭头指向准确
   - 文本框大小适中

### 3.2 技术实现

- 使用`matplotlib.annotate`添加标注
- 自动计算最佳点位置
- 智能调整标注位置避免重叠
- 高分辨率输出（300 DPI）

## 四、使用示例

### 4.1 生成带标注的图表

```bash
python visualize_training.py \
    --dataset="iemocap_4" \
    --modalities="atv"
```

### 4.2 输出信息

运行后会显示最佳点信息：

```
✓ Chart 1 saved: training_plots/iemocap_4_atv_loss_curve.png (Best: Epoch 25, Loss: 0.1234)
✓ Chart 2 saved: training_plots/iemocap_4_atv_f1_curves.png (Best Dev: Epoch 28, Dev F1: 0.6543, Test F1: 0.6421)
✓ Chart 3 saved: training_plots/iemocap_4_atv_class_f1_curves.png
   Happy: Epoch 25, F1: 0.7123
   Sad: Epoch 28, F1: 0.6543
   Neutral: Epoch 30, F1: 0.6890
   Angry: Epoch 27, F1: 0.6234
```

## 五、最佳点选择逻辑

### 5.1 损失曲线

- **选择标准**：损失最低的点
- **方法**：`np.argmin(train_losses)`

### 5.2 F1曲线

- **主要标注**：Dev F1最高的点（模型选择标准）
- **方法**：`np.argmax(dev_f1s)`
- **次要标注**：Test F1最高的点（如果不同）
- **方法**：`np.argmax(test_f1s)`

### 5.3 类别F1曲线

- **选择标准**：每个类别F1最高的点
- **方法**：对每个类别使用`np.argmax(class_f1s[class_name])`

## 六、注意事项

1. **最佳点可能不同**：
   - Dev F1最佳和Test F1最佳可能不在同一epoch
   - 不同类别的最佳epoch也可能不同

2. **标注位置**：
   - 自动调整避免重叠
   - 如果点太密集，可能需要手动查看

3. **数值精度**：
   - F1分数显示4位小数
   - Loss显示4位小数

## 七、总结

### 功能特点

- ✅ 自动标注最佳点
- ✅ 显示详细信息（Epoch + 数值）
- ✅ 清晰的视觉标记
- ✅ 智能布局避免重叠

### 使用价值

1. **快速识别**：一眼看出最佳训练轮次
2. **性能对比**：对比不同指标的最佳点
3. **模型选择**：基于Dev F1选择最佳模型
4. **类别分析**：了解每个类别的最佳表现

**推荐**：训练完成后查看标注，快速了解模型最佳性能！






