# 损失函数分析与优化方案

## 一、当前损失函数结构分析

### 1.1 总损失组成

在`Coach.train_epoch`中，总损失为：
```python
total_loss = classification_loss + cl_loss_weight * contrastive_loss + 0.05 * encoder_loss
```

**详细分解**：

```
total_loss = 
    classification_loss (来自model.get_loss)
    + cl_loss_weight * contrastive_loss (默认0.2)
    + 0.05 * encoder_loss (硬编码)
```

### 1.2 各组件损失

#### 1.2.1 分类损失 (classification_loss)

**来源**：`model.get_loss()` → `clf.get_loss()`

**组成**：
- **多分类任务**：NLL Loss（可能加权）
  ```python
  loss = NLLLoss(log_prob, label_tensor)
  ```
- **多标签任务**：BCE Loss（可能加权）
  ```python
  loss = BCEWithLogitsLoss(scores, label_tensor.float())
  if class_weight:
      loss = (loss * loss_weights).mean()
  ```

**特点**：
- 支持类别权重平衡（`--class_weight`）
- 不同数据集有不同的权重配置

#### 1.2.2 对比学习损失 (contrastive_loss)

**来源**：`model.get_loss()` → `gcl()` → `contrastive_loss_wo_cross_network()`

**组成**：
- InfoNCE损失（tau=0.2）
- 视图内对比损失

**权重**：`cl_loss_weight`（默认0.2）

**计算**：
```python
cl_loss = contrastive_loss_wo_cross_network(h1, h2, ho)
total_loss += cl_loss_weight * cl_loss
```

#### 1.2.3 融合重构损失 (encoder_loss)

**来源**：`AutoFusion.forward()` 或 `AutoFusion_Hierarchical.forward()`

**组成**（原始AutoFusion）：
```python
loss = globalLoss + interLoss
```

**组成**（改进的AutoFusion_Hierarchical）：
```python
loss = globalLoss + interLoss + gate_regularization
```

**权重**：`0.05`（硬编码在Coach中）

**详细**：
- `globalLoss`：全局重构损失（MSE）
- `interLoss`：局部重构损失（MSE）
- `gate_regularization`：门控正则化（权重0.01）

## 二、损失函数问题分析

### 2.1 权重不平衡问题

**问题**：
1. **硬编码权重**：encoder_loss权重0.05是硬编码的，无法调整
2. **权重比例不当**：不同损失的量级可能差异很大
3. **缺乏自适应**：权重在整个训练过程中固定不变

**影响**：
- 某些损失可能主导训练过程
- 不同损失之间的平衡可能不佳
- 难以根据训练进度调整

### 2.2 损失量级差异

**问题**：
- 分类损失：通常在0.1-10之间
- 对比学习损失：通常在0.01-1之间
- 重构损失：可能在0.1-100之间（取决于特征维度）

**影响**：
- 固定权重可能无法平衡不同量级的损失
- 需要根据实际损失值动态调整

### 2.3 损失函数选择

**问题**：
- MSE损失对异常值敏感
- 没有考虑不同模态的重要性
- 重构损失可能过大，影响分类任务

## 三、优化方案

### 3.1 可配置的损失权重

#### 方案1：添加损失权重参数

在`train.py`中添加参数：
```python
parser.add_argument("--encoder_loss_weight", type=float, default=0.05,
                    help="Weight for encoder reconstruction loss")
parser.add_argument("--gate_reg_weight", type=float, default=0.01,
                    help="Weight for gate regularization")
parser.add_argument("--cl_loss_weight", type=float, default=0.2,
                    help="Weight for contrastive learning loss")
```

#### 方案2：自适应损失权重

根据损失值动态调整权重：
```python
class AdaptiveLossWeight:
    def __init__(self, initial_weights, momentum=0.9):
        self.weights = initial_weights
        self.momentum = momentum
        self.loss_history = {k: [] for k in initial_weights.keys()}
    
    def update(self, losses):
        """根据损失值更新权重"""
        for key, loss in losses.items():
            self.loss_history[key].append(loss.item())
            # 使用移动平均
            if len(self.loss_history[key]) > 1:
                avg_loss = np.mean(self.loss_history[key][-10:])
                # 权重与损失成反比
                self.weights[key] = self.momentum * self.weights[key] + \
                                   (1 - self.momentum) * (1.0 / (avg_loss + 1e-8))
        return self.weights
```

### 3.2 改进损失函数

#### 方案1：使用Smooth L1 Loss替代MSE

Smooth L1对异常值更鲁棒：
```python
# 在fusion_methods_hierarchical.py中
self.criterion = nn.SmoothL1Loss()  # 替代MSE
```

#### 方案2：加权MSE损失

根据模态重要性加权：
```python
class WeightedMSELoss(nn.Module):
    def __init__(self, modal_weights=None):
        super().__init__()
        self.modal_weights = modal_weights or [1.0, 1.0, 1.0]  # a, t, v
    
    def forward(self, pred, target):
        # 按模态分割
        a_pred, t_pred, v_pred = pred.split([100, 768, 512], dim=0)
        a_target, t_target, v_target = target.split([100, 768, 512], dim=0)
        
        loss = (self.modal_weights[0] * F.mse_loss(a_pred, a_target) +
                self.modal_weights[1] * F.mse_loss(t_pred, t_target) +
                self.modal_weights[2] * F.mse_loss(v_pred, v_target))
        return loss
```

#### 方案3：Focal Loss用于分类

处理类别不平衡：
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()
```

### 3.3 损失归一化

#### 方案：损失值归一化

在计算总损失前归一化各损失：
```python
def normalize_losses(losses):
    """归一化损失值到相似量级"""
    normalized = {}
    for key, loss in losses.items():
        # 使用log归一化或min-max归一化
        normalized[key] = torch.log(loss + 1.0)
    return normalized
```

### 3.4 梯度平衡

#### 方案：梯度平衡损失权重

根据梯度大小调整权重：
```python
def compute_gradient_balanced_weights(model, losses, weights):
    """根据梯度大小平衡损失权重"""
    grads = {}
    for key, loss in losses.items():
        loss.backward(retain_graph=True)
        grad_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.data.norm(2).item()
        grads[key] = grad_norm
        model.zero_grad()
    
    # 权重与梯度成反比
    total_grad = sum(grads.values())
    balanced_weights = {k: total_grad / (v + 1e-8) for k, v in grads.items()}
    return balanced_weights
```

## 四、推荐配置

### 4.1 基础配置（当前）

```python
# 损失权重
cl_loss_weight = 0.2
encoder_loss_weight = 0.05  # 硬编码
gate_reg_weight = 0.01  # 在融合模块内

# 损失函数
classification_loss = NLLLoss / BCEWithLogitsLoss
contrastive_loss = InfoNCE
reconstruction_loss = MSELoss
```

### 4.2 优化配置（推荐）

```python
# 损失权重（可配置）
cl_loss_weight = 0.2
encoder_loss_weight = 0.03  # 降低，避免过度关注重构
gate_reg_weight = 0.005  # 降低，避免过度约束

# 损失函数改进
classification_loss = FocalLoss(alpha=1, gamma=2)  # 处理不平衡
contrastive_loss = InfoNCE  # 保持不变
reconstruction_loss = SmoothL1Loss  # 更鲁棒
```

### 4.3 自适应配置（高级）

```python
# 使用自适应权重
adaptive_weights = AdaptiveLossWeight({
    'classification': 1.0,
    'contrastive': 0.2,
    'encoder': 0.05,
    'gate_reg': 0.01
})

# 每个epoch更新权重
for epoch in range(epochs):
    losses = compute_losses(...)
    weights = adaptive_weights.update(losses)
    total_loss = sum(weights[k] * losses[k] for k in weights.keys())
```

## 五、实现建议

### 5.1 立即实施（简单）

1. **添加损失权重参数**
   - 在`train.py`中添加`--encoder_loss_weight`
   - 在`Coach`中使用参数而非硬编码

2. **改进损失函数**
   - 使用`SmoothL1Loss`替代`MSELoss`
   - 添加可选的`FocalLoss`

### 5.2 中期实施（中等）

1. **损失归一化**
   - 实现损失值归一化
   - 确保各损失在相似量级

2. **加权重构损失**
   - 根据模态重要性加权
   - 考虑不同模态的贡献

### 5.3 长期实施（复杂）

1. **自适应权重**
   - 实现自适应损失权重
   - 根据训练进度动态调整

2. **梯度平衡**
   - 实现梯度平衡机制
   - 确保各损失对训练的贡献平衡

## 六、实验建议

### 6.1 消融实验

1. **权重消融**：
   - 测试不同`encoder_loss_weight`（0.01, 0.03, 0.05, 0.1）
   - 测试不同`cl_loss_weight`（0.1, 0.2, 0.3, 0.5）

2. **损失函数消融**：
   - MSE vs SmoothL1
   - NLL vs Focal Loss

### 6.2 超参数搜索

使用网格搜索或贝叶斯优化：
```python
# 搜索空间
search_space = {
    'encoder_loss_weight': [0.01, 0.03, 0.05, 0.1],
    'cl_loss_weight': [0.1, 0.2, 0.3],
    'gate_reg_weight': [0.001, 0.005, 0.01, 0.05]
}
```

## 七、监控指标

### 7.1 损失监控

建议监控以下指标：
1. **各损失值**：分类损失、对比损失、重构损失
2. **损失比例**：各损失占总损失的比例
3. **梯度范数**：各模块的梯度范数
4. **权重变化**：如果使用自适应权重

### 7.2 可视化

```python
# 损失曲线
plt.plot(epochs, classification_losses, label='Classification')
plt.plot(epochs, contrastive_losses, label='Contrastive')
plt.plot(epochs, encoder_losses, label='Encoder')
plt.plot(epochs, total_losses, label='Total')
plt.legend()
plt.show()
```

## 八、总结

### 当前问题

1. ✅ 硬编码权重（encoder_loss_weight = 0.05）
2. ✅ 损失量级差异大
3. ✅ MSE对异常值敏感
4. ✅ 缺乏自适应机制

### 优化方向

1. ✅ **可配置权重**：添加参数控制
2. ✅ **改进损失函数**：SmoothL1, Focal Loss
3. ✅ **损失归一化**：平衡不同损失
4. ✅ **自适应权重**：动态调整（可选）

### 推荐优先级

1. **高优先级**：添加可配置权重参数
2. **中优先级**：改进损失函数（SmoothL1）
3. **低优先级**：自适应权重机制

