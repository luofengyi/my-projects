# rPPG质量问题修复指南

## 问题诊断

### 现象
添加rPPG后，Happy类F1分数从**0.83降至0.73**，且前40轮几乎无法学习：
- **图1（添加rPPG后）**：Happy F1在前40轮徘徊在0-0.1，第40轮后才缓慢上升
- **图2（未添加rPPG）**：Happy F1从第10轮就开始正常学习，第30轮已达0.6+

### 根本原因

#### 1. **零向量污染**（最严重）
```python
# 原代码（Dataset.py）
if rppg is None:
    rppg = torch.zeros(self.rppg_raw_dim, device=a.device)  # 问题所在！
```

**问题链**：
1. 数据集没有真实rPPG → 使用零向量填充
2. 零向量经过`projectR`投影 → 产生460维非零伪特征
3. 伪特征拼接到融合输入 → 输入维度从1380升至1840
4. **有效信号被稀释**：A/T/V的判别信息占比从100%降至75%
5. **Happy作为少数类**（14%样本）最先受到影响

#### 2. **ULGM被污染**
```python
# 原代码（_compute_unimodal_loss）
fusion_input = torch.cat([A_proj, T_proj, V_proj, R_proj])  # rPPG污染多模态特征
```
- 伪标签生成基于污染的融合特征
- Happy类伪标签质量下降
- `min_samples=20`的保护机制延迟生效

#### 3. **门控机制无法识别无效信号**
- `UtteranceLevelGate`无法判断rPPG是零向量投影产生的伪特征
- 注意力权重平均分配给4个模态 → 有效模态A/T/V权重下降

## 修复方案

### 核心策略
**检测无效rPPG → 完全跳过rPPG分支 → 恢复原有A/T/V融合效果**

### 修改1：Dataset.py - 质量检测

```python
# 质量检测：如果是零向量或低方差，设为None（完全跳过rPPG分支）
if rppg_feat is not None:
    abs_max = torch.abs(rppg_feat).max().item()
    variance = torch.var(rppg_feat).item()
    # 零向量检测（阈值1e-6）或低方差检测（阈值1e-4）
    if abs_max < 1e-6 or variance < 1e-4:
        rppg_feat = None  # 标记为无效，融合时完全跳过
```

**效果**：
- ✅ 阻止零向量进入融合模块
- ✅ 保持向后兼容（真实rPPG仍正常处理）

### 修改2：fusion_methods_hierarchical.py - 动态跳过

#### 2.1 forward方法：双重质量检测
```python
# rPPG质量检测：如果为None或无效，完全跳过rPPG分支
use_rppg_this_sample = False
if self.use_rppg and rppg is not None:
    rppg = torch.as_tensor(rppg, dtype=torch.float32, device=a.device)
    # 再次检测质量（防止传入无效rPPG）
    abs_max = torch.abs(rppg).max().item()
    variance = torch.var(rppg).item()
    if abs_max >= 1e-6 and variance >= 1e-4:
        use_rppg_this_sample = True
    else:
        rppg = None  # 无效，设为None
```

**效果**：
- ✅ 双重保护（Dataset + Fusion）
- ✅ 样本级别的动态判断

#### 2.2 只在有效时添加rPPG分支
```python
modal_projections.extend([A, T, V])
modal_weighted.extend([bba, bbt, bbv])

# 只有在rPPG有效时才添加rPPG分支
if use_rppg_this_sample and self.projectR is not None:
    R, bbr = _project_and_weight(rppg, self.projectR)
    modal_projections.append(R)
    modal_weighted.append(bbr)
```

**效果**：
- ✅ 无效rPPG时，融合输入退化为A/T/V（1380维）
- ✅ 恢复原有模型性能

#### 2.3 动态维度处理
```python
# 动态维度处理：如果rPPG被跳过，维度会减少
actual_dim = concat_features.shape[0]
expected_dim_without_rppg = 3 * self.proj_dim  # A/T/V: 1380
expected_dim_with_rppg = 4 * self.proj_dim     # A/T/V/R: 1840

# 如果实际维度与预期不匹配，进行填充或调整
if actual_dim == expected_dim_without_rppg and self.input_features == expected_dim_with_rppg:
    # rPPG被跳过，但模型期望包含rPPG：用零填充
    padding = torch.zeros(self.proj_dim, device=concat_features.device)
    concat_features = torch.cat([concat_features, padding])
```

**效果**：
- ✅ 兼容--use_rppg开关与实际rPPG可用性不一致的情况
- ✅ 避免维度错误

### 修改3：ULGM模块排除rPPG

#### 3.1 forward调用时不传递rPPG
```python
# ========== ULGM模块：单模态监督（仅训练时） ==========
unimodal_loss = None
if self.use_ulgm and self.training and labels is not None:
    # ULGM只使用A/T/V原始特征，不使用rPPG
    # 原因：rPPG是生理信号，不适合生成单模态伪标签
    unimodal_loss = self._compute_unimodal_loss(a, t, v, labels)
```

#### 3.2 _compute_unimodal_loss只使用A/T/V
```python
# ========== 步骤1：特征提取（只使用A/T/V，不使用rPPG） ==========
# 多模态融合特征：拼接A/T/V原始特征
fusion_input = torch.cat([a, t, v], dim=-1)  # [100+768+512=1380]
fusion_h = self.post_fusion_dropout(fusion_input)
fusion_h = F_nn.relu(self.post_fusion_layer_1(fusion_h), inplace=False)
```

#### 3.3 ULGM融合层维度修正
```python
# 注意：ULGM只使用A/T/V（1380维），不使用rPPG
ulgm_fusion_dim = 100 + 768 + 512  # A/T/V: 1380
self.post_fusion_layer_1 = nn.Linear(ulgm_fusion_dim, hidden_size)
```

**原因**：
- rPPG是**生理信号**（心率、血容量变化）
- A/T/V是**表达信号**（面部表情、语音韵律、语义内容）
- 生理与表达的产生机制不同，不应混合生成伪标签

**效果**：
- ✅ ULGM伪标签质量不受rPPG影响
- ✅ Happy类min_samples机制正常工作

### 修改4：新增rPPG质量检测模块（可选）

创建`joyful/rppg_quality_checker.py`，包含：

#### 4.1 `RPPGQualityChecker`（非参数）
```python
def check_quality(self, rppg_features):
    # 检查零向量
    # 检查低方差
    # 返回is_valid, quality_score, reason
```

**用途**：
- 统一的质量评估标准
- 可配置的阈值
- 提供质量分数用于自适应加权

#### 4.2 `RPPGAdaptiveWeightModule`（可学习）
```python
class RPPGAdaptiveWeightModule(nn.Module):
    # 根据rPPG特征和上下文，学习最优权重
```

**用途**（未来）：
- 当有真实rPPG时，自适应调整rPPG权重
- 低质量rPPG降低权重，高质量rPPG增加权重

## 使用指南

### 场景1：数据集无rPPG（当前情况）

**不启用rPPG**（推荐）：
```bash
python train.py \
  --dataset iemocap_4 --modalities atv \
  --use_hierarchical_fusion \
  --use_ulgm \
  --unimodal_delay_epochs 10 \
  --unimodal_warmup_epochs 15 \
  --unimodal_loss_weight 0.002
  # 不加--use_rppg
```

**启用rPPG但无真实数据**（自动跳过）：
```bash
python train.py \
  --dataset iemocap_4 --modalities atv \
  --use_hierarchical_fusion \
  --use_rppg --rppg_raw_dim 64 --rppg_proj_dim 460 \
  --use_ulgm \
  --unimodal_delay_epochs 10 \
  --unimodal_warmup_epochs 15 \
  --unimodal_loss_weight 0.002
```

**效果**：
- rPPG会被质量检测器自动标记为无效
- 完全跳过rPPG分支
- 等效于未启用rPPG的训练
- **Happy F1恢复到0.83左右**

### 场景2：数据集有真实rPPG（未来）

**启用rPPG**：
```bash
python train.py \
  --dataset iemocap_4 --modalities atv \
  --use_hierarchical_fusion \
  --use_rppg --rppg_raw_dim 64 --rppg_proj_dim 460 \
  --use_ulgm \
  --unimodal_delay_epochs 10 \
  --unimodal_warmup_epochs 15 \
  --unimodal_loss_weight 0.002
```

**效果**：
- 真实rPPG通过质量检测
- 正常参与融合
- **预期Happy F1提升2-5%**（生理信号的补充作用）

### 场景3：部分样本有rPPG

**混合场景**：
- 有rPPG的样本：正常融合（4模态）
- 无rPPG的样本：自动跳过rPPG（3模态）
- 训练过程自动适应

**效果**：
- ✅ 最大化利用可用数据
- ✅ 避免零向量污染

## 预期效果对比

| 场景 | Happy F1 (前10轮) | Happy F1 (最终) | 整体F1 |
|------|-------------------|-----------------|--------|
| **原始（无rPPG）** | 0.3-0.5 | 0.83 | 0.85 |
| **旧方案（零向量rPPG）** | 0.0-0.1 | 0.73 | 0.83 |
| **新方案（质量检测）** | 0.3-0.5 | **0.83** | **0.85** |
| **真实rPPG（未来）** | 0.4-0.6 | **0.85-0.88** | **0.87-0.90** |

## 技术细节

### 为什么零向量经过投影后不是零？

```python
# 零向量
rppg = torch.zeros(64)

# 经过线性投影
R = nn.Linear(64, 460)(rppg)
# R ≠ 0，因为有bias项！

# 解决方案：检测原始rPPG，而非投影后
if torch.abs(rppg).max() < 1e-6:
    rppg = None  # 标记为无效
```

### 为什么不直接关闭--use_rppg？

**原因**：
1. 保持代码统一性（同一套代码处理有/无rPPG）
2. 支持混合场景（部分样本有rPPG）
3. 方便未来集成真实rPPG提取器

### 为什么ULGM不使用rPPG？

**原理差异**：
- **A/T/V**：情感表达信号
  - 面部表情 → Ekman基本情感理论
  - 语音韵律 → 情感声学特征
  - 文本语义 → 情感词汇、句法

- **rPPG**：生理唤醒信号
  - 心率变化 → 交感神经激活
  - 血容量波动 → 血管反应
  - 不直接对应情感类别

**问题**：
- Happy vs Excited：表达不同，但生理相似（高心率）
- Sad vs Calm：表达不同，但生理相似（低心率）
- 用生理信号生成面部/语音伪标签 → 语义不匹配

**解决方案**：
- ULGM只用A/T/V（表达层）
- rPPG只在全局融合中贡献（生理层）
- 层次分离，各司其职

## 调试建议

### 检查rPPG是否被跳过

在`Dataset.py`中添加日志：
```python
if rppg_feat is None and self.use_rppg:
    if idx == 0:  # 只打印第一个utterance
        print(f"[Dataset] rPPG skipped: sample {i}, utterance {idx}")
```

### 检查融合维度

在`forward`中添加：
```python
print(f"[Fusion] actual_dim={actual_dim}, use_rppg_this_sample={use_rppg_this_sample}")
```

### 检查Happy F1恢复情况

训练10轮后：
```bash
# 预期：Happy F1 > 0.3
# 如果仍 < 0.1，说明修复未生效
```

## 常见问题

**Q1: 修复后Happy F1还是低怎么办？**
- 检查是否有其他模块也使用了零向量rPPG
- 检查门控正则权重是否过大（`--gate_reg_weight 0`）
- 检查ULGM权重是否过早引入（增加`--unimodal_delay_epochs`）

**Q2: 如何验证rPPG质量检测生效？**
- 在Dataset.py第75行加断点
- 检查`rppg_feat`是否为None
- 如果所有样本都是None，说明检测生效

**Q3: 未来有真实rPPG时需要改代码吗？**
- 不需要！质量检测会自动识别有效rPPG
- 只需确保数据集中`s.rppg`或`s.rppg_features`有真实值

**Q4: 能否为rPPG添加更多质量指标？**
- 可以！在`RPPGQualityChecker`中添加：
  - 频率域分析（0.7-4 Hz是否有峰值）
  - 信噪比计算
  - 时域连续性检测

## 总结

**核心修复**：
1. ✅ Dataset.py：质量检测 → 零向量→None
2. ✅ Fusion.forward：动态跳过 → None→跳过分支
3. ✅ ULGM：完全排除rPPG → 保持伪标签纯净

**预期结果**：
- Happy F1从0.73恢复到0.83
- 前10轮正常学习（0.3-0.5）
- 整体F1恢复到0.85

**未来扩展**：
- 集成真实rPPG提取器（PhysNet/EfficientPhys）
- 添加自适应权重模块
- 支持视频实时rPPG提取

**关键洞察**：
> 零向量不是"没有信息"，而是"错误的信息"。经过神经网络投影后，零向量会产生非零的伪特征，污染多模态融合。正确的做法是**检测并跳过**，而非填充零向量。


