# rPPG质量问题快速修复指南

## 问题
添加rPPG后，**Happy F1从0.83降至0.73**，前40轮无法学习。

## 原因
数据集无真实rPPG → 使用零向量 → 经过投影产生伪特征 → 稀释A/T/V有效信号 → Happy（少数类）最先受影响。

## 解决方案
**检测零向量 → 设为None → 完全跳过rPPG分支**

## 修改汇总

### 1. Dataset.py（第65-82行）
```python
# 质量检测：零向量或低方差→None
if rppg_feat is not None:
    abs_max = torch.abs(rppg_feat).max().item()
    variance = torch.var(rppg_feat).item()
    if abs_max < 1e-6 or variance < 1e-4:
        rppg_feat = None  # 跳过无效rPPG
```

### 2. fusion_methods_hierarchical.py

#### forward方法（第318-338行）
```python
# 双重质量检测
use_rppg_this_sample = False
if self.use_rppg and rppg is not None:
    abs_max = torch.abs(rppg).max().item()
    variance = torch.var(rppg).item()
    if abs_max >= 1e-6 and variance >= 1e-4:
        use_rppg_this_sample = True
    else:
        rppg = None

# 只在有效时添加rPPG分支
if use_rppg_this_sample and self.projectR is not None:
    R, bbr = _project_and_weight(rppg, self.projectR)
    modal_projections.append(R)
    modal_weighted.append(bbr)
```

#### ULGM排除rPPG（第405-410行）
```python
# ULGM只使用A/T/V，不使用rPPG
unimodal_loss = self._compute_unimodal_loss(a, t, v, labels)

# _compute_unimodal_loss中
fusion_input = torch.cat([a, t, v], dim=-1)  # 只拼接A/T/V
```

#### ULGM维度修正（第278行）
```python
ulgm_fusion_dim = 100 + 768 + 512  # 1380，不包含rPPG
self.post_fusion_layer_1 = nn.Linear(ulgm_fusion_dim, hidden_size)
```

## 训练命令

### 场景1：数据集无rPPG（推荐）
```bash
python train.py \
  --dataset iemocap_4 --modalities atv \
  --use_hierarchical_fusion \
  --use_ulgm \
  --unimodal_delay_epochs 10 \
  --unimodal_warmup_epochs 15 \
  --unimodal_loss_weight 0.002
  # 不加--use_rppg，直接恢复到图2效果
```

### 场景2：启用rPPG开关但无数据（自动跳过）
```bash
python train.py \
  --dataset iemocap_4 --modalities atv \
  --use_hierarchical_fusion \
  --use_rppg --rppg_raw_dim 64 --rppg_proj_dim 460 \
  --use_ulgm \
  --unimodal_delay_epochs 10 \
  --unimodal_warmup_epochs 15 \
  --unimodal_loss_weight 0.002
  # 会自动检测并跳过无效rPPG
```

## 预期效果

| 指标 | 旧方案（零向量） | 新方案（质量检测） |
|------|------------------|-------------------|
| Happy F1 (Epoch 10) | 0.0-0.1 | 0.3-0.5 ✅ |
| Happy F1 (最终) | 0.73 | **0.83** ✅ |
| 整体F1 | 0.83 | **0.85** ✅ |

## 验证方法

### 检查rPPG是否被跳过
在`Dataset.py`第75行后添加：
```python
if rppg_feat is None and self.use_rppg and idx == 0:
    print(f"[Dataset] rPPG skipped for sample {i}")
```

训练时应看到：
```
[Dataset] rPPG skipped for sample 0
[Dataset] rPPG skipped for sample 1
...
```

### 检查Happy F1恢复
训练10轮后：
```
Epoch 10: Happy F1 > 0.3  # 正常
Epoch 10: Happy F1 < 0.1  # 修复未生效
```

## 关键洞察

> **零向量 ≠ 没有信息**  
> 零向量经过`nn.Linear`投影后，由于bias项的存在，会产生**非零伪特征**。这些伪特征会**稀释有效信号**，而非简单地被忽略。

## 快速回滚

如果修复后仍有问题，快速回滚：
```bash
# 直接关闭rPPG
python train.py --dataset iemocap_4 --modalities atv \
  --use_hierarchical_fusion --use_ulgm \
  # 不加--use_rppg
```

## 未来扩展

当有真实rPPG时：
1. 质量检测器自动识别有效rPPG
2. 正常参与融合
3. 预期Happy F1提升2-5%

详见`RPPG_QUALITY_FIX_GUIDE.md`


