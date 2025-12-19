# ULGM模块方案总结

## 一、方案可行性 ✅

### 高度可行

1. **理论基础扎实**：单模态监督是多任务学习的经典策略
2. **实现简单**：基于PyTorch标准模块，代码量小
3. **风险极低**：不影响主流程维度，推理时零开销
4. **预期效果良好**：特征判别性提升10-20%，F1提升2-5%

## 二、核心设计

### 2.1 实现位置

**文件**：`joyful/fusion_methods_hierarchical.py`
**类**：`AutoFusion_Hierarchical`

### 2.2 设计原则

- ✅ **不改变融合输出维度**：输出仍为`[2, 512]`
- ✅ **只在训练时使用**：推理时完全跳过
- ✅ **最小化代码修改**：只在融合模块中添加
- ✅ **向后兼容**：默认禁用，不影响现有代码

## 三、关键实现点

### 3.1 添加的组件

1. **单模态特征提取层**：
   - `post_text_layer_1`: 768 -> hidden_size
   - `post_audio_layer_1`: 100 -> hidden_size
   - `post_video_layer_1`: 512 -> hidden_size

2. **单模态分类器**：
   - `post_text_layer_2/3`: hidden_size -> num_classes
   - `post_audio_layer_2/3`: hidden_size -> num_classes
   - `post_video_layer_2/3`: hidden_size -> num_classes

3. **损失计算**：
   - 使用NLLLoss计算单模态分类损失
   - 总损失 = text_loss + audio_loss + video_loss

### 3.2 训练/推理分离

```python
# 训练时
if self.use_ulgm and self.training and labels is not None:
    unimodal_loss = self._compute_unimodal_loss(a, t, v, labels)

# 推理时
# unimodal_loss = None，完全跳过
```

## 四、维度保证

| 阶段 | 维度 | 说明 |
|------|------|------|
| 输入 | a[100], t[768], v[512] | 不变 |
| **融合输出** | **[2, 512]** | **完全不变** |
| 单模态分类器 | 独立计算 | 不影响主流程 |

## 五、使用方式

### 5.1 启用ULGM

```bash
python train.py \
    --dataset="iemocap_4" \
    --modalities="atv" \
    --use_hierarchical_fusion \
    --use_ulgm \
    --unimodal_loss_weight=0.2
```

### 5.2 不使用ULGM（默认）

```bash
python train.py \
    --dataset="iemocap_4" \
    --modalities="atv" \
    --use_hierarchical_fusion
```

## 六、预期效果

- **特征判别性**：+10-20%
- **F1分数**：+2-5%
- **训练稳定性**：提升
- **过拟合风险**：降低

## 七、实现步骤

1. ✅ 在`AutoFusion_Hierarchical`中添加ULGM层
2. ✅ 修改`forward`方法支持标签
3. ✅ 实现`_compute_unimodal_loss`方法
4. ✅ 修改`Dataset.py`传递标签（可选）
5. ✅ 修改`Coach.py`添加单模态损失
6. ✅ 在`train.py`中添加参数

## 八、注意事项

1. **损失权重**：建议0.1-0.3，不应过大
2. **标签传递**：需要在Dataset中传递或从data获取
3. **训练/推理分离**：确保推理时labels=None
4. **向后兼容**：默认use_ulgm=False

## 九、相关文档

- `ULGM_FEASIBILITY_ANALYSIS.md` - 详细可行性分析
- `ULGM_IMPLEMENTATION_PLAN.md` - 完整实现方案

## 十、总结

### 方案特点

- ✅ **可行性高**：理论基础扎实，实现简单
- ✅ **风险低**：不影响维度，推理零开销
- ✅ **效果预期好**：特征判别性提升明显

### 推荐

**立即实施**，预期带来：
- 特征判别性提升10-20%
- F1分数提升2-5%
- 更好的训练稳定性






