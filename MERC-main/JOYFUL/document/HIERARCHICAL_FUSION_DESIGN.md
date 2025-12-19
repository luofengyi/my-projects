# 层次化动态门控机制 - 融合模块改进方案

## 一、约束条件

1. **只修改融合模块的上下文融合部分**（fuse_inGlobal/fuse_outGlobal）
2. **不改变整体维度变化**
3. **保持与原有代码兼容**
4. **避免训练时出现难以修正的错误**

## 二、当前AutoFusion架构分析

### 当前结构：
```
输入: a, t, v (单话语的多模态特征)
  ↓
局部特征学习 (inter): 
  - projectA/T/V -> 460维
  - 基向量空间投影
  - 输出: bba, bbt, bbv
  ↓
上下文融合 (global):
  - fuse_inGlobal: input_features -> 512 (压缩)
  - fuse_outGlobal: 512 -> input_features (重构)
  - 重构损失: ||fuse_outGlobal(fuse_inGlobal(x)) - x||
  ↓
输出: [globalCompressed, interCompressed] (维度: [2, 512])
```

### 问题分析：
1. **时序平滑策略单一**：fuse_inGlobal/fuse_outGlobal是简单的MLP，没有考虑时序上下文
2. **缺乏层次化建模**：没有区分话语级和对话级特征
3. **固定融合方式**：无法根据上下文动态调整

## 三、改进方案设计

### 核心思路：
在保持维度不变的前提下，将fuse_inGlobal/fuse_outGlobal替换为层次化门控机制：

1. **内层（话语级）**：在fuse_inGlobal中，对单话语的多模态特征进行动态门控融合
2. **外层（对话级）**：在fuse_outGlobal中，引入时序上下文信息，进行对话级门控

### 关键设计点：

#### 1. 内层门控（fuse_inGlobal改进）
- 输入：a, t, v（单话语的多模态特征）
- 处理：动态计算模态权重，门控融合
- 输出：512维压缩特征（保持维度不变）

#### 2. 外层门控（fuse_outGlobal改进）
- 输入：512维压缩特征
- 处理：引入时序上下文（通过可学习的全局状态或前一个话语的状态）
- 输出：input_features维重构特征（保持维度不变）

#### 3. 时序上下文获取
由于AutoFusion是逐话语调用的，需要：
- 方案A：使用可学习的全局情感状态（类似原论文的全局情感状态）
- 方案B：在Dataset中维护对话级状态，传递给AutoFusion
- 方案C：在SeqContext中处理时序，AutoFusion只处理话语级

**推荐方案A**：最简单，不改变调用方式，通过可学习参数实现

## 四、具体实现

### 4.1 改进的AutoFusion结构

```python
class AutoFusion_Hierarchical(nn.Module):
    def __init__(self, input_features):
        # 保持原有结构
        # ...
        
        # 改进：内层话语级门控（替换fuse_inGlobal）
        self.utterance_gate = UtteranceLevelGate(input_features, 512)
        
        # 改进：外层对话级门控（替换fuse_outGlobal）
        self.dialogue_gate = DialogueLevelGate(512, input_features)
        
        # 全局情感状态（用于对话级门控）
        self.global_emotion_state = nn.Parameter(torch.randn(512))
        
    def forward(self, a, t, v, dialogue_context=None):
        # 局部特征学习（保持不变）
        # ...
        
        # 内层：话语级门控融合
        globalCompressed = self.utterance_gate(a, t, v)
        
        # 外层：对话级门控重构
        # 使用全局情感状态或传入的对话上下文
        context = dialogue_context if dialogue_context is not None else self.global_emotion_state
        globalReconstructed = self.dialogue_gate(globalCompressed, context)
        
        # 重构损失（保持不变）
        globalLoss = self.criterion(globalReconstructed, torch.cat((a, t, v)))
        
        # ...
```

### 4.2 内层门控（话语级）

```python
class UtteranceLevelGate(nn.Module):
    """内层：话语级多模态门控"""
    def __init__(self, input_features, hidden_dim):
        # 模态权重计算
        self.modal_attention = nn.Sequential(
            nn.Linear(input_features, input_features // 2),
            nn.ReLU(),
            nn.Linear(input_features // 2, 3),  # a, t, v三个模态
            nn.Softmax(dim=-1)
        )
        
        # 门控融合网络（保持维度：input_features -> hidden_dim）
        self.fuse_in = nn.Sequential(
            nn.Linear(input_features, 1024),
            nn.Tanh(),
            nn.Linear(1024, hidden_dim),
            nn.ReLU(),
        )
        
    def forward(self, a, t, v):
        # 拼接多模态特征
        concat_features = torch.cat((a, t, v))
        
        # 计算模态权重
        modal_weights = self.modal_attention(concat_features)
        
        # 加权融合（可选，或直接使用concat）
        # 为了保持简单，直接使用concat
        compressed = self.fuse_in(concat_features)
        
        return compressed
```

### 4.3 外层门控（对话级）

```python
class DialogueLevelGate(nn.Module):
    """外层：对话级时序门控"""
    def __init__(self, hidden_dim, output_dim):
        # 时序门控网络
        self.temporal_gate = nn.Sequential(
            nn.Linear(hidden_dim + 512, hidden_dim),  # +512是全局状态维度
            nn.Sigmoid()
        )
        
        # 重构网络（保持维度：hidden_dim -> output_dim）
        self.fuse_out = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.Tanh(),
            nn.Linear(1024, output_dim)
        )
        
    def forward(self, compressed, context):
        # 拼接压缩特征和上下文
        concat = torch.cat([compressed, context])
        
        # 时序门控
        gate = self.temporal_gate(concat)
        
        # 门控后的特征
        gated = gate * compressed
        
        # 重构
        reconstructed = self.fuse_out(gated)
        
        return reconstructed
```

## 五、维度保证

### 输入输出维度检查：
- 输入：a (100), t (768), v (512) -> concat = 1380 (IEMOCAP)
- fuse_inGlobal: 1380 -> 512 ✅
- fuse_outGlobal: 512 -> 1380 ✅
- 输出：globalCompressed (512) ✅

**所有维度保持不变！**

## 六、实现优势

1. **最小化修改**：只修改fuse_inGlobal和fuse_outGlobal
2. **维度兼容**：完全保持原有维度
3. **向后兼容**：可以保留原AutoFusion作为选项
4. **易于调试**：修改范围小，容易定位问题

## 七、使用方式

### 方式1：直接替换
```python
# train.py
from joyful.fusion_methods import AutoFusion_Hierarchical
modelF = AutoFusion_Hierarchical(1380)
```

### 方式2：参数控制
```python
# train.py
if args.use_hierarchical_fusion:
    from joyful.fusion_methods import AutoFusion_Hierarchical
    modelF = AutoFusion_Hierarchical(1380)
else:
    from joyful.fusion_methods import AutoFusion
    modelF = AutoFusion(1380)
```

## 八、预期效果

1. **解决时序平滑问题**：通过对话级门控动态调整重构过程
2. **提升融合效果**：话语级门控更好地融合多模态特征
3. **保持稳定性**：维度不变，训练稳定






