"""
层次化动态门控机制模块
包含：
1. 内层：话语级多模态门控模块
2. 外层：对话级时序门控模块
3. 层次融合模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalAttentionGate(nn.Module):
    """
    模态间注意力门控
    计算每个模态的重要性权重
    """
    def __init__(self, modal_dims):
        """
        Args:
            modal_dims: dict, 各模态的维度 {'a': dim_a, 't': dim_t, 'v': dim_v}
        """
        super(ModalAttentionGate, self).__init__()
        self.modal_dims = modal_dims
        total_dim = sum(modal_dims.values())
        
        # 模态权重计算网络
        self.attention_net = nn.Sequential(
            nn.Linear(total_dim, total_dim // 2),
            nn.ReLU(),
            nn.Linear(total_dim // 2, len(modal_dims)),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, a=None, t=None, v=None):
        """
        计算模态权重
        Args:
            a: 音频特征 [batch, dim_a] 或 None
            t: 文本特征 [batch, dim_t] 或 None
            v: 视觉特征 [batch, dim_v] 或 None
        Returns:
            weights: 模态权重 [batch, num_modals]
        """
        modals = []
        if a is not None:
            modals.append(a)
        if t is not None:
            modals.append(t)
        if v is not None:
            modals.append(v)
        
        # 拼接所有模态
        concat_features = torch.cat(modals, dim=-1)
        # 计算注意力权重
        weights = self.attention_net(concat_features)
        return weights


class FusionGate(nn.Module):
    """
    多模态融合门控
    使用门控机制控制模态间的信息流
    """
    def __init__(self, modal_dims, hidden_dim):
        """
        Args:
            modal_dims: dict, 各模态的维度
            hidden_dim: 输出隐藏维度
        """
        super(FusionGate, self).__init__()
        self.modal_dims = modal_dims
        total_dim = sum(modal_dims.values())
        self.hidden_dim = hidden_dim
        
        # 门控网络
        self.gate_net = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, len(modal_dims)),
            nn.Sigmoid()
        )
        
        # 特征投影层
        self.projection_layers = nn.ModuleDict()
        for modal_name, dim in modal_dims.items():
            self.projection_layers[modal_name] = nn.Linear(dim, hidden_dim)
        
        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * len(modal_dims), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, a=None, t=None, v=None, modal_weights=None):
        """
        门控融合多模态特征
        Args:
            a, t, v: 各模态特征
            modal_weights: 模态权重（可选）
        Returns:
            fused: 融合后的特征 [batch, hidden_dim]
        """
        modals = {}
        if a is not None:
            modals['a'] = a
        if t is not None:
            modals['t'] = t
        if v is not None:
            modals['v'] = v
        
        # 投影到统一维度
        projected = []
        for modal_name, feature in modals.items():
            proj = self.projection_layers[modal_name](feature)
            projected.append(proj)
        
        # 拼接所有投影特征
        concat_proj = torch.cat(projected, dim=-1)
        
        # 计算门控权重
        gate_weights = self.gate_net(concat_proj)
        
        # 如果提供了模态权重，则结合使用
        if modal_weights is not None:
            gate_weights = gate_weights * modal_weights
        
        # 加权融合
        weighted_features = []
        for i, (modal_name, feature) in enumerate(modals.items()):
            proj = self.projection_layers[modal_name](feature)
            weighted = gate_weights[:, i:i+1] * proj
            weighted_features.append(weighted)
        
        # 拼接并融合
        weighted_concat = torch.cat(weighted_features, dim=-1)
        fused = self.fusion_layer(weighted_concat)
        
        return fused


class FeatureEnhancement(nn.Module):
    """
    特征增强模块
    对融合后的特征进行增强
    """
    def __init__(self, hidden_dim):
        super(FeatureEnhancement, self).__init__()
        self.enhancement = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(self, x):
        # 残差连接
        enhanced = self.enhancement(x) + x
        return enhanced


class UtteranceLevelGating(nn.Module):
    """
    内层：话语级多模态门控模块
    处理单话语内的多模态融合
    """
    def __init__(self, modal_dims, hidden_dim):
        """
        Args:
            modal_dims: dict, 各模态的维度 {'a': dim_a, 't': dim_t, 'v': dim_v}
            hidden_dim: 输出隐藏维度
        """
        super(UtteranceLevelGating, self).__init__()
        self.modal_attention = ModalAttentionGate(modal_dims)
        self.fusion_gate = FusionGate(modal_dims, hidden_dim)
        self.enhancement = FeatureEnhancement(hidden_dim)
        self.hidden_dim = hidden_dim
    
    def forward(self, a=None, t=None, v=None):
        """
        处理单话语的多模态融合
        Args:
            a: 音频特征 [batch, dim_a] 或 None
            t: 文本特征 [batch, dim_t] 或 None
            v: 视觉特征 [batch, dim_v] 或 None
        Returns:
            utterance_rep: 话语级表示 [batch, hidden_dim]
        """
        # 1. 计算模态权重
        modal_weights = self.modal_attention(a, t, v)
        
        # 2. 门控融合
        fused = self.fusion_gate(a, t, v, modal_weights)
        
        # 3. 特征增强
        enhanced = self.enhancement(fused)
        
        return enhanced


class TemporalMemory(nn.Module):
    """
    时序记忆单元
    使用LSTM/GRU/Transformer处理时序依赖
    """
    def __init__(self, input_dim, hidden_dim, args):
        super(TemporalMemory, self).__init__()
        self.args = args
        self.hidden_dim = hidden_dim
        
        if args.temporal_rnn == "lstm":
            self.memory = nn.LSTM(
                input_dim,
                hidden_dim // 2,
                num_layers=args.temporal_nlayer,
                dropout=args.drop_rate if args.temporal_nlayer > 1 else 0,
                bidirectional=True,
                batch_first=True
            )
        elif args.temporal_rnn == "gru":
            self.memory = nn.GRU(
                input_dim,
                hidden_dim // 2,
                num_layers=args.temporal_nlayer,
                dropout=args.drop_rate if args.temporal_nlayer > 1 else 0,
                bidirectional=True,
                batch_first=True
            )
        elif args.temporal_rnn == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=args.temporal_nhead,
                dim_feedforward=hidden_dim,
                dropout=args.drop_rate,
                batch_first=True
            )
            self.memory = nn.TransformerEncoder(
                encoder_layer,
                num_layers=args.temporal_nlayer
            )
            self.memory_proj = nn.Linear(input_dim, hidden_dim)
        else:
            raise ValueError(f"Unsupported temporal_rnn: {args.temporal_rnn}")
    
    def forward(self, utterance_features, lengths=None):
        """
        处理时序依赖
        Args:
            utterance_features: [batch, seq_len, input_dim]
            lengths: 序列长度 [batch]
        Returns:
            memory_state: [batch, seq_len, hidden_dim]
        """
        if self.args.temporal_rnn == "transformer":
            # Transformer不需要pack
            memory_out = self.memory(utterance_features)
            memory_state = self.memory_proj(memory_out)
        else:
            # LSTM/GRU需要pack
            if lengths is not None:
                from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
                packed = pack_padded_sequence(
                    utterance_features, lengths.cpu(), batch_first=True, enforce_sorted=False
                )
                memory_out, _ = self.memory(packed)
                memory_state, _ = pad_packed_sequence(memory_out, batch_first=True)
            else:
                memory_out, _ = self.memory(utterance_features)
                memory_state = memory_out
        
        return memory_state


class TemporalGate(nn.Module):
    """
    动态时序门控
    控制情感状态在时序中的传播
    """
    def __init__(self, hidden_dim, speaker_dim=0, position_dim=0):
        super(TemporalGate, self).__init__()
        self.hidden_dim = hidden_dim
        self.speaker_dim = speaker_dim
        self.position_dim = position_dim
        
        # 计算输入维度
        input_dim = hidden_dim * 2 + speaker_dim + position_dim
        
        # 遗忘门：决定遗忘多少历史信息
        self.forget_gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # 输入门：决定接受多少新信息
        self.input_gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # 输出门：决定输出多少信息
        self.output_gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # 候选值
        self.candidate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        )
    
    def forward(self, current_state, previous_state=None, speaker_info=None, position_info=None):
        """
        动态时序门控
        Args:
            current_state: 当前状态 [batch, hidden_dim]
            previous_state: 前一状态 [batch, hidden_dim] 或 None
            speaker_info: 说话人信息 [batch, speaker_dim] 或 None
            position_info: 位置信息 [batch, position_dim] 或 None
        Returns:
            gated_state: 门控后的状态 [batch, hidden_dim]
        """
        if previous_state is None:
            # 如果没有前一状态，直接返回当前状态
            return current_state
        
        # 拼接当前状态和前一状态
        concat_state = torch.cat([current_state, previous_state], dim=-1)
        
        # 如果有额外信息，也拼接进去
        if speaker_info is not None:
            concat_state = torch.cat([concat_state, speaker_info], dim=-1)
        if position_info is not None:
            concat_state = torch.cat([concat_state, position_info], dim=-1)
        
        # 计算门控值
        forget = self.forget_gate(concat_state)
        input_gate = self.input_gate(concat_state)
        output = self.output_gate(concat_state)
        candidate = self.candidate(concat_state)
        
        # 更新状态
        new_state = forget * previous_state + input_gate * candidate
        gated_state = output * torch.tanh(new_state)
        
        return gated_state


class EmotionPropagation(nn.Module):
    """
    情感状态传播控制
    控制情感在对话中的传播强度
    """
    def __init__(self, hidden_dim):
        super(EmotionPropagation, self).__init__()
        self.propagation_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, state):
        """
        情感状态传播
        Args:
            state: 当前情感状态 [batch, hidden_dim]
        Returns:
            propagated_state: 传播后的状态 [batch, hidden_dim]
        """
        # 残差连接
        propagated = self.propagation_net(state) + state
        return propagated


class DialogueLevelGating(nn.Module):
    """
    外层：对话级时序门控模块
    处理对话中的时序依赖
    """
    def __init__(self, input_dim, hidden_dim, args, speaker_dim=16, position_dim=16):
        """
        Args:
            input_dim: 输入维度（话语级特征的维度）
            hidden_dim: 隐藏维度
            args: 配置参数
            speaker_dim: 说话人信息维度
            position_dim: 位置信息维度
        """
        super(DialogueLevelGating, self).__init__()
        self.temporal_memory = TemporalMemory(input_dim, hidden_dim, args)
        self.temporal_gate = TemporalGate(hidden_dim, speaker_dim=speaker_dim, position_dim=position_dim)
        self.emotion_propagation = EmotionPropagation(hidden_dim)
        self.hidden_dim = hidden_dim
        self.speaker_dim = speaker_dim
        self.position_dim = position_dim
    
    def forward(self, utterance_features, lengths=None, speaker_info=None, position_info=None):
        """
        处理对话级时序依赖
        Args:
            utterance_features: 话语级特征 [batch, seq_len, input_dim]
            lengths: 序列长度 [batch]
            speaker_info: 说话人信息 [batch, seq_len, speaker_dim] 或 None
            position_info: 位置信息 [batch, seq_len, position_dim] 或 None
        Returns:
            dialogue_state: 对话级状态 [batch, seq_len, hidden_dim]
        """
        # 1. 时序记忆更新
        memory_state = self.temporal_memory(utterance_features, lengths)
        
        # 2. 逐时间步应用动态门控
        batch_size, seq_len, hidden_dim = memory_state.shape
        gated_states = []
        previous_state = None
        
        for t in range(seq_len):
            current_state = memory_state[:, t, :]  # [batch, hidden_dim]
            
            # 获取当前时间步的额外信息
            curr_speaker = speaker_info[:, t, :] if speaker_info is not None else None
            curr_position = position_info[:, t, :] if position_info is not None else None
            
            # 动态门控
            gated_state = self.temporal_gate(
                current_state, previous_state, curr_speaker, curr_position
            )
            
            # 情感传播
            propagated_state = self.emotion_propagation(gated_state)
            
            gated_states.append(propagated_state)
            previous_state = propagated_state
        
        # 堆叠所有时间步
        dialogue_state = torch.stack(gated_states, dim=1)  # [batch, seq_len, hidden_dim]
        
        return dialogue_state


class HierarchicalFusion(nn.Module):
    """
    层次融合模块
    融合话语级和对话级特征
    """
    def __init__(self, utterance_dim, dialogue_dim, output_dim):
        """
        Args:
            utterance_dim: 话语级特征维度
            dialogue_dim: 对话级特征维度
            output_dim: 输出维度
        """
        super(HierarchicalFusion, self).__init__()
        self.fusion_gate = nn.Sequential(
            nn.Linear(utterance_dim + dialogue_dim, output_dim),
            nn.Sigmoid()
        )
        self.fusion_layer = nn.Sequential(
            nn.Linear(utterance_dim + dialogue_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, h_utterance, h_dialogue):
        """
        融合话语级和对话级特征
        Args:
            h_utterance: 话语级特征 [batch, seq_len, utterance_dim]
            h_dialogue: 对话级特征 [batch, seq_len, dialogue_dim]
        Returns:
            fused: 融合后的特征 [batch, seq_len, output_dim]
        """
        # 拼接特征
        concat_features = torch.cat([h_utterance, h_dialogue], dim=-1)
        
        # 计算融合权重
        gate = self.fusion_gate(concat_features)  # [batch, seq_len, output_dim]
        
        # 投影到统一维度（如果需要）
        if h_utterance.shape[-1] != gate.shape[-1]:
            h_utterance_proj = nn.Linear(h_utterance.shape[-1], gate.shape[-1]).to(h_utterance.device)(h_utterance)
        else:
            h_utterance_proj = h_utterance
        
        if h_dialogue.shape[-1] != gate.shape[-1]:
            h_dialogue_proj = nn.Linear(h_dialogue.shape[-1], gate.shape[-1]).to(h_dialogue.device)(h_dialogue)
        else:
            h_dialogue_proj = h_dialogue
        
        # 门控融合
        fused = gate * h_utterance_proj + (1 - gate) * h_dialogue_proj
        
        # 最终融合层
        output = self.fusion_layer(concat_features)
        
        # 残差连接
        if output.shape[-1] == fused.shape[-1]:
            output = output + fused
        
        return output

