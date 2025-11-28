"""
改进的AutoFusion - 集成层次化动态门控机制
只在上下文融合部分（fuse_inGlobal/fuse_outGlobal）进行修改
保持所有维度不变
"""

import torch
from torch import nn
from torch.nn import functional as F
from joyful.loss_utils import create_reconstruction_loss
import torch.nn.functional as F_nn


class UtteranceLevelGate(nn.Module):
    """
    内层：话语级多模态门控
    替换原来的fuse_inGlobal，添加动态门控机制
    """
    def __init__(self, input_features, hidden_dim):
        super(UtteranceLevelGate, self).__init__()
        self.input_features = input_features
        self.hidden_dim = hidden_dim
        
        # 模态权重计算网络（用于动态调整模态重要性）
        # 输入是拼接后的特征，输出是3个模态的权重
        self.modal_attention = nn.Sequential(
            nn.Linear(input_features, input_features // 2),
            nn.ReLU(),
            nn.Linear(input_features // 2, 3),  # a, t, v三个模态
            nn.Softmax(dim=-1)
        )
        
        # 门控网络：控制信息流
        # 使用HardSigmoid或改进的Sigmoid，避免梯度消失
        # HardSigmoid在[-3, 3]范围内是线性的，梯度更稳定
        self.gate_net = nn.Sequential(
            nn.Linear(input_features, hidden_dim),
            nn.Hardtanh(min_val=0.0, max_val=1.0)  # 替代Sigmoid，避免饱和
        )
        
        # 特征压缩网络（保持原有结构，但添加门控）
        self.fuse_in = nn.Sequential(
            nn.Linear(input_features, 1024),
            nn.Tanh(),
            nn.Linear(1024, hidden_dim),
            nn.ReLU(),
        )
        
    def forward(self, concat_features):
        """
        Args:
            concat_features: 拼接后的多模态特征 [input_features]
        Returns:
            compressed: 压缩后的特征 [hidden_dim]
        """
        # 计算模态权重（用于后续可能的加权）
        modal_weights = self.modal_attention(concat_features)
        
        # 计算门控值
        gate = self.gate_net(concat_features)
        
        # 特征压缩
        compressed = self.fuse_in(concat_features)
        
        # 应用门控（门控控制信息流，避免过度压缩）
        gated_compressed = gate * compressed
        
        return gated_compressed


class DialogueLevelGate(nn.Module):
    """
    外层：对话级时序门控
    替换原来的fuse_outGlobal，添加时序上下文门控
    用于重构压缩特征到原始维度
    """
    def __init__(self, hidden_dim, output_dim):
        super(DialogueLevelGate, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 全局情感状态维度（可学习参数）
        self.context_dim = hidden_dim
        
        # 时序门控网络
        # 输入：压缩特征 + 上下文状态
        # 使用Hardtanh替代Sigmoid，避免梯度消失
        self.temporal_gate = nn.Sequential(
            nn.Linear(hidden_dim + self.context_dim, hidden_dim),
            nn.Hardtanh(min_val=0.0, max_val=1.0)  # 替代Sigmoid
        )
        
        # 遗忘门：控制历史信息的保留
        self.forget_gate = nn.Sequential(
            nn.Linear(hidden_dim + self.context_dim, hidden_dim),
            nn.Hardtanh(min_val=0.0, max_val=1.0)  # 替代Sigmoid
        )
        
        # 输入门：控制新信息的接受
        self.input_gate = nn.Sequential(
            nn.Linear(hidden_dim + self.context_dim, hidden_dim),
            nn.Hardtanh(min_val=0.0, max_val=1.0)  # 替代Sigmoid
        )
        
        # 重构网络（保持原有结构：hidden_dim -> output_dim）
        self.fuse_out = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.Tanh(),
            nn.Linear(1024, output_dim)
        )
        
    def forward(self, compressed, context=None, return_gates=False):
        """
        Args:
            compressed: 压缩后的特征 [hidden_dim]
            context: 上下文状态（全局情感状态）[context_dim]
                    如果为None，则使用零向量
            return_gates: 是否返回门控值（用于正则化）
        Returns:
            reconstructed: 重构后的特征 [output_dim]
            gates: 门控值字典（如果return_gates=True）
        """
        # 如果没有提供上下文，使用零向量（第一次调用时）
        if context is None:
            context = torch.zeros(self.context_dim, device=compressed.device)
        
        # 确保context是1D向量
        if context.dim() > 1:
            context = context.flatten()
        if context.shape[0] != self.context_dim:
            # 如果维度不匹配，进行投影或截断
            if context.shape[0] > self.context_dim:
                context = context[:self.context_dim]
            else:
                padding = torch.zeros(self.context_dim - context.shape[0], device=context.device)
                context = torch.cat([context, padding])
        
        # 拼接压缩特征和上下文
        concat = torch.cat([compressed, context])  # [hidden_dim + context_dim]
        
        # 计算时序门控
        temporal_gate = self.temporal_gate(concat)  # [hidden_dim]
        
        # 计算遗忘门和输入门
        forget = self.forget_gate(concat)  # [hidden_dim]
        input_gate = self.input_gate(concat)  # [hidden_dim]
        
        # 门控融合：结合当前特征和上下文
        # 遗忘门控制保留多少上下文信息，输入门控制接受多少新信息
        # 这里context的前hidden_dim个元素用于融合
        context_part = context[:self.hidden_dim] if context.shape[0] >= self.hidden_dim else context
        if context_part.shape[0] < self.hidden_dim:
            padding = torch.zeros(self.hidden_dim - context_part.shape[0], device=context_part.device)
            context_part = torch.cat([context_part, padding])
        
        gated_compressed = forget * context_part + input_gate * compressed
        
        # 应用时序门控
        final_compressed = temporal_gate * gated_compressed  # [hidden_dim]
        
        # 重构到原始维度
        reconstructed = self.fuse_out(final_compressed)  # [output_dim]
        
        if return_gates:
            gates = {
                'temporal_gate': temporal_gate,
                'forget_gate': forget,
                'input_gate': input_gate
            }
            return reconstructed, gates
        else:
            return reconstructed


class AutoFusion_Hierarchical(nn.Module):
    """
    改进的AutoFusion - 集成层次化动态门控机制
    只在上下文融合部分进行修改，保持其他部分不变
    """
    def __init__(self, input_features, use_smooth_l1=False, 
                 use_ulgm=False, num_classes=4, hidden_size=128, drop_rate=0.3):
        """
        Args:
            input_features: 输入特征维度
            use_smooth_l1: 是否使用SmoothL1Loss（基础优化方案），默认False使用MSELoss
            use_ulgm: 是否启用ULGM模块（单模态监督）
            num_classes: 情感类别数量
            hidden_size: 单模态特征提取的隐藏层维度
            drop_rate: Dropout率
        """
        super(AutoFusion_Hierarchical, self).__init__()
        self.input_features = input_features
        self.use_smooth_l1 = use_smooth_l1
        self.use_ulgm = use_ulgm

        # 改进：内层话语级门控（替换原来的fuse_inGlobal）
        self.utterance_gate = UtteranceLevelGate(input_features, 512)
        
        # 改进：外层对话级门控（替换原来的fuse_outGlobal）
        self.dialogue_gate = DialogueLevelGate(512, input_features)
        
        # 全局情感状态（可学习参数，用于对话级门控）
        # 使用小标准差初始化，避免初始重构误差过大
        # 初始化为接近零的小值，让模型从接近恒等映射开始学习
        self.global_emotion_state = nn.Parameter(torch.randn(512) * 0.01)

        # 局部特征学习部分（保持不变）
        self.fuse_inInter = nn.Sequential(
            nn.Linear(input_features, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.ReLU(),
        )
        self.fuse_outInter = nn.Sequential(
            nn.Linear(512, 1024),
            nn.Tanh(),
            nn.Linear(1024, input_features)
        )

        # allow switching to SmoothL1Loss for reconstruction when requested
        self.criterion = nn.SmoothL1Loss() if use_smooth_l1 else nn.MSELoss()

        # 局部特征学习的投影层（保持不变）
        self.projectA = nn.Linear(100, 460)
        self.projectT = nn.Linear(768, 460)
        self.projectV = nn.Linear(512, 460)
        self.projectB = nn.Sequential(
            nn.Linear(460, 460),
        )
        
        # ========== ULGM模块：单模态监督（可选） ==========
        if use_ulgm:
            # 单模态特征提取层
            # 文本特征提取
            self.post_text_dropout = nn.Dropout(drop_rate)
            self.post_text_layer_1 = nn.Linear(768, hidden_size)
            
            # 音频特征提取
            self.post_audio_dropout = nn.Dropout(drop_rate)
            self.post_audio_layer_1 = nn.Linear(100, hidden_size)
            
            # 视觉特征提取
            self.post_video_dropout = nn.Dropout(drop_rate)
            self.post_video_layer_1 = nn.Linear(512, hidden_size)
            
            # 单模态分类器
            # 文本分类器
            self.post_text_layer_2 = nn.Linear(hidden_size, hidden_size)
            self.post_text_layer_3 = nn.Linear(hidden_size, num_classes)
            
            # 音频分类器
            self.post_audio_layer_2 = nn.Linear(hidden_size, hidden_size)
            self.post_audio_layer_3 = nn.Linear(hidden_size, num_classes)
            
            # 视觉分类器
            self.post_video_layer_2 = nn.Linear(hidden_size, hidden_size)
            self.post_video_layer_3 = nn.Linear(hidden_size, num_classes)
            
            # 单模态损失函数
            self.unimodal_criterion = nn.NLLLoss()

    def forward(self, a, t, v, labels=None):
        """
        Args:
            a: 音频特征 [100]
            t: 文本特征 [768]
            v: 视觉特征 [512]
            labels: 标签（可选，仅训练时需要，用于ULGM）
        Returns:
            output: 融合后的特征 [2, 512] (与原始AutoFusion保持一致)
            loss: 重构损失
            unimodal_loss: 单模态损失（仅训练时，如果启用ULGM，否则为None）
        """
        # ========== 局部特征学习部分（保持不变） ==========
        B = self.projectB(torch.ones(460, device=a.device))
        A = self.projectA(a)
        T = self.projectT(t)
        V = self.projectV(v)

        BA = torch.softmax(torch.mul((torch.unsqueeze(B, dim=1)), A), dim=1)
        BT = torch.softmax(torch.mul((torch.unsqueeze(B, dim=1)), T), dim=1)
        BV = torch.softmax(torch.mul((torch.unsqueeze(B, dim=1)), V), dim=1)

        bba = torch.mm(BA, torch.unsqueeze(A, dim=1)).squeeze(1)
        bbt = torch.mm(BT, torch.unsqueeze(T, dim=1)).squeeze(1)
        bbv = torch.mm(BV, torch.unsqueeze(V, dim=1)).squeeze(1)

        interCompressed = self.fuse_inInter(torch.cat((bba, bbt, bbv)))
        interLoss = self.criterion(self.fuse_outInter(interCompressed), torch.cat((bba, bbt, bbv)))

        # ========== 上下文融合部分（改进：添加层次化门控） ==========
        # 拼接多模态特征
        concat_features = torch.cat((a, t, v))  # [input_features]
        
        # 内层：话语级门控融合（替换原来的fuse_inGlobal）
        globalCompressed = self.utterance_gate(concat_features)  # [512]
        
        # 外层：对话级门控重构（替换原来的fuse_outGlobal）
        # 使用全局情感状态作为上下文，同时返回门控值用于正则化
        globalReconstructed, gates = self.dialogue_gate(
            globalCompressed, 
            self.global_emotion_state,
            return_gates=True
        )  # [input_features]
        
        # 重构损失（保持原有计算方式）
        globalLoss = self.criterion(globalReconstructed, concat_features)
        
        # 门控正则化：防止门控值过度饱和，增强训练稳定性
        # 鼓励门控值接近0.5（中间值），避免过度偏向0或1
        # 这有助于避免梯度消失，保持门控的灵活性
        gate_regularization = (
            torch.mean((gates['temporal_gate'] - 0.5) ** 2) +
            torch.mean((gates['forget_gate'] - 0.5) ** 2) +
            torch.mean((gates['input_gate'] - 0.5) ** 2)
        ) * 0.01  # 正则化系数，可根据需要调整

        # 总损失：重构损失 + 门控正则化
        loss = globalLoss + interLoss + gate_regularization

        # 输出格式保持与原始AutoFusion一致
        # 原始：torch.cat((globalCompressed, interCompressed), 0)
        # globalCompressed: [512], interCompressed: [512]
        # cat后: [1024]
        output = torch.cat((globalCompressed, interCompressed), 0)
        
        # ========== ULGM模块：单模态监督（仅训练时） ==========
        unimodal_loss = None
        if self.use_ulgm and self.training and labels is not None:
            unimodal_loss = self._compute_unimodal_loss(a, t, v, labels)
        
        return output, loss, unimodal_loss

