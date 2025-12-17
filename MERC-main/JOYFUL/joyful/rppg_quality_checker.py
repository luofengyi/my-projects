"""
rPPG信号质量检测模块
用于判断rPPG特征是否有效，避免零向量或低质量信号污染多模态融合
"""

import torch
import torch.nn as nn


class RPPGQualityChecker:
    """
    rPPG信号质量检测器（非参数模块）
    
    功能：
    1. 检测是否为零向量或常数向量
    2. 计算信号方差（低方差=低质量）
    3. 返回质量分数和是否应该使用该rPPG特征
    """
    
    def __init__(self, 
                 zero_threshold=1e-6, 
                 variance_threshold=1e-4,
                 use_adaptive_weight=True):
        """
        Args:
            zero_threshold: 判定为零向量的阈值
            variance_threshold: 最小方差阈值（低于此值认为是常数/低质量）
            use_adaptive_weight: 是否使用自适应权重（基于质量分数）
        """
        self.zero_threshold = zero_threshold
        self.variance_threshold = variance_threshold
        self.use_adaptive_weight = use_adaptive_weight
        
    def check_quality(self, rppg_features):
        """
        检测rPPG特征质量
        
        Args:
            rppg_features: rPPG特征向量 [feature_dim]
        
        Returns:
            is_valid: bool，是否应该使用该rPPG
            quality_score: float（0-1），质量分数（可用于加权）
            reason: str，质量判断原因
        """
        if rppg_features is None:
            return False, 0.0, "rPPG is None"
        
        # 检查是否为零向量
        abs_max = torch.abs(rppg_features).max().item()
        if abs_max < self.zero_threshold:
            return False, 0.0, "Zero vector detected"
        
        # 计算方差（低方差说明是常数或噪声）
        variance = torch.var(rppg_features).item()
        if variance < self.variance_threshold:
            return False, 0.1, f"Low variance ({variance:.6f})"
        
        # 计算质量分数（基于方差，归一化到0-1）
        # 使用sigmoid映射：variance越大，质量越高
        quality_score = torch.sigmoid(
            torch.tensor((variance - self.variance_threshold) * 100)
        ).item()
        quality_score = max(0.0, min(1.0, quality_score))
        
        # 判断是否有效（质量分数>0.3认为可用）
        is_valid = quality_score > 0.3
        reason = f"Valid (quality={quality_score:.3f})" if is_valid else f"Low quality ({quality_score:.3f})"
        
        return is_valid, quality_score, reason
    
    def get_adaptive_weight(self, quality_score):
        """
        根据质量分数计算自适应权重
        
        Args:
            quality_score: 质量分数（0-1）
        
        Returns:
            weight: 权重系数（0-1）
        """
        if not self.use_adaptive_weight:
            return 1.0 if quality_score > 0.3 else 0.0
        
        # 平滑权重：quality_score^2，避免低质量信号过度影响
        return quality_score ** 2


class RPPGAdaptiveWeightModule(nn.Module):
    """
    可学习的rPPG自适应权重模块
    根据rPPG特征和其他模态特征，学习rPPG的最优权重
    """
    
    def __init__(self, rppg_dim=64, context_dim=512):
        """
        Args:
            rppg_dim: rPPG原始特征维度
            context_dim: 上下文特征维度（来自其他模态的融合）
        """
        super().__init__()
        self.rppg_dim = rppg_dim
        self.context_dim = context_dim
        
        # 权重预测网络：输入rPPG+上下文，输出权重（0-1）
        self.weight_predictor = nn.Sequential(
            nn.Linear(rppg_dim + context_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 输出0-1权重
        )
        
        # 初始化：默认输出0.5（中等权重）
        with torch.no_grad():
            self.weight_predictor[-2].bias.fill_(0.0)
    
    def forward(self, rppg_features, context_features):
        """
        预测rPPG的自适应权重
        
        Args:
            rppg_features: rPPG特征 [rppg_dim]
            context_features: 其他模态的融合特征 [context_dim]
        
        Returns:
            weight: 标量权重（0-1）
        """
        # 拼接rPPG和上下文
        combined = torch.cat([rppg_features, context_features], dim=-1)
        
        # 预测权重
        weight = self.weight_predictor(combined)
        
        return weight.squeeze()


