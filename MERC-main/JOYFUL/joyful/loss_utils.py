"""
损失函数工具模块
提供改进的损失函数和损失权重管理

基础优化方案：
1. SmoothL1Loss替代MSELoss（更鲁棒）
2. 基础损失权重配置（可配置权重）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class WeightedMSELoss(nn.Module):
    """
    加权MSE损失
    根据模态重要性对不同模态的损失进行加权
    """
    def __init__(self, modal_weights=None, reduction='mean'):
        """
        Args:
            modal_weights: 模态权重 [a_weight, t_weight, v_weight]
            reduction: 'mean' or 'sum'
        """
        super(WeightedMSELoss, self).__init__()
        self.modal_weights = modal_weights or [1.0, 1.0, 1.0]
        self.reduction = reduction
    
    def forward(self, pred, target, modal_dims=None):
        """
        Args:
            pred: 预测值 [total_dim]
            target: 目标值 [total_dim]
            modal_dims: 各模态的维度 [a_dim, t_dim, v_dim]，如果为None则自动推断
        """
        if modal_dims is None:
            # 默认维度（IEMOCAP）
            modal_dims = [100, 768, 512]
        
        # 按模态分割
        a_dim, t_dim, v_dim = modal_dims
        a_pred, t_pred, v_pred = pred.split([a_dim, t_dim, v_dim], dim=0)
        a_target, t_target, v_target = target.split([a_dim, t_dim, v_dim], dim=0)
        
        # 计算各模态的MSE损失
        a_loss = F.mse_loss(a_pred, a_target, reduction='sum')
        t_loss = F.mse_loss(t_pred, t_target, reduction='sum')
        v_loss = F.mse_loss(v_pred, v_target, reduction='sum')
        
        # 加权求和
        weighted_loss = (self.modal_weights[0] * a_loss +
                        self.modal_weights[1] * t_loss +
                        self.modal_weights[2] * v_loss)
        
        if self.reduction == 'mean':
            return weighted_loss / (a_dim + t_dim + v_dim)
        else:
            return weighted_loss


class FocalLoss(nn.Module):
    """
    Focal Loss
    用于处理类别不平衡问题
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        """
        Args:
            alpha: 平衡因子
            gamma: 聚焦参数，gamma越大，对难样本的关注越大
            reduction: 'mean' or 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: 模型输出 [batch, num_classes]
            targets: 真实标签 [batch]
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class AdaptiveLossWeight:
    """
    自适应损失权重管理器
    根据损失值动态调整权重
    """
    def __init__(self, initial_weights, momentum=0.9, min_weight=0.01, max_weight=10.0):
        """
        Args:
            initial_weights: 初始权重字典 {'loss_name': weight}
            momentum: 移动平均的动量
            min_weight: 最小权重
            max_weight: 最大权重
        """
        self.initial_weights = initial_weights.copy()
        self.weights = initial_weights.copy()
        self.momentum = momentum
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.loss_history = {k: [] for k in initial_weights.keys()}
        self.avg_losses = {k: 0.0 for k in initial_weights.keys()}
    
    def update(self, losses):
        """
        根据当前损失值更新权重
        
        Args:
            losses: 损失字典 {'loss_name': loss_tensor}
        
        Returns:
            updated_weights: 更新后的权重字典
        """
        updated_weights = {}
        
        for key, loss in losses.items():
            if key not in self.weights:
                continue
            
            loss_value = loss.item() if isinstance(loss, torch.Tensor) else loss
            self.loss_history[key].append(loss_value)
            
            # 计算移动平均
            if len(self.loss_history[key]) == 1:
                self.avg_losses[key] = loss_value
            else:
                self.avg_losses[key] = (self.momentum * self.avg_losses[key] +
                                       (1 - self.momentum) * loss_value)
        
        # 计算总平均损失
        total_avg = sum(self.avg_losses.values())
        if total_avg > 0:
            # 权重与损失成反比，归一化
            for key in self.weights.keys():
                if self.avg_losses[key] > 0:
                    # 反比例权重
                    inverse_weight = total_avg / (self.avg_losses[key] + 1e-8)
                    # 归一化到初始权重范围
                    normalized_weight = (inverse_weight / sum(
                        total_avg / (v + 1e-8) for v in self.avg_losses.values()
                    )) * sum(self.initial_weights.values())
                    
                    # 更新权重（使用动量）
                    self.weights[key] = (self.momentum * self.weights[key] +
                                       (1 - self.momentum) * normalized_weight)
                    
                    # 限制权重范围
                    self.weights[key] = max(self.min_weight,
                                          min(self.max_weight, self.weights[key]))
        
        return self.weights.copy()
    
    def get_weights(self):
        """获取当前权重"""
        return self.weights.copy()
    
    def reset(self):
        """重置权重到初始值"""
        self.weights = self.initial_weights.copy()
        self.loss_history = {k: [] for k in self.initial_weights.keys()}
        self.avg_losses = {k: 0.0 for k in self.initial_weights.keys()}


class LossNormalizer:
    """
    损失归一化器
    将不同量级的损失归一化到相似范围
    """
    def __init__(self, method='log', target_range=(0, 1)):
        """
        Args:
            method: 归一化方法 'log', 'minmax', 'zscore'
            target_range: 目标范围 (min, max)
        """
        self.method = method
        self.target_min, self.target_max = target_range
        self.loss_stats = {}  # 存储损失统计信息
    
    def normalize(self, losses):
        """
        归一化损失值
        
        Args:
            losses: 损失字典 {'loss_name': loss_tensor}
        
        Returns:
            normalized_losses: 归一化后的损失字典
        """
        normalized = {}
        
        for key, loss in losses.items():
            loss_value = loss.item() if isinstance(loss, torch.Tensor) else loss
            
            if self.method == 'log':
                # 对数归一化
                normalized_value = torch.log(loss + 1.0)
            elif self.method == 'minmax':
                # Min-Max归一化（需要历史统计）
                if key not in self.loss_stats:
                    self.loss_stats[key] = {'min': loss_value, 'max': loss_value}
                else:
                    self.loss_stats[key]['min'] = min(self.loss_stats[key]['min'], loss_value)
                    self.loss_stats[key]['max'] = max(self.loss_stats[key]['max'], loss_value)
                
                min_val = self.loss_stats[key]['min']
                max_val = self.loss_stats[key]['max']
                if max_val > min_val:
                    normalized_value = (loss_value - min_val) / (max_val - min_val)
                    normalized_value = normalized_value * (self.target_max - self.target_min) + self.target_min
                else:
                    normalized_value = loss
            elif self.method == 'zscore':
                # Z-score归一化（需要历史统计）
                if key not in self.loss_stats:
                    self.loss_stats[key] = {'mean': loss_value, 'std': 1.0, 'count': 1}
                else:
                    # 更新均值和标准差
                    old_mean = self.loss_stats[key]['mean']
                    old_count = self.loss_stats[key]['count']
                    new_count = old_count + 1
                    new_mean = (old_mean * old_count + loss_value) / new_count
                    new_std = np.sqrt(((old_count - 1) * self.loss_stats[key]['std']**2 +
                                     (loss_value - old_mean)**2) / old_count)
                    
                    self.loss_stats[key]['mean'] = new_mean
                    self.loss_stats[key]['std'] = new_std
                    self.loss_stats[key]['count'] = new_count
                
                mean = self.loss_stats[key]['mean']
                std = self.loss_stats[key]['std']
                if std > 0:
                    normalized_value = (loss_value - mean) / std
                else:
                    normalized_value = loss
            else:
                normalized_value = loss
            
            if isinstance(loss, torch.Tensor):
                normalized[key] = normalized_value if isinstance(normalized_value, torch.Tensor) else torch.tensor(normalized_value)
            else:
                normalized[key] = normalized_value
        
        return normalized


def compute_total_loss(losses, weights, normalize=False, normalizer=None):
    """
    计算总损失
    
    Args:
        losses: 损失字典 {'loss_name': loss_tensor}
        weights: 权重字典 {'loss_name': weight}
        normalize: 是否归一化损失
        normalizer: LossNormalizer实例
    
    Returns:
        total_loss: 总损失
        loss_components: 各损失组件（用于监控）
    """
    if normalize and normalizer is not None:
        losses = normalizer.normalize(losses)
    
    loss_components = {}
    total_loss = 0.0
    
    for key, loss in losses.items():
        weight = weights.get(key, 1.0)
        weighted_loss = weight * loss
        loss_components[key] = weighted_loss
        total_loss += weighted_loss
    
    return total_loss, loss_components


# ============================================================================
# 基础优化方案实现
# ============================================================================

def create_reconstruction_loss(use_smooth_l1=False, reduction='mean'):
    """
    创建重构损失函数（基础优化方案）
    
    Args:
        use_smooth_l1: 是否使用SmoothL1Loss（推荐），否则使用MSELoss
        reduction: 'mean' or 'sum'
    
    Returns:
        loss_fn: 损失函数实例
    
    Examples:
        >>> criterion = create_reconstruction_loss(use_smooth_l1=True)
        >>> loss = criterion(pred, target)
    """
    if use_smooth_l1:
        return nn.SmoothL1Loss(reduction=reduction)
    else:
        return nn.MSELoss(reduction=reduction)


class LossWeightConfig:
    """
    基础损失权重配置类
    提供可配置的损失权重管理（基础优化方案）
    """
    def __init__(self, 
                 encoder_loss_weight=0.03,
                 cl_loss_weight=0.2,
                 gate_reg_weight=0.01):
        """
        初始化损失权重配置
        
        Args:
            encoder_loss_weight: 融合重构损失权重（推荐0.03，原为0.05）
            cl_loss_weight: 对比学习损失权重（默认0.2）
            gate_reg_weight: 门控正则化权重（默认0.01）
        """
        self.encoder_loss_weight = encoder_loss_weight
        self.cl_loss_weight = cl_loss_weight
        self.gate_reg_weight = gate_reg_weight
    
    def get_weights(self):
        """
        获取权重字典
        
        Returns:
            weights: 权重字典
        """
        return {
            'encoder': self.encoder_loss_weight,
            'contrastive': self.cl_loss_weight,
            'gate_reg': self.gate_reg_weight
        }
    
    def update_encoder_weight(self, new_weight):
        """更新encoder损失权重"""
        self.encoder_loss_weight = new_weight
    
    def update_cl_weight(self, new_weight):
        """更新对比学习损失权重"""
        self.cl_loss_weight = new_weight
    
    def update_gate_reg_weight(self, new_weight):
        """更新门控正则化权重"""
        self.gate_reg_weight = new_weight
    
    @classmethod
    def from_args(cls, args):
        """
        从args对象创建配置
        
        Args:
            args: 命令行参数对象
        
        Returns:
            LossWeightConfig实例
        """
        encoder_weight = getattr(args, 'encoder_loss_weight', 0.03)
        cl_weight = getattr(args, 'cl_loss_weight', 0.2)
        gate_reg_weight = getattr(args, 'gate_reg_weight', 0.01)
        
        return cls(
            encoder_loss_weight=encoder_weight,
            cl_loss_weight=cl_weight,
            gate_reg_weight=gate_reg_weight
        )
    
    def __repr__(self):
        return (f"LossWeightConfig("
                f"encoder={self.encoder_loss_weight}, "
                f"cl={self.cl_loss_weight}, "
                f"gate_reg={self.gate_reg_weight})")


def compute_training_loss(classification_loss, 
                         contrastive_loss, 
                         encoder_loss,
                         weight_config=None,
                         gate_reg_loss=None):
    """
    计算训练总损失（基础优化方案）
    
    Args:
        classification_loss: 分类损失
        contrastive_loss: 对比学习损失
        encoder_loss: 融合重构损失
        weight_config: LossWeightConfig实例，如果为None则使用默认值
        gate_reg_loss: 门控正则化损失（可选）
    
    Returns:
        total_loss: 总损失
        loss_dict: 各损失组件字典（用于监控）
    
    Examples:
        >>> weight_config = LossWeightConfig(encoder_loss_weight=0.03)
        >>> total_loss, loss_dict = compute_training_loss(
        ...     classification_loss, contrastive_loss, encoder_loss, weight_config
        ... )
    """
    if weight_config is None:
        weight_config = LossWeightConfig()
    
    weights = weight_config.get_weights()
    
    # 计算加权损失
    weighted_classification = classification_loss
    weighted_contrastive = weights['contrastive'] * contrastive_loss
    weighted_encoder = weights['encoder'] * encoder_loss
    
    total_loss = weighted_classification + weighted_contrastive + weighted_encoder
    
    loss_dict = {
        'classification': classification_loss,
        'contrastive': contrastive_loss,
        'encoder': encoder_loss,
        'total': total_loss
    }
    
    # 如果有门控正则化损失，添加进去
    if gate_reg_loss is not None:
        weighted_gate_reg = weights['gate_reg'] * gate_reg_loss
        total_loss = total_loss + weighted_gate_reg
        loss_dict['gate_reg'] = gate_reg_loss
        loss_dict['total'] = total_loss
    
    return total_loss, loss_dict

