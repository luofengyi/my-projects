"""
ULGM (Unimodal Label Generation Module)
非参数模块，基于多模态和单模态特征到类中心的相对距离生成单模态伪标签
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ULGM(nn.Module):
    """
    Unimodal Label Generation Module (非参数)
    
    基于多模态注释和模态表示生成单模态监督值：
    1. 维护每个类别的中心向量（动态更新）
    2. 计算特征到类中心的相对距离
    3. 生成软伪标签（而非硬标签，减少训练早期的干扰）
    """
    
    def __init__(self, num_classes=4, feature_dim=128, momentum=0.9, 
                 temp=1.0, use_hard_labels=False, class_specific_config=None):
        """
        Args:
            num_classes: 情感类别数量
            feature_dim: 特征维度（用于类中心）
            momentum: 类中心更新的动量（EMA）
            temp: 软标签温度系数（越小越接近硬标签）
            use_hard_labels: 是否使用硬标签（False则使用软标签，更稳定）
            class_specific_config: 类别特定配置（为Happy等少数类提供更保守策略）
        """
        super(ULGM, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.momentum = momentum
        self.temp = temp
        self.use_hard_labels = use_hard_labels
        
        # 类别特定配置（默认配置：happy=0对应更保守策略）
        if class_specific_config is None:
            self.class_specific_config = {
                0: {'min_samples': 20, 'true_label_weight': 0.5},  # Happy：需要更多样本，保留更多真实标签
                1: {'min_samples': 10, 'true_label_weight': 0.3},  # Sad
                2: {'min_samples': 10, 'true_label_weight': 0.3},  # Neutral
                3: {'min_samples': 10, 'true_label_weight': 0.3},  # Angry
            }
        else:
            self.class_specific_config = class_specific_config
        
        # 类中心：使用buffer而非parameter，避免被优化器更新
        # 初始化为零向量，训练中逐步更新
        self.register_buffer('multimodal_centers', 
                           torch.zeros(num_classes, feature_dim))
        self.register_buffer('text_centers', 
                           torch.zeros(num_classes, feature_dim))
        self.register_buffer('audio_centers', 
                           torch.zeros(num_classes, feature_dim))
        self.register_buffer('video_centers', 
                           torch.zeros(num_classes, feature_dim))
        
        # 类中心初始化标志
        self.register_buffer('centers_initialized', torch.tensor(False))
        
        # 类样本计数（用于稳定早期更新）
        self.register_buffer('class_counts', torch.zeros(num_classes))
    
    def update_centers(self, multimodal_feat, text_feat, audio_feat, video_feat, 
                      labels, min_samples=5):
        """
        更新类中心（使用指数移动平均）
        
        Args:
            multimodal_feat: 多模态特征 [feature_dim]
            text_feat: 文本特征 [feature_dim]
            audio_feat: 音频特征 [feature_dim]
            video_feat: 视觉特征 [feature_dim]
            labels: 真实标签（标量）
            min_samples: 开始使用EMA前每类需要的最小样本数
        """
        if not self.training:
            return
        
        label = labels.item() if labels.dim() > 0 else labels
        label = int(label)
        
        # 更新样本计数
        self.class_counts[label] += 1
        
        # 早期：直接平均；后期：EMA
        if self.class_counts[label] <= min_samples:
            # 累积平均
            alpha = 1.0 / self.class_counts[label].item()
        else:
            # EMA
            alpha = 1 - self.momentum
        
        # 更新各模态的类中心
        with torch.no_grad():
            self.multimodal_centers[label] = (
                self.momentum * self.multimodal_centers[label] + 
                alpha * multimodal_feat.detach()
            )
            self.text_centers[label] = (
                self.momentum * self.text_centers[label] + 
                alpha * text_feat.detach()
            )
            self.audio_centers[label] = (
                self.momentum * self.audio_centers[label] + 
                alpha * audio_feat.detach()
            )
            self.video_centers[label] = (
                self.momentum * self.video_centers[label] + 
                alpha * video_feat.detach()
            )
        
        # 标记已初始化
        if not self.centers_initialized and self.class_counts.min() > 0:
            self.centers_initialized = torch.tensor(True)
    
    def compute_relative_distance(self, feature, centers, eps=1e-8):
        """
        计算特征到所有类中心的相对距离（归一化后的距离）
        
        Args:
            feature: 特征向量 [feature_dim]
            centers: 类中心 [num_classes, feature_dim]
            eps: 数值稳定项
        
        Returns:
            relative_distances: 相对距离 [num_classes]，值越小表示越接近该类
        """
        # 计算欧氏距离
        distances = torch.norm(feature.unsqueeze(0) - centers, dim=1)  # [num_classes]
        
        # 归一化为相对距离（与空间尺度无关）
        # 使用softmax的反转：距离越小，值越大
        relative_distances = distances / (distances.sum() + eps)
        
        return relative_distances
    
    def generate_pseudo_label(self, multimodal_feat, unimodal_feat, 
                             multimodal_logits, true_label, 
                             confidence_threshold=0.5, modality_name='text'):
        """
        基于相对距离生成单模态伪标签（类别特定策略）
        
        策略：
        1. 如果类中心未充分初始化，返回真实标签（稳定早期训练）
        2. Happy等少数类使用更保守的融合策略（更多保留真实标签）
        3. 计算多模态和单模态特征到各类中心的相对距离
        4. 基于多模态预测置信度决定伪标签生成策略
        
        Args:
            multimodal_feat: 多模态特征 [feature_dim]
            unimodal_feat: 单模态特征 [feature_dim]
            multimodal_logits: 多模态分类器输出 [num_classes]
            true_label: 真实标签（标量）
            confidence_threshold: 置信度阈值
            modality_name: 模态名称（用于选择类中心）
        
        Returns:
            pseudo_label: 伪标签（软标签 [num_classes] 或硬标签 标量）
        """
        label_idx = true_label.item() if true_label.dim() > 0 else int(true_label)
        
        # 获取该类别的特定配置
        class_config = self.class_specific_config.get(
            label_idx, 
            {'min_samples': 10, 'true_label_weight': 0.3}
        )
        min_samples = class_config['min_samples']
        true_label_base_weight = class_config['true_label_weight']
        
        # 如果该类样本数不足，直接返回真实标签
        if self.class_counts[label_idx] < min_samples:
            if self.use_hard_labels:
                return true_label
            else:
                soft_label = torch.zeros(self.num_classes, device=multimodal_feat.device)
                soft_label[label_idx] = 1.0
                return soft_label
        
        # 如果类中心未初始化，直接返回真实标签（软标签形式）
        if not self.centers_initialized:
            if self.use_hard_labels:
                return true_label
            else:
                # 返回one-hot软标签
                soft_label = torch.zeros(self.num_classes, device=multimodal_feat.device)
                soft_label[label_idx] = 1.0
                return soft_label
        
        with torch.no_grad():
            # 计算多模态预测的置信度
            multimodal_probs = F.softmax(multimodal_logits / self.temp, dim=-1)
            confidence = multimodal_probs.max().item()
            
            # 选择对应模态的类中心
            if modality_name == 'text':
                unimodal_centers = self.text_centers
            elif modality_name == 'audio':
                unimodal_centers = self.audio_centers
            elif modality_name == 'video':
                unimodal_centers = self.video_centers
            else:
                unimodal_centers = self.text_centers
            
            # 计算多模态和单模态特征到类中心的相对距离
            multimodal_rel_dist = self.compute_relative_distance(
                multimodal_feat, self.multimodal_centers
            )
            unimodal_rel_dist = self.compute_relative_distance(
                unimodal_feat, unimodal_centers
            )
            
            # 生成软伪标签：反转距离得到相似度分数
            similarity_scores = 1.0 / (unimodal_rel_dist + 1e-8)
            pseudo_probs = F.softmax(similarity_scores / self.temp, dim=-1)
            
            # 准备真实标签的one-hot编码
            true_label_onehot = torch.zeros(self.num_classes, device=multimodal_feat.device)
            true_label_onehot[label_idx] = 1.0
            
            # 类别特定的融合策略
            # Happy等少数类：保留更多真实标签，减少对不稳定预测的依赖
            if confidence > confidence_threshold:
                # 高置信度：更多依赖多模态预测，但Happy类仍保留更多真实标签
                distance_weight = 0.2
                multimodal_weight = 0.8 - true_label_base_weight
                true_label_weight = true_label_base_weight
            else:
                # 低置信度：大幅增加真实标签权重，尤其是Happy类
                distance_weight = 0.3
                multimodal_weight = 0.7 - true_label_base_weight * 1.5
                true_label_weight = true_label_base_weight * 1.5
            
            # 确保权重和为1
            total_weight = distance_weight + multimodal_weight + true_label_weight
            distance_weight /= total_weight
            multimodal_weight /= total_weight
            true_label_weight /= total_weight
            
            # 融合生成伪标签
            pseudo_label_soft = (
                distance_weight * pseudo_probs + 
                multimodal_weight * multimodal_probs + 
                true_label_weight * true_label_onehot
            )
            
            # 归一化
            pseudo_label_soft = pseudo_label_soft / (pseudo_label_soft.sum() + 1e-8)
            
            if self.use_hard_labels:
                return pseudo_label_soft.argmax()
            else:
                return pseudo_label_soft
    
    def generate_text_pseudo_label(self, multimodal_feat, text_feat, 
                                   multimodal_logits, true_label, **kwargs):
        """为文本模态生成伪标签"""
        return self.generate_pseudo_label(
            multimodal_feat, text_feat, multimodal_logits, true_label, 
            modality_name='text', **kwargs
        )
    
    def generate_audio_pseudo_label(self, multimodal_feat, audio_feat, 
                                    multimodal_logits, true_label, **kwargs):
        """为音频模态生成伪标签"""
        return self.generate_pseudo_label(
            multimodal_feat, audio_feat, multimodal_logits, true_label, 
            modality_name='audio', **kwargs
        )
    
    def generate_video_pseudo_label(self, multimodal_feat, video_feat, 
                                    multimodal_logits, true_label, **kwargs):
        """为视觉模态生成伪标签"""
        return self.generate_pseudo_label(
            multimodal_feat, video_feat, multimodal_logits, true_label, 
            modality_name='video', **kwargs
        )

