"""
改进的rPPG特征提取模块
借鉴论文: Deep learning-based remote-photoplethysmography measurement from short-time facial video

核心思想：
1. 3D时空卷积网络：从短时视频序列中提取rPPG特征
2. 多尺度特征提取：逐步提取不同尺度的时空特征
3. 时空注意力机制：聚焦于与rPPG信号高度相关的特征
4. 残差连接：保留浅层皮肤颜色变化信息
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpatialTemporalAttention3D(nn.Module):
    """
    3D时空注意力机制
    论文中的3D-S/T attention: 分别处理空间和时间维度
    """
    def __init__(self, in_channels):
        super(SpatialTemporalAttention3D, self).__init__()
        
        # 空间注意力（关注面部哪些区域）
        self.spatial_attention = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // 8, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 时间注意力（关注哪些时间帧）
        # 使用全局平均池化压缩空间维度，只保留时间和通道
        self.temporal_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d((None, 1, 1)),  # 保留T维度，压缩H,W
            nn.Conv3d(in_channels, in_channels // 8, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // 8, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, C, T, H, W] - 3D特征图
        Returns:
            attended_x: 加权后的特征图
        """
        # 空间注意力：[B, 1, T, H, W]
        spatial_weights = self.spatial_attention(x)
        x_spatial = x * spatial_weights
        
        # 时间注意力：[B, C, T, 1, 1]
        temporal_weights = self.temporal_attention(x)
        x_temporal = x_spatial * temporal_weights
        
        return x_temporal


class Conv3DBlock(nn.Module):
    """
    3D卷积块：Conv3D + BatchNorm + ReLU
    论文中的基础构建模块
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv3DBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResidualConstantBlock3D(nn.Module):
    """
    3D残差常量块
    论文中用于在特征降维时传递浅层信息
    """
    def __init__(self, channels):
        super(ResidualConstantBlock3D, self).__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm3d(channels)
    
    def forward(self, x):
        """保留原始特征的残差连接"""
        residual = F.adaptive_avg_pool3d(x, output_size=(x.size(2)//2, x.size(3)//2, x.size(4)//2))
        out = self.bn(self.conv(x))
        return out + residual


class RPPGEncoder3D(nn.Module):
    """
    rPPG 3D编码器
    论文中的backbone network (a)
    
    架构：
    - 5个卷积块（ConvB_1 到 ConvB_5）
    - 多尺度特征提取
    - 残差连接保留浅层信息
    """
    def __init__(self, in_channels=3):
        super(RPPGEncoder3D, self).__init__()
        
        # ConvB_1: 单层3D卷积
        self.conv1 = Conv3DBlock(in_channels, 16, kernel_size=5, stride=1, padding=2)
        
        # ConvB_2-5: 双层3D卷积块
        self.conv2 = nn.Sequential(
            Conv3DBlock(16, 32, kernel_size=3, stride=1, padding=1),
            Conv3DBlock(32, 32, kernel_size=3, stride=1, padding=1)
        )
        self.pool2 = nn.AvgPool3d(kernel_size=(1, 2, 2))  # 降低空间分辨率
        
        self.conv3 = nn.Sequential(
            Conv3DBlock(32, 64, kernel_size=3, stride=1, padding=1),
            Conv3DBlock(64, 64, kernel_size=3, stride=1, padding=1)
        )
        self.pool3 = nn.AvgPool3d(kernel_size=(1, 2, 2))
        
        self.conv4 = nn.Sequential(
            Conv3DBlock(64, 64, kernel_size=3, stride=1, padding=1),
            Conv3DBlock(64, 64, kernel_size=3, stride=1, padding=1)
        )
        self.pool4 = nn.AvgPool3d(kernel_size=(1, 2, 2))
        
        self.conv5 = nn.Sequential(
            Conv3DBlock(64, 128, kernel_size=3, stride=1, padding=1),
            Conv3DBlock(128, 128, kernel_size=3, stride=1, padding=1)
        )
        self.pool5 = nn.AvgPool3d(kernel_size=(1, 2, 2))
        
        # 残差连接（论文中的Residual Structure (b)）
        self.residual_blocks = nn.ModuleList([
            ResidualConstantBlock3D(32),
            ResidualConstantBlock3D(64),
            ResidualConstantBlock3D(64)
        ])
        
        # 全局平均池化：压缩空间维度，保留时间维度
        self.global_pool = nn.AdaptiveAvgPool3d((None, 1, 1))  # [B, 128, T, 1, 1]
    
    def forward(self, x):
        """
        Args:
            x: [B, C, T, H, W] - 输入视频序列（C=3 for RGB）
        Returns:
            features: [B, 128, T] - 提取的rPPG特征
            mid_features: 中间特征（用于分支监督）
        """
        # Stage 1
        x1 = self.conv1(x)  # [B, 16, T, H, W]
        
        # Stage 2
        x2 = self.conv2(x1)  # [B, 32, T, H, W]
        x2 = self.pool2(x2)  # [B, 32, T, H/2, W/2]
        
        # Stage 3
        x3 = self.conv3(x2)  # [B, 64, T, H/2, W/2]
        x3 = self.pool3(x3)  # [B, 64, T, H/4, W/4]
        
        # 中间特征（用于分支监督）
        mid_features = x3
        
        # Stage 4
        x4 = self.conv4(x3)  # [B, 64, T, H/4, W/4]
        x4 = self.pool4(x4)  # [B, 64, T, H/8, W/8]
        
        # Stage 5
        x5 = self.conv5(x4)  # [B, 128, T, H/8, W/8]
        x5 = self.pool5(x5)  # [B, 128, T, H/16, W/16]
        
        # 全局池化：压缩空间维度
        features = self.global_pool(x5)  # [B, 128, T, 1, 1]
        features = features.squeeze(-1).squeeze(-1)  # [B, 128, T]
        
        return features, mid_features


class RPPGDecoder1D(nn.Module):
    """
    rPPG 1D解码器
    论文中的decoder + smoothing
    将时空特征转换为1D rPPG信号
    """
    def __init__(self):
        super(RPPGDecoder1D, self).__init__()
        
        # 反卷积恢复时间分辨率
        self.deconv1 = nn.ConvTranspose1d(128, 128, kernel_size=4, stride=1, padding=0)
        self.deconv2 = nn.ConvTranspose1d(128, 128, kernel_size=4, stride=1, padding=0)
        self.deconv3 = nn.ConvTranspose1d(128, 64, kernel_size=4, stride=1, padding=0)
        
        # SGAP: Spatial Global Average Pooling (already done in encoder)
        # Smoothing: 1D卷积平滑信号
        self.smooth = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=5, stride=1, padding=2),
            nn.Tanh(),
            nn.Conv1d(32, 1, kernel_size=3, stride=1, padding=1)  # 输出单通道rPPG信号
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, 128, T] - 编码器输出的特征
        Returns:
            rppg_signal: [B, T'] - 恢复的rPPG信号
        """
        x = F.relu(self.deconv1(x))  # [B, 128, T+3]
        x = F.relu(self.deconv2(x))  # [B, 128, T+6]
        x = F.relu(self.deconv3(x))  # [B, 64, T+9]
        
        rppg_signal = self.smooth(x)  # [B, 1, T+9]
        rppg_signal = rppg_signal.squeeze(1)  # [B, T+9]
        
        return rppg_signal


class RPPGExtractor(nn.Module):
    """
    完整的rPPG特征提取器
    论文中的完整pipeline：Encoder + Attention + Decoder
    
    输入：短时面部视频序列（例如5秒，约150帧）
    输出：rPPG特征向量（用于情感识别融合）
    """
    def __init__(self, feature_dim=64, use_attention=True):
        """
        Args:
            feature_dim: 输出特征维度（用于与其他模态融合）
            use_attention: 是否使用时空注意力机制
        """
        super(RPPGExtractor, self).__init__()
        
        self.encoder = RPPGEncoder3D(in_channels=3)
        self.use_attention = use_attention
        
        if use_attention:
            self.attention = SpatialTemporalAttention3D(in_channels=128)
        
        self.decoder = RPPGDecoder1D()
        
        # 特征投影：将时序特征投影到固定维度
        self.feature_projection = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, feature_dim)
        )
        
        # 信号质量评估网络
        self.quality_estimator = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 输出0-1的质量分数
        )
    
    def forward(self, video_frames, return_signal=False):
        """
        Args:
            video_frames: [B, T, H, W, C] - 输入视频帧序列
            return_signal: 是否返回重构的rPPG信号
        Returns:
            features: [B, feature_dim] - rPPG特征向量
            quality_score: [B] - 信号质量分数
            rppg_signal: [B, T'] - 重构的rPPG信号（可选）
        """
        # 调整维度：[B, T, H, W, C] -> [B, C, T, H, W]
        video_frames = video_frames.permute(0, 4, 1, 2, 3)
        
        # 编码
        encoded_features, mid_features = self.encoder(video_frames)  # [B, 128, T]
        
        # 时空注意力（可选）
        if self.use_attention:
            # 恢复5D张量以应用3D注意力
            B, C, T = encoded_features.shape
            encoded_5d = encoded_features.unsqueeze(-1).unsqueeze(-1)  # [B, 128, T, 1, 1]
            attended_5d = self.attention(encoded_5d)
            encoded_features = attended_5d.squeeze(-1).squeeze(-1)  # [B, 128, T]
        
        # 时间平均池化：聚合时序信息
        pooled_features = torch.mean(encoded_features, dim=2)  # [B, 128]
        
        # 特征投影
        features = self.feature_projection(pooled_features)  # [B, feature_dim]
        
        # 质量评估
        quality_score = self.quality_estimator(pooled_features).squeeze(-1)  # [B]
        
        outputs = {
            'features': features,
            'quality_score': quality_score
        }
        
        # 解码rPPG信号（可选）
        if return_signal:
            rppg_signal = self.decoder(encoded_features)  # [B, T']
            outputs['rppg_signal'] = rppg_signal
            outputs['mid_features'] = mid_features
        
        return outputs


class RPPGQualityChecker:
    """
    rPPG信号质量检测器（改进版）
    结合统计特性和深度学习质量评估
    """
    @staticmethod
    def check_statistical_quality(rppg_signal, abs_max_threshold=1e-6, variance_threshold=1e-4):
        """
        统计质量检测：检测零方差或极小振幅的无效信号
        """
        if rppg_signal is None:
            return False, 0.0
        
        abs_max = torch.abs(rppg_signal).max().item()
        variance = torch.var(rppg_signal).item()
        
        is_valid = abs_max >= abs_max_threshold and variance >= variance_threshold
        
        # 简单的质量分数：基于方差和振幅
        quality_score = min(1.0, (variance / 0.1) * (abs_max / 1.0))
        
        return is_valid, quality_score
    
    @staticmethod
    def check_frequency_quality(rppg_signal, fs=30, expected_hr_range=(40, 180)):
        """
        频域质量检测：检测信号是否在合理的心率频率范围内
        
        Args:
            rppg_signal: [T] - rPPG信号
            fs: 采样率（帧率）
            expected_hr_range: 期望的心率范围（bpm）
        """
        if rppg_signal is None or len(rppg_signal) < 32:
            return False, 0.0
        
        # 转换为numpy进行FFT
        if torch.is_tensor(rppg_signal):
            signal = rppg_signal.detach().cpu().numpy()
        else:
            signal = rppg_signal
        
        # 去除直流分量
        signal = signal - np.mean(signal)
        
        # FFT
        n = len(signal)
        fft_vals = np.fft.rfft(signal)
        fft_freq = np.fft.rfftfreq(n, 1/fs)
        
        # 转换为心率频率范围
        hr_freq_range = (expected_hr_range[0] / 60, expected_hr_range[1] / 60)
        
        # 在期望频率范围内的能量占比
        freq_mask = (fft_freq >= hr_freq_range[0]) & (fft_freq <= hr_freq_range[1])
        if not np.any(freq_mask):
            return False, 0.0
        
        total_power = np.sum(np.abs(fft_vals) ** 2)
        valid_power = np.sum(np.abs(fft_vals[freq_mask]) ** 2)
        
        if total_power < 1e-6:
            return False, 0.0
        
        power_ratio = valid_power / total_power
        
        # 质量判断：期望范围内的能量应占主导
        is_valid = power_ratio > 0.5
        quality_score = min(1.0, power_ratio)
        
        return is_valid, quality_score
    
    @staticmethod
    def comprehensive_check(rppg_signal, fs=30):
        """
        综合质量检测：结合统计和频域检测
        """
        stat_valid, stat_score = RPPGQualityChecker.check_statistical_quality(rppg_signal)
        
        if not stat_valid:
            return False, 0.0
        
        freq_valid, freq_score = RPPGQualityChecker.check_frequency_quality(rppg_signal, fs)
        
        # 综合质量分数
        overall_score = 0.4 * stat_score + 0.6 * freq_score
        overall_valid = stat_valid and freq_valid
        
        return overall_valid, overall_score


class NegativePearsonLoss(nn.Module):
    """
    负Pearson相关系数损失
    论文中的L_NP损失：最小化线性相关误差
    """
    def __init__(self):
        super(NegativePearsonLoss, self).__init__()
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: [B, T] - 预测的rPPG信号
            targets: [B, T] - 真实的BVP信号
        Returns:
            loss: 负Pearson相关系数（越小越好）
        """
        # 计算均值
        pred_mean = torch.mean(predictions, dim=1, keepdim=True)
        target_mean = torch.mean(targets, dim=1, keepdim=True)
        
        # 中心化
        pred_centered = predictions - pred_mean
        target_centered = targets - target_mean
        
        # Pearson相关系数
        numerator = torch.sum(pred_centered * target_centered, dim=1)
        
        pred_std = torch.sqrt(torch.sum(pred_centered ** 2, dim=1))
        target_std = torch.sqrt(torch.sum(target_centered ** 2, dim=1))
        
        denominator = pred_std * target_std + 1e-8
        
        pearson = numerator / denominator
        
        # 负Pearson作为损失（最大化相关性 = 最小化负相关性）
        loss = 1 - torch.mean(pearson)
        
        return loss


# ==================== 用于JOYFUL集成的轻量级rPPG提取器 ====================

class LightweightRPPGExtractor(nn.Module):
    """
    轻量级rPPG提取器（适配JOYFUL模型）
    
    简化的3D CNN + 质量检测，适合与现有情感识别模型集成
    不需要完整的encoder-decoder，只需要提取判别性特征
    """
    def __init__(self, feature_dim=64, temporal_window=150):
        """
        Args:
            feature_dim: 输出特征维度（用于融合）
            temporal_window: 时间窗口大小（帧数）
        """
        super(LightweightRPPGExtractor, self).__init__()
        
        self.feature_dim = feature_dim
        self.temporal_window = temporal_window
        
        # 简化的3D卷积提取器
        self.conv3d_blocks = nn.Sequential(
            # Block 1
            nn.Conv3d(3, 16, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            
            # Block 2
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            
            # Block 3
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            
            # 全局池化
            nn.AdaptiveAvgPool3d((temporal_window // 4, 1, 1))  # 压缩空间维度
        )
        
        # 时间注意力
        self.temporal_attention = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # 特征投影
        self.feature_fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, feature_dim)
        )
        
        # 质量评估
        self.quality_fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, video_frames):
        """
        Args:
            video_frames: [B, T, H, W, C] - 输入视频帧
        Returns:
            features: [B, feature_dim]
            quality_score: [B]
        """
        # 调整维度
        video_frames = video_frames.permute(0, 4, 1, 2, 3)  # [B, C, T, H, W]
        
        # 3D卷积提取
        conv_features = self.conv3d_blocks(video_frames)  # [B, 64, T', 1, 1]
        conv_features = conv_features.squeeze(-1).squeeze(-1)  # [B, 64, T']
        
        # 时间注意力
        attention_weights = self.temporal_attention(conv_features)  # [B, 64, T']
        attended_features = conv_features * attention_weights
        
        # 时间池化
        pooled_features = torch.mean(attended_features, dim=2)  # [B, 64]
        
        # 特征投影
        features = self.feature_fc(pooled_features)  # [B, feature_dim]
        
        # 质量评估
        quality_score = self.quality_fc(pooled_features).squeeze(-1)  # [B]
        
        return features, quality_score


if __name__ == "__main__":
    # 测试代码
    print("Testing RPPGExtractor...")
    
    # 创建模拟数据：5秒视频，30fps，64x64分辨率
    batch_size = 2
    num_frames = 150
    height, width = 64, 64
    
    video = torch.randn(batch_size, num_frames, height, width, 3)
    
    # 测试完整提取器
    extractor = RPPGExtractor(feature_dim=64, use_attention=True)
    outputs = extractor(video, return_signal=True)
    
    print(f"Features shape: {outputs['features'].shape}")  # [2, 64]
    print(f"Quality scores: {outputs['quality_score']}")  # [2]
    print(f"rPPG signal shape: {outputs['rppg_signal'].shape}")  # [2, T']
    
    # 测试轻量级提取器
    print("\nTesting LightweightRPPGExtractor...")
    lightweight_extractor = LightweightRPPGExtractor(feature_dim=64)
    features, quality = lightweight_extractor(video)
    
    print(f"Features shape: {features.shape}")  # [2, 64]
    print(f"Quality scores: {quality}")  # [2]
    
    print("\nAll tests passed!")

