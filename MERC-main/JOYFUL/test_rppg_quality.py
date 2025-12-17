"""
测试rPPG质量检测功能
验证综合质量检测是否正常工作
"""

import torch
import numpy as np
from joyful.rppg_extractor import RPPGQualityChecker, LightweightRPPGExtractor

def generate_synthetic_rppg(length=150, hr=70, fs=30, noise_level=0.1):
    """
    生成合成的rPPG信号
    
    Args:
        length: 信号长度（帧数）
        hr: 心率（bpm）
        fs: 采样率（fps）
        noise_level: 噪声水平
    """
    t = np.arange(length) / fs
    freq = hr / 60  # 转换为Hz
    
    # 生成正弦波（模拟rPPG信号）
    signal = np.sin(2 * np.pi * freq * t)
    
    # 添加噪声
    noise = np.random.normal(0, noise_level, length)
    signal = signal + noise
    
    return torch.tensor(signal, dtype=torch.float32)


def test_quality_checker():
    """测试质量检测器"""
    print("=" * 60)
    print("测试rPPG质量检测器")
    print("=" * 60)
    
    # 测试1：高质量信号（正常心率70 bpm）
    print("\n测试1: 高质量信号（HR=70 bpm, 低噪声）")
    good_signal = generate_synthetic_rppg(length=150, hr=70, fs=30, noise_level=0.05)
    
    stat_valid, stat_score = RPPGQualityChecker.check_statistical_quality(good_signal)
    print(f"  统计检测: valid={stat_valid}, score={stat_score:.3f}")
    
    freq_valid, freq_score = RPPGQualityChecker.check_frequency_quality(good_signal, fs=30)
    print(f"  频域检测: valid={freq_valid}, score={freq_score:.3f}")
    
    overall_valid, overall_score = RPPGQualityChecker.comprehensive_check(good_signal, fs=30)
    print(f"  综合检测: valid={overall_valid}, score={overall_score:.3f}")
    print(f"  ✅ 预期: 高质量，通过检测")
    
    # 测试2：中等质量信号（正常心率但噪声较大）
    print("\n测试2: 中等质量信号（HR=70 bpm, 中等噪声）")
    medium_signal = generate_synthetic_rppg(length=150, hr=70, fs=30, noise_level=0.3)
    
    stat_valid, stat_score = RPPGQualityChecker.check_statistical_quality(medium_signal)
    print(f"  统计检测: valid={stat_valid}, score={stat_score:.3f}")
    
    freq_valid, freq_score = RPPGQualityChecker.check_frequency_quality(medium_signal, fs=30)
    print(f"  频域检测: valid={freq_valid}, score={freq_score:.3f}")
    
    overall_valid, overall_score = RPPGQualityChecker.comprehensive_check(medium_signal, fs=30)
    print(f"  综合检测: valid={overall_valid}, score={overall_score:.3f}")
    print(f"  ⚠️  预期: 中等质量，可能通过/不通过（取决于阈值）")
    
    # 测试3：异常心率信号（200 bpm，超出合理范围）
    print("\n测试3: 异常心率信号（HR=200 bpm）")
    abnormal_signal = generate_synthetic_rppg(length=150, hr=200, fs=30, noise_level=0.05)
    
    stat_valid, stat_score = RPPGQualityChecker.check_statistical_quality(abnormal_signal)
    print(f"  统计检测: valid={stat_valid}, score={stat_score:.3f}")
    
    freq_valid, freq_score = RPPGQualityChecker.check_frequency_quality(abnormal_signal, fs=30)
    print(f"  频域检测: valid={freq_valid}, score={freq_score:.3f}")
    
    overall_valid, overall_score = RPPGQualityChecker.comprehensive_check(abnormal_signal, fs=30)
    print(f"  综合检测: valid={overall_valid}, score={overall_score:.3f}")
    print(f"  ❌ 预期: 频率异常，频域检测应该失败")
    
    # 测试4：零向量（无效信号）
    print("\n测试4: 零向量（无效信号）")
    zero_signal = torch.zeros(150, dtype=torch.float32)
    
    stat_valid, stat_score = RPPGQualityChecker.check_statistical_quality(zero_signal)
    print(f"  统计检测: valid={stat_valid}, score={stat_score:.3f}")
    
    freq_valid, freq_score = RPPGQualityChecker.check_frequency_quality(zero_signal, fs=30)
    print(f"  频域检测: valid={freq_valid}, score={freq_score:.3f}")
    
    overall_valid, overall_score = RPPGQualityChecker.comprehensive_check(zero_signal, fs=30)
    print(f"  综合检测: valid={overall_valid}, score={overall_score:.3f}")
    print(f"  ❌ 预期: 完全无效，统计检测应该失败")
    
    # 测试5：纯噪声信号
    print("\n测试5: 纯噪声信号（随机噪声）")
    noise_signal = torch.randn(150, dtype=torch.float32) * 0.5
    
    stat_valid, stat_score = RPPGQualityChecker.check_statistical_quality(noise_signal)
    print(f"  统计检测: valid={stat_valid}, score={stat_score:.3f}")
    
    freq_valid, freq_score = RPPGQualityChecker.check_frequency_quality(noise_signal, fs=30)
    print(f"  频域检测: valid={freq_valid}, score={freq_score:.3f}")
    
    overall_valid, overall_score = RPPGQualityChecker.comprehensive_check(noise_signal, fs=30)
    print(f"  综合检测: valid={overall_valid}, score={overall_score:.3f}")
    print(f"  ❌ 预期: 频率分散，频域检测应该失败")
    
    # 测试6：低方差信号
    print("\n测试6: 低方差信号（常数 + 微小噪声）")
    low_var_signal = torch.ones(150, dtype=torch.float32) + torch.randn(150) * 0.001
    
    stat_valid, stat_score = RPPGQualityChecker.check_statistical_quality(low_var_signal)
    print(f"  统计检测: valid={stat_valid}, score={stat_score:.3f}")
    
    freq_valid, freq_score = RPPGQualityChecker.check_frequency_quality(low_var_signal, fs=30)
    print(f"  频域检测: valid={freq_valid}, score={freq_score:.3f}")
    
    overall_valid, overall_score = RPPGQualityChecker.comprehensive_check(low_var_signal, fs=30)
    print(f"  综合检测: valid={overall_valid}, score={overall_score:.3f}")
    print(f"  ❌ 预期: 方差过低，统计检测应该失败")
    
    print("\n" + "=" * 60)
    print("质量检测器测试完成")
    print("=" * 60)


def test_lightweight_extractor():
    """测试轻量级rPPG提取器"""
    print("\n" + "=" * 60)
    print("测试轻量级rPPG提取器")
    print("=" * 60)
    
    # 创建模拟视频数据：批量大小2，150帧，64x64分辨率
    batch_size = 2
    num_frames = 150
    height, width = 64, 64
    
    print(f"\n输入视频形状: [{batch_size}, {num_frames}, {height}, {width}, 3]")
    video = torch.randn(batch_size, num_frames, height, width, 3)
    
    # 创建提取器
    print("初始化轻量级rPPG提取器...")
    extractor = LightweightRPPGExtractor(feature_dim=64, temporal_window=150)
    
    # 前向传播
    print("提取rPPG特征...")
    with torch.no_grad():
        features, quality_scores = extractor(video)
    
    print(f"\n输出特征形状: {features.shape}")
    print(f"预期形状: [{batch_size}, 64]")
    print(f"✅ 形状正确" if features.shape == (batch_size, 64) else "❌ 形状错误")
    
    print(f"\n质量分数: {quality_scores}")
    print(f"预期范围: [0, 1]")
    all_in_range = torch.all((quality_scores >= 0) & (quality_scores <= 1))
    print(f"✅ 范围正确" if all_in_range else "❌ 范围错误")
    
    print("\n" + "=" * 60)
    print("轻量级提取器测试完成")
    print("=" * 60)


def test_quality_threshold_effect():
    """测试不同质量阈值的效果"""
    print("\n" + "=" * 60)
    print("测试不同质量阈值的效果")
    print("=" * 60)
    
    # 生成不同质量的信号
    num_samples = 100
    signals = []
    quality_scores = []
    
    print(f"\n生成{num_samples}个不同质量的rPPG信号...")
    for i in range(num_samples):
        # 随机心率（40-180 bpm）
        hr = np.random.uniform(40, 180)
        # 随机噪声水平（0.01-0.5）
        noise = np.random.uniform(0.01, 0.5)
        
        signal = generate_synthetic_rppg(length=150, hr=hr, fs=30, noise_level=noise)
        signals.append(signal)
        
        # 计算质量分数
        _, score = RPPGQualityChecker.comprehensive_check(signal, fs=30)
        quality_scores.append(score)
    
    quality_scores = np.array(quality_scores)
    
    # 统计不同阈值下的有效样本比例
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6]
    
    print(f"\n质量分数分布:")
    print(f"  最小值: {quality_scores.min():.3f}")
    print(f"  最大值: {quality_scores.max():.3f}")
    print(f"  平均值: {quality_scores.mean():.3f}")
    print(f"  中位数: {np.median(quality_scores):.3f}")
    
    print(f"\n不同阈值下的有效样本比例:")
    print(f"{'阈值':<10} {'有效样本':<15} {'比例':<10}")
    print("-" * 35)
    for threshold in thresholds:
        valid_count = np.sum(quality_scores >= threshold)
        valid_ratio = valid_count / num_samples
        print(f"{threshold:<10.1f} {valid_count:<15d} {valid_ratio:<10.1%}")
    
    print(f"\n推荐阈值: 0.3 (平衡质量与数量)")
    
    print("\n" + "=" * 60)
    print("阈值效果测试完成")
    print("=" * 60)


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("rPPG质量检测和提取器综合测试")
    print("=" * 60)
    
    # 测试1：质量检测器
    test_quality_checker()
    
    # 测试2：轻量级提取器
    test_lightweight_extractor()
    
    # 测试3：质量阈值效果
    test_quality_threshold_effect()
    
    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)
    print("\n下一步:")
    print("1. 运行训练命令（见 RPPG_IMPROVEMENT_QUICKSTART.md）")
    print("2. 使用 --rppg_quality_check comprehensive")
    print("3. 对比 Happy F1 和 Overall F1")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

