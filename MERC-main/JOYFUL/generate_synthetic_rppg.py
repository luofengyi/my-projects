"""
生成合成rPPG数据用于IEMOCAP数据集
当没有真实视频时，可以使用此脚本生成逼真的rPPG信号用于测试

使用方法:
    python generate_synthetic_rppg.py --data_dir ./data --output_dir ./data/rppg_features
"""

import os
import argparse
import pickle
import numpy as np
import torch
from tqdm import tqdm


def generate_realistic_rppg(num_frames, hr_mean=70, hr_std=5, fs=30, noise_level=0.1):
    """
    生成逼真的rPPG信号
    
    Args:
        num_frames: 帧数（信号长度）
        hr_mean: 平均心率（bpm）
        hr_std: 心率标准差
        fs: 采样率（fps）
        noise_level: 噪声水平
    
    Returns:
        rppg_signal: [num_frames] rPPG信号
    """
    # 随机心率（模拟个体差异和情绪影响）
    hr = np.random.normal(hr_mean, hr_std)
    hr = np.clip(hr, 50, 120)  # 合理范围
    
    # 时间轴
    t = np.arange(num_frames) / fs
    
    # 基础心跳信号（正弦波）
    freq = hr / 60  # 转换为Hz
    base_signal = np.sin(2 * np.pi * freq * t)
    
    # 添加谐波（更逼真）
    harmonic_2 = 0.3 * np.sin(2 * np.pi * 2 * freq * t)
    harmonic_3 = 0.1 * np.sin(2 * np.pi * 3 * freq * t)
    
    # 添加呼吸影响（低频调制）
    breathing_freq = 0.25  # 15次/分钟
    breathing_modulation = 0.1 * np.sin(2 * np.pi * breathing_freq * t)
    
    # 组合信号
    clean_signal = base_signal + harmonic_2 + harmonic_3
    clean_signal = clean_signal * (1 + breathing_modulation)
    
    # 添加生理噪声（1/f噪声 + 白噪声）
    white_noise = np.random.normal(0, noise_level, num_frames)
    
    # 1/f噪声（粉红噪声）
    pink_noise = np.cumsum(np.random.randn(num_frames))
    pink_noise = pink_noise / np.std(pink_noise) * noise_level * 0.5
    
    # 最终信号
    rppg_signal = clean_signal + white_noise + pink_noise
    
    # 归一化到[-1, 1]
    rppg_signal = rppg_signal / (np.max(np.abs(rppg_signal)) + 1e-8)
    
    return rppg_signal.astype(np.float32)


def emotion_specific_hr(emotion):
    """
    根据情绪调整心率参数（基于心理学研究）
    
    Args:
        emotion: 情绪标签（hap, sad, neu, ang）
    
    Returns:
        hr_mean, hr_std: 平均心率和标准差
    """
    emotion_hr_map = {
        'hap': (80, 8),   # Happy: 高心率，高变异
        'exc': (85, 10),  # Excited: 更高心率
        'ang': (85, 7),   # Angry: 高心率
        'sad': (65, 5),   # Sad: 低心率，低变异
        'neu': (70, 6),   # Neutral: 正常心率
        'fru': (75, 8),   # Frustrated: 中高心率
    }
    return emotion_hr_map.get(emotion, (70, 6))


def load_iemocap_data(data_dir):
    """
    加载IEMOCAP数据集
    
    Args:
        data_dir: 数据目录
    
    Returns:
        train_data, dev_data, test_data: 数据集
    """
    print(f"Loading IEMOCAP data from {data_dir}...")
    
    # 尝试不同的pickle文件名
    possible_names = [
        'train.pkl', 'dev.pkl', 'test.pkl',
        'iemocap_4_train.pkl', 'iemocap_4_dev.pkl', 'iemocap_4_test.pkl'
    ]
    
    data = {}
    for split in ['train', 'dev', 'test']:
        found = False
        for name_template in [f'{split}.pkl', f'iemocap_4_{split}.pkl']:
            pkl_path = os.path.join(data_dir, name_template)
            if os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as f:
                    data[split] = pickle.load(f)
                print(f"  Loaded {split}: {len(data[split])} samples")
                found = True
                break
        
        if not found:
            print(f"  Warning: {split} data not found")
            data[split] = []
    
    return data['train'], data['dev'], data['test']


def generate_rppg_for_dataset(samples, rppg_dim=64, fs=30):
    """
    为数据集中的所有样本生成rPPG特征
    
    Args:
        samples: 样本列表
        rppg_dim: rPPG特征维度
        fs: 采样率
    
    Returns:
        samples_with_rppg: 添加了rPPG特征的样本列表
    """
    print(f"Generating rPPG features for {len(samples)} samples...")
    
    for sample in tqdm(samples):
        num_utterances = len(sample.text)
        sample.rppg = []
        
        for idx in range(num_utterances):
            # 获取情绪标签
            emotion_label = sample.label[idx]
            
            # 情绪到字符串的映射（IEMOCAP-4）
            emotion_map = {0: 'hap', 1: 'sad', 2: 'neu', 3: 'ang'}
            emotion_str = emotion_map.get(emotion_label, 'neu')
            
            # 根据情绪调整心率参数
            hr_mean, hr_std = emotion_specific_hr(emotion_str)
            
            # 生成rPPG信号
            rppg_signal = generate_realistic_rppg(
                num_frames=rppg_dim,
                hr_mean=hr_mean,
                hr_std=hr_std,
                fs=fs,
                noise_level=0.15  # 适度噪声
            )
            
            sample.rppg.append(rppg_signal)
    
    return samples


def save_rppg_features(samples, output_path):
    """
    保存带有rPPG特征的样本
    
    Args:
        samples: 样本列表
        output_path: 输出文件路径
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(samples, f)
    
    print(f"Saved to {output_path}")


def verify_rppg_quality(samples, num_check=5):
    """
    验证生成的rPPG信号质量
    
    Args:
        samples: 样本列表
        num_check: 检查的样本数量
    """
    from joyful.rppg_extractor import RPPGQualityChecker
    
    print(f"\nVerifying rPPG quality for {num_check} random samples...")
    
    check_samples = np.random.choice(len(samples), min(num_check, len(samples)), replace=False)
    
    valid_count = 0
    quality_scores = []
    
    for idx in check_samples:
        sample = samples[idx]
        if len(sample.rppg) > 0:
            rppg_signal = torch.tensor(sample.rppg[0], dtype=torch.float32)
            
            # 质量检测
            is_valid, quality_score = RPPGQualityChecker.comprehensive_check(
                rppg_signal, fs=30
            )
            
            if is_valid:
                valid_count += 1
            quality_scores.append(quality_score)
            
            print(f"  Sample {idx}: valid={is_valid}, quality={quality_score:.3f}")
    
    avg_quality = np.mean(quality_scores)
    print(f"\nQuality Summary:")
    print(f"  Valid rate: {valid_count}/{len(check_samples)} ({valid_count/len(check_samples)*100:.1f}%)")
    print(f"  Average quality: {avg_quality:.3f}")
    
    if avg_quality > 0.5:
        print("  ✅ Quality check passed!")
    else:
        print("  ⚠️  Warning: Low average quality")


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic rPPG features for IEMOCAP')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory containing IEMOCAP pickle files')
    parser.add_argument('--output_dir', type=str, default='./data/rppg_features',
                        help='Output directory for rPPG features')
    parser.add_argument('--rppg_dim', type=int, default=64,
                        help='Dimension of rPPG features')
    parser.add_argument('--fs', type=int, default=30,
                        help='Sampling rate (fps)')
    parser.add_argument('--verify', action='store_true',
                        help='Verify rPPG quality after generation')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Synthetic rPPG Feature Generation")
    print("=" * 60)
    
    # 加载数据集
    train_data, dev_data, test_data = load_iemocap_data(args.data_dir)
    
    if not train_data and not dev_data and not test_data:
        print("\n❌ Error: No data found!")
        print("Please check the data directory path.")
        return
    
    # 为每个split生成rPPG特征
    splits = {
        'train': train_data,
        'dev': dev_data,
        'test': test_data
    }
    
    for split_name, split_data in splits.items():
        if not split_data:
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing {split_name} split")
        print(f"{'='*60}")
        
        # 生成rPPG特征
        split_data_with_rppg = generate_rppg_for_dataset(
            split_data,
            rppg_dim=args.rppg_dim,
            fs=args.fs
        )
        
        # 保存
        output_path = os.path.join(args.output_dir, f'{split_name}_with_rppg.pkl')
        save_rppg_features(split_data_with_rppg, output_path)
        
        # 验证质量
        if args.verify:
            verify_rppg_quality(split_data_with_rppg, num_check=5)
    
    print(f"\n{'='*60}")
    print("Generation Complete!")
    print(f"{'='*60}")
    print(f"\nNext steps:")
    print(f"1. Update data loading in train.py to use:")
    print(f"   --data_dir {args.output_dir}")
    print(f"2. Run training with --use_rppg flag")
    print(f"3. Monitor rPPG usage rate (should be ~80-90%)")


if __name__ == '__main__':
    main()





