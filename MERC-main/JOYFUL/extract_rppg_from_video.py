"""
从视频中提取真实rPPG信号
需要原始视频文件（.avi, .mp4等）

依赖:
    pip install opencv-python mediapipe scipy

使用方法:
    python extract_rppg_from_video.py --video_dir /path/to/videos --output_dir ./data/rppg_features

注意: 此脚本需要视频文件，不能使用音频文件
"""

import os
import argparse
import pickle
import cv2
import numpy as np
import torch
from tqdm import tqdm
import mediapipe as mp
from scipy import signal
from joyful.rppg_extractor import RPPGQualityChecker


class FaceDetector:
    """面部检测器（使用MediaPipe）"""
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
    
    def detect(self, frame):
        """
        检测面部区域
        
        Args:
            frame: [H, W, 3] RGB图像
        
        Returns:
            bbox: (x, y, w, h) 面部边界框，或None
        """
        results = self.face_detection.process(frame)
        
        if not results.detections:
            return None
        
        # 使用第一个检测到的面部
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        
        h, w, _ = frame.shape
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        width = int(bbox.width * w)
        height = int(bbox.height * h)
        
        # 边界检查
        x = max(0, x)
        y = max(0, y)
        width = min(width, w - x)
        height = min(height, h - y)
        
        return (x, y, width, height)


class SkinSegmentation:
    """皮肤分割（基于颜色空间）"""
    @staticmethod
    def segment(face_roi):
        """
        分割皮肤区域
        
        Args:
            face_roi: [H, W, 3] RGB面部图像
        
        Returns:
            skin_mask: [H, W] 二值掩码
        """
        # 转换到YCrCb色彩空间
        ycrcb = cv2.cvtColor(face_roi, cv2.COLOR_RGB2YCrCb)
        
        # 皮肤颜色范围（经验值）
        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)
        
        skin_mask = cv2.inRange(ycrcb, lower, upper)
        
        # 形态学操作去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        
        return skin_mask


class RPPGExtractorFromVideo:
    """从视频中提取rPPG信号"""
    def __init__(self, method='green'):
        """
        Args:
            method: 提取方法 ('green', 'ica', 'pos')
        """
        self.method = method
        self.face_detector = FaceDetector()
        self.skin_segmentation = SkinSegmentation()
    
    def extract_from_video(self, video_path, target_frames=150, fps=30):
        """
        从视频中提取rPPG信号
        
        Args:
            video_path: 视频文件路径
            target_frames: 目标帧数
            fps: 目标帧率
        
        Returns:
            rppg_signal: [target_frames] rPPG信号，或None（如果失败）
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"  ❌ Cannot open video: {video_path}")
            return None
        
        # 读取所有帧
        frames = []
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            print(f"  ❌ No frames extracted from {video_path}")
            return None
        
        # 重采样到目标帧率
        if original_fps != fps:
            frames = self._resample_frames(frames, original_fps, fps)
        
        # 限制帧数
        if len(frames) > target_frames:
            # 取中间部分
            start = (len(frames) - target_frames) // 2
            frames = frames[start:start + target_frames]
        elif len(frames) < target_frames:
            # 填充（重复最后一帧）
            frames.extend([frames[-1]] * (target_frames - len(frames)))
        
        # 提取rPPG信号
        rppg_signal = self._extract_rppg_signal(frames, fps)
        
        return rppg_signal
    
    def _resample_frames(self, frames, original_fps, target_fps):
        """重采样帧序列"""
        num_frames = len(frames)
        original_times = np.arange(num_frames) / original_fps
        target_times = np.arange(int(num_frames * target_fps / original_fps)) / target_fps
        
        # 使用线性插值重采样（简化版）
        indices = np.interp(target_times, original_times, np.arange(num_frames))
        indices = np.round(indices).astype(int)
        indices = np.clip(indices, 0, num_frames - 1)
        
        return [frames[i] for i in indices]
    
    def _extract_rppg_signal(self, frames, fps):
        """
        从帧序列中提取rPPG信号
        
        Args:
            frames: 帧列表
            fps: 帧率
        
        Returns:
            rppg_signal: rPPG信号
        """
        # 提取面部ROI和皮肤区域的RGB信号
        rgb_signals = []
        
        for frame in frames:
            # 检测面部
            bbox = self.face_detector.detect(frame)
            
            if bbox is None:
                # 使用上一帧的信号（或零）
                if len(rgb_signals) > 0:
                    rgb_signals.append(rgb_signals[-1])
                else:
                    rgb_signals.append([0, 0, 0])
                continue
            
            x, y, w, h = bbox
            face_roi = frame[y:y+h, x:x+w]
            
            # 皮肤分割
            skin_mask = self.skin_segmentation.segment(face_roi)
            
            # 计算皮肤区域的平均RGB值
            if np.sum(skin_mask) > 0:
                skin_pixels = face_roi[skin_mask > 0]
                mean_rgb = np.mean(skin_pixels, axis=0)
            else:
                # 如果没有皮肤区域，使用整个面部
                mean_rgb = np.mean(face_roi.reshape(-1, 3), axis=0)
            
            rgb_signals.append(mean_rgb)
        
        rgb_signals = np.array(rgb_signals)  # [num_frames, 3]
        
        # 根据方法提取rPPG信号
        if self.method == 'green':
            # Green channel method（最简单）
            raw_signal = rgb_signals[:, 1]  # 绿色通道
        elif self.method == 'chrom':
            # CHROM method
            raw_signal = self._chrom_method(rgb_signals)
        else:
            # 默认使用绿色通道
            raw_signal = rgb_signals[:, 1]
        
        # 去趋势（去除直流分量和低频漂移）
        raw_signal = signal.detrend(raw_signal)
        
        # 带通滤波（0.7-4 Hz，对应42-240 bpm）
        b, a = signal.butter(4, [0.7, 4.0], btype='bandpass', fs=fps)
        filtered_signal = signal.filtfilt(b, a, raw_signal)
        
        # 归一化
        filtered_signal = filtered_signal / (np.std(filtered_signal) + 1e-8)
        
        return filtered_signal.astype(np.float32)
    
    def _chrom_method(self, rgb_signals):
        """
        CHROM方法（Chrominance-based method）
        论文: De Haan, G., & Jeanne, V. (2013)
        """
        # 归一化
        mean_rgb = np.mean(rgb_signals, axis=0, keepdims=True)
        normalized = rgb_signals / (mean_rgb + 1e-8)
        
        # 色度信号
        x_s = 3 * normalized[:, 0] - 2 * normalized[:, 1]
        y_s = 1.5 * normalized[:, 0] + normalized[:, 1] - 1.5 * normalized[:, 2]
        
        # CHROM信号
        alpha = np.std(x_s) / (np.std(y_s) + 1e-8)
        rppg = x_s - alpha * y_s
        
        return rppg


def load_iemocap_data(data_dir):
    """加载IEMOCAP数据集"""
    print(f"Loading IEMOCAP data from {data_dir}...")
    
    data = {}
    for split in ['train', 'dev', 'test']:
        pkl_path = os.path.join(data_dir, f'{split}.pkl')
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                data[split] = pickle.load(f)
            print(f"  Loaded {split}: {len(data[split])} samples")
        else:
            data[split] = []
    
    return data['train'], data['dev'], data['test']


def find_video_file(video_dir, dialog_id, utterance_id):
    """
    查找视频文件
    
    Args:
        video_dir: 视频目录
        dialog_id: 对话ID（例如：Ses01F_impro01）
        utterance_id: 话语ID（例如：Ses01F_impro01_F000）
    
    Returns:
        video_path: 视频文件路径，或None
    """
    # 可能的视频文件名模式
    possible_patterns = [
        f"{utterance_id}.avi",
        f"{utterance_id}.mp4",
        f"{dialog_id}.avi",
        f"{dialog_id}.mp4",
    ]
    
    for pattern in possible_patterns:
        video_path = os.path.join(video_dir, pattern)
        if os.path.exists(video_path):
            return video_path
    
    return None


def extract_rppg_for_dataset(samples, video_dir, rppg_dim=64, fps=30, method='green'):
    """
    为数据集提取rPPG特征
    
    Args:
        samples: 样本列表
        video_dir: 视频目录
        rppg_dim: rPPG特征维度
        fps: 目标帧率
        method: rPPG提取方法
    
    Returns:
        samples_with_rppg: 添加了rPPG特征的样本列表
        extraction_stats: 提取统计信息
    """
    extractor = RPPGExtractorFromVideo(method=method)
    
    stats = {
        'total_utterances': 0,
        'successful': 0,
        'failed': 0,
        'video_not_found': 0
    }
    
    print(f"Extracting rPPG features from videos...")
    
    for sample in tqdm(samples):
        num_utterances = len(sample.text)
        sample.rppg = []
        
        # 获取对话ID（从第一个句子推断）
        if hasattr(sample, 'sentence') and len(sample.sentence) > 0:
            first_utterance = sample.sentence[0]
            # 例如：Ses01F_impro01_F000 -> Ses01F_impro01
            dialog_id = '_'.join(first_utterance.split('_')[:3])
        else:
            dialog_id = None
        
        for idx in range(num_utterances):
            stats['total_utterances'] += 1
            
            # 获取话语ID
            utterance_id = sample.sentence[idx] if hasattr(sample, 'sentence') else None
            
            # 查找视频文件
            video_path = find_video_file(video_dir, dialog_id, utterance_id)
            
            if video_path is None:
                # 视频未找到，使用零向量占位
                sample.rppg.append(np.zeros(rppg_dim, dtype=np.float32))
                stats['video_not_found'] += 1
                continue
            
            # 提取rPPG信号
            rppg_signal = extractor.extract_from_video(video_path, target_frames=rppg_dim, fps=fps)
            
            if rppg_signal is not None:
                # 验证质量
                rppg_tensor = torch.tensor(rppg_signal, dtype=torch.float32)
                is_valid, quality_score = RPPGQualityChecker.comprehensive_check(rppg_tensor, fs=fps)
                
                if is_valid:
                    sample.rppg.append(rppg_signal)
                    stats['successful'] += 1
                else:
                    # 质量不合格，使用零向量
                    sample.rppg.append(np.zeros(rppg_dim, dtype=np.float32))
                    stats['failed'] += 1
            else:
                # 提取失败
                sample.rppg.append(np.zeros(rppg_dim, dtype=np.float32))
                stats['failed'] += 1
    
    return samples, stats


def main():
    parser = argparse.ArgumentParser(description='Extract rPPG features from videos')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory containing IEMOCAP pickle files')
    parser.add_argument('--video_dir', type=str, required=True,
                        help='Directory containing video files (.avi, .mp4)')
    parser.add_argument('--output_dir', type=str, default='./data/rppg_features',
                        help='Output directory for rPPG features')
    parser.add_argument('--rppg_dim', type=int, default=64,
                        help='Dimension of rPPG features')
    parser.add_argument('--fps', type=int, default=30,
                        help='Target frame rate')
    parser.add_argument('--method', type=str, default='green', choices=['green', 'chrom'],
                        help='rPPG extraction method')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("rPPG Feature Extraction from Videos")
    print("=" * 60)
    
    # 检查视频目录
    if not os.path.exists(args.video_dir):
        print(f"\n❌ Error: Video directory not found: {args.video_dir}")
        return
    
    # 加载数据集
    train_data, dev_data, test_data = load_iemocap_data(args.data_dir)
    
    if not train_data and not dev_data and not test_data:
        print("\n❌ Error: No data found!")
        return
    
    # 为每个split提取rPPG特征
    splits = {
        'train': train_data,
        'dev': dev_data,
        'test': test_data
    }
    
    total_stats = {'total_utterances': 0, 'successful': 0, 'failed': 0, 'video_not_found': 0}
    
    for split_name, split_data in splits.items():
        if not split_data:
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing {split_name} split")
        print(f"{'='*60}")
        
        # 提取rPPG特征
        split_data_with_rppg, stats = extract_rppg_for_dataset(
            split_data,
            video_dir=args.video_dir,
            rppg_dim=args.rppg_dim,
            fps=args.fps,
            method=args.method
        )
        
        # 更新总统计
        for key in total_stats:
            total_stats[key] += stats[key]
        
        # 打印统计信息
        print(f"\nExtraction Stats for {split_name}:")
        print(f"  Total utterances: {stats['total_utterances']}")
        print(f"  Successful: {stats['successful']} ({stats['successful']/stats['total_utterances']*100:.1f}%)")
        print(f"  Failed: {stats['failed']}")
        print(f"  Video not found: {stats['video_not_found']}")
        
        # 保存
        output_path = os.path.join(args.output_dir, f'{split_name}_with_rppg.pkl')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(split_data_with_rppg, f)
        print(f"Saved to {output_path}")
    
    # 总体统计
    print(f"\n{'='*60}")
    print("Overall Extraction Stats")
    print(f"{'='*60}")
    print(f"Total utterances: {total_stats['total_utterances']}")
    print(f"Successful: {total_stats['successful']} ({total_stats['successful']/total_stats['total_utterances']*100:.1f}%)")
    print(f"Failed: {total_stats['failed']}")
    print(f"Video not found: {total_stats['video_not_found']}")
    
    if total_stats['successful'] / total_stats['total_utterances'] > 0.5:
        print("\n✅ Extraction successful!")
    else:
        print("\n⚠️  Warning: Low success rate. Check video directory structure.")


if __name__ == '__main__':
    main()





