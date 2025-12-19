import math
import random
import torch

import numpy as np

from threading import current_thread
from joyful.rppg_extractor import RPPGQualityChecker
import pickle


class Dataset:
    def __init__(self, samples, modelF, WT, args) -> None:
        self.samples = samples
        self.modelF = modelF
        self.batch_size = args.batch_size
        self.num_batches = math.ceil(len(self.samples) / args.batch_size)
        self.speaker_to_idx = {"M": 0, "F": 1}
        self.modalities = args.modalities
        self.dataset = args.dataset
        # rPPG相关配置（可选）
        self.use_rppg = getattr(args, "use_rppg", False)
        self.rppg_raw_dim = getattr(args, "rppg_raw_dim", 64)
        self.rppg_quality_check = getattr(args, "rppg_quality_check", "basic")  # basic / comprehensive
        self.rppg_quality_threshold = getattr(args, "rppg_quality_threshold", 0.3)
        self.rppg_fs = getattr(args, "rppg_fs", 30)  # 采样率（帧率）
        # 离线rPPG特征映射：utterance_id -> feature(64,)
        self.rppg_feature_map_path = getattr(args, "rppg_feature_map", None)
        self.rppg_feature_map = None
        if self.use_rppg and self.rppg_feature_map_path:
            try:
                with open(self.rppg_feature_map_path, "rb") as f:
                    self.rppg_feature_map = pickle.load(f)
            except Exception:
                self.rppg_feature_map = None
        
        # rPPG统计信息
        self.rppg_stats = {
            'total_samples': 0,
            'valid_samples': 0,
            'invalid_samples': 0,
            'zero_samples': 0,
            'low_quality_samples': 0
        }
        
        if WT:
            self.modelF.train()
        else:
            self.modelF.eval()

        self.embedding_dim = args.dataset_embedding_dims[args.dataset][args.modalities]
    
    def reset_rppg_stats(self):
        """重置rPPG统计信息（每个epoch开始时调用）"""
        self.rppg_stats = {
            'total_samples': 0,
            'valid_samples': 0,
            'invalid_samples': 0,
            'zero_samples': 0,
            'low_quality_samples': 0
        }
    
    def get_rppg_stats(self):
        """获取rPPG统计信息"""
        if self.rppg_stats['total_samples'] > 0:
            valid_ratio = self.rppg_stats['valid_samples'] / self.rppg_stats['total_samples']
        else:
            valid_ratio = 0.0
        return self.rppg_stats, valid_ratio

    def __len__(self):
        return self.num_batches

    def __getitem__(self, index):
        batch = self.raw_batch(index)
        return self.padding(batch)

    def raw_batch(self, index):
        assert index < self.num_batches, "batch_idx %d > %d" % (index, self.num_batches)
        batch = self.samples[index * self.batch_size : (index + 1) * self.batch_size]

        return batch

    def padding(self, samples):
        batch_size = len(samples)
        text_len_tensor = torch.tensor([len(s.text) for s in samples]).long()
        mx = torch.max(text_len_tensor).item()

        input_tensor = torch.zeros((batch_size, mx, self.embedding_dim))
        speaker_tensor = torch.zeros((batch_size, mx)).long()
        labels = []
        utterances = []
        for i, s in enumerate(samples):
            cur_len = len(s.text)
            utterances.append(s.sentence)
            tmp = []
            losst = 0
            unimodal_losses = []  # ULGM单模态损失
            
            # 逐utterance处理，兼容可选的rPPG特征
            for idx in range(cur_len):
                t = torch.tensor(s.sbert_sentence_embeddings[idx], dtype=torch.float32)
                a = torch.tensor(s.audio[idx], dtype=torch.float32)
                v = torch.tensor(s.visual[idx], dtype=torch.float32)
                label = s.label[idx]
                
                # rPPG特征：如不存在则使用零向量，保证向后兼容
                if self.use_rppg:
                    self.rppg_stats['total_samples'] += 1
                    
                    # 优先：Sample对象内已存在rppg / rppg_features
                    if hasattr(s, "rppg") and len(s.rppg) > idx:
                        rppg_feat = torch.tensor(s.rppg[idx], dtype=torch.float32)
                    elif hasattr(s, "rppg_features") and len(s.rppg_features) > idx:
                        rppg_feat = torch.tensor(s.rppg_features[idx], dtype=torch.float32)
                    # 次优：从离线feature_map按utterance_id加载（推荐）
                    elif self.rppg_feature_map is not None and hasattr(s, "sentence") and len(s.sentence) > idx:
                        utt_id = s.sentence[idx]
                        feat = self.rppg_feature_map.get(utt_id, None)
                        if feat is not None:
                            rppg_feat = torch.tensor(feat, dtype=torch.float32)
                        else:
                            rppg_feat = torch.zeros(self.rppg_raw_dim, dtype=torch.float32)
                            self.rppg_stats['zero_samples'] += 1
                    else:
                        rppg_feat = torch.zeros(self.rppg_raw_dim, dtype=torch.float32)
                        self.rppg_stats['zero_samples'] += 1
                    
                    # 改进的质量检测：根据配置选择检测方法
                    if rppg_feat is not None:
                        if self.rppg_quality_check == "comprehensive":
                            # 综合质量检测：统计 + 频域
                            is_valid, quality_score = RPPGQualityChecker.comprehensive_check(
                                rppg_feat, fs=self.rppg_fs
                            )
                            if not is_valid or quality_score < self.rppg_quality_threshold:
                                rppg_feat = None
                                self.rppg_stats['low_quality_samples'] += 1
                            else:
                                self.rppg_stats['valid_samples'] += 1
                        else:
                            # 基础质量检测：零向量或低方差
                            abs_max = torch.abs(rppg_feat).max().item()
                            variance = torch.var(rppg_feat).item()
                            if abs_max < 1e-6 or variance < 1e-4:
                                rppg_feat = None  # 标记为无效，融合时完全跳过
                                self.rppg_stats['invalid_samples'] += 1
                            else:
                                self.rppg_stats['valid_samples'] += 1
                else:
                    rppg_feat = None
                
                if self.modalities == "atv":
                    # 检查是否启用ULGM，如果启用则传递标签
                    if hasattr(self.modelF, 'use_ulgm') and self.modelF.use_ulgm:
                        result = self.modelF(a, t, v, rppg=rppg_feat, labels=label)
                        # 处理返回值（可能是2个或3个）
                        if isinstance(result, tuple) and len(result) == 3:
                            output, loss, unimodal_loss = result
                            if unimodal_loss is not None:
                                unimodal_losses.append(unimodal_loss)
                        else:
                            if isinstance(result, tuple) and len(result) >= 2:
                                output, loss = result[0], result[1]
                            else:
                                output, loss = result
                    else:
                        result = self.modelF(a, t, v, rppg=rppg_feat)
                        # 为了兼容性，如果forward返回3个值，只取前2个
                        if isinstance(result, tuple) and len(result) == 3:
                            output, loss = result[0], result[1]
                        else:
                            output, loss = result
                    
                    tmp.append(output)
                    losst += loss
                elif self.modalities == "at":
                    output, loss = self.modelF(a, t)
                    tmp.append(output)
                    losst += loss
                elif self.modalities == "tv":
                    output, loss = self.modelF(t, v)
                    tmp.append(output)
                    losst += loss
                elif self.modalities == "av":
                    output, loss = self.modelF(a, v)
                    tmp.append(output)
                    losst += loss
                elif self.modalities == "a":
                    output, loss = self.modelF(a)
                    tmp.append(output.squeeze(0))
                    losst += loss
                elif self.modalities == "t":
                    output, loss = self.modelF(t)
                    tmp.append(output.squeeze(0))
                    losst += loss
                elif self.modalities == "v":
                    output, loss = self.modelF(v)
                    tmp.append(output.squeeze(0))
                    losst += loss

            tmp = torch.stack(tmp)
            input_tensor[i, :cur_len, :] = tmp
            if self.dataset in ["meld", "dailydialog"]:
                embed = torch.argmax(torch.tensor(s.speaker), dim=1)
                speaker_tensor[i, :cur_len] = embed
            else:
                speaker_tensor[i, :cur_len] = torch.tensor(
                    [self.speaker_to_idx[c] for c in s.speaker]
                )

            labels.extend(s.label)

            # 将encoder子损失按utterance数量归一化，避免长序列导致loss偏大
            if cur_len > 0:
                losst = losst / cur_len

        label_tensor = torch.tensor(labels).long()
        data = {
            "text_len_tensor": text_len_tensor,
            "input_tensor": input_tensor,
            "speaker_tensor": speaker_tensor,
            "label_tensor": label_tensor,
            "utterance_texts": utterances,
            "encoder_loss": losst
        }
        
        # 如果有单模态损失，添加到data中（归一化：按utterance数量平均）
        if unimodal_losses:
            # 归一化：避免batch中utterance数量导致的损失累积过大
            data["unimodal_loss"] = sum(unimodal_losses) / len(unimodal_losses) if len(unimodal_losses) > 0 else torch.tensor(0.0)
        
        return data

    def shuffle(self):
        random.shuffle(self.samples)
