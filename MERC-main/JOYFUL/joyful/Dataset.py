import math
import random
import torch

import numpy as np

from threading import current_thread
from joyful.rppg_extractor import RPPGQualityChecker
import pickle
import re
import os


class Dataset:
    @staticmethod
    def _safe_pickle_load(path: str):
        """
        兼容不同 NumPy 版本的 pickle：
        - 旧环境(NumPy1.x) 读取由 NumPy2.x dump 的对象时，可能报：No module named 'numpy._core'
        这里用自定义 Unpickler 将 numpy._core -> numpy.core 做重定向。
        """
        with open(path, "rb") as f:
            try:
                return pickle.load(f)
            except ModuleNotFoundError as e:
                if "numpy._core" not in str(e):
                    raise
                # fallback: remap numpy._core.* -> numpy.core.*
                class _CompatUnpickler(pickle.Unpickler):
                    def find_class(self, module, name):
                        if module.startswith("numpy._core"):
                            module = module.replace("numpy._core", "numpy.core", 1)
                        return super().find_class(module, name)

                f.seek(0)
                return _CompatUnpickler(f).load()

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
                self.rppg_feature_map = Dataset._safe_pickle_load(self.rppg_feature_map_path)
            except Exception:
                self.rppg_feature_map = None

        # IEMOCAP utterance_id 规范格式（也是当前 rPPG feature_map 的 key）：
        #   Ses01F_impro01_F000 / Ses03M_impro08a_F012 / Ses03M_impro05b_M023 / Ses01F_script02_1_F003
        # 使用分组正则并“规范化重建”，将大小写/路径等差异统一到标准 key。
        self._utt_id_regex = re.compile(
            r"Ses(?P<sess>\d{2})(?P<dg>[FM])_"
            r"(?P<kind>impro|script)(?P<num>\d+)(?P<suf>[a-z]?)"
            r"(?P<part>_\d+)?_"
            r"(?P<spk>[FM])(?P<idx>\d{3})",
            re.IGNORECASE,
        )
        
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

    def _canonical_utt_id(self, text: str):
        if not text:
            return None
        if not isinstance(text, str):
            text = str(text)
        m = self._utt_id_regex.search(text)
        if not m:
            return None
        sess = m.group("sess")
        dg = m.group("dg").upper()
        kind = m.group("kind").lower()
        num = m.group("num")
        suf = (m.group("suf") or "").lower()
        part = m.group("part") or ""
        spk = m.group("spk").upper()
        idx = m.group("idx")
        return f"Ses{sess}{dg}_{kind}{num}{suf}{part}_{spk}{idx}"

    def _speaker_count_index(self, speakers, idx: int, spk: str):
        """
        IEMOCAP 的 utterance 命名通常是按 speaker 各自从 000 递增（F000... / M000...），
        而训练数据的 idx 是 turn index。这里计算“截至 idx 的该 speaker 第几次出现(从0开始)”。
        """
        if speakers is None:
            return None
        try:
            spk = (spk or "").strip().upper()
            if spk not in ("F", "M"):
                return None
            cnt = 0
            for j in range(0, idx + 1):
                sj = speakers[j]
                if isinstance(sj, (list, tuple)) and len(sj) > 0:
                    sj = sj[0]
                if isinstance(sj, str) and sj.strip():
                    if sj.strip()[0].upper() == spk:
                        cnt += 1
            return max(cnt - 1, 0)
        except Exception:
            return None

    def _lookup_rppg_from_map(self, sample, idx: int):
        """
        从离线 feature_map 中尽可能鲁棒地获取 utterance 的 rPPG 特征。
        关键：iemocap_4 的 sample.sentence[idx] 多为“文本”，不是 utterance_id；
        需要用 dialog_id(sample.vid) + speaker + (turn_idx 或 speaker_idx) 去拼 key。
        """
        if self.rppg_feature_map is None:
            return None

        tried = set()

        def _try_key(k: str):
            if not k:
                return None
            if k in tried:
                return None
            tried.add(k)
            return self.rppg_feature_map.get(k, None)

        def _yield_keys_from_string(x):
            if x is None:
                return
            if not isinstance(x, str):
                x = str(x)
            x = x.strip()
            if not x:
                return
            # 1) 原字符串
            yield x
            # 2) basename/stem
            base = os.path.basename(x)
            if base and base != x:
                yield base
            stem, _ext = os.path.splitext(base)
            if stem and stem != base:
                yield stem
            # 3) 规范化 utterance_id（若包含）
            canon = self._canonical_utt_id(x)
            if canon:
                yield canon

        # A) 先尝试 sentence[idx] / vid / vid[idx] 里如果直接包含 utterance_id
        try:
            if hasattr(sample, "sentence") and sample.sentence is not None and len(sample.sentence) > idx:
                for k in _yield_keys_from_string(sample.sentence[idx]):
                    feat = _try_key(k)
                    if feat is not None:
                        return feat
        except Exception:
            pass

        dialog_id = None
        try:
            if hasattr(sample, "vid"):
                v = sample.vid
                if isinstance(v, (list, tuple)) and len(v) > idx:
                    dialog_id = v[idx]
                elif isinstance(v, str):
                    dialog_id = v
        except Exception:
            dialog_id = None

        if dialog_id is not None:
            for k in _yield_keys_from_string(dialog_id):
                # 有些数据里 vid 直接就是 utterance_id
                feat = _try_key(k)
                if feat is not None:
                    return feat

        # B) 主路径：dialog_id + speaker + index
        spk = None
        speakers = None
        try:
            if hasattr(sample, "speaker"):
                speakers = sample.speaker
                if speakers is not None and len(speakers) > idx:
                    s0 = speakers[idx]
                    if isinstance(s0, (list, tuple)) and len(s0) > 0:
                        s0 = s0[0]
                    if isinstance(s0, str) and s0.strip():
                        spk = s0.strip()[0].upper()
        except Exception:
            spk = None

        if isinstance(dialog_id, str) and dialog_id.strip():
            did = dialog_id.strip()
            # 同时尝试 turn_idx 和 speaker_idx（以及 +1 的变体，规避不同实现的 off-by-one）
            idx_turn0 = idx
            idx_turn1 = idx + 1

            spk_candidates = []
            if spk in ("F", "M"):
                spk_candidates = [spk]
            else:
                # speaker 缺失时，保守尝试两种（宁可多查几次，也要把命中率拉满）
                spk_candidates = ["F", "M"]

            for sp in spk_candidates:
                idx_sp0 = self._speaker_count_index(speakers, idx, sp)
                idx_sp1 = (idx_sp0 + 1) if idx_sp0 is not None else None

                candidate_numbers = []
                candidate_numbers.append(("turn", idx_turn0))
                candidate_numbers.append(("turn", idx_turn1))
                if idx_sp0 is not None:
                    candidate_numbers.append(("spk", idx_sp0))
                if idx_sp1 is not None:
                    candidate_numbers.append(("spk", idx_sp1))

                for _kind, num in candidate_numbers:
                    k = f"{did}_{sp}{int(num):03d}"
                    feat = _try_key(k)
                    if feat is not None:
                        return feat

        return None
    
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
                        feat = self._lookup_rppg_from_map(s, idx)
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
