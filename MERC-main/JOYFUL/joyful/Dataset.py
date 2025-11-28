import math
import random
import torch

import numpy as np

from threading import current_thread


class Dataset:
    def __init__(self, samples, modelF, WT, args) -> None:
        self.samples = samples
        self.modelF = modelF
        self.batch_size = args.batch_size
        self.num_batches = math.ceil(len(self.samples) / args.batch_size)
        self.speaker_to_idx = {"M": 0, "F": 1}
        self.modalities = args.modalities
        self.dataset = args.dataset
        if WT:
            self.modelF.train()
        else:
            self.modelF.eval()

        self.embedding_dim = args.dataset_embedding_dims[args.dataset][args.modalities]

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
            
            for t, a, v, label in zip(s.sbert_sentence_embeddings, s.audio, s.visual, s.label):
                t = torch.tensor(t, dtype=torch.float32)
                a = torch.tensor(a, dtype=torch.float32)
                v = torch.tensor(v, dtype=torch.float32)
                
                if self.modalities == "atv":
                    # 检查是否启用ULGM，如果启用则传递标签
                    if hasattr(self.modelF, 'use_ulgm') and self.modelF.use_ulgm:
                        result = self.modelF(a, t, v, labels=label)
                        # 处理返回值（可能是2个或3个）
                        if isinstance(result, tuple) and len(result) == 3:
                            output, loss, unimodal_loss = result
                            if unimodal_loss is not None:
                                unimodal_losses.append(unimodal_loss)
                        else:
                            output, loss = result
                    else:
                        result = self.modelF(a, t, v)
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

        label_tensor = torch.tensor(labels).long()
        data = {
            "text_len_tensor": text_len_tensor,
            "input_tensor": input_tensor,
            "speaker_tensor": speaker_tensor,
            "label_tensor": label_tensor,
            "utterance_texts": utterances,
            "encoder_loss": losst
        }
        
        # 如果有单模态损失，添加到data中
        if unimodal_losses:
            data["unimodal_loss"] = sum(unimodal_losses)
        
        return data

    def shuffle(self):
        random.shuffle(self.samples)
