"""
JOYFUL模型 - 集成层次化动态门控机制版本
在原有JOYFUL基础上，增加：
1. 内层话语级多模态门控
2. 外层对话级时序门控
3. 层次融合模块
"""

import torch
import torch.nn as nn

from .SeqContext import SeqContext
from .GNN import GNN
from .Classifier import Classifier
from .functions import batch_graphify
from .HierarchicalGating import (
    UtteranceLevelGating,
    DialogueLevelGating,
    HierarchicalFusion
)
import joyful

log = joyful.utils.get_logger()


class JOYFUL_Hierarchical(nn.Module):
    """
    JOYFUL模型 - 层次化动态门控版本
    """
    def __init__(self, args):
        super(JOYFUL_Hierarchical, self).__init__()
        u_dim = 100
        if args.rnn == "transformer":
            g_dim = args.hidden_size
        else:
            g_dim = 200
        h1_dim = args.hidden_size
        h2_dim = args.hidden_size
        hc_dim = args.hidden_size
        
        dataset_label_dict = {
            "iemocap": {"hap": 0, "sad": 1, "neu": 2, "ang": 3, "exc": 4, "fru": 5},
            "iemocap_4": {"hap": 0, "sad": 1, "neu": 2, "ang": 3},
            "mosei": {"Negative": 0, "Positive": 1},
            "meld": {"Neutral": 0, "Surprise": 1, "Fear": 2, "Sadness": 3, "Joy": 4, "Disgust": 5, "Angry": 6}
        }

        dataset_speaker_dict = {
            "iemocap": 2,
            "iemocap_4": 2,
            "mosei": 1,
            "meld": 9
        }

        if args.dataset and args.emotion == "multilabel":
            dataset_label_dict["mosei"] = {
                "happiness": 0,
                "sadness": 1,
                "anger": 2,
                "surprise": 3,
                "disgust": 4,
                "fear": 5,
            }

        tag_size = len(dataset_label_dict[args.dataset])
        args.n_speakers = dataset_speaker_dict[args.dataset]
        self.args = args  # 保存args以便后续使用
        self.concat_gin_gout = args.concat_gin_gout
        self.cl_loss_weight = args.cl_loss_weight
        self.use_hierarchical = getattr(args, 'use_hierarchical', True)

        self.wp = args.wp
        self.wf = args.wf
        self.device = args.device

        # 原始JOYFUL组件（保留用于兼容）
        self.rnn = SeqContext(u_dim, g_dim, args)
        self.gcl = GNN(g_dim, h1_dim, h2_dim, args)
        
        # 层次化门控模块
        if self.use_hierarchical:
            # 获取各模态的维度
            modal_dims = self._get_modal_dims(args)
            
            # 内层：话语级多模态门控
            self.utterance_gating = UtteranceLevelGating(
                modal_dims=modal_dims,
                hidden_dim=g_dim
            )
            
            # 外层：对话级时序门控
            # 设置默认参数（如果未提供）
            if not hasattr(args, 'temporal_rnn'):
                args.temporal_rnn = args.rnn  # 默认使用相同的RNN类型
            if not hasattr(args, 'temporal_nlayer'):
                args.temporal_nlayer = args.seqcontext_nlayer
            if not hasattr(args, 'temporal_nhead'):
                args.temporal_nhead = 8
            
            speaker_dim = 16
            position_dim = 16
            
            self.dialogue_gating = DialogueLevelGating(
                input_dim=g_dim,
                hidden_dim=g_dim,
                args=args,
                speaker_dim=speaker_dim,
                position_dim=position_dim
            )
            
            # 层次融合模块
            self.hierarchical_fusion = HierarchicalFusion(
                utterance_dim=g_dim,
                dialogue_dim=g_dim,
                output_dim=g_dim
            )
        
        # 分类器
        if args.concat_gin_gout:
            self.clf = Classifier(
                g_dim + h2_dim * args.gnn_nheads, hc_dim, tag_size, args
            )
        else:
            self.clf = Classifier(h2_dim * args.gnn_nheads, hc_dim, tag_size, args)

        edge_type_to_idx = {}
        for j in range(args.n_speakers):
            for k in range(args.n_speakers):
                edge_type_to_idx[str(j) + str(k) + "0"] = len(edge_type_to_idx)
                edge_type_to_idx[str(j) + str(k) + "1"] = len(edge_type_to_idx)
        self.edge_type_to_idx = edge_type_to_idx
        log.debug(self.edge_type_to_idx)

    def _get_modal_dims(self, args):
        """
        根据数据集和模态配置获取各模态的维度
        """
        dataset = args.dataset
        modalities = args.modalities
        
        # 定义各数据集的模态维度
        dims = {
            "iemocap": {"a": 50, "t": 256, "v": 256},
            "iemocap_4": {"a": 50, "t": 256, "v": 256},
            "mosei": {"a": 80, "t": 768, "v": 35},
            "meld": {"a": 100, "t": 768, "v": 512}
        }
        
        base_dims = dims.get(dataset, {"a": 50, "t": 256, "v": 256})
        
        # 根据实际使用的模态构建modal_dims
        modal_dims = {}
        if "a" in modalities:
            modal_dims["a"] = base_dims["a"]
        if "t" in modalities:
            modal_dims["t"] = base_dims["t"]
        if "v" in modalities:
            modal_dims["v"] = base_dims["v"]
        
        return modal_dims

    def _extract_modal_features(self, data):
        """
        从输入数据中提取各模态特征
        注意：当前Dataset已经通过AutoFusion融合了特征
        这个方法用于未来支持原始模态特征的情况
        """
        # 如果数据中包含原始模态特征，直接使用
        if "audio_tensor" in data and "text_tensor" in data and "visual_tensor" in data:
            return data["audio_tensor"], data["text_tensor"], data["visual_tensor"]
        
        # 否则，从融合后的特征中无法分离，返回None
        # 这种情况下，内层门控将使用融合后的特征
        return None, None, None

    def get_rep(self, data=None, whetherT=None):
        """
        获取表示
        如果使用层次化门控，则先经过内层和外层处理
        """
        if self.use_hierarchical:
            return self.get_rep_hierarchical(data, whetherT)
        else:
            # 使用原始JOYFUL的方式
            return self.get_rep_original(data, whetherT)

    def get_rep_original(self, data=None, whetherT=None):
        """
        原始JOYFUL的表示获取方式
        """
        node_features = self.rnn(data["text_len_tensor"], data["input_tensor"])

        features, edge_index, edge_type, edge_index_lengths = batch_graphify(
            node_features,
            data["text_len_tensor"],
            data["speaker_tensor"],
            self.wp,
            self.wf,
            self.edge_type_to_idx,
            self.device,
        )

        graph_out, cl_loss = self.gcl(features, edge_index, edge_type, whetherT)
        return graph_out, features, cl_loss

    def get_rep_hierarchical(self, data=None, whetherT=None):
        """
        层次化门控的表示获取方式
        """
        batch_size = data["input_tensor"].shape[0]
        seq_lens = data["text_len_tensor"]
        max_len = data["input_tensor"].shape[1]
        
        # 提取各模态特征（如果可用）
        a_features, t_features, v_features = self._extract_modal_features(data)
        
        # 如果无法提取原始模态特征，使用融合后的特征
        # 这种情况下，内层门控将直接使用融合特征
        if a_features is None:
            # 使用融合后的特征作为输入
            fused_features = data["input_tensor"]
            
            # 内层：话语级处理（对每个话语应用门控）
            utterance_reps = []
            for i in range(batch_size):
                cur_len = seq_lens[i].item()
                utterance_seq = []
                for t in range(cur_len):
                    # 获取当前时间步的融合特征
                    feat = fused_features[i, t, :]  # [embedding_dim]
                    
                    # 由于无法分离模态，直接使用融合特征
                    # 这里可以添加一个投影层来适配
                    if not hasattr(self, '_feat_proj'):
                        self._feat_proj = nn.Linear(
                            fused_features.shape[-1],
                            self.utterance_gating.hidden_dim
                        ).to(self.device)
                    
                    proj_feat = self._feat_proj(feat.unsqueeze(0))  # [1, hidden_dim]
                    utterance_seq.append(proj_feat)
                
                utterance_seq = torch.cat(utterance_seq, dim=0)  # [cur_len, hidden_dim]
                utterance_reps.append(utterance_seq)
            
            # 填充到相同长度
            max_utterance_len = max([ur.shape[0] for ur in utterance_reps])
            padded_utterance_reps = []
            for ur in utterance_reps:
                pad_len = max_utterance_len - ur.shape[0]
                if pad_len > 0:
                    ur = torch.cat([ur, torch.zeros(pad_len, ur.shape[-1]).to(self.device)], dim=0)
                padded_utterance_reps.append(ur)
            
            utterance_features = torch.stack(padded_utterance_reps, dim=0)  # [batch, max_len, hidden_dim]
        else:
            # 如果有原始模态特征，使用内层门控
            utterance_features = []
            for i in range(batch_size):
                cur_len = seq_lens[i].item()
                utterance_seq = []
                for t in range(cur_len):
                    a = a_features[i, t, :] if a_features is not None else None
                    t_feat = t_features[i, t, :] if t_features is not None else None
                    v = v_features[i, t, :] if v_features is not None else None
                    
                    # 内层门控
                    utterance_rep = self.utterance_gating(a, t_feat, v)  # [1, hidden_dim]
                    utterance_seq.append(utterance_rep)
                
                utterance_seq = torch.cat(utterance_seq, dim=0)  # [cur_len, hidden_dim]
                utterance_features.append(utterance_seq)
            
            # 填充
            max_utterance_len = max([ur.shape[0] for ur in utterance_features])
            padded_utterance_reps = []
            for ur in utterance_features:
                pad_len = max_utterance_len - ur.shape[0]
                if pad_len > 0:
                    ur = torch.cat([ur, torch.zeros(pad_len, ur.shape[-1]).to(self.device)], dim=0)
                padded_utterance_reps.append(ur)
            
            utterance_features = torch.stack(padded_utterance_reps, dim=0)
        
        # 准备说话人信息（用于外层门控）
        speaker_info = None
        if "speaker_tensor" in data:
            # 将speaker转换为embedding
            if not hasattr(self, '_speaker_embedding'):
                num_speakers = data["speaker_tensor"].max().item() + 1
                self._speaker_embedding = nn.Embedding(num_speakers, 16).to(self.device)
            
            speaker_emb = self._speaker_embedding(data["speaker_tensor"])  # [batch, max_len, 16]
            speaker_info = speaker_emb
        
        # 准备位置信息
        position_info = None
        use_position_encoding = getattr(self.args, 'use_position_encoding', False)
        if use_position_encoding:
            # 位置编码
            positions = torch.arange(max_len).unsqueeze(0).expand(batch_size, -1).to(self.device)
            if not hasattr(self, '_position_embedding'):
                self._position_embedding = nn.Embedding(max_len, 16).to(self.device)
            position_info = self._position_embedding(positions)
        
        # 外层：对话级时序门控
        dialogue_features = self.dialogue_gating(
            utterance_features,
            lengths=seq_lens,
            speaker_info=speaker_info,
            position_info=position_info
        )  # [batch, max_len, hidden_dim]
        
        # 层次融合
        hierarchical_features = self.hierarchical_fusion(
            utterance_features,
            dialogue_features
        )  # [batch, max_len, hidden_dim]
        
        # 构建图（使用层次化特征）
        features, edge_index, edge_type, edge_index_lengths = batch_graphify(
            hierarchical_features,
            data["text_len_tensor"],
            data["speaker_tensor"],
            self.wp,
            self.wf,
            self.edge_type_to_idx,
            self.device,
        )
        
        # GNN处理
        graph_out, cl_loss = self.gcl(features, edge_index, edge_type, whetherT)
        
        return graph_out, features, cl_loss

    def forward(self, data, whetherT):
        graph_out, features, cl_loss = self.get_rep(data, whetherT)
        if self.concat_gin_gout:
            out = self.clf(
                torch.cat([features, graph_out], dim=-1), data["text_len_tensor"]
            )
        else:
            out = self.clf(graph_out, data["text_len_tensor"])

        return out

    def get_loss(self, data, whetherT):
        graph_out, features, cl_loss = self.get_rep(data, whetherT)
        if self.concat_gin_gout:
            loss = self.clf.get_loss(
                torch.cat([features, graph_out], dim=-1),
                data["label_tensor"],
                data["text_len_tensor"],
            )
        else:
            loss = self.clf.get_loss(
                graph_out, data["label_tensor"], data["text_len_tensor"]
            )
        return loss + self.cl_loss_weight * cl_loss

