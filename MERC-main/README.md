# Joyful
This is the official implementation for the paper "Joyful: Joint Modality Fusion and Graph Contrastive Learning for Multimodal Emotion Recognition" accepted by EMNLP2023.

# Structure of Joyful
![image](structure.png)

## Table of Contents

- [Dependencies](#security)
- [Requirement](#background)
- [Install](#install)
- [Dataset](#dataset)
- [Usage](#usage)
- [Maintainers](#maintainers)
- [Contributing](#contributing)
- [License](#license)

## Dependencies

- Python 3.9.1
- PyTorch toolbox (1.12.1+cu113)
- Linux 5.11.0-46-generic

The version of CUDA is very important, please use the same CUDA version for training. 
For fast check, the check point is provided.

## Requirement
- We use PyG (PyTorch Geometric) for the GNN component in our architecture. [RGCNConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.RGCNConv) and [TransformerConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.TransformerConv)
- We use PyGCL for GCL (Graph Contrastive Learning) network in out framework. [PyGCL](https://github.com/PyGCL/PyGCL)
- We use sentence transfomer for text feature extraction. [Sentence Embedding](https://huggingface.co/sentence-transformers/paraphrase-distilroberta-base-v1)


## Install
To easily reproduce our results, you can install the environments by
```
pip install -r requirements.txt
```


## Dataset

The IEMOCAP 4 classification dataset is store in
```
./data/iemocap_4/data_iemocap_4.pkl
```

The IEMOCAP 6 classification dataset is store in
```
./data/iemocap/data_iemocap.pkl
```

The Mosei dataset is avaliable on this official website
[Mosei](http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/) 


## Usage

### Train the model

```
python train.py --dataset="iemocap_4" --modalities="atv" --from_begin --epochs=50
```

### Evaluate the model

```
python eval.py --dataset="iemocap_4" --modalities="atv"
```

### Train the model use single modality (text)

```
python eval.py --dataset="iemocap_4" --modalities="t"
```

python train.py \
    --dataset="iemocap_4" \
    --modalities="atv" \
    --use_hierarchical_fusion \
    --encoder_loss_weight=0.03 \
    --use_smooth_l1 \
    --from_begin \
    --ulgm_text_only \
    --epochs=50

python train.py \
    --dataset="iemocap_4" \
    --modalities="atv" \
    --use_hierarchical_fusion \
    --from_begin \
    --epochs=50

python train.py \
    --dataset="iemocap_4" \
    --modalities="atv" \
    --use_hierarchical_fusion --encoder_loss_weight=0.03 \
    --from_begin \
    --epochs=50

python train.py \
    --dataset="iemocap_4" \
    --modalities="atv" \
    --use_hierarchical_fusion --encoder_loss_weight=0.03 --use_smooth_l1 \
    --from_begin \
    --epochs=50

<!-- 可以在项目根目录执行示例命令（含ULGM、自动类别权重、小学习率、开启梯度裁剪）： -->
cd MERC-mainpython JOYFUL/train.py \  --dataset iemocap_4 \  --use_ulgm \  --auto_class_weight \  --learning_rate 1e-5 \  --max_grad_norm 1.0 \  --max_grad_value -1 \  --unimodal_init_weight 0.0005 \  --unimodal_warmup_epochs 8 \  --unimodal_delay_epochs 3 \  --ulgm_happy_min_samples 10 \  --ulgm_happy_true_label_weight 0.7 \  --happy_early_boost 1.5
<!-- --use_ulgm 启用单模态伪标签监督模块。
--auto_class_weight 自动计算类别权重。
--learning_rate 1e-5 使用较小学习率。
--max_grad_norm 1.0（可改更小）开启梯度裁剪，--max_grad_value -1 表示仅按范数裁剪。 -->





