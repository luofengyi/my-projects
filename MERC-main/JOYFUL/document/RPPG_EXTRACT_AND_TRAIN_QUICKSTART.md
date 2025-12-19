# rPPG：按论文流程从IEMOCAP视频提取 + 接入训练（最短路径）

## 1) 先离线构建 rPPG feature map（utterance_id -> 64维）

> 依据论文图示：人脸定位 → 皮肤分割 → rPPG波形(POS/CHROM) → 带通 → PSD特征(64维)

在 PowerShell 中执行：

```powershell
cd MERC-main\JOYFUL
pip install opencv-python
python build_rppg_feature_map_iemocap.py `
  --iemocap_root .\data `
  --out_feature_map .\data\rppg\iemocap_rppg_map.pkl `
  --target_frames 160 `
  --feature_dim 64 `
  --method pos
```

运行结束会输出：
- `extracted utterances: N`

如果 `N` 很小或为0，说明你当前`data`目录下**缺少对应Session的dialog视频**（比如Session4没有`dialog/avi`），需要补齐视频文件或换到含完整视频的IEMOCAP目录。

## 2) 训练时启用 rPPG（真正使用，不再是零向量）

```powershell
cd MERC-main\JOYFUL
python train.py `
  --dataset iemocap_4 `
  --modalities atv `
  --device cuda `
  --epochs 100 `
  --batch_size 32 `
  --learning_rate 3e-5 `
  --use_hierarchical_fusion `
  --use_ulgm `
  --use_rppg `
  --rppg_feature_map .\data\rppg\iemocap_rppg_map.pkl `
  --rppg_raw_dim 64 `
  --rppg_proj_dim 460 `
  --rppg_quality_check comprehensive `
  --rppg_quality_threshold 0.3 `
  --rppg_fs 30 `
  --encoder_loss_weight 0.01 `
  --fusion_recon_weight 0.02 `
  --gate_reg_weight 0 `
  --global_residual_alpha 0.3
```

你应该在训练日志里看到：

```
Epoch 1: rPPG valid samples: xxx/xxxx (yy.y%)
```

如果仍然是 `0/xxxx (0.0%)`：
- 说明 feature map 里没有匹配到训练时的 `utterance_id`（常见于数据pkl里的`sentence`字段格式不同）
- 或者你的训练数据不是来自同一套IEMOCAP utterance id

这时把你训练时 `data_iemocap_4.pkl` 里任意一个 `s.sentence[0]` 的样例字符串发我（或让我读取一下该pkl路径），我可以把映射规则对齐。


