"""
从 IEMOCAP 原始对话视频(avi)中离线提取 rPPG 频域特征，并保存为 feature_map:
  utterance_id(str) -> List[float] (纯 Python list，避免 numpy pickle 跨版本不兼容)

适配你的数据结构（示例）：
  MERC-main/data/Session1/dialog/EmoEvaluation/*.txt
  MERC-main/data/Session1/dialog/avi/DivX/*.avi

注意：
- 若 Session2~5 缺少 dialog/avi 视频文件，本脚本会跳过并打印统计信息（此时无法生成 Ses02~Ses05 的 key）。
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass
class UtteranceSeg:
    utterance_id: str
    start: float
    end: float
    dialog_id: str


def parse_emoeval_file(txt_path: str) -> List[UtteranceSeg]:
    """
    解析 IEMOCAP dialog/EmoEvaluation/*.txt，抽取每条 utterance 的时间段与 utterance_id。
    常见行格式：
      [12.34 - 15.67]  Ses01F_impro01_F000  ...
    """
    segs: List[UtteranceSeg] = []
    patt = re.compile(
        r"\[(?P<st>\d+(?:\.\d+)?)\s*-\s*(?P<ed>\d+(?:\.\d+)?)\]\s+(?P<utt>Ses\d{2}[FM]_(?:impro|script)\d+[a-z]?(?:_\d+)?_[FM]\d{3})",
        re.IGNORECASE,
    )
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = patt.search(line)
            if not m:
                continue
            st = float(m.group("st"))
            ed = float(m.group("ed"))
            utt = m.group("utt")
            dialog_id = utt.rsplit("_", 1)[0]
            if ed > st:
                segs.append(UtteranceSeg(utterance_id=utt, start=st, end=ed, dialog_id=dialog_id))
    return segs


def find_dialog_video(session_dir: str, dialog_id: str) -> Optional[str]:
    """
    在 session 目录下寻找对话级视频文件：{dialog_id}.avi
    """
    # 常见目录：dialog/avi/DivX/
    candidates = [
        os.path.join(session_dir, "dialog", "avi", "DivX", f"{dialog_id}.avi"),
        os.path.join(session_dir, "dialog", "avi", f"{dialog_id}.avi"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    # 兜底：递归搜索（慢，但能救一些非常规结构）
    patt = os.path.join(session_dir, "**", "avi", "**", f"{dialog_id}.avi")
    hits = glob.glob(patt, recursive=True)
    for p in hits:
        if os.path.exists(p):
            return p
    return None


def _pos_rppg(rgb: np.ndarray) -> np.ndarray:
    """
    POS rPPG (Wang et al.)：输入 [T,3] 的 RGB 均值序列，输出 [T] rPPG waveform
    """
    eps = 1e-6
    X = rgb.copy().astype(np.float64)
    X = X / (X.mean(axis=0, keepdims=True) + eps) - 1.0

    # Projection
    x = 3.0 * X[:, 0] - 2.0 * X[:, 1]
    y = 1.5 * X[:, 0] + X[:, 1] - 1.5 * X[:, 2]
    sx = np.std(x) + eps
    sy = np.std(y) + eps
    s = x - (sx / sy) * y
    s = s - s.mean()
    return s.astype(np.float32)


def _bandpass_fft(x: np.ndarray, fs: float, fmin: float, fmax: float) -> np.ndarray:
    """
    纯 numpy FFT 频域带通滤波（避免 scipy 依赖）
    """
    n = len(x)
    if n <= 1:
        return x
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mask = (freqs >= fmin) & (freqs <= fmax)
    X_f = X * mask
    y = np.fft.irfft(X_f, n=n)
    y = y - y.mean()
    return y.astype(np.float32)


def _psd_feature(x: np.ndarray, fs: float, fmin: float, fmax: float, dim: int) -> Optional[np.ndarray]:
    """
    计算频带内 PSD，并等距聚合到 dim 维特征。
    """
    n = len(x)
    if n < 8:
        return None
    # Hann window
    w = np.hanning(n).astype(np.float32)
    xw = (x.astype(np.float32) * w).astype(np.float32)

    X = np.fft.rfft(xw)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    psd = (np.abs(X) ** 2).astype(np.float32)

    band = (freqs >= fmin) & (freqs <= fmax)
    if band.sum() < 4:
        return None
    f = freqs[band].astype(np.float32)
    p = psd[band].astype(np.float32)

    # dim bins over [fmin,fmax]
    edges = np.linspace(fmin, fmax, dim + 1, dtype=np.float32)
    feat = np.zeros((dim,), dtype=np.float32)
    for i in range(dim):
        lo, hi = edges[i], edges[i + 1]
        m = (f >= lo) & (f < hi) if i < dim - 1 else (f >= lo) & (f <= hi)
        if m.any():
            feat[i] = float(p[m].mean())

    s = float(feat.sum())
    if s <= 0:
        return None
    feat = feat / s
    return feat


def extract_rgb_trace_from_video(
    video_path: str,
    start_s: float,
    end_s: float,
    min_frames: int,
    face_scale: float = 0.5,
) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """
    从对话级视频里截取 [start_s, end_s] 段，提取每帧 ROI 的 RGB 均值序列 [T,3]。
    """
    try:
        import cv2
    except Exception:
        # 用英文避免 Windows 终端编码导致的乱码
        raise RuntimeError("Missing dependency: please install opencv-python (cv2)")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-3:
        fps = 30.0
    start_f = max(int(start_s * fps), 0)
    end_f = max(int(end_s * fps), start_f + 1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    rgb_list: List[List[float]] = []
    last_face = None
    fi = start_f
    while fi < end_f:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        fi += 1

        # BGR -> RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape

        # detect face (downscale for speed)
        small = cv2.resize(rgb, (int(w * face_scale), int(h * face_scale)))
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(40, 40))

        if len(faces) > 0:
            x, y, fw, fh = max(faces, key=lambda b: b[2] * b[3])
            # scale back
            x = int(x / face_scale)
            y = int(y / face_scale)
            fw = int(fw / face_scale)
            fh = int(fh / face_scale)
            last_face = (x, y, fw, fh)
        elif last_face is not None:
            x, y, fw, fh = last_face
        else:
            # fallback: center crop
            fw = int(w * 0.5)
            fh = int(h * 0.5)
            x = (w - fw) // 2
            y = (h - fh) // 2

        x0 = max(x, 0)
        y0 = max(y, 0)
        x1 = min(x + fw, w)
        y1 = min(y + fh, h)
        roi = rgb[y0:y1, x0:x1, :]
        if roi.size == 0:
            continue

        # 简单皮肤阈值（YCrCb），降低背景干扰
        ycrcb = cv2.cvtColor(roi, cv2.COLOR_RGB2YCrCb)
        Y, Cr, Cb = cv2.split(ycrcb)
        skin = (Cr >= 133) & (Cr <= 173) & (Cb >= 77) & (Cb <= 127)
        if skin.any():
            rr = roi[:, :, 0][skin].mean()
            gg = roi[:, :, 1][skin].mean()
            bb = roi[:, :, 2][skin].mean()
        else:
            rr, gg, bb = roi[:, :, 0].mean(), roi[:, :, 1].mean(), roi[:, :, 2].mean()
        rgb_list.append([rr, gg, bb])

    cap.release()
    if len(rgb_list) < min_frames:
        return None, float(fps)
    return np.asarray(rgb_list, dtype=np.float32), float(fps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iemocap_root", required=True, help="包含 Session1...Session5 的目录（例如 MERC-main/data）")
    ap.add_argument("--out_feature_map", required=True, help="输出 pkl 路径")
    ap.add_argument("--feature_dim", type=int, default=64)
    ap.add_argument("--min_frames", type=int, default=32)
    ap.add_argument("--fmin", type=float, default=0.7)
    ap.add_argument("--fmax", type=float, default=4.0)
    ap.add_argument("--max_utts", type=int, default=0, help="仅调试：限制处理 utterance 数量（0=不限制）")
    args = ap.parse_args()

    iemocap_root = os.path.abspath(args.iemocap_root)
    out_path = os.path.abspath(args.out_feature_map)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # 兼容：有些工程把 IEMOCAP 数据放在 MERC-main/JOYFUL/data 下
    # 这里自动检测 Session1..5 的真实根目录
    def _detect_root(root: str) -> str:
        # root 本身包含 Session?
        if any(os.path.isdir(os.path.join(root, f"Session{i}")) for i in range(1, 6)):
            return root
        # root/..（例如传了 .../JOYFUL/data，但 Session 在 .../data）
        parent = os.path.abspath(os.path.join(root, os.pardir))
        if any(os.path.isdir(os.path.join(parent, f"Session{i}")) for i in range(1, 6)):
            return parent
        # root/data（例如传了 .../JOYFUL，但 Session 在 .../JOYFUL/data）
        child = os.path.join(root, "data")
        if any(os.path.isdir(os.path.join(child, f"Session{i}")) for i in range(1, 6)):
            return child
        return root

    iemocap_root = _detect_root(iemocap_root)
    sessions = [os.path.join(iemocap_root, f"Session{i}") for i in range(1, 6)]

    feature_map: Dict[str, List[float]] = {}
    stats = {
        "sessions_seen": 0,
        "eval_files": 0,
        "utterances_total": 0,
        "utterances_video_missing": 0,
        "utterances_too_short": 0,
        "utterances_extracted": 0,
        "videos_missing": 0,
    }

    processed = 0
    for session_dir in sessions:
        if not os.path.isdir(session_dir):
            continue
        stats["sessions_seen"] += 1

        eval_dir = os.path.join(session_dir, "dialog", "EmoEvaluation")
        if not os.path.isdir(eval_dir):
            continue

        # 只取根目录下的 *.txt（不取 Attribute/Categorical 等子目录）
        eval_files = sorted(glob.glob(os.path.join(eval_dir, "Ses*.txt")))
        stats["eval_files"] += len(eval_files)

        # 每个 session 建一个 dialog_id -> video_path 缓存
        dialog_video_cache: Dict[str, Optional[str]] = {}

        for ef in eval_files:
            segs = parse_emoeval_file(ef)
            if not segs:
                continue
            stats["utterances_total"] += len(segs)

            for seg in segs:
                if args.max_utts > 0 and processed >= args.max_utts:
                    break

                if seg.dialog_id not in dialog_video_cache:
                    dialog_video_cache[seg.dialog_id] = find_dialog_video(session_dir, seg.dialog_id)
                video_path = dialog_video_cache[seg.dialog_id]
                if not video_path:
                    stats["utterances_video_missing"] += 1
                    continue

                rgb_trace, fps = extract_rgb_trace_from_video(
                    video_path,
                    seg.start,
                    seg.end,
                    min_frames=args.min_frames,
                )
                if rgb_trace is None or fps is None:
                    stats["utterances_too_short"] += 1
                    continue

                wave = _pos_rppg(rgb_trace)
                wave = _bandpass_fft(wave, fs=fps, fmin=args.fmin, fmax=args.fmax)
                feat = _psd_feature(wave, fs=fps, fmin=args.fmin, fmax=args.fmax, dim=args.feature_dim)
                if feat is None:
                    stats["utterances_too_short"] += 1
                    continue

                # 纯 Python list，避免 numpy pickle 版本问题
                feature_map[seg.utterance_id] = [float(x) for x in feat.tolist()]
                stats["utterances_extracted"] += 1
                processed += 1

            if args.max_utts > 0 and processed >= args.max_utts:
                break

        # 若 session 下完全没有视频目录，给一个明显提示
        if not os.path.isdir(os.path.join(session_dir, "dialog", "avi")):
            stats["videos_missing"] += 1

        if args.max_utts > 0 and processed >= args.max_utts:
            break

    with open(out_path, "wb") as f:
        pickle.dump(feature_map, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("[Done] Saved:", out_path)
    print("[Done] iemocap_root:", iemocap_root)
    print("[Done] Keys:", len(feature_map))
    print("[Stats]", stats)
    print(
        "提示：如果你希望覆盖 Ses01~Ses05，但 stats['utterances_video_missing'] 很高，"
        "通常是 Session2~5 缺少 dialog/avi/*.avi，需要先把视频补齐到对应 Session 目录。"
    )


if __name__ == "__main__":
    main()


