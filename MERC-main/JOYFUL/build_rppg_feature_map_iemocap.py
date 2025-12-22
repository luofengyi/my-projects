"""
从IEMOCAP对话视频中离线提取 rPPG 特征（utterance_id -> 64维）

本脚本对 IEMOCAP 目录结构做了“容错搜索”：
- 你可以把 --iemocap_root 指向 IEMOCAP 根目录（包含 Session1..5）
- 也可以指向更深的子目录（例如 dialog/avi/DivX），脚本会从视频路径反推 dialog/EmoEvaluation

输出：
- pickle：utterance_id(str) -> np.ndarray(shape=(64,), dtype=float32)

依赖：
- opencv-python（用于读视频、人脸检测）
"""

from __future__ import annotations

import argparse
import os
import re
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set

import numpy as np


def _require_cv2():
    try:
        import cv2  # type: ignore
    except Exception as e:
        raise RuntimeError("缺少依赖 OpenCV（cv2）。请先安装：pip install opencv-python") from e
    return cv2


def _iter_files(root: str, exts: Tuple[str, ...]) -> List[str]:
    out: List[str] = []
    for r, _, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(exts):
                out.append(os.path.join(r, fn))
    return out


def _discover_emoeval_paths(iemocap_root: str) -> List[str]:
    """
    尽可能在给定 root 下找到 dialog/EmoEvaluation/*.txt
    """
    root = os.path.abspath(iemocap_root)
    emoeval_paths: Set[str] = set()

    # 1) 直接扫描名为 EmoEvaluation 的目录（大小写不敏感）
    for r, dirs, files in os.walk(root):
        base = os.path.basename(r)
        if base.lower() == "emoevaluation":
            for fn in files:
                if fn.lower().endswith(".txt"):
                    emoeval_paths.add(os.path.join(r, fn))
        # 小优化：不改 dirs

    if emoeval_paths:
        return sorted(emoeval_paths)

    # 2) 若用户给的是视频子目录：从 *.avi 反推 dialog/EmoEvaluation
    avi_paths = _iter_files(root, (".avi", ".mp4", ".mov", ".mkv", ".webm"))
    for vp in avi_paths:
        # 找到包含 dialog 的祖先目录
        parts = os.path.normpath(vp).split(os.sep)
        dialog_idx = None
        for i, p in enumerate(parts):
            if p.lower() == "dialog":
                dialog_idx = i
        if dialog_idx is None:
            continue
        dialog_dir = os.sep.join(parts[: dialog_idx + 1])
        cand = os.path.join(dialog_dir, "EmoEvaluation")
        if os.path.isdir(cand):
            for fn in os.listdir(cand):
                if fn.lower().endswith(".txt"):
                    emoeval_paths.add(os.path.join(cand, fn))

    if emoeval_paths:
        return sorted(emoeval_paths)

    # 3) 常见Session结构的直接探测（兼容 SessionX/SessionX 与 SessionX）
    for sess in ["Session1", "Session2", "Session3", "Session4", "Session5"]:
        candidates = [
            os.path.join(root, sess, sess, "dialog", "EmoEvaluation"),
            os.path.join(root, sess, "dialog", "EmoEvaluation"),
        ]
        for cand in candidates:
            if os.path.isdir(cand):
                for fn in os.listdir(cand):
                    if fn.lower().endswith(".txt"):
                        emoeval_paths.add(os.path.join(cand, fn))

    return sorted(emoeval_paths)


def _fft_bandpass(x: np.ndarray, fs: float, low: float, high: float) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - np.mean(x)
    n = x.shape[0]
    if n < 8:
        return x.astype(np.float32)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mask = (freqs >= low) & (freqs <= high)
    X[~mask] = 0
    y = np.fft.irfft(X, n=n)
    y = y / (np.std(y) + 1e-8)
    return y.astype(np.float32)


def _psd_feature(x: np.ndarray, fs: float, low: float, high: float, dim: int) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - np.mean(x)
    n = x.shape[0]
    if n < 8:
        return np.zeros((dim,), dtype=np.float32)
    X = np.fft.rfft(x * np.hanning(n))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    power = (np.abs(X) ** 2)

    band_mask = (freqs >= low) & (freqs <= high)
    if not np.any(band_mask):
        return np.zeros((dim,), dtype=np.float32)
    freqs_b = freqs[band_mask]
    power_b = power[band_mask]

    target_freqs = np.linspace(low, high, dim)
    feat = np.interp(target_freqs, freqs_b, power_b)
    feat = np.log1p(feat)
    feat = feat / (np.linalg.norm(feat) + 1e-8)
    return feat.astype(np.float32)


def _pos_rppg(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float64)
    if rgb.shape[0] < 8:
        return np.zeros((rgb.shape[0],), dtype=np.float32)

    mean_rgb = np.mean(rgb, axis=0, keepdims=True) + 1e-8
    cn = rgb / mean_rgb

    r, g, b = cn[:, 0], cn[:, 1], cn[:, 2]
    s1 = g - b
    s2 = g + b - 2 * r

    std1 = np.std(s1) + 1e-8
    std2 = np.std(s2) + 1e-8
    h = s1 - (std1 / std2) * s2
    h = h - np.mean(h)
    h = h / (np.std(h) + 1e-8)
    return h.astype(np.float32)


def _chrom_rppg(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float64)
    if rgb.shape[0] < 8:
        return np.zeros((rgb.shape[0],), dtype=np.float32)
    mean_rgb = np.mean(rgb, axis=0, keepdims=True) + 1e-8
    cn = rgb / mean_rgb
    x_s = 3 * cn[:, 0] - 2 * cn[:, 1]
    y_s = 1.5 * cn[:, 0] + cn[:, 1] - 1.5 * cn[:, 2]
    alpha = (np.std(x_s) + 1e-8) / (np.std(y_s) + 1e-8)
    rppg = x_s - alpha * y_s
    rppg = rppg - np.mean(rppg)
    rppg = rppg / (np.std(rppg) + 1e-8)
    return rppg.astype(np.float32)


def _skin_mask_ycrcb(face_rgb: np.ndarray) -> np.ndarray:
    cv2 = _require_cv2()
    ycrcb = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2YCrCb)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def _detect_face_bbox(frame_bgr: np.ndarray, face_cascade) -> Optional[Tuple[int, int, int, int]]:
    cv2 = _require_cv2()
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
    if faces is None or len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    return int(x), int(y), int(w), int(h)


def _read_video_segment_rgb_means(
    video_path: str,
    start_t: float,
    end_t: float,
    target_frames: int,
    face_cascade,
) -> Tuple[np.ndarray, float]:
    cv2 = _require_cv2()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    start_f = int(max(0, np.floor(start_t * fps)))
    end_f = int(min(total_frames - 1, np.ceil(end_t * fps))) if total_frames > 0 else int(np.ceil(end_t * fps))
    if end_f <= start_f:
        end_f = start_f + int(1.0 * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    rgb_means: List[np.ndarray] = []
    last_rgb = np.zeros((3,), dtype=np.float32)

    for _ in range(end_f - start_f + 1):
        ok, frame_bgr = cap.read()
        if not ok:
            break
        bbox = _detect_face_bbox(frame_bgr, face_cascade)
        if bbox is None:
            rgb_means.append(last_rgb)
            continue
        x, y, w, h = bbox
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = x0 + w, y0 + h
        face_bgr = frame_bgr[y0:y1, x0:x1]
        if face_bgr.size == 0:
            rgb_means.append(last_rgb)
            continue
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        mask = _skin_mask_ycrcb(face_rgb)
        if np.sum(mask) > 0:
            skin_pixels = face_rgb[mask > 0]
            mean_rgb = np.mean(skin_pixels, axis=0)
        else:
            mean_rgb = np.mean(face_rgb.reshape(-1, 3), axis=0)
        mean_rgb = mean_rgb.astype(np.float32)
        rgb_means.append(mean_rgb)
        last_rgb = mean_rgb

    cap.release()

    if len(rgb_means) < 8:
        return np.zeros((target_frames, 3), dtype=np.float32), float(fps)

    rgb_arr = np.stack(rgb_means, axis=0)

    # 统一长度重采样到 target_frames
    t_src = np.linspace(0.0, 1.0, rgb_arr.shape[0])
    t_tgt = np.linspace(0.0, 1.0, target_frames)
    rgb_resampled = np.zeros((target_frames, 3), dtype=np.float32)
    for c in range(3):
        rgb_resampled[:, c] = np.interp(t_tgt, t_src, rgb_arr[:, c])
    return rgb_resampled, float(fps)


@dataclass
class UtteranceSeg:
    utt_id: str
    start: float
    end: float


_SEG_LINE = re.compile(r"^\[(?P<s>[\d\.]+)\s*-\s*(?P<e>[\d\.]+)\]\s+(?P<utt>\S+)\s+")


def parse_emoeval_file(path: str) -> List[UtteranceSeg]:
    segs: List[UtteranceSeg] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            m = _SEG_LINE.match(line)
            if not m:
                continue
            segs.append(UtteranceSeg(utt_id=m.group("utt"), start=float(m.group("s")), end=float(m.group("e"))))
    return segs


def find_dialog_video(iemocap_root: str, dialog_id: str) -> Optional[str]:
    root = os.path.abspath(iemocap_root)
    candidates = []
    for sess in ["Session1", "Session2", "Session3", "Session4", "Session5"]:
        candidates.extend(
            [
                os.path.join(root, sess, sess, "dialog", "avi", "DivX", f"{dialog_id}.avi"),
                os.path.join(root, sess, "dialog", "avi", "DivX", f"{dialog_id}.avi"),
            ]
        )
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iemocap_root", type=str, required=True, help="IEMOCAP根目录（包含Session1..5），或任意子目录（脚本会容错搜索）")
    parser.add_argument("--out_feature_map", type=str, required=True, help="输出的rPPG特征map路径（pkl）")
    parser.add_argument("--target_frames", type=int, default=160, help="每个utterance抽取的目标帧数")
    parser.add_argument("--feature_dim", type=int, default=64, help="输出rPPG特征维度（默认64）")
    parser.add_argument("--method", type=str, default="pos", choices=["pos", "chrom"], help="rPPG波形恢复方法")
    parser.add_argument("--band_low", type=float, default=0.7, help="带通下限Hz")
    parser.add_argument("--band_high", type=float, default=4.0, help="带通上限Hz")
    parser.add_argument("--max_dialogs", type=int, default=-1, help="仅处理前N个dialog（调试用，-1表示全量）")
    args = parser.parse_args()

    cv2 = _require_cv2()
    face_xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(face_xml)
    if face_cascade.empty():
        raise RuntimeError(f"无法加载人脸检测模型：{face_xml}")

    emoeval_paths = _discover_emoeval_paths(args.iemocap_root)
    if not emoeval_paths:
        root = os.path.abspath(args.iemocap_root)
        top = []
        try:
            top = sorted(os.listdir(root))[:30]
        except Exception:
            pass
        avi_count = len(_iter_files(root, (".avi", ".mp4", ".mov", ".mkv", ".webm")))
        msg = (
            "未找到EmoEvaluation/*.txt，请确认iemocap_root路径正确。\n"
            f"- iemocap_root = {root}\n"
            f"- top-level entries (first 30) = {top}\n"
            f"- found videos under root = {avi_count}\n"
            "期望结构示例：<root>/Session1/Session1/dialog/EmoEvaluation/*.txt\n"
            "如果你把root指到了.../dialog/avi/DivX，请改为更上层的IEMOCAP目录或包含Session的目录。"
        )
        raise RuntimeError(msg)

    feature_map: Dict[str, np.ndarray] = {}
    dialogs_done = 0

    for emo_path in emoeval_paths:
        dialog_id = os.path.splitext(os.path.basename(emo_path))[0]
        video_path = find_dialog_video(args.iemocap_root, dialog_id)
        if video_path is None:
            continue
        segs = parse_emoeval_file(emo_path)
        if not segs:
            continue

        for seg in segs:
            try:
                rgb_trace, _fps = _read_video_segment_rgb_means(
                    video_path=video_path,
                    start_t=seg.start,
                    end_t=seg.end,
                    target_frames=args.target_frames,
                    face_cascade=face_cascade,
                )

                if args.method == "pos":
                    rppg = _pos_rppg(rgb_trace)
                else:
                    rppg = _chrom_rppg(rgb_trace)

                # 这里用 target_frames 作为“等间隔采样点数”的 fs 近似（用于FFT频率轴归一化）
                rppg = _fft_bandpass(rppg, fs=float(args.target_frames), low=args.band_low, high=args.band_high)
                feat = _psd_feature(rppg, fs=float(args.target_frames), low=args.band_low, high=args.band_high, dim=args.feature_dim)
                feature_map[seg.utt_id] = feat
            except Exception:
                continue

        dialogs_done += 1
        if args.max_dialogs > 0 and dialogs_done >= args.max_dialogs:
            break

    out_dir = os.path.dirname(os.path.abspath(args.out_feature_map))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out_feature_map, "wb") as f:
        pickle.dump(feature_map, f)

    print(f"[Done] rPPG feature map saved: {args.out_feature_map}")
    print(f"[Info] extracted utterances: {len(feature_map)}")


if __name__ == "__main__":
    main()


