"""
检查 rPPG feature_map (utterance_id -> feature) 与训练数据 (Sample.sentence) 的 key 命中率。

用途：
- 快速定位为什么训练日志里 rPPG valid ratio 还是 0%
- 给出 rPPG map 的 key 样例、训练数据的 sentence 样例、以及候选 key 规范化后的命中率

示例：
  python scripts/check_rppg_key_match.py ^
    --rppg_map MERC-main/JOYFUL/data/rppg/iemocap_rppg_map.pkl ^
    --data_pkl MERC-main/JOYFUL/data/iemocap_4/data_iemocap_4.pkl
"""

from __future__ import annotations

import argparse
import os
import pickle
import random
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


UTT_ID_RE = re.compile(
    r"(Ses\d{2}[FM]_(?:impro|script)\d+[a-z]?(?:_\d+)?_[FM]\d{3})",
    re.IGNORECASE,
)


def iter_candidate_keys(x: Any) -> Iterable[str]:
    if x is None:
        return
    if not isinstance(x, str):
        x = str(x)
    x = x.strip()
    if not x:
        return

    yield x

    base = os.path.basename(x)
    if base and base != x:
        yield base

    stem, _ext = os.path.splitext(base)
    if stem and stem != base:
        yield stem

    m = UTT_ID_RE.search(x)
    if m:
        yield m.group(1)


def extract_fields(sample: Any, idx: int) -> List[Tuple[str, Any]]:
    out: List[Tuple[str, Any]] = []
    try:
        if hasattr(sample, "sentence") and sample.sentence is not None and len(sample.sentence) > idx:
            out.append(("sentence", sample.sentence[idx]))
    except Exception:
        pass
    try:
        if hasattr(sample, "vid"):
            v = sample.vid
            if isinstance(v, (list, tuple)) and len(v) > idx:
                out.append(("vid[idx]", v[idx]))
            else:
                out.append(("vid", v))
    except Exception:
        pass
    return out


def build_candidates(sample: Any, idx: int) -> List[str]:
    candidates: List[str] = []
    seen = set()
    for src, val in extract_fields(sample, idx):
        for k in iter_candidate_keys(val):
            if k in seen:
                continue
            seen.add(k)
            candidates.append(k)

    # 兜底：dialog_id(vid) + speaker + idx
    try:
        dialog_id: Optional[str] = None
        if hasattr(sample, "vid") and isinstance(sample.vid, str):
            dialog_id = sample.vid.strip()
        spk: Optional[str] = None
        if hasattr(sample, "speaker") and sample.speaker is not None and len(sample.speaker) > idx:
            spk = str(sample.speaker[idx]).strip()
        if dialog_id and spk:
            spk = spk[0].upper()
            if spk in ("F", "M"):
                k = f"{dialog_id}_{spk}{idx:03d}"
                if k not in seen:
                    candidates.append(k)
    except Exception:
        pass

    return candidates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rppg_map", required=True, help="rPPG feature map pkl path")
    ap.add_argument("--data_pkl", required=True, help="training data pkl path (contains dict with train/dev/test)")
    ap.add_argument("--split", default="train", choices=["train", "dev", "test"], help="which split to sample")
    ap.add_argument("--max_samples", type=int, default=200, help="max dialog samples to draw from split")
    ap.add_argument("--max_utts_per_sample", type=int, default=5, help="max utterances per dialog sample to test")
    ap.add_argument("--max_checks", type=int, default=1000, help="max utterances to check overall")
    ap.add_argument("--print_rppg_keys", type=int, default=20, help="print first N rppg keys")
    ap.add_argument("--print_miss", type=int, default=10, help="print first N miss examples")
    args = ap.parse_args()

    with open(args.rppg_map, "rb") as f:
        rppg_map: Dict[str, Any] = pickle.load(f)
    rppg_keys = list(rppg_map.keys())
    print(f"[Info] rPPG map keys: {len(rppg_keys)}")
    print(f"[Info] rPPG map key samples (first {min(args.print_rppg_keys, len(rppg_keys))}):")
    for k in rppg_keys[: args.print_rppg_keys]:
        print(" ", k)

    with open(args.data_pkl, "rb") as f:
        data = pickle.load(f)
    split_data: Sequence[Any] = data.get(args.split, [])
    print(f"[Info] data split '{args.split}' samples: {len(split_data)}")

    if not split_data:
        print("[Error] split is empty.")
        return

    random.seed(0)
    chosen = random.sample(list(split_data), min(args.max_samples, len(split_data)))
    pairs: List[Tuple[Any, int]] = []
    for s in chosen:
        try:
            L = len(getattr(s, "sentence", [])) if getattr(s, "sentence", None) is not None else 0
        except Exception:
            L = 0
        for i in range(min(L, args.max_utts_per_sample)):
            pairs.append((s, i))
    random.shuffle(pairs)
    pairs = pairs[: args.max_checks]

    hit = 0
    hit_by = Counter()
    misses_printed = 0

    for s, i in pairs:
        fields = extract_fields(s, i)
        candidates = build_candidates(s, i)
        found = None
        for k in candidates:
            if k in rppg_map:
                found = k
                break
        if found is not None:
            hit += 1
            hit_by[fields[0][0] if fields else "unknown"] += 1
        elif misses_printed < args.print_miss:
            misses_printed += 1
            print("\n[Miss example]")
            print(" fields:", fields)
            print(" candidates(first 12):", candidates[:12])

    total = len(pairs) if pairs else 1
    print(f"\n[Result] Hit rate: {hit}/{total} ({hit/total:.2%})")
    print("[Result] Hit by primary source:", dict(hit_by))


if __name__ == "__main__":
    main()


