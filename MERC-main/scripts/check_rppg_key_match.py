import argparse
import pickle
from typing import Any, Dict, Iterable, List, Optional, Tuple


def safe_pickle_load(path: str):
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except ModuleNotFoundError as e:
            if "numpy._core" not in str(e):
                raise

            class _CompatUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    if module.startswith("numpy._core"):
                        module = module.replace("numpy._core", "numpy.core", 1)
                    return super().find_class(module, name)

            f.seek(0)
            return _CompatUnpickler(f).load()


def _canonical_utt_id(text: str):
    import re

    if not text:
        return None
    if not isinstance(text, str):
        text = str(text)
    # Ses01F_impro01_F000 / Ses01F_script02_1_F003 / Ses03M_impro08a_F012 ...
    patt = re.compile(
        r"Ses(?P<sess>\d{2})(?P<dg>[FM])_"
        r"(?P<kind>impro|script)(?P<num>\d+)(?P<suf>[a-z]?)"
        r"(?P<part>_\d+)?_"
        r"(?P<spk>[FM])(?P<idx>\d{3})",
        re.IGNORECASE,
    )
    m = patt.search(text)
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


def _speaker_count_index(speakers, idx: int, spk: str) -> Optional[int]:
    if speakers is None:
        return None
    spk = (spk or "").strip().upper()
    if spk not in ("F", "M"):
        return None
    try:
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


def _yield_keys_from_string(x: Any) -> Iterable[str]:
    import os

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
    canon = _canonical_utt_id(x)
    if canon:
        yield canon


def build_candidates(sample, idx: int) -> List[str]:
    tried = []
    seen = set()

    def add(k: Optional[str]):
        if not k:
            return
        if k in seen:
            return
        seen.add(k)
        tried.append(k)

    # sentence[idx] / vid
    try:
        if hasattr(sample, "sentence") and sample.sentence is not None and len(sample.sentence) > idx:
            for k in _yield_keys_from_string(sample.sentence[idx]):
                add(k)
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
            add(k)

    # dialog_id + speaker + index variants
    speakers = getattr(sample, "speaker", None)
    spk = None
    try:
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
        spk_candidates = [spk] if spk in ("F", "M") else ["F", "M"]
        for sp in spk_candidates:
            idx_turn0 = idx
            idx_turn1 = idx + 1
            idx_sp0 = _speaker_count_index(speakers, idx, sp)
            idx_sp1 = (idx_sp0 + 1) if idx_sp0 is not None else None

            for num in (idx_turn0, idx_turn1):
                add(f"{did}_{sp}{int(num):03d}")
            if idx_sp0 is not None:
                add(f"{did}_{sp}{int(idx_sp0):03d}")
            if idx_sp1 is not None:
                add(f"{did}_{sp}{int(idx_sp1):03d}")

    return tried


def iter_split_samples(data_obj, split: str):
    # data pkl 常见：dict with keys train/dev/test or a namespace-like object
    if isinstance(data_obj, dict):
        if split in data_obj:
            return data_obj[split]
        # 兼容 iemocap_4：可能是 {"train": [...], "valid": [...], ...}
        if split == "dev" and "valid" in data_obj:
            return data_obj["valid"]
    return getattr(data_obj, split)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rppg_map", required=True)
    ap.add_argument("--data_pkl", required=True)
    ap.add_argument("--split", default="train", choices=["train", "dev", "test"])
    ap.add_argument("--max_samples", type=int, default=200)
    ap.add_argument("--max_utts_per_sample", type=int, default=5)
    ap.add_argument("--max_checks", type=int, default=1000)
    ap.add_argument("--print_rppg_keys", type=int, default=20)
    ap.add_argument("--print_miss", type=int, default=5)
    args = ap.parse_args()

    rppg_map: Dict[str, Any] = safe_pickle_load(args.rppg_map)

    print(f"[Info] rPPG map keys: {len(rppg_map)}")
    if args.print_rppg_keys > 0:
        ks = list(rppg_map.keys())[: args.print_rppg_keys]
        print(f"\n[Info] rPPG map key samples (first {len(ks)}):\n")
        for k in ks:
            print(f"  {k}")

    data_obj = safe_pickle_load(args.data_pkl)
    samples = list(iter_split_samples(data_obj, args.split))

    hit = 0
    total = 0
    miss_printed = 0
    hit_by_primary: Dict[str, int] = {"sentence": 0, "vid": 0, "constructed": 0}

    for si, s in enumerate(samples[: args.max_samples]):
        # 每个 sample 是一个对话（多个 utterance）
        cur_len = len(getattr(s, "text", []))
        for idx in range(min(cur_len, args.max_utts_per_sample)):
            if total >= args.max_checks:
                break

            candidates = build_candidates(s, idx)
            total += 1

            found = None
            for c in candidates:
                if c in rppg_map:
                    found = c
                    break

            if found is not None:
                hit += 1
                # 粗略归因：前几项是 sentence/vid，后面是 constructed
                if hasattr(s, "sentence") and s.sentence is not None and len(s.sentence) > idx:
                    if found == s.sentence[idx]:
                        hit_by_primary["sentence"] += 1
                    else:
                        hit_by_primary["constructed"] += 1
                else:
                    hit_by_primary["constructed"] += 1
            else:
                if miss_printed < args.print_miss:
                    fields: List[Tuple[str, Any]] = []
                    for name in ("sentence", "vid", "speaker"):
                        if hasattr(s, name):
                            v = getattr(s, name)
                            if name == "sentence" and isinstance(v, list) and len(v) > idx:
                                fields.append((name, v[idx]))
                            elif name == "speaker" and isinstance(v, list) and len(v) > idx:
                                fields.append((name, v[idx]))
                            else:
                                fields.append((name, v))
                    print("\n[Miss example]")
                    print(f" fields: {fields}")
                    print(f" candidates(first 12): {candidates[:12]}")
                    miss_printed += 1

        if total >= args.max_checks:
            break

    hr = (hit / total) if total else 0.0
    print(f"\n[Result] Hit rate: {hit}/{total} ({hr:.2%})")
    print(f"[Result] Hit by primary source: {hit_by_primary}")


if __name__ == "__main__":
    main()


