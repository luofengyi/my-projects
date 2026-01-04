import argparse
import csv
import re
from pathlib import Path


STRICT_UTT_RE = re.compile(
    r"^Ses\d{2}[FM]_(?:impro|script)\d{2}(?:_\d+)?_[FM]\d{3}\s+\[\d+\.\d+-\d+\.\d+\]:"
)
SPEAKER_ONLY_RE = re.compile(r"^[FM]\s*:")


def count_file(path: Path) -> dict:
    strict = 0
    speaker_only = 0
    nonempty = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            nonempty += 1
            if STRICT_UTT_RE.match(line):
                strict += 1
            elif SPEAKER_ONLY_RE.match(line):
                speaker_only += 1
    return {
        "file": path.name,
        "strict_utterances": strict,
        "speaker_only_lines": speaker_only,
        "nonempty_lines": nonempty,
        "total_dialog_lines_loose": strict + speaker_only,
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Count utterances per dialog transcription file. "
            "Strict utterances are lines like 'Ses04..._[FM]000 [t-t]: text'."
        )
    )
    p.add_argument(
        "--transcriptions_dir",
        required=True,
        help="Path to dialog/transcriptions (e.g., JOYFUL/data/Session4/dialog/transcriptions).",
    )
    p.add_argument(
        "--out_csv",
        default="",
        help="Optional output CSV path. If omitted, a CSV is not written.",
    )
    args = p.parse_args()

    d = Path(args.transcriptions_dir)
    if not d.exists() or not d.is_dir():
        raise SystemExit(f"Not a directory: {d}")

    rows = [count_file(fp) for fp in sorted(d.glob("*.txt"))]
    if not rows:
        raise SystemExit(f"No .txt files found in: {d}")

    # Print a compact table
    header = ["file", "strict_utterances", "speaker_only_lines", "nonempty_lines", "total_dialog_lines_loose"]
    print("\t".join(header))
    for r in rows:
        print("\t".join(str(r[k]) for k in header))

    if args.out_csv:
        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote CSV: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())



