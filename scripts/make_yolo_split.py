"""Re-split a YOLO-format dataset by video sequence (no near-duplicate frames across splits).

  .venv\\Scripts\\python.exe scripts\\make_yolo_split.py "D:\\Data Sentir\\pereezd_dataset" --out C:\\data\\pereezd_split

Frames cut from one video are near-duplicates (0.2 s apart), so a random frame split
leaks train frames into val and inflates every metric. Here the original train/ and val/
images are pooled, grouped by sequence (file name without the trailing frame number),
and whole sequences go to val until it holds ~`--val-frac` of the images. The original
dataset is not touched: the result is two lists of image paths plus a report.
"""
import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sarbon.data.yolo import IMG_EXT, label_path, read_labels  # noqa: E402


def sequence(name):
    """'PERZ_02_0012' -> 'PERZ_02'; '192.168.170.160_02_20251023183425270' -> '192.168.170.160_02'."""
    return re.sub(r"\d{6,}.*$", "", re.sub(r"[_-]?\d+$", "", name)).rstrip("_-") or name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--num-classes", type=int, default=2, help="label ids >= this are reported as invalid")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    root = Path(args.root)
    imgs = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT and "images" in p.parts)
    seqs = defaultdict(list)
    for p in imgs:
        seqs[sequence(p.stem)].append(p)
    # per-sequence class counts, so val gets heavy vehicles too
    counts, bad = {}, []
    for s, ps in seqs.items():
        c = Counter()
        for p in ps:
            rows, invalid = read_labels(label_path(p), args.num_classes)
            c.update(int(r[0]) for r in rows)
            bad += [(str(label_path(p)), line) for line in invalid]
        counts[s] = c
    names = sorted(seqs)
    random.Random(args.seed).shuffle(names)
    # sequences with the rarest class first, then the rest, until the val budget is used
    rare = min(range(args.num_classes), key=lambda k: sum(c[k] for c in counts.values()))
    names.sort(key=lambda s: counts[s][rare] == 0)
    budget, val = args.val_frac * len(imgs), []
    for s in names:
        if sum(len(seqs[v]) for v in val) >= budget:
            break
        if len(seqs[s]) <= 0.5 * budget:  # one huge sequence would make val a single scene
            val.append(s)
    train = [s for s in seqs if s not in val]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"root": str(root), "images": len(imgs), "sequences": len(seqs), "invalid_rows": len(bad),
              "invalid_examples": bad[:20]}
    for split, ss in (("train", train), ("val", val)):
        paths = sorted(str(p) for s in ss for p in seqs[s])
        (out / f"{split}.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
        c = sum((counts[s] for s in ss), Counter())
        report[split] = {"images": len(paths), "sequences": sorted(ss) if split == "val" else len(ss),
                         "boxes_per_class": {str(k): c[k] for k in range(args.num_classes)}}
    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "invalid_examples"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
