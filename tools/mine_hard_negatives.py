r"""Hard negatives from own labelled data, with a human (or reviewer) in the loop.

Why two steps: the crossing labels are incomplete (many vehicles are unlabelled), so a confident
detection with no label is often a real vehicle, not a false positive. Feeding every such frame
back as "nothing here" would teach the detector to miss vehicles. So:

  1. mine   -- run the detector over a split, keep confident detections that overlap no label at
               all (IoU < 0.1), write numbered review sheets + review.csv (verdict column empty).
  2. build  -- read the filled review.csv and write a new split folder:
                 n            real false positive (steel coil, signal post, ...): the frame is
                              repeated --repeat times in train.txt, its labels unchanged
                 o            real object the labels missed: the box is added to a copy of that
                              frame's labels with the predicted class
                 o=<class>    same, with another class (yengil / ogir / ... by name)
                 anything else (empty, ?, x): ignored
               Unchanged frames keep their original paths; changed ones are hard links (or copies)
               under <out>/images with patched <out>/labels. The source dataset is only read.

  python tools\mine_hard_negatives.py mine  --split C:/data/pereezd_split --out C:/data/pereezd_hardneg
  (fill C:/data/pereezd_hardneg/review.csv, looking at sheet_*.jpg)
  python tools\mine_hard_negatives.py build --split C:/data/pereezd_split --out C:/data/pereezd_hardneg
  then train with  yolo: C:/data/pereezd_hardneg/split
"""
import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sarbon.data.coco import RAILSAFE  # noqa: E402
from sarbon.data.yolo import YoloDetection, imread_at_least, label_path  # noqa: E402
from sarbon.engine import prep  # noqa: E402
from sarbon.models.box_ops import cxcywh_to_xyxy  # noqa: E402
from sarbon.models.sarbon_det import build, load_weights  # noqa: E402
from sarbon.paths import DEFAULT_DETECTOR  # noqa: E402

FIELDS = ["id", "image", "class", "score", "x0", "y0", "x1", "y1", "verdict"]
TILE, COLS, ROWS = 200, 8, 6


def parse_map(s):
    return {int(a): int(b) for a, b in (kv.split(":") for kv in s.split(","))}


def iou(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area = lambda x: (x[:, 2:] - x[:, :2]).prod(-1)  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def tile(path, box, text):
    """Context crop around a box (3x its size, at least 96 px), box drawn, number + score written."""
    img = imread_at_least(path, 1 << 30)
    x0, y0, x1, y1 = box
    half = max(48, 1.5 * max(x1 - x0, y1 - y0))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    a = [int(max(0, cx - half)), int(max(0, cy - half))]
    crop = img[a[1]:int(cy + half), a[0]:int(cx + half)].copy()
    k = TILE / max(crop.shape[:2])
    crop = cv2.resize(crop, None, fx=k, fy=k)
    p0 = (int((x0 - a[0]) * k), int((y0 - a[1]) * k))
    p1 = (int((x1 - a[0]) * k), int((y1 - a[1]) * k))
    cv2.rectangle(crop, p0, p1, (0, 0, 255), 2)
    out = np.zeros((TILE, TILE, 3), np.uint8)
    out[:crop.shape[0], :crop.shape[1]] = crop
    cv2.rectangle(out, (0, TILE - 22), (TILE, TILE), (0, 0, 0), -1)
    cv2.putText(out, text, (4, TILE - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return out


@torch.no_grad()
def mine(args):
    out = Path(args.out)
    if (out / "review.csv").exists():
        sys.exit(f"{out / 'review.csv'} exists; it may hold your verdicts. Use another --out.")
    out.mkdir(parents=True, exist_ok=True)
    cmap = parse_map(args.map)
    ds = YoloDetection(Path(args.split) / f"{args.part}.txt", args.img, cmap, train=False, limit=args.limit)
    allowed = torch.tensor(sorted(set(cmap.values())))  # judge only the classes this data labels
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_weights(build(args.size, len(RAILSAFE), halo_zero=False), args.weights).to(dev).eval()
    cands = []
    for start in range(0, len(ds), args.batch):
        items = [ds.load(i) for i in range(start, min(start + args.batch, len(ds)))]
        x = torch.from_numpy(np.stack([np.ascontiguousarray(it[0]) for it in items])).permute(0, 3, 1, 2)
        with torch.autocast(dev.type, torch.bfloat16, enabled=dev.type == "cuda"):
            o = model(prep(x, dev))
        prob = o["pred_logits"].float().sigmoid()[..., allowed]
        score, k = prob.max(-1)
        boxes = cxcywh_to_xyxy(o["pred_boxes"].float()) * args.img
        for b, (_, gt, _, (iid, s, px, py)) in enumerate(items):
            keep = (score[b] >= args.conf).nonzero()[:, 0]
            keep = keep[score[b, keep].argsort(descending=True)]
            bx = boxes[b, keep].cpu().numpy()
            sc, cl = score[b, keep].cpu().numpy(), allowed[k[b, keep].cpu()].numpy()
            taken = []
            for j in range(len(bx)):
                if len(gt) and iou(bx[j:j + 1], gt).max() >= 0.1:
                    continue
                if taken and iou(bx[j:j + 1], bx[taken]).max() >= 0.5:
                    continue  # one candidate per object
                taken.append(j)
                if len(taken) > args.per_image:
                    break
                orig = (bx[j] - np.array([px, py, px, py])) / s
                cands.append({"image": ds.images[iid]["file_name"], "class": RAILSAFE[int(cl[j])],
                              "score": round(float(sc[j]), 3), "box": orig.round(1).tolist()})
        print(f"\r{min(start + args.batch, len(ds))}/{len(ds)} frames, {len(cands)} candidates", end="")
    print()
    cands.sort(key=lambda c: -c["score"])
    cands = cands[: args.max]
    with open(out / "review.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        for n, c in enumerate(cands):
            w.writerow([n, c["image"], c["class"], c["score"], *c["box"], ""])
    per = COLS * ROWS
    for p in range(0, len(cands), per):
        tiles = [tile(c["image"], c["box"], f"#{p + n} {c['class']} {c['score']:.2f}")
                 for n, c in enumerate(cands[p:p + per])]
        tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % COLS)
        sheet = np.vstack([np.hstack(tiles[r:r + COLS]) for r in range(0, len(tiles), COLS)])
        cv2.imwrite(str(out / f"sheet_{p // per:03d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    (out / "mine.json").write_text(json.dumps({**vars(args), "candidates": len(cands)}, indent=1), encoding="utf-8")
    print(f"{len(cands)} candidates -> {out / 'review.csv'} and {out}/sheet_*.jpg")


def link(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build_split(args):
    out = Path(args.out)
    cmap = parse_map(args.map)
    inv = {v: k for k, v in cmap.items()}  # RailSafe class -> own YOLO id
    rows = list(csv.DictReader(open(out / "review.csv", encoding="utf-8")))
    neg, add = set(), {}
    for r in rows:
        v = r["verdict"].strip().lower()
        if v == "n":
            neg.add(r["image"])
        elif v == "o" or v.startswith("o="):
            name = v[2:] if v.startswith("o=") else r["class"]
            if name not in RAILSAFE or RAILSAFE.index(name) not in inv:
                sys.exit(f"row {r['id']}: class {name!r} is not labelled in this data ({sorted(inv)})")
            add.setdefault(r["image"], []).append((inv[RAILSAFE.index(name)], [float(r[k]) for k in "x0 y0 x1 y1".split()]))
    split = out / "split"
    (split / "images").mkdir(parents=True, exist_ok=True)
    (split / "labels").mkdir(exist_ok=True)
    moved = {}
    for img, boxes in add.items():
        src = Path(img)
        dst = split / "images" / src.name
        if dst.exists():
            dst.unlink()
        link(src, dst)
        h, w = imread_at_least(img, 1 << 30).shape[:2]
        lines = label_path(src).read_text(encoding="utf-8").splitlines() if label_path(src).exists() else []
        for c, (x0, y0, x1, y1) in boxes:
            x0, x1 = np.clip([x0, x1], 0, w)
            y0, y1 = np.clip([y0, y1], 0, h)
            lines.append(f"{c} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
        (split / "labels" / f"{src.stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        moved[img] = str(dst)
    train = []
    for p in (Path(args.split) / "train.txt").read_text(encoding="utf-8").splitlines():
        p = p.strip()
        if p:
            train += [moved.get(p, p)] * (args.repeat if p in neg else 1)
    (split / "train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    shutil.copy2(Path(args.split) / "val.txt", split / "val.txt")  # val stays the untouched reference
    rep = {"reviewed": sum(r["verdict"].strip() != "" for r in rows), "negative_frames": len(neg),
           "added_boxes": sum(map(len, add.values())), "patched_frames": len(add), "train_lines": len(train)}
    (split / "build.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1), f"\ntrain with  yolo: {split.as_posix()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("mine", "build"):
        p = sub.add_parser(name)
        p.add_argument("--split", default="C:/data/pereezd_split", help="folder with train.txt / val.txt")
        p.add_argument("--out", required=True)
        p.add_argument("--map", default="0:3,1:4", help="own YOLO id -> RailSafe id (the labelled classes)")
    m = sub.choices["mine"]
    m.add_argument("--part", default="train", choices=["train", "val"])
    m.add_argument("--weights", default=str(DEFAULT_DETECTOR))
    m.add_argument("--size", default="s")
    m.add_argument("--img", type=int, default=640)
    m.add_argument("--conf", type=float, default=0.5)
    m.add_argument("--per-image", type=int, default=3)
    m.add_argument("--max", type=int, default=480, help="keep the most confident N (10 sheets of 48)")
    m.add_argument("--batch", type=int, default=16)
    m.add_argument("--limit", type=int)
    sub.choices["build"].add_argument("--repeat", type=int, default=3, help="copies of each confirmed-negative frame")
    args = ap.parse_args()
    mine(args) if args.cmd == "mine" else build_split(args)


if __name__ == "__main__":
    main()
