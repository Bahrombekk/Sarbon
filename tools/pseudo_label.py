"""Pseudo-labels from a teacher detector, for distilling it into a smaller student.

Two kinds of input, written as one YOLO split (`images/`, `labels/`, `train.txt`, `val.txt`) whose
class ids are already our labels (train with the identity map "0:0,1:1,2:2,3:3,4:4,5:5"):

  --videos : frames sampled every --every seconds; all teacher classes become labels.
             The last --val-frac of every video (a time block, not random frames) goes to val.
  --list   : images of an existing labelled split (its train.txt / val.txt). Their own labels are
             kept (--keep "yolo_class:our_label,..."), the teacher adds only --add classes (the
             ones the split never annotated, e.g. person on the crossing set).

Images of --list are hard-linked, not copied (same volume); the source dataset is not modified.
Teacher: per query its best class, kept when the score is >= --conf (per-class with --conf-class).

    python tools/pseudo_label.py --weights weights/sarbon-s-railsafe-1088.pt --out C:/data/pereezd_pseudo
        --videos "C:/data/recordings/*.mp4" --every 2
    python tools/pseudo_label.py --weights ... --out C:/data/pereezd_plus --list C:/data/pereezd_yolo_split
        --keep 0:3,1:4 --add 0,1,2,5
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sarbon.data.coco import RAILSAFE, letterbox  # noqa: E402
from sarbon.data.yolo import imread_at_least, label_path, read_labels  # noqa: E402
from sarbon.engine import prep  # noqa: E402
from sarbon.models.sarbon_det import load_detector  # noqa: E402


class Teacher:
    def __init__(self, weights, size, conf, conf_class, batch=8):
        self.m = load_detector(weights, "s", len(RAILSAFE), halo_zero=False).cuda().eval()
        self.size, self.batch = size, batch
        self.conf = np.full(len(RAILSAFE), conf, np.float32)
        for k, v in conf_class.items():
            self.conf[k] = v

    @torch.no_grad()
    def __call__(self, imgs_bgr):
        """BGR images -> per image a list of (label, cx, cy, w, h) normalised to that image."""
        out = []
        for i in range(0, len(imgs_bgr), self.batch):
            chunk = imgs_bgr[i:i + self.batch]
            boxes = [letterbox(im[:, :, ::-1], self.size) for im in chunk]
            x = torch.from_numpy(np.stack([b[0] for b in boxes])).permute(0, 3, 1, 2)
            with torch.autocast("cuda", torch.bfloat16):
                o = self.m(prep(x, torch.device("cuda")))
            sc, lb = o["pred_logits"].float().sigmoid().max(-1)
            bx = o["pred_boxes"].float() * self.size  # cx cy w h, letterboxed pixels
            for j, (im, (_, s, (px, py))) in enumerate(zip(chunk, boxes)):
                h, w = im.shape[:2]
                keep = sc[j] >= torch.from_numpy(self.conf).cuda()[lb[j]]
                rows = []
                for (cx, cy, bw, bh), l in zip(bx[j][keep].tolist(), lb[j][keep].tolist()):
                    cx, cy, bw, bh = (cx - px) / s / w, (cy - py) / s / h, bw / s / w, bh / s / h
                    x0, y0 = max(0.0, cx - bw / 2), max(0.0, cy - bh / 2)
                    x1, y1 = min(1.0, cx + bw / 2), min(1.0, cy + bh / 2)
                    if x1 - x0 > 1e-3 and y1 - y0 > 1e-3:
                        rows.append((l, (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0))
                out.append(rows)
        return out


def write(rows, path):
    path.write_text("".join(f"{l} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for l, cx, cy, w, h in rows),
                    encoding="utf-8")


def from_videos(teacher, pattern, every, val_frac, out):
    lists, n_box = {"train": [], "val": []}, np.zeros(len(RAILSAFE), int)
    for video in sorted(glob.glob(pattern)):
        cap = cv2.VideoCapture(video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, round(every * fps))
        idx = list(range(0, total, step))
        frames, names = [], []
        for f in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if not ok:
                break
            frames.append(img)
            names.append(f"{Path(video).stem}_{f:06d}.jpg")
        cut = int(len(frames) * (1 - val_frac))
        for k, (img, name, rows) in enumerate(zip(frames, names, teacher(frames))):
            p = out / "images" / name
            cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            write(rows, label_path(p))
            lists["val" if k >= cut else "train"].append(str(p))
            for r in rows:
                n_box[r[0]] += 1
        print(f"{Path(video).name}: {len(frames)} frames", flush=True)
    return lists, n_box


def from_list(teacher, split, keep, add, out):
    lists, n_box = {"train": [], "val": []}, np.zeros(len(RAILSAFE), int)
    n_cls = max(keep) + 1
    for part in ("train", "val"):
        paths = [l.strip() for l in (Path(split) / f"{part}.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
        for i in range(0, len(paths), 64):
            chunk = paths[i:i + 64]
            imgs = [imread_at_least(p, teacher.size) for p in chunk]
            for src, rows in zip(chunk, teacher(imgs)):
                own, _ = read_labels(label_path(src), n_cls)
                rows = [(keep[int(c)], *b) for c, *b in own if int(c) in keep] + [r for r in rows if r[0] in add]
                dst = out / "images" / Path(src).name
                if not dst.exists():
                    os.link(src, dst)
                write(rows, label_path(dst))
                lists[part].append(str(dst))
                for r in rows:
                    n_box[r[0]] += 1
            print(f"{part}: {min(i + 64, len(paths))}/{len(paths)}", flush=True)
    return lists, n_box


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=1088)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--conf-class", default="", help="per-class thresholds, e.g. 0:0.45,5:0.6")
    ap.add_argument("--videos", help="glob of video files")
    ap.add_argument("--every", type=float, default=2.0, help="seconds between sampled frames")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--list", help="an existing split folder with train.txt / val.txt")
    ap.add_argument("--keep", default="0:3,1:4", help="with --list: its own classes to keep, yolo:ours")
    ap.add_argument("--add", default="0,1,2,5", help="with --list: teacher classes to add")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    conf_class = {int(a): float(b) for a, b in (p.split(":") for p in args.conf_class.split(",") if p)}
    teacher = Teacher(args.weights, args.size, args.conf, conf_class)
    if args.videos:
        lists, n_box = from_videos(teacher, args.videos, args.every, args.val_frac, out)
    else:
        keep = {int(a): int(b) for a, b in (p.split(":") for p in args.keep.split(","))}
        lists, n_box = from_list(teacher, args.list, keep, {int(c) for c in args.add.split(",")}, out)
    for part, paths in lists.items():
        (out / f"{part}.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
    info = {"teacher": args.weights, "size": args.size, "conf": args.conf, "conf_class": conf_class,
            "source": args.videos or args.list, "every_s": args.every if args.videos else None,
            "keep": args.keep if args.list else None, "add": args.add if args.list else None,
            "images": {k: len(v) for k, v in lists.items()},
            "boxes": {RAILSAFE[i]: int(n) for i, n in enumerate(n_box)}}
    (out / "build.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(json.dumps(info))


if __name__ == "__main__":
    main()
