"""Build a fully labelled test set from own val frames: pick frames evenly over every val sequence,
copy them, and pre-label every RailSafe class so a person only has to correct, not draw from zero.

Pre-labels = the existing human boxes (yengil / ogir, kept as they are) + Sarbon detections that do
not overlap them (any of the 6 classes, score >= --conf). Output is plain YOLO format
(images/, labels/, classes.txt) that CVAT, X-AnyLabeling, LabelImg and Roboflow import directly.
The source dataset is only read, never changed.

  .venv\\Scripts\\python.exe scripts\\make_relabel_set.py --out C:\\data\\pereezd_relabel --n 300
"""
import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from make_yolo_split import sequence  # noqa: E402
from sarbon.data.coco import RAILSAFE  # noqa: E402
from sarbon.data.yolo import YoloDetection, label_path  # noqa: E402
from sarbon.models.box_ops import cxcywh_to_xyxy  # noqa: E402
from sarbon.engine import prep  # noqa: E402
from sarbon.models.sarbon_det import load_detector  # noqa: E402

OWN_TO_RAILSAFE = {0: RAILSAFE.index("yengil"), 1: RAILSAFE.index("ogir")}


def iou(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area = lambda x: (x[:, 2:] - x[:, :2]).prod(-1)
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def pick(paths, n):
    """~n frames spread evenly over every sequence (sorted by name = by time)."""
    seqs = defaultdict(list)
    for i, p in enumerate(paths):
        seqs[sequence(Path(p).stem)].append(i)
    per = max(1, n // len(seqs))
    out = []
    for idx in seqs.values():
        idx = sorted(idx, key=lambda i: Path(paths[i]).name)
        out += [idx[int(j)] for j in np.linspace(0, len(idx) - 1, min(per, len(idx)))]
    return sorted(set(out)), {k: len(v) for k, v in seqs.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="C:/data/pereezd_split")
    ap.add_argument("--weights", default=str(ROOT / "runs" / "railsafe" / "s_railsafe" / "best.pt"))
    ap.add_argument("--out", default="C:/data/pereezd_relabel")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--img", type=int, default=640)
    args = ap.parse_args()

    ds = YoloDetection(Path(args.split) / "val.txt", args.img, OWN_TO_RAILSAFE, train=False)
    paths = [im["file_name"] for im in ds.images]
    chosen, seq_sizes = pick(paths, args.n)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} is not empty; choose another --out (nothing was changed)")
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(exist_ok=True)

    model = load_detector(args.weights, "s", len(RAILSAFE), halo_zero=False).cuda().eval()
    stats = {"frames": 0, "human_boxes": 0, "prelabels": defaultdict(int)}
    for i in chosen:
        img, gt, labels, (_, s, px, py) = ds.load(i)
        x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)[None]
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            o = model(prep(x, torch.device("cuda")))
        prob = o["pred_logits"][0].float().sigmoid()
        score, cls = prob.max(-1)
        keep = (score >= args.conf).nonzero()[:, 0]
        keep = keep[score[keep].argsort(descending=True)]
        boxes = (cxcywh_to_xyxy(o["pred_boxes"][0].float()[keep]) * args.img).cpu().numpy()
        cls, score = cls[keep].cpu().numpy(), score[keep].cpu().numpy()

        # predictions that duplicate a human box or a stronger prediction are dropped
        new = []
        for j in range(len(boxes)):
            if len(gt) and iou(boxes[j:j + 1], gt).max() >= 0.5:
                continue
            if new and iou(boxes[j:j + 1], boxes[new]).max() >= 0.7:
                continue
            new.append(j)

        src = Path(paths[i])
        w, h = (args.img - 2 * px) / s, (args.img - 2 * py) / s  # decoded size the boxes refer to

        def to_yolo(b, c):
            x0, y0, x1, y1 = (b - np.array([px, py, px, py])) / s
            x0, x1 = np.clip([x0, x1], 0, w)
            y0, y1 = np.clip([y0, y1], 0, h)
            return f"{c} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}"

        # human boxes copied verbatim (only the class id is renumbered), so nothing drifts
        lines = []
        for row in label_path(src).read_text(encoding="utf-8").splitlines():
            parts = row.split()
            if parts and int(parts[0]) in OWN_TO_RAILSAFE:
                lines.append(" ".join([str(OWN_TO_RAILSAFE[int(parts[0])]), *parts[1:5]]))
        lines += [to_yolo(boxes[j], int(cls[j])) for j in new]
        shutil.copy2(src, out / "images" / src.name)
        (out / "labels" / f"{src.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        stats["frames"] += 1
        stats["human_boxes"] += len(gt)
        for j in new:
            stats["prelabels"][RAILSAFE[int(cls[j])]] += 1

    (out / "classes.txt").write_text("\n".join(RAILSAFE) + "\n", encoding="utf-8")
    (out / "data.yaml").write_text(
        "path: .\ntrain: images\nval: images\nnc: %d\nnames:\n%s" % (
            len(RAILSAFE), "".join(f"  {k}: {n}\n" for k, n in enumerate(RAILSAFE))), encoding="utf-8")
    stats["prelabels"] = dict(stats["prelabels"])
    stats["sequences"] = len(seq_sizes)
    (out / "report.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
