"""Where does Sarbon lose small objects? Class-agnostic recall of small ground-truth boxes
(area < 32^2 in the letterboxed frame) at three points of the detector, plus YOLO for reference:

  enc   : the 300 encoder proposals that become decoder queries (selection stage)
  final : all 300 decoder boxes, any score (localisation stage)
  @conf : decoder boxes with score >= conf (ranking / classification stage)

Low `enc` recall -> features or query selection miss them; high `enc` but low `final` ->
the decoder moves boxes away; high `final` but low `@conf` -> scores too low.

  .venv\\Scripts\\python.exe bench\\small_recall.py --sarbon runs\\railsafe\\s_railsafe\\best.pt \\
      --yolo-json runs\\yolo\\yolo11s_val.json
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sarbon.data.coco import collate  # noqa: E402
from sarbon.cli.compare import yolo_dets  # noqa: E402
from sarbon.data.yolo import YoloDetection  # noqa: E402
from sarbon.engine import prep  # noqa: E402
from sarbon.models.box_ops import cxcywh_to_xyxy  # noqa: E402
from sarbon.models.sarbon_det import build, load_weights  # noqa: E402

BINS = [(0, 16), (16, 32), (32, 96), (96, 1e9)]  # sqrt(area) in letterbox pixels


def iou(a, b):
    """a [N,4], b [M,4] xyxy -> [N,M]"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area = lambda x: (x[:, 2:] - x[:, :2]).prod(-1)
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


class Recall:
    def __init__(self):
        self.hit = {k: np.zeros(len(BINS)) for k in []}
        self.n = np.zeros(len(BINS))

    def add(self, key, gt, boxes, thr=0.5):
        side = np.sqrt((gt[:, 2:] - gt[:, :2]).prod(-1))
        found = iou(gt, boxes).max(1) >= thr if len(boxes) else np.zeros(len(gt), bool)
        h = self.hit.setdefault(key, np.zeros(len(BINS)))
        for j, (lo, hi) in enumerate(BINS):
            m = (side >= lo) & (side < hi)
            h[j] += found[m].sum()
            if key == "enc":
                self.n[j] += m.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="C:/data/pereezd_split")
    ap.add_argument("--sarbon", required=True)
    ap.add_argument("--yolo-json")
    ap.add_argument("--img", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--iou", type=float, default=0.5)
    args = ap.parse_args()
    cmap = {0: 3, 1: 4}
    ds = YoloDetection(Path(args.split) / "val.txt", args.img, cmap, train=False)
    model = load_weights(build("s", 6, halo_zero=False), args.sarbon).cuda().eval()
    dl = DataLoader(ds, 16, num_workers=4, collate_fn=collate)
    gts = {i: b for i, b, _ in ds.gt_letterboxed()}
    r = Recall()
    for imgs, _, _, meta in dl:
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            out = model(prep(imgs, torch.device("cuda")))
        enc = (cxcywh_to_xyxy(out["enc_boxes"].float()) * args.img).cpu().numpy()
        fin = (cxcywh_to_xyxy(out["pred_boxes"].float()) * args.img).cpu().numpy()
        score = out["pred_logits"].float().sigmoid().max(-1).values.cpu().numpy()
        for k, m in enumerate(meta):
            gt = np.asarray(gts[m[0]], np.float32).reshape(-1, 4)
            if not len(gt):
                continue
            r.add("enc", gt, enc[k], args.iou)
            r.add("final", gt, fin[k], args.iou)
            r.add(f"@{args.conf}", gt, fin[k][score[k] >= args.conf], args.iou)
    if args.yolo_json:
        by_img = {}
        for d in yolo_dets(ds, args.yolo_json, cmap):
            x, y, w, h = d["bbox"]
            by_img.setdefault(d["image_id"], []).append((x, y, x + w, y + h, d["score"]))
        for i, gt in gts.items():
            gt = np.asarray(gt, np.float32).reshape(-1, 4)
            if not len(gt):
                continue
            d = np.asarray(by_img.get(i, []), np.float32).reshape(-1, 5)
            r.add("yolo_any", gt, d[:, :4], args.iou)
            r.add(f"yolo@{args.conf}", gt, d[d[:, 4] >= args.conf, :4], args.iou)
    head = "".join(f"{f'{lo:g}-{hi:g}px' if hi < 1e9 else f'>{lo:g}px':>12s}" for lo, hi in BINS)
    print(f"{'GT boxes':14s}" + "".join(f"{int(n):12d}" for n in r.n))
    print(f"{f'recall@{args.iou:g}':14s}" + head)
    for k, h in r.hit.items():
        print(f"{k:14s}" + "".join(f"{100 * a / max(n, 1):11.1f}%" for a, n in zip(h, r.n)))


if __name__ == "__main__":
    main()
