"""SafeZone test: Sarbon models vs SmartGUI YOLO helmet models, same images, same evaluator.

YOLO detections come from tools/safezone_yolo_dets.py (run in an env with Ultralytics, e.g.
SmartGUI's venv), so no AGPL code is imported here.

    python tools/safezone_eval.py --split test light=runs/safezone/s_safezone_light_1024/best.pt@1024 \
        heavy=runs/safezone/s_safezone_heavy_1280/best.pt@1280 --yolo helmet_v4=C:/data/safezone/yolo_helmet_v4_test.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sarbon.cli.compare import yolo_dets  # noqa: E402
from sarbon.data.coco import SAFEZONE, collate  # noqa: E402
from sarbon.data.yolo import YoloDetection  # noqa: E402
from sarbon.evaluation import coco_metrics  # noqa: E402
from sarbon.evaluation.coco_eval import postprocess, prep  # noqa: E402
from sarbon.models.sarbon_det import load_detector  # noqa: E402

IDENT = {0: 0, 1: 1, 2: 2}


def sarbon_dets(weights, ds, size):
    m = load_detector(weights, "s", len(SAFEZONE), halo_zero=False).cuda().eval()
    out = []
    for imgs, _, _, meta in DataLoader(ds, 8, num_workers=4, collate_fn=collate):
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            o = m(prep(imgs, torch.device("cuda")))
        for (bx, sc, lb), mm in zip(postprocess(o, size), meta):
            out += [{"image_id": mm[0], "category_id": l, "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]], "score": s}
                    for b, s, l in zip(bx.float().tolist(), sc.float().tolist(), lb.tolist())]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", help="name=weights.pt@size")
    ap.add_argument("--yolo", nargs="*", default=[], help="name=detections.json (normalised xyxy, score, class)")
    ap.add_argument("--data", default="C:/data/safezone")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", help="write results JSON here")
    args = ap.parse_args()
    sets = {}

    def at(size):
        if size not in sets:
            ds = YoloDetection(Path(args.data) / f"{args.split}.txt", size, IDENT, train=False)
            sets[size] = ds, ds.gt_letterboxed()
        return sets[size]

    dets = {}
    for spec in args.models:
        name, w = spec.split("=", 1)
        w, size = w.rsplit("@", 1)
        size = int(size)
        dets[name] = size, sarbon_dets(w, at(size)[0], size)
    for spec in args.yolo:
        name, p = spec.split("=", 1)
        dets[name] = 1024, yolo_dets(at(1024)[0], p, IDENT)  # normalised boxes: any frame works
    res = {}
    for name, (size, d) in dets.items():
        gt = at(size)[1]
        r = {"AP": coco_metrics(gt, d, 3, [0, 1, 2])["AP"]}
        for c, cname in enumerate(SAFEZONE):
            m = coco_metrics(gt, d, 3, [c])
            r.update({f"{cname}_AP": m["AP"], f"{cname}_AP50": m["AP50"], f"{cname}_APs": m["AP_small"]})
        res[name] = {k: round(v, 1) for k, v in r.items()}
        print(f"{name:24s}", res[name], flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
