"""Sarbon vs YOLO on our own labelled val set: same images, same COCO evaluator, same GPU.

  .venv\\Scripts\\python.exe bench\\compare_own.py --sarbon runs\\railsafe\\s_railsafe\\best.pt \\
      --yolo-json runs\\yolo\\yolo11s_val.json --yolo-speed runs\\yolo\\yolo11s_speed.json

YOLO detections come from scripts/yolo_own.py (run in the Ultralytics env) as normalised
xyxy boxes; they are mapped into the same letterboxed frame as the ground truth, so both
models are scored by the identical pycocotools call. Speed is network-only, batch 1,
fp16/bf16 at the same input size (dense Sarbon; the block-sparse video gain is extra).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from sarbon.data.coco import collate  # noqa: E402
from sarbon.data.yolo import YoloDetection, imread_at_least  # noqa: E402
from sarbon.models.sarbon_det import build  # noqa: E402
from train_sarbon import coco_metrics, evaluate  # noqa: E402


def yolo_dets(ds, path, class_map):
    raw = json.loads(Path(path).read_text())
    dets = []
    for im in ds.images:
        img = imread_at_least(im["file_name"], 1)
        h, w = img.shape[:2]
        s = min(ds.size / h, ds.size / w)
        px, py = (ds.size - round(w * s)) // 2, (ds.size - round(h * s)) // 2
        for x0, y0, x1, y1, score, c in raw.get(im["file_name"], []):
            if int(c) not in class_map:
                continue
            a, b = x0 * w * s + px, y0 * h * s + py
            dets.append({"image_id": im["id"], "category_id": class_map[int(c)],
                         "bbox": [a, b, (x1 - x0) * w * s, (y1 - y0) * h * s], "score": score})
    return dets


def sarbon_speed(model, size):
    x = torch.randn(1, 3, size, size, device="cuda")
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        for _ in range(30):
            model(x)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(200):
            model(x)
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / 200 * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="C:/data/pereezd_split")
    ap.add_argument("--sarbon", required=True)
    ap.add_argument("--size", default="s")
    ap.add_argument("--num-classes", type=int, default=6)
    ap.add_argument("--map", default="0:3,1:4", help="own YOLO class -> Sarbon label")
    ap.add_argument("--yolo-json")
    ap.add_argument("--yolo-speed", help="json written by yolo_own.py speed")
    ap.add_argument("--img", type=int, default=640)
    ap.add_argument("--out", default="runs/compare_own.json")
    args = ap.parse_args()
    cmap = {int(a): int(b) for a, b in (p.split(":") for p in args.map.split(","))}
    cats = sorted(set(cmap.values()))
    ds = YoloDetection(Path(args.split) / "val.txt", args.img, cmap, train=False)
    model = build(args.size, args.num_classes, halo_zero=False).cuda().eval()
    model.load_state_dict(torch.load(args.sarbon, map_location="cpu"))
    dl = DataLoader(ds, 16, num_workers=4, collate_fn=collate)
    res = {"val_images": len(ds), "classes": cats,
           "sarbon": evaluate(model, ds, dl, torch.device("cuda"), args.num_classes, args.img, cats)}
    res["sarbon"]["ms_per_frame"] = sarbon_speed(model, args.img)
    res["sarbon"]["params_M"] = sum(p.numel() for p in model.parameters()) / 1e6
    if args.yolo_json:
        # YOLO was trained on the same label ids as the own data, so the same map applies
        res["yolo"] = coco_metrics(ds.gt_letterboxed(), yolo_dets(ds, args.yolo_json, cmap), args.num_classes, cats)
        if args.yolo_speed:
            res["yolo"]["ms_per_frame"] = json.loads(Path(args.yolo_speed).read_text())["ms_per_frame"]
    Path(args.out).write_text(json.dumps(res, indent=1))
    keys = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "ms_per_frame"]
    print(f"{'':10s}" + "".join(f"{k:>13s}" for k in keys))
    for name in ("sarbon", "yolo"):
        if name in res:
            print(f"{name:10s}" + "".join(f"{res[name].get(k, float('nan')):13.2f}" for k in keys))


if __name__ == "__main__":
    main()
