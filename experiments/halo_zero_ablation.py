"""Ablation: standard stem vs halo-zero (block-local) stem, everything else identical.

Decision rule (DESIGN.md): adopt halo-zero if AP drop <= 1.0 and AP_small drop <= 1.5.

GPU run (COCO 2017, surveillance classes, commercial-license images only):
  .venv\\Scripts\\python.exe experiments\\halo_zero_ablation.py --coco D:\\data\\coco --variant standard
  .venv\\Scripts\\python.exe experiments\\halo_zero_ablation.py --coco D:\\data\\coco --variant halo
CPU smoke test:
  .venv\\Scripts\\python.exe experiments\\halo_zero_ablation.py --synthetic --device cpu --epochs 2
"""
import argparse
import contextlib
import io
import json
import math
import sys
import time
from pathlib import Path

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sarbon.data.coco import CocoDetection, SyntheticDetection, TargetCollate, collate  # noqa: E402
from sarbon.models.proxy_detector import (STRIDE, ProxyDetector, build_targets, decode,  # noqa: E402
                                          detection_loss)

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255


def prep(imgs, device):
    x = imgs.to(device, non_blocking=True).float()
    return (x - MEAN.to(device)) / STD.to(device)


def coco_eval(model, ds, loader, device, num_classes):
    gt = {"images": [], "annotations": [], "categories": [{"id": c} for c in range(num_classes)]}
    for img_id, boxes, labels in ds.gt_letterboxed():
        gt["images"].append({"id": img_id})
        for b, l in zip(boxes, labels):
            w, h = b[2] - b[0], b[3] - b[1]
            gt["annotations"].append({"id": len(gt["annotations"]) + 1, "image_id": img_id,
                                      "category_id": l, "bbox": [b[0], b[1], w, h],
                                      "area": w * h, "iscrowd": 0})
    dets = []
    model.eval()
    with torch.no_grad():
        for imgs, _, _, meta in loader:
            with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
                heat, reg = model(prep(imgs, device))
            for (boxes, scores, labels), m in zip(decode(heat, reg), meta):
                for b, s, l in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
                    dets.append({"image_id": m[0], "category_id": l,
                                 "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]], "score": s})
    model.train()
    if not dets:
        return {"AP": 0.0, "AP50": 0.0, "AP_small": 0.0, "AP_medium": 0.0, "AP_large": 0.0}
    with contextlib.redirect_stdout(io.StringIO()):
        cg = COCO()
        cg.dataset = gt
        cg.createIndex()
        e = COCOeval(cg, cg.loadRes(dets), "bbox")
        e.evaluate()
        e.accumulate()
        e.summarize()
    s = e.stats
    return {"AP": s[0] * 100, "AP50": s[1] * 100, "AP_small": s[3] * 100,
            "AP_medium": s[4] * 100, "AP_large": s[5] * 100}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["standard", "halo"], default="halo")
    ap.add_argument("--coco", type=str)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="runs/halo_zero")
    args = ap.parse_args()
    torch.manual_seed(0)
    device = torch.device(args.device)

    if args.synthetic:
        train_ds = SyntheticDetection(64, args.size if args.size != 640 else 256)
        val_ds = SyntheticDetection(16, train_ds.size, seed=1000)
        num_classes, size = 3, train_ds.size
        args.workers = 0
    else:
        train_ds = CocoDetection(args.coco, "train2017", args.size, limit=args.limit)
        val_ds = CocoDetection(args.coco, "val2017", args.size, train=False, commercial_only=False)
        num_classes, size = len(train_ds.cat_ids), args.size
    print(f"train images {len(train_ds)} (commercial licenses only), val {len(val_ds)}")
    out_h = out_w = size // STRIDE
    train_dl = DataLoader(train_ds, args.batch, shuffle=True, num_workers=args.workers,
                          collate_fn=TargetCollate(build_targets, num_classes, out_h, out_w),
                          drop_last=True, pin_memory=device.type == "cuda",
                          persistent_workers=args.workers > 0)
    # few, persistent eval workers: respawning 16 per eval exhausted the Windows commit limit
    val_dl = DataLoader(val_ds, args.batch, num_workers=min(4, args.workers), collate_fn=collate,
                        persistent_workers=args.workers > 0)

    model = ProxyDetector(num_classes, halo_zero=args.variant == "halo").to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=0.05)
    steps = args.epochs * len(train_dl)
    warm = min(500, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1, (i + 1) / max(1, warm)) * 0.5 * (1 + math.cos(math.pi * i / steps)))
    out_dir = Path(args.out) / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    log, t0 = [], time.time()
    for ep in range(args.epochs):
        for it, (imgs, heat, reg, mask, _) in enumerate(train_dl):
            heat, reg, mask = (x.to(device, non_blocking=True) for x in (heat, reg, mask))
            with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
                hp, rp = model(prep(imgs, device))
            loss, parts = detection_loss(hp, rp, heat, reg, mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            sched.step()
            if it % 50 == 0:
                print(f"ep {ep} it {it}/{len(train_dl)} loss {loss.item():.3f} {parts} "
                      f"{time.time() - t0:.0f}s", flush=True)
        if ep == args.epochs - 1 or (ep + 1) % 4 == 0:
            m = coco_eval(model, val_ds, val_dl, device, num_classes)
            m["epoch"] = ep + 1
            log.append(m)
            print("eval", json.dumps(m), flush=True)
            (out_dir / "metrics.json").write_text(json.dumps({"args": vars(args), "evals": log}, indent=1))
    torch.save(model.state_dict(), out_dir / "model.pt")


if __name__ == "__main__":
    main()
