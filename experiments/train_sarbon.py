"""Train the Sarbon detector on COCO.

Two modes:
  commercial (default): surveillance classes, commercial-license images only -> product weights
  --benchmark         : all 80 classes, all images -> comparable with published YOLO/D-FINE numbers
                        (research use only; these weights are not for the product)

  .venv\\Scripts\\python.exe experiments\\train_sarbon.py --coco D:\\data\\coco --size s
  .venv\\Scripts\\python.exe experiments\\train_sarbon.py --coco D:\\data\\coco --size s --benchmark
CPU smoke test:
  .venv\\Scripts\\python.exe experiments\\train_sarbon.py --synthetic --device cpu --size n --epochs 2
"""
import argparse
import contextlib
import copy
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
from sarbon.data.coco import (ALL, SURVEILLANCE, CocoDetection, MosaicDetection,  # noqa: E402
                              SyntheticDetection, collate)
from sarbon.models.box_ops import cxcywh_to_xyxy, xyxy_to_cxcywh  # noqa: E402
from sarbon.models.criterion import HungarianMatcher, SetCriterion  # noqa: E402
from sarbon.models.sarbon_det import build  # noqa: E402

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255


def prep(imgs, device):
    x = imgs.to(device, non_blocking=True).float()
    return (x - MEAN.to(device)) / STD.to(device)


def to_targets(boxes, labels, size, device):
    return [{"labels": l.to(device), "boxes": xyxy_to_cxcywh(b.to(device) / size).clamp(0, 1)}
            for b, l in zip(boxes, labels)]


class ModelEMA:
    def __init__(self, model, decay=0.9999, warmup=2000):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay, self.warmup, self.updates = decay, warmup, 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / self.warmup))
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)


@torch.no_grad()
def postprocess(out, size, k=300):
    """DETR-style: top-k over (query, class) pairs; boxes in input pixels xyxy."""
    scores = out["pred_logits"].sigmoid()
    b, q, c = scores.shape
    s, idx = scores.flatten(1).topk(min(k, q * c))
    boxes = cxcywh_to_xyxy(out["pred_boxes"]).gather(1, (idx // c)[..., None].expand(-1, -1, 4)) * size
    return [(boxes[i], s[i], idx[i] % c) for i in range(b)]


def evaluate(model, ds, loader, device, num_classes, size):
    gt = {"images": [], "annotations": [], "categories": [{"id": c} for c in range(num_classes)]}
    for img_id, boxes, labels in ds.gt_letterboxed():
        gt["images"].append({"id": img_id})
        for b, l in zip(boxes, labels):
            w, h = b[2] - b[0], b[3] - b[1]
            gt["annotations"].append({"id": len(gt["annotations"]) + 1, "image_id": img_id,
                                      "category_id": l, "bbox": [b[0], b[1], w, h], "area": w * h,
                                      "iscrowd": 0})
    dets = []
    model.eval()
    for imgs, _, _, meta in loader:
        with torch.no_grad(), torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
            out = model(prep(imgs, device))
        for (boxes, scores, labels), m in zip(postprocess(out, size), meta):
            for bb, s, l in zip(boxes.float().tolist(), scores.float().tolist(), labels.tolist()):
                dets.append({"image_id": m[0], "category_id": l,
                             "bbox": [bb[0], bb[1], bb[2] - bb[0], bb[3] - bb[1]], "score": s})
    if not dets:
        return {"AP": 0.0}
    with contextlib.redirect_stdout(io.StringIO()):
        cg = COCO()
        cg.dataset = gt
        cg.createIndex()
        e = COCOeval(cg, cg.loadRes(dets), "bbox")
        e.evaluate()
        e.accumulate()
        e.summarize()
    s = e.stats
    return {"AP": s[0] * 100, "AP50": s[1] * 100, "AP75": s[2] * 100, "AP_small": s[3] * 100,
            "AP_medium": s[4] * 100, "AP_large": s[5] * 100}


def param_groups(model, wd):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or n.endswith(".bias") else decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--benchmark", action="store_true", help="all classes + all images (research only)")
    ap.add_argument("--size", default="s", choices=["n", "s", "m"])
    ap.add_argument("--img", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=72)
    ap.add_argument("--no-aug-epochs", type=int, default=4, help="final epochs without mosaic")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--nwd", type=float, default=0.0, help="NWD weight in the matching cost")
    ap.add_argument("--no-halo-zero", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="runs/sarbon")
    args = ap.parse_args()
    torch.manual_seed(0)
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    if args.synthetic:
        size = 256
        base, val_ds = SyntheticDetection(64, size), SyntheticDetection(16, size, seed=1000)
        num_classes, args.workers = 3, 0
    else:
        size = args.img
        classes = ALL if args.benchmark else SURVEILLANCE
        base = CocoDetection(args.coco, "train2017", size, classes, commercial_only=not args.benchmark)
        val_ds = CocoDetection(args.coco, "val2017", size, classes, commercial_only=False, train=False)
        num_classes = len(base.cat_ids)
    train_ds = MosaicDetection(base, prob=0.5)
    print(f"train {len(base)} images, val {len(val_ds)}, classes {num_classes}, "
          f"{'benchmark (research only)' if args.benchmark else 'commercial'}")
    mk = lambda: DataLoader(train_ds, args.batch, shuffle=True, num_workers=args.workers,  # noqa: E731
                            collate_fn=collate, drop_last=True, pin_memory=device.type == "cuda",
                            persistent_workers=False)
    train_dl = mk()
    val_dl = DataLoader(val_ds, args.batch, num_workers=args.workers, collate_fn=collate)

    model = build(args.size, num_classes, halo_zero=not args.no_halo_zero).to(device)
    print(f"Sarbon-{args.size.upper()}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
    crit = SetCriterion(num_classes, HungarianMatcher(w_nwd=args.nwd), project=model.project)
    opt = torch.optim.AdamW(param_groups(model, args.wd), args.lr, betas=(0.9, 0.999))
    steps = args.epochs * len(train_dl)
    warm = min(2000, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / warm) * (0.5 * (1 + math.cos(math.pi * min(i, steps) / steps)) * 0.95 + 0.05))
    ema = ModelEMA(model)
    out_dir = Path(args.out) / f"{args.size}{'_bench' if args.benchmark else ''}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log, best, t0 = [], -1.0, time.time()
    for ep in range(args.epochs):
        if ep == args.epochs - args.no_aug_epochs:
            train_ds.prob = 0.0
            train_dl = mk()  # workers must see the new prob
        model.train()
        for it, (imgs, boxes, labels, _) in enumerate(train_dl):
            targets = to_targets(boxes, labels, size, device)
            with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
                out = model(prep(imgs, device), targets=targets)
            loss, parts, _ = crit(out, targets)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            opt.step()
            sched.step()
            ema.update(model)
            if it % 100 == 0:
                print(f"ep {ep} it {it}/{len(train_dl)} loss {loss.item():.3f} "
                      + " ".join(f"{k} {v:.3f}" for k, v in parts.items())
                      + f" lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s", flush=True)
        if (ep + 1) % 4 == 0 or ep == args.epochs - 1:
            m = evaluate(ema.ema, val_ds, val_dl, device, num_classes, size)
            m["epoch"] = ep + 1
            log.append(m)
            print("eval", json.dumps(m), flush=True)
            (out_dir / "metrics.json").write_text(json.dumps({"args": vars(args), "evals": log}, indent=1))
            torch.save(ema.ema.state_dict(), out_dir / "last.pt")
            if m["AP"] > best:
                best = m["AP"]
                torch.save(ema.ema.state_dict(), out_dir / "best.pt")


if __name__ == "__main__":
    main()
