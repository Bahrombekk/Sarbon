"""Train the Sarbon detector on COCO.

Two modes:
  commercial (default): surveillance classes, commercial-license images only -> product weights
  --benchmark         : all 80 classes, all images -> comparable with published YOLO/D-FINE numbers
                        (research use only; these weights are not for the product)
  --taxonomy railsafe --yolo DIR : RailSafe classes (person, bicycle, motorcycle, yengil, og'ir,
                        train) from commercial COCO + own labelled crossing data; the own data
                        labels vehicles only, so its other classes are masked out of the loss

  .venv\\Scripts\\python.exe experiments\\train_sarbon.py --coco D:\\data\\coco --size s
  .venv\\Scripts\\python.exe experiments\\train_sarbon.py --coco D:\\data\\coco --size s --benchmark
  .venv\\Scripts\\python.exe experiments\\train_sarbon.py --coco C:\\data\\coco --taxonomy railsafe \\
      --yolo C:\\data\\pereezd_split --init runs\\sarbon\\s_distill\\best.pt --init-classes surveillance
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
from torch.utils.data import ConcatDataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sarbon.data.coco import (ALL, RAILSAFE, RAILSAFE_FROM_COCO, SURVEILLANCE, CocoDetection,  # noqa: E402
                              MosaicDetection, SyntheticDetection, collate)
from sarbon.data.yolo import WithClassMask, YoloDetection  # noqa: E402
from sarbon.models.box_ops import cxcywh_to_xyxy, xyxy_to_cxcywh  # noqa: E402
from sarbon.models.criterion import HungarianMatcher, SetCriterion  # noqa: E402
from sarbon.models.sarbon_det import build  # noqa: E402

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255


def prep(imgs, device):
    x = imgs.to(device, non_blocking=True).float()
    return (x - MEAN.to(device)) / STD.to(device)


def to_targets(boxes, labels, size, device, meta=None):
    out = [{"labels": l.to(device), "boxes": xyxy_to_cxcywh(b.to(device) / size).clamp(0, 1)}
           for b, l in zip(boxes, labels)]
    for t, m in zip(out, meta or []):
        if torch.is_tensor(m[-1]):  # WithClassMask: classes annotated in this image
            t["valid"] = m[-1].to(device)
    return out


CLASS_KEYS = ("enc_cls.", "cls_heads.", "dn_label.")


def load_init(model, path, src_names=None, dst_names=None, alias=None):
    """Load every tensor whose shape fits. Class-indexed tensors are taken only when the
    source class names are given: rows are then copied by name (alias: dst -> src name),
    and classes the source never saw keep their fresh initialisation."""
    sd = torch.load(path, map_location="cpu")
    own = model.state_dict()
    take, remapped = {}, 0
    for k, v in sd.items():
        if k not in own:
            continue
        if k.startswith(CLASS_KEYS):
            if not (src_names and dst_names):
                continue
            new = own[k].clone()
            for j, name in enumerate(dst_names):
                src = (alias or {}).get(name, name)
                if src in src_names:
                    new[j] = v[src_names.index(src)]
                    remapped += 1
            if k.startswith("dn_label."):  # last row: padding token
                new[-1] = v[-1]
            take[k] = new
        elif v.shape == own[k].shape:
            take[k] = v
    model.load_state_dict(take, strict=False)
    return len(take), remapped


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


def evaluate(model, ds, loader, device, num_classes, size, cats=None):
    """COCO AP; cats: evaluate only these classes (the ones a dataset annotates)."""
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
        if cats is not None:
            e.params.catIds = list(cats)
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
    ap.add_argument("--limit", type=int, default=None, help="use only N train/val images (smoke tests)")
    ap.add_argument("--fusion", default="concat", choices=["concat", "weighted", "gated"])
    ap.add_argument("--block", default="res", choices=["res", "rep", "dw"])
    ap.add_argument("--act", default="relu", choices=["relu", "silu", "gelu"])
    ap.add_argument("--tag", default="", help="suffix for the output folder (ablations)")
    ap.add_argument("--init", help="pretrained weights: distillation output or a detector checkpoint")
    ap.add_argument("--init-classes", choices=["surveillance"],
                    help="class list of the --init detector; its class heads are then remapped by name")
    ap.add_argument("--taxonomy", default="surveillance", choices=["surveillance", "railsafe"])
    ap.add_argument("--yolo", help="folder with train.txt / val.txt (make_yolo_split.py) of own labelled data")
    ap.add_argument("--yolo-map", default="0:3,1:4", help="YOLO class -> our label (railsafe: yengil, ogir)")
    ap.add_argument("--eval-every", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="runs/sarbon")
    args = ap.parse_args()
    torch.manual_seed(0)
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    names, yolo_val = None, None
    if args.synthetic:
        size = 256
        base, val_ds = SyntheticDetection(64, size), SyntheticDetection(16, size, seed=1000)
        num_classes, args.workers = 3, 0
        mosaics = [MosaicDetection(base, prob=0.5)]
        train_ds = mosaics[0]
    else:
        size = args.img
        if args.benchmark:
            kw = {"classes": ALL}
        elif args.taxonomy == "railsafe":
            kw, names = {"class_map": RAILSAFE_FROM_COCO}, RAILSAFE
        else:
            kw, names = {"classes": SURVEILLANCE}, SURVEILLANCE
        base = CocoDetection(args.coco, "train2017", size, commercial_only=not args.benchmark,
                             limit=args.limit, **kw)
        val_ds = CocoDetection(args.coco, "val2017", size, commercial_only=False, train=False,
                               limit=args.limit, **kw)
        num_classes = base.num_classes
        mosaics = [MosaicDetection(base, prob=0.5)]
        train_ds = mosaics[0]
        if args.yolo:
            # own labelled data: annotates only the mapped classes (partial labels)
            ymap = {int(a): int(b) for a, b in (p.split(":") for p in args.yolo_map.split(","))}
            ytrain = YoloDetection(Path(args.yolo) / "train.txt", size, ymap, limit=args.limit)
            yolo_val = YoloDetection(Path(args.yolo) / "val.txt", size, ymap, train=False, limit=args.limit)
            mosaics.append(MosaicDetection(ytrain, prob=0.5))
            # mosaic stays within one source, so every sample has a single annotation mask
            train_ds = ConcatDataset([WithClassMask(mosaics[0], num_classes, range(num_classes)),
                                      WithClassMask(mosaics[1], num_classes, sorted(set(ymap.values())))])
            print(f"own data {args.yolo}: train {len(ytrain)}, val {len(yolo_val)}, classes {sorted(set(ymap.values()))}")
    print(f"train {len(train_ds)} images, val {len(val_ds)}, classes {num_classes}, "
          f"{'benchmark (research only)' if args.benchmark else 'commercial'}")
    # persistent workers: on Windows every respawn re-imports torch (~1 min) and repeated
    # respawns exhausted the commit limit (WinError 1114 loading shm.dll)
    mk = lambda: DataLoader(train_ds, args.batch, shuffle=True, num_workers=args.workers,  # noqa: E731
                            collate_fn=collate, drop_last=True, pin_memory=device.type == "cuda",
                            persistent_workers=args.workers > 0)
    train_dl = mk()
    val_dl = DataLoader(val_ds, args.batch, num_workers=min(4, args.workers), collate_fn=collate,
                        persistent_workers=args.workers > 0)
    yolo_dl = yolo_val and DataLoader(yolo_val, args.batch, num_workers=min(4, args.workers),
                                      collate_fn=collate, persistent_workers=args.workers > 0)

    model = build(args.size, num_classes, halo_zero=not args.no_halo_zero, fusion=args.fusion,
                  block_type=args.block, act=args.act).to(device)
    print(f"Sarbon-{args.size.upper()}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
    if args.init:
        src = SURVEILLANCE if args.init_classes == "surveillance" else None
        # RailSafe classes start from their nearest COCO class (car -> yengil, truck -> og'ir)
        n, rows = load_init(model, args.init, src, names, alias={"yengil": "car", "ogir": "truck"})
        print(f"init from {args.init}: {n} tensors, {rows} class rows remapped"
              + ("" if src else "; class heads start fresh"))
    crit = SetCriterion(num_classes, HungarianMatcher(w_nwd=args.nwd), project=model.project)
    opt = torch.optim.AdamW(param_groups(model, args.wd), args.lr, betas=(0.9, 0.999))
    steps = args.epochs * len(train_dl)
    warm = max(1, min(2000, steps // 20))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / warm) * (0.5 * (1 + math.cos(math.pi * min(i, steps) / steps)) * 0.95 + 0.05))
    ema = ModelEMA(model)
    out_dir = Path(args.out) / f"{args.size}{'_bench' if args.benchmark else ''}{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log, best, t0 = [], -1.0, time.time()
    for ep in range(args.epochs):
        if ep == args.epochs - args.no_aug_epochs:
            for m in mosaics:
                m.prob = 0.0
            del train_dl  # shut the old workers down before starting ones that see prob = 0
            train_dl = mk()
        model.train()
        for it, (imgs, boxes, labels, meta) in enumerate(train_dl):
            targets = to_targets(boxes, labels, size, device, meta)
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
        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            m = evaluate(ema.ema, val_ds, val_dl, device, num_classes, size)
            if yolo_val is not None:
                # own data is scored on the classes it annotates; it also picks the best checkpoint
                cats = sorted(set(yolo_val.class_map.values()))
                m.update({f"own_{k}": v for k, v in
                          evaluate(ema.ema, yolo_val, yolo_dl, device, num_classes, size, cats).items()})
            m["epoch"] = ep + 1
            log.append(m)
            print("eval", json.dumps(m), flush=True)
            (out_dir / "metrics.json").write_text(json.dumps({"args": vars(args), "evals": log}, indent=1))
            torch.save(ema.ema.state_dict(), out_dir / "last.pt")
            score = m.get("own_AP", m["AP"])
            if score > best:
                best = score
                torch.save(ema.ema.state_dict(), out_dir / "best.pt")


if __name__ == "__main__":
    main()
