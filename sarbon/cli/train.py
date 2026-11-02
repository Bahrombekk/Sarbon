"""Train the Sarbon detector.

Three modes:
  commercial (default): surveillance classes, commercial-license COCO images only -> product weights
  --benchmark         : all 80 classes, all images -> comparable with published YOLO/D-FINE numbers
                        (research use only; these weights are not for the product)
  --taxonomy railsafe --yolo DIR : RailSafe classes (person, bicycle, motorcycle, yengil, og'ir,
                        train) from commercial COCO + own labelled crossing data; the own data
                        labels vehicles only, so its other classes are masked out of the loss

Every option can also come from a YAML file (--config configs/train/*.yaml); flags given on the
command line override the file.

  sarbon-train --config configs/train/railsafe_s.yaml
  sarbon-train --coco C:\data\coco --size s --benchmark
CPU smoke test:
  sarbon-train --synthetic --device cpu --size n --epochs 2
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader

from ..config import parse_with_config
from ..data.coco import (ALL, RAILSAFE, RAILSAFE_FROM_COCO, SURVEILLANCE, CocoDetection,
                         MosaicDetection, SyntheticDetection, collate)
from ..data.yolo import WithClassMask, YoloDetection
from ..engine import ModelEMA, load_init, param_groups, prep, to_targets
from ..evaluation import evaluate
from ..models.criterion import HungarianMatcher, SetCriterion
from ..models.sarbon_det import build
from ..paths import RUNS


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
    ap.add_argument("--widths", help="stem widths at stride 2/4/8, e.g. 48,96,192 (backbone ablation)")
    ap.add_argument("--depths", help="Res blocks at stride 4/8, e.g. 2,3")
    ap.add_argument("--c16", type=int, help="channels of the stride-16 stage")
    ap.add_argument("--tag", default="", help="suffix for the output folder (ablations)")
    ap.add_argument("--init", help="pretrained weights: distillation output or a detector checkpoint")
    ap.add_argument("--init-classes", choices=["surveillance"],
                    help="class list of the --init detector; its class heads are then remapped by name")
    ap.add_argument("--taxonomy", default="surveillance", choices=["surveillance", "railsafe"])
    ap.add_argument("--yolo", help="folder with train.txt / val.txt (make_yolo_split.py) of own labelled data")
    ap.add_argument("--yolo-map", default="0:3,1:4", help="YOLO class -> our label (railsafe: yengil, ogir)")
    ap.add_argument("--own", action="append", default=[], metavar="SPLIT=MAP",
                    help="more own labelled data: split folder (train.txt/val.txt) = class map, "
                         "e.g. C:/data/smart_split=0:0 (repeatable; YAML: a list)")
    ap.add_argument("--eval-every", type=int, default=4)
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False, help="continue from <out>/ckpt.pt if present")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(RUNS / "sarbon"))
    args = parse_with_config(ap)
    torch.manual_seed(0)
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    names, own_vals = None, []
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
        # own labelled data sets: each annotates only its mapped classes (partial labels)
        sources = ([("own", args.yolo, args.yolo_map)] if args.yolo else []) + [
            (Path(s.split("=", 1)[0]).name.removesuffix("_split"), *s.split("=", 1)) for s in args.own]
        parts = [WithClassMask(mosaics[0], num_classes, range(num_classes))]
        for name, split, cmap in sources:
            ymap = {int(a): int(b) for a, b in (p.split(":") for p in cmap.split(","))}
            ytrain = YoloDetection(Path(split) / "train.txt", size, ymap, limit=args.limit)
            yval = YoloDetection(Path(split) / "val.txt", size, ymap, train=False, limit=args.limit)
            mosaics.append(MosaicDetection(ytrain, prob=0.5))
            # mosaic stays within one source, so every sample has a single annotation mask
            parts.append(WithClassMask(mosaics[-1], num_classes, sorted(set(ymap.values()))))
            own_vals.append((name, yval))
            print(f"own data {name} ({split}): train {len(ytrain)}, val {len(yval)}, "
                  f"classes {sorted(set(ymap.values()))}")
        if sources:
            train_ds = ConcatDataset(parts)
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
    own_dls = [(name, ds, DataLoader(ds, args.batch, num_workers=min(4, args.workers), collate_fn=collate,
                                     persistent_workers=args.workers > 0)) for name, ds in own_vals]

    arch = {}
    if args.widths:
        arch["widths"] = tuple(int(v) for v in args.widths.split(","))
    if args.depths:
        arch["depths"] = tuple(int(v) for v in args.depths.split(","))
    if args.c16:
        arch["c16"] = args.c16
    model = build(args.size, num_classes, halo_zero=not args.no_halo_zero, fusion=args.fusion,
                  block_type=args.block, act=args.act, **arch).to(device)
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
    log, best, t0, start = [], -1.0, time.time(), 0
    ckpt = out_dir / "ckpt.pt"
    if args.resume and ckpt.exists():
        c = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(c["model"])
        ema.ema.load_state_dict(c["ema"])
        ema.updates = c["ema_updates"]
        opt.load_state_dict(c["opt"])
        sched.load_state_dict(c["sched"])
        start, log, best = c["epoch"], c["log"], c["best"]
        print(f"resumed from {ckpt} at epoch {start}", flush=True)
    for ep in range(start, args.epochs):
        if ep >= args.epochs - args.no_aug_epochs and mosaics[0].prob > 0:
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
            for name, ds, dl in own_dls:
                # own data is scored on the classes it annotates; it also picks the best checkpoint
                cats = sorted(set(ds.class_map.values()))
                m.update({f"{name}_{k}": v for k, v in
                          evaluate(ema.ema, ds, dl, device, num_classes, size, cats).items()})
            m["epoch"] = ep + 1
            log.append(m)
            print("eval", json.dumps(m), flush=True)
            (out_dir / "metrics.json").write_text(json.dumps({"args": vars(args), "evals": log}, indent=1,
                                                             default=str))
            torch.save(ema.ema.state_dict(), out_dir / "last.pt")
            # best checkpoint: mean AP over the own data sets (the target domain), else COCO AP
            score = sum(m[f"{n}_AP"] for n, _, _ in own_dls) / len(own_dls) if own_dls else m["AP"]
            if score > best:
                best = score
                torch.save(ema.ema.state_dict(), out_dir / "best.pt")
        # full training state every epoch: a killed run loses at most one epoch (--resume)
        torch.save({"model": model.state_dict(), "ema": ema.ema.state_dict(), "ema_updates": ema.updates,
                    "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": ep + 1, "log": log,
                    "best": best}, out_dir / "ckpt.tmp")
        (out_dir / "ckpt.tmp").replace(ckpt)


if __name__ == "__main__":
    main()
