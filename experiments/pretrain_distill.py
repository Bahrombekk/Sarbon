"""Label-free backbone pretraining: distil DINOv2 features into the Sarbon stem + dense core.

YOLO/D-FINE start from backbones pretrained on ImageNet or Objects365, which are not
licensed for commercial use. DINOv2 (code and weights Apache-2.0) is used as a frozen
teacher on our own commercially licensed images: the student's stride-16 features are
projected to the teacher's dimension and pulled towards the teacher's patch tokens with
a cosine loss (RT-DETRv4, 2510.25257, found cosine better than MSE). No labels needed.

  .venv\\Scripts\\python.exe experiments\\pretrain_distill.py --coco C:\\data\\coco --epochs 20
  then: train_sarbon.py ... --init runs\\pretrain\\s_distill.pt
CPU smoke test (random stand-in teacher, nothing downloaded):
  .venv\\Scripts\\python.exe experiments\\pretrain_distill.py --fake-teacher --synthetic --device cpu --size n --epochs 2
"""
import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sarbon.data.coco import ALL, CocoDetection, MosaicDetection, SyntheticDetection, collate  # noqa: E402
from sarbon.models.sarbon_det import build  # noqa: E402
from train_sarbon import param_groups, prep  # noqa: E402


class FakeTeacher(nn.Module):
    """Frozen random conv net with DINOv2's interface (patch 14) for offline smoke tests."""

    embed_dim = 64
    patch_size = 14

    def __init__(self):
        super().__init__()
        self.net = nn.Conv2d(3, 64, 14, 14)

    def forward_features(self, x):
        t = self.net(x).flatten(2).transpose(1, 2)
        return {"x_norm_patchtokens": t}


def teacher_tokens(teacher, x):
    """Patch tokens as a [B, C, h, w] map. Input is resized to a multiple of the patch size."""
    p = teacher.patch_size
    h, w = (round(s / p) * p for s in x.shape[-2:])
    x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
    t = teacher.forward_features(x)["x_norm_patchtokens"]
    return t.transpose(1, 2).reshape(x.shape[0], -1, h // p, w // p)


class Student(nn.Module):
    """Sarbon stem + dense core with a projection to the teacher's token width."""

    def __init__(self, det, teacher_dim):
        super().__init__()
        self.stem, self.core = det.stem, det.core
        d = det.cfg.hidden
        self.proj = nn.Sequential(nn.Conv2d(d, d, 1), nn.GELU(), nn.Conv2d(d, teacher_dim, 1))

    def forward(self, x):
        _, p8 = self.stem.forward_blocks(x)
        o8, o16, o32 = self.core(p8)
        return self.proj(o16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--teacher", default="dinov2_vits14", help="torch.hub name in facebookresearch/dinov2")
    ap.add_argument("--fake-teacher", action="store_true")
    ap.add_argument("--size", default="s")
    ap.add_argument("--img", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="runs/pretrain")
    args = ap.parse_args()
    torch.manual_seed(0)
    device = torch.device(args.device)
    if args.synthetic:
        size, ds, args.workers = 256, SyntheticDetection(64, 256), 0
    else:
        size = args.img
        # all classes: every commercially licensed COCO image is useful without labels
        ds = MosaicDetection(CocoDetection(args.coco, "train2017", size, ALL), prob=0.3)
    dl = DataLoader(ds, args.batch, shuffle=True, num_workers=args.workers, collate_fn=collate,
                    drop_last=True, persistent_workers=args.workers > 0)
    if args.fake_teacher:
        teacher = FakeTeacher()
    else:
        teacher = torch.hub.load("facebookresearch/dinov2", args.teacher)
    teacher = teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    det = build(args.size, 6, halo_zero=False)
    student = Student(det, teacher.embed_dim).to(device)
    opt = torch.optim.AdamW(param_groups(student, 0.05), args.lr)
    steps = args.epochs * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / max(1, steps // 20)) * 0.5 * (1 + math.cos(math.pi * min(i, steps) / steps)))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for ep in range(args.epochs):
        tot, n = 0.0, 0
        for it, (imgs, _, _, _) in enumerate(dl):
            x = prep(imgs, device)
            with torch.no_grad(), torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
                tgt = teacher_tokens(teacher, x).float()
            with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
                pred = student(x)
            pred = F.interpolate(pred.float(), size=tgt.shape[-2:], mode="bilinear", align_corners=False)
            loss = (1 - F.cosine_similarity(pred, tgt, dim=1)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            sched.step()
            tot, n = tot + loss.item(), n + 1
            if it % 100 == 0:
                print(f"ep {ep} it {it}/{len(dl)} cos-loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
        print(f"ep {ep}: mean cos-loss {tot / max(1, n):.4f}", flush=True)
        # save only detector weights the detector understands (stem + core)
        sd = {k: v for k, v in det.state_dict().items() if k.startswith(("stem.", "core."))}
        torch.save(sd, out_dir / f"{args.size}_distill.pt")


if __name__ == "__main__":
    main()
