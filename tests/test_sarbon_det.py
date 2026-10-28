"""Sarbon detector: sizes, forward/backward, overfit a tiny batch, track-query frame step.

CPU-only. Run: .venv\\Scripts\\python.exe tests\\test_sarbon_det.py
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sarbon.data.coco import SyntheticDetection  # noqa: E402
from sarbon.models.box_ops import box_iou, cxcywh_to_xyxy, xyxy_to_cxcywh  # noqa: E402
from sarbon.models.criterion import SetCriterion  # noqa: E402
from sarbon.models.sarbon_det import SIZES, build  # noqa: E402


def batch(n=2, size=256):
    ds = SyntheticDetection(n, size, num_classes=3, seed=5)
    imgs, targets = [], []
    for i in range(n):
        img, b, l, _ = ds[i]
        imgs.append(img.float() / 255 - 0.5)
        targets.append({"labels": l, "boxes": xyxy_to_cxcywh(b / size),
                        "ids": torch.arange(len(l)) + 100 * i})
    return torch.stack(imgs), targets


def main():
    torch.manual_seed(0)
    for s in SIZES:
        m = build(s, num_classes=6)
        n_params = sum(p.numel() for p in m.parameters()) / 1e6
        print(f"Sarbon-{s.upper()}: {n_params:.1f}M params")

    m = build("s", num_classes=6).eval()
    with torch.no_grad():
        out = m(torch.randn(1, 3, 384, 640))
    assert out["pred_logits"].shape == (1, 300, 6) and out["pred_boxes"].shape == (1, 300, 4)
    assert len(out["aux"]) == 3 and torch.isfinite(out["pred_boxes"]).all()

    # overfit a tiny batch: proves model + matcher + losses agree
    imgs, targets = batch()
    m = build("n", num_classes=3, queries=50)
    crit = SetCriterion(3, project=m.project)  # MAL + L1/GIoU + FGL + DDF
    opt = torch.optim.AdamW(m.parameters(), 2e-4, weight_decay=1e-4)
    t0 = time.time()
    for it in range(301):
        out = m(imgs, targets=targets)  # training mode: + contrastive denoising queries
        assert "dn" in out and out["pred_logits"].shape[1] == 50
        loss, logs, _ = crit(out, targets)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 0.1)
        opt.step()
        if it % 100 == 0:
            print(f"iter {it}: loss {loss.item():.3f} {logs} ({time.time() - t0:.0f}s)")
    m.eval()
    with torch.no_grad():
        out = m(imgs)
    for i, t in enumerate(targets):
        scores = out["pred_logits"][i].sigmoid().max(-1).values
        keep = scores > 0.3
        iou, _ = box_iou(cxcywh_to_xyxy(t["boxes"]), cxcywh_to_xyxy(out["pred_boxes"][i][keep]))
        best = iou.max(1).values if keep.any() else torch.zeros(len(t["boxes"]))
        print(f"image {i}: {len(t['boxes'])} objects, {int(keep.sum())} confident preds, "
              f"best IoU per object {[round(v, 2) for v in best.tolist()]}")
        assert (best > 0.5).all(), "overfit failed"

    # next frame with track queries: matched queries become tracks bound to their IDs
    out = m(imgs)
    _, _, matches = crit(out, targets)
    tracks, fixed = {"embed": [], "boxes": []}, []
    t_max = max(len(q) for q, _ in matches)
    for i, (q, g) in enumerate(matches):
        pad = t_max - len(q)
        tracks["embed"].append(torch.cat([out["embed"][i, q], out["embed"][i, q[:1]].repeat(pad, 1) * 0]))
        tracks["boxes"].append(torch.cat([out["boxes"][i, q], torch.full((pad, 4), 0.5)]))
        fixed.append((torch.arange(len(q)), g))  # track j -> the gt it was matched to (same ID)
    tracks = {k: torch.stack(v) for k, v in tracks.items()}
    shifted = torch.roll(imgs, 4, -1)
    targets2 = [{**t, "boxes": t["boxes"] + torch.tensor([4 / 256, 0, 0, 0])} for t in targets]
    out2 = m(shifted, tracks=tracks)
    loss2, logs2, matches2 = crit(out2, targets2, fixed=fixed)
    assert out2["n_track"] == t_max and torch.isfinite(loss2)
    for (q, g), (q1, g1) in zip(matches2, fixed):
        assert torch.equal(q[:len(q1)], q1) and torch.equal(g[:len(g1)], g1)  # IDs kept
    print(f"track step: {t_max} track queries, loss {loss2.item():.3f}, IDs preserved")
    print("OK")


if __name__ == "__main__":
    main()
