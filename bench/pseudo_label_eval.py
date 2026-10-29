"""How good are background-subtraction pseudo-labels? Synthetic fixed-camera video with known
ground truth: textured static background, moving textured objects, sensor noise, an
illumination drift and a swaying "tree" (moving texture that is not an object).

Reports box precision / recall (IoU >= 0.5) of the kept pseudo tracks and identity purity
(fraction of a pseudo track's boxes that belong to its majority ground-truth object).
Run: .venv\\Scripts\\python.exe bench\\pseudo_label_eval.py
"""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sarbon.data.pseudo import PseudoConfig, pseudo_label  # noqa: E402
from sarbon.track.tracker import iou_matrix  # noqa: E402
from scenes import Camera  # noqa: E402

H, W, N = 360, 640, 500


def video(seed, noise=0.02):
    """-> (frames BGR uint8, per-frame GT [(id, xyxy)])"""
    cam = Camera(H, W, "moderate", seed=seed, device="cpu")
    rng = np.random.default_rng(seed)
    frames, gts = [], []
    tree = (rng.integers(50, W - 120), rng.integers(20, H - 140))
    for t in range(N):
        f = cam.next().numpy()  # RGB float, moves objects
        f = f * (1 + 0.1 * np.sin(t / 80))  # slow illumination drift
        tx, ty = tree  # swaying tree: texture shifts back and forth, but goes nowhere
        shift = int(4 * np.sin(t / 3))
        f[ty:ty + 120, tx:tx + 80] = np.roll(f[ty:ty + 120, tx:tx + 80], shift, axis=1)
        f = f + rng.normal(0, noise, f.shape)
        frames.append((np.clip(f[:, :, ::-1], 0, 1) * 255).astype(np.uint8))
        boxes = []
        for k, ((y, x), (sh, sw)) in enumerate(zip(cam.pos.int().tolist(), cam.size.tolist())):
            boxes.append((k, np.array([x, y, x + sw, y + sh], float)))
        gts.append(boxes)
    return frames, gts


def painted_out(labels, i, gt_boxes, cfg):
    """GT boxes whose area is >= 50% painted over in the training image of frame i."""
    un = labels.unlabelled.get(i, np.zeros((0, 4)))
    if not cfg.paint or not len(un) or i not in labels.backgrounds:
        return np.zeros(len(gt_boxes), bool)
    m = np.zeros((H, W), bool)
    keep = np.zeros((H, W), bool)
    for x0, y0, x1, y1 in [b for _, b in labels.get(i, [])]:
        keep[max(0, int(y0)):int(y1) + 1, max(0, int(x0)):int(x1) + 1] = True
    for x0, y0, x1, y1 in un:
        g = cfg.paint_margin
        m[max(0, int(y0) - g):int(y1) + g + 1, max(0, int(x0) - g):int(x1) + g + 1] = True
    m &= ~keep
    out = []
    for x0, y0, x1, y1 in gt_boxes:
        reg = m[max(0, int(y0)):int(y1), max(0, int(x0)):int(x1)]
        out.append(reg.size > 0 and reg.mean() >= 0.5)
    return np.array(out, bool)


def evaluate(labels, gts, warmup, cfg=None, written_only=False):
    """Box P/R and ID purity. With written_only: only frames that become training images,
    GT painted out of the image is not counted as missed (it is no longer in the image)."""
    tp = fp = fn = painted = 0
    purity = {}
    for i in range(warmup, N):
        if written_only and (i % cfg.stride or not labels.get(i)):
            continue
        pr = labels.get(i, [])
        gt = gts[i]
        if written_only:
            out = painted_out(labels, i, [b for _, b in gt], cfg)
            painted += int(out.sum())
            gt = [g for g, o in zip(gt, out) if not o]
        pb = np.array([b for _, b in pr]).reshape(-1, 4)
        gb = np.array([b for _, b in gt]).reshape(-1, 4)
        iou = iou_matrix(pb, gb)
        if iou.size:
            r, c = linear_sum_assignment(-iou)
            ok = iou[r, c] >= 0.5
            r, c = r[ok], c[ok]
        else:
            r = c = np.array([], int)
        tp += len(r)
        fp += len(pb) - len(r)
        fn += len(gb) - len(r)
        for a, b in zip(r, c):
            purity.setdefault(pr[a][0], []).append(gt[b][0])
    pur = [np.max(np.bincount(v)) / len(v) for v in purity.values()]
    return (tp / max(1, tp + fp), tp / max(1, tp + fn), float(np.mean(pur)) if pur else 0.0,
            len(purity), fn / max(1, tp + fn + painted))


if __name__ == "__main__":
    variants = {"tracks only": PseudoConfig(warmup=60, interp_gap=0, paint=False),
                "+ gap interpolation": PseudoConfig(warmup=60, paint=False),
                "+ interpolation + background painting": PseudoConfig(warmup=60)}
    data = [video(seed) for seed in range(3)]
    print("training images: precision | recall of what is visible | ID purity | pseudo IDs (GT 6)"
          " | visible but unlabelled")
    for name, cfg in variants.items():
        rows = []
        for frames, gts in data:
            labels = pseudo_label(frames, cfg)
            rows.append(evaluate(labels, gts, cfg.warmup, cfg, written_only=True))
        p, r, pur, n, miss = np.mean(rows, 0)
        print(f"{name:40s} P {p:.3f} | R {r:.3f} | purity {pur:.3f} | IDs {n:4.1f} | unlabelled {miss:.3f}")
