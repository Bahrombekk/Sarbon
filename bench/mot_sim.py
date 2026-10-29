"""Synthetic fixed-camera MOT benchmark for tracker development (CPU only).

Simulates people walking in a static 1920x1080 view: entries/exits at the borders,
smooth turns, mutual occlusion when paths cross (the farther person gets weaker,
less frequent detections), static occluders (pillars) that hide people for long
gaps, false positives, box/score noise and appearance embeddings (with look-alike
groups). Detector outputs mimic a DETR head: boxes, scores and an embedding each.

Metrics follow MOTChallenge definitions: MOTA (FN + FP + IDSW), IDF1 (global
identity matching) and ID switches, at IoU >= 0.5.
Run: .venv\\Scripts\\python.exe bench\\mot_sim.py
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sarbon.track import SarbonTracker, TrackerConfig  # noqa: E402
from sarbon.track.tracker import iou_matrix  # noqa: E402

W, H = 1920, 1080


@dataclass
class SimConfig:
    frames: int = 600
    crowd: int = 18  # mean people on screen
    pillars: int = 3
    fp_rate: float = 2.0  # false positives per frame
    miss: float = 0.05  # random misses for visible people
    emb_dim: int = 64
    emb_noise: float = 0.35
    lookalike_groups: int = 4  # identities share a base appearance within a group
    seed: int = 0


def simulate(cfg: SimConfig):
    """-> list per frame of (gt_ids, gt_boxes, det_boxes, det_scores, det_feats)."""
    rng = np.random.default_rng(cfg.seed)
    # pillars / trees / parked trucks: wide enough to hide a walker for 40-100 frames
    pillars = [np.array([x, 200, x + rng.uniform(120, 220), 1080]) for x in rng.uniform(250, 1500, cfg.pillars)]
    groups = rng.normal(size=(cfg.lookalike_groups, cfg.emb_dim))
    people, next_id, frames = [], 1, []

    def spawn():
        nonlocal next_id
        h = rng.uniform(40, 200)
        side = rng.integers(4)
        x, y = [(0, rng.uniform(350, H)), (W, rng.uniform(350, H)), (rng.uniform(0, W), H),
                (rng.uniform(0, W), 300)][side]
        # walk through the scene towards a point on the opposite side (people pass by)
        target = [(W, rng.uniform(350, H)), (0, rng.uniform(350, H)), (rng.uniform(0, W), 300),
                  (rng.uniform(0, W), H)][side]
        v = np.array(target, float) - np.array([x, y])
        v = v / np.linalg.norm(v) * rng.uniform(0.8, 3.0) * h / 120
        app = groups[rng.integers(cfg.lookalike_groups)] * 0.8 + rng.normal(size=cfg.emb_dim) * 0.6
        people.append({"id": next_id, "p": np.array([x, y], float), "v": v, "h": h,
                       "app": app / np.linalg.norm(app)})
        next_id += 1

    for _ in range(cfg.crowd):
        spawn()
    for _ in range(cfg.frames):
        if len(people) < cfg.crowd and rng.random() < 0.3:
            spawn()
        for p in people:  # gentle random turns, feet position p
            ang = rng.normal(0, 0.02)
            c, s = np.cos(ang), np.sin(ang)
            p["v"] = np.array([[c, -s], [s, c]]) @ p["v"]
            p["p"] += p["v"]
        people = [p for p in people if -100 < p["p"][0] < W + 100 and 250 < p["p"][1] < H + 150]
        gts = []
        for p in people:
            w = 0.42 * p["h"]
            b = np.array([p["p"][0] - w / 2, p["p"][1] - p["h"], p["p"][0] + w / 2, p["p"][1]])
            vis_b = np.clip(b, [0, 0, 0, 0], [W, H, W, H])
            if (vis_b[2] - vis_b[0]) * (vis_b[3] - vis_b[1]) < 0.4 * (b[2] - b[0]) * (b[3] - b[1]):
                continue
            gts.append((p, vis_b))
        boxes = np.array([g[1] for g in gts]).reshape(-1, 4)
        iou = iou_matrix(boxes, boxes)
        dets, scores, feats = [], [], []
        for i, (p, b) in enumerate(gts):
            occl = 0.0
            for j in range(len(gts)):  # people nearer the camera (larger bottom y) occlude
                if j != i and gts[j][1][3] > b[3] and iou[i, j] > 0.15:
                    occl = max(occl, iou[i, j])
            for pl in pillars:
                ov = iou_matrix(b[None], pl[None])[0, 0]
                inter_w = max(0, min(b[2], pl[2]) - max(b[0], pl[0]))
                if inter_w > 0.5 * (b[2] - b[0]):
                    occl = max(occl, 0.95)
                elif ov > 0:
                    occl = max(occl, 0.4)
            p_det = (1 - cfg.miss) * (1 - occl) ** 1.5
            if rng.random() > p_det:
                continue
            w, h = b[2] - b[0], b[3] - b[1]
            jit = rng.normal(0, [0.04 * w, 0.03 * h, 0.04 * w, 0.03 * h]) * (1 + 2 * occl)
            dets.append(b + jit)
            scores.append(float(np.clip(rng.normal(0.85 - 0.6 * occl, 0.1), 0.02, 0.99)))
            f = p["app"] + rng.normal(0, cfg.emb_noise * (1 + 2 * occl), cfg.emb_dim) / np.sqrt(cfg.emb_dim) * 4
            feats.append(f)
        for _ in range(rng.poisson(cfg.fp_rate)):
            h = rng.uniform(30, 160)
            x, y = rng.uniform(0, W - 0.4 * h), rng.uniform(250, H - h)
            dets.append(np.array([x, y, x + 0.42 * h, y + h]))
            scores.append(float(rng.uniform(0.05, 0.5)))
            feats.append(rng.normal(size=cfg.emb_dim))
        frames.append(([g[0]["id"] for g in gts], boxes, np.array(dets).reshape(-1, 4),
                       np.array(scores), np.array(feats).reshape(-1, cfg.emb_dim)))
    return frames


def evaluate(frames, outputs, thr=0.5):
    """MOTA, IDF1, IDSW, FP, FN over the sequence."""
    n_gt = fp = fn = idsw = 0
    last = {}
    pair_count, gt_count, pr_count = {}, {}, {}
    for (gt_ids, gt_boxes, *_), out in zip(frames, outputs):
        pr_ids = [o[0] for o in out]
        pr_boxes = np.array([o[1] for o in out]).reshape(-1, 4)
        n_gt += len(gt_ids)
        iou = iou_matrix(gt_boxes, pr_boxes)
        for g in gt_ids:
            gt_count[g] = gt_count.get(g, 0) + 1
        for p in pr_ids:
            pr_count[p] = pr_count.get(p, 0) + 1
        if iou.size:
            r, c = linear_sum_assignment(-iou)
            ok = iou[r, c] >= thr
            r, c = r[ok], c[ok]
        else:
            r = c = np.array([], int)
        matched_g = set(r.tolist())
        fn += len(gt_ids) - len(r)
        fp += len(pr_ids) - len(c)
        for i, j in zip(r, c):
            g, p = gt_ids[i], pr_ids[j]
            if g in last and last[g] != p:
                idsw += 1
            last[g] = p
            pair_count[(g, p)] = pair_count.get((g, p), 0) + 1
        del matched_g
    gids, pids = sorted(gt_count), sorted(pr_count)
    gi, pi = {g: i for i, g in enumerate(gids)}, {p: i for i, p in enumerate(pids)}
    m = np.zeros((len(gids), len(pids)))
    for (g, p), v in pair_count.items():
        m[gi[g], pi[p]] = v
    if m.size:
        r, c = linear_sum_assignment(-m)
        idtp = m[r, c].sum()
    else:
        idtp = 0
    idf1 = 2 * idtp / max(1, sum(gt_count.values()) + sum(pr_count.values()))
    return {"MOTA": 100 * (1 - (fn + fp + idsw) / max(1, n_gt)), "IDF1": 100 * idf1, "IDSW": idsw,
            "FP": fp, "FN": fn, "GT_ids": len(gids), "pred_ids": len(pids)}


def run(frames, cfg):
    tr = SarbonTracker(cfg)
    return [tr.update(b, s, f) for _, _, b, s, f in frames]


VARIANTS = {
    "ByteTrack (Ultralytics default logic)": TrackerConfig(),
    "+ NSA Kalman": TrackerConfig(nsa=True),
    "+ OC-SORT ORU": TrackerConfig(oru=True),
    "+ OC-SORT OCR": TrackerConfig(ocr=True),
    "+ OC-SORT OCM": TrackerConfig(ocm=0.2),
    "+ embeddings (BoT-SORT-style)": TrackerConfig(emb=True),
    "+ long-term memory re-ID (Sarbon)": TrackerConfig(emb=True, reid=True),
    "Sarbon: re-ID + ORU + NSA": TrackerConfig(emb=True, reid=True, oru=True, nsa=True),
}


if __name__ == "__main__":
    seqs = [simulate(SimConfig(seed=s, crowd=c)) for s, c in ((0, 12), (1, 18), (2, 28))]
    print(f"{'tracker':40s} {'MOTA':>6s} {'IDF1':>6s} {'IDSW':>5s} {'FP':>5s} {'FN':>6s} {'IDs':>5s}")
    for name, cfg in VARIANTS.items():
        r = [evaluate(f, run(f, cfg)) for f in seqs]
        avg = {k: np.mean([x[k] for x in r]) for k in r[0]}
        print(f"{name:40s} {avg['MOTA']:6.1f} {avg['IDF1']:6.1f} {avg['IDSW']:5.0f} "
              f"{avg['FP']:5.0f} {avg['FN']:6.0f} {avg['pred_ids']:5.0f} (GT {avg['GT_ids']:.0f})")
