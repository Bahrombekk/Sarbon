"""Replay a compute policy on a dense reference (tools/dense_reference.py) and score it.

On a computed frame the tracker gets that frame's dense detections (exactly what the live
detector would output); on a skipped frame it only coasts. The reference output is the same
tracker fed on every frame, so the scores measure what skipping costs, nothing else:

  recall      reference boxes matched (IoU >= 0.5) by the policy's output on the same frame
  precision   policy boxes that match a reference box
  delay       frames from a reference track's first appearance until the policy outputs it
  missed      reference tracks (alive >= 1 s) the policy never outputs
  compute     fraction of frames the detector ran on
Per-class numbers use the class of the detection a track was last matched to.
"""
import numpy as np

from ..track import SarbonTracker, TrackerConfig

TRACKER = dict(emb=False, color=0.0, reid=False, ground=True)  # no embeddings stored in the reference


def load(path):
    z = np.load(path)
    return {"dets": z["dets"], "off": z["off"], "thumb": z["thumb"], "fps": float(z["fps"]),
            "size": tuple(int(v) for v in z["size"])}


def iou(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area = lambda x: (x[:, 2:] - x[:, :2]).prod(-1)  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def run(ref, policy, frames=None):
    """Policy output per frame: list of (track id, box, class) and the computed-frame mask."""
    n = len(ref["thumb"]) if frames is None else min(frames, len(ref["thumb"]))
    cam, mask = _Cam(ref, policy), np.zeros(n, bool)
    for t in range(n):
        mask[t] = policy.should_compute(t, ref["thumb"][t], cam.tr, ref["fps"])
        cam.step(t, mask[t])
    return cam.out, mask


class _Cam:
    def __init__(self, ref, policy):
        self.ref, self.policy = ref, policy
        self.tr = SarbonTracker(TrackerConfig(**TRACKER), ref["size"])
        self.cls_of, self.out = {}, []

    def step(self, t, compute):
        ref, tr = self.ref, self.tr
        if compute:
            d = ref["dets"][ref["off"][t]:ref["off"][t + 1]]
            res = tr.update(d[:, :4], d[:, 4])
            if len(res) and len(d):
                near = iou(np.array([b for _, b, _, _ in res]), d[:, :4]).argmax(1)
                for (tid, _, _, _), j in zip(res, near):
                    self.cls_of[tid] = int(d[j, 5])
            self.policy.computed(t, ref["thumb"][t])
        else:
            res = tr.coast()
        self.out.append([(tid, b, self.cls_of.get(tid, -1)) for tid, b, _, _ in res])


def run_shared(refs, policies, per_tick, frames, max_age=2.0):
    """Cameras share one detector budget: each tick the `per_tick` cameras with the highest
    perception debt compute (cameras over `max_age` seconds without a compute go first).
    Returns per camera (output, computed mask), like run()."""
    cams = [_Cam(r, p) for r, p in zip(refs, policies)]
    masks = [np.zeros(frames, bool) for _ in cams]
    last = [-10 ** 9] * len(cams)
    for t in range(frames):
        debts = []
        for k, c in enumerate(cams):
            d = c.policy.debt(t, c.ref["thumb"][t], c.tr, c.ref["fps"])
            if (t - last[k]) / c.ref["fps"] >= max_age:
                d = float("inf")
            debts.append(d)
        chosen = set(np.argsort(debts)[::-1][:per_tick].tolist())
        for k, c in enumerate(cams):
            c.step(t, k in chosen)
            if k in chosen:
                masks[k][t], last[k] = True, t
    return [(c.out, m) for c, m in zip(cams, masks)]


def score(reference, output, mask, fps, classes=None):
    """Fidelity of `output` to `reference` (both from run())."""
    tp = n_ref = n_out = 0
    per_cls = {}
    first_seen, first_hit = {}, {}
    life = {}
    for t, (r, o) in enumerate(zip(reference, output)):
        n_ref, n_out = n_ref + len(r), n_out + len(o)
        if not r:
            continue
        rb = np.array([b for _, b, _ in r])
        hit = np.zeros(len(r), bool)
        if o:
            m = iou(rb, np.array([b for _, b, _ in o]))
            # greedy one-to-one matching, highest IoU first
            for i, j in sorted(zip(*np.nonzero(m >= 0.5)), key=lambda ij: -m[ij]):
                if not hit[i] and m[i, j] >= 0.5 and not np.isnan(m[i, j]):
                    hit[i] = True
                    m[:, j] = np.nan
        tp += hit.sum()
        for (tid, _, c), h in zip(r, hit):
            first_seen.setdefault(tid, t)
            life[tid] = life.get(tid, 0) + 1
            if h:
                first_hit.setdefault(tid, t)
            if classes and c in classes:
                k = classes[c]
                a, b = per_cls.get(k, (0, 0))
                per_cls[k] = (a + h, b + 1)
    long = [tid for tid, n in life.items() if n >= fps]
    delays = np.array([first_hit[tid] - first_seen[tid] for tid in long if tid in first_hit], float)
    return {"compute": float(mask.mean()), "recall": tp / max(n_ref, 1), "precision": tp / max(n_out, 1),
            "delay_mean_s": float(delays.mean() / fps) if len(delays) else 0.0,
            "delay_p95_s": float(np.percentile(delays, 95) / fps) if len(delays) else 0.0,
            "missed": sum(tid not in first_hit for tid in long) / max(len(long), 1),
            **{f"recall_{k}": a / max(b, 1) for k, (a, b) in per_cls.items()}}
