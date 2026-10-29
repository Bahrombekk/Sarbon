"""Sarbon tracker: ByteTrack-style association with switchable improvements.

Baseline (all options off) reproduces the ByteTrack logic Ultralytics ships as its
default tracker, written from the paper (2110.06864), not from AGPL code:
  1) high-score detections vs tracked+lost tracks (IoU, score-fused);
  2) low-score detections vs still-unmatched tracked tracks (IoU);
  3) unconfirmed tracks vs leftover high detections; new tracks from the rest.
Options (each measurable on its own):
  emb   : appearance cost from the detector's decoder embeddings (BoT-SORT-style
          gated min of IoU and cosine costs, 2206.14651); EMA feature per track
  oru   : OC-SORT observation-centric re-update -- after a gap, re-run the Kalman
          filter along a virtual straight path so velocity is not corrupted (2203.14360)
  ocr   : OC-SORT observation-centric recovery -- last *observed* box vs leftover dets
  ocm   : OC-SORT momentum -- direction-consistency cost term
  nsa   : confidence-scaled measurement noise (NSA Kalman)
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .kalman import KalmanXYAH, xyah_to_xyxy, xyxy_to_xyah

TENTATIVE, TRACKED, LOST = 0, 1, 2


@dataclass
class TrackerConfig:
    high: float = 0.25  # Ultralytics bytetrack.yaml: track_high_thresh
    low: float = 0.1  # track_low_thresh
    new: float = 0.25  # new_track_thresh
    buffer: int = 30  # track_buffer (frames a lost track survives)
    match: float = 0.8  # match_thresh (max cost in stage 1)
    fuse_score: bool = True
    emb: bool = False
    emb_thresh: float = 0.25  # max cosine distance accepted from appearance
    iou_gate: float = 0.5  # appearance only for pairs with IoU cost below this (BoT-SORT proximity)
    long_gate: float = 0.85  # for long-lost tracks appearance may rescue weaker overlaps
    ema: float = 0.9
    oru: bool = False
    ocr: bool = False
    ocm: float = 0.0  # weight of the direction-consistency cost
    nsa: bool = False
    # long-term memory re-identification (needs embeddings): lost tracks are kept for
    # `reid_buffer` frames and matched by appearance + motion plausibility + size,
    # without any IoU requirement -- static occluders hide people longer than `buffer`
    reid: bool = False
    reid_buffer: int = 300
    reid_thresh: float = 0.35  # max cosine distance
    reid_speed_slack: float = 1.5  # allowed distance = slack * |v| * gap + reach * height
    reid_reach: float = 1.5


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])  # noqa: E731
    return inter / np.maximum(area(a)[:, None] + area(b)[None] - inter, 1e-9)


def assign(cost, thresh):
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    r, c = linear_sum_assignment(cost)
    keep = cost[r, c] <= thresh
    matches = list(zip(r[keep], c[keep]))
    mr, mc = {i for i, _ in matches}, {j for _, j in matches}
    return matches, [i for i in range(cost.shape[0]) if i not in mr], [j for j in range(cost.shape[1]) if j not in mc]


class Track:
    _next = 1

    def __init__(self, box, score, feat, frame, kf, nsa):
        self.kf, self.nsa = kf, nsa
        self.mean, self.cov = kf.initiate(xyxy_to_xyah(box))
        self.state, self.hits, self.frame = TENTATIVE, 1, frame
        self.last_box, self.last_frame = box.copy(), frame
        self.snapshot = (self.mean.copy(), self.cov.copy())
        self.obs = {frame: box.copy()}
        self.feat = None if feat is None else feat / (np.linalg.norm(feat) + 1e-9)
        self.score, self.id = score, 0

    def activate(self):
        if self.id == 0:
            self.id, Track._next = Track._next, Track._next + 1
        self.state = TRACKED

    def predict(self):
        if self.state != TRACKED:
            self.mean[7] = 0.0  # ByteTrack: freeze height velocity while not tracked
        self.mean, self.cov = self.kf.predict(self.mean, self.cov)

    @property
    def box(self):
        return xyah_to_xyxy(self.mean)

    def velocity(self, delta=10):
        """Mean centre velocity (px/frame) over the last ~delta observed frames."""
        frames = sorted(self.obs)
        if len(frames) < 2:
            return np.zeros(2)
        f0 = next((f for f in frames if self.last_frame - f <= delta), frames[0])
        if f0 == self.last_frame:
            f0 = frames[-2]
        a, b = self.obs[f0], self.obs[self.last_frame]
        return ((b[:2] + b[2:]) - (a[:2] + a[2:])) / 2 / (self.last_frame - f0)

    def velocity_dir(self, delta=3):
        """Unit direction of motion from observations ~delta frames apart (OC-SORT)."""
        frames = sorted(self.obs)
        if len(frames) < 2:
            return None
        f0 = next((f for f in reversed(frames[:-1]) if self.last_frame - f >= delta), frames[0])
        a, b = self.obs[f0], self.obs[self.last_frame]
        d = ((b[:2] + b[2:]) - (a[:2] + a[2:])) / 2
        n = np.linalg.norm(d)
        return d / n if n > 1e-6 else None

    def update(self, box, score, feat, frame, ema, oru):
        gap = frame - self.last_frame
        if oru and gap > 1:
            # observation-centric re-update: replay the gap along a straight virtual path
            self.mean, self.cov = (x.copy() for x in self.snapshot)
            for k in range(1, gap):
                self.mean, self.cov = self.kf.predict(self.mean, self.cov)
                virt = self.last_box + (box - self.last_box) * k / gap
                self.mean, self.cov = self.kf.update(self.mean, self.cov, xyxy_to_xyah(virt))
            self.mean, self.cov = self.kf.predict(self.mean, self.cov)
        self.mean, self.cov = self.kf.update(self.mean, self.cov, xyxy_to_xyah(box),
                                             score if self.nsa else None)
        self.snapshot = (self.mean.copy(), self.cov.copy())
        self.last_box, self.last_frame, self.score = box.copy(), frame, score
        self.obs[frame] = box.copy()
        if len(self.obs) > 30:
            self.obs.pop(min(self.obs))
        if feat is not None:
            f = feat / (np.linalg.norm(feat) + 1e-9)
            self.feat = f if self.feat is None else ema * self.feat + (1 - ema) * f
            self.feat /= np.linalg.norm(self.feat) + 1e-9
        self.hits += 1
        self.frame = frame


class SarbonTracker:
    def __init__(self, cfg: TrackerConfig = None):
        self.cfg = cfg or TrackerConfig()
        self.kf = KalmanXYAH()
        self.tracks, self.frame = [], 0

    def _cost(self, tracks, boxes, scores, feats, use_emb, use_last=False):
        tb = np.array([t.last_box if use_last else t.box for t in tracks]).reshape(-1, 4)
        iou = iou_matrix(tb, boxes)
        cost = 1 - (iou * scores[None] if self.cfg.fuse_score else iou)
        if self.cfg.ocm and len(tracks) and len(boxes):
            dc = np.zeros_like(cost)
            centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            for i, t in enumerate(tracks):
                v = t.velocity_dir()
                if v is None:
                    continue
                d = centers - (t.last_box[:2] + t.last_box[2:]) / 2
                n = np.linalg.norm(d, axis=1, keepdims=True)
                cos = (d / np.maximum(n, 1e-6)) @ v
                dc[i] = (1 - cos) / 2 * (n[:, 0] > 1e-6)
            cost = cost + self.cfg.ocm * dc
        if use_emb and feats is not None and len(tracks) and len(boxes):
            tf = np.array([t.feat for t in tracks])
            f = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)
            emb = 1 - tf @ f.T
            lost = np.array([t.state == LOST and self.frame - t.last_frame > 2 for t in tracks])
            gate = np.where(lost[:, None], self.cfg.long_gate, self.cfg.iou_gate)
            emb = np.where((emb > self.cfg.emb_thresh) | (1 - iou > gate), 1.0, emb)
            cost = np.minimum(cost, emb)
        return cost

    def _reid_cost(self, tracks, boxes, feats):
        """Appearance distance, gated by where the object could have walked and its size."""
        cfg = self.cfg
        f = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)
        cost = 1 - np.array([t.feat for t in tracks]) @ f.T
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        heights = boxes[:, 3] - boxes[:, 1]
        for i, t in enumerate(tracks):
            gap = self.frame - t.last_frame
            h = t.last_box[3] - t.last_box[1]
            v = t.velocity()
            expect = (t.last_box[:2] + t.last_box[2:]) / 2 + v * gap
            reach = cfg.reid_speed_slack * np.linalg.norm(v) * gap + cfg.reid_reach * h
            far = np.linalg.norm(centers - expect, axis=1) > reach
            ratio = heights / max(h, 1e-6)
            cost[i, far | (ratio < 0.67) | (ratio > 1.5)] = 1.0
        return cost

    def update(self, boxes, scores, feats=None):
        """boxes [N, 4] xyxy, scores [N], feats [N, D] or None. Returns [(id, box, score)]."""
        cfg = self.cfg
        self.frame += 1
        boxes, scores = np.asarray(boxes, float).reshape(-1, 4), np.asarray(scores, float)
        feats = None if feats is None else np.asarray(feats, float)
        use_emb = cfg.emb and feats is not None
        for t in self.tracks:
            t.predict()
        hi = np.where(scores >= cfg.high)[0]
        lo = np.where((scores >= cfg.low) & (scores < cfg.high))[0]
        confirmed = [t for t in self.tracks if t.state != TENTATIVE]
        tentative = [t for t in self.tracks if t.state == TENTATIVE]
        sub = lambda idx: (boxes[idx], scores[idx], None if feats is None else feats[idx])  # noqa: E731

        def apply(ts, idx, matches):
            b, s, f = sub(idx)
            for i, j in matches:
                ts[i].update(b[j], s[j], None if f is None else f[j], self.frame, cfg.ema, cfg.oru)
                ts[i].activate()

        # stage 1: high-score detections vs tracked + lost tracks
        m1, ut, ud = assign(self._cost(confirmed, *sub(hi), use_emb), cfg.match)
        apply(confirmed, hi, m1)
        rem_hi = hi[ud]
        # stage 2: low-score detections vs tracks that were tracked last frame (IoU only)
        cand = [confirmed[i] for i in ut if confirmed[i].state == TRACKED]
        rest = [confirmed[i] for i in ut if confirmed[i].state != TRACKED]
        b, s, _ = sub(lo)
        m2, ut2, _ = assign(1 - iou_matrix(np.array([t.box for t in cand]).reshape(-1, 4), b), 0.5)
        apply(cand, lo, m2)
        unmatched = [cand[i] for i in ut2] + rest
        # OC-SORT recovery: last observed box vs leftover high detections
        if cfg.ocr and len(unmatched) and len(rem_hi):
            m3, ut3, ud3 = assign(self._cost(unmatched, *sub(rem_hi), False, use_last=True), 0.5)
            apply(unmatched, rem_hi, m3)
            unmatched = [unmatched[i] for i in ut3]
            rem_hi = rem_hi[ud3]
        for t in unmatched:
            t.state = LOST
        # long-term memory: re-identify tracks lost for a while, by appearance + motion
        if cfg.reid and use_emb and len(rem_hi):
            mem = [t for t in self.tracks if t.state == LOST and t.feat is not None
                   and self.frame - t.last_frame > 1]
            if mem:
                b, s, f = sub(rem_hi)
                m5, _, ud5 = assign(self._reid_cost(mem, b, f), cfg.reid_thresh)
                apply(mem, rem_hi, m5)
                rem_hi = rem_hi[ud5]
        # stage 3: tentative tracks vs leftover high detections
        m4, ut4, ud4 = assign(self._cost(tentative, *sub(rem_hi), use_emb), 0.7)
        apply(tentative, rem_hi, m4)
        dead = {id(tentative[i]) for i in ut4}
        rem_hi = rem_hi[ud4]
        # births
        for j in rem_hi:
            if scores[j] >= cfg.new:
                t = Track(boxes[j], scores[j], None if feats is None else feats[j], self.frame, self.kf, cfg.nsa)
                if self.frame == 1:
                    t.activate()
                self.tracks.append(t)
        keep_for = cfg.reid_buffer if (cfg.reid and use_emb) else cfg.buffer
        self.tracks = [t for t in self.tracks if id(t) not in dead
                       and not (t.state == LOST and self.frame - t.last_frame > keep_for)]
        return [(t.id, t.last_box.copy(), t.score) for t in self.tracks
                if t.state == TRACKED and t.last_frame == self.frame]

    def predictions(self):
        """Predicted boxes of live tracks for the next frame (feeds the compute gate)."""
        return [(t.id, t.box) for t in self.tracks if t.state != TENTATIVE]
