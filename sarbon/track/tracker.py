"""Sarbon tracker: ByteTrack-style association with switchable improvements.

Baseline (all options off) reproduces the ByteTrack logic Ultralytics ships as its
default tracker, written from the paper (2110.06864), not from AGPL code:
  1) high-score detections vs tracked+lost tracks (IoU, score-fused);
  2) low-score detections vs still-unmatched tracked tracks (IoU);
  3) unconfirmed tracks vs leftover high detections; new tracks from the rest.
Options (each measurable on its own):
  emb    : appearance cost from the detector's decoder embeddings (BoT-SORT-style
           gated min of IoU and cosine costs, 2206.14651); EMA feature per track
  color  : colour histograms (upper/lower body) mixed into the appearance distance
  reid   : long-term memory -- lost tracks are kept for `reid_buffer` frames and
           re-identified by appearance + motion plausibility + size, without IoU
  ground : self-calibrating ground plane (ground.py): constant-velocity Kalman in
           metres, Mahalanobis gating with camera-propagated noise (UCMCTrack-style),
           re-ID extrapolation on the ground, per-track distance from the camera
  size_prior : with the ground plane, refuse births whose height cannot be a person
           standing at that foot row (static-camera false-positive filter)
  oru / ocr / ocm : OC-SORT re-update, recovery and momentum (2203.14360)
  nsa    : confidence-scaled measurement noise (NSA Kalman)
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .appearance import bhattacharyya
from .ground import GroundKalman, GroundModel
from .kalman import KalmanXYAH, xyah_to_xyxy, xyxy_to_xyah

TENTATIVE, TRACKED, LOST = 0, 1, 2
CHI2_2DOF_99 = 9.21


@dataclass
class TrackerConfig:
    high: float = 0.25  # Ultralytics bytetrack.yaml: track_high_thresh
    low: float = 0.1  # track_low_thresh
    new: float = 0.25  # new_track_thresh
    buffer: int = 30  # track_buffer (frames a lost track survives)
    match: float = 0.8  # match_thresh (max cost in stage 1)
    fuse_score: bool = True
    emb: bool = False
    emb_thresh: float = 0.25  # max appearance distance accepted in the short-term stages
    iou_gate: float = 0.5  # appearance only for pairs with IoU cost below this (BoT-SORT proximity)
    long_gate: float = 0.85
    ema: float = 0.9
    color: float = 0.0  # weight of the colour distance inside the appearance distance
    oru: bool = False
    ocr: bool = False
    ocm: float = 0.0
    nsa: bool = False
    reid: bool = False
    reid_buffer: int = 300
    reid_thresh: float = 0.35
    reid_speed_slack: float = 1.5
    reid_reach: float = 1.5  # in box heights (image mode)
    ground: bool = False
    ground_reach: float = 1.0  # metres of slack for re-ID on the ground
    size_prior: bool = False


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


def _unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


class Track:
    _next = 1

    def __init__(self, box, score, feat, color, frame, kf, nsa):
        self.kf, self.nsa = kf, nsa
        self.mean, self.cov = kf.initiate(xyxy_to_xyah(box))
        self.state, self.hits, self.frame = TENTATIVE, 1, frame
        self.last_box, self.last_frame = box.copy(), frame
        self.snapshot = (self.mean.copy(), self.cov.copy())
        self.obs = {frame: box.copy()}
        self.feat = None if feat is None else _unit(feat)
        self.color = None if color is None else color.copy()
        self.score, self.id = score, 0
        self.g = None  # ground state (mean, cov) once the ground plane is calibrated
        self.g_last = None  # last observed ground position
        self.size_norm = None  # image height / expected person height at that row (EMA)

    def activate(self):
        if self.id == 0:
            self.id, Track._next = Track._next, Track._next + 1
        self.state = TRACKED

    def predict(self, gkf):
        if self.state != TRACKED:
            self.mean[7] = 0.0  # ByteTrack: freeze height velocity while not tracked
        self.mean, self.cov = self.kf.predict(self.mean, self.cov)
        if self.g is not None:
            self.g = gkf.predict(*self.g)

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
        frames = sorted(self.obs)
        if len(frames) < 2:
            return None
        f0 = next((f for f in reversed(frames[:-1]) if self.last_frame - f >= delta), frames[0])
        a, b = self.obs[f0], self.obs[self.last_frame]
        d = ((b[:2] + b[2:]) - (a[:2] + a[2:])) / 2
        n = np.linalg.norm(d)
        return d / n if n > 1e-6 else None

    def update(self, box, score, feat, color, frame, cfg, ground=None, gkf=None, size_norm=None):
        gap = frame - self.last_frame
        if cfg.oru and gap > 1:
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
        if ground is not None:
            z, r = ground
            self.g = gkf.initiate(z, r) if self.g is None else gkf.update(*self.g, z, r)
            self.g_last = z.copy()
        if size_norm is not None:
            self.size_norm = size_norm if self.size_norm is None else 0.9 * self.size_norm + 0.1 * size_norm
        self.last_box, self.last_frame, self.score = box.copy(), frame, score
        self.obs[frame] = box.copy()
        if len(self.obs) > 30:
            self.obs.pop(min(self.obs))
        if feat is not None:
            f = _unit(feat)
            self.feat = f if self.feat is None else _unit(cfg.ema * self.feat + (1 - cfg.ema) * f)
        if color is not None:
            self.color = color.copy() if self.color is None else cfg.ema * self.color + (1 - cfg.ema) * color
        self.hits += 1
        self.frame = frame


class SarbonTracker:
    def __init__(self, cfg: TrackerConfig = None, img_size=(1920, 1080)):
        self.cfg = cfg or TrackerConfig()
        self.kf, self.gkf = KalmanXYAH(), GroundKalman()
        self.gm = GroundModel(*img_size)
        self.tracks, self.frame = [], 0

    # ---------------- per-frame detection geometry ----------------
    def _ground(self, boxes):
        """Ground position, measurement covariance and normalised size per detection."""
        if not (self.cfg.ground and self.gm.ready) or len(boxes) == 0:
            return None
        out = []
        for b in boxes:
            x, y = self.gm.foot(b)
            w, h = b[2] - b[0], b[3] - b[1]
            j = self.gm.jacobian(x, y)
            r = j @ np.diag([(0.05 * w + 1) ** 2, (0.03 * h + 1) ** 2]) @ j.T
            out.append((self.gm.to_ground(x, y), r, h / max(self.gm.expected_height(y), 1e-6)))
        return out

    # ---------------- costs ----------------
    def _appearance(self, tracks, feats, colors):
        """Appearance distance in [0, 1]: embedding cosine, optionally mixed with colour."""
        d = None
        if feats is not None and all(t.feat is not None for t in tracks):
            d = 1 - np.array([t.feat for t in tracks]) @ _unit(feats).T
        if self.cfg.color and colors is not None and all(t.color is not None for t in tracks):
            c = bhattacharyya(np.array([t.color for t in tracks]), colors)
            d = c if d is None else (1 - self.cfg.color) * d + self.cfg.color * c
        return d

    def _ground_cost(self, tracks, geo):
        """Squared Mahalanobis on the ground / chi2 gate; inf where a track has no ground state."""
        g = np.full((len(tracks), len(geo)), np.inf)
        zs = np.array([x[0] for x in geo])
        rs = np.array([x[1] for x in geo])
        for i, t in enumerate(tracks):
            if t.g is not None:
                g[i] = self.gkf.mahalanobis(*t.g, zs, rs) / CHI2_2DOF_99
        return g

    def _cost(self, tracks, boxes, scores, feats, colors, geo, use_app, use_last=False):
        tb = np.array([t.last_box if use_last else t.box for t in tracks]).reshape(-1, 4)
        iou = iou_matrix(tb, boxes)
        cost = 1 - (iou * scores[None] if self.cfg.fuse_score else iou)
        if not len(tracks) or not len(boxes):
            return cost
        if geo is not None and not use_last:
            g = self._ground_cost(tracks, geo)
            has = np.isfinite(g).any(1)
            # the ground plane gates and refines, it does not replace the image overlap:
            # impossible on the ground = no match however much boxes overlap, and among
            # plausible pairs a small Mahalanobis penalty breaks ties. Letting the ground
            # cost alone match IoU-0 pairs attached far-range false positives (depth noise
            # far away is metres), so IoU stays the base cost.
            ground_cost = np.where(g > 1, 1.0, cost + 0.2 * np.minimum(g, 1.0))
            cost = np.where(has[:, None], ground_cost, cost)
        if self.cfg.ocm:
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
        if use_app:
            app = self._appearance(tracks, feats, colors)
            if app is not None:
                lost = np.array([t.state == LOST and self.frame - t.last_frame > 2 for t in tracks])
                gate = np.where(lost[:, None], self.cfg.long_gate, self.cfg.iou_gate)
                app = np.where((app > self.cfg.emb_thresh) | (1 - iou > gate), 1.0, app)
                cost = np.minimum(cost, app)
        return cost

    def _reid_cost(self, tracks, boxes, feats, colors, geo):
        """Appearance distance, gated by where the object could have gone and its size."""
        cfg = self.cfg
        cost = self._appearance(tracks, feats, colors)
        if cost is None:
            return np.ones((len(tracks), len(boxes)))
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        heights = boxes[:, 3] - boxes[:, 1]
        for i, t in enumerate(tracks):
            gap = self.frame - t.last_frame
            if geo is not None and t.g is not None:
                # on the ground: walkers keep a constant speed in metres behind occluders
                zs = np.array([x[0] for x in geo])
                expect = t.g[0][:2]
                reach = cfg.reid_speed_slack * np.linalg.norm(t.g[0][2:]) * gap + cfg.ground_reach
                far = np.linalg.norm(zs - expect, axis=1) > reach
                sn = np.array([x[2] for x in geo])
                ratio = sn / max(t.size_norm or 1.0, 1e-6)  # same real size wherever it stands
            else:
                h = t.last_box[3] - t.last_box[1]
                expect = (t.last_box[:2] + t.last_box[2:]) / 2 + t.velocity() * gap
                reach = cfg.reid_speed_slack * np.linalg.norm(t.velocity()) * gap + cfg.reid_reach * h
                far = np.linalg.norm(centers - expect, axis=1) > reach
                ratio = heights / max(h, 1e-6)
            cost[i, far | (ratio < 0.67) | (ratio > 1.5)] = 1.0
        return cost

    # ---------------- main loop ----------------
    def update(self, boxes, scores, feats=None, colors=None):
        """boxes [N, 4] xyxy, scores [N], feats [N, D], colors [N, K] (optional).

        Returns [(id, box, score, ground_xz or None)] for tracks updated this frame;
        ground_xz is the position on the ground in metres (Z = distance from camera).
        """
        cfg = self.cfg
        self.frame += 1
        boxes, scores = np.asarray(boxes, float).reshape(-1, 4), np.asarray(scores, float)
        feats = None if (feats is None or not cfg.emb) else np.asarray(feats, float)
        colors = None if (colors is None or not cfg.color) else np.asarray(colors, float)
        use_app = feats is not None or colors is not None
        geo = self._ground(boxes)
        for t in self.tracks:
            t.predict(self.gkf)
        hi = np.where(scores >= cfg.high)[0]
        lo = np.where((scores >= cfg.low) & (scores < cfg.high))[0]
        confirmed = [t for t in self.tracks if t.state != TENTATIVE]
        tentative = [t for t in self.tracks if t.state == TENTATIVE]

        def sub(idx):
            return (boxes[idx], scores[idx], None if feats is None else feats[idx],
                    None if colors is None else colors[idx], None if geo is None else [geo[k] for k in idx])

        def apply(ts, idx, matches):
            b, s, f, c, g = sub(idx)
            for i, j in matches:
                gm = None if g is None else (g[j][0], g[j][1])
                ts[i].update(b[j], s[j], None if f is None else f[j], None if c is None else c[j],
                             self.frame, cfg, gm, self.gkf, None if g is None else g[j][2])
                ts[i].activate()

        # stage 1: high-score detections vs tracked + lost tracks
        m1, ut, ud = assign(self._cost(confirmed, *sub(hi), use_app), cfg.match)
        apply(confirmed, hi, m1)
        rem_hi = hi[ud]
        # stage 2: low-score detections vs tracks that were tracked last frame (IoU only)
        cand = [confirmed[i] for i in ut if confirmed[i].state == TRACKED]
        rest = [confirmed[i] for i in ut if confirmed[i].state != TRACKED]
        b = boxes[lo]
        m2, ut2, _ = assign(1 - iou_matrix(np.array([t.box for t in cand]).reshape(-1, 4), b), 0.5)
        apply(cand, lo, m2)
        unmatched = [cand[i] for i in ut2] + rest
        if cfg.ocr and len(unmatched) and len(rem_hi):
            m3, ut3, ud3 = assign(self._cost(unmatched, *sub(rem_hi), False, use_last=True), 0.5)
            apply(unmatched, rem_hi, m3)
            unmatched = [unmatched[i] for i in ut3]
            rem_hi = rem_hi[ud3]
        for t in unmatched:
            t.state = LOST
        # long-term memory re-identification
        if cfg.reid and use_app and len(rem_hi):
            mem = [t for t in self.tracks if t.state == LOST and self.frame - t.last_frame > 1
                   and (t.feat is not None or t.color is not None)]
            if mem:
                _, _, f, c, g = sub(rem_hi)
                m5, _, ud5 = assign(self._reid_cost(mem, boxes[rem_hi], f, c, g), cfg.reid_thresh)
                apply(mem, rem_hi, m5)
                rem_hi = rem_hi[ud5]
        # stage 3: tentative tracks vs leftover high detections
        m4, ut4, ud4 = assign(self._cost(tentative, *sub(rem_hi), use_app), 0.7)
        apply(tentative, rem_hi, m4)
        dead = {id(tentative[i]) for i in ut4}
        rem_hi = rem_hi[ud4]
        # births (optionally rejected by the per-row size prior)
        for j in rem_hi:
            if scores[j] < cfg.new:
                continue
            if cfg.size_prior and geo is not None and not 0.5 < geo[j][2] < 2.0:
                continue
            t = Track(boxes[j], scores[j], None if feats is None else feats[j],
                      None if colors is None else colors[j], self.frame, self.kf, cfg.nsa)
            if geo is not None:
                t.update(boxes[j], scores[j], None, None, self.frame, cfg, (geo[j][0], geo[j][1]),
                         self.gkf, geo[j][2])
                t.hits = 1
            if self.frame == 1:
                t.activate()
            self.tracks.append(t)
        keep_for = cfg.reid_buffer if (cfg.reid and use_app) else cfg.buffer
        self.tracks = [t for t in self.tracks if id(t) not in dead
                       and not (t.state == LOST and self.frame - t.last_frame > keep_for)]
        self._calibrate(boxes, scores)
        return [(t.id, t.last_box.copy(), t.score, None if t.g_last is None else t.g_last.copy())
                for t in self.tracks if t.state == TRACKED and t.last_frame == self.frame]

    def _calibrate(self, boxes, scores):
        """Feed confident, unoccluded, well-established tracks to the ground model."""
        if not self.cfg.ground or (self.gm.ready and self.frame % 500):
            return
        live = [t for t in self.tracks if t.state == TRACKED and t.last_frame == self.frame
                and t.hits >= 5 and t.score >= 0.6]
        if live:
            others = iou_matrix(np.array([t.last_box for t in live]), boxes)
            for t, row in zip(live, others):
                if np.sort(row)[-2:][0] < 0.02 if len(row) > 1 else True:  # nothing else overlaps
                    self.gm.observe(t.last_box)
        if self.frame % 50 == 0:
            self.gm.fit()

    def predictions(self):
        """Predicted boxes of live tracks for the next frame (feeds the compute gate)."""
        return [(t.id, t.box) for t in self.tracks if t.state != TENTATIVE]
