"""SarbonVideo: multi-camera detection + tracking with change-proportional compute.

Per step (one frame from each camera):
  1. stem   : on GPU the block-sparse engine recomputes only the blocks that changed
              (plus, with tau > 0, the blocks where tracks are predicted to be);
              on CPU the stem runs densely (same numbers, for tests/debugging)
  2. detector: dense core + decoder on the cached stride-8 features
  3. tracker : one SarbonTracker per camera, fed with boxes, scores, decoder
              embeddings and colour histograms
  4. steering: tracks' predicted boxes become next step's must-recompute blocks
"""
import cv2
import numpy as np
import torch

from .models.box_ops import cxcywh_to_xyxy
from .sparse.convert import stem_to_local
from .track import SarbonTracker, TrackerConfig
from .track.appearance import color_descriptor

MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255
STD = np.array([0.229, 0.224, 0.225], np.float32) * 255


def boxes_to_block_mask(boxes, hb, wb, block, margin=0):
    """Blocks [hb, wb] touched by xyxy boxes (engine pixels), dilated by `margin` blocks."""
    m = np.zeros((hb, wb), bool)
    for x0, y0, x1, y1 in boxes:
        c0, r0 = max(0, int(x0 // block) - margin), max(0, int(y0 // block) - margin)
        c1, r1 = min(wb - 1, int(x1 // block) + margin), min(hb - 1, int(y1 // block) + margin)
        if c0 <= c1 and r0 <= r1:
            m[r0:r1 + 1, c0:c1 + 1] = True
    return m


class SarbonVideo:
    def __init__(self, detector, n_cams, size=(640, 1152), tracker_cfg: TrackerConfig = None,
                 tau=0.0, keep_score=0.1, max_dets=100, use_color=True, track_steer=True,
                 class_names=None, device="cuda"):
        self.dev = torch.device(device)
        self.det = detector.eval().to(self.dev)
        self.n, (self.h, self.w) = n_cams, size
        assert self.h % 32 == 0 and self.w % 32 == 0
        self.keep_score, self.max_dets = keep_score, max_dets
        self.use_color, self.track_steer, self.tau = use_color, track_steer, tau
        self.class_names = class_names
        self.tracker_cfg = tracker_cfg or TrackerConfig(emb=True, color=0.7 if use_color else 0.0,
                                                        reid=True, ground=True)
        self.trackers = None  # created on the first frame (needs the camera resolution)
        self.engine = None
        if self.dev.type == "cuda":
            from .sparse.fused_engine import FusedSparseEngine
            local = stem_to_local(self.det.stem).to(self.dev)
            self.engine = FusedSparseEngine(local, n_cams, self.h, self.w, dtype=torch.float16, tau=tau)
        self.started = False
        self.stats = {"frames": 0, "active_blocks": []}

    # ---------------- helpers ----------------
    def _prep(self, frames):
        """BGR uint8 frames -> normalised NHWC float tensor at engine size."""
        x = np.stack([cv2.resize(f, (self.w, self.h), interpolation=cv2.INTER_LINEAR)[:, :, ::-1]
                      for f in frames]).astype(np.float32)
        return torch.from_numpy((x - MEAN) / STD).to(self.dev)

    @torch.no_grad()
    def _stem(self, x):
        if self.engine is None:
            return self.det.stem.forward_blocks(x.permute(0, 3, 1, 2).contiguous())[1]
        x = x.half()
        if not self.started:
            self.engine.reset(x)
            k = self.engine.n_blocks
        else:
            _, k = self.engine.step(x)
        self.stats["active_blocks"].append(k / self.engine.n_blocks if k >= 0 else 1.0)
        return self.engine.outputs_nchw()[1]

    def _steer(self, scale):
        if self.engine is None or not (self.track_steer and self.tau > 0):
            return
        b = self.engine.block
        masks = []
        for tr in self.trackers:
            boxes = [box / np.r_[scale, scale] for _, box in tr.predictions()]
            masks.append(boxes_to_block_mask(boxes, self.engine.hb, self.engine.wb, b, margin=1))
        self.engine.set_extra(torch.from_numpy(np.stack(masks)).to(self.dev))

    # ---------------- main ----------------
    @torch.no_grad()
    def step(self, frames):
        """frames: list of N BGR uint8 images (same resolution). Returns per camera a list of
        {"id", "box" (xyxy, frame pixels), "score", "label", "ground" ([X, Z] metres or None)}."""
        assert len(frames) == self.n
        h0, w0 = frames[0].shape[:2]
        if self.trackers is None:
            self.trackers = [SarbonTracker(self.tracker_cfg, (w0, h0)) for _ in range(self.n)]
        scale = np.array([w0 / self.w, h0 / self.h])
        p8 = self._stem(self._prep(frames))
        self.started = True
        with torch.autocast(self.dev.type, torch.float16, enabled=self.dev.type == "cuda"):
            out = self.det(None, p8=p8)
        scores_all = out["pred_logits"].float().sigmoid()
        boxes_all = cxcywh_to_xyxy(out["pred_boxes"].float())
        results = []
        for i in range(self.n):
            s, lab = scores_all[i].max(-1)
            keep = (s >= self.keep_score).nonzero()[:, 0]
            keep = keep[s[keep].argsort(descending=True)][: self.max_dets]
            boxes = (boxes_all[i, keep] * torch.tensor([self.w, self.h, self.w, self.h], device=self.dev)).cpu().numpy()
            boxes = boxes * np.r_[scale, scale]
            sc, lb = s[keep].cpu().numpy(), lab[keep].cpu().numpy()
            emb = out.get("id_embed", out["embed"])[i, keep].float().cpu().numpy()
            col = np.stack([color_descriptor(frames[i], b) for b in boxes]) if self.use_color and len(boxes) else None
            tracks = self.trackers[i].update(boxes, sc, emb, col)
            res = []
            for tid, box, score, ground in tracks:
                j = int(np.argmin(np.abs(boxes - box).sum(1))) if len(boxes) else -1
                label = int(lb[j]) if j >= 0 else -1
                res.append({"id": tid, "box": box, "score": float(score),
                            "label": self.class_names[label] if self.class_names and label >= 0 else label,
                            "ground": None if ground is None else ground.tolist()})
            results.append(res)
        self._steer(scale)
        self.stats["frames"] += 1
        return results
