"""Free labels from fixed cameras: background subtraction -> blobs -> tracks -> MOT dataset.

A static camera sees the same background for hours, so anything that moves stands out
against a learned background model. Moving blobs are tracked; only tracks that live
long enough and actually travel are kept (waving trees, flicker and noise do not), and
each kept track becomes a pseudo-labelled object with a consistent identity. The
result is written in the MOTChallenge layout, so `MOTClips` / `train_track.py` can
pretrain objectness + re-identification on our own unlabelled footage, no external
weights or human labels involved.

Known limits (what the detector must still learn from labelled data): objects that
stop moving fade into the background; people walking together merge into one blob;
camera shake or sudden light changes produce whole-frame foreground (such frames are
skipped).
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..track import SarbonTracker, TrackerConfig
from ..track.tracker import iou_matrix


@dataclass
class PseudoConfig:
    history: int = 500  # MOG2 background memory (frames)
    var_threshold: float = 16.0
    warmup: int = 100  # frames used only to learn the background
    min_area: float = 0.0004  # blob area / frame area
    max_area: float = 0.25
    max_fg: float = 0.4  # skip frames whose foreground exceeds this (shake, light switch)
    min_len: int = 15  # frames a track must live
    min_move: float = 1.0  # total centre travel, in units of the box height
    close: int = 7  # morphological closing kernel (joins body parts)
    stride: int = 3  # write every `stride`-th frame
    interp_gap: int = 10  # fill gaps of kept tracks up to this many frames (merged / missed blobs)
    paint: bool = True  # paint unlabelled moving blobs with the background model (no false negatives)
    paint_margin: int = 4  # px around a painted blob


def blobs(fg, cfg: PseudoConfig, frame_area):
    """Foreground mask (MOG2: 255 = fg, 127 = shadow) -> xyxy boxes and fill ratios."""
    m = (fg == 255).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((cfg.close, cfg.close), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    boxes, fill = [], []
    for x, y, w, h, area in stats[1:]:
        if cfg.min_area * frame_area <= w * h <= cfg.max_area * frame_area:
            boxes.append([x, y, x + w, y + h])
            fill.append(area / (w * h))
    return np.array(boxes, float).reshape(-1, 4), np.array(fill)


class PseudoLabels(dict):
    """{frame_index: [(track_id, xyxy), ...]} plus what painting needs:
    `unlabelled[i]` -- moving blobs not covered by a kept track; `backgrounds[i]` -- the
    background model image (JPEG bytes) at frame i, for frames that may be written."""

    def __init__(self, labels, unlabelled, backgrounds):
        super().__init__(labels)
        self.unlabelled, self.backgrounds = unlabelled, backgrounds


def _interpolate(obs, max_gap):
    """Fill short gaps of one track [(frame, box)] linearly."""
    out = list(obs)
    for (f0, b0), (f1, b1) in zip(obs, obs[1:]):
        if 1 < f1 - f0 <= max_gap:
            for f in range(f0 + 1, f1):
                a = (f - f0) / (f1 - f0)
                out.append((f, b0 + (b1 - b0) * a))
    return sorted(out, key=lambda x: x[0])


def pseudo_label(frames, cfg: PseudoConfig = None):
    """frames: iterable of BGR images -> PseudoLabels of the kept tracks."""
    cfg = cfg or PseudoConfig()
    bg = cv2.createBackgroundSubtractorMOG2(cfg.history, cfg.var_threshold, detectShadows=True)
    tracker = None
    per_frame, raw, backgrounds = {}, {}, {}
    for i, frame in enumerate(frames):
        fg = bg.apply(frame)
        if i < cfg.warmup:
            continue
        h, w = frame.shape[:2]
        if tracker is None:
            tracker = SarbonTracker(TrackerConfig(), (w, h))
        if (fg == 255).mean() > cfg.max_fg:
            continue
        b, fill = blobs(fg, cfg, h * w)
        # a compact blob is more object-like than a sparse one: use the fill ratio as score
        out = tracker.update(b, np.clip(0.3 + fill, 0, 0.99))
        per_frame[i] = [(tid, box) for tid, box, _, _ in out]
        raw[i] = b
        if cfg.paint and i % cfg.stride == 0:
            backgrounds[i] = cv2.imencode(".png", bg.getBackgroundImage())[1].tobytes()
    # keep tracks that live long enough and travel (rejects swaying trees, flicker)
    tracks = {}
    for i, objs in per_frame.items():
        for tid, box in objs:
            tracks.setdefault(tid, []).append((i, box))
    labels = {}
    for tid, obs in tracks.items():
        if len(obs) < cfg.min_len:
            continue
        c = np.array([(b[:2] + b[2:]) / 2 for _, b in obs])
        hgt = np.median([b[3] - b[1] for _, b in obs])
        if np.linalg.norm(c[-1] - c[0]) < cfg.min_move * hgt:
            continue
        for f, box in _interpolate(obs, cfg.interp_gap):
            labels.setdefault(f, []).append((tid, box))
    # moving blobs that no kept label explains: painted over before training
    unlabelled = {}
    for i, b in raw.items():
        kept = np.array([box for _, box in labels.get(i, [])]).reshape(-1, 4)
        if len(b):
            cover = iou_matrix(b, kept).max(1) if len(kept) else np.zeros(len(b))
            unlabelled[i] = b[cover < 0.3]
    return PseudoLabels(labels, unlabelled, backgrounds)


def paint(frame, boxes, background, labelled, margin):
    """Replace unlabelled moving regions with the background model, sparing labelled boxes."""
    out = frame.copy()
    h, w = frame.shape[:2]
    keep = np.zeros((h, w), bool)
    for x0, y0, x1, y1 in labelled:
        keep[max(0, int(y0)):int(y1) + 1, max(0, int(x0)):int(x1) + 1] = True
    for x0, y0, x1, y1 in boxes:
        ya, yb = max(0, int(y0) - margin), min(h, int(y1) + margin + 1)
        xa, xb = max(0, int(x0) - margin), min(w, int(x1) + margin + 1)
        region = ~keep[ya:yb, xa:xb]
        out[ya:yb, xa:xb][region] = background[ya:yb, xa:xb][region]
    return out


def write_mot(frames, labels, out_dir, cfg: PseudoConfig = None):
    """Write kept frames (every cfg.stride-th with objects) as img1/*.jpg + gt/gt.txt."""
    cfg = cfg or PseudoConfig()
    out = Path(out_dir)
    (out / "img1").mkdir(parents=True, exist_ok=True)
    (out / "gt").mkdir(parents=True, exist_ok=True)
    rows, n = [], 0
    for i, frame in enumerate(frames):
        objs = labels.get(i)
        if not objs or i % cfg.stride:
            continue
        n += 1
        un = getattr(labels, "unlabelled", {}).get(i)
        bgb = getattr(labels, "backgrounds", {}).get(i)
        if cfg.paint and un is not None and len(un) and bgb is not None:
            bgi = cv2.imdecode(np.frombuffer(bgb, np.uint8), cv2.IMREAD_COLOR)
            frame = paint(frame, un, bgi, [b for _, b in objs], cfg.paint_margin)
        cv2.imwrite(str(out / "img1" / f"{n:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        for tid, (x0, y0, x1, y1) in objs:
            rows.append(f"{n},{tid},{x0:.1f},{y0:.1f},{x1 - x0:.1f},{y1 - y0:.1f},1,1,1")
    (out / "gt" / "gt.txt").write_text("\n".join(rows) + "\n")
    return n, len(rows)


def video_frames(path, max_frames=None, resize_w=None):
    """Read a video file or stream (cv2.VideoCapture source) as BGR frames."""
    cap = cv2.VideoCapture(str(path))
    i = 0
    while max_frames is None or i < max_frames:
        ok, f = cap.read()
        if not ok:
            break
        if resize_w and f.shape[1] != resize_w:
            f = cv2.resize(f, (resize_w, round(f.shape[0] * resize_w / f.shape[1])))
        yield f
        i += 1
    cap.release()
