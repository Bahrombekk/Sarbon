"""When to run the detector on a camera frame: per-camera compute policies.

Every policy sees only what is cheap on a skipped frame -- a 96x54 grayscale thumbnail and the
tracker's own state -- and answers `should_compute`. After a computed frame `computed` is called
with that frame's thumbnail.

  EveryFrame      the usual pipeline (reference)
  FixedStride     every N-th frame
  MotionGate      thumbnail change since the last computed frame above a threshold
  PerceptionDebt  how wrong the current picture probably is, from several sources (below)

Perception debt of a camera at frame t (each term divided by its budget; compute when the
largest reaches 1):
  local change   strongest cell change since the last computed frame, after removing the
                 frame-wide change (lighting): something appeared, left or moved there
  global change  frame-wide change (camera moved, light switched, scene cut): cache is stale
  motion         the live track expected to have moved furthest since its last measurement,
                 in units of its own size (its predicted box is drifting from the truth)
  edge           a live track predicted to touch the border: it may be leaving and new ones
                 enter there; recompute sooner
  unconfirmed    seconds an unconfirmed (just born) track has waited: a new object is only
                 reported once a second detection confirms it, so it is the most urgent debt
  age            seconds since the last computed frame (a hard staleness bound)
Budgets scale together with `scale`: sweeping it traces the accuracy / compute curve.
"""
import numpy as np

CELLS = (6, 12)  # rows, cols of the change grid on the 54x96 thumbnail (9x8 px cells)


def cell_change(a, b):
    """Global and strongest local mean |a - b| (0..255) of two thumbnails."""
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).astype(np.float32)
    g = float(d.mean())
    h, w = d.shape
    r, c = CELLS
    cells = d[: h // r * r, : w // c * c].reshape(r, h // r, c, w // c).mean((1, 3))
    return g, float(cells.max() - np.median(cells))


class EveryFrame:
    name = "every frame"

    def should_compute(self, t, thumb, tracker, fps):
        return True

    def computed(self, t, thumb):
        pass


class FixedStride:
    def __init__(self, n):
        self.n, self.name = n, f"every {n}"

    def should_compute(self, t, thumb, tracker, fps):
        return t % self.n == 0

    def computed(self, t, thumb):
        pass


class MotionGate:
    """Classic change gating: compute when the frame differs enough from the last computed one."""

    def __init__(self, thresh, max_age=2.0):
        self.thresh, self.max_age = thresh, max_age
        self.name = f"motion>{thresh:g}"
        self.last, self.last_t = None, -10 ** 9

    def should_compute(self, t, thumb, tracker, fps):
        if self.last is None or (t - self.last_t) / fps >= self.max_age:
            return True
        return float(np.abs(thumb.astype(np.int16) - self.last).mean()) > self.thresh

    def computed(self, t, thumb):
        self.last, self.last_t = thumb.astype(np.int16), t


class RoundRobin:
    """Shared-budget baseline: the debt is just the time since the last compute."""

    name = "round robin"

    def __init__(self):
        self.last_t = -10 ** 9

    def debt(self, t, thumb, tracker, fps):
        return (t - self.last_t) / fps

    def computed(self, t, thumb):
        self.last_t = t


class OracleDebt:
    """Research upper bound, not deployable: debt = reference boxes of THIS frame that the
    tracker's coasted prediction would miss (IoU < 0.5). It knows the future frame, so it shows
    how much any learned debt could gain on the data at all."""

    name = "oracle"

    def __init__(self, reference):
        self.reference = reference

    def debt(self, t, thumb, tracker, fps):
        from .simulate import iou
        from ..track.kalman import xyah_to_xyxy

        r = self.reference[t]
        if not r:
            return 0.0
        pred = [xyah_to_xyxy(tracker.kf.F @ tr.mean) for tr in tracker.tracks if tr.state == 1]
        if not pred:
            return float(len(r))
        m = iou(np.array([b for _, b, _ in r]), np.array(pred))
        return float((m.max(1) < 0.5).sum())

    def computed(self, t, thumb):
        pass


class PerceptionDebt:
    def __init__(self, scale=1.0, local=12.0, global_=6.0, motion=0.25, edge=0.5, age=2.0, unconfirmed=0.05,
                 img_size=None):
        self.scale = scale
        self.budget = {"local": local, "global": global_, "motion": motion, "edge": edge, "age": age,
                       "unconfirmed": unconfirmed}
        self.name = f"debt x{scale:g}"
        self.img_size = img_size
        self.last, self.last_t = None, -10 ** 9
        self.last_terms = {}

    def terms(self, t, thumb, tracker, fps):
        g, loc = cell_change(thumb, self.last)
        motion, edge, unconfirmed = 0.0, 0.0, 0.0
        W, H = self.img_size
        for tr in tracker.tracks:
            if tr.state == 0:  # TENTATIVE
                unconfirmed = max(unconfirmed, (tracker.frame + 1 - tr.last_frame) / fps)
            if tr.state != 1:  # TRACKED only: lost tracks are the re-ID memory, not on screen
                continue
            b, gap = tr.box, tracker.frame + 1 - tr.last_frame
            v = np.abs(tr.mean[4:6]) * gap  # predicted centre shift since the last measurement
            size = max(b[2] - b[0], b[3] - b[1], 8.0)
            motion = max(motion, float(np.hypot(*v)) / size)
            margin = 0.02 * W
            if gap > 1 and (b[0] < margin or b[1] < margin or b[2] > W - margin or b[3] > H - margin):
                edge = max(edge, gap / fps)
        return {"local": loc, "global": g, "motion": motion, "edge": edge, "age": (t - self.last_t) / fps,
                "unconfirmed": unconfirmed}

    def debt(self, t, thumb, tracker, fps):
        """Largest term over its budget (>= 1: this camera should compute now)."""
        if self.last is None:
            return float("inf")
        self.last_terms = self.terms(t, thumb, tracker, fps)
        return max(v / (self.budget[k] * self.scale) for k, v in self.last_terms.items())

    def should_compute(self, t, thumb, tracker, fps):
        return self.debt(t, thumb, tracker, fps) >= 1.0

    def computed(self, t, thumb):
        self.last, self.last_t = thumb, t
