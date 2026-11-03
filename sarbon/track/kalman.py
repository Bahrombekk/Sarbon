"""Constant-velocity Kalman filter on (cx, cy, aspect, height), as in SORT/ByteTrack.

Process and measurement noise scale with the box height. With `nsa=True` the
measurement noise is scaled by (1 - score) (NSA Kalman, GIAOTracker 2202.11983):
confident detections pull the state harder than doubtful ones.
"""
import numpy as np

_STD_POS, _STD_VEL = 1.0 / 20, 1.0 / 160


def inv2(s):
    """Closed-form inverse of a batch of 2x2 matrices [..., 2, 2]."""
    det = s[..., 0, 0] * s[..., 1, 1] - s[..., 0, 1] * s[..., 1, 0]
    out = np.stack([np.stack([s[..., 1, 1], -s[..., 0, 1]], -1),
                    np.stack([-s[..., 1, 0], s[..., 0, 0]], -1)], -2)
    return out / det[..., None, None]


def inv4(s):
    """Block (Schur complement) inverse of a batch of 4x4 matrices [..., 4, 4].

    np.linalg.inv costs ~20 us per tiny matrix (LAPACK call overhead), which dominated the
    tracker with many live tracks; this is ~50x faster and exact for the SPD covariances here.
    """
    a, b, c, d = s[..., :2, :2], s[..., :2, 2:], s[..., 2:, :2], s[..., 2:, 2:]
    ai = inv2(a)
    si = inv2(d - c @ ai @ b)
    aib, cai = ai @ b, c @ ai
    top = np.concatenate([ai + aib @ si @ cai, -aib @ si], -1)
    return np.concatenate([top, np.concatenate([-si @ cai, si], -1)], -2)


class KalmanXYAH:
    ndim = 4

    def __init__(self):
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)
        self.H = np.eye(4, 8)

    def initiate(self, z):
        mean = np.r_[z, np.zeros(4)]
        h = z[3]
        std = [2 * _STD_POS * h, 2 * _STD_POS * h, 1e-2, 2 * _STD_POS * h,
               10 * _STD_VEL * h, 10 * _STD_VEL * h, 1e-5, 10 * _STD_VEL * h]
        return mean, np.diag(np.square(std))

    def predict(self, mean, cov):
        h = mean[3]
        q = np.diag(np.square([_STD_POS * h, _STD_POS * h, 1e-2, _STD_POS * h,
                               _STD_VEL * h, _STD_VEL * h, 1e-5, _STD_VEL * h]))
        return self.F @ mean, self.F @ cov @ self.F.T + q

    def predict_batch(self, means, covs, dt=1):
        """predict() for N tracks at once: means [N, 8], covs [N, 8, 8].

        dt: source frames elapsed (live cameras drop frames when the pipeline is slower than the
        stream); the noise per frame is a random walk, so its variance grows linearly with dt.
        """
        std = means[:, 3:4] * np.array([_STD_POS, _STD_POS, 0, _STD_POS, _STD_VEL, _STD_VEL, 0, _STD_VEL])
        std[:, 2], std[:, 6] = 1e-2, 1e-5
        q = dt * np.square(std)[:, :, None] * np.eye(8)
        f = self.F
        if dt != 1:
            f = np.eye(8)
            f[:4, 4:] = dt * np.eye(4)
        return means @ f.T, f @ covs @ f.T + q

    def update_batch(self, means, covs, zs, scores=None):
        """update() for N tracks at once (H selects the first 4 state entries)."""
        r = np.square(means[:, 3:4] * np.array([_STD_POS, _STD_POS, 0, _STD_POS]))
        r[:, 2] = 1e-2
        if scores is not None:
            r = r * (1.0 - scores[:, None])
        k = covs[:, :, :4] @ inv4(covs[:, :4, :4] + r[:, :, None] * np.eye(4))
        means = means + (k @ (zs - means[:, :4])[..., None])[..., 0]
        return means, covs - k @ covs[:, :4, :]

    def update(self, mean, cov, z, score=None):
        h = mean[3]
        r = np.square([_STD_POS * h, _STD_POS * h, 1e-1, _STD_POS * h])
        if score is not None:
            r = r * (1.0 - score)  # NSA: trust confident detections more
        s = self.H @ cov @ self.H.T + np.diag(r)
        k = cov @ self.H.T @ np.linalg.inv(s)
        mean = mean + k @ (z - self.H @ mean)
        cov = (np.eye(8) - k @ self.H) @ cov
        return mean, cov


def xyxy_to_xyah(b):
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, w / max(h, 1e-6), h])


def xyah_to_xyxy(m):
    cx, cy, a, h = m[:4]
    w = a * h
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
