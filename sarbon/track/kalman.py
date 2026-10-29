"""Constant-velocity Kalman filter on (cx, cy, aspect, height), as in SORT/ByteTrack.

Process and measurement noise scale with the box height. With `nsa=True` the
measurement noise is scaled by (1 - score) (NSA Kalman, GIAOTracker 2202.11983):
confident detections pull the state harder than doubtful ones.
"""
import numpy as np

_STD_POS, _STD_VEL = 1.0 / 20, 1.0 / 160


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
