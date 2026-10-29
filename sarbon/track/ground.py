"""Self-calibrating ground plane for a fixed camera (no manual calibration).

Pinhole camera, no roll, flat ground, people of roughly constant real height H.
For a person standing at depth Z, foot row y and image height h satisfy
    y - y0 = f * Hc / Z      and      h = f * H / Z      =>      h = a * (y - y0),  a = H / Hc,
where y0 is the horizon row and Hc the camera height. Fitting h against y over
confident tracks gives y0 and Hc for free; ground coordinates follow:
    X = (x - cx) * Hc / (y - y0)   (metres, exact)
    Z = f * Hc / (y - y0)          (metres up to the focal length f; f ~ image width by default)
So every detection gets a ground position and a distance from the camera (depth)
from geometry alone. Motion is then constant-velocity in metres, which it is for
walkers -- in pixels the same walker speeds up and grows when approaching.
"""
import numpy as np


class GroundModel:
    def __init__(self, img_w, img_h, person_h=1.7, focal=None, min_samples=150, max_samples=4000):
        self.cx, self.img_h = img_w / 2, img_h
        self.f = focal or float(img_w)
        self.H, self.min_samples, self.max_samples = person_h, min_samples, max_samples
        self.samples = []
        self.y0 = self.a = None

    @property
    def ready(self):
        return self.a is not None

    @property
    def cam_height(self):
        return self.H / self.a

    def observe(self, box):
        """Feed a confidently tracked, unoccluded person box (xyxy)."""
        self.samples.append((box[3], box[3] - box[1]))
        if len(self.samples) > self.max_samples:
            self.samples = self.samples[-self.max_samples:]

    def fit(self, iters=5):
        """Robust (iteratively reweighted) line fit h = a*y + b; y0 = -b / a."""
        if len(self.samples) < self.min_samples:
            return False
        s = np.array(self.samples)
        y, h = s[:, 0], s[:, 1]
        w = np.ones_like(y)
        for _ in range(iters):
            A = np.stack([y, np.ones_like(y)], 1) * w[:, None]
            a, b = np.linalg.lstsq(A, h * w, rcond=None)[0]
            r = h - (a * y + b)
            scale = 1.4826 * np.median(np.abs(r)) + 1e-6
            w = 1.0 / np.maximum(1.0, np.abs(r) / (2.0 * scale))  # Huber weights
        if a <= 1e-4:
            return False
        y0 = -b / a
        if y0 >= np.percentile(y, 2):  # horizon must lie above (almost) every foot
            return False
        self.a, self.y0 = a, y0
        return True

    def expected_height(self, foot_y):
        return self.a * (foot_y - self.y0)

    def to_ground(self, x, y):
        d = np.maximum(np.asarray(y, float) - self.y0, 1.0)
        hc = self.cam_height
        return np.stack([(np.asarray(x, float) - self.cx) * hc / d, self.f * hc / d], -1)

    def jacobian(self, x, y):
        """d(X, Z) / d(x, y) at an image foot point."""
        d = max(y - self.y0, 1.0)
        hc = self.cam_height
        return np.array([[hc / d, -(x - self.cx) * hc / d ** 2],
                         [0.0, -self.f * hc / d ** 2]])

    def foot(self, box):
        return (box[0] + box[2]) / 2, box[3]


class GroundKalman:
    """Constant velocity on the ground plane; image noise propagated through the camera.

    Measurement covariance R = J diag(sx^2, sy^2) J^T (UCMCTrack's correlated noise,
    2312.08952): a pixel of error far away is metres on the ground, near it is cm.
    """

    def __init__(self, accel=0.02):
        self.F = np.eye(4)
        self.F[:2, 2:] = np.eye(2)
        self.Hm = np.eye(2, 4)
        self.q = accel

    def initiate(self, z, r):
        mean = np.r_[z, 0.0, 0.0]
        cov = np.zeros((4, 4))
        cov[:2, :2] = r
        cov[2:, 2:] = np.eye(2) * 0.1 ** 2
        return mean, cov

    def predict(self, mean, cov):
        g = np.array([[0.5, 0], [0, 0.5], [1, 0], [0, 1]])
        return self.F @ mean, self.F @ cov @ self.F.T + g @ g.T * self.q ** 2

    def project(self, mean, cov, r):
        return self.Hm @ mean, self.Hm @ cov @ self.Hm.T + r

    def update(self, mean, cov, z, r):
        m, s = self.project(mean, cov, r)
        k = cov @ self.Hm.T @ np.linalg.inv(s)
        return mean + k @ (z - m), (np.eye(4) - k @ self.Hm) @ cov

    def mahalanobis(self, mean, cov, zs, rs):
        """Squared Mahalanobis distances of measurements zs [N, 2] with covariances rs [N, 2, 2]."""
        m = self.Hm @ mean
        p = self.Hm @ cov @ self.Hm.T
        out = np.empty(len(zs))
        for i, (z, r) in enumerate(zip(zs, rs)):
            d = z - m
            out[i] = d @ np.linalg.solve(p + r, d)
        return out
