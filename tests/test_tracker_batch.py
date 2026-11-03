"""Batched tracker maths == the per-track reference (Kalman predict/update, inverses, Mahalanobis).

Run: .venv\\Scripts\\python.exe tests\\test_tracker_batch.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sarbon.track.ground import GroundKalman  # noqa: E402
from sarbon.track.kalman import KalmanXYAH, inv2, inv4  # noqa: E402
from sarbon.track.tracker import SarbonTracker, TrackerConfig  # noqa: E402


def spd(rng, n, d):
    a = rng.normal(size=(n, d, d))
    return a @ a.transpose(0, 2, 1) + np.eye(d)


def main():
    rng = np.random.default_rng(0)
    for d, f in ((2, inv2), (4, inv4)):
        s = spd(rng, 50, d)
        err = np.abs(f(s) - np.linalg.inv(s)).max()
        assert err < 1e-9, (d, err)

    kf, n = KalmanXYAH(), 20
    means = np.c_[rng.uniform(0, 1000, (n, 2)), rng.uniform(0.3, 0.6, n), rng.uniform(20, 300, n),
                  rng.normal(size=(n, 4))]
    covs = spd(rng, n, 8)
    zs, scores = means[:, :4] + rng.normal(size=(n, 4)), rng.uniform(0.1, 0.9, n)
    pm, pc = kf.predict_batch(means, covs)
    for sc in (None, scores):
        um, uc = kf.update_batch(pm, pc, zs, sc)
        for i in range(n):
            rm, rc = kf.predict(means[i], covs[i])
            assert np.allclose(rm, pm[i]) and np.allclose(rc, pc[i])
            rm, rc = kf.update(rm, rc, zs[i], None if sc is None else sc[i])
            assert np.allclose(rm, um[i]) and np.allclose(rc, uc[i])

    gk = GroundKalman()
    gm, gc = rng.normal(size=(n, 4)) * 5, spd(rng, n, 4) * 0.1
    gz, gr = rng.normal(size=(7, 2)) * 5, spd(rng, 7, 2) * 0.05
    bm, bc = gk.predict_batch(gm, gc)
    um, uc = gk.update_batch(bm, bc, gz[:1].repeat(n, 0), gr[:1].repeat(n, 0))
    pairs = gk.mahalanobis_pairs(bm, bc, gz, gr)
    for i in range(n):
        rm, rc = gk.predict(gm[i], gc[i])
        assert np.allclose(rm, bm[i]) and np.allclose(rc, bc[i])
        assert np.allclose(gk.mahalanobis(rm, rc, gz, gr), pairs[i])
        rm, rc = gk.update(rm, rc, gz[0], gr[0])
        assert np.allclose(rm, um[i]) and np.allclose(rc, uc[i])

    # dt > 1 moves the mean exactly like dt single steps; the uncertainty is not smaller
    for f, m, c in ((kf, means, covs), (gk, gm, gc)):
        m3, c3 = f.predict_batch(m, c, 3)
        r, rc = m, c
        for _ in range(3):
            r, rc = f.predict_batch(r, rc)
        assert np.allclose(m3, r)
        assert (np.linalg.eigvalsh(c3) > 0).all()

    # dropped frames (live camera slower than the stream): a box moving 3 px/frame, seen on
    # every 4th frame, keeps one ID and its predicted box stays on it
    for steps in (1, 4):
        tr = SarbonTracker(TrackerConfig())
        ids = set()
        for f in range(0, 120, steps):
            x = 100 + 3.0 * f
            out = tr.update(np.array([[x, 200, x + 40, 300]]), np.array([0.9]), steps=steps)
            ids |= {i for i, *_ in out}
        pred = tr.tracks[0].box  # after the update; predict one interval ahead
        tr._predict_all(steps)
        assert len(ids) == 1, (steps, ids)
        assert abs(tr.tracks[0].box[0] - (100 + 3.0 * (f + steps))) < 3, (steps, tr.tracks[0].box, pred)
    print("OK")


if __name__ == "__main__":
    main()
