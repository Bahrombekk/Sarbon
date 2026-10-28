"""Sensor noise vs the change threshold tau (approximate mode).

With noise every pixel changes, so lossless gating (tau=0) recomputes everything.
For each tau we report the active-block fraction and the feature error against a
dense pass on the *same noisy frame*, relative to the error the noise itself causes
(dense on the noisy frame vs dense on the clean frame): ratio < 1 means the cache
error is smaller than the noise the detector sees anyway.
Run: .venv\\Scripts\\python.exe bench\\bench_noise.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sarbon.sparse import LocalBackbone  # noqa: E402
from sarbon.sparse.fused_engine import FusedSparseEngine  # noqa: E402
from scenes import Camera  # noqa: E402

H, W, DT, N = 1088, 1920, torch.float16, 40


def rel(a, b):
    return ((a - b).float().norm() / b.float().norm()).item()


@torch.no_grad()
def main(sigma=2.0 / 255):
    torch.manual_seed(0)
    net = LocalBackbone().to("cuda", DT).eval().to(memory_format=torch.channels_last)
    clean_cam = Camera(H, W, "moderate", seed=3)
    clean = [clean_cam.next().to(DT) for _ in range(N)]
    gen = torch.Generator(device="cuda").manual_seed(7)
    noisy = [(c + torch.randn(c.shape, generator=gen, device="cuda", dtype=DT) * sigma) for c in clean]

    def dense(f):
        return net(f[None].permute(0, 3, 1, 2))[1]  # stride-8 output

    noise_err = sum(rel(dense(n), dense(c)) for n, c in zip(noisy[-10:], clean[-10:])) / 10
    print(f"sensor noise sigma={sigma * 255:.1f}/255; feature error caused by noise itself: {noise_err:.3f}")
    print("   tau*255 | refresh | active blocks | err vs dense(noisy) | / noise err "
          "| err vs dense(clean) | / noise err")
    for tau, refresh in ((0, 0), (2, 0), (4, 0), (6, 0), (6, 30), (10, 30)):
        eng = FusedSparseEngine(net, 1, H, W, dtype=DT, tau=tau / 255, refresh=refresh)
        eng.reset(noisy[0][None])
        acts, errs, errs_clean = [], [], []
        for i, f in enumerate(noisy[1:], 1):
            _, k = eng.step(f[None])
            acts.append(k / eng.n_blocks)
            if i >= N - 10:
                errs.append(rel(eng.outputs_nchw()[1], dense(f)))
                errs_clean.append(rel(eng.outputs_nchw()[1], dense(clean[i])))
        e, ec = sum(errs) / len(errs), sum(errs_clean) / len(errs_clean)
        print(f"   {tau:7d} | {refresh:7d} | {sum(acts) / len(acts):12.1%} | {e:19.3f} | "
              f"{e / noise_err:10.2f} | {ec:19.3f} | {ec / noise_err:10.2f}")


if __name__ == "__main__":
    main()
