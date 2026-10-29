"""GPU: the sparse-engine video pipeline must give the detector the same features as a dense
stem, frame after frame (lossless, tau = 0), and report its per-step latency.

Run: .venv\\Scripts\\python.exe tests\\test_pipeline_gpu.py [checkpoint.pt]
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from sarbon.models.sarbon_det import build, load_weights  # noqa: E402
from sarbon.pipeline import SarbonVideo  # noqa: E402
from scenes import Camera  # noqa: E402


def main(ckpt=None):
    torch.manual_seed(0)
    det = build("s", 6, halo_zero=False)
    if ckpt:
        load_weights(det, ckpt)
        print(f"loaded {ckpt}")
    video = SarbonVideo(det, n_cams=2, size=(640, 1152), device="cuda", keep_score=0.3)
    cams = [Camera(720, 1280, "moderate", seed=s, device="cuda") for s in range(2)]
    worst, times = 0.0, []
    for t in range(12):
        frames = [(c.next().flip(-1).clamp(0, 1) * 255).byte().cpu().numpy() for c in cams]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = video.step(frames)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
        # dense reference for the same frames
        x = video._prep(frames).permute(0, 3, 1, 2).contiguous().half()
        with torch.no_grad():
            ref = det.stem.half().forward_blocks(x)[1].float()
            det.stem.float()
        got = video.engine.outputs_nchw()[1].float()
        worst = max(worst, ((got - ref).abs().max() / ref.abs().max()).item())
    print(f"sparse vs dense stem features over 12 frames: max rel err {worst:.1e} (fp16)")
    print(f"step latency (2 cams, 1152x640, incl. tracking): median {np.median(times[2:]):.1f} ms")
    print(f"active blocks per step: {[round(a, 3) for a in video.stats['active_blocks']]}")
    print(f"tracks in last frame: {[len(r) for r in res]}")
    assert worst < 2e-2
    print("OK")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
