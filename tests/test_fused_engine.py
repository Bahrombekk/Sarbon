"""Correctness of the fused multi-camera engine against a plain dense forward pass.

Run: .venv\\Scripts\\python.exe tests\\test_fused_engine.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sarbon.sparse import LocalBackbone  # noqa: E402
from sarbon.sparse.fused_engine import FusedSparseEngine  # noqa: E402

DEV = "cuda"


def rel_err(engine, net, frames):
    with torch.no_grad():
        ref = net(frames.permute(0, 3, 1, 2).contiguous())
    return max(((o - r).abs().max() / r.abs().max()).item()
               for o, r in zip(engine.outputs_nchw(), ref))


def run(dtype, n_cams=3, h=256, w=384, block=32):
    torch.manual_seed(0)
    net = LocalBackbone().to(DEV, dtype).eval()
    eng = FusedSparseEngine(net, n_cams, h, w, block=block, dtype=dtype)
    frames = torch.rand(n_cams, h, w, 3, device=DEV, dtype=dtype)
    eng.reset(frames)
    errs = {"reset": rel_err(eng, net, frames)}
    for i in range(4):  # chained frames: errors must not accumulate
        frames = frames.clone()
        cam = i % n_cams
        y, x = 40 * i + 7, 60 * i + 13  # unaligned to blocks on purpose
        frames[cam, y:y + 37, x:x + 51] = torch.rand(3, device=DEV, dtype=dtype)
        if i == 3:
            frames[:, :, -20:] = 0.5  # touches the right border of every camera
        mode, k = eng.step(frames)
        assert mode == "sparse", mode
        errs[f"step{i} k={k}"] = rel_err(eng, net, frames)
    mode, k = eng.step(frames)
    assert mode == "skip" and k == 0
    eng.frames.copy_(torch.rand_like(frames))
    eng.dense_graph.replay()
    errs["dense"] = rel_err(eng, net, eng.frames)
    # sparse steps must keep working on the caches the dense graph produced (pool aliasing)
    frames = eng.frames.clone()
    for i in range(2):
        frames[i, 100:140, 200:260] = 0.25 * (i + 1)
        mode, k = eng.step(frames)
        assert mode == "sparse", mode
        errs[f"after-dense{i} k={k}"] = rel_err(eng, net, frames)
    # dense regime with hysteresis, then back to sparse
    budget, eng.fallback = eng.fallback, 0
    seen = []
    for i in range(eng.dense_check_every + 1):
        frames[i % n_cams, 10 * i:10 * i + 30, 50:90] = 0.1 * i
        seen.append(eng.step(frames)[0])
    assert seen == ["dense"] * len(seen), seen
    eng.fallback = budget
    for i in range(eng.dense_check_every + 1):
        frames[0, 150:170, 20 * i:20 * i + 25] = 0.9
        mode, _ = eng.step(frames)
    assert mode == "sparse", mode
    errs["after-hysteresis"] = rel_err(eng, net, frames)
    return errs


if __name__ == "__main__":
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    for dtype, tol in ((torch.float32, 1e-5), (torch.float16, 1e-2)):
        for block in (32, 64):
            errs = run(dtype, block=block)
            worst = max(errs.values())
            print(f"{str(dtype):14s} block={block}: worst rel err {worst:.2e}  "
                  + " ".join(f"{k}:{v:.1e}" for k, v in errs.items()))
            assert worst < tol, (dtype, block, errs)
    print("OK")
