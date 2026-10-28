"""Exactness + wall-clock benchmark of the block-sparse temporal engine.

Run: .venv\\Scripts\\python.exe bench\\bench_block_sparse.py
"""
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sarbon.sparse import BlockSparseEngine, LocalBackbone  # noqa: E402
from sarbon.sparse.block_sparse import _bucket  # noqa: E402

H, W = 1088, 1920  # 1080p padded to a multiple of the block size
DEV = "cuda"


def moving_objects(frame, frac, gen):
    """Copy of `frame` with random rectangles repainted (≈ frac of the image area)."""
    out = frame.clone()
    target, done = frac * H * W, 0.0
    while done < target:
        h = int(torch.randint(24, 160, (1,), generator=gen))
        w = int(torch.randint(24, 160, (1,), generator=gen))
        y = int(torch.randint(0, H - h, (1,), generator=gen))
        x = int(torch.randint(0, W - w, (1,), generator=gen))
        out[:, y:y + h, x:x + w] = torch.rand(3, 1, 1, device=DEV, generator=None)
        done += h * w
    return out


def timeit(fn, reps=30, warm=10):
    for _ in range(warm):
        fn()
    times = []
    for _ in range(reps):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b))
    return statistics.median(times)


def exactness(block):
    torch.backends.cudnn.allow_tf32 = False  # TF32 rounding would mask real indexing bugs
    torch.manual_seed(0)
    net = LocalBackbone().to(DEV).float()
    eng = BlockSparseEngine(net, block)
    gen = torch.Generator().manual_seed(1)
    f0 = torch.rand(3, H, W, device=DEV)
    eng.full(f0)
    prev, worst = f0, 0.0
    cap = (H // block) * (W // block)  # budget = all blocks, so no overflow
    for frac in (0.01, 0.05, 0.2):  # several consecutive frames: errors must not accumulate
        f1 = moving_objects(prev, frac, gen)
        outs, over = eng.step(f1, cap)
        assert not over
        with torch.no_grad():
            ref = net(f1[None])
        for o, r in zip(outs, ref):
            worst = max(worst, ((o - r[0]).abs().max() / r.abs().max()).item())
        prev = f1
    torch.backends.cudnn.allow_tf32 = True
    return worst


def speed(block):
    torch.manual_seed(0)
    net = LocalBackbone().to(DEV).half()
    eng = BlockSparseEngine(net, block)
    gen = torch.Generator().manual_seed(2)
    f0 = torch.rand(3, H, W, device=DEV).half()
    eng.full(f0)
    with torch.no_grad():
        dense = timeit(lambda: net(f0[None]), reps=200)
    n_blocks = (H // block) * (W // block)
    rows = []
    for frac in (0.005, 0.02, 0.05, 0.10, 0.20, 0.40):
        f1 = moving_objects(f0, frac, gen)
        changed = eng.changed_blocks(f1, f0)
        k = int(eng.active_blocks(changed, eng.nodes[-1].radius).sum())
        cap = min(_bucket(k), n_blocks)
        # cost depends only on the budget, so replaying the same frame is representative
        t = timeit(lambda: eng.step(f1, cap), reps=200)
        rows.append((frac, changed.float().mean().item(), k / n_blocks, cap / n_blocks, t))
    return dense, rows


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    print(torch.cuda.get_device_name(0), f"{W}x{H}")
    for block in (32, 64):
        err = exactness(block)
        dense, rows = speed(block)
        print(f"\nblock={block}px  lossless max rel. error (fp32, 3 chained frames) = {err:.2e}")
        print(f"dense fp16: {dense:.3f} ms")
        print(" area changed | blocks changed | active@last | budget | gate+sparse ms | speedup")
        for frac, ch, act, cap, t in rows:
            print(f"   {frac:6.1%}    |    {ch:6.1%}      |   {act:6.1%}   | {cap:6.1%} |"
                  f"     {t:6.3f}     |  {dense / t:4.2f}x")
