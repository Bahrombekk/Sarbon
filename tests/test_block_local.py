"""Halo-zero stem: sparse updates must equal a full pass; only changed blocks are encoded.

CPU-only. Run: .venv\\Scripts\\python.exe tests\\test_block_local.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sarbon.models import BlockLocalCache, BlockLocalStem  # noqa: E402
from sarbon.models.block_local import from_blocks, to_blocks  # noqa: E402


def main():
    torch.manual_seed(0)
    n, h, w, b = 2, 128, 192, 32
    x = torch.rand(n, 3, h, w)
    assert torch.equal(from_blocks(to_blocks(x, b), n, h // b, w // b), x)

    stem = BlockLocalStem(b).eval()
    with torch.no_grad():  # non-trivial BN statistics
        for m in stem.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.running_mean.uniform_(-0.1, 0.1)
                m.running_var.uniform_(0.5, 1.5)

    # locality: perturbing one block changes the outputs of that block only
    p4, p8 = stem(x)
    y = x.clone()
    y[0, :, 40:50, 70:80] += 0.5  # inside block (1, 2) of camera 0
    q4, q8 = stem(y)
    diff = (q8 - p8).abs().amax(1)  # [N, H/8, W/8]
    rows, cols = diff[0].nonzero(as_tuple=True)
    assert diff[1].max() == 0 and rows.min() >= 4 and rows.max() < 8 and cols.min() >= 8 and cols.max() < 12

    cache = BlockLocalCache(stem)
    cache.reset(x)
    frame, worst = x, 0.0
    for step in range(3):
        frame = frame.clone()
        frame[step % n, :, 10 + 30 * step:40 + 30 * step, 20:90] = torch.rand(3, 1, 1)
        changed = (frame - x if step == 0 else frame - prev).abs().amax(1)
        changed = changed.view(n, h // b, b, w // b, b).amax((2, 4)) > 0
        k = cache.update(frame, changed)
        r4, r8 = stem(frame)
        err = max((cache.p4 - r4).abs().max().item(), (cache.p8 - r8).abs().max().item())
        worst = max(worst, err)
        print(f"step {step}: encoded {k}/{changed.numel()} blocks (= changed, no halo), max err {err:.1e}")
        assert k == int(changed.sum())
        prev = frame
    assert worst < 1e-5, worst
    print("OK")


if __name__ == "__main__":
    main()
