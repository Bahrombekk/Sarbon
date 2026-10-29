"""Architecture variants: every block/activation builds and runs; RepVGG fusion is exact.

CPU-only. Run: .venv\\Scripts\\python.exe tests\\test_arch_variants.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sarbon.models.block_local import BlockLocalCache, BlockLocalStem, reparameterize  # noqa: E402
from sarbon.models.sarbon_det import build  # noqa: E402


def randomize_bn(m):
    with torch.no_grad():
        for b in m.modules():
            if isinstance(b, torch.nn.BatchNorm2d):
                b.running_mean.uniform_(-0.2, 0.2)
                b.running_var.uniform_(0.5, 2.0)
                b.weight.uniform_(0.5, 1.5)
                b.bias.uniform_(-0.2, 0.2)


def main():
    torch.manual_seed(0)
    x = torch.randn(1, 3, 128, 192)
    for block in ("res", "rep", "dw"):
        for act in ("relu", "silu", "gelu"):
            m = build("s", 6, block_type=block, act=act).eval()
            with torch.no_grad():
                out = m(x)
            assert torch.isfinite(out["pred_boxes"]).all()
            n = sum(p.numel() for p in m.parameters()) / 1e6
            print(f"block={block:3s} act={act:4s}: Sarbon-S {n:5.1f}M params, forward ok")

    # neck fusion variants
    ref_keys = set(build("s", 6).state_dict())
    for fusion in ("concat", "weighted", "gated"):
        m = build("s", 6, fusion=fusion).eval()
        with torch.no_grad():
            out = m(x)
        n = sum(p.numel() for p in m.parameters()) / 1e6
        print(f"fusion={fusion:8s}: Sarbon-S {n:5.1f}M params, forward ok")
        if fusion == "concat":
            assert set(m.state_dict()) == ref_keys  # old checkpoints keep loading
        else:
            f = m.core.td16
            a, b = torch.randn(2, 256, 8, 8), torch.randn(2, 256, 8, 8)
            w = f.weights(a, b)
            assert torch.allclose(w.sum(1), torch.ones(2, 8, 8), atol=1e-4)
            assert torch.allclose(w, torch.full_like(w, 0.5), atol=1e-3)  # starts as an average

    # RepVGG: fused single-3x3 model == multi-branch model
    stem = BlockLocalStem(block_type="rep").eval()
    randomize_bn(stem)
    with torch.no_grad():
        a4, a8 = stem(x)
        reparameterize(stem)
        b4, b8 = stem(x)
    err = max((a4 - b4).abs().max().item(), (a8 - b8).abs().max().item())
    print(f"RepVGG fuse: max abs diff {err:.1e}")
    assert err < 1e-4

    # fused rep stem keeps the halo-zero sparse-update property
    cache = BlockLocalCache(stem)
    cache.reset(x)
    y = x.clone()
    y[0, :, 40:60, 70:100] = 0.3
    changed = (y - x).abs().amax(1).view(1, 4, 32, 6, 32).amax((2, 4)) > 0
    cache.update(y, changed)
    with torch.no_grad():
        r4, r8 = stem(y)
    assert (cache.p8 - r8).abs().max() < 1e-5
    print("fused rep stem: halo-zero sparse update exact")
    print("OK")


if __name__ == "__main__":
    main()
