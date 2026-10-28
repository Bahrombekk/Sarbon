"""CPU analysis: GPU work implied by block-level vs cell-level gating (no GPU needed).

Work model: a layer's sparse cost is (active blocks) x (FLOPs per block), so the
total is the FLOP-weighted active fraction of the local backbone. Also checks that
cell-level gating is never less conservative than the exact receptive field.
Run: .venv\\Scripts\\python.exe bench\\analyze_gating.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sarbon.sparse import LocalBackbone  # noqa: E402
from sarbon.sparse.gating import (active_blocks_block_level, active_blocks_cell_level,  # noqa: E402
                                  cells_to_blocks, changed_cells)
from scenes import Camera  # noqa: E402

H, W, B = 1088, 1920, 32


def layer_costs(net):
    """(radius, flops per image pixel) per conv node of the sparse program."""
    nodes, _ = net.program()
    out = []
    for n in nodes:
        if n.kind == "conv":
            c = n.conv
            flops = 2 * c.in_channels * c.out_channels * 9 / (n.stride ** 2)
            out.append((n.radius, flops))
    return out


def exact_active(cells_px, radius, stride):
    """Ground truth on a pixel mask: block active iff its dependency window holds a change."""
    n, h, w = cells_px.shape
    m = cells_px.float()[:, None]
    k = 2 * radius + 1  # square dilation as two separable 1-D max pools (fast on CPU)
    m = torch.nn.functional.max_pool2d(m, (k, 1), 1, (radius, 0))
    ii = torch.nn.functional.max_pool2d(m, (1, k), 1, (0, radius))[:, 0] > 0
    # a block depends on output pixels at fy*S in [bB, (b+1)B - S]; sample those positions
    sub = ii[:, ::stride, ::stride]
    t = B // stride
    return sub.view(n, h // B, t, w // B, t).any(4).any(2)


def main():
    torch.manual_seed(0)
    net = LocalBackbone()
    costs = layer_costs(net)
    total = sum(f for _, f in costs)
    print(f"local backbone: {len(costs)} convs, receptive radius {costs[-1][0]} px, block {B} px")
    print("halo-zero = block-local high-res stages (no cross-block halo): active = changed blocks")
    print("preset    | changed px | cell | block-level | cell-level | saving | exact | halo-zero")
    for preset in ("quiet", "moderate", "busy"):
        cam = Camera(H, W, preset, seed=1, device="cpu")
        frames = [cam.next()[None] for _ in range(4)]
        pairs = list(zip(frames, frames[1:]))
        pix = [changed_cells(c, p, 1) for p, c in pairs]
        hz = sum(cells_to_blocks(m, B).float().mean().item() for m in pix) / len(pairs)
        for cell in (16, 8, 4):
            wb = wc = 0.0
            ok = True
            for (prev, cur), pm in zip(pairs, pix):
                cells = changed_cells(cur, prev, cell)
                for (r, f) in costs:
                    a = active_blocks_block_level(cells, r, cell, B)
                    b = active_blocks_cell_level(cells, r, cell, B)
                    wb += f * a.float().mean().item()
                    wc += f * b.float().mean().item()
                    if r == costs[-1][0]:
                        ok &= bool((b | ~exact_active(pm, r, 8)).all())  # must cover exact set
            k = len(pairs)
            print(f"{preset:9s} | {sum(m.float().mean().item() for m in pix) / k:9.2%} | "
                  f"{cell:4d} | {wb / k / total:10.1%} | {wc / k / total:9.1%} | "
                  f"{1 - wc / wb:6.1%} | {'ok' if ok else 'FAIL'}  | {hz:8.1%}")


if __name__ == "__main__":
    main()
