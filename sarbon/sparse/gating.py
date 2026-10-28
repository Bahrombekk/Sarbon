"""Which blocks must be recomputed, given where the image changed.

Block b of a layer with total stride S and receptive radius R (image px) depends on
image pixels [b*B - R, (b+1)*B - S + R] along each axis (receptive fields of stacked
'same' convs are symmetric around fy*S). Two conservative (hence exact) estimates:

- block-level  : changed *blocks* dilated by ceil(R / B) blocks (what M0/M1 used);
- cell-level   : changed *cells* (c x c px, c | B) dilated by ceil(R / c) cells, then
                 a block is active if any of its cells is. Tighter: a change in the
                 middle of a block no longer activates a full ring of neighbours.
"""
import math

import torch
import torch.nn.functional as F


def changed_cells(frame, prev, cell, tau=0.0):
    """[N, H/cell, W/cell] bool from NHWC frames (mean abs diff per cell > tau)."""
    n, h, w, c = frame.shape
    d = (frame.float() - prev.float()).abs().view(n, h // cell, cell, w // cell, cell, c)
    return d.mean((2, 4, 5)) > tau


def dilate(mask, r):
    if r == 0:
        return mask
    return F.max_pool2d(mask.float()[:, None], 2 * r + 1, 1, r)[:, 0] > 0


def cells_to_blocks(mask, k):
    """Any-pool k x k cells into one block."""
    n, hc, wc = mask.shape
    return mask.view(n, hc // k, k, wc // k, k).any(4).any(2)


def active_blocks_block_level(cells, radius, cell, block):
    blocks = cells_to_blocks(cells, block // cell)
    return dilate(blocks, math.ceil(radius / block))


def active_blocks_cell_level(cells, radius, cell, block):
    return cells_to_blocks(dilate(cells, math.ceil(radius / cell)), block // cell)
