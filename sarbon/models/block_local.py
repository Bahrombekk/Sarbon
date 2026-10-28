"""Halo-zero high-resolution stages: every B x B image block is encoded independently.

The stem reshapes the image into a batch of blocks, so convolutions zero-pad at block
borders and no information crosses a block boundary below the dense core. Then:
- temporal sparsity is exact and trivial: re-encode exactly the changed blocks, no
  receptive-field dilation (the M0/M1 halo engine activates a ring around changes);
- the sparse path is plain batched cuDNN on gathered blocks, no custom kernel;
- it is the CNN analogue of ViT's non-overlapping patchify (RF-DETR, LW-DETR), where
  cross-patch context comes from later global layers -- here, the dense core.
Open question (needs training): accuracy cost of the block borders vs a standard stem.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_act(cin, cout, s=1, act=True):
    layers = [nn.Conv2d(cin, cout, 3, s, 1, bias=False), nn.BatchNorm2d(cout)]
    if act:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class Res(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.body = nn.Sequential(conv_bn_act(c, c), conv_bn_act(c, c, act=False))

    def forward(self, x):
        return F.relu(self.body(x) + x)


def to_blocks(img, b):
    """[N, C, H, W] -> [N*Hb*Wb, C, b, b] (block-major, row by row)."""
    n, c, h, w = img.shape
    x = img.view(n, c, h // b, b, w // b, b).permute(0, 2, 4, 1, 3, 5)
    return x.reshape(-1, c, b, b)


def from_blocks(x, n, hb, wb):
    """Inverse of to_blocks for feature tiles [N*Hb*Wb, C, t, t] -> [N, C, Hb*t, Wb*t]."""
    k, c, t, _ = x.shape
    return x.view(n, hb, wb, c, t, t).permute(0, 3, 1, 4, 2, 5).reshape(n, c, hb * t, wb * t)


class BlockLocalStem(nn.Module):
    """Stride-4 and stride-8 features computed per block (block = 32 px by default)."""

    def __init__(self, block=32, widths=(32, 64, 128), depths=(2, 2)):
        super().__init__()
        self.block = block
        self.stem = nn.Sequential(conv_bn_act(3, widths[0], 2), conv_bn_act(widths[0], widths[1], 2))
        self.s4 = nn.Sequential(*[Res(widths[1]) for _ in range(depths[0])])
        self.down = conv_bn_act(widths[1], widths[2], 2)
        self.s8 = nn.Sequential(*[Res(widths[2]) for _ in range(depths[1])])

    def forward_blocks(self, x):
        p4 = self.s4(self.stem(x))
        return p4, self.s8(self.down(p4))

    def forward(self, img):
        n, _, h, w = img.shape
        hb, wb = h // self.block, w // self.block
        p4, p8 = self.forward_blocks(to_blocks(img, self.block))
        return from_blocks(p4, n, hb, wb), from_blocks(p8, n, hb, wb)


class BlockLocalCache:
    """Keeps the stem's outputs for a set of cameras and re-encodes only changed blocks."""

    def __init__(self, stem: BlockLocalStem):
        self.stem = stem.eval()
        self.p4 = self.p8 = None

    @torch.no_grad()
    def reset(self, img):
        self.p4, self.p8 = self.stem(img)
        return self.p4, self.p8

    @staticmethod
    def _tiles(feat, b, stride):
        """Writable view [N, Hb, Wb, C, t, t] of a full feature map."""
        n, c, h, w = feat.shape
        t = b // stride
        return feat.view(n, c, h // t, t, w // t, t).permute(0, 2, 4, 1, 3, 5)

    @torch.no_grad()
    def update(self, img, changed):
        """img [N, 3, H, W]; changed [N, Hb, Wb] bool. Returns the number of blocks encoded."""
        b = self.stem.block
        blocks = self._tiles(img, b, 1)[changed]  # [K, 3, b, b]
        if len(blocks):
            p4, p8 = self.stem.forward_blocks(blocks)
            self._tiles(self.p4, b, 4)[changed] = p4
            self._tiles(self.p8, b, 8)[changed] = p8
        return len(blocks)
