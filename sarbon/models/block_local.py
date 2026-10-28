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


ACTS = {"relu": nn.ReLU, "silu": nn.SiLU, "gelu": nn.GELU}


def make_act(name):
    return ACTS[name]() if name == "gelu" else ACTS[name](inplace=True)


def conv_bn_act(cin, cout, s=1, act=True, act_name="relu", k=3, groups=1):
    layers = [nn.Conv2d(cin, cout, k, s, k // 2, groups=groups, bias=False), nn.BatchNorm2d(cout)]
    if act:
        layers.append(make_act(act_name))
    return nn.Sequential(*layers)


class Res(nn.Module):
    """ResNet basic block: 3x3 -> 3x3 + identity."""

    def __init__(self, c, act="relu"):
        super().__init__()
        self.body = nn.Sequential(conv_bn_act(c, c, act_name=act), conv_bn_act(c, c, act=False))
        self.act = make_act(act)

    def forward(self, x):
        return self.act(self.body(x) + x)


class RepConv(nn.Module):
    """RepVGG (2101.03697): 3x3 + 1x1 + identity branches while training, one 3x3 at inference.

    After `fuse()` it is a single dense 3x3 conv: the cheapest shape for cuDNN, TensorRT
    and the Triton tile kernel, with the optimisation benefits of the multi-branch form.
    """

    def __init__(self, c, act="relu"):
        super().__init__()
        self.c3 = conv_bn_act(c, c, act=False)
        self.c1 = conv_bn_act(c, c, act=False, k=1)
        self.idn = nn.BatchNorm2d(c)
        self.act = make_act(act)
        self.fused = None

    def forward(self, x):
        if self.fused is not None:
            return self.act(self.fused(x))
        return self.act(self.c3(x) + self.c1(x) + self.idn(x))

    @staticmethod
    def _fold(kernel, bn):
        std = (bn.running_var + bn.eps).sqrt()
        t = bn.weight / std
        return kernel * t.view(-1, 1, 1, 1), bn.bias - bn.running_mean * t

    @torch.no_grad()
    def fuse(self):
        c = self.c3[0].out_channels
        k3, b3 = self._fold(self.c3[0].weight, self.c3[1])
        k1, b1 = self._fold(F.pad(self.c1[0].weight, [1, 1, 1, 1]), self.c1[1])
        eye = torch.zeros_like(k3)
        eye[torch.arange(c), torch.arange(c), 1, 1] = 1.0
        ki, bi = self._fold(eye, self.idn)
        self.fused = nn.Conv2d(c, c, 3, 1, 1).to(k3)
        self.fused.weight.copy_(k3 + k1 + ki)
        self.fused.bias.copy_(b3 + b1 + bi)
        del self.c3, self.c1, self.idn


class RepBlock(nn.Module):
    """Two RepConvs (same 3x3 depth as Res, but plain at inference)."""

    def __init__(self, c, act="relu"):
        super().__init__()
        self.body = nn.Sequential(RepConv(c, act), RepConv(c, act))

    def forward(self, x):
        return self.body(x)


class DWBlock(nn.Module):
    """Inverted bottleneck (MobileNetV2 / ConvNeXt-lite): 1x1 expand, 3x3 depthwise, 1x1 project."""

    def __init__(self, c, act="relu", expand=3):
        super().__init__()
        e = c * expand
        self.body = nn.Sequential(conv_bn_act(c, e, act_name=act, k=1),
                                  conv_bn_act(e, e, act_name=act, groups=e),
                                  conv_bn_act(e, c, act=False, k=1))

    def forward(self, x):
        return x + self.body(x)


BLOCKS = {"res": Res, "rep": RepBlock, "dw": DWBlock}


def reparameterize(model):
    """Fuse every RepConv in place (call after training, before export/benchmark)."""
    for m in model.modules():
        if isinstance(m, RepConv) and m.fused is None:
            m.fuse()
    return model


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

    def __init__(self, block=32, widths=(32, 64, 128), depths=(2, 2), block_type="res", act="relu"):
        super().__init__()
        self.block = block
        unit = BLOCKS[block_type]
        self.stem = nn.Sequential(conv_bn_act(3, widths[0], 2, act_name=act),
                                  conv_bn_act(widths[0], widths[1], 2, act_name=act))
        self.s4 = nn.Sequential(*[unit(widths[1], act) for _ in range(depths[0])])
        self.down = conv_bn_act(widths[1], widths[2], 2, act_name=act)
        self.s8 = nn.Sequential(*[unit(widths[2], act) for _ in range(depths[1])])

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
