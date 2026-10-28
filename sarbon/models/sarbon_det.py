"""Sarbon detector: halo-zero stem + dense core + track-query DETR decoder.

    image ─► BlockLocalStem (per 32 px block, stride 4/8; cacheable per block)
          ─► DenseCore: stride 16/32 stages, one attention layer at stride 32 (AIFI),
             top-down + bottom-up fusion -> P8, P16, P32 (hidden channels)
          ─► encoder head scores every token; top-K become new object queries
          ─► decoder: [track queries from the previous frame] + [new queries],
             self-attention, box-relative multi-scale deformable cross-attention,
             iterative box refinement; NMS-free, one box per query.

Track queries carry identity: if a track query wins, the object keeps its ID; the
decoder's output embeddings are fed back as next frame's track queries. Training
assigns track queries to their own ground-truth ID (no Hungarian), MOTR-style.
Not yet included (planned): D-FINE distribution regression, contrastive denoising,
the foveal small-object branch.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block_local import BlockLocalStem, Res, conv_bn_act
from .box_ops import inverse_sigmoid
from .deform_attn import MSDeformAttn


@dataclass
class SarbonConfig:
    widths: tuple = (32, 64, 128)  # stem stride 2/4/8
    depths: tuple = (2, 2)  # Res blocks at stride 4/8
    c16: int = 256
    c32: int = 256
    hidden: int = 256
    heads: int = 8
    ffn: int = 1024
    dec_layers: int = 4
    queries: int = 300
    points: int = 4
    block: int = 32
    halo_zero: bool = True


SIZES = {
    "n": SarbonConfig(widths=(16, 32, 64), depths=(1, 1), c16=128, c32=256, hidden=128, heads=4,
                      ffn=512, dec_layers=3, queries=150),
    "s": SarbonConfig(),
    "m": SarbonConfig(widths=(48, 96, 192), depths=(2, 3), c16=384, c32=512, dec_layers=6),
}


def mlp(d_in, d_hidden, d_out, n):
    layers = []
    for i in range(n):
        layers += [nn.Linear(d_in if i == 0 else d_hidden, d_out if i == n - 1 else d_hidden)]
        if i < n - 1:
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def fuse(cin, cout):
    """Cheap fusion: 1x1 channel mix, then 3x3 depthwise spatial mix."""
    return nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, 1, 1, groups=cout, bias=False), nn.BatchNorm2d(cout),
                         nn.ReLU(inplace=True))


def sincos_2d(h, w, dim, device, temperature=10000.0):
    y, x = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32),
                          torch.arange(w, device=device, dtype=torch.float32), indexing="ij")
    d = dim // 4
    omega = 1.0 / temperature ** (torch.arange(d, device=device, dtype=torch.float32) / d)
    ox, oy = x.flatten()[:, None] * omega, y.flatten()[:, None] * omega
    return torch.cat([ox.sin(), ox.cos(), oy.sin(), oy.cos()], 1)[None]


class DenseCore(nn.Module):
    """Cross-block context: everything here runs densely every frame at stride >= 8."""

    def __init__(self, cfg: SarbonConfig):
        super().__init__()
        c8, d = cfg.widths[2], cfg.hidden
        self.to16 = nn.Sequential(conv_bn_act(c8, cfg.c16, 2), Res(cfg.c16), Res(cfg.c16))
        self.to32 = nn.Sequential(conv_bn_act(cfg.c16, cfg.c32, 2), Res(cfg.c32))
        self.in8, self.in16, self.in32 = (nn.Sequential(nn.Conv2d(c, d, 1, bias=False), nn.BatchNorm2d(d))
                                          for c in (c8, cfg.c16, cfg.c32))
        self.aifi = nn.TransformerEncoderLayer(d, cfg.heads, cfg.ffn, 0.0, batch_first=True,
                                               activation="gelu")
        self.td16, self.td8 = fuse(2 * d, d), fuse(2 * d, d)
        self.down8, self.down16 = conv_bn_act(d, d, 2), conv_bn_act(d, d, 2)
        self.bu16, self.bu32 = fuse(2 * d, d), fuse(2 * d, d)

    def forward(self, p8):
        p16 = self.to16(p8)
        p32 = self.to32(p16)
        x8, x16, x32 = self.in8(p8), self.in16(p16), self.in32(p32)
        b, d, h, w = x32.shape
        t = x32.flatten(2).transpose(1, 2)
        pos = sincos_2d(h, w, d, t.device).to(t.dtype)
        # attention only at stride 32: global context for <1/1000 of the pixels
        t = self.aifi(t + pos) - pos
        x32 = t.transpose(1, 2).reshape(b, d, h, w)
        t16 = self.td16(torch.cat([x16, F.interpolate(x32, scale_factor=2.0)], 1))
        o8 = self.td8(torch.cat([x8, F.interpolate(t16, scale_factor=2.0)], 1))
        o16 = self.bu16(torch.cat([t16, self.down8(o8)], 1))
        o32 = self.bu32(torch.cat([x32, self.down16(o16)], 1))
        return [o8, o16, o32]


class DecoderLayer(nn.Module):
    def __init__(self, cfg: SarbonConfig, n_levels=3):
        super().__init__()
        d = cfg.hidden
        self.self_attn = nn.MultiheadAttention(d, cfg.heads, batch_first=True)
        self.cross = MSDeformAttn(d, cfg.heads, n_levels, cfg.points)
        self.ffn = nn.Sequential(nn.Linear(d, cfg.ffn), nn.ReLU(inplace=True), nn.Linear(cfg.ffn, d))
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, tgt, pos, ref, memory, shapes, attn_mask=None):
        q = tgt + pos
        tgt = self.n1(tgt + self.self_attn(q, q, tgt, attn_mask=attn_mask, need_weights=False)[0])
        tgt = self.n2(tgt + self.cross(tgt + pos, ref, memory, shapes))
        return self.n3(tgt + self.ffn(tgt))


class SarbonDetector(nn.Module):
    def __init__(self, num_classes, cfg: SarbonConfig = None):
        super().__init__()
        cfg = cfg or SarbonConfig()
        self.cfg, self.num_classes = cfg, num_classes
        d = cfg.hidden
        self.stem = BlockLocalStem(cfg.block, cfg.widths, cfg.depths)
        self.core = DenseCore(cfg)
        # encoder proposal heads (query selection)
        self.enc_proj = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d))
        self.enc_cls = nn.Linear(d, num_classes)
        self.enc_box = mlp(d, d, 4, 3)
        # decoder
        self.layers = nn.ModuleList(DecoderLayer(cfg) for _ in range(cfg.dec_layers))
        self.query_pos = mlp(4, 2 * d, d, 2)
        self.cls_heads = nn.ModuleList(nn.Linear(d, num_classes) for _ in range(cfg.dec_layers))
        self.box_heads = nn.ModuleList(mlp(d, d, 4, 3) for _ in range(cfg.dec_layers))
        self._reset()

    def _reset(self):
        prior = -math.log((1 - 0.01) / 0.01)
        for head in [self.enc_cls, *self.cls_heads]:
            nn.init.constant_(head.bias, prior)
        for head in [self.enc_box, *self.box_heads]:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    # ---------------- features ----------------
    def features(self, img):
        if self.cfg.halo_zero:
            _, p8 = self.stem(img)
        else:
            _, p8 = self.stem.forward_blocks(img)
        return self.core(p8)

    @staticmethod
    def anchors(shapes, device, dtype):
        """Cell-centre anchors (logit space), size 0.05 * 2^level; invalid near the border."""
        out = []
        for lvl, (h, w) in enumerate(shapes):
            y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device),
                                  indexing="ij")
            xy = torch.stack([(x + 0.5) / w, (y + 0.5) / h], -1).view(-1, 2)
            wh = torch.full_like(xy, 0.05 * 2.0 ** lvl)
            out.append(torch.cat([xy, wh], -1))
        a = torch.cat(out)[None]
        valid = ((a > 0.01) & (a < 0.99)).all(-1, keepdim=True)
        return inverse_sigmoid(a).masked_fill(~valid, float("inf")).to(dtype), valid

    def forward(self, img, tracks=None):
        """img [B, 3, H, W] normalised. tracks: optional {"embed": [B, T, D], "boxes": [B, T, 4]}.

        Returns a dict with per-layer predictions (last layer first-class), the encoder
        proposals (for their own loss) and `embed`/`boxes` to feed back as tracks.
        Queries [0, T) are the track queries, the rest are new.
        """
        feats = self.features(img)
        shapes = [f.shape[-2:] for f in feats]
        memory = torch.cat([f.flatten(2).transpose(1, 2) for f in feats], 1)  # [B, S, D]
        anchors, valid = self.anchors(shapes, memory.device, memory.dtype)

        enc = self.enc_proj(memory * valid.to(memory.dtype))
        enc_logits = self.enc_cls(enc)
        enc_boxes_logit = self.enc_box(enc) + anchors
        k = self.cfg.queries
        top = enc_logits.max(-1).values.masked_fill(~valid[..., 0], float("-inf")).topk(k, 1).indices
        idx = top[..., None]
        ref_logit = enc_boxes_logit.gather(1, idx.expand(-1, -1, 4))
        out = {"enc_logits": enc_logits.gather(1, idx.expand(-1, -1, self.num_classes)),
               "enc_boxes": ref_logit.sigmoid()}
        tgt = enc.gather(1, idx.expand(-1, -1, enc.shape[-1])).detach()
        ref = ref_logit.sigmoid().detach()

        n_track = 0
        if tracks is not None and tracks["embed"].shape[1]:
            n_track = tracks["embed"].shape[1]
            tgt = torch.cat([tracks["embed"], tgt], 1)
            ref = torch.cat([tracks["boxes"].detach(), ref], 1)

        logits, boxes = [], []
        for layer, cls_head, box_head in zip(self.layers, self.cls_heads, self.box_heads):
            tgt = layer(tgt, self.query_pos(ref), ref, memory, shapes)
            new_ref = (box_head(tgt) + inverse_sigmoid(ref)).sigmoid()
            logits.append(cls_head(tgt))
            boxes.append(new_ref)
            ref = new_ref.detach()
        out.update(pred_logits=logits[-1], pred_boxes=boxes[-1], n_track=n_track,
                   aux=[{"pred_logits": l, "pred_boxes": b} for l, b in zip(logits[:-1], boxes[:-1])],
                   embed=tgt, boxes=boxes[-1])
        return out


def build(size="s", num_classes=6, **overrides):
    cfg = SarbonConfig(**{**SIZES[size].__dict__, **overrides})
    return SarbonDetector(num_classes, cfg)
