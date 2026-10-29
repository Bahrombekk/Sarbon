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

from .block_local import BLOCKS, BlockLocalStem, conv_bn_act
from .box_ops import distance2bbox, fdr_project, inverse_sigmoid
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
    reg_max: int = 32  # FDR bins per box edge (reg_max + 1 values)
    reg_scale: float = 4.0
    block_type: str = "res"  # "res" | "rep" (RepVGG, fused for inference) | "dw" (inverted bottleneck)
    act: str = "relu"  # "relu" | "silu" | "gelu"
    fusion: str = "concat"  # neck fusion: "concat" | "weighted" (BiFPN) | "gated" (per-pixel)
    num_denoising: int = 100  # contrastive denoising queries during training (0 = off)
    label_noise: float = 0.5
    box_noise: float = 1.0


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


class DynamicFuse(nn.Module):
    """Fuse two same-width feature maps with learned weights, then mix (1x1 + dw 3x3).

    weighted: BiFPN fast normalised fusion (1911.09070) -- one learned, non-negative
              weight per input, shared over the whole map;
    gated   : a per-pixel softmax gate (1x1 conv on both inputs) decides, location by
              location, how much to trust each scale; starts as a plain average.
    """

    def __init__(self, d, mode):
        super().__init__()
        self.mode = mode
        if mode == "weighted":
            self.w = nn.Parameter(torch.ones(2))
        else:
            self.gate = nn.Conv2d(2 * d, 2, 1)
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)
        self.mix = fuse(d, d)

    def weights(self, a, b):
        if self.mode == "weighted":
            w = F.relu(self.w)
            return (w / (w.sum() + 1e-4)).view(1, 2, 1, 1).expand(a.shape[0], 2, *a.shape[-2:])
        return self.gate(torch.cat([a, b], 1)).softmax(1)

    def forward(self, a, b):
        w = self.weights(a, b)
        return self.mix(w[:, :1] * a + w[:, 1:] * b)


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
        unit, a = BLOCKS[cfg.block_type], cfg.act
        self.to16 = nn.Sequential(conv_bn_act(c8, cfg.c16, 2, act_name=a), unit(cfg.c16, a), unit(cfg.c16, a))
        self.to32 = nn.Sequential(conv_bn_act(cfg.c16, cfg.c32, 2, act_name=a), unit(cfg.c32, a))
        self.in8, self.in16, self.in32 = (nn.Sequential(nn.Conv2d(c, d, 1, bias=False), nn.BatchNorm2d(d))
                                          for c in (c8, cfg.c16, cfg.c32))
        self.aifi = nn.TransformerEncoderLayer(d, cfg.heads, cfg.ffn, 0.0, batch_first=True,
                                               activation="gelu")
        # "concat" keeps the original module layout so existing checkpoints still load
        self.fusion = cfg.fusion
        mk = (lambda: fuse(2 * d, d)) if cfg.fusion == "concat" else (lambda: DynamicFuse(d, cfg.fusion))
        self.td16, self.td8 = mk(), mk()
        self.down8, self.down16 = conv_bn_act(d, d, 2), conv_bn_act(d, d, 2)
        self.bu16, self.bu32 = mk(), mk()

    def _fuse(self, m, a, b):
        return m(torch.cat([a, b], 1)) if self.fusion == "concat" else m(a, b)

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
        t16 = self._fuse(self.td16, x16, F.interpolate(x32, scale_factor=2.0))
        o8 = self._fuse(self.td8, x8, F.interpolate(t16, scale_factor=2.0))
        o16 = self._fuse(self.bu16, t16, self.down8(o8))
        o32 = self._fuse(self.bu32, x32, self.down16(o16))
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
        self.stem = BlockLocalStem(cfg.block, cfg.widths, cfg.depths, cfg.block_type, cfg.act)
        self.core = DenseCore(cfg)
        # encoder proposal heads (query selection)
        self.enc_proj = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d))
        self.enc_cls = nn.Linear(d, num_classes)
        self.enc_box = mlp(d, d, 4, 3)
        # decoder
        self.layers = nn.ModuleList(DecoderLayer(cfg) for _ in range(cfg.dec_layers))
        self.query_pos = mlp(4, 2 * d, d, 2)
        self.cls_heads = nn.ModuleList(nn.Linear(d, num_classes) for _ in range(cfg.dec_layers))
        # FDR: layer 0 sets an initial box; every layer then refines per-edge distributions
        self.pre_box = mlp(d, d, 4, 3)
        self.corner_heads = nn.ModuleList(mlp(d, d, 4 * (cfg.reg_max + 1), 3)
                                          for _ in range(cfg.dec_layers))
        self.register_buffer("project", fdr_project(cfg.reg_max, 0.5, cfg.reg_scale), persistent=False)
        self.dn_label = nn.Embedding(num_classes + 1, d)  # last row: padding
        self._reset()

    def _reset(self):
        prior = -math.log((1 - 0.01) / 0.01)
        for head in [self.enc_cls, *self.cls_heads]:
            nn.init.constant_(head.bias, prior)
        for head in [self.enc_box, self.pre_box, *self.corner_heads]:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def denoising(self, targets, device):
        """Contrastive denoising queries (DINO, 2203.03605), RT-DETR style.

        Each group holds a positive (small box noise, must reconstruct the object) and a
        negative (large noise, must say "no object") copy of every ground truth.
        Returns content [B, N, D], boxes [B, N, 4], matches per image, groups, N.
        """
        cfg = self.cfg
        n_gt = [len(t["labels"]) for t in targets]
        mx = max(n_gt) if n_gt else 0
        if mx == 0 or cfg.num_denoising == 0:
            return None
        groups = max(1, cfg.num_denoising // (2 * mx))
        b = len(targets)
        labels = torch.full((b, mx), self.num_classes, dtype=torch.long, device=device)
        boxes = torch.zeros(b, mx, 4, device=device)
        valid = torch.zeros(b, mx, dtype=torch.bool, device=device)
        for i, t in enumerate(targets):
            labels[i, :n_gt[i]] = t["labels"]
            boxes[i, :n_gt[i]] = t["boxes"]
            valid[i, :n_gt[i]] = True
        labels = labels.repeat(1, 2 * groups)  # layout: [g0 pos, g0 neg, g1 pos, g1 neg, ...]
        boxes = boxes.repeat(1, 2 * groups, 1)
        valid = valid.repeat(1, 2 * groups)
        neg = torch.zeros(b, mx * 2 * groups, dtype=torch.bool, device=device)
        neg.view(b, groups, 2, mx)[:, :, 1] = True
        if cfg.label_noise > 0:
            flip = torch.rand(labels.shape, device=device) < cfg.label_noise * 0.5
            labels = torch.where(flip & valid, torch.randint_like(labels, self.num_classes), labels)
        if cfg.box_noise > 0:
            xyxy = torch.cat([boxes[..., :2] - boxes[..., 2:] / 2, boxes[..., :2] + boxes[..., 2:] / 2], -1)
            diff = torch.cat([boxes[..., 2:] / 2, boxes[..., 2:] / 2], -1) * cfg.box_noise
            sign = torch.randint_like(xyxy, 0, 2) * 2 - 1
            r = torch.rand_like(xyxy) + neg[..., None].float()  # |r| < 1 positives, in [1, 2) negatives
            xyxy = (xyxy + sign * r * diff).clamp(0, 1)
            boxes = torch.cat([(xyxy[..., :2] + xyxy[..., 2:]) / 2, (xyxy[..., 2:] - xyxy[..., :2]).clamp(min=1e-4)], -1)
        content = self.dn_label(labels.masked_fill(~valid, self.num_classes))
        matches = []
        for i in range(b):
            q = torch.cat([torch.arange(n_gt[i], device=device) + g * 2 * mx for g in range(groups)])
            gidx = torch.arange(n_gt[i], device=device).repeat(groups)
            matches.append((q.cpu(), gidx.cpu()))
        return content, boxes.clamp(1e-4, 1 - 1e-4), matches, groups, mx * 2 * groups

    @staticmethod
    def dn_attn_mask(n_dn, n_match, group_size, device):
        """True = blocked. Matching queries never see denoising ones; groups are isolated."""
        n = n_dn + n_match
        mask = torch.zeros(n, n, dtype=torch.bool, device=device)
        mask[n_dn:, :n_dn] = True
        for g in range(n_dn // group_size):
            s, e = g * group_size, (g + 1) * group_size
            mask[s:e, :s] = True
            mask[s:e, e:n_dn] = True
        return mask

    def integral(self, corners):
        """[B, Q, 4*(R+1)] logits -> expected edge offsets [B, Q, 4] in bin-value units."""
        p = corners.view(*corners.shape[:-1], 4, -1).softmax(-1)
        return p @ self.project.to(p.dtype)

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

    def forward(self, img, tracks=None, targets=None, p8=None):
        """img [B, 3, H, W] normalised. tracks: optional {"embed": [B, T, D], "boxes": [B, T, 4]}.
        targets (training only): enables contrastive denoising queries.
        p8: stride-8 stem features computed elsewhere (the block-sparse video engine);
        when given, `img` is not used.

        Returns a dict with per-layer predictions (last layer first-class), the encoder
        proposals (for their own loss) and `embed`/`boxes` to feed back as tracks.
        Queries [0, T) are the track queries, the rest are new; denoising outputs, if
        any, are returned separately under "dn".
        """
        feats = self.features(img) if p8 is None else self.core(p8)
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

        n_dn, attn_mask, dn = 0, None, None
        if self.training and targets is not None:
            dn = self.denoising(targets, tgt.device)
            if dn is not None:
                dn_content, dn_boxes, dn_matches, groups, n_dn = dn
                tgt = torch.cat([dn_content, tgt], 1)
                ref = torch.cat([dn_boxes, ref], 1)
                attn_mask = self.dn_attn_mask(n_dn, tgt.shape[1] - n_dn, n_dn // groups, tgt.device)

        logits, boxes, corners_all = [], [], []
        corners = tgt_detach = ref_init = pre = pre_logits = None
        for i, (layer, cls_head, corner_head) in enumerate(zip(self.layers, self.cls_heads,
                                                               self.corner_heads)):
            tgt = layer(tgt, self.query_pos(ref), ref, memory, shapes, attn_mask)
            if i == 0:
                pre = (self.pre_box(tgt) + inverse_sigmoid(ref)).sigmoid()
                pre_logits = cls_head(tgt)
                ref_init = pre.detach()
                corners = corner_head(tgt)
            else:
                # residual refinement of the distributions (gradient flows through `tgt`
                # once directly and once via the previous layers' corners)
                corners = corner_head(tgt + tgt_detach) + corners
            box = distance2bbox(ref_init, self.integral(corners), self.cfg.reg_scale)
            logits.append(cls_head(tgt))
            boxes.append(box)
            corners_all.append(corners)
            ref, tgt_detach = box.detach(), tgt.detach()

        def pack(sl):
            return {"pred_logits": logits[-1][:, sl], "pred_boxes": boxes[-1][:, sl],
                    "pred_corners": corners_all[-1][:, sl], "ref_init": ref_init[:, sl],
                    "pre": {"pred_logits": pre_logits[:, sl], "pred_boxes": pre[:, sl]},
                    "aux": [{"pred_logits": l[:, sl], "pred_boxes": b[:, sl], "pred_corners": c[:, sl]}
                            for l, b, c in zip(logits[:-1], boxes[:-1], corners_all[:-1])]}

        out.update(pack(slice(n_dn, None)), n_track=n_track, embed=tgt[:, n_dn:], boxes=boxes[-1][:, n_dn:])
        if n_dn:
            out["dn"] = {**pack(slice(0, n_dn)), "matches": dn_matches, "groups": groups}
        return out


def build(size="s", num_classes=6, **overrides):
    cfg = SarbonConfig(**{**SIZES[size].__dict__, **overrides})
    return SarbonDetector(num_classes, cfg)
