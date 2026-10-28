"""Multi-scale deformable attention (Deformable DETR, 2010.04159) in plain PyTorch.

Box-relative sampling as in RT-DETR: offsets are scaled by the reference box size,
so a query looks around its own box. grid_sample keeps it portable (CPU, ONNX); a
fused kernel can replace `ms_deform_attn_core` later without changing weights.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def ms_deform_attn_core(value, shapes, loc, weights):
    """value [B, S, H, Dh]; loc [B, Q, H, L, P, 2] in [0, 1]; weights [B, Q, H, L, P]."""
    b, _, nh, dh = value.shape
    _, q, _, nl, npnt, _ = loc.shape
    splits = value.split([h * w for h, w in shapes], 1)
    grids = 2 * loc - 1
    sampled = []
    for lvl, (h, w) in enumerate(shapes):
        v = splits[lvl].permute(0, 2, 3, 1).reshape(b * nh, dh, h, w)
        g = grids[:, :, :, lvl].permute(0, 2, 1, 3, 4).reshape(b * nh, q, npnt, 2)
        sampled.append(F.grid_sample(v, g, mode="bilinear", padding_mode="zeros", align_corners=False))
    s = torch.stack(sampled, -2).flatten(-2)  # [B*H, Dh, Q, L*P]
    w = weights.permute(0, 2, 1, 3, 4).reshape(b * nh, 1, q, nl * npnt)
    out = (s * w).sum(-1)  # [B*H, Dh, Q]
    return out.view(b, nh * dh, q).transpose(1, 2)


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_levels=3, n_points=4):
        super().__init__()
        self.h, self.l, self.p = n_heads, n_levels, n_points
        self.offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attn = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self._reset()

    def _reset(self):
        nn.init.zeros_(self.offsets.weight)
        theta = torch.arange(self.h, dtype=torch.float32) * (2 * math.pi / self.h)
        grid = torch.stack([theta.cos(), theta.sin()], -1)
        grid = grid / grid.abs().max(-1, keepdim=True).values
        grid = grid.view(self.h, 1, 1, 2).repeat(1, self.l, self.p, 1)
        grid = grid * torch.arange(1, self.p + 1, dtype=torch.float32).view(1, 1, -1, 1)
        with torch.no_grad():
            self.offsets.bias.copy_(grid.flatten())
        nn.init.zeros_(self.attn.weight)
        nn.init.zeros_(self.attn.bias)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, query, ref_boxes, value, shapes):
        """query [B, Q, D]; ref_boxes [B, Q, 4] cxcywh in [0, 1]; value [B, S, D]."""
        b, q, d = query.shape
        v = self.value_proj(value).view(b, -1, self.h, d // self.h)
        off = self.offsets(query).view(b, q, self.h, self.l, self.p, 2)
        w = self.attn(query).view(b, q, self.h, self.l * self.p).softmax(-1)
        w = w.view(b, q, self.h, self.l, self.p)
        ref = ref_boxes[:, :, None, None, None]
        loc = ref[..., :2] + off / self.p * ref[..., 2:] * 0.5
        return self.out_proj(ms_deform_attn_core(v, shapes, loc, w))
