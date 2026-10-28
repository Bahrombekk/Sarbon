"""Small CenterNet-style detector used to measure what the halo-zero stem costs in accuracy.

Both variants share every weight shape; the only difference is whether the high-res
stem sees the whole image (standard) or each 32 px block on its own (halo-zero).
Dense core: stride 16/32 stages + top-down fusion back to stride 8, where a
heatmap / size / offset head predicts object centres.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .block_local import BlockLocalStem, Res, conv_bn_act


class DenseCore(nn.Module):
    def __init__(self, c8=128, c16=256, c32=256):
        super().__init__()
        self.to16 = nn.Sequential(conv_bn_act(c8, c16, 2), Res(c16), Res(c16))
        self.to32 = nn.Sequential(conv_bn_act(c16, c32, 2), Res(c32))
        self.lat16 = nn.Conv2d(c16, c8, 1)
        self.lat32 = nn.Conv2d(c32, c8, 1)
        self.fuse16 = conv_bn_act(c8, c8)
        self.fuse8 = conv_bn_act(c8, c8)

    def forward(self, p8):
        p16 = self.to16(p8)
        p32 = self.to32(p16)
        t16 = self.fuse16(self.lat16(p16) + F.interpolate(self.lat32(p32), scale_factor=2))
        return self.fuse8(p8 + F.interpolate(t16, scale_factor=2))


class ProxyDetector(nn.Module):
    def __init__(self, num_classes, halo_zero, block=32, c8=128):
        super().__init__()
        self.halo_zero = halo_zero
        self.stem = BlockLocalStem(block, widths=(32, 64, c8))
        self.core = DenseCore(c8)
        self.head = nn.Sequential(conv_bn_act(c8, c8), nn.Conv2d(c8, num_classes + 4, 1))
        nn.init.constant_(self.head[-1].bias[:num_classes], -2.19)  # prior 0.1 (CenterNet)
        self.num_classes = num_classes

    def forward(self, img):
        if self.halo_zero:
            _, p8 = self.stem(img)  # per-block encoding
        else:
            _, p8 = self.stem.forward_blocks(img)  # same layers over the whole image
        out = self.head(self.core(p8))
        return out[:, :self.num_classes], out[:, self.num_classes:]  # heat logits, (w, h, dx, dy)


# ---------------- targets, loss, decoding ----------------
STRIDE = 8


def gaussian_radius(w, h, min_overlap=0.7):
    """CenterNet radius so that a shifted box keeps IoU >= min_overlap (simplified)."""
    a = 4 * min_overlap
    b = 2 * min_overlap * (w + h)
    c = (min_overlap - 1) * w * h
    return max(0.0, (b + (b * b - 4 * a * c) ** 0.5) / (2 * a) * 0.5)


def build_targets(boxes, labels, num_classes, out_h, out_w):
    """boxes [M, 4] xyxy in input pixels -> heatmap, regression targets and their mask."""
    heat = torch.zeros(num_classes, out_h, out_w)
    reg = torch.zeros(4, out_h, out_w)
    mask = torch.zeros(out_h, out_w, dtype=torch.bool)
    for (x0, y0, x1, y1), c in zip(boxes.tolist(), labels.tolist()):
        w, h = (x1 - x0) / STRIDE, (y1 - y0) / STRIDE
        if w <= 0 or h <= 0:
            continue
        cx, cy = (x0 + x1) / 2 / STRIDE, (y0 + y1) / 2 / STRIDE
        ix = min(max(int(cx), 0), out_w - 1)
        iy = min(max(int(cy), 0), out_h - 1)
        sigma = max(gaussian_radius(w, h), 0.5) / 3 + 0.3
        r = int(3 * sigma) + 1  # draw only the local window (fast enough for loader workers)
        ya, yb, xa, xb = max(0, iy - r), min(out_h, iy + r + 1), max(0, ix - r), min(out_w, ix + r + 1)
        yy = torch.arange(ya, yb).view(-1, 1) - iy
        xx = torch.arange(xa, xb).view(1, -1) - ix
        g = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        heat[c, ya:yb, xa:xb] = torch.maximum(heat[c, ya:yb, xa:xb], g)
        reg[:, iy, ix] = torch.tensor([w, h, cx - ix, cy - iy])
        mask[iy, ix] = True
    return heat, reg, mask


def focal_loss(logits, target):
    """CenterNet penalty-reduced focal loss."""
    p = logits.sigmoid().clamp(1e-4, 1 - 1e-4)
    pos = target.eq(1).float()
    neg = 1 - pos
    pos_loss = (torch.log(p) * (1 - p) ** 2 * pos).sum()
    neg_loss = (torch.log(1 - p) * p ** 2 * (1 - target) ** 4 * neg).sum()
    return -(pos_loss + neg_loss) / pos.sum().clamp(min=1)


def detection_loss(heat_logits, reg_pred, heat, reg, mask):
    loss_h = focal_loss(heat_logits.float(), heat)
    m = mask[:, None].expand_as(reg_pred)
    n = mask.sum().clamp(min=1)
    loss_wh = F.l1_loss(reg_pred[:, :2].float()[m[:, :2]], reg[:, :2][m[:, :2]], reduction="sum") / n
    loss_off = F.l1_loss(reg_pred[:, 2:].float()[m[:, 2:]], reg[:, 2:][m[:, 2:]], reduction="sum") / n
    return loss_h + 0.1 * loss_wh + loss_off, {"heat": loss_h.item(), "wh": loss_wh.item(),
                                                 "off": loss_off.item()}


@torch.no_grad()
def decode(heat_logits, reg_pred, k=100):
    """-> per image list of (boxes xyxy input px [k, 4], scores [k], labels [k])."""
    heat = heat_logits.sigmoid()
    keep = F.max_pool2d(heat, 3, 1, 1) == heat
    heat = heat * keep
    n, c, h, w = heat.shape
    scores, idx = heat.reshape(n, -1).topk(k)  # reshape: outputs may be channels_last
    labels = idx // (h * w)
    pix = idx % (h * w)
    ys, xs = (pix // w).float(), (pix % w).float()
    out = []
    for i in range(n):
        r = reg_pred[i].reshape(4, -1)[:, pix[i]].float()
        cx, cy = xs[i] + r[2], ys[i] + r[3]
        boxes = torch.stack([cx - r[0] / 2, cy - r[1] / 2, cx + r[0] / 2, cy + r[1] / 2], 1) * STRIDE
        out.append((boxes, scores[i], labels[i]))
    return out
