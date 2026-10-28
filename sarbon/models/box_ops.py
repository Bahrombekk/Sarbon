"""Box utilities. Boxes are normalised to [0, 1] unless stated otherwise."""
import torch


def cxcywh_to_xyxy(b):
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)


def xyxy_to_cxcywh(b):
    x0, y0, x1, y1 = b.unbind(-1)
    return torch.stack([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0], -1)


def box_area(b):
    return (b[..., 2] - b[..., 0]).clamp(min=0) * (b[..., 3] - b[..., 1]).clamp(min=0)


def box_iou(a, b):
    """Pairwise IoU of xyxy boxes [N, 4] x [M, 4] -> ([N, M] iou, [N, M] union)."""
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    union = box_area(a)[:, None] + box_area(b)[None] - inter
    return inter / union.clamp(min=1e-7), union


def generalized_box_iou(a, b):
    iou, union = box_iou(a, b)
    lt = torch.min(a[:, None, :2], b[None, :, :2])
    rb = torch.max(a[:, None, 2:], b[None, :, 2:])
    hull = (rb - lt).clamp(min=0).prod(-1)
    return iou - (hull - union) / hull.clamp(min=1e-7)


def nwd(a, b, c):
    """Pairwise Normalized Wasserstein Distance of cxcywh boxes (Wang et al., 2110.13389).

    Boxes are 2-D Gaussians N((cx, cy), diag(w^2/4, h^2/4)); `c` is a size constant in
    the same units as the boxes. Unlike IoU it stays smooth for tiny, non-overlapping
    boxes, which is why it helps small-object matching.
    """
    d = a[:, None, :] - b[None, :, :]
    w2 = d[..., 0] ** 2 + d[..., 1] ** 2 + (d[..., 2] / 2) ** 2 + (d[..., 3] / 2) ** 2
    return torch.exp(-w2.clamp(min=0).sqrt() / c)


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(eps, 1 - eps)
    return torch.log(x / (1 - x))
