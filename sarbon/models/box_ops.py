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


# ---------------- Fine-grained Distribution Refinement (D-FINE, 2410.13842) ----------------
# Each box edge is a distribution over reg_max + 1 bins whose values W(n) are dense near
# 0 (fine corrections) and sparse at the ends (coarse jumps). Edges are measured from an
# initial box in units of (box size / reg_scale).

def fdr_project(reg_max=32, up=0.5, reg_scale=4.0):
    ub1 = abs(up) * abs(reg_scale)
    ub2 = ub1 * 2
    step = (ub1 + 1) ** (2 / (reg_max - 2))
    left = [-(step ** i) + 1 for i in range(reg_max // 2 - 1, 0, -1)]
    right = [step ** i - 1 for i in range(1, reg_max // 2)]
    return torch.tensor([-ub2] + left + [0.0] + right + [ub2])


def distance2bbox(points, dist, reg_scale):
    """points cxcywh [..., 4]; dist (left, top, right, bottom) in bin-value units -> cxcywh."""
    cx, cy, w, h = points.unbind(-1)
    sw, sh = w / reg_scale, h / reg_scale
    x0 = cx - (0.5 * reg_scale + dist[..., 0]) * sw
    y0 = cy - (0.5 * reg_scale + dist[..., 1]) * sh
    x1 = cx + (0.5 * reg_scale + dist[..., 2]) * sw
    y1 = cy + (0.5 * reg_scale + dist[..., 3]) * sh
    return xyxy_to_cxcywh(torch.stack([x0, y0, x1, y1], -1))


def bbox2distance(points, gt, reg_scale, project):
    """Target edges of `gt` w.r.t. `points` as (left bin index, left weight, right weight)."""
    cx, cy, w, h = points.unbind(-1)
    x0, y0, x1, y1 = cxcywh_to_xyxy(gt).unbind(-1)
    sw, sh = w / reg_scale + 1e-16, h / reg_scale + 1e-16
    d = torch.stack([(cx - x0) / sw, (cy - y0) / sh, (x1 - cx) / sw, (y1 - cy) / sh], -1) - 0.5 * reg_scale
    p = project.to(d)
    d = d.clamp(p[0], p[-1] - 1e-6)
    idx = (torch.searchsorted(p, d.contiguous(), right=True) - 1).clamp(0, len(p) - 2)
    lo, hi = p[idx], p[idx + 1]
    wr = ((d - lo) / (hi - lo)).clamp(0, 1)
    return idx, 1 - wr, wr


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(eps, 1 - eps)
    return torch.log(x / (1 - x))
