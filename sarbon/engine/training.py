"""Training building blocks shared by every training entry point (detector, tracking, distillation)."""
import copy
import math

import torch

from ..models.box_ops import xyxy_to_cxcywh

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255


def prep(imgs, device):
    x = imgs.to(device, non_blocking=True).float()
    return (x - MEAN.to(device)) / STD.to(device)


def to_targets(boxes, labels, size, device, meta=None):
    out = [{"labels": l.to(device), "boxes": xyxy_to_cxcywh(b.to(device) / size).clamp(0, 1)}
           for b, l in zip(boxes, labels)]
    for t, m in zip(out, meta or []):
        if torch.is_tensor(m[-1]):  # WithClassMask: classes annotated in this image
            t["valid"] = m[-1].to(device)
    return out


CLASS_KEYS = ("enc_cls.", "cls_heads.", "dn_label.")


def load_init(model, path, src_names=None, dst_names=None, alias=None):
    """Load every tensor whose shape fits. Class-indexed tensors are taken only when the
    source class names are given: rows are then copied by name (alias: dst -> src name),
    and classes the source never saw keep their fresh initialisation."""
    sd = torch.load(path, map_location="cpu")
    own = model.state_dict()
    take, remapped = {}, 0
    for k, v in sd.items():
        if k not in own:
            continue
        if k.startswith(CLASS_KEYS):
            if not (src_names and dst_names):
                continue
            new = own[k].clone()
            for j, name in enumerate(dst_names):
                src = (alias or {}).get(name, name)
                if src in src_names:
                    new[j] = v[src_names.index(src)]
                    remapped += 1
            if k.startswith("dn_label."):  # last row: padding token
                new[-1] = v[-1]
            take[k] = new
        elif v.shape == own[k].shape:
            take[k] = v
    model.load_state_dict(take, strict=False)
    return len(take), remapped


class ModelEMA:
    def __init__(self, model, decay=0.9999, warmup=2000):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay, self.warmup, self.updates = decay, warmup, 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / self.warmup))
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)


def param_groups(model, wd):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or n.endswith(".bias") else decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def add_arch_args(ap):
    """Backbone overrides shared by the training CLIs (default: the size preset)."""
    ap.add_argument("--widths", help="stem widths at stride 2/4/8, e.g. 48,96,192")
    ap.add_argument("--depths", help="Res blocks at stride 4/8, e.g. 2,3")
    ap.add_argument("--c16", type=int, help="channels of the stride-16 stage")


def arch_overrides(args):
    out = {}
    if args.widths:
        out["widths"] = tuple(int(v) for v in str(args.widths).split(","))
    if args.depths:
        out["depths"] = tuple(int(v) for v in str(args.depths).split(","))
    if args.c16:
        out["c16"] = args.c16
    return out
