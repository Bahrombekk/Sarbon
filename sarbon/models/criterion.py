"""Set matching and losses for the Sarbon detector.

Matching cost = focal class cost + L1 + (-GIoU) [+ (1 - NWD) for small objects].
Track queries (indices [0, T)) never enter the Hungarian step: each one is bound to
the ground truth with its track ID, or to background if that object left the scene.
Losses: sigmoid focal (class), L1 + GIoU (boxes), on every decoder layer and on the
encoder proposals.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .box_ops import cxcywh_to_xyxy, generalized_box_iou, nwd


class HungarianMatcher(nn.Module):
    def __init__(self, w_cls=2.0, w_l1=5.0, w_giou=2.0, w_nwd=0.0, nwd_c=0.02, alpha=0.25, gamma=2.0):
        super().__init__()
        self.w_cls, self.w_l1, self.w_giou, self.w_nwd = w_cls, w_l1, w_giou, w_nwd
        self.nwd_c, self.alpha, self.gamma = nwd_c, alpha, gamma

    @torch.no_grad()
    def forward(self, logits, boxes, targets, fixed=None, skip=0):
        """logits [B, Q, C], boxes [B, Q, 4]; targets: list of {"labels", "boxes"(cxcywh)}.

        fixed: optional per-image list of (query_idx, gt_idx) pairs decided beforehand
        (track queries). Queries [0, skip) are excluded from the Hungarian step.
        Returns per-image (query_idx, gt_idx) LongTensors.
        """
        out = []
        for i, t in enumerate(targets):
            fq, fg = (fixed[i] if fixed else (torch.empty(0, dtype=torch.long),) * 2)
            n_gt = len(t["labels"])
            free_gt = torch.ones(n_gt, dtype=torch.bool)
            free_gt[fg] = False
            gt_idx = free_gt.nonzero()[:, 0]
            q_idx = torch.arange(skip, logits.shape[1])
            if len(gt_idx) == 0:
                out.append((fq, fg))
                continue
            p = logits[i, q_idx].float().sigmoid()
            lab = t["labels"][gt_idx]
            pos = self.alpha * (1 - p) ** self.gamma * -(p + 1e-8).log()
            neg = (1 - self.alpha) * p ** self.gamma * -(1 - p + 1e-8).log()
            c = self.w_cls * (pos[:, lab] - neg[:, lab])
            pb, gb = boxes[i, q_idx].float(), t["boxes"][gt_idx].float()
            c = c + self.w_l1 * torch.cdist(pb, gb, p=1)
            c = c - self.w_giou * generalized_box_iou(cxcywh_to_xyxy(pb), cxcywh_to_xyxy(gb))
            if self.w_nwd:
                c = c + self.w_nwd * (1 - nwd(pb, gb, self.nwd_c))
            r, col = linear_sum_assignment(c.cpu().numpy())
            out.append((torch.cat([fq, q_idx[r]]), torch.cat([fg, gt_idx[col]])))
        return out


def sigmoid_focal(logits, target, alpha=0.25, gamma=2.0):
    p = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pt = p * target + (1 - p) * (1 - target)
    return (ce * (1 - pt) ** gamma * (alpha * target + (1 - alpha) * (1 - target))).sum()


class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher=None, w_cls=2.0, w_l1=5.0, w_giou=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher or HungarianMatcher()
        self.w = {"cls": w_cls, "l1": w_l1, "giou": w_giou}

    def _single(self, logits, boxes, targets, matches, n_norm):
        b, q, c = logits.shape
        tcls = torch.zeros_like(logits, dtype=torch.float32)
        src_b, tgt_b = [], []
        for i, (qi, gi) in enumerate(matches):
            if len(qi):
                tcls[i, qi, targets[i]["labels"][gi]] = 1.0
                src_b.append(boxes[i, qi])
                tgt_b.append(targets[i]["boxes"][gi])
        loss = {"cls": sigmoid_focal(logits.float(), tcls) / n_norm}
        if src_b:
            sb, tb = torch.cat(src_b).float(), torch.cat(tgt_b).float()
            loss["l1"] = F.l1_loss(sb, tb, reduction="sum") / n_norm
            loss["giou"] = (1 - torch.diag(generalized_box_iou(cxcywh_to_xyxy(sb), cxcywh_to_xyxy(tb)))).sum() / n_norm
        else:
            zero = boxes.sum() * 0
            loss["l1"] = loss["giou"] = zero
        return loss

    def forward(self, out, targets, fixed=None):
        """out: SarbonDetector output; fixed: track-query assignments (see matcher)."""
        skip = out.get("n_track", 0)
        n_norm = max(1.0, float(sum(len(t["labels"]) for t in targets)))
        total, logs = 0.0, {}
        layers = [*out["aux"], {"pred_logits": out["pred_logits"], "pred_boxes": out["pred_boxes"]}]
        final_matches = None
        for li, o in enumerate(layers):
            m = self.matcher(o["pred_logits"], o["pred_boxes"], targets, fixed, skip)
            if li == len(layers) - 1:
                final_matches = m
            for k, v in self._single(o["pred_logits"], o["pred_boxes"], targets, m, n_norm).items():
                total = total + self.w[k] * v
                if li == len(layers) - 1:
                    logs[k] = v.item()
        # encoder proposals: new objects only (no track queries there)
        m = self.matcher(out["enc_logits"], out["enc_boxes"], targets)
        for k, v in self._single(out["enc_logits"], out["enc_boxes"], targets, m, n_norm).items():
            total = total + self.w[k] * v
        return total, logs, final_matches
