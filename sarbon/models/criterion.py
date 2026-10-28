"""Set matching and losses for the Sarbon detector.

Matching cost = focal class cost + L1 + (-GIoU) [+ (1 - NWD) for small objects].
Track queries (indices [0, T)) never enter the Hungarian step: each one is bound to
the ground truth with its track ID, or to background if that object left the scene.
Losses, on every decoder layer, the pre-box layer and the encoder proposals:
- class: MAL (DEIM, 2412.04234) -- the positive target is IoU^gamma, so a query is
  rewarded for confidence in proportion to how well its box matches; or sigmoid focal;
- boxes: L1 + GIoU, plus FGL (D-FINE): per-edge distribution focal loss towards the
  two bins around the target edge, weighted by IoU;
- DDF (D-FINE's GO-LSD): earlier layers' edge distributions distil from the final one.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .box_ops import bbox2distance, box_iou, cxcywh_to_xyxy, generalized_box_iou, nwd


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


def mal_loss(logits, target_q, pos_mask, gamma=1.5):
    """Matchability-Aware Loss: pos -> BCE towards q^gamma; neg -> p^gamma-weighted BCE to 0."""
    p = logits.sigmoid().detach()
    t = torch.where(pos_mask, target_q.pow(gamma), torch.zeros_like(target_q))
    w = torch.where(pos_mask, torch.ones_like(p), p.pow(gamma))
    return (F.binary_cross_entropy_with_logits(logits, t, reduction="none") * w).sum()


class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher=None, cls_loss="mal", w_cls=1.0, w_l1=5.0, w_giou=2.0,
                 w_fgl=0.15, w_ddf=1.5, reg_scale=4.0, project=None, ddf_t=5.0):
        super().__init__()
        self.num_classes, self.cls_loss = num_classes, cls_loss
        self.matcher = matcher or HungarianMatcher()
        self.w = {"cls": w_cls, "l1": w_l1, "giou": w_giou, "fgl": w_fgl, "ddf": w_ddf}
        self.reg_scale, self.ddf_t = reg_scale, ddf_t
        self.project = project

    @staticmethod
    def _gather(matches, *tensors):
        """Concatenate matched rows of [B, Q, ...] tensors over the batch."""
        return [torch.cat([t[i, qi] for i, (qi, _) in enumerate(matches)]) for t in tensors]

    def _single(self, o, targets, matches, n_norm, ref_init=None):
        logits, boxes = o["pred_logits"], o["pred_boxes"]
        tgt_b = torch.cat([t["boxes"][gi] for t, (_, gi) in zip(targets, matches)]).float()
        sb = self._gather(matches, boxes)[0].float()
        has = len(sb) > 0
        iou = box_iou(cxcywh_to_xyxy(sb), cxcywh_to_xyxy(tgt_b))[0].diag().detach() if has else sb.new_zeros(0)

        pos = torch.zeros_like(logits, dtype=torch.bool)
        q = torch.zeros_like(logits, dtype=torch.float32)
        k = 0
        for i, (qi, gi) in enumerate(matches):
            lab = targets[i]["labels"][gi]
            pos[i, qi, lab] = True
            q[i, qi, lab] = iou[k:k + len(qi)]
            k += len(qi)
        if self.cls_loss == "mal":
            loss = {"cls": mal_loss(logits.float(), q, pos) / n_norm}
        else:
            loss = {"cls": sigmoid_focal(logits.float(), pos.float()) / n_norm}
        if not has:
            zero = boxes.sum() * 0
            loss.update(l1=zero, giou=zero)
            return loss, iou
        loss["l1"] = F.l1_loss(sb, tgt_b, reduction="sum") / n_norm
        loss["giou"] = (1 - generalized_box_iou(cxcywh_to_xyxy(sb), cxcywh_to_xyxy(tgt_b)).diag()).sum() / n_norm
        if "pred_corners" in o and ref_init is not None:
            corners, ref = self._gather(matches, o["pred_corners"], ref_init)
            corners = corners.float().view(-1, 4, self.project.numel())
            idx, wl, wr = bbox2distance(ref.float(), tgt_b, self.reg_scale, self.project)
            logp = corners.log_softmax(-1)
            dfl = -(logp.gather(-1, idx[..., None])[..., 0] * wl + logp.gather(-1, idx[..., None] + 1)[..., 0] * wr)
            loss["fgl"] = (dfl.mean(-1) * iou).sum() / n_norm
        return loss, iou

    def _ddf(self, layers, final, matches, iou, n_norm):
        """GO-LSD: matched queries' edge distributions in earlier layers follow the final layer."""
        t = self.ddf_t
        tgt = self._gather(matches, final["pred_corners"])[0].float().detach()
        tgt = tgt.view(-1, 4, self.project.numel())
        total = 0.0
        for o in layers:
            src = self._gather(matches, o["pred_corners"])[0].float().view_as(tgt)
            kl = F.kl_div((src / t).log_softmax(-1), (tgt / t).softmax(-1), reduction="none").sum(-1).mean(-1)
            total = total + (kl * iou).sum() * t * t / n_norm
        return total

    def forward(self, out, targets, fixed=None):
        """out: SarbonDetector output; fixed: track-query assignments (see matcher)."""
        skip = out.get("n_track", 0)
        n_norm = max(1.0, float(sum(len(t["labels"]) for t in targets)))
        total, logs = 0.0, {}
        final = {k: out[k] for k in ("pred_logits", "pred_boxes", "pred_corners") if k in out}
        layers = [*out["aux"], final]
        ref_init = out.get("ref_init")
        final_matches = self.matcher(final["pred_logits"], final["pred_boxes"], targets, fixed, skip)
        final_iou = None
        for li, o in enumerate(layers):
            last = li == len(layers) - 1
            m = final_matches if last else self.matcher(o["pred_logits"], o["pred_boxes"], targets, fixed, skip)
            parts, iou = self._single(o, targets, m, n_norm, ref_init)
            for k, v in parts.items():
                total = total + self.w[k] * v
                if last:
                    logs[k] = v.item()
            if last:
                final_iou = iou
        if "pred_corners" in final and len(out["aux"]) and final_iou is not None and len(final_iou):
            ddf = self._ddf(out["aux"], final, final_matches, final_iou, n_norm)
            total = total + self.w["ddf"] * ddf
            logs["ddf"] = ddf.item()
        # pre-box layer and encoder proposals (new objects only for the encoder)
        if "pre" in out:
            m = self.matcher(out["pre"]["pred_logits"], out["pre"]["pred_boxes"], targets, fixed, skip)
            for k, v in self._single(out["pre"], targets, m, n_norm)[0].items():
                total = total + self.w[k] * v
        enc = {"pred_logits": out["enc_logits"], "pred_boxes": out["enc_boxes"]}
        m = self.matcher(enc["pred_logits"], enc["pred_boxes"], targets)
        for k, v in self._single(enc, targets, m, n_norm)[0].items():
            total = total + self.w[k] * v
        if "dn" in out:  # denoising: assignments are known, no matching
            dn = out["dn"]
            m, norm = dn["matches"], n_norm * dn["groups"]
            dn_layers = [*dn["aux"], {k: dn[k] for k in ("pred_logits", "pred_boxes", "pred_corners")}, dn["pre"]]
            for o in dn_layers:
                for k, v in self._single(o, targets, m, norm, dn["ref_init"])[0].items():
                    total = total + self.w[k] * v
            logs["dn"] = sum(self._single(dn_layers[-2], targets, m, norm)[0].values()).item()
        return total, logs, final_matches
