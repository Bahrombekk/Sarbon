"""Set matching and losses for the Sarbon detector (vectorised over the batch).

Matching cost = focal class cost + L1 + (-GIoU) [+ (1 - NWD) for small objects].
Track queries (indices [0, T)) never enter the Hungarian step: each one is bound to
the ground truth with its track ID, or to background if that object left the scene.
Losses, on every decoder layer, the pre-box layer and the encoder proposals:
- class: MAL (DEIM, 2412.04234) -- the positive target is IoU^gamma, so a query is
  rewarded for confidence in proportion to how well its box matches; or sigmoid focal;
- boxes: L1 + GIoU, plus FGL (D-FINE): per-edge distribution focal loss towards the
  two bins around the target edge, weighted by IoU;
- DDF (D-FINE's GO-LSD): earlier layers' edge distributions distil from the final one.

Speed: one cost matrix per layer for the whole batch, one device->host copy for the
Hungarian step, and every loss computed with a single gather over all matches (the
per-image Python loops cost ~17k kernel launches per step before).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .box_ops import aligned_giou, bbox2distance, cxcywh_to_xyxy, generalized_box_iou, nwd

EMPTY = torch.empty(0, dtype=torch.long)


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
        Returns per-image (query_idx, gt_idx) CPU LongTensors.
        """
        b, q, _ = logits.shape
        sizes = [len(t["labels"]) for t in targets]
        fixed = fixed or [(EMPTY, EMPTY)] * b
        if sum(sizes) == 0 or q == skip:
            return [(fq.cpu(), fg.cpu()) for fq, fg in fixed]
        lab = torch.cat([t["labels"] for t in targets])
        gtb = torch.cat([t["boxes"] for t in targets]).float()
        p = logits[:, skip:].flatten(0, 1).float().sigmoid()
        pb = boxes[:, skip:].flatten(0, 1).float()
        pos = self.alpha * (1 - p) ** self.gamma * -(p + 1e-8).log()
        neg = (1 - self.alpha) * p ** self.gamma * -(1 - p + 1e-8).log()
        c = self.w_cls * (pos[:, lab] - neg[:, lab])
        c = c + self.w_l1 * torch.cdist(pb, gtb, p=1)
        c = c - self.w_giou * generalized_box_iou(cxcywh_to_xyxy(pb), cxcywh_to_xyxy(gtb))
        if self.w_nwd:
            c = c + self.w_nwd * (1 - nwd(pb, gtb, self.nwd_c))
        c = c.view(b, q - skip, -1).cpu()
        out, start = [], 0
        for i, n in enumerate(sizes):
            fq, fg = fixed[i][0].cpu(), fixed[i][1].cpu()
            free = torch.ones(n, dtype=torch.bool)
            free[fg] = False
            gt_idx = free.nonzero()[:, 0]
            if len(gt_idx):
                r, col = linear_sum_assignment(c[i, :, start + gt_idx].numpy())
                fq = torch.cat([fq, torch.as_tensor(r, dtype=torch.long) + skip])
                fg = torch.cat([fg, gt_idx[torch.as_tensor(col, dtype=torch.long)]])
            out.append((fq, fg))
            start += n
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


def id_contrastive(embs, ids, temperature=0.07):
    """Supervised contrastive loss (SupCon, 2004.11362) over identity embeddings.

    embs: [M, D] L2-normalised embeddings of matched queries from all frames of a clip
    batch; ids: [M] global identity labels (unique per video). Anchors are pulled towards
    the same identity in other frames and pushed from every other identity.
    """
    if len(ids) < 2:
        return embs.sum() * 0
    sim = embs @ embs.T / temperature
    eye = torch.eye(len(ids), dtype=torch.bool, device=embs.device)
    pos = (ids[:, None] == ids[None]) & ~eye
    has = pos.any(1)
    if not has.any():
        return embs.sum() * 0
    logits = sim.masked_fill(eye, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, 1, keepdim=True)
    loss = -(log_prob.masked_fill(~pos, 0).sum(1) / pos.sum(1).clamp(min=1))
    return loss[has].mean()


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
    def _flat(matches, offsets, device):
        """Per-image matches -> (batch idx, query idx, global gt idx) on `device`."""
        bi = torch.cat([torch.full((len(q),), i, dtype=torch.long) for i, (q, _) in enumerate(matches)])
        qi = torch.cat([q for q, _ in matches])
        gi = torch.cat([g + offsets[i] for i, (_, g) in enumerate(matches)])
        return bi.to(device), qi.to(device), gi.to(device)

    def _single(self, o, flat, all_lab, all_box, n_norm, ref_init=None):
        logits, boxes = o["pred_logits"], o["pred_boxes"]
        bi, qi, gi = flat
        has = len(bi) > 0
        pos = torch.zeros_like(logits, dtype=torch.bool)
        q = torch.zeros_like(logits, dtype=torch.float32)
        if has:
            lab, tgt_b = all_lab[gi], all_box[gi].float()
            sb = boxes[bi, qi].float()
            iou, giou = aligned_giou(cxcywh_to_xyxy(sb), cxcywh_to_xyxy(tgt_b))
            iou = iou.detach()
            pos[bi, qi, lab] = True
            q[bi, qi, lab] = iou
        if self.cls_loss == "mal":
            loss = {"cls": mal_loss(logits.float(), q, pos) / n_norm}
        else:
            loss = {"cls": sigmoid_focal(logits.float(), pos.float()) / n_norm}
        if not has:
            zero = boxes.sum() * 0
            loss.update(l1=zero, giou=zero)
            return loss, None
        loss["l1"] = F.l1_loss(sb, tgt_b, reduction="sum") / n_norm
        loss["giou"] = (1 - giou).sum() / n_norm
        if "pred_corners" in o and ref_init is not None:
            corners = o["pred_corners"][bi, qi].float().view(-1, 4, self.project.numel())
            idx, wl, wr = bbox2distance(ref_init[bi, qi].float(), tgt_b, self.reg_scale, self.project)
            logp = corners.log_softmax(-1)
            dfl = -(logp.gather(-1, idx[..., None])[..., 0] * wl + logp.gather(-1, idx[..., None] + 1)[..., 0] * wr)
            loss["fgl"] = (dfl.mean(-1) * iou).sum() / n_norm
        return loss, iou

    def _ddf(self, layers, final, flat, iou, n_norm):
        """GO-LSD: matched queries' edge distributions in earlier layers follow the final layer."""
        t, (bi, qi, _) = self.ddf_t, flat
        tgt = final["pred_corners"][bi, qi].float().detach().view(-1, 4, self.project.numel())
        total = 0.0
        for o in layers:
            src = o["pred_corners"][bi, qi].float().view_as(tgt)
            kl = F.kl_div((src / t).log_softmax(-1), (tgt / t).softmax(-1), reduction="none").sum(-1).mean(-1)
            total = total + (kl * iou).sum() * t * t / n_norm
        return total

    def forward(self, out, targets, fixed=None):
        """out: SarbonDetector output; fixed: track-query assignments (see matcher)."""
        dev = out["pred_logits"].device
        skip = out.get("n_track", 0)
        sizes = [len(t["labels"]) for t in targets]
        offsets = [0]
        for n in sizes[:-1]:
            offsets.append(offsets[-1] + n)
        all_lab = torch.cat([t["labels"] for t in targets]).to(dev)
        all_box = torch.cat([t["boxes"] for t in targets]).to(dev)
        n_norm = max(1.0, float(sum(sizes)))
        total, logs = 0.0, {}

        def run(o, matches, norm, ref_init=None, log=False):
            nonlocal total
            flat = self._flat(matches, offsets, dev)
            parts, iou = self._single(o, flat, all_lab, all_box, norm, ref_init)
            for k, v in parts.items():
                total = total + self.w[k] * v
                if log:
                    logs[k] = v.item()
            return flat, iou

        final = {k: out[k] for k in ("pred_logits", "pred_boxes", "pred_corners") if k in out}
        ref_init = out.get("ref_init")
        final_matches = self.matcher(final["pred_logits"], final["pred_boxes"], targets, fixed, skip)
        for o in out["aux"]:
            run(o, self.matcher(o["pred_logits"], o["pred_boxes"], targets, fixed, skip), n_norm, ref_init)
        flat, iou = run(final, final_matches, n_norm, ref_init, log=True)
        if "pred_corners" in final and len(out["aux"]) and iou is not None:
            ddf = self._ddf(out["aux"], final, flat, iou, n_norm)
            total = total + self.w["ddf"] * ddf
            logs["ddf"] = ddf.item()
        if "pre" in out:  # pre-box layer
            run(out["pre"], self.matcher(out["pre"]["pred_logits"], out["pre"]["pred_boxes"], targets,
                                         fixed, skip), n_norm)
        enc = {"pred_logits": out["enc_logits"], "pred_boxes": out["enc_boxes"]}
        run(enc, self.matcher(enc["pred_logits"], enc["pred_boxes"], targets), n_norm)
        if "dn" in out:  # denoising: assignments are known, no matching
            dn = out["dn"]
            norm = n_norm * dn["groups"]
            before = total
            for o in [*dn["aux"], {k: dn[k] for k in ("pred_logits", "pred_boxes", "pred_corners")}]:
                run(o, dn["matches"], norm, dn["ref_init"])
            run(dn["pre"], dn["matches"], norm)
            logs["dn"] = float(total - before) if not torch.is_tensor(total) else (total - before).item()
        return total, logs, final_matches
