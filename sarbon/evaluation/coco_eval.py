"""COCO-style evaluation: DETR post-processing and pycocotools AP on letterboxed ground truth."""
import contextlib
import io

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from ..engine.training import prep
from ..models.box_ops import cxcywh_to_xyxy


@torch.no_grad()
def postprocess(out, size, k=300):
    """DETR-style: top-k over (query, class) pairs; boxes in input pixels xyxy."""
    scores = out["pred_logits"].sigmoid()
    b, q, c = scores.shape
    s, idx = scores.flatten(1).topk(min(k, q * c))
    boxes = cxcywh_to_xyxy(out["pred_boxes"]).gather(1, (idx // c)[..., None].expand(-1, -1, 4)) * size
    return [(boxes[i], s[i], idx[i] % c) for i in range(b)]


def evaluate(model, ds, loader, device, num_classes, size, cats=None):
    """COCO AP; cats: evaluate only these classes (the ones a dataset annotates)."""
    dets = []
    model.eval()
    for imgs, _, _, meta in loader:
        with torch.no_grad(), torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
            out = model(prep(imgs, device))
        for (boxes, scores, labels), m in zip(postprocess(out, size), meta):
            for bb, s, l in zip(boxes.float().tolist(), scores.float().tolist(), labels.tolist()):
                dets.append({"image_id": m[0], "category_id": l,
                             "bbox": [bb[0], bb[1], bb[2] - bb[0], bb[3] - bb[1]], "score": s})
    return coco_metrics(ds.gt_letterboxed(), dets, num_classes, cats)


def coco_metrics(gt_items, dets, num_classes, cats=None):
    """gt_items: (image_id, xyxy boxes, labels); dets: COCO result dicts (xywh), same pixel frame."""
    gt = {"images": [], "annotations": [], "categories": [{"id": c} for c in range(num_classes)]}
    for img_id, boxes, labels in gt_items:
        gt["images"].append({"id": img_id})
        for b, l in zip(boxes, labels):
            w, h = b[2] - b[0], b[3] - b[1]
            gt["annotations"].append({"id": len(gt["annotations"]) + 1, "image_id": img_id,
                                      "category_id": l, "bbox": [b[0], b[1], w, h], "area": w * h,
                                      "iscrowd": 0})
    if not dets:
        return {"AP": 0.0}
    with contextlib.redirect_stdout(io.StringIO()):
        cg = COCO()
        cg.dataset = gt
        cg.createIndex()
        e = COCOeval(cg, cg.loadRes(dets), "bbox")
        if cats is not None:
            e.params.catIds = list(cats)
        e.evaluate()
        e.accumulate()
        e.summarize()
    s = e.stats
    return {"AP": s[0] * 100, "AP50": s[1] * 100, "AP75": s[2] * 100, "AP_small": s[3] * 100,
            "AP_medium": s[4] * 100, "AP_large": s[5] * 100}
