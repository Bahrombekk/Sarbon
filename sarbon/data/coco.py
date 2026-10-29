"""COCO-format detection data with commercial-license filtering, plus a synthetic set.

COCO images carry per-image Flickr licenses. For training we keep only licenses that
allow commercial use (CC BY, CC BY-SA, no known copyright, US government work); the
NC / ND variants are dropped. Annotations themselves are CC BY 4.0.
"""
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

COMMERCIAL_OK = {"Attribution License", "Attribution-ShareAlike License",
                 "No known copyright restrictions", "United States Government Work"}
SURVEILLANCE = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]
ALL = None  # every COCO category (research benchmark against published numbers)


def letterbox(img, size):
    """Resize keeping aspect ratio and pad to size x size. Returns image, scale, (pad_x, pad_y)."""
    h, w = img.shape[:2]
    s = min(size / h, size / w)
    nh, nw = round(h * s), round(w * s)
    out = np.full((size, size, 3), 114, np.uint8)
    py, px = (size - nh) // 2, (size - nw) // 2
    out[py:py + nh, px:px + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, s, (px, py)


class CocoDetection(Dataset):
    def __init__(self, root, split, size=640, classes=SURVEILLANCE, commercial_only=True,
                 train=True, limit=None):
        root = Path(root)
        ann = json.loads((root / "annotations" / f"instances_{split}.json").read_text())
        self.img_dir = root / split
        self.size, self.train = size, train
        names = {c["id"]: c["name"] for c in ann["categories"]}
        self.cat_ids = [cid for cid, n in names.items() if classes is None or n in classes]
        self.label_of = {cid: i for i, cid in enumerate(self.cat_ids)}
        ok_lic = {l["id"] for l in ann["licenses"] if l["name"] in COMMERCIAL_OK}
        boxes = {}
        for a in ann["annotations"]:
            if a["category_id"] in self.label_of and not a.get("iscrowd", 0):
                boxes.setdefault(a["image_id"], []).append(a)
        self.images = [{k: im[k] for k in ("id", "file_name", "width", "height")} for im in ann["images"]
                       if im["id"] in boxes and (not commercial_only or im["license"] in ok_lic)]
        if limit:
            self.images = self.images[:limit]
        # keep only what __getitem__ needs: on Windows the dataset is pickled into every
        # DataLoader worker, and the full COCO json there exhausted the commit limit
        keep = {im["id"] for im in self.images}
        self.boxes = {i: [{"bbox": a["bbox"], "category_id": a["category_id"]} for a in v]
                      for i, v in boxes.items() if i in keep}

    def __len__(self):
        return len(self.images)

    def gt_letterboxed(self):
        """Ground truth as (image_id, [[x0,y0,x1,y1], ...], [labels]) in letterboxed pixels."""
        out = []
        for im in self.images:
            s = min(self.size / im["height"], self.size / im["width"])
            px = (self.size - round(im["width"] * s)) // 2
            py = (self.size - round(im["height"] * s)) // 2
            anns = self.boxes[im["id"]]
            b = [[a["bbox"][0] * s + px, a["bbox"][1] * s + py,
                  (a["bbox"][0] + a["bbox"][2]) * s + px, (a["bbox"][1] + a["bbox"][3]) * s + py]
                 for a in anns]
            out.append((im["id"], b, [self.label_of[a["category_id"]] for a in anns]))
        return out

    def load(self, i):
        """Letterboxed RGB uint8 image, xyxy boxes (pixels), labels, meta. No augmentation."""
        im = self.images[i]
        img = cv2.imread(str(self.img_dir / im["file_name"]))[:, :, ::-1]
        img, s, (px, py) = letterbox(img, self.size)
        b = np.array([a["bbox"] for a in self.boxes[im["id"]]], np.float32).reshape(-1, 4)
        b[:, 2:] += b[:, :2]
        b = b * s + np.array([px, py, px, py], np.float32)
        labels = np.array([self.label_of[a["category_id"]] for a in self.boxes[im["id"]]])
        return img, b, labels, (im["id"], s, px, py)

    def __getitem__(self, i):
        img, b, labels, meta = self.load(i)
        if self.train and random.random() < 0.5:
            img = img[:, ::-1]
            b[:, [0, 2]] = self.size - b[:, [2, 0]]
        t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)
        return t, torch.from_numpy(b), torch.from_numpy(labels), meta


class MosaicDetection(Dataset):
    """Mosaic of 4 images + random scale/crop + flip (YOLOv4 mosaic; DEIM's Dense O2O).

    More objects per image means more one-to-one positives per step, which DEIM shows
    speeds up DETR training. `prob` is lowered to 0 for the final epochs.
    """

    def __init__(self, base, prob=0.5, scale=(0.5, 1.5), min_px=2.0):
        self.base, self.prob, self.scale, self.min_px = base, prob, scale, min_px
        self.size = base.size

    def __len__(self):
        return len(self.base)

    def _clip(self, b, labels, s):
        if not len(b):
            return b, labels
        area0 = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        b = b.clip(0, s)
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        keep = ((b[:, 2] - b[:, 0]) > self.min_px) & ((b[:, 3] - b[:, 1]) > self.min_px) & (area > 0.25 * area0)
        return b[keep], labels[keep]

    def __getitem__(self, i):
        s = self.size
        if random.random() < self.prob:
            canvas = np.full((2 * s, 2 * s, 3), 114, np.uint8)
            xc, yc = (int(random.uniform(0.5 * s, 1.5 * s)) for _ in range(2))
            boxes, labels = [], []
            for k, j in enumerate([i] + random.choices(range(len(self.base)), k=3)):
                img, b, l, _ = self.base.load(j)
                # quadrant placement around (xc, yc), as in YOLO mosaic
                x0, y0 = (xc - s, yc - s) if k == 0 else (xc, yc - s) if k == 1 else \
                         (xc - s, yc) if k == 2 else (xc, yc)
                cx0, cy0 = max(x0, 0), max(y0, 0)
                cx1, cy1 = min(x0 + s, 2 * s), min(y0 + s, 2 * s)
                canvas[cy0:cy1, cx0:cx1] = img[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
                boxes.append(b + np.array([x0, y0, x0, y0], np.float32))
                labels.append(l)
            b, l = np.concatenate(boxes), np.concatenate(labels)
            # random scale, then a random s x s crop
            f = random.uniform(*self.scale) * 0.5
            canvas = cv2.resize(canvas, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR)
            b = b * f
            ch, cw = canvas.shape[:2]
            if ch < s or cw < s:
                pad = np.full((max(ch, s), max(cw, s), 3), 114, np.uint8)
                pad[:ch, :cw] = canvas
                canvas, ch, cw = pad, max(ch, s), max(cw, s)
            ox, oy = random.randint(0, cw - s), random.randint(0, ch - s)
            img = canvas[oy:oy + s, ox:ox + s]
            b, l = self._clip(b - np.array([ox, oy, ox, oy], np.float32), l, s)
        else:
            img, b, l, _ = self.base.load(i)
        if random.random() < 0.5:
            img = img[:, ::-1]
            b = b.copy()
            b[:, [0, 2]] = s - b[:, [2, 0]]
        t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)
        return t, torch.from_numpy(b.astype(np.float32)), torch.from_numpy(l.astype(np.int64)), (i, 1.0, 0, 0)


class SyntheticDetection(Dataset):
    """Textured background with random rectangles; for smoke tests without real data."""

    def __init__(self, n=64, size=256, num_classes=3, seed=0):
        self.n, self.size, self.k, self.seed = n, size, num_classes, seed

    def __len__(self):
        return self.n

    def gt_letterboxed(self):
        out = []
        for i in range(self.n):
            _, b, l, _ = self[i]
            out.append((i, b.tolist(), l.tolist()))
        return out

    def load(self, i):
        t, b, l, meta = self[i]
        return t.permute(1, 2, 0).numpy(), b.numpy(), l.numpy(), meta

    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed + i)
        s = self.size
        img = (rng.random((s // 8, s // 8, 3)) * 255).astype(np.uint8)
        img = cv2.resize(img, (s, s), interpolation=cv2.INTER_LINEAR)
        boxes, labels = [], []
        for _ in range(rng.integers(1, 6)):
            c = int(rng.integers(self.k))
            w, h = rng.integers(8, s // 3, 2)
            x, y = rng.integers(0, s - w), rng.integers(0, s - h)
            img[y:y + h, x:x + w] = (np.array([255, 0, 0]) if c == 0 else
                                     np.array([0, 255, 0]) if c == 1 else np.array([0, 0, 255]))
            boxes.append([x, y, x + w, y + h])
            labels.append(c)
        t = torch.from_numpy(img).permute(2, 0, 1)
        return t, torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels), (i, 1.0, 0, 0)


def collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    return imgs, [b[1] for b in batch], [b[2] for b in batch], [b[3] for b in batch]


class TargetCollate:
    """Collate that also renders dense targets, so it runs inside DataLoader workers."""

    def __init__(self, build, num_classes, out_h, out_w):
        self.build, self.args = build, (num_classes, out_h, out_w)

    def __call__(self, batch):
        imgs, boxes, labels, meta = collate(batch)
        tg = [self.build(b, l, *self.args) for b, l in zip(boxes, labels)]
        heat, reg, mask = (torch.stack(x) for x in zip(*tg))
        return imgs, heat, reg, mask, meta
