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
        self.cat_ids = [cid for cid, n in names.items() if n in classes]
        self.label_of = {cid: i for i, cid in enumerate(self.cat_ids)}
        ok_lic = {l["id"] for l in ann["licenses"] if l["name"] in COMMERCIAL_OK}
        boxes = {}
        for a in ann["annotations"]:
            if a["category_id"] in self.label_of and not a.get("iscrowd", 0):
                boxes.setdefault(a["image_id"], []).append(a)
        self.images = [im for im in ann["images"] if im["id"] in boxes
                       and (not commercial_only or im["license"] in ok_lic)]
        if limit:
            self.images = self.images[:limit]
        self.boxes = boxes
        self.coco_json = ann  # for evaluation

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

    def __getitem__(self, i):
        im = self.images[i]
        img = cv2.imread(str(self.img_dir / im["file_name"]))[:, :, ::-1]
        img, s, (px, py) = letterbox(img, self.size)
        b = np.array([a["bbox"] for a in self.boxes[im["id"]]], np.float32).reshape(-1, 4)
        b[:, 2:] += b[:, :2]
        b = b * s + np.array([px, py, px, py], np.float32)
        labels = np.array([self.label_of[a["category_id"]] for a in self.boxes[im["id"]]])
        if self.train and random.random() < 0.5:
            img = img[:, ::-1]
            b[:, [0, 2]] = self.size - b[:, [2, 0]]
        t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)
        return t, torch.from_numpy(b), torch.from_numpy(labels), (im["id"], s, px, py)


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
