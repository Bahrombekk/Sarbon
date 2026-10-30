"""YOLO-format detection data (one .txt per image: `class cx cy w h`, normalised).

Used for our own labelled camera data (e.g. the RailSafe crossing set). Same interface
as `CocoDetection` (`load`, `gt_letterboxed`, `__getitem__`), so it plugs into
`MosaicDetection` and the evaluator unchanged.

Large frames (2560x1440 here) are decoded at 1/2, 1/4 or 1/8 scale straight from the
JPEG DCT (`IMREAD_REDUCED_*`) when that is still at least the training size: about 5x
faster than decoding full resolution and resizing, which kept the loader from starving
the GPU.
"""
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .coco import letterbox

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
REDUCED = {8: cv2.IMREAD_REDUCED_COLOR_8, 4: cv2.IMREAD_REDUCED_COLOR_4, 2: cv2.IMREAD_REDUCED_COLOR_2}


def label_path(img):
    """.../images/x.jpg -> .../labels/x.txt (last `images` folder in the path)."""
    p = Path(img)
    parts = list(p.parts)
    i = len(parts) - 1 - parts[::-1].index("images")
    parts[i] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_labels(path, num_classes):
    """-> (rows [[cls, cx, cy, w, h], ...] with 0 <= cls < num_classes, invalid lines)."""
    rows, bad = [], []
    if not Path(path).exists():
        return rows, bad
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        v = line.split()
        if not v:
            continue
        try:
            c, *box = int(float(v[0])), *map(float, v[1:5])
        except ValueError:
            bad.append(line)
            continue
        if len(box) == 4 and 0 <= c < num_classes and box[2] > 0 and box[3] > 0:
            rows.append([c, *box])
        else:
            bad.append(line)
    return rows, bad


def imread_at_least(path, size):
    """BGR image, JPEG-downscaled by the largest of 8/4/2 that keeps the long side >= size."""
    buf = np.fromfile(str(path), np.uint8)
    head = cv2.imdecode(buf, cv2.IMREAD_REDUCED_GRAYSCALE_8)  # cheap probe of the size
    if head is not None:
        long8 = max(head.shape[:2])
        for f in (8, 4, 2):
            if long8 * 8 / f >= size:
                img = cv2.imdecode(buf, REDUCED[f])
                if img is not None:
                    return img
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


class YoloDetection(Dataset):
    """list_file: one image path per line. class_map: YOLO class id -> our label (others dropped)."""

    def __init__(self, list_file, size=640, class_map=None, train=True, limit=None):
        self.size, self.train = size, train
        self.class_map = class_map or {0: 0}
        paths = [l.strip() for l in Path(list_file).read_text(encoding="utf-8").splitlines() if l.strip()]
        if limit:
            paths = paths[:limit]
        self.images, self.boxes = [], {}
        n_cls = max(self.class_map) + 1
        for i, p in enumerate(paths):
            rows, _ = read_labels(label_path(p), n_cls)
            rows = [r for r in rows if r[0] in self.class_map]
            self.images.append({"id": i, "file_name": p})
            # float32 array per image: small to pickle into DataLoader workers
            self.boxes[i] = np.array([[self.class_map[r[0]], *r[1:]] for r in rows], np.float32).reshape(-1, 5)

    def __len__(self):
        return len(self.images)

    def _scaled(self, i, w, h):
        """Normalised cxcywh rows -> letterboxed xyxy pixels for an image of size (w, h)."""
        s = min(self.size / h, self.size / w)
        px, py = (self.size - round(w * s)) // 2, (self.size - round(h * s)) // 2
        r = self.boxes[i]
        x0, y0 = (r[:, 1] - r[:, 3] / 2) * w, (r[:, 2] - r[:, 4] / 2) * h
        x1, y1 = (r[:, 1] + r[:, 3] / 2) * w, (r[:, 2] + r[:, 4] / 2) * h
        b = np.stack([x0, y0, x1, y1], 1).clip(0, [w, h, w, h]) * s + np.array([px, py, px, py], np.float32)
        return b.astype(np.float32), r[:, 0].astype(np.int64), s, px, py

    def gt_letterboxed(self):
        """Ground truth in letterboxed pixels (needs each image's aspect ratio; cached)."""
        if getattr(self, "_gt", None) is None:
            out = []
            for im in self.images:
                img = imread_at_least(im["file_name"], 1)  # smallest decode, aspect ratio only
                h, w = img.shape[:2]
                b, l, *_ = self._scaled(im["id"], w, h)
                out.append((im["id"], b.tolist(), l.tolist()))
            self._gt = out
        return self._gt

    def load(self, i):
        im = self.images[i]
        img = imread_at_least(im["file_name"], self.size)[:, :, ::-1]
        h, w = img.shape[:2]
        b, labels, s, px, py = self._scaled(i, w, h)
        img, _, _ = letterbox(img, self.size)
        return img, b, labels, (im["id"], s, px, py)

    def __getitem__(self, i):
        img, b, labels, meta = self.load(i)
        if self.train and random.random() < 0.5:
            img = img[:, ::-1]
            b = b.copy()
            b[:, [0, 2]] = self.size - b[:, [2, 0]]
        t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)
        return t, torch.from_numpy(b), torch.from_numpy(labels), meta


class WithClassMask(Dataset):
    """Tags every sample with the classes its source annotates (partial-label training).

    The criterion then never treats an unannotated class as background on these images
    (e.g. the crossing set labels vehicles only: its pedestrians and trains are unlabelled,
    not absent). The mask is appended to the sample's meta tuple.
    """

    def __init__(self, ds, num_classes, annotated):
        self.ds = ds
        self.valid = torch.zeros(num_classes, dtype=torch.bool)
        self.valid[list(annotated)] = True

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        img, b, l, meta = self.ds[i]
        return img, b, l, (*meta, self.valid)
