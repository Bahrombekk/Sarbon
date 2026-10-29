"""Video clips with identities for re-ID training.

MOTClips reads the MOTChallenge layout (also used by many fixed-camera datasets once
converted, e.g. MEVA / SmartSpaces exports):
    <root>/<seq>/img1/000001.jpg ...
    <root>/<seq>/gt/gt.txt   rows: frame, id, x, y, w, h, conf, class, visibility
A sample is a clip of `clip` frames from one sequence, `gap` frames apart, so the same
people appear under different poses / positions -- the positives of the ID loss.
Identities are made global (sequence index * 1e6 + id) so negatives span the batch.
"""
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .coco import letterbox


class MOTClips(Dataset):
    def __init__(self, root, size=640, clip=2, max_gap=15, classes=None, min_vis=0.15, samples=None):
        self.size, self.clip, self.max_gap = size, clip, max_gap
        self.classes = classes or {1: 0}  # MOT class -> our label (1 = pedestrian)
        self.seqs = []
        for si, seq in enumerate(sorted(p for p in Path(root).iterdir() if (p / "gt" / "gt.txt").exists())):
            gt = np.loadtxt(seq / "gt" / "gt.txt", delimiter=",", ndmin=2)
            keep = (gt[:, 6] > 0) & np.isin(gt[:, 7], list(self.classes)) & (gt[:, 8] >= min_vis)
            gt = gt[keep]
            frames = {}
            for row in gt:
                frames.setdefault(int(row[0]), []).append(row)
            imgs = sorted((seq / "img1").glob("*.jpg"))
            self.seqs.append({"dir": seq, "imgs": {int(p.stem): p for p in imgs},
                              "gt": {f: np.array(r) for f, r in frames.items()}, "index": si})
        self.items = [(i, f) for i, s in enumerate(self.seqs) for f in sorted(s["gt"])]
        self.samples = samples or len(self.items)

    def __len__(self):
        return self.samples

    def _frame(self, s, f):
        img = cv2.imdecode(np.fromfile(str(s["imgs"][f]), np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]
        img, sc, (px, py) = letterbox(img, self.size)
        rows = s["gt"].get(f, np.zeros((0, 9)))
        b = rows[:, 2:6].astype(np.float32).copy()
        b[:, 2:] += b[:, :2]
        b = b * sc + np.array([px, py, px, py], np.float32)
        labels = np.array([self.classes[int(c)] for c in rows[:, 7]], np.int64)
        ids = (s["index"] * 1_000_000 + rows[:, 1]).astype(np.int64)
        return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1), b, labels, ids

    def __getitem__(self, i):
        si, f0 = self.items[random.randrange(len(self.items))]
        s = self.seqs[si]
        frames = [f0]
        for _ in range(self.clip - 1):
            nxt = [f for f in range(frames[-1] + 1, frames[-1] + self.max_gap + 1) if f in s["gt"]]
            frames.append(random.choice(nxt) if nxt else frames[-1])
        flip = random.random() < 0.5
        out = []
        for f in frames:
            img, b, l, ids = self._frame(s, f)
            if flip:
                img = img.flip(-1)
                b[:, [0, 2]] = self.size - b[:, [2, 0]]
            out.append((img, torch.from_numpy(b), torch.from_numpy(l), torch.from_numpy(ids)))
        return out


class SyntheticClips(Dataset):
    """Moving objects with a fixed per-identity colour/pattern on a static textured background."""

    def __init__(self, videos=8, size=256, clip=2, objects=6, frames=60, seed=0):
        self.size, self.clip = size, clip
        rng = np.random.default_rng(seed)
        self.videos = []
        for v in range(videos):
            bg = cv2.resize((rng.random((size // 8, size // 8, 3)) * 255).astype(np.uint8), (size, size))
            objs = []
            for k in range(objects):
                w, h = rng.integers(16, size // 4, 2)
                p = rng.uniform([0, 0], [size - w, size - h])
                vel = rng.normal(0, 2, 2)
                color = rng.integers(0, 256, 3)
                stripe = rng.integers(2, 6)
                objs.append((v * 1000 + k, int(w), int(h), p, vel, color, stripe))
            self.videos.append((bg, objs, frames))
        self.n = videos * frames

    def __len__(self):
        return self.n

    def _render(self, vid, t):
        bg, objs, _ = self.videos[vid]
        img = bg.copy()
        boxes, ids = [], []
        for oid, w, h, p, vel, color, stripe in objs:
            x, y = p + vel * t
            x, y = int(np.clip(x, 0, self.size - w)), int(np.clip(y, 0, self.size - h))
            patch = np.tile(color, (h, w, 1)).astype(np.uint8)
            patch[::stripe] //= 2  # identity-specific stripe pattern
            img[y:y + h, x:x + w] = patch
            boxes.append([x, y, x + w, y + h])
            ids.append(oid)
        t_img = torch.from_numpy(img).permute(2, 0, 1)
        return (t_img, torch.tensor(boxes, dtype=torch.float32), torch.zeros(len(ids), dtype=torch.long),
                torch.tensor(ids))

    def __getitem__(self, i):
        vid = i % len(self.videos)
        t0 = random.randrange(self.videos[vid][2] - 20)
        ts = [t0] + [t0 + random.randint(1, 15) for _ in range(self.clip - 1)]
        return [self._render(vid, t) for t in ts]


def collate_clips(batch):
    """-> list over clip positions of (imgs [B,3,S,S], boxes list, labels list, ids list)."""
    out = []
    for t in range(len(batch[0])):
        frames = [b[t] for b in batch]
        out.append((torch.stack([f[0] for f in frames]), [f[1] for f in frames],
                    [f[2] for f in frames], [f[3] for f in frames]))
    return out
