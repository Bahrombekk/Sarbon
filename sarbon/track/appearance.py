"""Colour appearance descriptor: HSV histograms of the upper and lower body.

Cheap (no network), robust to pose, and complementary to learned embeddings while
those are not yet trained with an identity loss. Upper/lower halves separate shirt
and trousers, which is most of what distinguishes people on CCTV. Value (brightness)
is binned coarsely so slow illumination changes move little mass.
"""
import cv2
import numpy as np

BINS = (12, 4, 2)  # hue, saturation, value


def color_descriptor(frame_bgr, box):
    """L1-normalised concatenation of two HSV histograms (upper, lower body)."""
    h, w = frame_bgr.shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    x0, x1 = max(0, x0), min(w, x1)
    y0, y1 = max(0, y0), min(h, y1)
    if x1 - x0 < 2 or y1 - y0 < 4:
        return np.full(2 * np.prod(BINS), 1.0 / (2 * np.prod(BINS)), np.float32)
    # shrink horizontally: box edges are mostly background
    m = (x1 - x0) // 6
    crop = cv2.cvtColor(frame_bgr[y0:y1, x0 + m:x1 - m or x1], cv2.COLOR_BGR2HSV)
    mid = crop.shape[0] // 2
    out = []
    for part in (crop[: mid], crop[mid:]):
        hist = cv2.calcHist([part], [0, 1, 2], None, list(BINS), [0, 180, 0, 256, 0, 256]).ravel()
        out.append(hist / max(hist.sum(), 1.0))
    return np.concatenate(out).astype(np.float32) / 2


def bhattacharyya(a, b):
    """Pairwise Bhattacharyya distance in [0, 1] between rows of a [N, D] and b [M, D]."""
    bc = np.sqrt(np.clip(a, 0, None)) @ np.sqrt(np.clip(b, 0, None)).T
    return np.sqrt(np.clip(1.0 - bc, 0.0, 1.0))
