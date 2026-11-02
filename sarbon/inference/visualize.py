"""Drawing helpers for the live view: boxes with class, track ID and score; camera grid."""
import cv2
import numpy as np

CLASS_COLORS = {"person": (60, 200, 60), "bicycle": (200, 200, 60), "motorcycle": (200, 120, 60),
                "yengil": (255, 160, 40), "ogir": (40, 80, 255), "train": (200, 60, 200)}


def draw(frame, tracks, name, info):
    for t in tracks:
        x0, y0, x1, y1 = map(int, t["box"])
        label = t["label"] if isinstance(t["label"], str) else str(t["label"])
        c = CLASS_COLORS.get(label, (200, 200, 200))
        cv2.rectangle(frame, (x0, y0), (x1, y1), c, 2)
        txt = f"{label} #{t['id']} {t['score']:.2f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x0, max(0, y0 - th - 6)), (x0 + tw + 4, y0), c, -1)
        cv2.putText(frame, txt, (x0 + 2, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{name}  {info}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, f"{name}  {info}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def tile(frames):
    n = len(frames)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    h, w = frames[0].shape[:2]
    grid = np.zeros((rows * h, cols * w, 3), np.uint8)
    for i, f in enumerate(frames):
        r, c = divmod(i, cols)
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = f
    return grid


