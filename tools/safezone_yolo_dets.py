"""YOLO (Ultralytics) detections of the SmartGUI helmet models on a SafeZone split, for comparison only.

Run with an env that has Ultralytics (SmartGUI's venv); Sarbon itself never imports it (AGPL).
Output: {image path: [[x0, y0, x1, y1 (0..1), score, class], ...]} as read by tools/safezone_eval.py.

    C:/Users/User/Desktop/SmartGUI/venv/Scripts/python.exe tools/safezone_yolo_dets.py \
        --weights C:/Users/User/Desktop/SmartGUI/app/models/helmet_v4.pt --list C:/data/safezone/test.txt \
        --imgsz 1024 --out C:/data/safezone/yolo_helmet_v4_test.json
"""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--list", required=True)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    model = YOLO(args.weights)
    paths = [l.strip() for l in Path(args.list).read_text(encoding="utf-8").splitlines() if l.strip()]
    out = {}
    for i in range(0, len(paths), 16):
        for p, r in zip(paths[i:i + 16], model.predict(paths[i:i + 16], imgsz=args.imgsz, conf=0.001, iou=0.7,
                                                         max_det=300, verbose=False, half=True)):
            b = r.boxes
            out[p] = [[*xy, s, c] for xy, s, c in zip(b.xyxyn.tolist(), b.conf.tolist(), b.cls.int().tolist())]
    Path(args.out).write_text(json.dumps(out), encoding="utf-8")
    print(len(out), "images ->", args.out)


if __name__ == "__main__":
    main()
