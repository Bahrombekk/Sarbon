"""YOLO baseline on our own labelled data -- run with the Ultralytics env, NOT the Sarbon env.

Ultralytics is AGPL-3.0: it is used here only to measure the competitor, never imported by
the product. Same split as Sarbon (make_yolo_split.py lists), same input size.

  train : <ultralytics python> scripts\\yolo_own.py train --split C:\\data\\pereezd_split --out D:\\Sarbon\\runs\\yolo
  dump  : <ultralytics python> scripts\\yolo_own.py dump  --weights <best.pt> --split C:\\data\\pereezd_split --json <out.json>
  speed : <ultralytics python> scripts\\yolo_own.py speed --weights <best.pt>
`dump` writes raw detections (normalised xyxy, score, class) for every val image, so
bench/compare_own.py scores YOLO and Sarbon with the very same COCO evaluator.
"""
import argparse
import json
import time
from pathlib import Path


def train(args):
    from ultralytics import YOLO
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = out / "own.yaml"
    split = Path(args.split).as_posix()
    data.write_text(f"train: {split}/train.txt\nval: {split}/val.txt\nnames:\n  0: yengil\n  1: ogir\n")
    YOLO(args.model).train(data=str(data), epochs=args.epochs, imgsz=args.img, batch=args.batch,
                           workers=args.workers, project=str(out), name=Path(args.model).stem + "_own" + args.tag,
                           exist_ok=True, seed=0, fraction=args.fraction, device=args.device)


def dump(args):
    from ultralytics import YOLO
    model = YOLO(args.weights)
    paths = [l.strip() for l in (Path(args.split) / "val.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    paths = paths[:args.limit]
    res = {}
    for i in range(0, len(paths), 32):
        # Ultralytics' own val settings: conf 0.001, NMS IoU 0.7, up to 300 boxes
        for p, r in zip(paths[i:i + 32], model.predict(paths[i:i + 32], imgsz=args.img, conf=0.001, iou=0.7,
                                                        max_det=300, half=args.device != "cpu",
                                                        device=args.device, verbose=False)):
            b = r.boxes
            res[p] = [[*xy, s, int(c)] for xy, s, c in zip(b.xyxyn.tolist(), b.conf.tolist(), b.cls.tolist())]
    Path(args.json).write_text(json.dumps(res))
    print(f"{len(res)} images -> {args.json}")


def speed(args):
    import torch
    from ultralytics import YOLO
    net = YOLO(args.weights).model.fuse().cuda().half().eval()
    x = torch.randn(1, 3, args.img, args.img, device="cuda", dtype=torch.half)
    with torch.no_grad():
        for _ in range(30):
            net(x)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(200):
            net(x)
        torch.cuda.synchronize()
    res = json.dumps({"model": args.weights, "ms_per_frame": (time.perf_counter() - t) / 200 * 1e3,
                      "note": "PyTorch fp16, batch 1, network only (no NMS)"})
    print(res)
    if args.json:
        Path(args.json).write_text(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "dump", "speed"])
    ap.add_argument("--split", default="C:/data/pereezd_split")
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--weights")
    ap.add_argument("--out", default="D:/Sarbon/runs/yolo")
    ap.add_argument("--json")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--img", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--fraction", type=float, default=1.0, help="train on a fraction (smoke tests)")
    ap.add_argument("--limit", type=int, default=None, help="dump only N val images (smoke tests)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    {"train": train, "dump": dump, "speed": speed}[a.mode](a)
