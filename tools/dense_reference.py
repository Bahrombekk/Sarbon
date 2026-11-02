r"""Dense reference for compute-scheduling research: every frame of a recording through the detector.

Writes one .npz per video with, for every frame:
  dets    : detections with score >= --keep (frame pixels xyxy, score, class), concatenated;
            off[t]:off[t+1] are frame t's rows
  thumb   : 96x54 grayscale thumbnail (uint8) -- the cheap signal a scheduler is allowed to see
            on frames it does not compute
A scheduler is then simulated offline (bench/scheduler_eval.py): "compute frame t" = take the
dense detections of t, "skip" = the tracker's prediction. The model, input size and resize match
sarbon-run (stretch to 640x1152), so the reference equals what the live pipeline would see.

  python tools\dense_reference.py C:\data\recordings\*.mp4 --out C:\data\recordings\dense
"""
import argparse
import glob
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from sarbon.data.coco import RAILSAFE
from sarbon.models.box_ops import cxcywh_to_xyxy
from sarbon.models.sarbon_det import build, load_weights
from sarbon.paths import DEFAULT_DETECTOR

MEAN = torch.tensor([0.485, 0.456, 0.406]) * 255
STD = torch.tensor([0.229, 0.224, 0.225]) * 255
THUMB = (96, 54)


class GraphedDetector:
    """Fixed-shape detector replayed as a CUDA graph (identical numbers to eager fp16 autocast)."""

    def __init__(self, det, batch, h, w):
        self.det = det
        self.x = torch.zeros(batch, h, w, 3, device="cuda")
        self.mean, self.std = MEAN.cuda(), STD.cuda()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._run()
        torch.cuda.current_stream().wait_stream(s)
        self.g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g):
            self.out = self._run()

    @torch.no_grad()
    def _run(self):
        x = ((self.x - self.mean) / self.std).permute(0, 3, 1, 2)
        with torch.autocast("cuda", torch.float16):
            o = self.det(x)
        return o["pred_logits"].float().sigmoid(), cxcywh_to_xyxy(o["pred_boxes"].float())

    def __call__(self, frames_rgb):
        self.x[: len(frames_rgb)].copy_(torch.from_numpy(np.stack(frames_rgb)).cuda().float())
        self.g.replay()
        return self.out


def process(path, gd, args, h, w):
    cap = cv2.VideoCapture(str(path))
    fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = torch.tensor([W, H, W, H], device="cuda", dtype=torch.float32)
    rows, off, thumbs = [], [0], []
    t0, done = time.time(), False
    while not done:
        batch = []
        while len(batch) < args.batch:
            ok, f = cap.read()
            if not ok:
                done = True
                break
            thumbs.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), THUMB, interpolation=cv2.INTER_AREA))
            batch.append(cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR)[:, :, ::-1])
            if args.limit and len(thumbs) >= args.limit:
                done = True
                break
        if not batch:
            break
        prob, boxes = gd(batch)
        score, cls = prob[: len(batch)].max(-1)
        for k in range(len(batch)):
            keep = (score[k] >= args.keep).nonzero()[:, 0]
            keep = keep[score[k, keep].argsort(descending=True)][: args.max_dets]
            r = torch.cat([boxes[k, keep] * scale, score[k, keep, None], cls[k, keep, None].float()], 1)
            rows.append(r.cpu().numpy())
            off.append(off[-1] + len(keep))
        if len(thumbs) % (args.batch * 250) == 0:
            print(f"\r{path.name}: {len(thumbs)}/{n} frames, {len(thumbs) / (time.time() - t0):.0f} fps", end="")
    print(f"\r{path.name}: {len(thumbs)} frames in {time.time() - t0:.0f}s" + " " * 20)
    out = Path(args.out) / f"{path.stem}.npz"
    np.savez_compressed(out, dets=np.concatenate(rows).astype(np.float32), off=np.array(off, np.int64),
                        thumb=np.stack(thumbs), fps=fps, size=np.array([W, H]))
    return {"video": str(path), "frames": len(thumbs), "fps": fps, "size": [W, H], "dets": off[-1]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default=str(DEFAULT_DETECTOR))
    ap.add_argument("--size", default="640x1152")
    ap.add_argument("--keep", type=float, default=0.1, help="lowest score stored (the tracker's low threshold)")
    ap.add_argument("--max-dets", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="frames per video (0 = all)")
    args = ap.parse_args()
    h, w = (int(v) for v in args.size.lower().split("x"))
    Path(args.out).mkdir(parents=True, exist_ok=True)
    det = load_weights(build("s", len(RAILSAFE), halo_zero=False), args.weights).cuda().eval()
    gd = GraphedDetector(det, args.batch, h, w)
    videos = [Path(p) for v in args.videos for p in sorted(glob.glob(v))]
    report = [process(v, gd, args, h, w) for v in videos]
    (Path(args.out) / "reference.json").write_text(json.dumps({"weights": args.weights, "size": [h, w],
                                                                "videos": report}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
