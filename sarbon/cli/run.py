r"""Run Sarbon (detection + tracking) on live cameras or video files and show the result.

  one RTSP camera : sarbon-run rtsp://user:pass@ip:554/stream
  RailSafe config : sarbon-run C:\path\gui_config.json --cams 0,1
  video file      : sarbon-run C:\data\recordings\x.mp4 --save runs\demo.mp4
  webcam          : sarbon-run 0
  from a file     : sarbon-run --config configs/run/camera.yaml

Sources can be mixed; all of them go through the model as one batch. Keys in the window:
q / Esc quit, space pause. Camera passwords are never printed.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from ..config import parse_with_config
from ..data.coco import RAILSAFE
from ..inference.sources import Source, expand_sources
from ..inference.video import SarbonVideo
from ..inference.visualize import draw, tile
from ..models.sarbon_det import build, load_weights
from ..paths import DEFAULT_DETECTOR
from ..track import TrackerConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="*", help="rtsp URL, video file, webcam index or RailSafe gui_config.json")
    ap.add_argument("--cams", help="with a .json config: comma-separated camera indexes to use")
    ap.add_argument("--weights", default=str(DEFAULT_DETECTOR))
    ap.add_argument("--size", default="640x1152", help="model input HxW, multiples of 32 "
                    "(384x640 faster, 640x1152 sees small/far objects better)")
    ap.add_argument("--view", type=int, default=1280, help="width each camera is processed and shown at")
    ap.add_argument("--conf", type=float, default=0.3, help="detection score for starting a track")
    ap.add_argument("--no-graph", action="store_true", help="run the detector eagerly instead of as a CUDA graph")
    ap.add_argument("--sparse", action="store_true", help="block-sparse stem: recompute only changed blocks")
    ap.add_argument("--tau", type=float, default=8 / 255, help="with --sparse: per-pixel change below this "
                    "counts as noise (0 = lossless, every compression flicker recomputes the block)")
    ap.add_argument("--save", help="write the annotated grid to this .mp4")
    ap.add_argument("--log", help="write tracks per frame to this .jsonl")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    args = parse_with_config(ap)
    if not args.sources:
        ap.error("give at least one source (or sources: [...] in --config)")

    cam_filter = [int(i) for i in args.cams.split(",")] if args.cams else None
    srcs = []
    for name, src, live in expand_sources(args.sources, cam_filter):
        try:
            srcs.append(Source(name, src, live))
            print(f"opened {name}")
        except RuntimeError as e:
            print(f"skip: {e}")
    if not srcs:
        sys.exit("no source could be opened")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    h, w = (int(v) for v in args.size.lower().split("x"))
    det = load_weights(build("s", len(RAILSAFE), halo_zero=False), args.weights)
    cfg = TrackerConfig(high=args.conf, new=args.conf, emb=True, color=0.7, reid=True, ground=True)
    video = SarbonVideo(det, len(srcs), size=(h, w), tracker_cfg=cfg, class_names=RAILSAFE, device=dev,
                        tau=args.tau if args.sparse else 0.0, graph=not args.no_graph)
    if not args.sparse:
        video.engine = None
    print(f"model {Path(args.weights).name} on {dev}, input {h}x{w}, {len(srcs)} source(s)"
          f"{', dense stem' if video.engine is None else ', block-sparse stem'}")

    # wait for the first frame from every live source
    t0 = time.time()
    while any(s.live and s.frame is None for s in srcs) and time.time() - t0 < 15:
        time.sleep(0.05)

    writer = log = None
    if args.log:
        log = open(args.log, "w", encoding="utf-8")
    view_h = None
    last = [None] * len(srcs)
    n, paused, ms_hist = 0, False, []
    try:
        while True:
            if not paused:
                frames = []
                for i, s in enumerate(srcs):
                    f = s.read()
                    if f is None:
                        f = last[i]
                    if f is None:
                        f = np.zeros((720, 1280, 3), np.uint8)
                    last[i] = f
                    if view_h is None:
                        view_h = round(f.shape[0] * args.view / f.shape[1] / 2) * 2
                    frames.append(cv2.resize(f, (args.view, view_h)))
                if all(s.ended for s in srcs if not s.live) and not any(s.live for s in srcs):
                    break
                if dev == "cuda":
                    torch.cuda.synchronize()
                t = time.perf_counter()
                results = video.step(frames)
                if dev == "cuda":
                    torch.cuda.synchronize()
                ms = (time.perf_counter() - t) * 1e3
                ms_hist = (ms_hist + [ms])[-30:]
                info = f"{np.mean(ms_hist):.0f} ms  {1000 / np.mean(ms_hist):.0f} fps"
                shown = [draw(f.copy(), r, s.name, f"{len(r)} obj  {info}")
                         for f, r, s in zip(frames, results, srcs)]
                grid = tile(shown)
                if log:
                    log.write(json.dumps({"frame": n, "t": time.time(), "cams": [
                        [{**r, "box": [round(float(v), 1) for v in r["box"]]} for r in res]
                        for res in results]}) + "\n")
                if args.save:
                    if writer is None:
                        fps = srcs[0].fps if not srcs[0].live else 1000 / np.mean(ms_hist)
                        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                                 (grid.shape[1], grid.shape[0]))
                    writer.write(grid)
                n += 1
                if n % 100 == 0:
                    print(f"frame {n}: {info}, objects {[len(r) for r in results]}")
                if args.max_frames and n >= args.max_frames:
                    break
            if not args.no_show:
                cv2.imshow("Sarbon", cv2.resize(grid, None, fx=min(1.0, 1600 / grid.shape[1]),
                                                fy=min(1.0, 1600 / grid.shape[1])))
                k = cv2.waitKey(1) & 0xFF
                if k in (ord("q"), 27):
                    break
                if k == ord(" "):
                    paused = not paused
    except KeyboardInterrupt:
        pass
    finally:
        for s in srcs:
            s.close()
        if writer:
            writer.release()
        if log:
            log.close()
        cv2.destroyAllWindows()
    if ms_hist:
        print(f"done: {n} frames, last 30 avg {np.mean(ms_hist):.1f} ms per step ({len(srcs)} cameras)")
        if video.stats["active_blocks"]:
            print(f"block-sparse: {100 * np.mean(video.stats['active_blocks']):.0f}% of blocks recomputed on average, "
                  f"steps {video.engine.stats}, dense above {video.engine.fallback}/{video.engine.n_blocks} blocks")


if __name__ == "__main__":
    main()
