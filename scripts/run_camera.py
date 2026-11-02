"""Run Sarbon-S (detection + tracking) on live cameras or video files and show the result.

  one RTSP camera : .venv\\Scripts\\python.exe scripts\\run_camera.py rtsp://user:pass@ip:554/stream
  RailSafe config : .venv\\Scripts\\python.exe scripts\\run_camera.py C:\\path\\gui_config.json --cams 0,1
  video file      : .venv\\Scripts\\python.exe scripts\\run_camera.py C:\\data\\recordings\\x.mp4 --save runs\\demo.mp4
  webcam          : .venv\\Scripts\\python.exe scripts\\run_camera.py 0

Sources can be mixed; all of them go through the model as one batch. Live sources are read
in background threads that keep only the newest frame (no growing RTSP delay); files are
read frame by frame. Keys in the window: q / Esc quit, space pause.
Camera passwords are never printed (URLs are shown as rtsp://***:***@host/...).
"""
import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from camera_probe import cameras, safe  # noqa: E402
from sarbon.data.coco import RAILSAFE  # noqa: E402
from sarbon.models.sarbon_det import build, load_weights  # noqa: E402
from sarbon.pipeline import SarbonVideo  # noqa: E402
from sarbon.track import TrackerConfig  # noqa: E402

DEFAULT_WEIGHTS = ROOT / "runs" / "railsafe" / "s_railsafe" / "best.pt"
CLASS_COLORS = {"person": (60, 200, 60), "bicycle": (200, 200, 60), "motorcycle": (200, 120, 60),
                "yengil": (255, 160, 40), "ogir": (40, 80, 255), "train": (200, 60, 200)}


def expand_sources(items, cam_filter):
    """-> [(name, source, live)]; a .json is a RailSafe GUI config holding RTSP cameras."""
    out = []
    for s in items:
        if s.lower().endswith(".json"):
            cams = cameras(s)
            if cam_filter is not None:
                cams = [cams[i] for i in cam_filter]
            out += [(n, u, True) for n, u in cams]
        elif s.isdigit():
            out.append((f"webcam{s}", int(s), True))
        elif "://" in s:
            out.append((safe(s).split("@")[-1][:32], s, True))
        else:
            out.append((Path(s).stem[:32], s, False))
    return out


class Source:
    """One camera or file. Live sources: a thread keeps only the newest frame."""

    def __init__(self, name, src, live):
        self.name, self.src, self.live = name, src, live
        api = cv2.CAP_FFMPEG if isinstance(src, str) else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(src, api)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open {name} ({safe(src) if isinstance(src, str) else src})")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
        self.frame, self.ended, self.lock = None, False, threading.Lock()
        if live:
            threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        fails = 0
        while not self.ended:
            ok, f = self.cap.read()
            if not ok:
                fails += 1
                if fails > 50:  # camera dropped: reconnect
                    self.cap.release()
                    time.sleep(2)
                    self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
                    fails = 0
                continue
            fails = 0
            with self.lock:
                self.frame = f

    def read(self):
        if not self.live:
            ok, f = self.cap.read()
            if not ok:
                self.ended = True
                return None
            return f
        with self.lock:
            return self.frame

    def close(self):
        self.ended = True
        self.cap.release()


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="+", help="rtsp URL, video file, webcam index or RailSafe gui_config.json")
    ap.add_argument("--cams", help="with a .json config: comma-separated camera indexes to use")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--size", default="640x1152", help="model input HxW, multiples of 32 "
                    "(384x640 faster, 640x1152 sees small/far objects better)")
    ap.add_argument("--view", type=int, default=1280, help="width each camera is processed and shown at")
    ap.add_argument("--conf", type=float, default=0.3, help="detection score for starting a track")
    ap.add_argument("--sparse", action="store_true", help="block-sparse stem: recompute only changed blocks")
    ap.add_argument("--tau", type=float, default=8 / 255, help="with --sparse: per-pixel change below this "
                    "counts as noise (0 = lossless, every compression flicker recomputes the block)")
    ap.add_argument("--save", help="write the annotated grid to this .mp4")
    ap.add_argument("--log", help="write tracks per frame to this .jsonl")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

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
                        tau=args.tau if args.sparse else 0.0)
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
            print(f"block-sparse: {100 * np.mean(video.stats['active_blocks']):.0f}% of blocks recomputed on average")


if __name__ == "__main__":
    main()
