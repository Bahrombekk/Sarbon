"""Probe / record RTSP cameras listed in a RailSafe GUI config. Credentials never printed.

  probe : .venv\\Scripts\\python.exe scripts\\camera_probe.py probe <gui_config.json> --snap <dir>
  record: .venv\\Scripts\\python.exe scripts\\camera_probe.py record <gui_config.json> --out <dir> --minutes 15
Recording copies the camera's own H.264/H.265 stream (no re-encode) when ffmpeg is
available, else re-encodes frames with OpenCV at --width.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
import cv2  # noqa: E402


def cameras(cfg_path):
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    out = []
    for ci, c in enumerate(cfg.get("crossings", [])):
        for k, cam in enumerate(c.get("cameras", [])):
            src = cam.get("source", "")
            if src.lower().startswith("rtsp://"):
                name = f"x{ci}_{re.sub(r'[^A-Za-z0-9]+', '_', str(c.get('name', ci)))[:20]}_c{k}"
                out.append((name, src))
    return out


def safe(url):
    return re.sub(r"//[^@/]+@", "//***:***@", url)


def probe(cfg_path, snap_dir):
    Path(snap_dir).mkdir(parents=True, exist_ok=True)
    for name, url in cameras(cfg_path):
        t0 = time.time()
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        # joining mid-GOP decodes grey frames until the next keyframe: read ~3 s first
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        for _ in range(int(3 * fps)):
            ok2, f2 = cap.read()
            if not ok2:
                break
            ok, frame = ok2, f2
        dt = time.time() - t0
        if ok:
            fps = cap.get(cv2.CAP_PROP_FPS)
            cv2.imwrite(str(Path(snap_dir) / f"{name}.jpg"), cv2.resize(frame, (640, round(frame.shape[0] * 640 / frame.shape[1]))))
            print(f"OK   {name:28s} {frame.shape[1]}x{frame.shape[0]} @ {fps:.0f} fps  ({dt:.1f}s)  {safe(url)}")
        else:
            print(f"FAIL {name:28s} ({dt:.1f}s)  {safe(url)}")
        cap.release()


def record_ffmpeg(ffmpeg, url, out, seconds):
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-rtsp_transport", "tcp", "-i", url,
           "-t", str(seconds), "-c", "copy", "-an", "-y", str(out)]
    return subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE).returncode


def record_cv(url, out, seconds, width):
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    wr, n, t0 = None, 0, time.time()
    while time.time() - t0 < seconds:
        ok, f = cap.read()
        if not ok:
            break
        if f.shape[1] != width:
            f = cv2.resize(f, (width, round(f.shape[0] * width / f.shape[1])))
        if wr is None:
            wr = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (f.shape[1], f.shape[0]))
        wr.write(f)
        n += 1
    if wr:
        wr.release()
    cap.release()
    return n


def record(cfg_path, out_dir, minutes, width):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    threads, results = [], {}

    def job(name, url):
        path = out / f"{name}_{stamp}.mp4"
        if ffmpeg:
            results[name] = ("ffmpeg copy", record_ffmpeg(ffmpeg, url, path, minutes * 60), path)
        else:
            results[name] = ("opencv", record_cv(url, path, minutes * 60, width), path)

    for name, url in cameras(cfg_path):
        th = threading.Thread(target=job, args=(name, url))
        th.start()
        threads.append(th)
    for th in threads:
        th.join()
    for name, (how, res, path) in results.items():
        size = path.stat().st_size / 2**20 if path.exists() else 0
        print(f"{name:28s} {how:12s} result {res}  {size:7.1f} MB  {path.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["probe", "record"])
    ap.add_argument("config")
    ap.add_argument("--snap", default="runs/camera_snap")
    ap.add_argument("--out", default="C:/data/recordings")
    ap.add_argument("--minutes", type=float, default=15)
    ap.add_argument("--width", type=int, default=1280)
    a = ap.parse_args()
    probe(a.config, a.snap) if a.mode == "probe" else record(a.config, a.out, a.minutes, a.width)
    sys.exit(0)
