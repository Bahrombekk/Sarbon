"""Video sources for live inference: RTSP cameras, files, webcams and RailSafe GUI configs.

Live sources are read in a background thread that keeps only the newest frame, so a slow
consumer never builds up RTSP delay; files are read frame by frame. Camera passwords are never
printed: use `safe(url)` for anything that goes to a log or the screen.
"""
import json
import os
import re
import threading
import time
from pathlib import Path

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
import cv2  # noqa: E402


def cameras(cfg_path):
    """RTSP cameras of a RailSafe GUI config -> [(name, url)]."""
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
    """URL with the user:password part masked."""
    return re.sub(r"//[^@/]+@", "//***:***@", url)


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
    """One camera or file. Live sources: a thread keeps only the newest frame.

    `count` numbers the frames the source has produced and `gen` the (re)connections, so a
    consumer slower than the stream knows how many frames it skipped (tracker dt), when it was
    handed the same frame again, and when a reconnect broke the motion history.
    """

    def __init__(self, name, src, live):
        self.name, self.src, self.live = name, src, live
        api = cv2.CAP_FFMPEG if isinstance(src, str) else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(src, api)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open {name} ({safe(src) if isinstance(src, str) else src})")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
        self.frame, self.ended, self.lock = None, False, threading.Lock()
        self.count, self.gen = 0, 0
        self.thread = None
        if live:
            self.thread = threading.Thread(target=self._reader, daemon=True)
            self.thread.start()

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
                    with self.lock:
                        self.gen += 1
                continue
            fails = 0
            with self.lock:
                self.frame = f
                self.count += 1

    def read(self):
        return self.grab()[0]

    def grab(self):
        """(frame, count, gen): the newest frame with its sequence number and connection number."""
        if not self.live:
            ok, f = self.cap.read()
            if not ok:
                self.ended = True
                return None, self.count, self.gen
            self.count += 1
            return f, self.count, self.gen
        with self.lock:
            return self.frame, self.count, self.gen

    def close(self):
        self.ended = True
        if self.thread is not None:  # releasing while the reader is inside read() crashes FFmpeg (HEVC)
            self.thread.join(timeout=5)
        self.cap.release()


