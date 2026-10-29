"""Turn fixed-camera recordings into a pseudo-labelled MOT dataset (no human labels).

  .venv\\Scripts\\python.exe scripts\\make_pseudo_mot.py D:\\recordings --out D:\\data\\pseudo_mot
  (every video file in the folder becomes one sequence; RTSP URLs work too)
then:
  .venv\\Scripts\\python.exe experiments\\train_track.py --mot D:\\data\\pseudo_mot --classes 1 ...
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sarbon.data.pseudo import PseudoConfig, pseudo_label, video_frames, write_mot  # noqa: E402

VIDEO_EXT = {".mp4", ".avi", ".mkv", ".mov", ".ts", ".h264", ".h265"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="+", help="video files, folders of videos, or stream URLs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--width", type=int, default=1280, help="resize frames to this width")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--first-index", type=int, default=0, help="number sequences from here (append runs)")
    args = ap.parse_args()
    srcs = []
    for s in args.sources:
        p = Path(s)
        srcs += sorted(f for f in p.iterdir() if f.suffix.lower() in VIDEO_EXT) if p.is_dir() else [s]
    cfg = PseudoConfig(stride=args.stride)
    for k, src in enumerate(srcs, args.first_index):
        # ASCII-only folder names: OpenCV and many tools break on non-ASCII Windows paths
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(str(src)).stem).strip("_") or "video"
        name = f"cam{k:03d}_{stem}"[:60]
        # two passes over the source: labelling needs the whole sequence before writing
        labels = pseudo_label(video_frames(src, args.max_frames, args.width), cfg)
        n_img, n_box = write_mot(video_frames(src, args.max_frames, args.width), labels,
                                 Path(args.out) / name, cfg)
        n_ids = len({t for objs in labels.values() for t, _ in objs})
        print(f"{name}: {n_img} frames, {n_box} boxes, {n_ids} pseudo identities")


if __name__ == "__main__":
    main()
