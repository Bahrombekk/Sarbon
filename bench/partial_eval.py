r"""Skip vs cheap compute on the frames between full detector runs, at equal GPU cost.

Plans (per camera): full every N-th frame and on the frames between either
  skip     coast the tracker (what bench/scheduler_eval.py measured)
  low-S    run the detector at a smaller input size S (dense reference at that size)
  only-S   every frame at size S (no full runs)
Cost of a plan = full runs + cheap runs x the measured cost ratio of size S at batch 8 (many
cameras per call, the deployment case). Scores are fidelity to the every-frame full-size
pipeline (sarbon.schedule.simulate).

  python bench\partial_eval.py --full C:\data\recordings\dense ^
      --low 448=C:\data\recordings\dense_448=0.58 320=C:\data\recordings\dense_320=0.41
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sarbon.data.coco import RAILSAFE  # noqa: E402
from sarbon.schedule.policies import EveryFrame  # noqa: E402
from sarbon.schedule.simulate import load, run, run_plan, score  # noqa: E402

CLASSES = {RAILSAFE.index(c): c for c in ("person", "yengil", "ogir", "train")}


def plans(n_frames, lows):
    t = np.arange(n_frames)
    for n in (2, 3, 4, 6, 8):
        full = t % n == 0
        yield f"full/{n} + skip", n, np.where(full, 0, -1), full.mean()
        for k, (name, _, ratio) in enumerate(lows, 1):
            yield f"full/{n} + {name}", n, np.where(full, 0, k), full.mean() + (~full).mean() * ratio
    for k, (name, _, ratio) in enumerate(lows, 1):
        yield f"only {name}", 1, np.full(n_frames, k), ratio


def evaluate(args):
    full_path, lows, frames = args
    ref = load(full_path)
    srcs = [load(Path(d) / Path(full_path).name) for _, d, _ in lows]
    n = min([frames] + [len(s["off"]) - 1 for s in srcs])
    reference, _ = run(ref, EveryFrame(), n)
    rows = []
    for name, knob, plan, cost in plans(n, lows):
        out, mask = run_plan(ref, srcs, plan)
        rows.append({"camera": Path(full_path).stem, "plan": name, "stride": knob, "cost": float(cost),
                     **score(reference, out, mask, ref["fps"], CLASSES)})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", required=True, help="dense reference folder at the full input size")
    ap.add_argument("--low", nargs="+", required=True, metavar="NAME=DIR=COST")
    ap.add_argument("--frames", type=int, default=12000)
    ap.add_argument("--out", default="runs/partial")
    args = ap.parse_args()
    lows = [(f"low-{a}", b, float(c)) for a, b, c in (s.split("=") for s in args.low)]
    files = sorted(Path(args.full).glob("*.npz"))
    with ProcessPoolExecutor(len(files)) as ex:
        rows = [r for rs in ex.map(evaluate, [(str(f), lows, args.frames) for f in files]) for r in rs]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    keys = ["cost", "recall", "precision", "delay_p95_s", "missed", "recall_person"]
    groups = {}
    for r in rows:
        groups.setdefault(r["plan"], []).append(r)
    summary = sorted(({"plan": p, **{k: float(np.mean([r.get(k, np.nan) for r in rs])) for k in keys}}
                      for p, rs in groups.items()), key=lambda s: s["cost"])
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"{'plan':20s} " + " ".join(f"{k:>13s}" for k in keys))
    for s in summary:
        print(f"{s['plan']:20s} " + " ".join(f"{s[k]:13.3f}" for k in keys))


if __name__ == "__main__":
    main()
