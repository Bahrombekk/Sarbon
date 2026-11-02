r"""Accuracy / compute trade-off of camera compute policies on recorded video (Pareto table + chart).

Needs dense references from tools/dense_reference.py. Every policy is scored against the
pipeline that computes every frame (sarbon.schedule.simulate explains the metrics).

  python bench\scheduler_eval.py C:\data\recordings\dense --frames 6000 --out runs\scheduler
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sarbon.data.coco import RAILSAFE  # noqa: E402
from sarbon.schedule.policies import EveryFrame, FixedStride, MotionGate, OracleDebt, PerceptionDebt, RoundRobin  # noqa: E402
from sarbon.schedule.simulate import load, run, run_shared, score  # noqa: E402

CLASSES = {RAILSAFE.index(c): c for c in ("person", "yengil", "ogir", "train")}


def policies(size):
    yield "fixed", 1, FixedStride(1)
    for n in (2, 3, 4, 6, 8, 12):
        yield "fixed", n, FixedStride(n)
    for th in (0.5, 1, 2, 4, 8):
        yield "motion", th, MotionGate(th)
    for s in (0.25, 0.5, 0.75, 1, 1.5, 2, 3):
        yield "debt", s, PerceptionDebt(s, img_size=size)


def evaluate(args):
    path, frames = args
    ref = load(path)
    reference, _ = run(ref, EveryFrame(), frames)
    rows = []
    for family, knob, pol in policies(ref["size"]):
        out, mask = run(ref, pol, frames)
        rows.append({"camera": Path(path).stem, "family": family, "knob": knob,
                     **score(reference, out, mask, ref["fps"], CLASSES)})
    return rows


def shared(args):
    """All cameras on one budget of `k` detector runs per tick: debt-ranked vs round robin."""
    files, frames, k, family = args
    refs = [load(f) for f in files]
    frames = min([frames or 10 ** 9] + [len(r["thumb"]) for r in refs])
    references = [run(r, EveryFrame(), frames)[0] for r in refs]
    if family == "round robin":
        pols = [RoundRobin() for _ in refs]
    elif family == "oracle":
        pols = [OracleDebt(ref) for ref in references]
    else:
        pols = [PerceptionDebt(0.5, img_size=r["size"]) for r in refs]
    rows = []
    for r, f, reference, (out, mask) in zip(refs, files, references, run_shared(refs, pols, k, frames)):
        rows.append({"camera": Path(f).stem, "family": f"shared {family}", "knob": k,
                     **score(reference, out, mask, r["fps"], CLASSES)})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dense_dir")
    ap.add_argument("--frames", type=int, default=0, help="frames per camera (0 = all)")
    ap.add_argument("--out", default="runs/scheduler")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--shared", action="store_true",
                    help="cameras share a budget of K detector runs per tick (K = 1 .. cameras-1)")
    ap.add_argument("--families", nargs="+", default=["round robin", "debt"],
                    help="shared-budget schedulers: round robin, debt, oracle (upper bound)")
    args = ap.parse_args()
    files = sorted(Path(args.dense_dir).glob("*.npz"))
    if args.shared:
        jobs = [([str(f) for f in files], args.frames, k, fam) for k in range(1, len(files))
                for fam in args.families]
        with ProcessPoolExecutor(args.workers) as ex:
            rows = [r for rs in ex.map(shared, jobs) for r in rs]
        report(rows, Path(args.out) / "shared")
        return
    with ProcessPoolExecutor(args.workers) as ex:
        rows = [r for rs in ex.map(evaluate, [(str(f), args.frames or None) for f in files]) for r in rs]
    report(rows, Path(args.out))


def report(rows, out):
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")

    keys = ["compute", "recall", "precision", "delay_mean_s", "delay_p95_s", "missed", "recall_person"]
    table = {}
    for r in rows:
        table.setdefault((r["family"], r["knob"]), []).append(r)
    summary = []
    print(f"{'policy':14s} " + " ".join(f"{k:>13s}" for k in keys))
    for (fam, knob), rs in table.items():
        avg = {k: float(np.mean([r[k] for r in rs if k in r])) if any(k in r for r in rs) else float("nan")
               for k in keys}
        summary.append({"family": fam, "knob": knob, **avg})
        print(f"{fam + ' ' + format(knob, 'g'):14s} " + " ".join(f"{avg[k]:13.3f}" for k in keys))
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for fam, mark in zip(sorted({s["family"] for s in summary}), ("s-", "^-", "o-", "d-")):
            pts = sorted((s["compute"], s["recall"], s["delay_p95_s"]) for s in summary if s["family"] == fam)
            axes[0].plot([p[0] for p in pts], [p[1] for p in pts], mark, label=fam)
            axes[1].plot([p[0] for p in pts], [p[2] for p in pts], mark, label=fam)
        axes[0].set(xlabel="detector runs (fraction of frames)", ylabel="recall vs every-frame pipeline")
        axes[1].set(xlabel="detector runs (fraction of frames)", ylabel="new-object delay p95 (s)")
        for a in axes:
            a.grid(alpha=0.3)
            a.legend()
        fig.tight_layout()
        fig.savefig(out / "pareto.png", dpi=120)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
