r"""Detector latency on this GPU, one table for every way of running it.

Modes (each skipped with the reason if it cannot run here):
  eager fp32 | eager fp16 (autocast) | fp16 + channels_last | torch.compile (needs Triton)
  | CUDA graph (fp16, static shapes: removes per-kernel launch overhead, which dominates a
  small DETR at batch 1) | TensorRT fp16 from the ONNX export (needs the tensorrt package)

Numbers are only meaningful on an otherwise idle GPU: the script refuses to report if another
process holds the GPU busy unless --force (then they are marked unreliable).

  python bench\speed.py --weights weights\sarbon-s-railsafe.pt --batch 1 4 --img 640
"""
import argparse
import importlib
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sarbon.cli.export import Deploy  # noqa: E402
from sarbon.data.coco import RAILSAFE  # noqa: E402
from sarbon.models.sarbon_det import build, load_weights  # noqa: E402


def gpu_util():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return int(out.split()[0])
    except Exception:
        return -1


def timeit(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(1e3 * (time.perf_counter() - t0))
    ts.sort()
    return {"ms": round(ts[len(ts) // 2], 2), "p90": round(ts[int(len(ts) * 0.9)], 2)}


MODES = ["eager fp32", "eager fp16", "fp16 + channels_last", "torch.compile fp16", "CUDA graph fp16",
         "TensorRT fp16"]


def make(name, model, x):
    """A no-argument callable that runs one forward pass in the given mode."""
    def autocast(m, inp):
        def run():
            with torch.autocast("cuda", torch.float16):
                return m(inp)
        return run

    if name == "eager fp32":
        return lambda: model(x)
    if name == "eager fp16":
        return autocast(model, x)
    if name == "fp16 + channels_last":
        return autocast(model.to(memory_format=torch.channels_last), x.contiguous(memory_format=torch.channels_last))
    if name == "torch.compile fp16":
        return autocast(torch.compile(model, mode="max-autotune-no-cudagraphs"), x)
    if name == "CUDA graph fp16":
        half, static = model.half(), x.half().clone()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):  # warm up allocations outside the capture
                half(static)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            half(static)
        return g.replay
    if name == "TensorRT fp16":
        importlib.import_module("tensorrt")  # fails with a clear reason when not installed
        raise NotImplementedError("tensorrt found: build an engine from sarbon-export's ONNX "
                                  "(trtexec --onnx=... --fp16) and time it here")
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights")
    ap.add_argument("--size", default="s")
    ap.add_argument("--img", type=int, default=640)
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--out", default="runs/speed.json")
    ap.add_argument("--force", action="store_true", help="measure even if the GPU is busy (marked unreliable)")
    args = ap.parse_args()

    util = gpu_util()
    busy = util > 10
    if busy and not args.force:
        sys.exit(f"GPU is {util}% busy; latency would be meaningless. Rerun when idle or pass --force.")
    torch.backends.cudnn.benchmark = True
    res = {"gpu": torch.cuda.get_device_name(), "img": args.img, "unreliable_gpu_busy": busy, "rows": []}
    for b in args.batch:
        x = torch.rand(b, 3, args.img, args.img, device="cuda") * 255
        for name in MODES:
            det = build(args.size, len(RAILSAFE), halo_zero=False)
            if args.weights:
                det = load_weights(det, args.weights)
            model = Deploy(det, args.img).cuda().eval()
            try:
                with torch.no_grad():
                    fn = make(name, model, x)
                    r = timeit(fn, args.warmup, args.iters)
                r = {"batch": b, "mode": name, **r, "ms_per_image": round(r["ms"] / b, 2)}
            except Exception as e:
                r = {"batch": b, "mode": name, "skipped": f"{type(e).__name__}: {str(e).splitlines()[0][:120]}"}
            res["rows"].append(r)
            print(json.dumps(r), flush=True)
            del model
            torch.cuda.empty_cache()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"{'batch':>5} {'mode':24s} {'ms':>7} {'ms/img':>7}" + ("   (GPU busy: unreliable)" if busy else ""))
    for r in res["rows"]:
        print(f"{r['batch']:5d} {r['mode']:24s} " + (f"{r['ms']:7.2f} {r['ms_per_image']:7.2f}" if "ms" in r
                                                       else f"  skipped: {r['skipped']}"))


if __name__ == "__main__":
    main()
