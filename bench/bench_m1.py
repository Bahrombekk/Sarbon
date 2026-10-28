"""M1 benchmark: fused multi-camera engine vs dense cuDNN on synthetic fixed-camera video.

Robust to a shared GPU: dense baseline and Sarbon are timed back-to-back on every
frame (A/B interleaved), and the speedup is the median of the per-frame ratios.
Wall-clock per frame batch includes the host sync; the dense baseline is CUDA-graphed
(its most favourable setting).
Run: .venv\\Scripts\\python.exe bench\\bench_m1.py
"""
import gc
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sarbon.sparse import LocalBackbone  # noqa: E402
from sarbon.sparse.fused_engine import FusedSparseEngine  # noqa: E402
from scenes import Camera, make_sequence  # noqa: E402

H, W, DT = 1088, 1920, torch.float16


class DenseBaseline:
    def __init__(self, net, frames):
        self.net = net
        self.x = frames.permute(0, 3, 1, 2).clone()  # NHWC frames == channels_last NCHW
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            net(self.x)
        torch.cuda.current_stream().wait_stream(s)
        self.g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g), torch.no_grad():
            net(self.x)

    def __call__(self, frames):
        self.x.copy_(frames.permute(0, 3, 1, 2))
        self.g.replay()


def once(fn, *a):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn(*a)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3, out


def scenario(name, presets, n_frames=40, passes=2, block=32, show_table=False):
    torch.manual_seed(0)
    net = LocalBackbone().to("cuda", DT).eval().to(memory_format=torch.channels_last)
    cams = [Camera(H, W, p, seed=i) for i, p in enumerate(presets)]
    seq = make_sequence(cams, n_frames, DT)
    dense = DenseBaseline(net, seq[0])
    eng = FusedSparseEngine(net, len(cams), H, W, block=block, dtype=DT)
    eng.reset(seq[0])
    table = eng.calibrate()
    td, ts, ratios, ks = [], [], [], []
    modes = {"skip": 0, "sparse": 0, "dense": 0}
    for _ in range(passes):
        eng.reset(seq[0])
        for f in seq[1:]:
            a, _ = once(dense, f)
            b, (mode, k) = once(eng.step, f)
            td.append(a)
            ts.append(b)
            ratios.append(a / b)
            if k >= 0:  # -1: dense step taken without a gate look (hysteresis)
                ks.append(k)
            modes[mode] += 1
    med = statistics.median
    print(f"\n[{name}] {len(cams)} cam x {W}x{H}, block={block}")
    print(f"  dense cuDNN graphed   median {med(td):8.3f} ms   best {min(td):8.3f} ms")
    print(f"  Sarbon                median {med(ts):8.3f} ms   best {min(ts):8.3f} ms")
    print(f"  speedup (median of per-frame A/B ratios) x{med(ratios):.2f}   "
          f"best-vs-best x{min(td) / min(ts):.2f}")
    print(f"  active blocks mean {statistics.mean(ks) / eng.n_blocks:.1%}   fallback above "
          f"{eng.fallback / eng.n_blocks:.0%}   modes {modes}")
    if show_table:
        print("  calibration budget(blocks): ms sparse/dense ->",
              " ".join(f"{b}:{ms:.2f}/{r:.2f}" for b, ms, r in table[::max(1, len(table) // 12)]))
    del eng, dense, seq, cams, net
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    print(torch.cuda.get_device_name(0))
    scenario("1 cam quiet", ["quiet"], show_table=True)
    scenario("1 cam moderate", ["moderate"])
    scenario("1 cam busy", ["busy"])
    scenario("16 cam all quiet", ["quiet"] * 16, n_frames=12)
    mix = ["quiet"] * 10 + ["moderate"] * 4 + ["busy"] * 2
    scenario("16 cam mixed (10 quiet / 4 moderate / 2 busy)", mix, n_frames=12, show_table=True)
    scenario("16 cam all busy (worst case)", ["busy"] * 16, n_frames=12)
