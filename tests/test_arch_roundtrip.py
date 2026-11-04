"""A checkpoint rebuilds as the architecture it was trained with (load_detector), including the
light neck/decoder sizes; checkpoints from before n_queries was saved still load.

CPU-only. Run: .venv\Scripts\python.exe tests\test_arch_roundtrip.py
"""
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sarbon.models.sarbon_det import arch_from_state_dict, build, load_detector  # noqa: E402

LIGHT = dict(widths=(32, 64, 128), depths=(1, 1), c16=256, queries=150, dec_layers=3,
             hidden=192, heads=6, ffn=768)
WIDE = dict(widths=(48, 96, 192), depths=(2, 3), c16=384)


def main():
    torch.manual_seed(0)
    x = torch.randn(1, 3, 128, 192)
    with tempfile.TemporaryDirectory() as d:
        for name, arch in (("light", LIGHT), ("wide", WIDE)):
            m = build("s", 6, halo_zero=False, **arch).eval()
            path = Path(d) / f"{name}.pt"
            sd = m.state_dict()
            got = arch_from_state_dict(sd)
            for k, v in arch.items():
                assert got[k] == v, (name, k, got[k], v)
            torch.save(sd, path)
            r = load_detector(path, "s", 6, halo_zero=False).eval()
            assert r.cfg.queries == m.cfg.queries and r.cfg.hidden == m.cfg.hidden
            with torch.no_grad():
                a, b = m(x), r(x)
            err = (a["pred_boxes"] - b["pred_boxes"]).abs().max().item()
            assert err == 0.0, (name, err)
            # older checkpoint: no n_queries saved -> the preset's query count, still loads
            old = {k: v for k, v in sd.items() if k != "n_queries"}
            torch.save(old, path)
            r = load_detector(path, "s", 6, halo_zero=False)
            assert r.cfg.queries == 300
            print(f"{name}: {sum(p.numel() for p in m.parameters()) / 1e6:.1f}M params, round trip exact")
    print("OK")


if __name__ == "__main__":
    main()
