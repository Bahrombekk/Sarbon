"""Video pipeline pieces: stem -> LocalBackbone conversion, detector p8 entry, block masks,
and an end-to-end CPU run (dense stem path; the sparse engine path is exercised on GPU).

Run: .venv\\Scripts\\python.exe tests\\test_pipeline.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from sarbon.models.sarbon_det import build  # noqa: E402
from sarbon.pipeline import SarbonVideo, boxes_to_block_mask  # noqa: E402
from sarbon.sparse.convert import stem_to_local  # noqa: E402
from scenes import Camera  # noqa: E402


def randomize_bn(m):
    with torch.no_grad():
        for b in m.modules():
            if isinstance(b, torch.nn.BatchNorm2d):
                b.running_mean.uniform_(-0.2, 0.2)
                b.running_var.uniform_(0.5, 2.0)
                b.weight.uniform_(0.5, 1.5)
                b.bias.uniform_(-0.2, 0.2)


def main():
    torch.manual_seed(0)
    det = build("s", 6, halo_zero=False).eval()
    randomize_bn(det)
    x = torch.randn(2, 3, 128, 192)

    with torch.no_grad():
        p8 = det.stem.forward_blocks(x)[1]
        local = stem_to_local(det.stem)
        q8 = local(x)[1]
        err = ((p8 - q8).abs().max() / p8.abs().max()).item()
        print(f"stem -> LocalBackbone (BN folded): max rel err {err:.1e}")
        assert err < 1e-5

        a = det(x)
        b = det(None, p8=p8)
        assert torch.allclose(a["pred_boxes"], b["pred_boxes"], atol=1e-5)
        assert torch.allclose(a["pred_logits"], b["pred_logits"], atol=1e-5)
        print("detector p8 entry == full forward")

    m = boxes_to_block_mask([[40, 10, 70, 90]], 4, 6, 32)
    assert m.sum() == 6 and m[0, 1] and m[2, 2] and not m[3, 3]
    m = boxes_to_block_mask([[40, 10, 70, 90]], 4, 6, 32, margin=1)
    assert m.sum() == 16
    print("block masks ok")

    # end to end on synthetic video (random weights: keep every query so the tracker works)
    cams = [Camera(352, 640, "moderate", seed=s, device="cpu") for s in range(2)]
    video = SarbonVideo(build("n", 6, halo_zero=False), n_cams=2, size=(256, 448), keep_score=0.0,
                        max_dets=30, device="cpu")
    for t in range(5):
        frames = [(c.next().numpy()[:, :, ::-1] * 255).astype(np.uint8).copy() for c in cams]
        res = video.step(frames)
        assert len(res) == 2 and all(isinstance(r["id"], int) for cam in res for r in cam)
        if t == 4:
            print(f"pipeline: 5 steps, tracks per camera {[len(r) for r in res]}, "
                  f"example {({k: v for k, v in res[0][0].items() if k != 'box'} if res[0] else None)}")
    print("OK")


if __name__ == "__main__":
    main()
