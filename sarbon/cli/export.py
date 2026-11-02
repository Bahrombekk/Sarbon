r"""Export the detector to ONNX for deployment (TensorRT / ONNX Runtime) and check it.

The graph takes raw RGB pixels (float, 0..255, NCHW, fixed size) and does the normalisation itself,
so a deployment only letterboxes and feeds the frame. Outputs are NMS-free:
  scores [B, Q, C] (sigmoid), boxes [B, Q, 4] (xyxy, input pixels), embed [B, Q, D] (for tracking).

  sarbon-export --weights weights\sarbon-s-railsafe.pt --out weights\sarbon-s-railsafe.onnx
  sarbon-export ... --batch 4 --img 640       (fixed batch, e.g. 4 cameras per call)

After export the ONNX graph is run with ONNX Runtime on the same random frames and compared with
PyTorch (fp32); the max differences are printed and written next to the file.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..data.coco import RAILSAFE
from ..engine.training import MEAN, STD
from ..models.box_ops import cxcywh_to_xyxy
from ..models.sarbon_det import load_detector
from ..paths import DEFAULT_DETECTOR


class Deploy(torch.nn.Module):
    """Detector with input normalisation and output decoding inside the graph."""

    def __init__(self, det, img):
        super().__init__()
        self.det, self.img = det, img
        self.register_buffer("mean", MEAN.reshape(1, 3, 1, 1).float())
        self.register_buffer("std", STD.reshape(1, 3, 1, 1).float())

    def forward(self, x):
        out = self.det((x - self.mean) / self.std)
        boxes = cxcywh_to_xyxy(out["pred_boxes"]) * self.img
        return out["pred_logits"].sigmoid(), boxes, out.get("id_embed", out["embed"])


def export(model, path, batch, img, opset):
    """Trace with the model's own device and dtype (a half model on CUDA gives an fp16 graph)."""
    p = next(model.parameters())
    x = (torch.rand(batch, 3, img, img, device=p.device) * 255).to(p.dtype)
    # the fused eval fast path of nn.TransformerEncoderLayer (AIFI) has no ONNX symbolic
    torch.backends.mha.set_fastpath_enabled(False)
    torch.onnx.export(model, (x,), str(path), input_names=["images"], output_names=["scores", "boxes", "embed"],
                      opset_version=opset, do_constant_folding=True, dynamo=False)
    return x


def check(model, path, batch, img, runs=5):
    """ONNX Runtime vs PyTorch on random frames: max |diff| of scores and boxes, ORT latency."""
    import onnxruntime as ort

    have = ort.get_available_providers()
    sess = ort.InferenceSession(str(path), providers=[p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                                                      if p in have])
    rng = np.random.default_rng(0)
    x = (rng.random((batch, 3, img, img)) * 255).astype(np.float32)
    with torch.no_grad():
        ref = [t.numpy() for t in model(torch.from_numpy(x))]
    got = sess.run(None, {"images": x})
    t0 = time.perf_counter()
    for _ in range(runs):
        sess.run(None, {"images": x})
    return {"provider": sess.get_providers()[0],
            "max_abs_diff": {n: float(np.abs(a - b).max()) for n, a, b in zip(("scores", "boxes", "embed"), ref, got)},
            "ort_ms": 1e3 * (time.perf_counter() - t0) / runs}


def main(argv=None):
    from ..config import parse_with_config

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=str(DEFAULT_DETECTOR))
    ap.add_argument("--out", help="default: weights path with .onnx")
    ap.add_argument("--size", default="s")
    ap.add_argument("--classes", type=int, default=len(RAILSAFE))
    ap.add_argument("--img", type=int, default=640)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--half", action="store_true",
                    help="fp16 graph (traced on CUDA): what TensorRT >= 11 builds an fp16 engine from")
    ap.add_argument("--no-check", action="store_true", help="skip the ONNX Runtime comparison (fp32 only)")
    args = parse_with_config(ap, argv)

    det = load_detector(args.weights, args.size, args.classes, halo_zero=False).eval()
    model = Deploy(det, args.img).eval()
    if args.half:
        model = model.cuda().half()
    out = Path(args.out or Path(args.weights).with_suffix(".fp16.onnx" if args.half else ".onnx"))
    out.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        export(model, out, args.batch, args.img, args.opset)
    info = {"weights": str(args.weights), "img": args.img, "batch": args.batch, "opset": args.opset,
            "dtype": "float16" if args.half else "float32",
            "classes": RAILSAFE if args.classes == len(RAILSAFE) else args.classes,
            "mb": round(out.stat().st_size / 2 ** 20, 1)}
    if not (args.no_check or args.half):
        info["check"] = check(model, out, args.batch, args.img)
    out.with_suffix(".json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
