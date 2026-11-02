"""TensorRT engine for the exported detector: build once from ONNX, run on torch CUDA tensors.

TensorRT >= 11 builds strongly typed networks only: the precision comes from the ONNX graph, so an
fp16 engine is built from `sarbon-export --half`. The engine is tied to the GPU model and the
TensorRT version it was built with; build it on the deployment machine (it takes a few minutes).

  from sarbon.inference.trt import build_engine, TrtDetector
  build_engine("weights/sarbon-s-railsafe.fp16.onnx", "weights/sarbon-s-railsafe.fp16.engine")
  det = TrtDetector("weights/sarbon-s-railsafe.fp16.engine")
  scores, boxes, embed = det(images)   # images: CUDA tensor [B, 3, H, W], RGB 0..255
"""
from pathlib import Path

import torch

_DTYPES = {}


def _trt():
    import tensorrt as trt

    if not _DTYPES:
        _DTYPES.update({trt.float32: torch.float32, trt.float16: torch.float16, trt.int32: torch.int32,
                        trt.int64: torch.int64, trt.bool: torch.bool})
    return trt


def build_engine(onnx_path, engine_path, workspace_gb=4, opt_level=3):
    """Parse the ONNX graph and serialise an engine next to it; returns the engine path."""
    trt = _trt()
    log = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(log)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(net, log)
    if not parser.parse_from_file(str(onnx_path)):
        errs = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT could not parse {onnx_path}:\n{errs}")
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    cfg.builder_optimization_level = opt_level
    plan = builder.build_serialized_network(net, cfg)
    if plan is None:
        raise RuntimeError("TensorRT engine build failed (see the log above)")
    Path(engine_path).write_bytes(bytes(plan))
    return Path(engine_path)


class TrtDetector:
    """Runs a static-shape engine on the current CUDA stream; outputs are reused buffers."""

    def __init__(self, engine_path, device="cuda"):
        trt = _trt()
        self.dev = torch.device(device)
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        self.ctx = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.outputs = [n for n in names if n not in self.inputs]
        self.buf = {}
        for n in names:
            shape = tuple(self.engine.get_tensor_shape(n))
            self.buf[n] = torch.empty(shape, dtype=_DTYPES[self.engine.get_tensor_dtype(n)], device=self.dev)
            self.ctx.set_tensor_address(n, self.buf[n].data_ptr())
        self.input_shape = tuple(self.buf[self.inputs[0]].shape)
        self.stream = torch.cuda.Stream(self.dev)

    def __call__(self, images):
        # TensorRT syncs extra on the default stream, so it runs on its own, ordered after the caller
        cur = torch.cuda.current_stream(self.dev)
        self.stream.wait_stream(cur)
        with torch.cuda.stream(self.stream):
            self.buf[self.inputs[0]].copy_(images)  # casts to the engine's input dtype
            if not self.ctx.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError("TensorRT execution failed")
        cur.wait_stream(self.stream)
        return tuple(self.buf[n] for n in self.outputs)
