"""Multi-camera temporal block-sparse engine on fused Triton tile convolutions.

Per step (one frame from each of N cameras):
  1. gate graph   : per-block change mask (+ optional round-robin refresh), the dilated
                    masks every layer needs, and the active-block count -> 1 host sync.
  2. dispatch     : count == 0 -> nothing to do; count <= fallback -> sparse graph for
                    the smallest budget bucket >= count; otherwise the dense graph.
  3. sparse graph : masked image update, nonzero_static block lists, one fused Triton
                    kernel per conv (residual add fused into the second conv).
     dense graph  : plain cuDNN forward; its channels_last outputs *are* the caches
                    (NHWC memory), so the fallback costs the same as a dense network.

Active tiles from all cameras share one batch, so GPU work follows the total amount of
scene change, not the number of cameras. The fallback threshold is calibrated on the
device, so a step is never meaningfully slower than the engine's own dense path.
"""
import math
import statistics
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .block_sparse import LocalBackbone
from .triton_conv import tile_conv, tiles_per_program


@dataclass
class Layer:
    src: str
    dst: str
    res: str  # residual input cache (fused add + ReLU) or None
    act: bool
    stride: int  # conv stride
    out_stride: int  # output feature stride w.r.t. the image
    radius: int
    w_tile: torch.Tensor  # [k, k, cin, cout] for the Triton kernel
    w_dense: torch.Tensor  # [cout, cin, k, k] channels_last for cuDNN
    bias: torch.Tensor  # fp32 for the Triton epilogue
    bias_dense: torch.Tensor  # engine dtype for cuDNN


def budget_buckets(n_blocks, unit=16, ratio=1.15):
    """Fine-grained budgets: ~15% apart, multiples of `unit` (no 2x power-of-two waste)."""
    out, b = [], unit
    while b < n_blocks:
        out.append(b)
        b = max(b + unit, int(math.ceil(b * ratio / unit)) * unit)
    out.append(int(math.ceil(n_blocks / unit)) * unit)
    return out


class FusedSparseEngine:
    def __init__(self, net: LocalBackbone, n_cams, height, width, block=32,
                 dtype=torch.float16, tau=0.0, refresh=0, dense_check_every=4, device="cuda"):
        assert height % block == 0 and width % block == 0
        self.n, self.h, self.w, self.block = n_cams, height, width, block
        self.hb, self.wb = height // block, width // block
        self.n_blocks = n_cams * self.hb * self.wb
        self.dtype, self.tau, self.refresh = dtype, tau, refresh
        self.dev = device
        self.layers = self._fuse(net)
        self.ds = sorted({self.dilation(l.radius) for l in self.layers})
        self.outputs = net.program()[1]

        # NHWC caches; the layer caches are replaced by the dense graph's outputs on capture
        self.cache = {"img": self._zeros(1, 3)}
        for l in self.layers:
            self.cache[l.dst] = self._zeros(l.out_stride, l.bias.numel())

        # static buffers shared by all graphs
        self.frames = torch.zeros(n_cams, height, width, 3, device=device, dtype=dtype)
        self.changed = torch.zeros(n_cams, self.hb, self.wb, device=device, dtype=torch.bool)
        self.masks = {d: self.changed.clone() for d in self.ds}
        self.count = torch.zeros((), device=device, dtype=torch.int64)
        self.count_host = torch.zeros((), dtype=torch.int64).pin_memory()
        self.tick = torch.zeros((), device=device, dtype=torch.int64)
        self.lin = torch.arange(self.n_blocks, device=device).view(self.changed.shape)
        self.pool = torch.cuda.graph_pool_handle()
        self.buckets = budget_buckets(self.n_blocks)
        self.sparse_graphs = {}
        self.gate_graph = self._capture(self._gate)
        # The dense graph's outputs stay alive as the caches, so it must not share a pool:
        # other graphs' temporaries would land on blocks it later hands out as caches.
        self.dense_graph = self._capture(self._dense, pool=torch.cuda.graph_pool_handle())
        self.fallback = self.buckets[-1]  # until calibrate() says otherwise
        # hysteresis: while in the dense regime, look at the gate only every N steps
        self.dense_check_every, self._dense_run = dense_check_every, 0
        self.stats = {"skip": 0, "sparse": 0, "dense": 0}

    # ---------- construction ----------
    def _zeros(self, stride, c):
        return torch.zeros(self.n, self.h // stride, self.w // stride, c,
                           device=self.dev, dtype=self.dtype)

    def _fuse(self, net):
        nodes, _ = net.program()
        layers, i = [], 0
        while i < len(nodes):
            n = nodes[i]
            c = n.conv
            assert n.kind == "conv" and c.kernel_size[0] == 3 and c.padding[0] == 1
            nxt = nodes[i + 1] if i + 1 < len(nodes) else None
            fused = nxt is not None and nxt.kind == "add" and nxt.src[0] == n.dst
            w = c.weight.detach().to(self.dev)
            layers.append(Layer(
                src=n.src[0], dst=nxt.dst if fused else n.dst,
                res=nxt.src[1] if fused else None, act=True if fused else n.act,
                stride=c.stride[0], out_stride=n.stride,
                radius=nxt.radius if fused else n.radius,
                w_tile=w.permute(2, 3, 1, 0).contiguous().to(self.dtype),
                w_dense=w.to(self.dtype).contiguous(memory_format=torch.channels_last),
                bias=c.bias.detach().to(self.dev, torch.float32),
                bias_dense=c.bias.detach().to(self.dev, self.dtype)))
            i += 2 if fused else 1
        return layers

    def dilation(self, radius):
        return math.ceil(radius / self.block)

    def outputs_nchw(self):
        return [self.cache[o].permute(0, 3, 1, 2) for o in self.outputs]

    def _capture(self, body, pool=None):
        """Capture `body`. Graphs on the shared pool must not leave live outputs behind."""
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            body()  # warm-up: compiles Triton kernels, autotunes cuDNN
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=pool or self.pool):
            body()
        return g

    # ---------- graph bodies ----------
    def _gate(self):
        b = self.block
        diff = (self.frames - self.cache["img"]).abs().view(
            self.n, self.hb, b, self.wb, b, 3).sum((2, 4, 5), dtype=torch.float32)
        changed = diff > self.tau * b * b * 3
        if self.refresh:  # bounded staleness for tau > 0: every block recomputed every `refresh` steps
            changed |= (self.lin % self.refresh) == (self.tick % self.refresh)
            self.tick += 1
        self.changed.copy_(changed)
        m = changed.float()[:, None]
        for d in self.ds:
            self.masks[d].copy_((F.max_pool2d(m, 2 * d + 1, 1, d) if d else m)[:, 0] > 0)
        self.count.copy_(self.masks[self.ds[-1]].sum())
        self.count_host.copy_(self.count, non_blocking=True)

    def _sparse(self, cap):
        b = self.block
        img = self.cache["img"].view(self.n, self.hb, b, self.wb, b, 3)
        m = self.changed[:, :, None, :, None, None]
        # only changed blocks enter the image cache, so caches always describe one consistent image
        img.copy_(torch.where(m, self.frames.view(img.shape), img))
        lists, per_cam = {}, self.hb * self.wb
        for d in self.ds:
            flat = torch.nonzero_static(self.masks[d].flatten(), size=cap, fill_value=0)[:, 0]
            r = flat % per_cam
            lists[d] = ((flat // per_cam).int(), (r // self.wb).int(), (r % self.wb).int())
        for l in self.layers:
            cam, by, bx = lists[self.dilation(l.radius)]
            tile_conv(self.cache[l.src], l.w_tile, l.bias, self.cache[l.dst], cam, by, bx,
                      b // l.out_stride, l.stride, 3, 1, l.act,
                      None if l.res is None else self.cache[l.res])

    def _dense(self):
        self.cache["img"].copy_(self.frames)
        for l in self.layers:
            x = self.cache[l.src].permute(0, 3, 1, 2)  # NHWC memory == channels_last NCHW
            y = F.conv2d(x, l.w_dense, l.bias_dense, l.stride, 1)
            if l.res is not None:
                y = y + self.cache[l.res].permute(0, 3, 1, 2)
            if l.act:
                y = F.relu_(y)
            y = y.permute(0, 2, 3, 1)
            if not y.is_contiguous():  # cuDNN picked NCHW output: pay one copy
                y = y.contiguous()
            # no copy: the output tensor becomes the cache the sparse graphs update
            self.cache[l.dst] = y

    # ---------- runtime ----------
    def _bucket(self, k):
        lo = max(tiles_per_program(self.block // l.out_stride) for l in self.layers)
        for b in self.buckets:
            if b >= k and b % lo == 0:
                return b
        return self.buckets[-1]

    def _sparse_graph(self, cap):
        g = self.sparse_graphs.get(cap)
        if g is None:
            g = self.sparse_graphs[cap] = self._capture(lambda: self._sparse(cap))
        return g

    @torch.no_grad()
    def reset(self, frames):
        """Dense initialisation from [N, H, W, 3] frames."""
        self.frames.copy_(frames)
        self.dense_graph.replay()
        self._dense_run = 0

    @torch.no_grad()
    def step(self, frames):
        """Process one [N, H, W, 3] frame batch. Returns the mode used and the active count.

        The count is -1 for dense steps taken without looking at the gate (hysteresis):
        the dense path recomputes everything, so skipping the gate there is always exact.
        """
        self.frames.copy_(frames)
        if 0 < self._dense_run < self.dense_check_every:
            self._dense_run += 1
            self.dense_graph.replay()
            self.stats["dense"] += 1
            return "dense", -1
        self.gate_graph.replay()
        torch.cuda.current_stream().synchronize()
        k = int(self.count_host)
        if k == 0:
            mode = "skip"
        elif k <= self.fallback:
            self._sparse_graph(self._bucket(k)).replay()
            mode = "sparse"
        else:
            self.dense_graph.replay()
            mode = "dense"
        self._dense_run = 1 if mode == "dense" else 0
        self.stats[mode] += 1
        return mode, k

    @torch.no_grad()
    def calibrate(self, reps=15, margin=0.95, dense_ref=None):
        """Time every budget bucket against a dense reference; set the fallback crossover.

        Sparse and dense replays are interleaved and compared by their best times: on a
        GPU shared with other jobs the minimum is the least contended estimate.
        `dense_ref` (a callable) defaults to the engine's own dense graph. Replays use
        the current caches/masks; that changes which blocks are computed, not how many.
        """
        def once(fn):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            return time.perf_counter() - t0

        dense = dense_ref or self.dense_graph.replay
        saved = {k: v.clone() for k, v in self.cache.items()}
        self.fallback, table, above = 0, [], 0
        for b in sorted({self._bucket(x) for x in self.buckets}):
            g = self._sparse_graph(b)
            g.replay()
            pairs = [(once(g.replay), once(dense)) for _ in range(reps)]
            ratio = min(s for s, _ in pairs) / min(d for _, d in pairs)
            table.append((b, min(s for s, _ in pairs) * 1e3, ratio))
            if ratio > margin:
                above += 1
                if above == 2:  # two in a row: a real crossover, not a contention spike
                    break
            else:
                above, self.fallback = 0, b
        for k, v in saved.items():
            self.cache[k].copy_(v)
        return table
