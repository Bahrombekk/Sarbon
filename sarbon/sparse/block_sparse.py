"""Block-sparse temporal execution for fixed cameras.

Core idea: between consecutive frames of a static camera most pixels do not
change, so the high-resolution, purely *local* part of the backbone only needs
to be recomputed in blocks whose receptive field touches a changed pixel.
Every layer keeps a zero-padded cache of its last output; a frame update
gathers active tiles (with the halo the conv needs), runs dense cuDNN convs on
the tile batch, and scatters the results back into the cache.

With a change threshold of 0 this is *lossless*: the cache equals what a dense
forward pass on the new frame would produce. Only local ops (conv, pointwise
activation, residual add) are allowed here; global ops (attention, SE, pooling)
belong to the low-resolution dense part of the network.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

CACHE_PAD = 1  # every cache carries a 1-pixel zero border (enough for 3x3 convs)


@dataclass
class Node:
    kind: str            # "conv" or "add"
    src: tuple           # input cache names
    dst: str             # output cache name
    stride: int          # output feature stride w.r.t. the image
    radius: int          # cumulative receptive radius in image pixels
    conv: nn.Conv2d = None
    act: bool = True


class ConvAct(nn.Module):
    """Conv with folded BN (bias) + ReLU; the inference form of Conv-BN-ReLU."""

    def __init__(self, cin, cout, k=3, s=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, s, k // 2, bias=True)
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")
        nn.init.normal_(self.conv.bias, std=0.01)
        self.act = act

    def forward(self, x):
        x = self.conv(x)
        return F.relu(x) if self.act else x


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = ConvAct(c, c)
        self.c2 = ConvAct(c, c, act=False)
        self.c2.conv.weight.data *= 0.5  # keep activations bounded with random init

    def forward(self, x):
        return F.relu(self.c2(self.c1(x)) + x)


class LocalBackbone(nn.Module):
    """High-resolution local stages (stride 4 and 8), ResNet-18-like widths."""

    def __init__(self, w=(32, 64, 128), blocks=(2, 2)):
        super().__init__()
        self.stem1 = ConvAct(3, w[0], s=2)
        self.stem2 = ConvAct(w[0], w[1], s=2)
        self.stage4 = nn.Sequential(*[ResBlock(w[1]) for _ in range(blocks[0])])
        self.down = ConvAct(w[1], w[2], s=2)
        self.stage8 = nn.Sequential(*[ResBlock(w[2]) for _ in range(blocks[1])])

    def forward(self, x):
        p4 = self.stage4(self.stem2(self.stem1(x)))
        p8 = self.stage8(self.down(p4))
        return p4, p8

    def program(self):
        """Flatten the module into a list of Nodes the sparse engine can execute."""
        nodes, state = [], {"name": "img", "stride": 1, "radius": 0, "i": 0}

        def conv(unit, src=None):
            c = unit.conv
            s_in = state["stride"] if src is None else src[1]
            r_in = state["radius"] if src is None else src[2]
            name_in = state["name"] if src is None else src[0]
            radius = r_in + (c.kernel_size[0] // 2) * s_in
            stride = s_in * c.stride[0]
            dst = f"t{state['i']}"
            state["i"] += 1
            nodes.append(Node("conv", (name_in,), dst, stride, radius, c, unit.act))
            state.update(name=dst, stride=stride, radius=radius)
            return dst

        def res(block):
            x = (state["name"], state["stride"], state["radius"])
            conv(block.c1)
            conv(block.c2)
            dst = f"t{state['i']}"
            state["i"] += 1
            radius = max(state["radius"], x[2])
            nodes.append(Node("add", (state["name"], x[0]), dst, x[1], radius))
            state.update(name=dst, radius=radius)

        conv(self.stem1)
        conv(self.stem2)
        for b in self.stage4:
            res(b)
        p4 = state["name"]
        conv(self.down)
        for b in self.stage8:
            res(b)
        return nodes, (p4, state["name"])


def _bucket(n):
    """Round the active-block count up so cuDNN/TensorRT see few distinct shapes."""
    if n <= 16:
        return 16
    return 1 << math.ceil(math.log2(n))


class BlockSparseEngine:
    """Runs a LocalBackbone program densely once, then sparsely per frame."""

    def __init__(self, net: LocalBackbone, block=32):
        self.net = net.eval()
        self.block = block
        self.nodes, self.outputs = net.program()
        for n in self.nodes:
            assert n.stride <= block and block % n.stride == 0, "tile must be >= 1 px"
        self.cache = {}

    # ---------- cache helpers ----------
    def _alloc(self, name, c, h, w, like):
        self.cache[name] = like.new_zeros(c, h + 2 * CACHE_PAD, w + 2 * CACHE_PAD)

    def _interior(self, name):
        p = CACHE_PAD
        return self.cache[name][:, p:-p, p:-p]

    def _gather(self, cache, blocks, t_out, s, k, conv_pad):
        """Gather K input windows that produce K output tiles of size t_out."""
        by, bx, _ = blocks
        c, _, wp = cache.shape
        win = (t_out - 1) * s + k
        key = (id(blocks), t_out, s, k, conv_pad, wp)
        idx = self._idx.get(key)
        if idx is None:  # index maps are shared by every node with the same geometry
            off = CACHE_PAD - conv_pad
            r = torch.arange(win, device=cache.device)
            rows = (by * t_out * s + off)[:, None] + r
            cols = (bx * t_out * s + off)[:, None] + r
            idx = (rows[:, :, None] * wp + cols[:, None, :]).reshape(len(by), -1)
            self._idx[key] = idx
        tiles = cache.reshape(c, -1)[:, idx]
        return tiles.permute(1, 0, 2).reshape(len(by), c, win, win)

    def _scatter(self, name, tiles, by, bx, t):
        v = self._interior(name)
        c, h, w = v.shape
        v.view(c, h // t, t, w // t, t).permute(1, 3, 0, 2, 4)[by, bx] = tiles

    # ---------- execution ----------
    @torch.no_grad()
    def full(self, frame):
        """Dense pass over a [3,H,W] frame that (re)initialises every cache."""
        _, h, w = frame.shape
        assert h % self.block == 0 and w % self.block == 0
        self._alloc("img", 3, h, w, frame)
        self._interior("img").copy_(frame)
        maps = {"img": frame[None]}
        for n in self.nodes:
            if n.kind == "conv":
                y = F.conv2d(maps[n.src[0]], n.conv.weight, n.conv.bias,
                             n.conv.stride, n.conv.padding)
                y = F.relu(y) if n.act else y
            else:
                y = F.relu(maps[n.src[0]] + maps[n.src[1]])
            maps[n.dst] = y
            self._alloc(n.dst, y.shape[1], y.shape[2], y.shape[3], frame)
            self._interior(n.dst).copy_(y[0])
        return [self._interior(o) for o in self.outputs]

    def dilation(self, radius):
        return math.ceil(radius / self.block)

    def active_blocks(self, changed, radius):
        """Blocks whose receptive field (radius in image px) touches a changed block."""
        d = self.dilation(radius)
        m = changed.float()[None, None]
        if d:
            m = F.max_pool2d(m, 2 * d + 1, 1, d)
        return m[0, 0] > 0

    def _block_lists(self, changed):
        """Active block coordinates for every distinct dilation, with one host sync."""
        ds = sorted({self.dilation(n.radius) for n in self.nodes})
        masks = torch.stack([self.active_blocks(changed, d * self.block) for d in ds])
        counts = masks.flatten(1).sum(1).tolist()  # the only device->host sync per frame
        lists = {}
        for d, m, k in zip(ds, masks, counts):
            if k == 0:
                lists[d] = None
                continue
            by, bx = m.nonzero(as_tuple=True)
            pad = _bucket(k) - k  # duplicate writes of block 0 are idempotent
            lists[d] = (torch.cat([by, by[:1].expand(pad)]),
                        torch.cat([bx, bx[:1].expand(pad)]), k)
        return lists

    @torch.no_grad()
    def update(self, frame, changed, graph=True):
        """Sparse update. `changed` is a [H/block, W/block] bool mask of changed blocks.

        With graph=True the per-node work is replayed from a CUDA graph keyed by the
        bucketed active-block counts, which removes the Python/launch overhead.
        """
        lists = self._block_lists(changed)
        if not graph:
            self._interior("img").copy_(frame)
            return self._run(lists)
        key = tuple((d, None if b is None else len(b[0])) for d, b in sorted(lists.items()))
        g = self._graphs.get(key) if hasattr(self, "_graphs") else None
        if g is None:
            self._graphs = getattr(self, "_graphs", {})
            frame_buf = frame.clone()
            static = {d: None if b is None else (b[0].clone(), b[1].clone(), b[2])
                      for d, b in lists.items()}

            def body():
                self._interior("img").copy_(frame_buf)
                self._run(static)

            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                body()  # warm-up also autotunes cuDNN for these shapes
            torch.cuda.current_stream().wait_stream(s)
            cg = torch.cuda.CUDAGraph()
            with torch.cuda.graph(cg):
                body()
            g = self._graphs[key] = (cg, frame_buf, static)
        cg, frame_buf, static = g
        frame_buf.copy_(frame)
        for d, b in lists.items():
            if b is not None:
                static[d][0].copy_(b[0])
                static[d][1].copy_(b[1])
        cg.replay()
        return [self._interior(o) for o in self.outputs]

    @torch.no_grad()
    def step(self, frame, capacity, tau=0.0):
        """Gate + sparse update for the next frame as ONE CUDA graph with no host sync.

        `capacity` is a fixed per-frame budget of recomputed blocks, so latency is
        bounded by construction. Returns (outputs, overflow): overflow is a 0-dim
        bool tensor, True when more blocks changed than the budget; the caller
        should then schedule a dense refresh (checked lazily, off the hot path).
        """
        g = getattr(self, "_step_graphs", {}).get(capacity)
        if g is None:
            self._step_graphs = getattr(self, "_step_graphs", {})
            hb, wb = (s // self.block for s in self._interior("img").shape[1:])
            ds = sorted({self.dilation(n.radius) for n in self.nodes})
            bufs = {"frame": frame.clone(), "prev": self._interior("img").clone()}

            def body():
                changed = self.changed_blocks(bufs["frame"], bufs["prev"], tau)
                lists, over = {}, torch.zeros((), dtype=torch.bool, device=frame.device)
                for d in ds:
                    m = self.active_blocks(changed, d * self.block).flatten()
                    # padding entries point at block 0: recomputing it is redundant, not wrong
                    flat = torch.nonzero_static(m, size=capacity, fill_value=0)[:, 0]
                    lists[d] = (flat // wb, flat % wb, capacity)
                    over = over | (m.sum() > capacity)
                self._interior("img").copy_(bufs["frame"])
                self._run(lists)
                bufs["prev"].copy_(bufs["frame"])
                bufs["over"] = over

            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                body()
            torch.cuda.current_stream().wait_stream(s)
            cg = torch.cuda.CUDAGraph()
            with torch.cuda.graph(cg):
                body()
            g = self._step_graphs[capacity] = (cg, bufs)
        cg, bufs = g
        bufs["frame"].copy_(frame)
        cg.replay()
        return [self._interior(o) for o in self.outputs], bufs["over"]

    def reset_prev(self, frame):
        """Point the step graphs' reference frame at `frame` (after full())."""
        for _, bufs in getattr(self, "_step_graphs", {}).values():
            bufs["prev"].copy_(frame)

    def _run(self, lists):
        self._idx = {}
        for n in self.nodes:
            blocks = lists[self.dilation(n.radius)]
            if blocks is None:
                continue
            by, bx, _ = blocks
            t = self.block // n.stride
            if n.kind == "conv":
                c = n.conv
                tiles = self._gather(self.cache[n.src[0]], blocks, t, c.stride[0],
                                     c.kernel_size[0], c.padding[0])
                y = F.conv2d(tiles, c.weight, c.bias, c.stride, 0)
                y = F.relu(y) if n.act else y
            else:
                a = self._gather(self.cache[n.src[0]], blocks, t, 1, 1, 0)
                b = self._gather(self.cache[n.src[1]], blocks, t, 1, 1, 0)
                y = F.relu(a + b)
            self._scatter(n.dst, y, by, bx, t)
        return [self._interior(o) for o in self.outputs]

    def changed_blocks(self, frame, prev, tau=0.0):
        """Per-block change mask from the mean absolute frame difference."""
        d = (frame - prev).abs().mean(0, keepdim=True)[None]
        return F.avg_pool2d(d, self.block)[0, 0] > tau
