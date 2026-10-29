"""Turn a trained Sarbon stem into the LocalBackbone the sparse engines execute.

The standard Sarbon stem (block_type="res", act="relu") has exactly the LocalBackbone
topology: conv s2, conv s2, Res x d4, conv s2, Res x d8, every conv 3x3 / pad 1.
Folding each BatchNorm into its conv (inference statistics) gives a conv-with-bias
network that the block-sparse engines run exactly.
"""
import torch

from ..models.block_local import BlockLocalStem, Res
from .block_sparse import LocalBackbone


@torch.no_grad()
def fold(conv, bn):
    std = (bn.running_var + bn.eps).sqrt()
    t = bn.weight / std
    w = conv.weight * t.view(-1, 1, 1, 1)
    b = bn.bias - bn.running_mean * t
    if conv.bias is not None:
        b = b + conv.bias * t
    return w, b


@torch.no_grad()
def stem_to_local(stem: BlockLocalStem) -> LocalBackbone:
    s4, s8 = list(stem.s4), list(stem.s8)
    assert all(isinstance(b, Res) for b in s4 + s8), "only block_type='res' maps to LocalBackbone"
    assert isinstance(s4[0].act, torch.nn.ReLU), "only ReLU maps to LocalBackbone"
    widths = (stem.stem[0][0].out_channels, stem.stem[1][0].out_channels, stem.down[0].out_channels)
    net = LocalBackbone(widths, (len(s4), len(s8)))

    def put(dst, seq):
        w, b = fold(seq[0], seq[1])
        dst.conv.weight.copy_(w)
        dst.conv.bias.copy_(b)

    put(net.stem1, stem.stem[0])
    put(net.stem2, stem.stem[1])
    put(net.down, stem.down)
    for src, dst in zip(s4 + s8, list(net.stage4) + list(net.stage8)):
        put(dst.c1, src.body[0])
        put(dst.c2, src.body[1])
    return net.eval()
