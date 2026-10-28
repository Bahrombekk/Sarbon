"""Fused gather -> conv -> bias -> (+residual) -> ReLU -> scatter, as one Triton kernel.

Feature caches are plain NHWC tensors [n_cams, H, W, C] (the memory layout of a
channels_last NCHW tensor, so cuDNN can produce them directly). Each program
computes BLOCK_M output pixels of the active tiles for BLOCK_N output channels as
an implicit GEMM (K = KS*KS*C_in), reading the input halo straight from the cache
(zero padding emulated with load masks; no gathered copy) and writing the result
straight into the output cache (no scatter copy).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _tile_conv_kernel(
    x_ptr, w_ptr, b_ptr, r_ptr, y_ptr, cam_ptr, by_ptr, bx_ptr,
    cam_stride_x, h_x, w_x, cin, cam_stride_y, w_y, cout,
    T: tl.constexpr, S: tl.constexpr, KS: tl.constexpr,
    OFF: tl.constexpr, HAS_RES: tl.constexpr, RELU: tl.constexpr, IEEE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_n = tl.program_id(1)
    M: tl.constexpr = BLOCK_M
    q = pid * BLOCK_M + tl.arange(0, BLOCK_M)  # flat pixel index over all active tiles
    t = q // (T * T)  # which active tile each row belongs to
    p = q % (T * T)
    cam = tl.load(cam_ptr + t).to(tl.int64)
    gy = tl.load(by_ptr + t) * T + p // T  # output pixel (unpadded coords)
    gx = tl.load(bx_ptr + t) * T + p % T

    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    nm = n < cout
    acc = tl.zeros((M, BLOCK_N), dtype=tl.float32)
    for ky in tl.static_range(KS):
        for kx in tl.static_range(KS):
            iy = gy * S + ky + OFF
            ix = gx * S + kx + OFF
            inside = (iy >= 0) & (iy < h_x) & (ix >= 0) & (ix < w_x)  # zero padding
            pix = cam * cam_stride_x + (iy * w_x + ix).to(tl.int64) * cin
            for k0 in range(0, cin, BLOCK_K):
                kk = k0 + tl.arange(0, BLOCK_K)
                km = kk < cin
                a = tl.load(x_ptr + pix[:, None] + kk[None, :],
                            mask=inside[:, None] & km[None, :], other=0.0)
                w = tl.load(w_ptr + ((ky * KS + kx) * cin + kk)[:, None] * cout + n[None, :],
                            mask=km[:, None] & nm[None, :], other=0.0)
                if IEEE:
                    acc = tl.dot(a, w, acc=acc, input_precision="ieee")
                else:
                    acc = tl.dot(a, w, acc=acc)

    acc += tl.load(b_ptr + n, mask=nm, other=0.0)[None, :]
    out = cam * cam_stride_y + (gy * w_y + gx).to(tl.int64) * cout
    optr = out[:, None] + n[None, :]
    if HAS_RES:
        acc += tl.load(r_ptr + optr, mask=nm[None, :], other=0.0).to(tl.float32)
    if RELU:
        acc = tl.maximum(acc, 0.0)
    tl.store(y_ptr + optr, acc.to(y_ptr.dtype.element_ty), mask=nm[None, :])


BLOCK_M = 64  # output pixels per program, independent of tile size


def tiles_per_program(t):
    """Smallest tile count whose pixels fill whole programs (budgets must be multiples)."""
    return max(1, BLOCK_M // (t * t))


def tile_conv(x, w, b, y, cam, by, bx, t, stride, ks, pad, act, res=None):
    """x, y, res: NHWC caches. w: [ks, ks, cin, cout]. cam/by/bx: int32 tile lists."""
    cin, cout = x.shape[-1], y.shape[-1]
    n_tiles = cam.numel()
    assert n_tiles % tiles_per_program(t) == 0, "tile budget must fill whole programs"
    bn = min(64, max(16, triton.next_power_of_2(cout)))
    bk = min(32, max(16, triton.next_power_of_2(cin)))
    grid = (n_tiles * t * t // BLOCK_M, triton.cdiv(cout, bn))
    _tile_conv_kernel[grid](
        x, w, b, y if res is None else res, y, cam, by, bx,
        x.stride(0), x.shape[1], x.shape[2], cin, y.stride(0), y.shape[2], cout,
        T=t, S=stride, KS=ks, OFF=-pad, HAS_RES=res is not None, BLOCK_M=BLOCK_M,
        RELU=act, IEEE=x.dtype == torch.float32, BLOCK_N=bn, BLOCK_K=bk, num_warps=4,
        num_stages=2,  # the K loop is short (<= 4 chunks); deeper pipelines only cost smem
    )
