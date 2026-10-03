"""
FlashSlice's Triton kernels, called from JAX through jax-triton.

The four single-tile kernels (slice and deslice, forward and backward) and
their tile tables are copied verbatim from
https://github.com/Shizheng-Wen/flashslice (v0.3.0, commit bcb26a7,
`flashslice/kernels/slice_ops.py`), Copyright 2026 Shizheng Wen, Apache-2.0;
they are original work of that package. What is new here is the host side:
the PyTorch wrappers (`torch.empty`, `.stride()`, the `torch.library`
registration) become `jax_triton.triton_call` launches, row-major strides
computed from the shapes, `jnp.sum` reductions of the per-program partials,
and a `jax.custom_vjp`.

Serves what the reference's single-tile family serves: head width `D` and
slice count `G` powers of two in [16, 128], value width equal to `D`, 16- or
32-bit activations. `zoo._flashslice.fused_slice` / `fused_deslice` route
here with `backend="triton"`; every other shape stays on their pure-JAX path.

Requires a CUDA GPU, `jax>=0.11`, `jax-triton>=0.4` and `triton>=3.7`; the
reference was validated on Triton 3.0-3.4.
"""

import functools
import os

import jax
import jax.numpy as jnp
import jax_triton as jt
import triton
import triton.language as tl

_MAX_PROGRAMS = 512  # target total persistent programs across the (B*H) grid axis

# fmt: off
# ---------------------------------------------------------------------------
# Tile tables, verbatim from the reference
# ---------------------------------------------------------------------------
# (BLOCK_N, num_warps, num_stages) per G -> (kernel, input-is-16bit), from the
# GH200 sweeps of bench_kernels.py. G is a tiling axis, not just a shape: the
# whole slice axis lives in one tile, so the tile budget is BLOCK_N x G and the
# winner moves with G. Winners also differ by dtype: e.g. (256,4,2) is best for
# bf16 slice_fwd at G=32 but pathological for fp32 (job 3084301's ieee
# regression), whose winner is stages=1.
# Using the G=32 tiles at another G is not a small loss: the BLOCK_N x G
# accumulators spill to local memory, costing a median 2.4x at G=64 and 5.3x at
# G=128 (worst single kernel 40x / 80x), and leaving ~9% on the table at G=16.
#   G=32:  job 3084253 (+ 3088100 for the bf16-dot tier) — frozen; every number
#          reported in the study was measured with these tiles, and a fresh
#          sweep (3095916) reproduces 10 of its 12 entries exactly, the other
#          two within 2.3%.
#   G=16/64/128: jobs 3095933 / 3095917 / 3095918, one sweep each, winner =
#          lowest mean slowdown vs the per-N best over N = 262k and 1M.
_CFG = {
    16: {
        ("slice_fwd", False): (256, 4, 2),
        ("slice_fwd", True): (128, 4, 2),
        ("deslice_fwd", False): (256, 4, 1),
        ("deslice_fwd", True): (256, 4, 1),
        ("slice_bwd", False): (256, 4, 2),
        ("slice_bwd", True): (256, 4, 1),
        ("deslice_bwd", False): (128, 4, 1),
        ("deslice_bwd", True): (64, 4, 2),
    },
    32: {
        ("slice_fwd", False): (256, 4, 1),
        ("slice_fwd", True): (256, 4, 2),
        ("deslice_fwd", False): (128, 4, 1),
        ("deslice_fwd", True): (128, 4, 1),
        ("slice_bwd", False): (128, 4, 1),
        ("slice_bwd", True): (128, 4, 1),
        ("deslice_bwd", False): (128, 4, 2),
        ("deslice_bwd", True): (64, 4, 2),
    },
    64: {
        ("slice_fwd", False): (128, 4, 1),
        ("slice_fwd", True): (128, 4, 2),
        ("deslice_fwd", False): (128, 8, 1),
        ("deslice_fwd", True): (128, 4, 3),
        ("slice_bwd", False): (64, 4, 1),
        ("slice_bwd", True): (64, 4, 1),
        ("deslice_bwd", False): (64, 4, 2),
        ("deslice_bwd", True): (64, 4, 1),
    },
    128: {
        ("slice_fwd", False): (64, 4, 3),
        ("slice_fwd", True): (64, 4, 2),
        ("deslice_fwd", False): (64, 4, 1),
        ("deslice_fwd", True): (64, 4, 2),
        ("slice_bwd", False): (64, 8, 2),
        ("slice_bwd", True): (64, 8, 1),
        ("deslice_bwd", False): (64, 8, 2),
        ("deslice_bwd", True): (64, 8, 1),
    },
}


# bf16-dot (level 3) winners differ again — faster dots shift the optimum
# and stages>1 is safe there (same jobs as above).
_CFG_BF16 = {
    16: {
        "slice_fwd": (128, 4, 3),
        "deslice_fwd": (256, 4, 3),
        "slice_bwd": (64, 4, 3),
        "deslice_bwd": (128, 4, 2),
    },
    32: {
        "slice_fwd": (256, 4, 3),
        "deslice_fwd": (64, 4, 1),
        "slice_bwd": (64, 4, 3),
        "deslice_bwd": (64, 4, 3),
    },
    64: {
        "slice_fwd": (128, 4, 3),
        "deslice_fwd": (128, 8, 2),
        "slice_bwd": (64, 4, 1),
        "deslice_bwd": (64, 4, 2),
    },
    128: {
        "slice_fwd": (64, 4, 2),
        "deslice_fwd": (64, 4, 3),
        "slice_bwd": (64, 4, 1),
        "deslice_bwd": (64, 4, 1),
    },
}


# The same tables for GPUs with less shared memory per block than Hopper,
# swept on one RTX 4090 (sm_89, 99 KB opt-in shared memory, torch 2.8,
# Triton 3.4) with bench_kernels.py over N = 262k and 1M, fp32 and bf16
# inputs, ieee and bf16 dots; BLOCK_N in {64, 128, 256} ({32, 64, 128}
# at G=128), warps in {4, 8}, stages in {1, 2, 3}. On that GPU several
# Hopper entries do not compile (shared memory) and others spill: the
# Hopper tile is up to 15x (slice_fwd) and 22x (deslice_bwd) slower than
# these at G=64 in fp32.
_CFG_ADA = {
    16: {
        ('slice_fwd', False): (128, 4, 2),
        ('slice_fwd', True): (128, 4, 1),
        ('deslice_fwd', False): (128, 8, 3),
        ('deslice_fwd', True): (128, 4, 1),
        ('slice_bwd', False): (128, 8, 1),
        ('slice_bwd', True): (128, 4, 1),
        ('deslice_bwd', False): (128, 4, 3),
        ('deslice_bwd', True): (128, 4, 1),
    },
    32: {
        ('slice_fwd', False): (128, 4, 1),
        ('slice_fwd', True): (128, 4, 1),
        ('deslice_fwd', False): (64, 8, 3),
        ('deslice_fwd', True): (256, 4, 1),
        ('slice_bwd', False): (64, 4, 2),
        ('slice_bwd', True): (64, 4, 2),
        ('deslice_bwd', False): (64, 8, 1),
        ('deslice_bwd', True): (128, 4, 1),
    },
    64: {
        ('slice_fwd', False): (64, 4, 2),
        ('slice_fwd', True): (64, 4, 3),
        ('deslice_fwd', False): (64, 4, 3),
        ('deslice_fwd', True): (128, 4, 2),
        ('slice_bwd', False): (128, 8, 1),
        ('slice_bwd', True): (128, 8, 2),
        ('deslice_bwd', False): (64, 4, 1),
        ('deslice_bwd', True): (64, 4, 1),
    },
    128: {
        ('slice_fwd', False): (128, 8, 1),
        ('slice_fwd', True): (128, 8, 1),
        ('deslice_fwd', False): (64, 4, 3),
        ('deslice_fwd', True): (64, 4, 1),
        ('slice_bwd', False): (64, 8, 1),
        ('slice_bwd', True): (64, 8, 3),
        ('deslice_bwd', False): (64, 8, 1),
        ('deslice_bwd', True): (64, 8, 1),
    },
}

_CFG_BF16_ADA = {
    16: {
        'slice_fwd': (64, 4, 2),
        'deslice_fwd': (64, 8, 1),
        'slice_bwd': (128, 4, 1),
        'deslice_bwd': (128, 4, 1),
    },
    32: {
        'slice_fwd': (256, 8, 1),
        'deslice_fwd': (64, 8, 1),
        'slice_bwd': (64, 4, 1),
        'deslice_bwd': (64, 4, 3),
    },
    64: {
        'slice_fwd': (64, 4, 2),
        'deslice_fwd': (64, 4, 2),
        'slice_bwd': (64, 4, 1),
        'deslice_bwd': (64, 4, 3),
    },
    128: {
        'slice_fwd': (64, 4, 1),
        'deslice_fwd': (64, 4, 1),
        'slice_bwd': (64, 4, 2),
        'deslice_bwd': (64, 4, 2),
    },
}


# Ada tiles for the tf32-class dot levels (tf32 = 1, tf32x3 = 4; bf16v takes
# the tf32 entry), fp32 inputs, one stage (the tf32 pipeliner guard): same
# sweep protocol, BLOCK_N in {32, 64, 128}.
_CFG_TF32_ADA = {
    16: {
        ('slice_fwd', 1): (64, 4, 1),
        ('slice_fwd', 4): (64, 4, 1),
        ('deslice_fwd', 1): (64, 8, 1),
        ('deslice_fwd', 4): (64, 8, 1),
        ('slice_bwd', 1): (64, 4, 1),
        ('slice_bwd', 4): (32, 4, 1),
        ('deslice_bwd', 1): (64, 4, 1),
        ('deslice_bwd', 4): (64, 4, 1),
    },
    32: {
        ('slice_fwd', 1): (128, 4, 1),
        ('slice_fwd', 4): (64, 4, 1),
        ('deslice_fwd', 1): (128, 8, 1),
        ('deslice_fwd', 4): (128, 8, 1),
        ('slice_bwd', 1): (64, 4, 1),
        ('slice_bwd', 4): (64, 4, 1),
        ('deslice_bwd', 1): (64, 4, 1),
        ('deslice_bwd', 4): (32, 4, 1),
    },
    64: {
        ('slice_fwd', 1): (64, 4, 1),
        ('slice_fwd', 4): (64, 4, 1),
        ('deslice_fwd', 1): (64, 4, 1),
        ('deslice_fwd', 4): (64, 4, 1),
        ('slice_bwd', 1): (64, 4, 1),
        ('slice_bwd', 4): (64, 4, 1),
        ('deslice_bwd', 1): (32, 4, 1),
        ('deslice_bwd', 4): (32, 4, 1),
    },
    128: {
        ('slice_fwd', 1): (32, 4, 1),
        ('slice_fwd', 4): (64, 4, 1),
        ('deslice_fwd', 1): (128, 8, 1),
        ('deslice_fwd', 4): (64, 4, 1),
        ('deslice_bwd', 1): (32, 4, 1),
    },
}


# ---------------------------------------------------------------------------
# Kernels, verbatim from the reference
# ---------------------------------------------------------------------------

@triton.jit
def _dot(a, b, DOT: tl.constexpr):
    """Value/gradient dot at the requested precision; fp32 accumulation."""
    if DOT == 4:
        return tl.dot(a, b, input_precision="tf32x3")
    elif DOT >= 2:
        return tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16))
    else:
        return tl.dot(a, b, input_precision="tf32" if DOT == 1 else "ieee")


@triton.jit
def _dot_w(a, b, DOT: tl.constexpr):
    """Logits dot for w: ieee unless full-bf16 (3) or tf32x3 (4) mode."""
    if DOT == 3:
        return tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16))
    elif DOT == 4:
        return tl.dot(a, b, input_precision="tf32x3")
    else:
        return tl.dot(a, b, input_precision="ieee")

@triton.jit
def _slice_fwd_kernel(
    XM, FX, W, BS, TAU, PART_Z, PART_S,
    N, P, H,
    swb, swh, sbb, sbh,
    sxb, sxn, sxh, sxd,
    sfb, sfn, sfh, sfd,
    D: tl.constexpr, G: tl.constexpr, BN: tl.constexpr, DOT: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    W = W + b.to(tl.int64) * swb + h * swh
    BS = BS + b.to(tl.int64) * sbb + h * sbh
    offs_d = tl.arange(0, D)
    offs_g = tl.arange(0, G)

    w_mat = tl.load(W + offs_g[:, None] * D + offs_d[None, :]).to(tl.float32)
    bias = tl.load(BS + offs_g).to(tl.float32)
    tau = tl.load(TAU + h).to(tl.float32)

    acc_z = tl.zeros((G, D), dtype=tl.float32)
    acc_s = tl.zeros((G,), dtype=tl.float32)
    for start in range(pid * BN, N, P * BN):
        offs_n = start + tl.arange(0, BN)
        mask = offs_n < N
        offs_n64 = offs_n.to(tl.int64)  # n*stride overflows int32 past N~8M
        xm = tl.load(XM + b * sxb + h * sxh + offs_n64[:, None] * sxn
                     + offs_d[None, :] * sxd,
                     mask=mask[:, None], other=0.0).to(tl.float32)
        logits = _dot_w(xm, tl.trans(w_mat), DOT)
        logits = (logits + bias[None, :]) / tau
        m = tl.max(logits, axis=1)
        e = tl.exp(logits - m[:, None])
        w = e / tl.sum(e, axis=1)[:, None]
        w = tl.where(mask[:, None], w, 0.0)
        fx = tl.load(FX + b * sfb + h * sfh + offs_n64[:, None] * sfn
                     + offs_d[None, :] * sfd,
                     mask=mask[:, None], other=0.0).to(tl.float32)
        acc_z += _dot(tl.trans(w), fx, DOT)
        acc_s += tl.sum(w, axis=0)

    idx = bh * P + pid
    tl.store(PART_Z + idx * G * D + offs_g[:, None] * D + offs_d[None, :], acc_z)
    tl.store(PART_S + idx * G + offs_g, acc_s)


@triton.jit
def _deslice_fwd_kernel(
    XM, W, BS, TAU, TOK, OUT,
    N, H,
    swb, swh, sbb, sbh,
    sxb, sxn, sxh, sxd,
    sob, son, soh, sod,
    D: tl.constexpr, G: tl.constexpr, BN: tl.constexpr, DOT: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    W = W + b.to(tl.int64) * swb + h * swh
    BS = BS + b.to(tl.int64) * sbb + h * sbh
    offs_d = tl.arange(0, D)
    offs_g = tl.arange(0, G)

    w_mat = tl.load(W + offs_g[:, None] * D + offs_d[None, :]).to(tl.float32)
    bias = tl.load(BS + offs_g).to(tl.float32)
    tau = tl.load(TAU + h).to(tl.float32)
    tok = tl.load(TOK + bh * G * D + offs_g[:, None] * D
                  + offs_d[None, :]).to(tl.float32)

    offs_n = pid * BN + tl.arange(0, BN)
    mask = offs_n < N
    offs_n64 = offs_n.to(tl.int64)  # n*stride overflows int32 past N~8M
    xm = tl.load(XM + b * sxb + h * sxh + offs_n64[:, None] * sxn
                 + offs_d[None, :] * sxd,
                 mask=mask[:, None], other=0.0).to(tl.float32)
    logits = _dot_w(xm, tl.trans(w_mat), DOT)
    logits = (logits + bias[None, :]) / tau
    m = tl.max(logits, axis=1)
    e = tl.exp(logits - m[:, None])
    w = e / tl.sum(e, axis=1)[:, None]
    out = _dot(w, tok, DOT)
    tl.store(OUT + b * sob + h * soh + offs_n64[:, None] * son
             + offs_d[None, :] * sod,
             out.to(OUT.dtype.element_ty), mask=mask[:, None])


@triton.jit
def _slice_bwd_kernel(
    XM, FX, W, BS, TAU, DZN, DS,
    DXM, DFX, PDW, PDB, PDT,
    N, P, H,
    swb, swh, sbb, sbh,
    sxb, sxn, sxh, sxd,
    sfb, sfn, sfh, sfd,
    D: tl.constexpr, G: tl.constexpr, BN: tl.constexpr, DOT: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    W = W + b.to(tl.int64) * swb + h * swh
    BS = BS + b.to(tl.int64) * sbb + h * sbh
    offs_d = tl.arange(0, D)
    offs_g = tl.arange(0, G)

    w_mat = tl.load(W + offs_g[:, None] * D + offs_d[None, :]).to(tl.float32)
    bias = tl.load(BS + offs_g).to(tl.float32)
    tau = tl.load(TAU + h).to(tl.float32)
    dzn = tl.load(DZN + bh * G * D + offs_g[:, None] * D
                  + offs_d[None, :]).to(tl.float32)
    ds = tl.load(DS + bh * G + offs_g).to(tl.float32)

    acc_dw = tl.zeros((G, D), dtype=tl.float32)
    acc_db = tl.zeros((G,), dtype=tl.float32)
    acc_dt = 0.0
    for start in range(pid * BN, N, P * BN):
        offs_n = start + tl.arange(0, BN)
        mask = offs_n < N
        offs_n64 = offs_n.to(tl.int64)  # n*stride overflows int32 past N~8M
        xm = tl.load(XM + b * sxb + h * sxh + offs_n64[:, None] * sxn
                     + offs_d[None, :] * sxd,
                     mask=mask[:, None], other=0.0).to(tl.float32)
        logits = _dot_w(xm, tl.trans(w_mat), DOT)
        logits = (logits + bias[None, :]) / tau
        m = tl.max(logits, axis=1)
        e = tl.exp(logits - m[:, None])
        w = e / tl.sum(e, axis=1)[:, None]
        w = tl.where(mask[:, None], w, 0.0)

        fx = tl.load(FX + b * sfb + h * sfh + offs_n64[:, None] * sfn
                     + offs_d[None, :] * sfd,
                     mask=mask[:, None], other=0.0).to(tl.float32)
        # d fx_mid = w @ dz_num
        dfx = _dot(w, dzn, DOT)
        tl.store(DFX + b * sfb + h * sfh + offs_n64[:, None] * sfn
                 + offs_d[None, :] * sfd,
                 dfx.to(DFX.dtype.element_ty), mask=mask[:, None])
        # d w, softmax Jacobian, d logits (pre-division)
        dw = _dot(fx, tl.trans(dzn), DOT) + ds[None, :]
        gsum = tl.sum(dw * w, axis=1)
        dl = w * (dw - gsum[:, None])
        dlr = dl / tau
        dxm = _dot(dlr, w_mat, DOT)
        tl.store(DXM + b * sxb + h * sxh + offs_n64[:, None] * sxn
                 + offs_d[None, :] * sxd,
                 dxm.to(DXM.dtype.element_ty), mask=mask[:, None])
        acc_dw += _dot(tl.trans(dlr), xm, DOT)
        acc_db += tl.sum(dlr, axis=0)
        acc_dt += tl.sum(tl.sum(dl * (-logits / tau), axis=1), axis=0)

    idx = bh * P + pid
    tl.store(PDW + idx * G * D + offs_g[:, None] * D + offs_d[None, :], acc_dw)
    tl.store(PDB + idx * G + offs_g, acc_db)
    tl.store(PDT + idx, acc_dt)


@triton.jit
def _deslice_bwd_kernel(
    XM, W, BS, TAU, TOK, DOUT,
    DXM, PDTOK, PDW, PDB, PDT,
    N, P, H,
    swb, swh, sbb, sbh,
    sxb, sxn, sxh, sxd,
    sob, son, soh, sod,
    D: tl.constexpr, G: tl.constexpr, BN: tl.constexpr, DOT: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    W = W + b.to(tl.int64) * swb + h * swh
    BS = BS + b.to(tl.int64) * sbb + h * sbh
    offs_d = tl.arange(0, D)
    offs_g = tl.arange(0, G)

    w_mat = tl.load(W + offs_g[:, None] * D + offs_d[None, :]).to(tl.float32)
    bias = tl.load(BS + offs_g).to(tl.float32)
    tau = tl.load(TAU + h).to(tl.float32)
    tok = tl.load(TOK + bh * G * D + offs_g[:, None] * D
                  + offs_d[None, :]).to(tl.float32)

    acc_dtok = tl.zeros((G, D), dtype=tl.float32)
    acc_dw = tl.zeros((G, D), dtype=tl.float32)
    acc_db = tl.zeros((G,), dtype=tl.float32)
    acc_dt = 0.0
    for start in range(pid * BN, N, P * BN):
        offs_n = start + tl.arange(0, BN)
        mask = offs_n < N
        offs_n64 = offs_n.to(tl.int64)  # n*stride overflows int32 past N~8M
        xm = tl.load(XM + b * sxb + h * sxh + offs_n64[:, None] * sxn
                     + offs_d[None, :] * sxd,
                     mask=mask[:, None], other=0.0).to(tl.float32)
        logits = _dot_w(xm, tl.trans(w_mat), DOT)
        logits = (logits + bias[None, :]) / tau
        m = tl.max(logits, axis=1)
        e = tl.exp(logits - m[:, None])
        w = e / tl.sum(e, axis=1)[:, None]
        w = tl.where(mask[:, None], w, 0.0)

        dout = tl.load(DOUT + b * sob + h * soh + offs_n64[:, None] * son
                       + offs_d[None, :] * sod,
                       mask=mask[:, None], other=0.0).to(tl.float32)
        acc_dtok += _dot(tl.trans(w), dout, DOT)
        dw = _dot(dout, tl.trans(tok), DOT)
        gsum = tl.sum(dw * w, axis=1)
        dl = w * (dw - gsum[:, None])
        dlr = dl / tau
        dxm = _dot(dlr, w_mat, DOT)
        tl.store(DXM + b * sxb + h * sxh + offs_n64[:, None] * sxn
                 + offs_d[None, :] * sxd,
                 dxm.to(DXM.dtype.element_ty), mask=mask[:, None])
        acc_dw += _dot(tl.trans(dlr), xm, DOT)
        acc_db += tl.sum(dlr, axis=0)
        acc_dt += tl.sum(tl.sum(dl * (-logits / tau), axis=1), axis=0)

    idx = bh * P + pid
    tl.store(PDTOK + idx * G * D + offs_g[:, None] * D + offs_d[None, :], acc_dtok)
    tl.store(PDW + idx * G * D + offs_g[:, None] * D + offs_d[None, :], acc_dw)
    tl.store(PDB + idx * G + offs_g, acc_db)
    tl.store(PDT + idx, acc_dt)
# fmt: on


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------
# Accumulation is fp32 in every mode; the logits dot stays ieee below "bf16",
# because the softmax Jacobian amplifies noise in w (see the reference).
_DOT_LEVELS = {"ieee": 0, "tf32": 1, "bf16v": 2, "bf16": 3, "tf32x3": 4}
_ACTIVATION_DTYPES = (jnp.float32, jnp.bfloat16, jnp.float16)

# Device kinds with Hopper-class shared memory per block (>= 200 KB). The
# reference reads that from CUDA; JAX exposes the device kind instead.
_HOPPER_KINDS = ("H100", "H200", "H800", "H20", "GH200", "B100", "B200", "GB200")


def tile_table():
    """'hopper' or 'ada'; FLASHSLICE_TILE_TABLE forces one, as in the reference."""
    forced = os.environ.get("FLASHSLICE_TILE_TABLE", "").lower()
    if forced in ("hopper", "ada"):
        return forced
    kind = jax.devices()[0].device_kind.upper()
    return "hopper" if any(k in kind for k in _HOPPER_KINDS) else "ada"


def _cfg(name, is16, dot=0, g=32, d=32):
    """The reference's `_cfg`, keyed on (is16, dot) instead of a torch tensor."""
    table_class = tile_table()
    if table_class == "ada" and dot in (1, 2, 4):
        table, key = _CFG_TF32_ADA, (name, 4 if dot == 4 else 1)
    elif table_class == "ada":
        table = _CFG_BF16_ADA if dot == 3 else _CFG_ADA
        key = name if dot == 3 else (name, is16)
    else:
        table = _CFG_BF16 if dot == 3 else _CFG
        key = name if dot == 3 else (name, is16)
    bn, warps, stages = table.get(g, {}).get(key) or table[32][key]
    if d != 32:
        bn = max(16, min(256, bn * 32 // d))
    if dot and g <= 16 and warps > 4:
        warps = 4
    if dot and warps > 4 and name in ("slice_bwd", "deslice_bwd"):
        warps = 4
    if dot in (1, 4):  # the reference's `_stages`: single-stage tf32 dots
        stages = 1
    return bn, warps, stages


def _dot_level(mode, dtype):
    if mode not in _DOT_LEVELS:
        raise ValueError(f"dot must be one of {sorted(_DOT_LEVELS)}, got {mode!r}")
    level = _DOT_LEVELS[mode]
    if level in (2, 3) and dtype == jnp.float32:
        return 1  # bf16 dots only for 16-bit inputs; fp32 falls back to tf32
    return level


def single_tile_dims(d, g, dv=None):
    """True when the kernels serve (D, G): both powers of two in [16, 128], and
    the value width `dv` equal to `d`."""
    if dv is not None and dv != d:
        return False
    return all(16 <= v <= 128 and (v & (v - 1)) == 0 for v in (d, g))


def check_dims(d, g, dv=None):
    if not single_tile_dims(d, g, dv):
        raise ValueError(
            f"backend='triton' serves head width and slice count that are powers "
            f"of two in [16, 128] with equal value width; got head width {d}, "
            f"{g} slices, value width {d if dv is None else dv}. Use backend='jax'."
        )


def _n_programs(n, bh, block_n):
    return max(1, min(triton.cdiv(n, block_n), max(8, _MAX_PROGRAMS // max(bh, 1))))


def _strides(shape):
    """Element strides of a row-major (contiguous) 4-D array."""
    _, s1, s2, s3 = shape
    return dict(b=s1 * s2 * s3, n=s2 * s3, h=s3, d=1)


def _lead2(shape, trailing):
    return ((1, 1) + tuple(shape[:-trailing]))[-2:]


def _wb_layout(weight, bias, B, H):
    """Strides of the weight (.., G, D) and bias (.., G) along x_mid's batch and
    head axes: zero wherever the tensor is shared along that axis."""
    G, D = weight.shape[-2:]
    Bw, Hw = _lead2(weight.shape, 2)
    Bb, Hb = _lead2(bias.shape, 1)
    return dict(
        swb=Hw * G * D if Bw > 1 else 0,
        swh=G * D if Hw > 1 else 0,
        sbb=Hb * G if Bb > 1 else 0,
        sbh=G if Hb > 1 else 0,
    )


def _reduce_parts(pdw, pdb, B, H, P, weight, bias):
    """Per-program partial dW and db summed into the weight's and bias's own
    shapes: over the programs, and over any batch or head axis they share."""
    G, D = weight.shape[-2:]
    dw = pdw.reshape(B, H, P, G, D).sum(2)
    db = pdb.reshape(B, H, P, G).sum(2)
    Bw, Hw = _lead2(weight.shape, 2)
    Bb, Hb = _lead2(bias.shape, 1)
    if Bw == 1:
        dw = dw.sum(0, keepdims=True)
    if Hw == 1:
        dw = dw.sum(1, keepdims=True)
    if Bb == 1:
        db = db.sum(0, keepdims=True)
    if Hb == 1:
        db = db.sum(1, keepdims=True)
    return dw.reshape(weight.shape).astype(weight.dtype), db.reshape(bias.shape).astype(
        bias.dtype
    )


def _f32(*shape):
    return jax.ShapeDtypeStruct(shape, jnp.float32)


def _launch(kernel, name, x_mid, weight, bias, dot, out_type, *, per_n):
    """Common launch: tile config, strides of x_mid and the projection, and the
    shape constants. `per_n` launches one program per point tile (deslice
    forward); the others are persistent, `P` programs per (batch, head)."""
    B, N, H, D = x_mid.shape
    G = weight.shape[-2]
    bn, warps, stages = _cfg(name, x_mid.dtype != jnp.float32, dot, G, D)
    sx = _strides(x_mid.shape)
    P = None if per_n else _n_programs(N, B * H, bn)
    out_type = out_type(P)
    grid = (triton.cdiv(N, bn), B * H) if per_n else (P, B * H)

    def call(**operands):
        if P is not None:
            operands["P"] = P
        return jt.triton_call(
            kernel=kernel,
            out_type=out_type,
            grid=grid,
            num_warps=warps,
            num_stages=stages,
            W=weight,
            BS=bias,
            N=N,
            H=H,
            **_wb_layout(weight, bias, B, H),
            sxb=sx["b"], sxn=sx["n"], sxh=sx["h"], sxd=sx["d"],
            D=D, G=G, BN=bn, DOT=dot,
            **operands,
        )  # fmt: skip

    return call, P


def _slice_impl(x_mid, fx_mid, weight, bias, tau, dot):
    B, N, H, D = x_mid.shape
    G = weight.shape[-2]
    sf = _strides(fx_mid.shape)
    call, P = _launch(
        _slice_fwd_kernel, "slice_fwd", x_mid, weight, bias, dot,
        lambda P: (_f32(B * H * P, G, D), _f32(B * H * P, G)), per_n=False,
    )  # fmt: skip
    part_z, part_s = call(
        XM=x_mid, FX=fx_mid, TAU=tau,
        sfb=sf["b"], sfn=sf["n"], sfh=sf["h"], sfd=sf["d"],
    )  # fmt: skip
    return part_z.reshape(B, H, P, G, D).sum(2), part_s.reshape(B, H, P, G).sum(2)


def _slice_bwd_impl(x_mid, fx_mid, weight, bias, tau, dz_num, ds, dot):
    B, N, H, D = x_mid.shape
    G = weight.shape[-2]
    sf = _strides(fx_mid.shape)
    call, P = _launch(
        _slice_bwd_kernel, "slice_bwd", x_mid, weight, bias, dot,
        lambda P: (
            jax.ShapeDtypeStruct(x_mid.shape, x_mid.dtype),
            jax.ShapeDtypeStruct(fx_mid.shape, fx_mid.dtype),
            _f32(B * H * P, G, D), _f32(B * H * P, G), _f32(B * H * P),
        ), per_n=False,
    )  # fmt: skip
    dxm, dfx, pdw, pdb, pdt = call(
        XM=x_mid, FX=fx_mid, TAU=tau, DZN=dz_num, DS=ds,
        sfb=sf["b"], sfn=sf["n"], sfh=sf["h"], sfd=sf["d"],
    )  # fmt: skip
    dw, db = _reduce_parts(pdw, pdb, B, H, P, weight, bias)
    return dxm, dfx, dw, db, pdt.reshape(B, H, P).sum((0, 2)).astype(tau.dtype)


def _deslice_impl(x_mid, weight, bias, tau, tokens, dot):
    so = _strides(x_mid.shape)
    call, _ = _launch(
        _deslice_fwd_kernel, "deslice_fwd", x_mid, weight, bias, dot,
        lambda P: jax.ShapeDtypeStruct(x_mid.shape, x_mid.dtype), per_n=True,
    )  # fmt: skip
    return call(
        XM=x_mid, TAU=tau, TOK=tokens,
        sob=so["b"], son=so["n"], soh=so["h"], sod=so["d"],
    )  # fmt: skip


def _deslice_bwd_impl(x_mid, weight, bias, tau, tokens, d_out, dot):
    B, N, H, D = x_mid.shape
    G = weight.shape[-2]
    so = _strides(d_out.shape)
    call, P = _launch(
        _deslice_bwd_kernel, "deslice_bwd", x_mid, weight, bias, dot,
        lambda P: (
            jax.ShapeDtypeStruct(x_mid.shape, x_mid.dtype),
            _f32(B * H * P, G, D), _f32(B * H * P, G, D),
            _f32(B * H * P, G), _f32(B * H * P),
        ), per_n=False,
    )  # fmt: skip
    dxm, pdtok, pdw, pdb, pdt = call(
        XM=x_mid, TAU=tau, TOK=tokens, DOUT=d_out,
        sob=so["b"], son=so["n"], soh=so["h"], sod=so["d"],
    )  # fmt: skip
    dw, db = _reduce_parts(pdw, pdb, B, H, P, weight, bias)
    d_tokens = pdtok.reshape(B, H, P, G, D).sum(2).astype(tokens.dtype)
    return dxm, d_tokens, dw, db, pdt.reshape(B, H, P).sum((0, 2)).astype(tau.dtype)


@functools.partial(jax.custom_vjp, nondiff_argnums=(5,))
def _slice(x_mid, fx_mid, weight, bias, tau, dot):
    return _slice_impl(x_mid, fx_mid, weight, bias, tau, dot)


def _slice_fwd(x_mid, fx_mid, weight, bias, tau, dot):
    out = _slice_impl(x_mid, fx_mid, weight, bias, tau, dot)
    return out, (x_mid, fx_mid, weight, bias, tau)


def _slice_bwd(dot, residuals, cotangents):
    dz_num, ds = (c.astype(jnp.float32) for c in cotangents)
    return _slice_bwd_impl(*residuals, dz_num, ds, dot)


_slice.defvjp(_slice_fwd, _slice_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(5,))
def _deslice(x_mid, weight, bias, tau, tokens, dot):
    return _deslice_impl(x_mid, weight, bias, tau, tokens, dot)


def _deslice_fwd(x_mid, weight, bias, tau, tokens, dot):
    out = _deslice_impl(x_mid, weight, bias, tau, tokens, dot)
    return out, (x_mid, weight, bias, tau, tokens)


def _deslice_bwd(dot, residuals, d_out):
    x_mid, weight, bias, tau, tokens = residuals
    dxm, d_tokens, dw, db, dtau = _deslice_bwd_impl(
        x_mid, weight, bias, tau, tokens, d_out, dot
    )
    return dxm, dw, db, dtau, d_tokens


_deslice.defvjp(_deslice_fwd, _deslice_bwd)


def _check_activations(*arrays):
    for a in arrays:
        if a.dtype not in _ACTIVATION_DTYPES:
            raise ValueError(
                f"backend='triton' takes float32, bfloat16 or float16 activations, "
                f"got {a.dtype}"
            )


def slice_op(x_mid, fx_mid, weight, bias, tau, *, dot="ieee"):
    """`fused_slice` on the Triton kernels; `weight` and `bias` (not None) are
    already validated against `x_mid` and keep their own shapes."""
    _check_activations(x_mid, fx_mid)
    check_dims(x_mid.shape[3], weight.shape[-2], fx_mid.shape[3])
    return _slice(x_mid, fx_mid, weight, bias, tau, _dot_level(dot, x_mid.dtype))


def deslice_op(x_mid, weight, bias, tau, tokens, *, dot="ieee"):
    """`fused_deslice` on the Triton kernels; see `slice_op`."""
    _check_activations(x_mid)
    check_dims(x_mid.shape[3], weight.shape[-2], tokens.shape[3])
    tokens = tokens.astype(jnp.float32)
    return _deslice(x_mid, weight, bias, tau, tokens, _dot_level(dot, x_mid.dtype))
