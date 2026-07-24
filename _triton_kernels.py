"""Triton kernels for DDPM.

This file is intentionally isolated from ``triton_impl.py``: importing the
public module on a CPU-only machine must not require a Triton installation.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


_POINTWISE_CONFIGS = [
    triton.Config({"BLOCK_HW": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_HW": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_HW": 256}, num_warps=8, num_stages=2),
]


@triton.autotune(configs=_POINTWISE_CONFIGS, key=["C", "H", "W", "KS"])
@triton.jit
def _dynamic_cat_fwd(
    x,
    kernels,
    out,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    KS: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    n = pid_nc // C
    c = pid_nc - n * C
    hw = tl.program_id(1) * BLOCK_HW + tl.arange(0, BLOCK_HW)
    valid = hw < H * W
    h = hw // W
    w = hw - h * W
    x_base = (n * C + c) * H * W
    kernel_nc = (n * 3 * C + c) * KS * KS * H * W

    center = tl.load(x + x_base + hw, mask=valid, other=0.0)
    acc0 = tl.zeros((BLOCK_HW,), tl.float32)
    acc1 = tl.zeros((BLOCK_HW,), tl.float32)
    acc2 = tl.zeros((BLOCK_HW,), tl.float32)
    radius = KS // 2

    for ki in tl.static_range(0, KS):
        for kj in tl.static_range(0, KS):
            tap = ki * KS + kj
            dh = ki - radius
            dw = kj - radius

            nh0 = h + dh * D0
            nw0 = w + dw * D0
            mask0 = valid & (nh0 >= 0) & (nh0 < H) & (nw0 >= 0) & (nw0 < W)
            xv0 = tl.load(x + x_base + nh0 * W + nw0, mask=mask0, other=0.0)
            kv0 = tl.load(
                kernels + kernel_nc + tap * H * W + hw, mask=valid, other=0.0
            )
            acc0 += xv0 * kv0

            nh1 = h + dh * D1
            nw1 = w + dw * D1
            mask1 = valid & (nh1 >= 0) & (nh1 < H) & (nw1 >= 0) & (nw1 < W)
            xv1 = tl.load(x + x_base + nh1 * W + nw1, mask=mask1, other=0.0)
            kv1 = tl.load(
                kernels
                + kernel_nc
                + C * KS * KS * H * W
                + tap * H * W
                + hw,
                mask=valid,
                other=0.0,
            )
            acc1 += xv1 * kv1

            nh2 = h + dh * D2
            nw2 = w + dw * D2
            mask2 = valid & (nh2 >= 0) & (nh2 < H) & (nw2 >= 0) & (nw2 < W)
            xv2 = tl.load(x + x_base + nh2 * W + nw2, mask=mask2, other=0.0)
            kv2 = tl.load(
                kernels
                + kernel_nc
                + 2 * C * KS * KS * H * W
                + tap * H * W
                + hw,
                mask=valid,
                other=0.0,
            )
            acc2 += xv2 * kv2

    out_base = n * 4 * C * H * W + c * H * W + hw
    tl.store(out + out_base, center, mask=valid)
    tl.store(out + out_base + C * H * W, acc0, mask=valid)
    tl.store(out + out_base + 2 * C * H * W, acc1, mask=valid)
    tl.store(out + out_base + 3 * C * H * W, acc2, mask=valid)


@triton.autotune(configs=_POINTWISE_CONFIGS, key=["C", "H", "W", "KS"])
@triton.jit
def _dynamic_cat_bwd_x(
    grad_out,
    kernels,
    grad_x,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    KS: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    n = pid_nc // C
    c = pid_nc - n * C
    hw = tl.program_id(1) * BLOCK_HW + tl.arange(0, BLOCK_HW)
    valid = hw < H * W
    h = hw // W
    w = hw - h * W
    plane = H * W
    go_base = n * 4 * C * plane + c * plane
    kernel_nc = (n * 3 * C + c) * KS * KS * plane
    acc = tl.load(grad_out + go_base + hw, mask=valid, other=0.0).to(tl.float32)
    radius = KS // 2

    for ki in tl.static_range(0, KS):
        for kj in tl.static_range(0, KS):
            tap = ki * KS + kj
            dh = ki - radius
            dw = kj - radius

            oh0 = h - dh * D0
            ow0 = w - dw * D0
            mask0 = valid & (oh0 >= 0) & (oh0 < H) & (ow0 >= 0) & (ow0 < W)
            out_hw0 = oh0 * W + ow0
            go0 = tl.load(
                grad_out + go_base + C * plane + out_hw0, mask=mask0, other=0.0
            )
            k0 = tl.load(
                kernels + kernel_nc + tap * plane + out_hw0, mask=mask0, other=0.0
            )
            acc += go0 * k0

            oh1 = h - dh * D1
            ow1 = w - dw * D1
            mask1 = valid & (oh1 >= 0) & (oh1 < H) & (ow1 >= 0) & (ow1 < W)
            out_hw1 = oh1 * W + ow1
            go1 = tl.load(
                grad_out + go_base + 2 * C * plane + out_hw1,
                mask=mask1,
                other=0.0,
            )
            k1 = tl.load(
                kernels
                + kernel_nc
                + C * KS * KS * plane
                + tap * plane
                + out_hw1,
                mask=mask1,
                other=0.0,
            )
            acc += go1 * k1

            oh2 = h - dh * D2
            ow2 = w - dw * D2
            mask2 = valid & (oh2 >= 0) & (oh2 < H) & (ow2 >= 0) & (ow2 < W)
            out_hw2 = oh2 * W + ow2
            go2 = tl.load(
                grad_out + go_base + 3 * C * plane + out_hw2,
                mask=mask2,
                other=0.0,
            )
            k2 = tl.load(
                kernels
                + kernel_nc
                + 2 * C * KS * KS * plane
                + tap * plane
                + out_hw2,
                mask=mask2,
                other=0.0,
            )
            acc += go2 * k2

    tl.store(grad_x + (n * C + c) * plane + hw, acc, mask=valid)


@triton.autotune(configs=_POINTWISE_CONFIGS, key=["C", "H", "W", "KS"])
@triton.jit
def _dynamic_cat_bwd_k(
    x,
    grad_out,
    grad_k,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    KS: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    n = pid_nc // C
    c = pid_nc - n * C
    hw = tl.program_id(1) * BLOCK_HW + tl.arange(0, BLOCK_HW)
    valid = hw < H * W
    h = hw // W
    w = hw - h * W
    plane = H * W
    x_base = (n * C + c) * plane
    go_base = n * 4 * C * plane + c * plane
    gk_base = (n * 3 * C + c) * KS * KS * plane
    go0 = tl.load(grad_out + go_base + C * plane + hw, mask=valid, other=0.0)
    go1 = tl.load(grad_out + go_base + 2 * C * plane + hw, mask=valid, other=0.0)
    go2 = tl.load(grad_out + go_base + 3 * C * plane + hw, mask=valid, other=0.0)
    radius = KS // 2

    for ki in tl.static_range(0, KS):
        for kj in tl.static_range(0, KS):
            tap = ki * KS + kj
            dh = ki - radius
            dw = kj - radius

            nh0 = h + dh * D0
            nw0 = w + dw * D0
            mask0 = valid & (nh0 >= 0) & (nh0 < H) & (nw0 >= 0) & (nw0 < W)
            xv0 = tl.load(x + x_base + nh0 * W + nw0, mask=mask0, other=0.0)
            tl.store(grad_k + gk_base + tap * plane + hw, go0 * xv0, mask=valid)

            nh1 = h + dh * D1
            nw1 = w + dw * D1
            mask1 = valid & (nh1 >= 0) & (nh1 < H) & (nw1 >= 0) & (nw1 < W)
            xv1 = tl.load(x + x_base + nh1 * W + nw1, mask=mask1, other=0.0)
            tl.store(
                grad_k + gk_base + C * KS * KS * plane + tap * plane + hw,
                go1 * xv1,
                mask=valid,
            )

            nh2 = h + dh * D2
            nw2 = w + dw * D2
            mask2 = valid & (nh2 >= 0) & (nh2 < H) & (nw2 >= 0) & (nw2 < W)
            xv2 = tl.load(x + x_base + nh2 * W + nw2, mask=mask2, other=0.0)
            tl.store(
                grad_k + gk_base + 2 * C * KS * KS * plane + tap * plane + hw,
                go2 * xv2,
                mask=valid,
            )


_GENERATOR_CONFIGS = [
    triton.Config({"BLOCK_C": 16, "BLOCK_HW": 16}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_C": 16, "BLOCK_HW": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_C": 16, "BLOCK_HW": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_C": 32, "BLOCK_HW": 16}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_C": 32, "BLOCK_HW": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_C": 32, "BLOCK_HW": 32}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_C": 64, "BLOCK_HW": 16}, num_warps=8, num_stages=2),
]


@triton.autotune(configs=_GENERATOR_CONFIGS, key=["C", "H", "W", "KS"])
@triton.jit
def _generator_dynamic_cat_fwd(
    x,
    y,
    weight,
    bias,
    out,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    KS: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    BLOCK_IN: tl.constexpr,
    QUANTIZE_BF16: tl.constexpr,
    QUANTIZE_FP16: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    """Tensor-core tiled generator + dynamic filtering, without writing K."""
    n = tl.program_id(0)
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    hw = tl.program_id(2) * BLOCK_HW + tl.arange(0, BLOCK_HW)
    c2 = c[:, None]
    hw2 = hw[None, :]
    cin = tl.arange(0, BLOCK_IN)
    valid_c = c < C
    valid_hw = hw < H * W
    h = hw2 // W
    w = hw2 - h * W
    y_tile = tl.load(
        y + n * C * H * W + cin[:, None] * H * W + hw2,
        mask=(cin[:, None] < C) & valid_hw[None, :],
        other=0.0,
    )
    x_center = tl.load(
        x + (n * C + c2) * H * W + hw2,
        mask=valid_c[:, None] & valid_hw[None, :],
        other=0.0,
    )
    acc0 = tl.zeros((BLOCK_C, BLOCK_HW), tl.float32)
    acc1 = tl.zeros((BLOCK_C, BLOCK_HW), tl.float32)
    acc2 = tl.zeros((BLOCK_C, BLOCK_HW), tl.float32)
    radius = KS // 2

    for ki in tl.static_range(0, KS):
        for kj in tl.static_range(0, KS):
            tap = ki * KS + kj
            dh = ki - radius
            dw = kj - radius

            out_index0 = c * KS * KS + tap
            w0 = tl.load(
                weight + out_index0[:, None] * C + cin[None, :],
                mask=valid_c[:, None] & (cin[None, :] < C),
                other=0.0,
            )
            b0 = tl.load(bias + out_index0, mask=valid_c, other=0.0)
            generated0 = tl.dot(
                w0, y_tile, input_precision=DOT_PRECISION
            ) + b0[:, None]
            if QUANTIZE_BF16:
                generated0 = generated0.to(tl.bfloat16)
            elif QUANTIZE_FP16:
                generated0 = generated0.to(tl.float16)
            nh0 = h + dh * D0
            nw0 = w + dw * D0
            mask0 = (
                valid_c[:, None]
                & valid_hw[None, :]
                & (nh0 >= 0)
                & (nh0 < H)
                & (nw0 >= 0)
                & (nw0 < W)
            )
            xv0 = tl.load(
                x + (n * C + c2) * H * W + nh0 * W + nw0,
                mask=mask0,
                other=0.0,
            )
            acc0 += generated0 * xv0

            out_index1 = (C + c) * KS * KS + tap
            w1 = tl.load(
                weight + out_index1[:, None] * C + cin[None, :],
                mask=valid_c[:, None] & (cin[None, :] < C),
                other=0.0,
            )
            b1 = tl.load(bias + out_index1, mask=valid_c, other=0.0)
            generated1 = tl.dot(
                w1, y_tile, input_precision=DOT_PRECISION
            ) + b1[:, None]
            if QUANTIZE_BF16:
                generated1 = generated1.to(tl.bfloat16)
            elif QUANTIZE_FP16:
                generated1 = generated1.to(tl.float16)
            nh1 = h + dh * D1
            nw1 = w + dw * D1
            mask1 = (
                valid_c[:, None]
                & valid_hw[None, :]
                & (nh1 >= 0)
                & (nh1 < H)
                & (nw1 >= 0)
                & (nw1 < W)
            )
            xv1 = tl.load(
                x + (n * C + c2) * H * W + nh1 * W + nw1,
                mask=mask1,
                other=0.0,
            )
            acc1 += generated1 * xv1

            out_index2 = (2 * C + c) * KS * KS + tap
            w2 = tl.load(
                weight + out_index2[:, None] * C + cin[None, :],
                mask=valid_c[:, None] & (cin[None, :] < C),
                other=0.0,
            )
            b2 = tl.load(bias + out_index2, mask=valid_c, other=0.0)
            generated2 = tl.dot(
                w2, y_tile, input_precision=DOT_PRECISION
            ) + b2[:, None]
            if QUANTIZE_BF16:
                generated2 = generated2.to(tl.bfloat16)
            elif QUANTIZE_FP16:
                generated2 = generated2.to(tl.float16)
            nh2 = h + dh * D2
            nw2 = w + dw * D2
            mask2 = (
                valid_c[:, None]
                & valid_hw[None, :]
                & (nh2 >= 0)
                & (nh2 < H)
                & (nw2 >= 0)
                & (nw2 < W)
            )
            xv2 = tl.load(
                x + (n * C + c2) * H * W + nh2 * W + nw2,
                mask=mask2,
                other=0.0,
            )
            acc2 += generated2 * xv2

    out_base = n * 4 * C * H * W + c2 * H * W + hw2
    out_mask = valid_c[:, None] & valid_hw[None, :]
    tl.store(out + out_base, x_center, mask=out_mask)
    tl.store(out + out_base + C * H * W, acc0, mask=out_mask)
    tl.store(out + out_base + 2 * C * H * W, acc1, mask=out_mask)
    tl.store(out + out_base + 3 * C * H * W, acc2, mask=out_mask)


_GENERATOR_BWD_CONFIGS = [
    triton.Config({"BLOCK_C": 16, "BLOCK_HW": 16}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_C": 16, "BLOCK_HW": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_C": 32, "BLOCK_HW": 16}, num_warps=8, num_stages=2),
]


@triton.autotune(configs=_GENERATOR_BWD_CONFIGS, key=["C", "H", "W", "KS"])
@triton.jit
def _generator_dynamic_cat_bwd_x(
    y,
    weight,
    bias,
    grad_out,
    grad_x,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    KS: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    BLOCK_IN: tl.constexpr,
    QUANTIZE_BF16: tl.constexpr,
    QUANTIZE_FP16: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    """Recompute generated taps and gather their contribution to dX."""
    n = tl.program_id(0)
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    q = tl.program_id(2) * BLOCK_HW + tl.arange(0, BLOCK_HW)
    c2 = c[:, None]
    q2 = q[None, :]
    cin = tl.arange(0, BLOCK_IN)
    valid_c = c < C
    valid_q = q < H * W
    qh = q2 // W
    qw = q2 - qh * W
    plane = H * W
    grad_base = n * 4 * C * plane
    acc = tl.load(
        grad_out + grad_base + c2 * plane + q2,
        mask=valid_c[:, None] & valid_q[None, :],
        other=0.0,
    ).to(tl.float32)
    radius = KS // 2

    for ki in tl.static_range(0, KS):
        for kj in tl.static_range(0, KS):
            tap = ki * KS + kj
            dh = ki - radius
            dw = kj - radius
            for branch in tl.static_range(0, 3):
                if branch == 0:
                    dilation = D0
                elif branch == 1:
                    dilation = D1
                else:
                    dilation = D2
                ph = qh - dh * dilation
                pw = qw - dw * dilation
                valid_p = (
                    valid_q[None, :]
                    & (ph >= 0)
                    & (ph < H)
                    & (pw >= 0)
                    & (pw < W)
                )
                p = ph * W + pw
                y_tile = tl.load(
                    y + n * C * plane + cin[:, None] * plane + p,
                    mask=(cin[:, None] < C) & valid_p,
                    other=0.0,
                )
                out_index = (branch * C + c) * KS * KS + tap
                w_tile = tl.load(
                    weight + out_index[:, None] * C + cin[None, :],
                    mask=valid_c[:, None] & (cin[None, :] < C),
                    other=0.0,
                )
                b = tl.load(bias + out_index, mask=valid_c, other=0.0)
                generated = (
                    tl.dot(
                        w_tile, y_tile, input_precision=DOT_PRECISION
                    )
                    + b[:, None]
                )
                if QUANTIZE_BF16:
                    generated = generated.to(tl.bfloat16)
                elif QUANTIZE_FP16:
                    generated = generated.to(tl.float16)
                go = tl.load(
                    grad_out
                    + grad_base
                    + (branch + 1) * C * plane
                    + c2 * plane
                    + p,
                    mask=valid_c[:, None] & valid_p,
                    other=0.0,
                )
                acc += generated * go

    tl.store(
        grad_x + (n * C + c2) * plane + q2,
        acc,
        mask=valid_c[:, None] & valid_q[None, :],
    )


@triton.autotune(configs=_GENERATOR_BWD_CONFIGS, key=["C", "H", "W", "KS"])
@triton.jit
def _generator_dynamic_cat_bwd_y(
    x,
    weight,
    grad_out,
    grad_y,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    KS: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    """Compute generator input gradient while forming dK on chip."""
    n = tl.program_id(0)
    i = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    p = tl.program_id(2) * BLOCK_HW + tl.arange(0, BLOCK_HW)
    i2 = i[:, None]
    p2 = p[None, :]
    valid_i = i < C
    valid_p = p < H * W
    ph = p2 // W
    pw = p2 - ph * W
    plane = H * W
    grad_base = n * 4 * C * plane
    acc = tl.zeros((BLOCK_C, BLOCK_HW), tl.float32)
    radius = KS // 2

    for ki in tl.static_range(0, KS):
        for kj in tl.static_range(0, KS):
            tap = ki * KS + kj
            dh = ki - radius
            dw = kj - radius
            for branch in tl.static_range(0, 3):
                if branch == 0:
                    dilation = D0
                elif branch == 1:
                    dilation = D1
                else:
                    dilation = D2
                nh = ph + dh * dilation
                nw = pw + dw * dilation
                neighbor_valid = (
                    valid_p[None, :]
                    & (nh >= 0)
                    & (nh < H)
                    & (nw >= 0)
                    & (nw < W)
                )
                for c_start in tl.static_range(0, C, BLOCK_C):
                    c = c_start + tl.arange(0, BLOCK_C)
                    c2 = c[:, None]
                    valid_c = c < C
                    xv = tl.load(
                        x + (n * C + c2) * plane + nh * W + nw,
                        mask=valid_c[:, None] & neighbor_valid,
                        other=0.0,
                    )
                    go = tl.load(
                        grad_out
                        + grad_base
                        + (branch + 1) * C * plane
                        + c2 * plane
                        + p2,
                        mask=valid_c[:, None] & valid_p[None, :],
                        other=0.0,
                    )
                    dk = xv * go
                    out_index = (branch * C + c) * KS * KS + tap
                    w_tile = tl.load(
                        weight + out_index[:, None] * C + i[None, :],
                        mask=valid_c[:, None] & valid_i[None, :],
                        other=0.0,
                    )
                    acc += tl.dot(
                        tl.trans(w_tile),
                        dk,
                        input_precision=DOT_PRECISION,
                    )

    tl.store(
        grad_y + (n * C + i2) * plane + p2,
        acc,
        mask=valid_i[:, None] & valid_p[None, :],
    )


@triton.jit
def _generator_dynamic_cat_bwd_weight_partial(
    x,
    y,
    grad_out,
    partial_weight,
    partial_bias,
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    KS: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    SPLITS: tl.constexpr,
    TOKENS_PER_SPLIT: tl.constexpr,
    DIRECT_WRITE: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Form one FP32 spatial partial for dW/db without materializing dK."""
    branch_tap = tl.program_id(0)
    branch = branch_tap // (KS * KS)
    tap = branch_tap - branch * KS * KS
    ki = tap // KS
    kj = tap - ki * KS
    channel_tiles = tl.cdiv(C, BLOCK_C)
    channel_pair = tl.program_id(1)
    c_tile = channel_pair // channel_tiles
    i_tile = channel_pair - c_tile * channel_tiles
    split = tl.program_id(2)
    c = c_tile * BLOCK_C + tl.arange(0, BLOCK_C)
    i = i_tile * BLOCK_C + tl.arange(0, BLOCK_C)
    c2 = c[:, None]
    valid_c = c < C
    valid_i = i < C
    plane = H * W
    radius = KS // 2
    dh = ki - radius
    dw = kj - radius
    if branch == 0:
        dilation = D0
    elif branch == 1:
        dilation = D1
    else:
        dilation = D2
    acc = tl.zeros((BLOCK_C, BLOCK_C), tl.float32)
    bias_acc = tl.zeros((BLOCK_C,), tl.float32)
    token_base = split * TOKENS_PER_SPLIT

    for token_offset in tl.range(0, TOKENS_PER_SPLIT, BLOCK_K):
        token = token_base + token_offset + tl.arange(0, BLOCK_K)
        n = token // plane
        p = token - n * plane
        p2 = p[None, :]
        valid_p = (token < N * plane) & (
            token < token_base + TOKENS_PER_SPLIT
        )
        ph = p2 // W
        pw = p2 - ph * W
        nh = ph + dh * dilation
        nw = pw + dw * dilation
        neighbor_valid = (
            valid_p[None, :]
            & (nh >= 0)
            & (nh < H)
            & (nw >= 0)
            & (nw < W)
        )
        xv = tl.load(
            x + (n[None, :] * C + c2) * plane + nh * W + nw,
            mask=valid_c[:, None] & neighbor_valid,
            other=0.0,
        )
        go = tl.load(
            grad_out
            + n[None, :] * 4 * C * plane
            + (branch + 1) * C * plane
            + c2 * plane
            + p2,
            mask=valid_c[:, None] & valid_p[None, :],
            other=0.0,
        )
        dk = xv * go
        y_tile = tl.load(
            y + (n[None, :] * C + i[:, None]) * plane + p2,
            mask=valid_i[:, None] & valid_p[None, :],
            other=0.0,
        )
        acc += tl.dot(
            dk,
            tl.trans(y_tile),
            input_precision=DOT_PRECISION,
        )
        bias_acc += tl.sum(dk, axis=1)

    if DIRECT_WRITE:
        out_index = (branch * C + c) * KS * KS + tap
        tl.store(
            partial_weight + out_index[:, None] * C + i[None, :],
            acc,
            mask=valid_c[:, None] & valid_i[None, :],
        )
        tl.store(
            partial_bias + out_index,
            bias_acc,
            mask=valid_c & (i_tile == 0),
        )
    else:
        partial_base = (
            (split * 3 * KS * KS + branch_tap) * C * C
        )
        tl.store(
            partial_weight + partial_base + c[:, None] * C + i[None, :],
            acc,
            mask=valid_c[:, None] & valid_i[None, :],
        )
        tl.store(
            partial_bias
            + (split * 3 * KS * KS + branch_tap) * C
            + c,
            bias_acc,
            mask=valid_c & (i_tile == 0),
        )


@triton.jit
def _generator_dynamic_cat_bwd_weight_reduce(
    partial_weight,
    partial_bias,
    grad_weight,
    grad_bias,
    C: tl.constexpr,
    KS: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Reduce split spatial partials and restore Conv2d parameter layout."""
    branch_tap = tl.program_id(0)
    channel_tiles = tl.cdiv(C, BLOCK_C)
    channel_pair = tl.program_id(1)
    c_tile = channel_pair // channel_tiles
    i_tile = channel_pair - c_tile * channel_tiles
    c = c_tile * BLOCK_C + tl.arange(0, BLOCK_C)
    i = i_tile * BLOCK_C + tl.arange(0, BLOCK_C)
    valid_c = c < C
    valid_i = i < C
    acc = tl.zeros((BLOCK_C, BLOCK_C), tl.float32)
    bias_acc = tl.zeros((BLOCK_C,), tl.float32)

    for split in tl.static_range(0, SPLITS):
        partial_base = (
            (split * 3 * KS * KS + branch_tap) * C * C
        )
        acc += tl.load(
            partial_weight
            + partial_base
            + c[:, None] * C
            + i[None, :],
            mask=valid_c[:, None] & valid_i[None, :],
            other=0.0,
        )
        if i_tile == 0:
            bias_acc += tl.load(
                partial_bias
                + (split * 3 * KS * KS + branch_tap) * C
                + c,
                mask=valid_c,
                other=0.0,
            )

    branch = branch_tap // (KS * KS)
    tap = branch_tap - branch * KS * KS
    out_index = (branch * C + c) * KS * KS + tap
    tl.store(
        grad_weight + out_index[:, None] * C + i[None, :],
        acc,
        mask=valid_c[:, None] & valid_i[None, :],
    )
    tl.store(
        grad_bias + out_index,
        bias_acc,
        mask=valid_c & (i_tile == 0),
    )


class _DynamicCat(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, kernels, kernel_size, dilations):
        x = x.contiguous()
        kernels = kernels.contiguous()
        n, channels, height, width = x.shape
        out = torch.empty(
            (n, 4 * channels, height, width), dtype=kernels.dtype, device=x.device
        )
        grid = lambda meta: (
            n * channels,
            triton.cdiv(height * width, meta["BLOCK_HW"]),
        )
        _dynamic_cat_fwd[grid](
            x,
            kernels,
            out,
            C=channels,
            H=height,
            W=width,
            KS=kernel_size,
            D0=dilations[0],
            D1=dilations[1],
            D2=dilations[2],
        )
        ctx.save_for_backward(x, kernels)
        ctx.kernel_size = kernel_size
        ctx.dilations = dilations
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, kernels = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        n, channels, height, width = x.shape
        grid = lambda meta: (
            n * channels,
            triton.cdiv(height * width, meta["BLOCK_HW"]),
        )
        common = dict(
            C=channels,
            H=height,
            W=width,
            KS=ctx.kernel_size,
            D0=ctx.dilations[0],
            D1=ctx.dilations[1],
            D2=ctx.dilations[2],
        )
        grad_x = torch.empty_like(x)
        _dynamic_cat_bwd_x[grid](grad_out, kernels, grad_x, **common)
        grad_k = torch.empty_like(kernels)
        _dynamic_cat_bwd_k[grid](x, grad_out, grad_k, **common)
        return grad_x, grad_k, None, None


def fused_dynamic_cat(x, kernels, kernel_size, dilations):
    return _DynamicCat.apply(x, kernels, kernel_size, dilations)


def _launch_generator_dynamic_cat(x, y, weight, bias, kernel_size, dilations):
    x = x.contiguous()
    y = y.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    n, channels, height, width = x.shape
    out = torch.empty(
        (n, 4 * channels, height, width), dtype=x.dtype, device=x.device
    )
    block_in = max(16, triton.next_power_of_2(channels))
    grid = lambda meta: (
        n,
        triton.cdiv(channels, meta["BLOCK_C"]),
        triton.cdiv(height * width, meta["BLOCK_HW"]),
    )
    _generator_dynamic_cat_fwd[grid](
        x,
        y,
        weight,
        bias,
        out,
        C=channels,
        H=height,
        W=width,
        KS=kernel_size,
        D0=dilations[0],
        D1=dilations[1],
        D2=dilations[2],
        BLOCK_IN=block_in,
        QUANTIZE_BF16=x.dtype == torch.bfloat16,
        QUANTIZE_FP16=x.dtype == torch.float16,
        # Preserve the stricter inference error bound.  TF32 is reserved for
        # backward recomputation, where it materially improves throughput.
        DOT_PRECISION="ieee",
    )
    return out


class _GeneratorDynamicCat(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, weight, bias, kernel_size, dilations):
        x = x.contiguous()
        y = y.contiguous()
        weight = weight.contiguous()
        bias = bias.contiguous()
        out = _launch_generator_dynamic_cat(
            x, y, weight, bias, kernel_size, dilations
        )
        ctx.save_for_backward(x, y, weight, bias)
        ctx.kernel_size = kernel_size
        ctx.dilations = dilations
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, y, weight, bias = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        n, channels, height, width = x.shape
        kernel_size = ctx.kernel_size
        dilations = ctx.dilations
        block_in = max(16, triton.next_power_of_2(channels))
        common = dict(
            C=channels,
            H=height,
            W=width,
            KS=kernel_size,
            D0=dilations[0],
            D1=dilations[1],
            D2=dilations[2],
        )

        grad_x = torch.empty_like(x)
        spatial_grid = lambda meta: (
            n,
            triton.cdiv(channels, meta["BLOCK_C"]),
            triton.cdiv(height * width, meta["BLOCK_HW"]),
        )
        _generator_dynamic_cat_bwd_x[spatial_grid](
            y,
            weight,
            bias,
            grad_out,
            grad_x,
            BLOCK_IN=block_in,
            QUANTIZE_BF16=x.dtype == torch.bfloat16,
            QUANTIZE_FP16=x.dtype == torch.float16,
            DOT_PRECISION="tf32" if x.dtype == torch.float32 else "ieee",
            **common,
        )

        grad_y = torch.empty_like(y)
        _generator_dynamic_cat_bwd_y[spatial_grid](
            x,
            weight,
            grad_out,
            grad_y,
            DOT_PRECISION="tf32" if x.dtype == torch.float32 else "ieee",
            **common,
        )

        grad_weight = torch.empty_like(weight)
        grad_bias = torch.empty_like(bias)
        total_tokens = n * height * width
        if total_tokens <= 16384:
            block_c = 16
            channel_tiles = triton.cdiv(channels, block_c)
            direct_grid = (
                3 * kernel_size * kernel_size,
                channel_tiles * channel_tiles,
                1,
            )
            _generator_dynamic_cat_bwd_weight_partial[direct_grid](
                x,
                y,
                grad_out,
                grad_weight,
                grad_bias,
                N=n,
                SPLITS=1,
                TOKENS_PER_SPLIT=total_tokens,
                DIRECT_WRITE=True,
                DOT_PRECISION="ieee",
                BLOCK_C=block_c,
                BLOCK_K=16,
                **common,
            )
            return grad_x, grad_y, grad_weight, grad_bias, None, None

        splits = min(16, max(1, triton.cdiv(total_tokens, 4096)))
        block_c = 32 if channels >= 32 else 16
        block_k = 32
        tokens_per_split = triton.cdiv(total_tokens, splits)
        tokens_per_split = triton.cdiv(tokens_per_split, block_k) * block_k
        branch_taps = 3 * kernel_size * kernel_size
        channel_tiles = triton.cdiv(channels, block_c)
        partial_weight = torch.empty(
            splits * branch_taps * channels * channels,
            dtype=torch.float32,
            device=x.device,
        )
        partial_bias = torch.empty(
            splits * branch_taps * channels,
            dtype=torch.float32,
            device=x.device,
        )
        partial_grid = (
            3 * kernel_size * kernel_size,
            channel_tiles * channel_tiles,
            splits,
        )
        _generator_dynamic_cat_bwd_weight_partial[partial_grid](
            x,
            y,
            grad_out,
            partial_weight,
            partial_bias,
            N=n,
            SPLITS=splits,
            TOKENS_PER_SPLIT=tokens_per_split,
            DIRECT_WRITE=False,
            DOT_PRECISION="tf32",
            BLOCK_C=block_c,
            BLOCK_K=block_k,
            **common,
        )
        reduce_grid = (
            branch_taps,
            channel_tiles * channel_tiles,
        )
        _generator_dynamic_cat_bwd_weight_reduce[reduce_grid](
            partial_weight,
            partial_bias,
            grad_weight,
            grad_bias,
            C=channels,
            KS=kernel_size,
            SPLITS=splits,
            BLOCK_C=block_c,
            num_warps=8 if block_c == 32 else 4,
        )
        return grad_x, grad_y, grad_weight, grad_bias, None, None


def fused_generator_dynamic_cat(x, y, weight, bias, kernel_size, dilations):
    if torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (x, y, weight, bias)
    ):
        return _GeneratorDynamicCat.apply(
            x, y, weight, bias, kernel_size, dilations
        )
    return _launch_generator_dynamic_cat(
        x, y, weight, bias, kernel_size, dilations
    )


__all__ = ["fused_dynamic_cat", "fused_generator_dynamic_cat"]
