"""Focused CUDA benchmarks for the DDPM optimization paths."""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from _triton_kernels import fused_dynamic_cat
from triton_impl import DDPM as TritonDDPM
from unfold_impl import DDPM as UnfoldDDPM


def elapsed_ms(fn, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


def peak_memory_mb(fn) -> float:
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024**2


def error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    difference = (actual.float() - expected.float()).abs()
    scale = expected.float().abs().amax().clamp_min(torch.finfo(torch.float32).tiny)
    return {
        "max_abs": difference.amax().item(),
        "mean_abs": difference.mean().item(),
        "rmse": difference.square().mean().sqrt().item(),
        "max_abs_over_output_max": (difference.amax() / scale).item(),
    }


def model_tolerance(dtype: torch.dtype, kernel_size: int) -> float:
    if dtype == torch.bfloat16:
        return 5e-2
    if dtype == torch.float16:
        return 3e-2
    return 3e-3 if kernel_size >= 5 else 2e-3


def reference_dynamic_cat(x, kernels, kernel_size, dilations=(1, 3, 5)):
    n, channels, height, width = x.shape
    branches = [x]
    for branch, dilation in enumerate(dilations):
        patches = F.unfold(
            x,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=kernel_size // 2 * dilation,
        ).view(n, channels, kernel_size**2, height, width)
        branches.append((patches * kernels[:, branch]).sum(2))
    return torch.cat(branches, dim=1)


def dynamic_case(
    batch: int,
    channels: int,
    resolution: int,
    kernel_size: int,
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
) -> dict:
    torch.manual_seed(10)
    x = torch.randn(
        batch, channels, resolution, resolution, device="cuda", dtype=dtype
    )
    kernels = torch.randn(
        batch,
        3,
        channels,
        kernel_size**2,
        resolution,
        resolution,
        device="cuda",
        dtype=dtype,
    )
    reference = lambda: reference_dynamic_cat(x, kernels, kernel_size)
    candidate = lambda: fused_dynamic_cat(x, kernels, kernel_size, (1, 3, 5))
    with torch.inference_mode():
        expected = reference()
        actual = candidate()
        tolerance = 2e-2 if dtype != torch.float32 else 2e-3
        passed = torch.allclose(actual, expected, atol=tolerance, rtol=tolerance)
        reference_ms = elapsed_ms(reference, warmup, repeats)
        candidate_ms = elapsed_ms(candidate, warmup, repeats)
        reference_memory = peak_memory_mb(reference)
        candidate_memory = peak_memory_mb(candidate)

    elements = batch * channels * resolution * resolution
    lower_bound_bytes = (
        (3 * kernel_size**2 + 5) * elements * torch.empty((), dtype=dtype).element_size()
    )
    return {
        "kind": "dynamic_forward",
        "shape": [batch, channels, resolution, resolution],
        "kernel_size": kernel_size,
        "dtype": str(dtype),
        "passed": passed,
        "reference_ms": reference_ms,
        "candidate_ms": candidate_ms,
        "speedup": reference_ms / candidate_ms,
        "candidate_lower_bound_gbps": lower_bound_bytes / candidate_ms / 1e6,
        "reference_peak_mb": reference_memory,
        "candidate_peak_mb": candidate_memory,
    }


def dynamic_training_case(
    batch: int,
    channels: int,
    resolution: int,
    kernel_size: int,
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
) -> dict:
    torch.manual_seed(11)
    x_ref = torch.randn(
        batch,
        channels,
        resolution,
        resolution,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    kernels_ref = torch.randn(
        batch,
        3,
        channels,
        kernel_size**2,
        resolution,
        resolution,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    x_new = x_ref.detach().clone().requires_grad_()
    kernels_new = kernels_ref.detach().clone().requires_grad_()

    def reference():
        x_ref.grad = None
        kernels_ref.grad = None
        reference_dynamic_cat(x_ref, kernels_ref, kernel_size).square().mean().backward()

    def candidate():
        x_new.grad = None
        kernels_new.grad = None
        fused_dynamic_cat(x_new, kernels_new, kernel_size, (1, 3, 5)).square().mean().backward()

    reference()
    candidate()
    tolerance = 2e-2 if dtype != torch.float32 else 2e-3
    passed = torch.allclose(
        x_new.grad, x_ref.grad, atol=tolerance, rtol=tolerance
    ) and torch.allclose(
        kernels_new.grad, kernels_ref.grad, atol=tolerance, rtol=tolerance
    )
    reference_ms = elapsed_ms(reference, warmup, repeats)
    candidate_ms = elapsed_ms(candidate, warmup, repeats)
    return {
        "kind": "dynamic_training",
        "shape": [batch, channels, resolution, resolution],
        "kernel_size": kernel_size,
        "dtype": str(dtype),
        "passed": passed,
        "reference_ms": reference_ms,
        "candidate_ms": candidate_ms,
        "speedup": reference_ms / candidate_ms,
    }


def model_case(
    batch: int,
    channels: int,
    resolution: int,
    kernel_size: int,
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
) -> dict:
    torch.manual_seed(20)
    reference = UnfoldDDPM(channels, kernel_size).cuda().to(dtype).eval()
    materialized = (
        TritonDDPM(channels, kernel_size, fused_generator_inference=False)
        .cuda()
        .to(dtype)
        .eval()
    )
    fused = TritonDDPM(channels, kernel_size).cuda().to(dtype).eval()
    materialized.load_state_dict(reference.state_dict())
    fused.load_state_dict(reference.state_dict())
    x = torch.randn(
        batch, channels, resolution, resolution, device="cuda", dtype=dtype
    )
    y = torch.randn_like(x)
    reference_fn = lambda: reference(x, y)
    materialized_fn = lambda: materialized(x, y)
    fused_fn = lambda: fused(x, y)
    with torch.inference_mode():
        expected = reference_fn()
        materialized_out = materialized_fn()
        fused_out = fused_fn()
        tolerance = model_tolerance(dtype, kernel_size)
        materialized_passed = torch.allclose(
            materialized_out, expected, atol=tolerance, rtol=tolerance
        )
        fused_passed = torch.allclose(
            fused_out, expected, atol=tolerance, rtol=tolerance
        )
        materialized_error = error_metrics(materialized_out, expected)
        fused_error = error_metrics(fused_out, expected)
        reference_ms = elapsed_ms(reference_fn, warmup, repeats)
        materialized_ms = elapsed_ms(materialized_fn, warmup, repeats)
        fused_ms = elapsed_ms(fused_fn, warmup, repeats)
        reference_memory = peak_memory_mb(reference_fn)
        materialized_memory = peak_memory_mb(materialized_fn)
        fused_memory = peak_memory_mb(fused_fn)
    return {
        "kind": "model_inference",
        "shape": [batch, channels, resolution, resolution],
        "kernel_size": kernel_size,
        "dtype": str(dtype),
        "materialized_passed": materialized_passed,
        "fused_passed": fused_passed,
        "tolerance": tolerance,
        "materialized_error": materialized_error,
        "fused_error": fused_error,
        "reference_ms": reference_ms,
        "materialized_ms": materialized_ms,
        "fused_ms": fused_ms,
        "materialized_speedup": reference_ms / materialized_ms,
        "fused_speedup": reference_ms / fused_ms,
        "reference_peak_mb": reference_memory,
        "materialized_peak_mb": materialized_memory,
        "fused_peak_mb": fused_memory,
    }


def model_training_case(
    batch: int,
    channels: int,
    resolution: int,
    kernel_size: int,
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
) -> dict:
    torch.manual_seed(21)
    reference = UnfoldDDPM(channels, kernel_size).cuda().to(dtype).train()
    materialized = TritonDDPM(
        channels, kernel_size, fused_generator_training=False
    ).cuda().to(dtype).train()
    candidate = TritonDDPM(
        channels, kernel_size, fused_generator_training=True
    ).cuda().to(dtype).train()
    materialized.load_state_dict(reference.state_dict())
    candidate.load_state_dict(reference.state_dict())
    x_ref = torch.randn(
        batch,
        channels,
        resolution,
        resolution,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    y_ref = torch.randn_like(x_ref, requires_grad=True)
    x_new = x_ref.detach().clone().requires_grad_()
    y_new = y_ref.detach().clone().requires_grad_()
    x_materialized = x_ref.detach().clone().requires_grad_()
    y_materialized = y_ref.detach().clone().requires_grad_()

    def reference_step():
        reference.zero_grad(set_to_none=True)
        x_ref.grad = None
        y_ref.grad = None
        reference(x_ref, y_ref).square().mean().backward()

    def candidate_step():
        candidate.zero_grad(set_to_none=True)
        x_new.grad = None
        y_new.grad = None
        candidate(x_new, y_new).square().mean().backward()

    def materialized_step():
        materialized.zero_grad(set_to_none=True)
        x_materialized.grad = None
        y_materialized.grad = None
        materialized(x_materialized, y_materialized).square().mean().backward()

    reference_step()
    materialized_step()
    candidate_step()
    tolerance = 3e-2 if dtype != torch.float32 else 3e-3
    fused_passed = torch.allclose(
        x_new.grad, x_ref.grad, atol=tolerance, rtol=tolerance
    ) and torch.allclose(
        y_new.grad, y_ref.grad, atol=tolerance, rtol=tolerance
    ) and all(
        torch.allclose(
            candidate_parameter.grad,
            reference_parameter.grad,
            atol=tolerance,
            rtol=tolerance,
        )
        for candidate_parameter, reference_parameter in zip(
            candidate.parameters(), reference.parameters()
        )
    )
    materialized_passed = torch.allclose(
        x_materialized.grad, x_ref.grad, atol=tolerance, rtol=tolerance
    ) and torch.allclose(
        y_materialized.grad, y_ref.grad, atol=tolerance, rtol=tolerance
    ) and all(
        torch.allclose(
            materialized_parameter.grad,
            reference_parameter.grad,
            atol=tolerance,
            rtol=tolerance,
        )
        for materialized_parameter, reference_parameter in zip(
            materialized.parameters(), reference.parameters()
        )
    )
    reference_ms = elapsed_ms(reference_step, warmup, repeats)
    materialized_ms = elapsed_ms(materialized_step, warmup, repeats)
    candidate_ms = elapsed_ms(candidate_step, warmup, repeats)
    return {
        "kind": "model_training",
        "shape": [batch, channels, resolution, resolution],
        "kernel_size": kernel_size,
        "dtype": str(dtype),
        "passed": fused_passed and materialized_passed,
        "materialized_passed": materialized_passed,
        "fused_passed": fused_passed,
        "reference_ms": reference_ms,
        "materialized_ms": materialized_ms,
        "candidate_ms": candidate_ms,
        "materialized_speedup": reference_ms / materialized_ms,
        "speedup": reference_ms / candidate_ms,
        "fused_vs_materialized": materialized_ms / candidate_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--only-k5", action="store_true")
    parser.add_argument("--only-r256", action="store_true")
    parser.add_argument("--training", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    if args.only_k5:
        cases = [(1, 64, 128, 5)]
    elif args.only_r256:
        cases = [(1, 64, 256, 3)]
    elif args.quick:
        cases = [(1, 64, 128, 3)]
    else:
        cases = [
            (1, 64, 128, 3),
            (1, 64, 256, 3),
            (1, 64, 128, 5),
            (4, 64, 128, 3),
        ]
    print(json.dumps({"device": torch.cuda.get_device_name(0)}))
    for case in cases:
        for dtype in (torch.float32, torch.bfloat16):
            result = dynamic_case(
                *case, dtype=dtype, warmup=args.warmup, repeats=args.repeats
            )
            print(json.dumps(result))
            result = model_case(
                *case, dtype=dtype, warmup=args.warmup, repeats=args.repeats
            )
            print(json.dumps(result))
            if args.training:
                result = dynamic_training_case(
                    *case, dtype=dtype, warmup=args.warmup, repeats=args.repeats
                )
                print(json.dumps(result))
                result = model_training_case(
                    *case, dtype=dtype, warmup=args.warmup, repeats=args.repeats
                )
                print(json.dumps(result))


if __name__ == "__main__":
    main()
