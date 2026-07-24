"""Small CUDA validation suite used before performance tuning."""

from __future__ import annotations

import argparse
import json

import torch

from triton_impl import DDPM as TritonDDPM
from unfold_impl import DDPM as UnfoldDDPM


def max_error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    difference = (actual.float() - expected.float()).abs()
    denominator = expected.float().abs().clamp_min(1e-6)
    return {
        "abs": difference.max().item(),
        "rel": (difference / denominator).max().item(),
    }


def make_models(
    channels: int,
    kernel_size: int,
    dtype: torch.dtype = torch.float32,
):
    torch.manual_seed(123)
    reference = UnfoldDDPM(channels, kernel_size).cuda().to(dtype)
    candidate = TritonDDPM(
        channels,
        kernel_size,
        fused_generator_training=True,
    ).cuda().to(dtype)
    candidate.load_state_dict(reference.state_dict())
    return reference, candidate


def validate_training(kernel_size: int, dtype: torch.dtype) -> dict:
    # C=24 exercises all channel masks rather than only power-of-two tiles.
    channels, height, width = 24, 15, 17
    reference, candidate = make_models(channels, kernel_size, dtype)
    reference.train()
    candidate.train()
    torch.manual_seed(456)
    x_ref = torch.randn(
        2, channels, height, width, device="cuda", dtype=dtype, requires_grad=True
    )
    y_ref = torch.randn_like(x_ref, requires_grad=True)
    x_new = x_ref.detach().clone().requires_grad_()
    y_new = y_ref.detach().clone().requires_grad_()

    expected = reference(x_ref, y_ref)
    actual = candidate(x_new, y_new)
    expected_loss = expected.square().mean()
    actual_loss = actual.square().mean()

    expected_grads = torch.autograd.grad(
        expected_loss, (x_ref, y_ref, *reference.parameters())
    )
    actual_grads = torch.autograd.grad(
        actual_loss, (x_new, y_new, *candidate.parameters())
    )
    output_error = max_error(actual, expected)
    gradient_errors = [
        max_error(actual_grad, expected_grad)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads)
    ]
    tolerance = 2e-2 if dtype in (torch.float16, torch.bfloat16) else 2e-3
    passed = torch.allclose(actual, expected, atol=tolerance, rtol=tolerance) and all(
        torch.allclose(a, b, atol=tolerance * 5, rtol=tolerance * 5)
        for a, b in zip(actual_grads, expected_grads)
    )
    return {
        "mode": "training",
        "kernel_size": kernel_size,
        "dtype": str(dtype),
        "passed": passed,
        "output_error": output_error,
        "gradient_errors": gradient_errors,
    }


def validate_fused_inference(kernel_size: int) -> dict:
    channels, height, width = 16, 15, 17
    reference, candidate = make_models(channels, kernel_size)
    reference.eval()
    candidate.eval()
    torch.manual_seed(789)
    x = torch.randn(2, channels, height, width, device="cuda")
    y = torch.randn_like(x)
    with torch.inference_mode():
        expected = reference(x, y)
        actual = candidate(x, y)
    error = max_error(actual, expected)
    passed = torch.allclose(actual, expected, atol=3e-3, rtol=3e-3)
    return {
        "mode": "fused_inference",
        "kernel_size": kernel_size,
        "dtype": "torch.float32",
        "passed": passed,
        "output_error": error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel-size", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--skip-bf16", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
            }
        )
    )
    results = []
    for kernel_size in args.kernel_size:
        results.append(validate_training(kernel_size, torch.float32))
        if not args.skip_bf16:
            results.append(validate_training(kernel_size, torch.bfloat16))
        results.append(validate_fused_inference(kernel_size))
        print(json.dumps(results[-3 if not args.skip_bf16 else -2 :]))

    if not all(result["passed"] for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
