"""Isolated peak-memory measurement for one DDPM training implementation."""

from __future__ import annotations

import argparse
import json

import torch

from triton_impl import DDPM as TritonDDPM
from unfold_impl import DDPM as UnfoldDDPM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--implementation", choices=("unfold", "materialized", "fused"), required=True
    )
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), required=True)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    if args.implementation == "unfold":
        module = UnfoldDDPM(64, 3)
    else:
        module = TritonDDPM(
            64,
            3,
            fused_generator_training=args.implementation == "fused",
        )
    module = module.cuda().to(dtype).train()
    x = torch.randn(
        1,
        64,
        args.resolution,
        args.resolution,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    y = torch.randn_like(x, requires_grad=True)

    def step():
        module.zero_grad(set_to_none=True)
        x.grad = None
        y.grad = None
        module(x, y).square().mean().backward()

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    module.zero_grad(set_to_none=True)
    x.grad = None
    y.grad = None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    step()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    print(
        json.dumps(
            {
                "implementation": args.implementation,
                "dtype": args.dtype,
                "resolution": args.resolution,
                "baseline_mb": baseline / 1024**2,
                "peak_mb": peak / 1024**2,
                "temporary_peak_mb": (peak - baseline) / 1024**2,
            }
        )
    )


if __name__ == "__main__":
    main()
