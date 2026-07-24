"""Memory-efficient Dynamic Dilated Pyramid Module.

The CUDA path is implemented in ``_triton_kernels.py`` and imported lazily so
that this module remains usable on CPU-only PyTorch installations.  The native
path streams spatial tiles through the kernel generator instead of materializing
the complete ``N x 3 x C x K^2 x H x W`` tensor.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


_DEFAULT_DILATIONS = (1, 3, 5)


@lru_cache(maxsize=1)
def _load_triton_ops() -> tuple[Callable, Callable] | None:
    """Load Triton only when a CUDA tensor actually reaches the module."""
    try:
        from ._triton_kernels import fused_dynamic_cat, fused_generator_dynamic_cat
    except (ImportError, ModuleNotFoundError):
        try:
            from _triton_kernels import fused_dynamic_cat, fused_generator_dynamic_cat
        except (ImportError, ModuleNotFoundError):
            return None
    return fused_dynamic_cat, fused_generator_dynamic_cat


def _check_inputs(x: torch.Tensor, y: torch.Tensor, channels: int) -> None:
    if x.ndim != 4 or y.ndim != 4:
        raise ValueError(f"x and y must be NCHW tensors, got {x.shape=} and {y.shape=}")
    if x.shape != y.shape:
        raise ValueError(f"x and y must have identical shapes, got {x.shape=} and {y.shape=}")
    if x.shape[1] != channels:
        raise ValueError(f"expected {channels} channels, got {x.shape[1]}")
    if x.device != y.device:
        raise ValueError(f"x and y must be on the same device, got {x.device} and {y.device}")


def _native_dynamic_cat_from_kernels(
    x: torch.Tensor,
    kernels: torch.Tensor,
    kernel_size: int,
    dilations: tuple[int, int, int],
) -> torch.Tensor:
    """Unfold-free differentiable fallback used for tests and unusual devices."""
    n, channels, height, width = x.shape
    radius = kernel_size // 2
    halo = radius * max(dilations)
    padded = F.pad(x, (halo, halo, halo, halo))
    outputs = [x]

    for branch, dilation in enumerate(dilations):
        acc = torch.zeros_like(x)
        for ki in range(kernel_size):
            h0 = halo + (ki - radius) * dilation
            for kj in range(kernel_size):
                w0 = halo + (kj - radius) * dilation
                tap = ki * kernel_size + kj
                shifted = padded[:, :, h0 : h0 + height, w0 : w0 + width]
                acc = acc + shifted * kernels[:, branch, :, tap]
        outputs.append(acc)
    return torch.cat(outputs, dim=1)


class _NativeDynamicCat(torch.autograd.Function):
    """CPU-oriented backward with three batched unfold/fold operations.

    The forward remains unfold-free.  Backward reconstructs local patches once
    per dilation, which is substantially cheaper than asking autograd to replay
    every Python-level shifted multiply/add from the native forward.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        kernels: torch.Tensor,
        kernel_size: int,
        dilations: tuple[int, int, int],
    ) -> torch.Tensor:
        ctx.save_for_backward(x, kernels)
        ctx.kernel_size = kernel_size
        ctx.dilations = dilations
        return _native_dynamic_cat_from_kernels(x, kernels, kernel_size, dilations)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, kernels = ctx.saved_tensors
        kernel_size = ctx.kernel_size
        n, channels, height, width = x.shape
        kernel_area = kernel_size * kernel_size
        radius = kernel_size // 2
        grad_x = grad_out[:, :channels].contiguous()
        grad_kernel_branches = []

        for branch, dilation in enumerate(ctx.dilations):
            branch_grad = grad_out[
                :, (branch + 1) * channels : (branch + 2) * channels
            ].contiguous()
            patches = F.unfold(
                x,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=radius * dilation,
            ).view(n, channels, kernel_area, height, width)
            grad_kernel_branches.append(patches * branch_grad.unsqueeze(2))

            weighted_grad = (
                kernels[:, branch] * branch_grad.unsqueeze(2)
            ).reshape(n, channels * kernel_area, height * width)
            grad_x = grad_x + F.fold(
                weighted_grad,
                output_size=(height, width),
                kernel_size=kernel_size,
                dilation=dilation,
                padding=radius * dilation,
            )

        return (
            grad_x,
            torch.stack(grad_kernel_branches, dim=1),
            None,
            None,
        )


def _native_dynamic_cat(
    x: torch.Tensor,
    kernels: torch.Tensor,
    kernel_size: int,
    dilations: tuple[int, int, int],
) -> torch.Tensor:
    if torch.is_grad_enabled() and (x.requires_grad or kernels.requires_grad):
        return _NativeDynamicCat.apply(x, kernels, kernel_size, dilations)
    return _native_dynamic_cat_from_kernels(x, kernels, kernel_size, dilations)


def _cpu_training_dynamic_cat(
    x: torch.Tensor,
    kernels: torch.Tensor,
    kernel_size: int,
    unfolds: nn.ModuleList,
) -> torch.Tensor:
    """Throughput-first CPU training path.

    oneDNN's unfold backward retains the forward columns and is faster on CPU
    than recomputing them in a Python custom backward.  The memory-efficient
    streaming path is still selected automatically once the kernel budget is
    exceeded.
    """
    n, channels, height, width = x.shape
    outputs = [x]
    for branch_kernel, unfold in zip(kernels.unbind(dim=1), unfolds):
        patches = unfold(x).view(
            n, channels, kernel_size**2, height, width
        )
        outputs.append((patches * branch_kernel).sum(2))
    return torch.cat(outputs, dim=1)


def _streaming_generator_dynamic_cat(
    x: torch.Tensor,
    y: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    kernel_size: int,
    dilations: tuple[int, int, int],
    tile_size: int,
    recompute: bool,
) -> torch.Tensor:
    """Flash-style native path: generate and consume one spatial tile at a time.

    Peak temporary storage is ``O(N * tile_size * 3 * C * K^2)`` instead of
    ``O(N * H * W * 3 * C * K^2)``.  During training, checkpointing recomputes
    generated tile weights in backward rather than retaining all of them.
    """
    n, channels, height, width = x.shape
    spatial = height * width
    kernel_area = kernel_size * kernel_size
    radius = kernel_size // 2
    halo = radius * max(dilations)
    padded = F.pad(x, (halo, halo, halo, halo))
    y_tokens = y.flatten(2).transpose(1, 2)
    linear_weight = weight[:, :, 0, 0]
    tile_size = spatial if tile_size <= 0 else min(tile_size, spatial)
    branch_tiles: list[list[torch.Tensor]] = [[] for _ in dilations]

    for start in range(0, spatial, tile_size):
        end = min(start + tile_size, spatial)
        positions = torch.arange(start, end, device=x.device)
        h_index = torch.div(positions, width, rounding_mode="floor")
        w_index = positions.remainder(width)

        def compute_tile(
            y_tile: torch.Tensor,
            x_padded: torch.Tensor,
            generator_weight: torch.Tensor,
            generator_bias: torch.Tensor,
            h_idx: torch.Tensor = h_index,
            w_idx: torch.Tensor = w_index,
            tile_length: int = end - start,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            generated = F.linear(y_tile, generator_weight, generator_bias)
            generated = generated.view(
                n, tile_length, len(dilations), channels, kernel_area
            )
            tile_outputs = []
            for branch, dilation in enumerate(dilations):
                acc = torch.zeros(
                    (n, tile_length, channels), dtype=x.dtype, device=x.device
                )
                for ki in range(kernel_size):
                    hi = h_idx + halo + (ki - radius) * dilation
                    for kj in range(kernel_size):
                        wi = w_idx + halo + (kj - radius) * dilation
                        tap = ki * kernel_size + kj
                        shifted = x_padded[:, :, hi, wi].transpose(1, 2)
                        acc = acc + shifted * generated[:, :, branch, :, tap]
                tile_outputs.append(acc)
            return tile_outputs[0], tile_outputs[1], tile_outputs[2]

        args = (y_tokens[:, start:end], padded, linear_weight, bias)
        if recompute and torch.is_grad_enabled():
            tile_outputs = checkpoint(compute_tile, *args, use_reentrant=False)
        else:
            tile_outputs = compute_tile(*args)
        for branch, tile in enumerate(tile_outputs):
            branch_tiles[branch].append(tile)

    outputs = [x]
    for tiles in branch_tiles:
        branch = torch.cat(tiles, dim=1).transpose(1, 2).reshape(
            n, channels, height, width
        )
        outputs.append(branch)
    return torch.cat(outputs, dim=1)


class DDPM(nn.Module):
    """Dynamic Dilated Pyramid Module with streaming and Triton fast paths.

    Args:
        dim: Input/output channel count.
        kernel_size: Odd dynamic depthwise kernel size.
        spatial_tile_size: Number of pixels whose dynamic kernels may be
            materialized by the native path at once.  ``0`` means all pixels.
        max_generated_bytes: Use the faster full-generator native path while
            its dynamic-kernel tensor fits under this byte budget.  Set to
            ``0`` to always use spatial streaming.
        recompute: Recompute native generated tiles during backward to reduce
            saved-activation memory.
        fused_generator_inference: Fuse the 1x1 kernel generator and dynamic
            filtering into one Triton launch when gradients are disabled.
        fused_generator_training: Select recomputation-based fused generator
            backward for K<=3. ``None`` (default) uses a measured crossover:
            fuse above 16384 spatial tokens and also at exactly 16384 for
            FP32. ``True`` always fuses; ``False`` always materializes.
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        *,
        spatial_tile_size: int = 4096,
        max_generated_bytes: int = 256 * 1024**2,
        recompute: bool = True,
        fused_generator_inference: bool = True,
        fused_generator_training: bool | None = None,
    ) -> None:
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
        if dim < 1:
            raise ValueError(f"dim must be positive, got {dim}")

        self.dim = dim
        self.kernel_size = kernel_size
        self.spatial_tile_size = spatial_tile_size
        self.max_generated_bytes = max_generated_bytes
        self.recompute = recompute
        self.fused_generator_inference = fused_generator_inference
        self.fused_generator_training = fused_generator_training
        self.dilations = _DEFAULT_DILATIONS
        self.generator = nn.Conv2d(dim, 3 * dim * kernel_size**2, 1)
        self.fuse = nn.Conv2d(4 * dim, dim, 3, 1, 1)
        self._cpu_unfolds = nn.ModuleList(
            nn.Unfold(
                kernel_size,
                dilation,
                padding=kernel_size // 2 * dilation,
                stride=1,
            )
            for dilation in self.dilations
        )

    def _cuda_cat(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor | None:
        ops = _load_triton_ops()
        if ops is None:
            return None
        fused_dynamic_cat, fused_generator_dynamic_cat = ops

        spatial_tokens = x.shape[0] * x.shape[2] * x.shape[3]
        fuse_training_requested = self.fused_generator_training
        if fuse_training_requested is None:
            fuse_training_requested = spatial_tokens > 16384 or (
                spatial_tokens == 16384 and x.dtype == torch.float32
            )
        fuse_training = (
            torch.is_grad_enabled()
            and fuse_training_requested
            and self.kernel_size <= 3
        )
        can_fuse_generator = (
            (self.fused_generator_inference or fuse_training)
            and (not torch.is_grad_enabled() or fuse_training)
            and not torch.is_autocast_enabled()
            and x.dtype == y.dtype == self.generator.weight.dtype
            and self.generator.bias is not None
            and self.generator.bias.dtype == x.dtype
        )
        if can_fuse_generator:
            return fused_generator_dynamic_cat(
                x,
                y,
                self.generator.weight,
                self.generator.bias,
                self.kernel_size,
                self.dilations,
            )

        kernels = self.generator(y).reshape(
            x.shape[0],
            len(self.dilations),
            self.dim,
            self.kernel_size**2,
            x.shape[2],
            x.shape[3],
        )
        return fused_dynamic_cat(x, kernels, self.kernel_size, self.dilations)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        _check_inputs(x, y, self.dim)

        cat_out = self._cuda_cat(x, y) if x.is_cuda else None
        if cat_out is None:
            generated_bytes = (
                x.shape[0]
                * len(self.dilations)
                * self.dim
                * self.kernel_size**2
                * x.shape[2]
                * x.shape[3]
                * x.element_size()
            )
            if 0 < generated_bytes <= self.max_generated_bytes:
                kernels = self.generator(y).reshape(
                    x.shape[0],
                    len(self.dilations),
                    self.dim,
                    self.kernel_size**2,
                    x.shape[2],
                    x.shape[3],
                )
                if x.device.type == "cpu" and self.training and torch.is_grad_enabled():
                    cat_out = _cpu_training_dynamic_cat(
                        x, kernels, self.kernel_size, self._cpu_unfolds
                    )
                else:
                    cat_out = _native_dynamic_cat(
                        x, kernels, self.kernel_size, self.dilations
                    )
            else:
                cat_out = _streaming_generator_dynamic_cat(
                    x,
                    y,
                    self.generator.weight,
                    self.generator.bias,
                    self.kernel_size,
                    self.dilations,
                    self.spatial_tile_size,
                    self.recompute and self.training,
                )
        return self.fuse(cat_out)


__all__ = ["DDPM", "_native_dynamic_cat_from_kernels"]
