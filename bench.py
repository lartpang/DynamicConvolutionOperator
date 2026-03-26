import argparse
import itertools
import time
from contextlib import nullcontext
from dataclasses import dataclass
import tabulate
import torch
from torch import nn

# 引入两个已实现的模块
from triton_impl import DDPM as _DDPM_Triton
from unfold_impl import DDPM as _DDPM_Unfold

_REGISTRY: dict[str, type[nn.Module]] = {
    "unfold": _DDPM_Unfold,
    "triton": _DDPM_Triton,
}


def list_impls() -> list[str]:
    return sorted(_REGISTRY.keys())


def get_impl(name: str) -> type[nn.Module]:
    return _REGISTRY[name]


@dataclass
class BenchmarkResult:
    impl_name: str
    dim: int
    resolution: int
    batch_size: int
    amp: str
    mean_time_ms: float = 0.0
    peak_memory_mb: float = 0.0
    is_aligned: bool = False
    status_msg: str = ""
    output_tensor: torch.Tensor = None
    grad_x_tensor: torch.Tensor = None


def create_inputs(batch: int, dim: int, resolution: int, device: str, seed: int):
    torch.manual_seed(seed)
    # 使用 randn 提供高方差输入特征并开启支持梯度回传
    x = torch.randn(batch, dim, resolution, resolution, device=device, requires_grad=True)
    y = torch.randn(batch, dim, resolution, resolution, device=device, requires_grad=True)
    return x, y


def run_benchmark(
    model, x, y, impl_name, dim, res, batch, amp_str, warmup=10, repeats=50, test_backward=False
) -> BenchmarkResult:
    use_cuda = x.is_cuda

    amp_dtype = None
    if amp_str == "fp16":
        amp_dtype = torch.float16
    elif amp_str == "bf16":
        amp_dtype = torch.bfloat16

    if amp_dtype is not None:
        amp_ctx = torch.autocast("cuda", dtype=amp_dtype)
        scaler = torch.cuda.amp.GradScaler(init_scale=1024.0) if amp_dtype == torch.float16 and test_backward else None
    else:
        amp_ctx = nullcontext()
        scaler = None

    result = BenchmarkResult(impl_name=impl_name, dim=dim, resolution=res, batch_size=batch, amp=amp_str)

    # =============== 预热引擎 ===============
    with torch.set_grad_enabled(test_backward):
        for _ in range(warmup):
            if test_backward:
                x.grad, y.grad = None, None
                model.zero_grad(set_to_none=True)
            with amp_ctx:
                out = model(x, y)
            if test_backward:
                if scaler is not None:
                    scaler.scale(out.mean()).backward()
                else:
                    out.mean().backward()

    # =============== 统计资源 ===============
    if use_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(x.device)

    times = []
    with torch.set_grad_enabled(test_backward):
        for _ in range(repeats):
            if test_backward:
                x.grad, y.grad = None, None
                model.zero_grad(set_to_none=True)

            if use_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            with amp_ctx:
                out = model(x, y)
            if test_backward:
                if scaler is not None:
                    scaler.scale(out.mean()).backward()
                else:
                    out.mean().backward()

            if use_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            times.append((t1 - t0) * 1000.0)

    # 封裝分析结论
    times_tensor = torch.tensor(times)
    result.mean_time_ms = float(times_tensor.mean())
    result.output_tensor = out.detach().cpu()
    if test_backward:
        result.grad_x_tensor = x.grad.detach().cpu().clone()
    if use_cuda:
        result.peak_memory_mb = torch.cuda.max_memory_allocated(x.device) / (1024**2)

    return result


def evaluate_alignment(results: list[BenchmarkResult], has_bwd: bool):
    """进行矩阵数据维度的结果比对检查容差"""
    if not results:
        return
    ref = results[0]
    ref.is_aligned = True
    ref.status_msg = "参考(Ref)"

    dtype = ref.output_tensor.dtype
    tolerance = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-3

    for r in results[1:]:
        if ref.output_tensor.shape != r.output_tensor.shape:
            r.is_aligned = False
            r.status_msg = "形状不匹配"
            continue

        out_close = torch.allclose(ref.output_tensor, r.output_tensor, atol=tolerance, rtol=tolerance)
        bwd_close = True
        if has_bwd and r.grad_x_tensor is not None:
            bwd_close = torch.allclose(ref.grad_x_tensor, r.grad_x_tensor, atol=tolerance * 10, rtol=tolerance * 10)

        r.is_aligned = out_close and bwd_close
        r.status_msg = f"✓对齐" if r.is_aligned else f"✗数值悬殊"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", nargs="+", default=list_impls(), help="实现组合列表，默认为全部包含")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backward", action="store_true", help="启用反向传播测速", default=True)  # 默认直接双走
    return parser.parse_args()


def main():
    args = parse_args()

    # 笛卡尔积打包出所有的 Grid Search 配置项：(batch, dim, res, amp)
    configs = list(
        itertools.product(
            [1, 2, 4],  # batch size
            [1, 3, 5],  # kernel size
            [64],  # dim
            [128, 256],  # resolution
            ["none", "fp16", "bf16"],  # amp mode
        )
    )

    print("=========================================================================================")
    print(f" DDPM 动态卷积性能综测网格版 (总计测试流数: {len(configs)})")
    print(f" 比对实现组: {args.impl}")
    print(f" 固定种子及设备: seed={args.seed}, dev={args.device}")
    print("=========================================================================================\n")

    header = ["Batch", "Kernel", "Dim", "Res", "AMP", "Impl", "Mean Time (ms)", "Peak Mem (MB)", "Aligned", "Speedup"]
    table_lines = []
    for batch, kernel_size, dim, res, amp in configs:
        x, y = create_inputs(batch, dim, res, args.device, args.seed)

        cfg_results = []
        for name in args.impl:
            print(f"Processing B={batch}, K={kernel_size}, D={dim}, H,W={res}, Amp={amp}, Impl={name}")

            torch.manual_seed(args.seed)
            ImplClass = get_impl(name)
            model = ImplClass(dim=dim, kernel_size=kernel_size).to(args.device)
            model.eval()

            res_obj = run_benchmark(model, x, y, name, dim, res, batch, amp, args.warmup, args.repeats, args.backward)
            cfg_results.append(res_obj)
            del model
            torch.cuda.empty_cache()

        # 交叉对比参考对象与最新对象的数学差异
        evaluate_alignment(cfg_results, args.backward)

        # 汇总表行逻辑
        ref_time = cfg_results[0].mean_time_ms
        for r in cfg_results:
            table_lines.append(
                [
                    r.batch_size,
                    r.dim,
                    r.resolution,
                    r.amp.ljust(4),
                    r.impl_name.ljust(6),
                    r.mean_time_ms,
                    r.peak_memory_mb,
                    r.status_msg,
                    f"{ref_time / r.mean_time_ms:.2f}x" if r.mean_time_ms > 0 else "N/A",
                ]
            )
    print("\n" + tabulate.tabulate(table_lines, headers=header, tablefmt="pipe"))


if __name__ == "__main__":
    main()
