import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch.cuda.amp import custom_bwd, custom_fwd


@triton.jit
def _fused_ddpm_fwd_kernel(
    # --- 1. 数据指针 (Pointers) ---
    X,  # 输入特征图 X 的起始内存地址指针。逻辑形状: (N, C, H, W)
    K,  # 融合动态卷积核 K 的起始内存地址指针。逻辑形状: (N, 3, C, KS*KS, H, W)
    OUT,  # 输出特征图 OUT 的起始内存地址指针。逻辑形状: (N, 4*C, H, W)
    # --- 2. 维度尺寸 (Dimensions) ---
    C,  # 通道数 (Channels)
    H,  # 图像高度 (Height)
    W,  # 图像宽度 (Width)
    # --- 3. 内存跨度 (Strides: 沿某个维度走1步需要跳过的内存元素数) ---
    # X (输入特征图) 的跨度
    x_sn,
    x_sc,
    x_sh,
    x_sw,
    # K (融合卷积核) 的跨度，k_sb 为分支(branch)维度的跨度
    k_sn,
    k_sb,
    k_sc,
    k_sk,
    k_sh,
    k_sw,
    # OUT (输出特征图) 的跨度
    o_sn,
    o_sc,
    o_sh,
    o_sw,
    # --- 4. 编译期常量 (constexpr) ---
    HW: tl.constexpr,  # 一张特征图的像素总数 (H * W)
    KS: tl.constexpr,  # 卷积核的边长 Kernel Size (如 3 表示 3x3 卷积，5 表示 5x5)
    BLOCK: tl.constexpr,  # 每个 GPU 线程块 (Block) 一次性处理的像素个数 (决定并发粒度)
):
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)
    n = pid_nc // C
    c = pid_nc % C

    hw_off = pid_hw * BLOCK + tl.arange(0, BLOCK)
    mask = hw_off < HW
    h = hw_off // W
    w = hw_off % W

    x_base = X + n * x_sn + c * x_sc

    # 获取原始中心像素，直接存储到未来的 `x` 对应的通道位置 (等同于 concat 第一项)
    x_val_center = tl.load(x_base + h * x_sh + w * x_sw, mask=mask, other=0.0)

    # 计算对应本 Channel 和 Batch 的、分别属于第0,1,2个分支卷积核的存储起点
    k_base_0 = K + n * k_sn + 0 * k_sb + c * k_sc
    k_base_1 = K + n * k_sn + 1 * k_sb + c * k_sc
    k_base_2 = K + n * k_sn + 2 * k_sb + c * k_sc

    # 分别给三个分支积攒和计算
    acc1 = tl.zeros([BLOCK], dtype=tl.float32)
    acc3 = tl.zeros([BLOCK], dtype=tl.float32)
    acc5 = tl.zeros([BLOCK], dtype=tl.float32)

    r = KS // 2
    for ki in range(KS):
        for kj in range(KS):
            # dilation = 1 (Branch 1)
            nh1 = h + (ki - r) * 1
            nw1 = w + (kj - r) * 1
            v1 = mask & (nh1 >= 0) & (nh1 < H) & (nw1 >= 0) & (nw1 < W)
            x_v1 = tl.load(x_base + nh1 * x_sh + nw1 * x_sw, mask=v1, other=0.0)
            k_v1 = tl.load(
                k_base_0 + (ki * KS + kj) * k_sk + h * k_sh + w * k_sw,
                mask=mask,
                other=0.0,
            )
            acc1 += x_v1 * k_v1

            # dilation = 3 (Branch 3)
            nh3 = h + (ki - r) * 3
            nw3 = w + (kj - r) * 3
            v3 = mask & (nh3 >= 0) & (nh3 < H) & (nw3 >= 0) & (nw3 < W)
            x_v3 = tl.load(x_base + nh3 * x_sh + nw3 * x_sw, mask=v3, other=0.0)
            k_v3 = tl.load(
                k_base_1 + (ki * KS + kj) * k_sk + h * k_sh + w * k_sw,
                mask=mask,
                other=0.0,
            )
            acc3 += x_v3 * k_v3

            # dilation = 5 (Branch 5)
            nh5 = h + (ki - r) * 5
            nw5 = w + (kj - r) * 5
            v5 = mask & (nh5 >= 0) & (nh5 < H) & (nw5 >= 0) & (nw5 < W)
            x_v5 = tl.load(x_base + nh5 * x_sh + nw5 * x_sw, mask=v5, other=0.0)
            k_v5 = tl.load(
                k_base_2 + (ki * KS + kj) * k_sk + h * k_sh + w * k_sw,
                mask=mask,
                other=0.0,
            )
            acc5 += x_v5 * k_v5

    out_base = OUT + n * o_sn + h * o_sh + w * o_sw

    # Fused Concat: 直接跳跃着向通道数四倍于原特征图的 OUT 赋值 => 即实现了 concat(dim=1)
    tl.store(out_base + (0 * C + c) * o_sc, x_val_center, mask=mask)
    tl.store(out_base + (1 * C + c) * o_sc, acc1, mask=mask)
    tl.store(out_base + (2 * C + c) * o_sc, acc3, mask=mask)
    tl.store(out_base + (3 * C + c) * o_sc, acc5, mask=mask)


@triton.jit
def _fused_ddpm_bwd_dx_kernel(
    # --- 1. 数据指针 ---
    DOUT,  # 损失函数对输出 OUT 的梯度指针。形状: (N, 4*C, H, W)
    K,  # 前向传播时保存下来的融合卷积核 K 的指针。形状: (N, 3, C, KS*KS, H, W)
    DX,  # 需要计算并写入的：损失函数对输入 X 的梯度指针。形状: (N, C, H, W)
    # --- 2. 维度尺寸 ---
    C,
    H,
    W,
    # --- 3. 内存跨度 ---
    do_sn,
    do_sc,
    do_sh,
    do_sw,  # DOUT (输出特征图的梯度) 的跨度
    k_sn,
    k_sb,
    k_sc,
    k_sk,
    k_sh,
    k_sw,  # K (融合卷积核) 的跨度
    dx_sn,
    dx_sc,
    dx_sh,
    dx_sw,  # DX (输入特征图的梯度) 的跨度
    # --- 4. 编译期常量 ---
    HW: tl.constexpr,
    KS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """反向 dx: 计算前向 fused 操作对 x 所有采样的总梯度"""
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)
    n = pid_nc // C
    c = pid_nc % C

    hw_off = pid_hw * BLOCK + tl.arange(0, BLOCK)
    mask = hw_off < HW
    h = hw_off // W
    w = hw_off % W

    do_base = DOUT + n * do_sn + h * do_sh + w * do_sw

    # 因为 x 在前向被四倍 copy(concat)，其自身直接作为第一个块，天然有一份来自 0...C 的输出梯度
    acc = tl.load(do_base + (0 * C + c) * do_sc, mask=mask, other=0.0).to(tl.float32)

    k_base_0 = K + n * k_sn + 0 * k_sb + c * k_sc
    k_base_1 = K + n * k_sn + 1 * k_sb + c * k_sc
    k_base_2 = K + n * k_sn + 2 * k_sb + c * k_sc

    r = KS // 2
    for ki in range(KS):
        for kj in range(KS):
            # dilation = 1 (Branch 1 反推)
            oh1 = h - (ki - r) * 1
            ow1 = w - (kj - r) * 1
            v1 = mask & (oh1 >= 0) & (oh1 < H) & (ow1 >= 0) & (ow1 < W)
            do_v1 = tl.load(
                DOUT + n * do_sn + (1 * C + c) * do_sc + oh1 * do_sh + ow1 * do_sw,
                mask=v1,
                other=0.0,
            )
            k_v1 = tl.load(
                k_base_0 + (ki * KS + kj) * k_sk + oh1 * k_sh + ow1 * k_sw,
                mask=v1,
                other=0.0,
            )
            acc += do_v1 * k_v1

            # dilation = 3 (Branch 3 反推)
            oh3 = h - (ki - r) * 3
            ow3 = w - (kj - r) * 3
            v3 = mask & (oh3 >= 0) & (oh3 < H) & (ow3 >= 0) & (ow3 < W)
            do_v3 = tl.load(
                DOUT + n * do_sn + (2 * C + c) * do_sc + oh3 * do_sh + ow3 * do_sw,
                mask=v3,
                other=0.0,
            )
            k_v3 = tl.load(
                k_base_1 + (ki * KS + kj) * k_sk + oh3 * k_sh + ow3 * k_sw,
                mask=v3,
                other=0.0,
            )
            acc += do_v3 * k_v3

            # dilation = 5 (Branch 5 反推)
            oh5 = h - (ki - r) * 5
            ow5 = w - (kj - r) * 5
            v5 = mask & (oh5 >= 0) & (oh5 < H) & (ow5 >= 0) & (ow5 < W)
            do_v5 = tl.load(
                DOUT + n * do_sn + (3 * C + c) * do_sc + oh5 * do_sh + ow5 * do_sw,
                mask=v5,
                other=0.0,
            )
            k_v5 = tl.load(
                k_base_2 + (ki * KS + kj) * k_sk + oh5 * k_sh + ow5 * k_sw,
                mask=v5,
                other=0.0,
            )
            acc += do_v5 * k_v5

    dx_base = DX + n * dx_sn + c * dx_sc
    tl.store(dx_base + h * dx_sh + w * dx_sw, acc, mask=mask)


@triton.jit
def _fused_ddpm_bwd_dk_kernel(
    # --- 1. 数据指针 ---
    X,  # 前向传播时保存下来的原始输入 X 的指针。形状: (N, C, H, W)
    DOUT,  # 损失函数对输出 OUT 的梯度指针。形状: (N, 4*C, H, W)
    DK,  # 需要计算并写入的：损失函数对融合核 K 的梯度指针。形状: (N, 3, C, KS*KS, H, W)
    # --- 2. 维度尺寸 ---
    C,
    H,
    W,
    # --- 3. 内存跨度 ---
    x_sn,
    x_sc,
    x_sh,
    x_sw,  # X (原始输入特征图) 的跨度
    do_sn,
    do_sc,
    do_sh,
    do_sw,  # DOUT (输出特征图的梯度) 的跨度
    dk_sn,
    dk_sb,
    dk_sc,
    dk_sk,
    dk_sh,
    dk_sw,  # DK (融合卷积核的梯度) 的跨度
    # --- 4. 编译期常量 ---
    HW: tl.constexpr,
    KS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """反向 dk: 计算三个感受野分枝带来的 kernel 梯度"""
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)
    n = pid_nc // C
    c = pid_nc % C

    hw_off = pid_hw * BLOCK + tl.arange(0, BLOCK)
    mask = hw_off < HW
    h = hw_off // W
    w = hw_off % W

    x_base = X + n * x_sn + c * x_sc
    do_base = DOUT + n * do_sn + h * do_sh + w * do_sw

    dk_base_0 = DK + n * dk_sn + 0 * dk_sb + c * dk_sc
    dk_base_1 = DK + n * dk_sn + 1 * dk_sb + c * dk_sc
    dk_base_2 = DK + n * dk_sn + 2 * dk_sb + c * dk_sc

    # 预加载该通道三个分支的后传梯度
    do_v1 = tl.load(do_base + (1 * C + c) * do_sc, mask=mask, other=0.0)
    do_v3 = tl.load(do_base + (2 * C + c) * do_sc, mask=mask, other=0.0)
    do_v5 = tl.load(do_base + (3 * C + c) * do_sc, mask=mask, other=0.0)

    r = KS // 2
    for ki in range(KS):
        for kj in range(KS):
            # dilation = 1 计算对应块
            nh1 = h + (ki - r) * 1
            nw1 = w + (kj - r) * 1
            v1 = mask & (nh1 >= 0) & (nh1 < H) & (nw1 >= 0) & (nw1 < W)
            x_v1 = tl.load(x_base + nh1 * x_sh + nw1 * x_sw, mask=v1, other=0.0)
            tl.store(
                dk_base_0 + (ki * KS + kj) * dk_sk + h * dk_sh + w * dk_sw,
                do_v1 * x_v1,
                mask=mask,
            )

            # dilation = 3 计算对应块
            nh3 = h + (ki - r) * 3
            nw3 = w + (kj - r) * 3
            v3 = mask & (nh3 >= 0) & (nh3 < H) & (nw3 >= 0) & (nw3 < W)
            x_v3 = tl.load(x_base + nh3 * x_sh + nw3 * x_sw, mask=v3, other=0.0)
            tl.store(
                dk_base_1 + (ki * KS + kj) * dk_sk + h * dk_sh + w * dk_sw,
                do_v3 * x_v3,
                mask=mask,
            )

            # dilation = 5 计算对应块
            nh5 = h + (ki - r) * 5
            nw5 = w + (kj - r) * 5
            v5 = mask & (nh5 >= 0) & (nh5 < H) & (nw5 >= 0) & (nw5 < W)
            x_v5 = tl.load(x_base + nh5 * x_sh + nw5 * x_sw, mask=v5, other=0.0)
            tl.store(
                dk_base_2 + (ki * KS + kj) * dk_sk + h * dk_sh + w * dk_sw,
                do_v5 * x_v5,
                mask=mask,
            )


def _get_block(HW):
    if HW <= 1024:
        return 256
    elif HW <= 4096:
        return 512
    elif HW <= 16384:
        return 512
    return 1024


class FusedDDPMFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, x, kernels, ks):
        # x: (N, C, H, W)
        # kernels: (N, 3, C, KS*KS, H, W)

        # 规避 AMP 时数据类型对不上的导致非法内存访问问题
        kernels = kernels.to(x.dtype).contiguous()
        x = x.contiguous()

        N, C, H, W = x.shape
        HW = H * W
        BLK = _get_block(HW)

        # 全融合 Kernel，直接向融合通道 4*C 大小的空间灌数据
        out = torch.empty((N, 4 * C, H, W), device=x.device, dtype=x.dtype)

        _fused_ddpm_fwd_kernel[(N * C, triton.cdiv(HW, BLK))](
            x,
            kernels,
            out,
            C,
            H,
            W,
            *x.stride(),
            *kernels.stride(),
            *out.stride(),
            HW=HW,
            KS=ks,
            BLOCK=BLK,
        )

        ctx.save_for_backward(x, kernels)
        ctx.ks = ks
        return out

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        x, kernels = ctx.saved_tensors
        ks = ctx.ks
        grad_output = grad_output.contiguous()

        N, C, H, W = x.shape
        HW = H * W
        BLK = _get_block(HW)

        # 同样一个并行线程一次反推算出整个 grad_x
        grad_x = torch.empty_like(x)
        _fused_ddpm_bwd_dx_kernel[(N * C, triton.cdiv(HW, BLK))](
            grad_output,
            kernels,
            grad_x,
            C,
            H,
            W,
            *grad_output.stride(),
            *kernels.stride(),
            *grad_x.stride(),
            HW=HW,
            KS=ks,
            BLOCK=BLK,
        )

        # 以及一次算出整个由三部分组成的 grad_k
        grad_k = torch.empty_like(kernels)
        _fused_ddpm_bwd_dk_kernel[(N * C, triton.cdiv(HW, BLK))](
            x,
            grad_output,
            grad_k,
            C,
            H,
            W,
            *x.stride(),
            *grad_output.stride(),
            *grad_k.stride(),
            HW=HW,
            KS=ks,
            BLOCK=BLK,
        )

        return grad_x, grad_k, None


class DDPM(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super(DDPM, self).__init__()
        self.kernel_size = kernel_size
        self.generator = nn.Conv2d(dim, 3 * dim * self.kernel_size**2, 1)
        self.fuse = nn.Conv2d(4 * dim, dim, 3, 1, 1)

    def forward(self, x, y):
        N, xC, xH, xW = x.size()

        # 将参数重置为 [N, Branch, Channel, KernelSize, H, W] => (N, 3, C, KS², H, W)
        kernels = self.generator(y).reshape(N, 3, xC, self.kernel_size**2, xH, xW)

        # Triton 直接吐出 shape 为 (N, 4*C, H, W) 的等价 concat 结果
        cat_out = FusedDDPMFunction.apply(x, kernels, self.kernel_size)

        # 最后喂入线性熔断层 fuse
        return self.fuse(cat_out)
