import torch
from torch import nn


class DDPM(nn.Module):
    def __init__(self, dim, kernel_size=3):
        """DDPM，利用nn.Unfold实现的动态卷积模块

        Args:
            dim (int): 第一个输入的通道数
            kernel_size (int): 指定的生成的卷积核的大小
        """
        super(DDPM, self).__init__()
        self.kernel_size = kernel_size
        self.generator = nn.Conv2d(dim, 3 * dim * self.kernel_size**2, 1)
        self.branch_1 = DepthDCOp(kernel_size=self.kernel_size, dilation=1)
        self.branch_3 = DepthDCOp(kernel_size=self.kernel_size, dilation=3)
        self.branch_5 = DepthDCOp(kernel_size=self.kernel_size, dilation=5)
        self.fuse = nn.Conv2d(4 * dim, dim, 3, 1, 1)

    def forward(self, x, y):
        N, xC, xH, xW = x.size()
        kernels = self.generator(y).reshape([N, 3, xC, self.kernel_size**2, xH, xW])
        kernels = kernels.unbind(dim=1)

        result_1 = self.branch_1(x, kernels[0])
        result_3 = self.branch_3(x, kernels[1])
        result_5 = self.branch_5(x, kernels[2])
        return self.fuse(torch.cat((x, result_1, result_3, result_5), dim=1))


class DepthDCOp(nn.Module):
    def __init__(self, kernel_size, dilation=1):
        super().__init__()
        self.unfold = nn.Unfold(
            kernel_size, dilation, padding=kernel_size // 2 * dilation, stride=1
        )

    def forward(self, x, kernel):
        N, xC, xH, xW = x.size()
        unfold_x = self.unfold(x).reshape([N, xC, -1, xH, xW])
        return (unfold_x * kernel).sum(2)
