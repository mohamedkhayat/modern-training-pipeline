import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_out_shapes, kernels, strides, paddings):
        super(ConvBlock, self).__init__()
        self.in_out_shapes = in_out_shapes
        self.conv1 = nn.Conv2d(
            in_out_shapes[0], in_out_shapes[1], kernels[0], strides[0], paddings[0]
        )

        self.batchnorm1 = nn.BatchNorm2d(in_out_shapes[1])
        self.conv2 = nn.Conv2d(
            in_out_shapes[1], in_out_shapes[1], kernels[1], strides[1], paddings[1]
        )

        self.silu = nn.SiLU()
        self.identity = None
        if strides[0] != 1 or in_out_shapes[0] != in_out_shapes[1]:
            self.identity = nn.Conv2d(
                in_out_shapes[0], in_out_shapes[1], 1, strides[0], 0
            )
        self.batchnorm2 = nn.BatchNorm2d(in_out_shapes[1])
        self.batchnorm2.is_last_in_block = True

    def forward(self, x):
        out_conv1 = self.batchnorm1(self.conv1(x))
        out_conv1 = self.silu(out_conv1)
        out_conv2 = self.batchnorm2(self.conv2(out_conv1))

        identity = x if self.identity is None else self.identity(x)

        out = self.silu(out_conv2 + identity)

        return out
