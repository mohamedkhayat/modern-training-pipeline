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
        self.identity = None
        if strides[0] != 1 or in_out_shapes[0] != in_out_shapes[1]:
            self.identity = nn.Conv2d(
                in_out_shapes[0], in_out_shapes[1], 1, strides[0], 0
            )
        self.batchnorm2 = nn.BatchNorm2d(in_out_shapes[1])
        if self.identity is None:
            self.maxpool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        out_conv1 = self.batchnorm1(self.conv1(x))
        out_conv1 = F.relu(out_conv1)
        out_conv2 = self.batchnorm2(self.conv2(out_conv1))

        if self.identity is not None:
            identity = self.identity(x)
            out = F.relu(out_conv2 + identity)
        else:
            out = F.relu(out_conv2)
            out = self.maxpool(out)

        return out