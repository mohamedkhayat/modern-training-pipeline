import torch.nn as nn
from .convBlock import ConvBlock


class CNN_AVG_POOL(nn.Module):
    def __init__(self, out_size=9, last_filter_size=128):
        super(CNN_AVG_POOL, self).__init__()

        self.blocks = nn.ModuleList(
            [
                ConvBlock(
                    in_out_shapes=[3, 32],
                    kernels=[7, 3],
                    strides=[2, 1],
                    paddings=[3, 1],
                ),
                # 32 x 112 x 112
                ConvBlock(
                    in_out_shapes=[32, 32],
                    kernels=[3, 3],
                    strides=[1, 1],
                    paddings=[1, 1],
                ),
                # 32 x 56 x 56
                ConvBlock(
                    in_out_shapes=[32, 64],
                    kernels=[5, 3],
                    strides=[2, 1],
                    paddings=[2, 1],
                ),
                # 64 x 28 x 28
                ConvBlock(
                    in_out_shapes=[64, 64],
                    kernels=[3, 3],
                    strides=[1, 1],
                    paddings=[1, 1],
                ),
                # 64 x 14 x 14
                ConvBlock(
                    in_out_shapes=[64, last_filter_size],
                    kernels=[3, 3],
                    strides=[2, 1],
                    paddings=[1, 1],
                ),
                # 128 x 7 x 7
                ConvBlock(
                    in_out_shapes=[last_filter_size, last_filter_size],
                    kernels=[3, 3],
                    strides=[1, 1],
                    paddings=[1, 1],
                ),
                # 128 x 3 x 3
            ]
        )
        # shape after this block : 128 x 3 x 3
        self.convblocks = nn.Sequential(*self.blocks)

        self.gavgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(last_filter_size, out_size)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out_convblock = self.convblocks(x)
        out_avgpool = self.gavgpool(out_convblock)
        out_avgpool = out_avgpool.view(out_avgpool.size(0), -1)
        out = self.fc1(out_avgpool)
        return out
