import torch.nn as nn
from .convBlock import ConvBlock


class CNN_AVG_POOL(nn.Module):
    def __init__(self, out_size=9, last_filter_size=512, p=0.2, alpha=1):
        super(CNN_AVG_POOL, self).__init__()

        if alpha < 0:
            print("alpha cant be negative, defaulting to 1.0")
            alpha = 1.0

        self.blocks = nn.ModuleList(
            [
                ConvBlock(
                    in_out_shapes=[3, int(64 // alpha)],
                    kernels=[7, 3],
                    strides=[2, 1],
                    paddings=[3, 1],
                ),
                # 32 x 56 x 56
                ConvBlock(
                    in_out_shapes=[int(64 // alpha), int(128 // alpha)],
                    kernels=[5, 3],
                    strides=[2, 1],
                    paddings=[2, 1],
                ),
                # 64 x 14 x 14
                ConvBlock(
                    in_out_shapes=[int(128 // alpha), int(256 // alpha)],
                    kernels=[3, 3],
                    strides=[2, 1],
                    paddings=[1, 1],
                ),
                ConvBlock(
                    in_out_shapes=[int(256 // alpha), int(last_filter_size // alpha)],
                    kernels=[3, 3],
                    strides=[2, 1],
                    paddings=[1, 1],
                ),
            ]
        )
        self.convblocks = nn.Sequential(*self.blocks)

        self.gavgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p)
        self.fc1 = nn.Linear(int(last_filter_size // alpha), out_size)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            if getattr(m, "is_last_in_block", False):
                nn.init.constant_(m.weight, 0)
            else:
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
        out = self.fc1(self.dropout(out_avgpool))
        return out
