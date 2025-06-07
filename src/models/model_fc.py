import torch.nn as nn
from .convBlock import ConvBlock
import torch.nn.functional as F
import math

class CNN_FC(nn.Module):
    def __init__(self, hidden_size=256, out_size=9, p=0.3, last_filter_size=128):
        super(CNN_FC, self).__init__()

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
        last_filter_size = self.blocks[-1].in_out_shapes[1]
        # shape after this block : 128 x 3 x 3
        self.convblocks = nn.Sequential(*self.blocks)

        self.flatten = nn.Flatten()

        n_blocks = len([block for block in self.blocks if isinstance(block, ConvBlock)])
        fc1_in_features = last_filter_size * int(
            math.pow(math.floor(224 / (math.pow(2, n_blocks))), 2)
        )
        self.fc1 = nn.Linear(fc1_in_features, hidden_size)
        self.batchnorm1 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(p)

        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.batchnorm2 = nn.BatchNorm1d(hidden_size // 2)

        self.fc3 = nn.Linear(hidden_size // 2, out_size)
        
        self.fc1 = nn.Linear(last_filter_size,out_size)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity='relu')
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
        out_flatten = self.flatten(out_convblock)
        out_fc1 = self.batchnorm1(self.fc1(out_flatten))
        out_fc1 = self.batchnorm1(self.fc1(out_flatten))
        out_bn1 = self.dropout(F.relu(out_fc1))
        out_fc2 = self.batchnorm2(self.fc2(out_bn1))
        out_bn2 = F.relu(out_fc2)
        out = self.fc3(out_bn2)

        return out
