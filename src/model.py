import torch.nn as nn
import torch.nn.functional as F


# 3 32 64 128 64 n_classes
class ConvBlock(nn.Module):
    def __init__(self, in_out_shapes, kernels, strides, paddings):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_out_shapes[0], in_out_shapes[1], kernels[0], strides[0], paddings[0]
        )

        self.batchnorm1 = nn.BatchNorm2d(in_out_shapes[1])

        self.conv2 = nn.Conv2d(
            in_out_shapes[1], in_out_shapes[1], kernels[1], strides[1], paddings[1]
        )
        self.batchnorm2 = nn.BatchNorm2d(in_out_shapes[1])

        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        out_conv1 = F.relu(self.batchnorm1(self.conv1(x)))
        out_conv2 = F.relu(self.batchnorm2(self.conv2(out_conv1)))
        out = self.maxpool(out_conv2)

        return out


class CNN(nn.Module):
    def __init__(self, hidden_size, out_size, p):
        super(CNN, self).__init__()

        self.blocks = nn.ModuleList(
            [
                ConvBlock([3, 32], [7, 3], [2, 1], [3, 1]),
                ConvBlock([32, 64], [3, 3], [1, 1], [1, 1]),
                ConvBlock([64, 128], [3, 3], [1, 1], [1, 1]),
            ]
        )
        # shape after this block : 64 x 14 x 14
        self.convblocks = nn.Sequential(*self.blocks)

        self.gavgpool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(128, hidden_size)
        self.batchnorm1 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(p)

        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.batchnorm2 = nn.BatchNorm1d(hidden_size // 2)

        self.fc3 = nn.Linear(hidden_size // 2, out_size)

    def forward(self, x):
        out_convblock = self.convblocks(x)
        out_avgpool = self.gavgpool(out_convblock)
        out_avgpool = out_avgpool.view(out_avgpool.size(0), -1)
        out_fc1 = F.relu(self.fc1(out_avgpool))
        out_bn1 = self.dropout(self.batchnorm1(out_fc1))
        out_fc2 = F.relu(self.fc2(out_bn1))
        out_bn2 = self.dropout(self.batchnorm2(out_fc2))
        out = self.fc3(out_bn2)

        return out
