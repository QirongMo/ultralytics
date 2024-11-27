
import torch
import torch.nn as nn
from .conv import Conv

__all__ = (
    "CReLu",
    "InceptionV2"
)


class CReLu(nn.Module):
    def __init__(self):
        super(CReLu, self).__init__()
        self.relu = nn.ReLU()
    def forward(self, x):
        x = torch.cat([x, -x], axis=1)
        x = self.relu(x)
        return x


class InceptionV2(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        out_channel = in_channels//4
        self.branch1 = Conv(in_channels, out_channel, 1)

        self.branch2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            Conv(in_channels, out_channel, 1)
        )
        self.branch3 = nn.Sequential(
            Conv(in_channels, out_channel, 1),
            Conv(out_channel, out_channel, 3, 1)
        )
        self.branch4 = nn.Sequential(
            Conv(in_channels, out_channel, 1),
            Conv(out_channel, out_channel, 3, 1),
            Conv(out_channel, out_channel, 3, 1)
        )

    def forward(self, inputs):
        x1 = self.branch1(inputs)
        x2 = self.branch2(inputs)
        x3 = self.branch3(inputs)
        x4 = self.branch4(inputs)
        return torch.cat([x1, x2, x3, x4], axis=1)

