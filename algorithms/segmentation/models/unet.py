"""UNet（移植自项目 B `Unet/Unet.py`）。

层名使用 snake_case；加载旧版权重时由
`algorithms.segmentation.weight_compat` 负责键名映射。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from . import blocks as nd


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = nd.DoubleConv(n_channels, 32)
        self.down1 = nd.Down(32, 64)
        self.down2 = nd.Down(64, 128)
        self.down3 = nd.Down(128, 256)
        self.down4 = nd.Down(256, 256)
        self.up1 = nd.Up(512, 128, bilinear)
        self.up2 = nd.Up(256, 64, bilinear)
        self.up3 = nd.Up(128, 32, bilinear)
        self.up4 = nd.Up(64, 32, bilinear)
        self.outc = nd.OutConv(32, n_classes)
        self.out = torch.sigmoid

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.out(self.outc(x))
