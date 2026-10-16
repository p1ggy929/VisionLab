"""UNet（移植自项目 B `Unet/Unet.py`，保持层命名以匹配既有权重）。"""

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
        self.Down1 = nd.Down(32, 64)
        self.Down2 = nd.Down(64, 128)
        self.Down3 = nd.Down(128, 256)
        self.Down4 = nd.Down(256, 256)
        self.Up1 = nd.Up(512, 128, bilinear)
        self.Up2 = nd.Up(256, 64, bilinear)
        self.Up3 = nd.Up(128, 32, bilinear)
        self.Up4 = nd.Up(64, 32, bilinear)
        self.outc = nd.OutConv(32, n_classes)
        self.out = torch.sigmoid

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.Down1(x1)
        x3 = self.Down2(x2)
        x4 = self.Down3(x3)
        x5 = self.Down4(x4)
        x = self.Up1(x5, x4)
        x = self.Up2(x, x3)
        x = self.Up3(x, x2)
        x = self.Up4(x, x1)
        return self.out(self.outc(x))
