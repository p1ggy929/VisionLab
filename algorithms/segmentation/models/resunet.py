"""ResUNet（移植自项目 B `Unet/ResUnet.py`）。"""

from __future__ import annotations

import torch
import torch.nn as nn

from . import blocks as nd


class ResUNet(nn.Module):
    def __init__(self, in_channel, out_channel, pretrained=False):
        super().__init__()
        self.resnet = nd.resnet34_backbone(pretrained)

        self.layer0 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
        )
        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4

        self.bottleneck = nn.Sequential(
            nn.Conv2d(kernel_size=(3, 3), in_channels=512, out_channels=1024, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.Conv2d(kernel_size=(3, 3), in_channels=1024, out_channels=1024, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        )

        self.conv_decode4 = nd.ExpansiveBlock(1024 + 512, 512, 512)
        self.conv_decode3 = nd.ExpansiveBlock(512 + 256, 256, 256)
        self.conv_decode2 = nd.ExpansiveBlock(256 + 128, 128, 128)
        self.conv_decode1 = nd.ExpansiveBlock(128 + 64, 64, 64)
        self.conv_decode0 = nd.ExpansiveBlock(64, 32, 32)

        self.final_layer = nd.final_block(32, out_channel)
        self.out = torch.sigmoid

    def forward(self, x):
        x = self.layer0(x)
        encode_block1 = self.layer1(x)
        encode_block2 = self.layer2(encode_block1)
        encode_block3 = self.layer3(encode_block2)
        encode_block4 = self.layer4(encode_block3)

        bottleneck = self.bottleneck(encode_block4)

        decode_block4 = self.conv_decode4(bottleneck, encode_block4)
        decode_block3 = self.conv_decode3(decode_block4, encode_block3)
        decode_block2 = self.conv_decode2(decode_block3, encode_block2)
        decode_block1 = self.conv_decode1(decode_block2, encode_block1)
        decode_block0 = self.conv_decode0(decode_block1)

        return self.out(self.final_layer(decode_block0))
