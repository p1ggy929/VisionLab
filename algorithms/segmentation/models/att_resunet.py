"""Attention-ResUNet（移植自项目 B `Unet/AtteResUnet.py`）。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import blocks as nd
from .blocks import Attention_block, Attexpansive_block, expansive_block, final_block

double_conv = nd.contracting_block


class AttResUNet(nn.Module):
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

        self.conv_decode4 = Attexpansive_block(1024, 512)
        self.att4 = Attention_block(512, 512, 256)
        self.double_conv4 = double_conv(1024, 512)
        self.conv_decode3 = Attexpansive_block(512, 256)
        self.att3 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.double_conv3 = double_conv(512, 256)
        self.conv_decode2 = Attexpansive_block(256, 128)
        self.att2 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.double_conv2 = double_conv(256, 128)
        self.conv_decode1 = Attexpansive_block(128, 64)
        self.att1 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.double_conv1 = double_conv(128, 64)
        self.conv_decode0 = expansive_block(64, 32, 32)

        self.final_layer = final_block(32, out_channel)
        self.out = torch.sigmoid

    def forward(self, x):
        x = self.layer0(x)
        encode_block1 = self.layer1(x)
        encode_block2 = self.layer2(encode_block1)
        encode_block3 = self.layer3(encode_block2)
        encode_block4 = self.layer4(encode_block3)

        bottleneck = self.bottleneck(encode_block4)

        decode_block4 = self.conv_decode4(bottleneck)
        encode_block4 = self.att4(g=decode_block4, x=encode_block4)
        decode_block4 = torch.cat([encode_block4, decode_block4], dim=1)
        decode_block4 = self.double_conv4(decode_block4)

        decode_block3 = self.conv_decode3(encode_block4)
        encode_block3 = self.att3(g=decode_block3, x=encode_block3)
        decode_block3 = torch.cat((encode_block3, decode_block3), dim=1)
        decode_block3 = self.double_conv3(decode_block3)

        decode_block2 = self.conv_decode2(encode_block3)
        encode_block2 = self.att2(g=decode_block2, x=encode_block2)
        decode_block2 = torch.cat((encode_block2, decode_block2), dim=1)
        decode_block2 = self.double_conv2(decode_block2)

        decode_block1 = self.conv_decode1(encode_block2)
        encode_block1 = self.att1(g=decode_block1, x=encode_block1)
        decode_block1 = torch.cat((encode_block1, decode_block1), dim=1)
        decode_block1 = self.double_conv1(decode_block1)

        decode_block0 = self.conv_decode0(decode_block1)
        return self.out(self.final_layer(decode_block0))
