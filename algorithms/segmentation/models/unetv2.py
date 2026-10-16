"""UNetV2（PVTv2 骨干 + 深监督，移植自项目 B `Unet/unet_v2/UNet_v2.py`）。

依赖 `timm`（PVTv2 使用 DropPath / trunc_normal_）。
若未安装 timm，构造时抛出明确的 ModelError，由上层降级处理。
"""

from __future__ import annotations

import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.exceptions import ModelError


def _import_pvtv2():
    """导入 PVTv2 骨干，兼容 timm 0.9（timm.models.layers）与 timm 1.x（timm.layers）。"""
    try:
        import timm  # noqa: F401
    except ImportError as exc:  # pragma: no cover - 依赖缺失
        raise ModelError(
            "UNetV2 需要 timm 依赖，请执行: pip install timm==0.9.16"
        ) from exc

    # timm 1.x 移除了 timm.models.layers，这里做一次兼容垫片
    import sys
    import types

    try:
        import timm.models.layers  # noqa: F401
    except Exception:  # noqa: BLE001
        try:
            import timm.layers as _layers

            shim = types.ModuleType("timm.models.layers")
            for attr in ("DropPath", "to_2tuple", "trunc_normal_", "PatchEmbed",
                         "Mlp", "LayerNorm", "ConvNormAct"):
                if hasattr(_layers, attr):
                    setattr(shim, attr, getattr(_layers, attr))
            sys.modules["timm.models.layers"] = shim
        except Exception as exc:  # noqa: BLE001
            raise ModelError(f"timm 版本不兼容，无法导入 PVTv2 依赖: {exc}") from exc

    from ..backbones.pvtv2 import pvt_v2_b2

    return pvt_v2_b2


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(x))


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.bn(self.conv(x))


class Encoder(nn.Module):
    def __init__(self, pretrain_path):
        super().__init__()
        pvt_v2_b2 = _import_pvtv2()
        self.backbone = pvt_v2_b2()
        if pretrain_path is None:
            warnings.warn("please provide the pretrained pvt model. Not using pretrained model.")
        elif not os.path.isfile(pretrain_path):
            warnings.warn(f"path: {pretrain_path} does not exists. Not using pretrained model.")
        else:
            save_model = torch.load(pretrain_path, map_location="cpu")
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.backbone.load_state_dict(model_dict)

    def forward(self, x):
        return self.backbone(x)


class SDI(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1) for _ in range(4)]
        )

    def forward(self, xs, anchor):
        ans = torch.ones_like(anchor)
        target_h, target_w = anchor.shape[-2], anchor.shape[-1]
        for i, x in enumerate(xs):
            h, w = x.shape[-2], x.shape[-1]
            if h > target_h or w > target_w:
                x = F.adaptive_avg_pool2d(x, (target_h, target_w))
            elif h < target_h or w < target_w:
                x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=True)
            ans = ans * self.convs[i](x)
        return ans


class UNetV2(nn.Module):
    """use SpatialAtt + ChannelAtt（项目 B 原始实现）。"""

    def __init__(self, channel=32, n_classes=1, deep_supervision=True, pretrained_path=None):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = Encoder(pretrained_path)

        self.ca_1 = ChannelAttention(64)
        self.sa_1 = SpatialAttention()
        self.ca_2 = ChannelAttention(128)
        self.sa_2 = SpatialAttention()
        self.ca_3 = ChannelAttention(320)
        self.sa_3 = SpatialAttention()
        self.ca_4 = ChannelAttention(512)
        self.sa_4 = SpatialAttention()

        self.Translayer_1 = BasicConv2d(64, channel, 1)
        self.Translayer_2 = BasicConv2d(128, channel, 1)
        self.Translayer_3 = BasicConv2d(320, channel, 1)
        self.Translayer_4 = BasicConv2d(512, channel, 1)

        self.sdi_1 = SDI(channel)
        self.sdi_2 = SDI(channel)
        self.sdi_3 = SDI(channel)
        self.sdi_4 = SDI(channel)

        self.seg_outs = nn.ModuleList([nn.Conv2d(channel, n_classes, 1, 1) for _ in range(4)])

        self.deconv2 = nn.ConvTranspose2d(channel, channel, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv3 = nn.ConvTranspose2d(channel, channel, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv4 = nn.ConvTranspose2d(channel, channel, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv5 = nn.ConvTranspose2d(channel, channel, kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, x):
        seg_outs = []
        f1, f2, f3, f4 = self.encoder(x)

        # 严格保持项目 B 原始前向顺序：先通道注意力再空间注意力
        f1 = self.ca_1(f1) * f1
        f1 = self.Translayer_1(self.sa_1(f1) * f1)
        f2 = self.ca_2(f2) * f2
        f2 = self.Translayer_2(self.sa_2(f2) * f2)
        f3 = self.ca_3(f3) * f3
        f3 = self.Translayer_3(self.sa_3(f3) * f3)
        f4 = self.ca_4(f4) * f4
        f4 = self.Translayer_4(self.sa_4(f4) * f4)

        f41 = self.sdi_4([f1, f2, f3, f4], f4)
        f31 = self.sdi_3([f1, f2, f3, f4], f3)
        f21 = self.sdi_2([f1, f2, f3, f4], f2)
        f11 = self.sdi_1([f1, f2, f3, f4], f1)

        seg_outs.append(self.seg_outs[0](f41))
        y = self.deconv2(f41) + f31
        seg_outs.append(self.seg_outs[1](y))
        y = self.deconv3(y) + f21
        seg_outs.append(self.seg_outs[2](y))
        y = self.deconv4(y) + f11
        seg_outs.append(self.seg_outs[3](y))

        for i, o in enumerate(seg_outs):
            seg_outs[i] = F.interpolate(o, scale_factor=4, mode="bilinear")

        if self.deep_supervision:
            return seg_outs[::-1]
        return seg_outs[-1]
