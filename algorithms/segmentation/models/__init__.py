"""分割模型库（沿用项目 B 的四个结构与权重命名）。"""

from .att_resunet import AttResUNet  # noqa: F401
from .resunet import ResUNet  # noqa: F401
from .unet import UNet  # noqa: F401

__all__ = ["UNet", "ResUNet", "AttResUNet", "build_segmentation_model", "MODEL_FACTORY"]


def _unetv2(**kwargs):
    # 延迟导入：UNetV2 依赖 timm，未安装时不应影响其余三个模型
    from .unetv2 import UNetV2

    return UNetV2(**kwargs)


MODEL_FACTORY = {
    "unet": lambda **kw: UNet(n_channels=kw.get("n_channels", 3), n_classes=kw.get("n_classes", 1),
                              bilinear=kw.get("bilinear", True)),
    "resunet": lambda **kw: ResUNet(in_channel=kw.get("n_channels", 3),
                                    out_channel=kw.get("n_classes", 1), pretrained=False),
    "att_resunet": lambda **kw: AttResUNet(in_channel=kw.get("n_channels", 3),
                                           out_channel=kw.get("n_classes", 1), pretrained=False),
    "unetv2": lambda **kw: _unetv2(n_classes=kw.get("n_classes", 1),
                                   deep_supervision=kw.get("deep_supervision", True),
                                   pretrained_path=kw.get("pretrained_path")),
}


def build_segmentation_model(arch: str, **kwargs):
    """统一工厂：修复项目 B 图形界面中"切换模型实际未生效"的缺陷。"""
    key = str(arch).lower()
    if key not in MODEL_FACTORY:
        from core.exceptions import ModelError

        raise ModelError(f"不支持的分割结构: {arch}，可选: {list(MODEL_FACTORY)}")
    return MODEL_FACTORY[key](**kwargs)
