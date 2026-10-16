"""分割插件：四个 U-Net 变体 + 传统道路分割器。"""

from .classical_adapter import ClassicalRoadSegmenter  # noqa: F401
from .unet_adapter import ARCH_LABELS, UNetSegmenterAdapter  # noqa: F401

__all__ = ["UNetSegmenterAdapter", "ClassicalRoadSegmenter", "ARCH_LABELS"]
