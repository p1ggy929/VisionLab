"""统一测距内核。"""

from .base import RangingResult, RangingStrategy, split_left_right, summarize_points  # noqa: F401
from .box_boundary import BoxBoundaryStrategy, summarize_per_detection  # noqa: F401
from .depth_sampler import DepthSampler  # noqa: F401
from .mask_boundary import MaskBoundaryStrategy, road_summary  # noqa: F401

STRATEGIES = {
    "box_boundary": BoxBoundaryStrategy,
    "mask_boundary": MaskBoundaryStrategy,
}

__all__ = ["RangingResult", "RangingStrategy", "DepthSampler", "BoxBoundaryStrategy",
           "MaskBoundaryStrategy", "STRATEGIES", "split_left_right", "summarize_points",
           "summarize_per_detection", "road_summary"]
