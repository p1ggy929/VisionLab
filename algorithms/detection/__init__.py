"""检测插件：YOLO（Ultralytics）、深度基线检测器、仿真真值检测器。"""

from .depth_adapter import DepthObstacleDetector  # noqa: F401
from .sim_adapter import SimGroundTruthDetector  # noqa: F401
from .yolo_adapter import YoloDetectorAdapter, merge_detections  # noqa: F401

__all__ = ["YoloDetectorAdapter", "DepthObstacleDetector", "SimGroundTruthDetector",
           "merge_detections"]
