"""统一测距内核：结果契约与策略协议。

设计要点（二合一的技术枢纽）：
  "目标边界测距"与"道路边界测距"本质是同一件事——
  在指定图像区域内采样深度、剔除无效值、换算为物理距离；
  差异仅在于采样点来自"检测框"还是"分割掩膜轮廓"。
  因此共享 DepthSampler 与 RangingResult，仅策略不同。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Protocol

import numpy as np

from core.geometry import Intrinsics, deproject, euclidean


@dataclass
class RangingResult:
    """统一测距结果。"""

    target_kind: str            # object | road_left | road_right | road_center
    metric_kind: str            # min | max | mean | center
    depth_m: float              # 沿光轴深度（既有 A/B 的口径）
    raw_depth_m: float          # 校正前
    corrected_depth_m: float    # 校正后
    lateral_m: float | None = None      # 横向偏移 X=(u-ppx)Z/fx（平台新增）
    euclidean_m: float | None = None    # 欧氏距离（平台新增）
    valid_points: int = 0
    total_points: int = 0
    sample_points: list = field(default_factory=list)
    category: str | None = None
    size_group: str | None = None
    strategy: str = ""
    calibration_applied: str | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["depth_m"] = _round(self.depth_m)
        data["raw_depth_m"] = _round(self.raw_depth_m)
        data["corrected_depth_m"] = _round(self.corrected_depth_m)
        if self.lateral_m is not None:
            data["lateral_m"] = _round(self.lateral_m)
        if self.euclidean_m is not None:
            data["euclidean_m"] = _round(self.euclidean_m)
        return data


def _round(value: float | None, digits: int = 4) -> float | None:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return None
    return round(float(value), digits)


class RangingStrategy(Protocol):
    target_kind: str
    name: str

    def measure(self, depth_m: np.ndarray, intr: Intrinsics, prediction, **params) -> list[RangingResult]:
        ...


def summarize_points(values: list[float], points: list[tuple[int, int]], *,
                     target_kind: str, category: str | None, size_group: str | None,
                     strategy: str, total_points: int,
                     lateral: list[float] | None = None,
                     calibration=None) -> list[RangingResult]:
    """把一组有效采样点的深度值汇总为 min/max/mean 三条统一结果。"""
    if not values:
        return []

    arr = np.asarray(values, dtype=np.float64)
    idx_min = int(np.argmin(arr))
    idx_max = int(np.argmax(arr))
    results: list[RangingResult] = []

    def build(kind: str, value: float, index: int) -> RangingResult:
        raw = float(value)
        corrected, scale_info = raw, None
        if calibration is not None:
            corrected = float(calibration.apply(raw))
            scale_info = calibration.name
        lat = None
        euc = None
        if lateral is not None and index < len(lateral):
            lat = float(lateral[index])
            if target_kind.startswith("road"):
                euc = euclidean(lat, 0.0, corrected)
            else:
                euc = euclidean(lat, 0.0, corrected)
        extra = {}
        if scale_info:
            extra["calibration"] = scale_info
        if 0 <= index < len(points):
            extra["nearest_point" if kind == "min" else ("farthest_point" if kind == "max" else "point")] = list(points[index])
        return RangingResult(
            target_kind=target_kind, metric_kind=kind,
            depth_m=corrected, raw_depth_m=raw, corrected_depth_m=corrected,
            lateral_m=lat, euclidean_m=euc,
            valid_points=len(values), total_points=total_points,
            sample_points=[list(p) for p in points],
            category=category, size_group=size_group, strategy=strategy,
            calibration_applied=scale_info, extra=extra,
        )

    results.append(build("min", float(arr[idx_min]), idx_min))
    results.append(build("max", float(arr[idx_max]), idx_max))
    results.append(build("mean", float(arr.mean()), int(np.argmin(np.abs(arr - arr.mean())))))
    return results


def attach_visual_hints(result: RangingResult) -> dict:
    """供叠加渲染使用：最近/最远点坐标。"""
    extra = result.extra or {}
    return {
        "target_kind": result.target_kind,
        "metric_kind": result.metric_kind,
        "depth_m": result.depth_m,
        "sample_points": result.sample_points,
        "_nearest_point": extra.get("nearest_point"),
        "_farthest_point": extra.get("farthest_point"),
    }


def split_left_right(points: np.ndarray, center_x: float) -> tuple[np.ndarray, np.ndarray]:
    """按 x 相对质心把轮廓采样点分成左右两组。"""
    if points is None or len(points) == 0:
        return np.empty((0, 2), dtype=np.int32), np.empty((0, 2), dtype=np.int32)
    pts = np.asarray(points).reshape(-1, 2)
    left = pts[pts[:, 0] < center_x]
    right = pts[pts[:, 0] >= center_x]
    return left, right
