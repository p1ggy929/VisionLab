"""道路边界测距策略（源自项目 B，并新增两项平台能力）。

沿用：掩膜最大外轮廓 → 等距采样 20–30 点 → 逐点读深度 → 有效点过滤（>0 且 ≤20 m）→ 最近/最远距离。

平台新增：
  1. **左右边界分离**：按采样点 x 与掩膜质心 x 比较，拆成 road_left / road_right 两组；
  2. **针孔反投影**：X = (u − ppx)·Z/fx，输出横向偏移 lateral_m 与欧氏距离，
     并给出可通行走廊宽度（左右边界横向偏移之差）。
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from core.config import settings
from core.geometry import Intrinsics, deproject, euclidean
from ..base import Prediction
from .base import RangingResult, split_left_right, summarize_points
from .depth_sampler import DepthSampler


class MaskBoundaryStrategy:
    target_kind = "road"
    name = "mask_boundary"

    def __init__(self, sampler: DepthSampler | None = None) -> None:
        self.sampler = sampler or DepthSampler()

    # ------------------------------------------------------------------
    @staticmethod
    def _largest_contour(mask: np.ndarray | None):
        if mask is None:
            return None, None
        binary = (np.asarray(mask) > 0).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None, None
        largest = max(contours, key=cv2.contourArea)
        moments = cv2.moments(largest)
        if moments["m00"] <= 0:
            return largest, None
        centroid = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
        return largest, centroid

    def measure(self, depth_m: np.ndarray, intr: Intrinsics, prediction: Prediction,
                **params: Any) -> list[RangingResult]:
        if depth_m is None or prediction is None or prediction.task != "segmentation":
            return []
        mask = prediction.get("mask")
        if mask is None:
            return []

        binary = (np.asarray(mask) > 0).astype(np.uint8)
        if binary.sum() == 0:
            return []

        num_points = int(params.get("sample_points", settings.get("ranging.sample_points", 20)))
        calibration = params.get("calibration")
        do_split = bool(params.get("road_split", settings.get("ranging.road_split", True)))
        do_reproject = bool(params.get("reproject", settings.get("ranging.reproject", True)))

        if binary.shape[:2] != depth_m.shape[:2]:
            binary = cv2.resize(binary, (depth_m.shape[1], depth_m.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
            if mask is not None and np.shape(mask)[:2] != depth_m.shape[:2]:
                pass

        contour, centroid = self._largest_contour(binary)
        if contour is None:
            return []

        if centroid is None:
            ys, xs = np.nonzero(binary)
            centroid = (float(xs.mean()), float(ys.mean())) if xs.size else (depth_m.shape[1] / 2.0,
                                                                            depth_m.shape[0] / 2.0)

        sample = self.sampler.contour_points(contour, num_points)
        if len(sample) == 0:
            return []

        results: list[RangingResult] = []
        side_data: dict[str, tuple[list[float], list[float]]] = {}

        def measure_group(points: np.ndarray, kind: str) -> None:
            if points is None or len(points) == 0:
                return
            values, valid_points, pixel_x = self.sampler.sample_points(depth_m, points)
            if not values:
                return
            laterals = self._laterals(pixel_x, values, intr, do_reproject)
            if laterals:
                side_data[kind] = (list(values), list(laterals))
            results.extend(summarize_points(
                values, valid_points, target_kind=kind, category=None, size_group=None,
                strategy=self.name, total_points=len(points),
                lateral=laterals, calibration=calibration,
            ))

        if do_split:
            left_pts, right_pts = split_left_right(sample, float(centroid[0]))
            measure_group(left_pts, "road_left")
            measure_group(right_pts, "road_right")
        else:
            measure_group(sample, "road_center")

        # 掩膜质心距离（道路整体远近）
        center_value, center_pt = self.sampler.box_center_distance(
            depth_m, [0, 0, depth_m.shape[1] - 1, depth_m.shape[0] - 1], intr
        )
        cx, cy = int(centroid[0]), int(centroid[1])
        cy = int(np.clip(cy, 0, depth_m.shape[0] - 1))
        cx = int(np.clip(cx, 0, depth_m.shape[1] - 1))
        centroid_value = self.sampler.patch_value(depth_m, cx, cy)  # noqa: SLF001 - 同包内复用
        if centroid_value is not None:
            lat = self._laterals([cx], [centroid_value], intr, do_reproject)
            center_results = summarize_points(
                [centroid_value], [(cx, cy)], target_kind="road_center", category=None,
                size_group=None, strategy=self.name, total_points=1,
                lateral=lat, calibration=calibration,
            )
            if center_results:
                center = center_results[0]
                center.metric_kind = "center"
                center.extra["centroid"] = [cx, cy]
                results.append(center)

        params.setdefault("intrinsics", intr)
        self._attach_corridor(results, side_data, do_reproject, params)
        return results

    # ------------------------------------------------------------------
    @staticmethod
    def _laterals(pixel_x, values, intr: Intrinsics, enabled: bool) -> list[float] | None:
        if not enabled:
            return None
        out = []
        for px, z in zip(pixel_x, values):
            x, _, _ = deproject(float(px), 0.0, float(z), intr)
            out.append(x)
        return out

    @staticmethod
    def _lateral_at(depths: list[float], laterals: list[float], z_ref: float) -> float | None:
        """把「深度 -> 横向偏移」看作一条曲线，插值到参考深度。

        走廊宽度必须取同一深度处的左右横向差，否则把不同深度的两个边界点相减
        得到的不是走廊宽度（近处边界更窄、远处更宽，直接相减会显著偏小）。
        """
        if not depths or not laterals:
            return None
        pairs = sorted(zip(depths, laterals))
        zs = [p[0] for p in pairs]
        xs = [p[1] for p in pairs]
        return float(np.interp(z_ref, zs, xs))

    @classmethod
    def _attach_corridor(cls, results: list[RangingResult],
                         side_data: dict[str, tuple[list[float], list[float]]],
                         enabled: bool, params: dict) -> None:
        """计算可通行走廊宽度。

        必须在**同一参考深度**上取左右横向差：近处可见宽度受图像视场限制、
        远处更宽，直接把不同深度的两个边界点相减得到的不是走廊宽度。
        参考深度默认 5 m（可通过 ranging.corridor_ref_depth_m 配置），
        且不小于两侧最近边界深度（比最近边界更近处没有可用观测）。
        """
        if not enabled or "road_left" not in side_data or "road_right" not in side_data:
            return
        l_z, l_x = side_data["road_left"]
        r_z, r_x = side_data["road_right"]

        config_ref = float(params.get("corridor_ref_depth_m",
                                      settings.get("ranging.corridor_ref_depth_m", 5.0)))
        near = min(min(l_z), min(r_z))
        z_ref = max(config_ref, near)

        left_lat = cls._lateral_at(l_z, l_x, z_ref)
        right_lat = cls._lateral_at(r_z, r_x, z_ref)
        if left_lat is None or right_lat is None:
            return
        width = abs(right_lat - left_lat)

        intr = params.get("intrinsics")
        fov_limited = False
        if intr is not None:
            half_visible = float(intr.ppx) * z_ref / max(float(intr.fx), 1e-6)
            fov_limited = max(abs(left_lat), abs(right_lat)) >= half_visible * 0.95

        for item in results:
            item.extra["corridor_width_m"] = round(width, 3)
            item.extra["corridor_depth_ref_m"] = round(z_ref, 3)
            item.extra["corridor_fov_limited"] = bool(fov_limited)


def road_summary(results: list[RangingResult]) -> dict:
    """把道路测距结果整理成前端面板所需的紧凑结构。"""
    summary: dict[str, Any] = {"left": {}, "right": {}, "center": {}, "corridor_width_m": None}
    for item in results:
        if item.target_kind == "road_left":
            summary["left"][item.metric_kind] = item.to_dict()
        elif item.target_kind == "road_right":
            summary["right"][item.metric_kind] = item.to_dict()
        elif item.target_kind == "road_center":
            summary["center"][item.metric_kind] = item.to_dict()
        if item.extra.get("corridor_width_m") is not None:
            summary["corridor_width_m"] = item.extra["corridor_width_m"]
    return summary
