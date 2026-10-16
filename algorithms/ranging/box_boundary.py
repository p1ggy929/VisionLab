"""目标边界测距策略（源自项目 A）。

流程：检测框中心局部中位数 + 沿框边界多点采样 → 最近/最远/平均距离。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from core.config import settings
from core.geometry import Intrinsics
from ..base import Prediction
from .base import RangingResult, summarize_points
from .depth_sampler import DepthSampler


class BoxBoundaryStrategy:
    target_kind = "object"
    name = "box_boundary"

    def __init__(self, sampler: DepthSampler | None = None) -> None:
        self.sampler = sampler or DepthSampler()

    # ------------------------------------------------------------------
    def measure(self, depth_m: np.ndarray, intr: Intrinsics, prediction: Prediction,
                **params: Any) -> list[RangingResult]:
        """对一帧中的全部检测目标测距。结果以 target_kind='object' 返回。"""
        if depth_m is None or prediction is None or prediction.task != "detection":
            return []

        num_points = int(params.get("sample_points", settings.get("ranging.sample_points", 20)))
        calibration = params.get("calibration")
        per_target = params.get("per_target", False)

        results: list[RangingResult] = []
        for det in prediction.detections():
            box = det["bbox"]
            category = str(det.get("class_name") or "object")
            size_group = settings.class_group(category)

            # 1) 框中心局部中位数（项目 A 的做法，作为"整体距离"）
            center_value, center_pt = self.sampler.box_center_distance(depth_m, box, intr)
            if center_value is None:
                continue

            # 2) 框边界多点采样
            edge_points = self.sampler.box_edge_points(box, num_points)
            values, valid_points, pixel_x = self.sampler.sample_points(depth_m, edge_points)
            laterals = self.sampler.to_lateral_meters(np.asarray(pixel_x), values, intr)

            target_results: list[RangingResult] = []
            if values:
                target_results.extend(summarize_points(
                    values, valid_points,
                    target_kind=self.target_kind, category=category, size_group=size_group,
                    strategy=self.name, total_points=len(edge_points),
                    lateral=laterals, calibration=calibration,
                ))
            # 整体距离（中心）
            center_result = summarize_points(
                [center_value], [center_pt] if center_pt else [],
                target_kind=self.target_kind, category=category, size_group=size_group,
                strategy=self.name, total_points=1,
                lateral=[self.sampler.to_lateral_meters(np.asarray([center_pt[0]]), [center_value], intr)[0]]
                if center_pt else None,
                calibration=calibration,
            )
            if center_result:
                center = center_result[0]
                center.metric_kind = "center"
                target_results.append(center)

            for item in target_results:
                item.extra["bbox"] = [float(v) for v in box]
                item.extra["score"] = float(det.get("score", 0.0))
                item.extra["class_id"] = int(det.get("class_id", 0))

            if per_target:
                results.extend(target_results)
            else:
                # 逐目标保留全部指标，由上层按需筛选
                results.extend(target_results)

        return results


def summarize_per_detection(results: list[RangingResult]) -> list[dict]:
    """把逐目标结果折叠成"每个目标一条"的紧凑形式（供前端直接展示）。"""
    grouped: dict[tuple, dict] = {}
    for item in results:
        key = (item.category, tuple(item.extra.get("bbox", [])))
        entry = grouped.setdefault(key, {
            "category": item.category,
            "size_group": item.size_group,
            "bbox": item.extra.get("bbox"),
            "score": item.extra.get("score"),
            "ranging": {},
        })
        entry["ranging"][item.metric_kind] = item.to_dict()
    return list(grouped.values())
