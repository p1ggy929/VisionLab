"""深度基线检测器（无权重时的可运行兜底）。

这不是"假数据"，而是一个真实可用的传统算法：
  1. 从深度图稳健估计地面平面；
  2. 找出"显著近于地面"的连通区域（障碍/目标候选）；
  3. 用**物理尺寸**（宽/高/宽高比，单位米）做类别判据 —— 物理尺寸具有尺度不变性，
     同一目标在 2 m 与 15 m 处的像素尺寸相差数倍，但物理尺寸不变。

适用场景：没有训练权重时保证"检测 → 测距 → 误差分析"整条链路可跑通；
也可作为与 YOLO 对照的传统方法基线。
"""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

from core.config import settings
from core.exceptions import ModelError
from core.logging_setup import get_logger

from ..base import BasePlugin, Prediction
from ..ground import closer_than_ground_mask, estimate_ground_plane
from ..registry import register_plugin

log = get_logger("algorithms.depth_detector")


def _range_cost(value: float, low: float, high: float) -> float:
    """值落在 [low, high] 内为 0，超出按相对距离计代价。"""
    if low <= value <= high:
        return 0.0
    span = max(high - low, 1e-6)
    if value < low:
        return (low - value) / span
    return (value - high) / span


class DepthObstacleDetector(BasePlugin):
    task = "detection"
    name = "depth_obstacle"

    #: 基于物理尺寸的类别判据：类别 -> (宽 m 范围, 高 m 范围, 宽高比范围)
    PHYSICAL_RULES: list[tuple[str, tuple[float, float], tuple[float, float], tuple[float, float]]] = [
        ("斑马线", (1.80, 4.50), (0.01, 0.30), (5.0, 60.0)),
        ("减速带", (1.80, 4.50), (0.03, 0.40), (4.0, 60.0)),
        ("车辆", (1.30, 3.00), (1.00, 2.60), (0.60, 2.20)),
        ("行人", (0.30, 0.95), (1.20, 2.10), (0.20, 0.75)),
        ("路标", (0.25, 0.80), (0.45, 1.20), (0.30, 1.10)),
        ("交通标志", (0.35, 1.05), (0.45, 1.25), (0.45, 1.70)),
        ("井盖", (0.45, 1.05), (0.02, 0.35), (1.30, 4.50)),
        ("大理石球", (0.25, 0.75), (0.22, 0.75), (0.60, 1.70)),
        ("废瓶子", (0.04, 0.28), (0.10, 0.50), (0.20, 1.30)),
        ("落叶", (0.15, 0.70), (0.01, 0.25), (1.00, 5.00)),
        ("废纸", (0.10, 0.65), (0.01, 0.25), (1.00, 5.00)),
    ]
    FLAT_CLASSES = {"斑马线", "减速带", "井盖", "落叶", "废纸"}

    def __init__(self, ratio: float = 0.85, min_area_px: int = 120, max_objects: int = 30,
                 class_names: list[str] | None = None, **config: Any) -> None:
        super().__init__(**config)
        self.ratio = float(ratio)
        self.min_area_px = int(min_area_px)
        self.max_objects = int(max_objects)
        self.class_names = list(class_names or settings.detection_class_names())

    def describe(self) -> dict:
        info = super().describe()
        info.update({
            "arch": "depth_obstacle_baseline",
            "framework": "opencv",
            "label": "深度基线检测器（无需权重）",
            "input_size": "原生分辨率",
            "class_names": self.class_names,
            "params": {"ratio": self.ratio, "min_area_px": self.min_area_px,
                       "classify": "物理尺寸判据（尺度不变）"},
            "limits": "分类精度低于训练模型；用于流程兜底与传统方法对照",
        })
        return info

    def load(self, weights_uri: str | None = None, device: str = "auto", **kwargs: Any) -> None:
        self.device = "cpu"
        self.loaded = True
        log.info("深度基线检测器已就绪（无需权重）")

    # ------------------------------------------------------------------
    def _classify(self, width_m: float, height_m: float, depth_m: float) -> str:
        aspect = width_m / max(height_m, 1e-3)
        best_name, best_cost = None, float("inf")
        for name, (w_lo, w_hi), (h_lo, h_hi), (a_lo, a_hi) in self.PHYSICAL_RULES:
            cost = (_range_cost(width_m, w_lo, w_hi)
                    + _range_cost(height_m, h_lo, h_hi)
                    + _range_cost(aspect, a_lo, a_hi))
            if name in self.FLAT_CLASSES and depth_m > 12.0:
                cost += 0.6      # 远处不应出现贴地小目标
            if cost < best_cost:
                best_name, best_cost = name, cost
        if best_name is not None and best_cost < 1.2:
            return best_name
        if width_m > 1.2:
            return "车辆"
        if height_m > 0.9:
            return "行人"
        if max(width_m, height_m) < 0.28:
            return "废瓶子"
        return "废纸" if height_m < 0.16 else "大理石球"

    def predict(self, image: np.ndarray, **params: Any) -> Prediction:
        depth_m = params.get("depth_m")
        if depth_m is None:
            return Prediction.empty_detection("深度基线检测器需要 depth_m 参数")
        depth_m = np.asarray(depth_m, dtype=np.float32)
        if depth_m.shape[:2] != image.shape[:2]:
            depth_m = cv2.resize(depth_m, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        intr = params.get("intrinsics")
        if intr is None:
            from core.geometry import estimate_intrinsics_from_fov

            intr = estimate_intrinsics_from_fov(image.shape[1], image.shape[0])

        conf = float(params.get("conf", 0.35))
        min_m = float(settings.get("ranging.min_range_m", 0.3))
        max_m = float(settings.get("ranging.max_range_m", 20.0))
        keep = params.get("classes") or None

        t0 = time.perf_counter()
        plane = estimate_ground_plane(depth_m, min_m=min_m, max_m=max_m)
        raw = closer_than_ground_mask(depth_m, plane, ratio=self.ratio, min_m=min_m, max_m=max_m)

        clean = cv2.morphologyEx(raw * 255, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes, scores, classes, labels = [], [], [], []
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[: self.max_objects]:
            if cv2.contourArea(contour) < self.min_area_px:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            patch = depth_m[y:y + h, x:x + w]
            valid = patch[(patch >= min_m) & (patch <= max_m)]
            if valid.size == 0:
                continue
            z = float(np.median(valid))
            width_m = float(w) * z / max(intr.fx, 1e-6)
            height_m = float(h) * z / max(intr.fy, 1e-6)
            name = self._classify(width_m, height_m, z)
            if keep and name not in set(str(c) for c in keep):
                continue
            score = float(np.clip(0.92 - z / max_m * 0.40, 0.30, 0.96))
            if score < conf:
                continue
            boxes.append([float(x), float(y), float(x + w), float(y + h)])
            scores.append(score)
            classes.append(self.class_names.index(name) if name in self.class_names else 0)
            labels.append(name)

        order = np.argsort(-np.asarray(scores)) if scores else []
        return Prediction(
            task="detection",
            boxes=[boxes[i] for i in order],
            scores=[scores[i] for i in order],
            classes=[classes[i] for i in order],
            labels=[labels[i] for i in order],
            raw_labels=[labels[i] for i in order],
            sources=["depth_baseline"] * len(order),
            infer_ms=(time.perf_counter() - t0) * 1000.0,
            arch="depth_obstacle_baseline",
            weight_mode="none",
            ground_plane=plane.to_dict(),
        )

    def export(self, fmt: str = "onnx", out_dir: str | None = None) -> str:
        raise ModelError("深度基线检测器无需导出权重。")


register_plugin("detection", "depth_obstacle", DepthObstacleDetector,
                meta={"label": "深度基线检测器（无需权重）", "framework": "opencv"})
