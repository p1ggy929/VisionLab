"""传统道路分割器（无权重兜底）。

真实可用的几何 + 颜色方法：
  1. 从深度图估计地面平面，取"贴合地面"的像素作为候选；
  2. 在图像下部中央取道路种子区域，做 Lab 颜色相似度筛选；
  3. 形态学清理 + 保留包含底部中央的最大连通域 + 空洞填充。

用途：无分割权重时保证道路边界测距链路可跑通；也可作为深度学习方法对照基线。
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
from ..ground import GroundTolerance, estimate_ground_plane, on_ground_mask
from ..registry import register_plugin

log = get_logger("algorithms.classical_seg")


class ClassicalRoadSegmenter(BasePlugin):
    task = "segmentation"
    name = "classical_road"

    def __init__(self, color_tol: float = 46.0, ground_tol_abs: float = 0.22,
                 ground_tol_rel: float = 0.12, **config: Any) -> None:
        super().__init__(**config)
        self.color_tol = float(color_tol)
        self.ground_tol_abs = float(ground_tol_abs)
        self.ground_tol_rel = float(ground_tol_rel)

    def describe(self) -> dict:
        info = super().describe()
        info.update({
            "arch": "classical_road",
            "label": "传统道路分割器（地面平面 + 颜色，无需权重）",
            "framework": "opencv",
            "input_size": "原生分辨率",
            "limits": "依赖深度质量；无深度时退化为纯颜色分割，弱纹理路面效果下降",
        })
        return info

    def load(self, weights_uri: str | None = None, device: str = "auto", **kwargs: Any) -> None:
        self.device = "cpu"
        self.loaded = True
        log.info("传统道路分割器已就绪（无需权重）")

    def predict(self, image: np.ndarray, **params: Any) -> Prediction:
        h, w = image.shape[:2]
        t0 = time.perf_counter()
        depth_m = params.get("depth_m")
        min_m = float(settings.get("ranging.min_range_m", 0.3))
        max_m = float(settings.get("ranging.max_range_m", 20.0))

        if depth_m is not None:
            depth_m = np.asarray(depth_m, dtype=np.float32)
            if depth_m.shape[:2] != (h, w):
                depth_m = cv2.resize(depth_m, (w, h), interpolation=cv2.INTER_NEAREST)
            plane = estimate_ground_plane(depth_m, min_m=min_m, max_m=max_m)
            tolerance = GroundTolerance(tol_abs=self.ground_tol_abs,
                                        tol_rel=self.ground_tol_rel,
                                        min_m=min_m, max_m=max_m)
            ground = on_ground_mask(depth_m, plane, tolerance)
        else:
            plane = None
            ground = np.ones((h, w), dtype=np.uint8)

        # 颜色相似度：以底部中央条纹为道路种子
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        y0, y1 = int(h * 0.82), int(h * 0.97)
        x0, x1 = int(w * 0.40), int(w * 0.60)
        seed = lab[y0:y1, x0:x1].reshape(-1, 3)
        ref = np.median(seed, axis=0)
        dist = np.linalg.norm(lab - ref.reshape(1, 1, 3), axis=2)
        similar = (dist <= self.color_tol).astype(np.uint8)

        mask = (ground & similar).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))

        mask = self._keep_main_component(mask, h, w)
        mask = self._fill_holes(mask)
        mask = self._keep_lower_region(mask, h)

        prob = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 3.0)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return Prediction(
            task="segmentation", prob=prob, mask=mask, contours=list(contours),
            infer_ms=(time.perf_counter() - t0) * 1000.0,
            arch="classical_road", threshold=0.5,
            ground_plane=plane.to_dict() if plane else None,
        )

    @staticmethod
    def _keep_main_component(mask: np.ndarray, h: int, w: int) -> np.ndarray:
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num <= 1:
            return mask
        seed_label = labels[int(h * 0.95), int(w * 0.5)]
        if seed_label != 0:
            best = seed_label
        else:
            best, best_score = 0, -1.0
            for idx in range(1, num):
                area = stats[idx, cv2.CC_STAT_AREA]
                cy = centroids[idx][1]
                score = area * (1.0 + cy / max(h, 1))
                if score > best_score:
                    best, best_score = idx, score
        if best == 0:
            return mask
        return (labels == best).astype(np.uint8)

    @staticmethod
    def _fill_holes(mask: np.ndarray) -> np.ndarray:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
        return filled

    @staticmethod
    def _keep_lower_region(mask: np.ndarray, h: int) -> np.ndarray:
        """去掉上半部的孤立碎块，保留与下部连通的主体。"""
        top = mask.copy()
        top[int(h * 0.35):, :] = 0
        if top.sum() == 0:
            return mask
        num, labels = cv2.connectedComponents(top, connectivity=8)
        if num <= 1:
            return mask
        result = mask.copy()
        for idx in range(1, num):
            if (labels == idx).sum() > 0:
                result[labels == idx] = 1
        return result

    def export(self, fmt: str = "onnx", out_dir: str | None = None) -> str:
        raise ModelError("传统道路分割器无需导出权重。")


register_plugin("segmentation", "classical_road", ClassicalRoadSegmenter,
                meta={"label": "传统道路分割器（无需权重）", "framework": "opencv"})
