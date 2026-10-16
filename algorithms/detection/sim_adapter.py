"""仿真真值检测器（SimGroundTruth）。

用途：在与 MockBackend 合成场景配合时，由仿真器直接给出目标的真值框与真值距离，
用于**验证"测距 → 误差分析 → 校正"链路**（仿真器提供 ground truth 是常规做法，
例如 CARLA 的 ground truth 传感器）。平台在界面上明确标识为「仿真真值」，
不与真实模型检测结果混同。

同时按距离模拟一定比例的漏检（缺检率可配），使缺检分析链路也可被验证。
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from core.config import settings
from core.exceptions import ModelError
from core.logging_setup import get_logger

from ..base import BasePlugin, Prediction
from ..registry import register_plugin

log = get_logger("algorithms.sim_gt")


class SimGroundTruthDetector(BasePlugin):
    task = "detection"
    name = "sim_ground_truth"

    def __init__(self, miss_rate: float = 0.05, far_miss_boost: float = 2.0,
                 score_noise: float = 0.03, seed: int = 20260512,
                 class_names: list[str] | None = None, **config: Any) -> None:
        super().__init__(**config)
        self.miss_rate = float(miss_rate)
        self.far_miss_boost = float(far_miss_boost)
        self.score_noise = float(score_noise)
        self.class_names = list(class_names or settings.detection_class_names())
        self._rng = np.random.default_rng(seed)

    def describe(self) -> dict:
        info = super().describe()
        info.update({
            "arch": "sim_ground_truth",
            "framework": "simulator",
            "label": "仿真真值检测器（MockBackend 配套）",
            "input_size": "原生分辨率",
            "class_names": self.class_names,
            "params": {"miss_rate": self.miss_rate, "far_miss_boost": self.far_miss_boost},
            "limits": "仅在相机为合成场景（MockBackend）时可用；用于验证测距与误差分析链路，"
                      "不代表真实模型性能",
        })
        return info

    def load(self, weights_uri: str | None = None, device: str = "auto", **kwargs: Any) -> None:
        self.device = "cpu"
        self.loaded = True
        log.info("仿真真值检测器已就绪（依赖 MockBackend 提供的 ground_truth）")

    def predict(self, image: np.ndarray, **params: Any) -> Prediction:
        t0 = time.perf_counter()
        truth = params.get("ground_truth") or []
        if not truth:
            return Prediction.empty_detection(
                "无仿真真值可用：请把相机后端切换为 MockBackend（合成场景）")

        keep = params.get("classes") or None
        min_score = float(params.get("conf", 0.25))
        miss_rate = float(params.get("miss_rate", self.miss_rate))
        max_range = float(settings.get("ranging.max_range_m", 20.0))

        boxes, scores, classes, labels, gts = [], [], [], [], []
        for item in truth:
            bbox = item.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            name = str(item.get("category") or item.get("name") or "object")
            if keep and name not in set(str(c) for c in keep):
                continue
            depth = float(item.get("world_z", 0.0))
            # 越远越容易漏检（模拟真实检测器的距离退化）
            p_miss = miss_rate * (1.0 + self.far_miss_boost * min(depth / max(max_range, 1e-3), 1.0))
            if self._rng.random() < min(p_miss, 0.9):
                continue
            score = float(np.clip(0.97 - self._rng.normal(0, self.score_noise) - depth / max_range * 0.12,
                                  0.05, 0.99))
            if score < min_score:
                continue
            boxes.append([float(v) for v in bbox[:4]])
            scores.append(score)
            classes.append(self.class_names.index(name) if name in self.class_names else 0)
            labels.append(name)
            gts.append({"world_x": item.get("world_x"), "world_z": depth,
                        "size_group": item.get("size_group")})

        order = np.argsort(-np.asarray(scores)) if scores else []
        return Prediction(
            task="detection",
            boxes=[boxes[i] for i in order],
            scores=[scores[i] for i in order],
            classes=[classes[i] for i in order],
            labels=[labels[i] for i in order],
            raw_labels=[labels[i] for i in order],
            sources=["sim_ground_truth"] * len(order),
            ground_truth=[gts[i] for i in order],
            infer_ms=(time.perf_counter() - t0) * 1000.0,
            arch="sim_ground_truth",
            weight_mode="none",
        )

    def export(self, fmt: str = "onnx", out_dir: str | None = None) -> str:
        raise ModelError("仿真真值检测器无需导出权重。")


register_plugin("detection", "sim_ground_truth", SimGroundTruthDetector,
                meta={"label": "仿真真值检测器（仿真标注）", "framework": "simulator"})
