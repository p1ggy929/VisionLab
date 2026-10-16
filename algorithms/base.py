"""算法插件协议：检测与分割对上层完全同构的契约。

新增算法只需实现本文件的 AlgorithmPlugin（6 个方法）+ 在 registry 注册一份配置，
不需要改动任何业务代码。
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


class Prediction(dict):
    """统一推理输出容器。

    detection:
        {
          "task": "detection",
          "boxes":   [[x1, y1, x2, y2], ...],
          "scores":  [float, ...],
          "classes": [int, ...],
          "labels":  [str, ...],
          "infer_ms": float,
        }
    segmentation:
        {
          "task": "segmentation",
          "prob":   np.ndarray  HxW float32 in [0,1]  (可为 None)
          "mask":   np.ndarray  HxW uint8 {0,1}
          "contours": list[np.ndarray]
          "infer_ms": float,
        }
    """

    @property
    def task(self) -> str:
        return str(self.get("task", ""))

    @classmethod
    def empty_detection(cls, reason: str = "") -> "Prediction":
        return cls(task="detection", boxes=[], scores=[], classes=[], labels=[],
                   infer_ms=0.0, reason=reason)

    @classmethod
    def empty_segmentation(cls, shape: tuple[int, int] | None = None, reason: str = "") -> "Prediction":
        prob = np.zeros(shape, dtype=np.float32) if shape else None
        mask = np.zeros(shape, dtype=np.uint8) if shape else None
        return cls(task="segmentation", prob=prob, mask=mask, contours=[],
                   infer_ms=0.0, reason=reason)

    def detections(self) -> list[dict]:
        """把检测结果展开为逐目标字典列表。"""
        if self.task != "detection":
            return []
        out = []
        for i, box in enumerate(self.get("boxes") or []):
            out.append({
                "bbox": [float(v) for v in box[:4]],
                "score": float((self.get("scores") or [0.0])[i]) if i < len(self.get("scores") or []) else 0.0,
                "class_id": int((self.get("classes") or [0])[i]) if i < len(self.get("classes") or []) else 0,
                "class_name": str((self.get("labels") or ["object"])[i]) if i < len(self.get("labels") or []) else "object",
            })
        return out


@runtime_checkable
class AlgorithmPlugin(Protocol):
    """算法插件契约。"""

    task: str          # 'detection' | 'segmentation'
    name: str          # 插件标识，如 'yolo'

    def describe(self) -> dict:
        """模型卡片：arch / framework / input_size / classes / weights / metrics / limits。"""
        ...

    def load(self, weights_uri: str | None, device: str = "auto", **kwargs: Any) -> None:
        """加载权重。weights_uri 为相对存储根的 URI，None 表示使用内置/官方权重。"""
        ...

    def predict(self, image: np.ndarray, **params: Any) -> Prediction:
        ...

    def predict_batch(self, images: list[np.ndarray], **params: Any) -> list[Prediction]:
        ...

    def export(self, fmt: str = "onnx", out_dir: str | None = None) -> str:
        ...

    def release(self) -> None:
        ...


class BasePlugin:
    """插件基类：提供默认的空实现，子类按需覆写。"""

    task = "base"
    name = "base"

    def __init__(self, **config: Any) -> None:
        self.config = dict(config)
        self.loaded = False
        self.device = "cpu"
        self.weights_uri: str | None = None

    def describe(self) -> dict:
        return {"task": self.task, "plugin": self.name, "loaded": self.loaded,
                "device": self.device, "config": self.config}

    def load(self, weights_uri: str | None = None, device: str = "auto", **kwargs: Any) -> None:
        self.weights_uri = weights_uri
        self.device = self._resolve_device(device)
        self.loaded = True

    def predict(self, image: np.ndarray, **params: Any) -> Prediction:
        raise NotImplementedError

    def predict_batch(self, images: list[np.ndarray], **params: Any) -> list[Prediction]:
        return [self.predict(img, **params) for img in images]

    def export(self, fmt: str = "onnx", out_dir: str | None = None) -> str:
        raise NotImplementedError(f"{self.name} 暂不支持导出为 {fmt}")

    def release(self) -> None:
        self.loaded = False

    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_device(device: str = "auto") -> str:
        if device and device != "auto":
            return device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except Exception:  # noqa: BLE001
            pass
        return "cpu"
