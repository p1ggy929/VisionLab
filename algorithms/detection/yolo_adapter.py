"""YOLO 目标检测适配器。

沿用项目 A `backend/detector.py` 的核心能力：
  - 五个系列 v8/v9/v10/v11/v12 × n/s/m/l（v9 另含 t/c），共 20 个配置；
  - 三种权重模式 pretrained / trained / dual；
  - dual 模式下两路结果按 IoU 融合去重（阈值 0.55，项目 A 的验证值）；
  - 模型 LRU 缓存、类别过滤、单类别检测模式。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.config import settings
from core.exceptions import ModelError
from core.logging_setup import get_logger
from ..base import BasePlugin, Prediction
from ..registry import register_plugin

log = get_logger("algorithms.yolo")


def _iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def merge_detections(items: list[dict], iou_threshold: float = 0.55) -> list[dict]:
    """同类目标按 IoU 去重，保留置信度更高者（项目 A 的 dual 融合策略）。"""
    ordered = sorted(items, key=lambda d: float(d.get("score", 0.0)), reverse=True)
    kept: list[dict] = []
    for item in ordered:
        duplicated = False
        for exist in kept:
            if exist.get("class_id") != item.get("class_id"):
                continue
            if _iou(exist["bbox"], item["bbox"]) >= iou_threshold:
                duplicated = True
                break
        if not duplicated:
            kept.append(item)
    return kept


class YoloDetectorAdapter(BasePlugin):
    """Ultralytics YOLO 检测插件。"""

    task = "detection"
    name = "yolo"

    def __init__(self, version: str = "v8", size: str = "n", weight_mode: str | None = None,
                 class_names: list[str] | None = None, **config: Any) -> None:
        super().__init__(**config)
        self.version = str(version).lower().lstrip("yolo")
        self.size = str(size).lower()
        self.weight_mode = str(weight_mode or settings.get("detection.weight_mode", "dual")).lower()
        self.class_names = list(class_names or settings.detection_class_names())
        self._models: dict[str, Any] = {}
        self._names: dict[int, str] = {}
        self._source_map: dict[str, str] = {}

    # ------------------------------------------------------------ 命名
    @property
    def arch(self) -> str:
        """平台规范结构名。

        Ultralytics 的命名规则：v8/v9/v10 带 v（yolov8n），v11/v12 不带 v（yolo11n）。
        注意不能写成 "yolo" + size，否则 v11/v12 会退化成 "yolon"。
        """
        prefix = f"yolo{self.version}" if self.version in ("11", "12") else f"yolov{self.version}"
        return f"{prefix}{self.size}"

    @property
    def model_key(self) -> str:
        return self.arch

    @staticmethod
    def available_configs() -> list[dict]:
        families = settings.get("detection.families", {}) or {}
        out = []
        for version, info in families.items():
            for size in info.get("sizes", []):
                canonical = str(version).lstrip("v")
                prefix = f"yolo{canonical}" if canonical in ("11", "12") else f"yolov{canonical}"
                out.append({
                    "version": canonical,
                    "size": size,
                    "key": f"{prefix}{size}",
                    "arch_prefix": info.get("arch_prefix"),
                })
        return out

    # ------------------------------------------------------------ 权重解析
    def _arch_variants(self, arch: str) -> list[str]:
        """返回权重文件名中可能出现该结构的写法。

        原项目 A 的命名存在两种惯例：
          · 官方预训练权重：v11/v12 不带 v（`yolo11n.pt`）
          · 自训练产物    ：统一带 v（`yolov11n_cloud.pt`）
        两种都要能命中，否则 20 个配置里有 8 个会找不到权重。
        """
        variants = [arch]
        if arch.startswith("yolo") and not arch.startswith("yolov"):
            variants.append("yolov" + arch[4:])
        elif arch.startswith("yolov"):
            variants.append("yolo" + arch[5:])
        return variants

    def _candidate_weight(self, arch: str, origin: str) -> Path | None:
        """按命名约定在平台目录与原项目权重目录中查找权重。"""
        platform = settings.storage_dir("models") / ("pretrained" if origin == "pretrained"
                                                     else "trained")
        external = settings.external_source(
            "detection_pretrained" if origin == "pretrained" else "detection_trained")
        search: list[Path] = [platform]
        if external:
            search.append(external)

        for token in self._arch_variants(arch):
            names = [f"{token}.pt", f"{token}_best.pt", f"{token}_last.pt"]
            for base in search:
                for name in names:
                    path = base / name
                    if path.exists():
                        return path
        # 再退一步：模糊匹配（覆盖 `yolov11n_cloud.pt` 这类带后缀的自训练产物）
        for token in self._arch_variants(arch):
            for base in search:
                for path in sorted(base.glob(f"*{token}*.pt")):
                    return path
        return None

    def _resolve_weight(self, origin: str) -> Path | None:
        """origin: pretrained | trained，找不到返回 None。"""
        if self.weights_uri and origin == "explicit":
            path = Path(self.weights_uri)
            return path if path.exists() else None
        return self._candidate_weight(self.arch, origin)

    # ------------------------------------------------------------ 加载
    def load(self, weights_uri: str | None = None, device: str = "auto", **kwargs: Any) -> None:
        from core.storage import to_path

        self.device = self._resolve_device(device)
        self.weights_uri = weights_uri
        self._models.clear()
        self._names.clear()

        explicit = to_path(weights_uri) if weights_uri else None
        wanted: list[tuple[str, Path]] = []
        if explicit and explicit.exists():
            wanted.append(("explicit", explicit))
        else:
            if self.weight_mode in ("pretrained", "dual"):
                path = self._resolve_weight("pretrained")
                if path:
                    wanted.append(("pretrained", path))
            if self.weight_mode in ("trained", "dual"):
                path = self._resolve_weight("trained")
                if path:
                    wanted.append(("trained", path))

        if not wanted:
            raise ModelError(
                f"未找到 {self.arch} 的可用权重（模式={self.weight_mode}）。"
                f"请将权重放入 {settings.storage_dir('models')}/pretrained/ 或 trained/，"
                f"或改用深度基线检测器（depth_obstacle）。"
            )

        for origin, path in wanted:
            try:
                self._models[origin] = self._build_model(path)
                self._source_map[origin] = str(path)
                log.info("YOLO 权重已加载[%s]: %s", origin, path.name)
            except Exception as exc:  # noqa: BLE001
                log.warning("加载权重失败[%s] %s: %s", origin, path, exc)

        if not self._models:
            raise ModelError(f"{self.arch} 的所有候选权重均加载失败。")

        first = next(iter(self._models.values()))
        names = getattr(first, "names", {}) or {}
        self._names = {int(k): str(v) for k, v in names.items()}
        self.loaded = True

    def _build_model(self, path: Path):
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover
            raise ModelError("未安装 ultralytics，无法使用 YOLO 检测器。") from exc
        return YOLO(str(path))

    # ------------------------------------------------------------ 描述
    def describe(self) -> dict:
        info = super().describe()
        imgsz = settings.get("detection.default_imgsz", 640)
        info.update({
            "arch": self.arch,
            "framework": "ultralytics",
            "weight_mode": self.weight_mode,
            "sources": dict(self._source_map),
            "num_classes": len(self._names) or settings.detection_num_classes(),
            "class_names": [self._names[k] for k in sorted(self._names)] or self.class_names,
            "input_size": f"{imgsz}x{imgsz}",
        })
        return info

    # ------------------------------------------------------------ 推理
    def _predict_one(self, model, image: np.ndarray, options: "PredictOptions") -> list[dict]:
        conf, iou, imgsz, max_det, source = (options.conf, options.iou, options.imgsz,
                                             options.max_det, options.source)
        results = model.predict(image, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det,
                                device=self.device, verbose=False)
        out: list[dict] = []
        for res in results:
            boxes = getattr(res, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                xyxy = box.xyxy[0].tolist()
                cls_id = int(box.cls[0].item()) if box.cls is not None else 0
                score = float(box.conf[0].item()) if box.conf is not None else 0.0
                label = self._names.get(cls_id, str(cls_id))
                out.append({
                    "bbox": [float(v) for v in xyxy],
                    "score": score,
                    "class_id": cls_id,
                    "class_name": self._translate(label),
                    "raw_label": label,
                    "source": source,
                })
        return out

    def _translate(self, raw: str) -> str:
        """把权重里的英文类名映射到平台的 11 类中文名。"""
        aliases = settings.get("detection.aliases", {}) or {}
        low = str(raw).strip().lower()
        for cn, names in aliases.items():
            candidates = names if isinstance(names, list) else [names]
            if low in [str(c).lower() for c in candidates]:
                return str(cn)
        if raw in self.class_names:
            return raw
        return raw

    def predict(self, image: np.ndarray, **params: Any) -> Prediction:
        if not self.loaded:
            raise ModelError("检测器尚未加载权重。")
        conf = float(params.get("conf", settings.get("detection.default_conf", 0.25)))
        iou = float(params.get("iou", settings.get("detection.default_iou", 0.45)))
        imgsz = int(params.get("imgsz", settings.get("detection.default_imgsz", 640)))
        max_det = int(params.get("max_det", settings.get("detection.max_det", 100)))
        keep_classes = params.get("classes") or None

        t0 = time.perf_counter()
        collected: list[dict] = []
        for source, model in self._models.items():
            try:
                options = PredictOptions(conf=conf, iou=iou, imgsz=imgsz,
                                         max_det=max_det, source=source)
                collected.extend(self._predict_one(model, image, options))
            except Exception as exc:  # noqa: BLE001
                log.warning("推理失败[%s]: %s", source, exc)

        if len(self._models) > 1:
            collected = merge_detections(collected, float(settings.get("detection.merge_iou", 0.55)))

        if keep_classes:
            allowed = set(str(c) for c in keep_classes)
            collected = [d for d in collected if d["class_name"] in allowed]

        infer_ms = (time.perf_counter() - t0) * 1000.0
        collected.sort(key=lambda d: -d["score"])
        return Prediction(
            task="detection",
            boxes=[d["bbox"] for d in collected],
            scores=[d["score"] for d in collected],
            classes=[d["class_id"] for d in collected],
            labels=[d["class_name"] for d in collected],
            raw_labels=[d.get("raw_label", "") for d in collected],
            sources=[d.get("source", "") for d in collected],
            infer_ms=infer_ms,
            arch=self.arch,
            weight_mode=self.weight_mode,
        )

    def export(self, fmt: str = "onnx", out_dir: str | None = None) -> str:
        if not self.loaded:
            raise ModelError("检测器尚未加载权重，无法导出。")
        model = next(iter(self._models.values()))
        kwargs = {"format": fmt}
        if out_dir:
            kwargs["project"] = out_dir
        path = model.export(**kwargs)
        return str(path)


@dataclass(frozen=True)
class PredictOptions:
    """单模型推理参数（阈值、输入尺寸、来源权重）。"""

    conf: float
    iou: float
    imgsz: int
    max_det: int
    source: str


@register_plugin("detection", "yolo", YoloDetectorAdapter,
                 meta={"label": "YOLO (Ultralytics)", "framework": "ultralytics",
                       "configs": len(YoloDetectorAdapter.available_configs())})
def _register() -> None:  # pragma: no cover - 仅用于触发注册
    return None
