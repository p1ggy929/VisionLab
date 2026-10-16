"""改进 U-Net 道路语义分割适配器（沿用项目 B 的四个结构与推理流程）。

支持结构：unet / resunet / att_resunet / unetv2
权重格式兼容项目 B 的两种约定：
  - 裸 state_dict
  - {"model_state_dict": state_dict}
推理预处理与项目 B `Main.py` 完全一致：resize 到 1024x512 → /255 → CHW → 前向 → 阈值 0.30。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from core.config import settings
from core.exceptions import ModelError
from core.logging_setup import get_logger
from core.storage import to_path
from ..base import BasePlugin, Prediction
from ..registry import register_plugin

log = get_logger("algorithms.segmenter")

ARCH_LABELS = {
    "unet": "UNet",
    "resunet": "ResUNet",
    "att_resunet": "Attention-ResUNet",
    "unetv2": "UNetV2 (PVTv2)",
}


class UNetSegmenterAdapter(BasePlugin):
    task = "segmentation"
    name = "unet"

    def __init__(self, arch: str = "unet", n_classes: int = 1, **config: Any) -> None:
        super().__init__(**config)
        self.arch = str(arch).lower()
        self.n_classes = int(n_classes)
        self._model = None
        self._input_w = int(settings.get("segmentation.input_width", 1024))
        self._input_h = int(settings.get("segmentation.input_height", 512))

    # ------------------------------------------------------------ 加载
    def load(self, weights_uri: str | None = None, device: str = "auto", **kwargs: Any) -> None:
        import torch

        from .models import build_segmentation_model

        self.device = self._resolve_device(device)
        self.weights_uri = weights_uri
        try:
            model = build_segmentation_model(
                self.arch,
                n_classes=self.n_classes,
                n_channels=3,
                pretrained_path=kwargs.get("pretrained_path"),
            )
        except ModelError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ModelError(f"构建 {self.arch} 失败: {exc}") from exc

        if weights_uri:
            path = to_path(weights_uri)
            if path is None or not Path(path).exists():
                raise ModelError(f"权重文件不存在: {weights_uri}")
            try:
                state = torch.load(str(path), map_location="cpu", weights_only=False)
            except Exception as exc:  # noqa: BLE001
                raise ModelError(f"权重加载失败 {path.name}: {exc}") from exc
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            try:
                model.load_state_dict(state)
            except Exception as exc:  # noqa: BLE001
                raise ModelError(
                    f"{self.arch} 权重与结构不匹配（常见原因：选了 UNet 结构却加载 ResUNet 权重）: {exc}"
                ) from exc

        model.to(self.device)
        model.eval()
        self._model = model
        self.loaded = True
        log.info("分割模型已加载: %s on %s (%s)", self.arch, self.device, weights_uri or "no-weights")

    def describe(self) -> dict:
        info = super().describe()
        info.update({
            "arch": self.arch,
            "label": ARCH_LABELS.get(self.arch, self.arch),
            "framework": "pytorch",
            "input_size": f"{self._input_w}x{self._input_h}",
            "num_classes": self.n_classes,
            "threshold_default": float(settings.get("segmentation.default_threshold", 0.30)),
            "device": self.device,
        })
        return info

    # ------------------------------------------------------------ 推理
    def _forward(self, tensor):
        import torch

        with torch.no_grad():
            out = self._model(tensor)
            if isinstance(out, (list, tuple)):
                out = out[0]
        return out

    @staticmethod
    def _to_probability(raw: np.ndarray) -> tuple[np.ndarray, str]:
        """把模型输出统一为 [0,1] 概率。

        原项目 B 的四个结构输出语义并不一致：UNet / ResUNet / AttResUNet 的 forward
        末尾带 sigmoid，而 UNetV2 的 seg_outs 输出的是 **logits**。
        原 `Main.py` 直接对 logits 用 0.30 阈值，导致阈值在不同结构上含义不同。
        平台统一：输出超出 [0,1] 即视为 logits，自动套用 sigmoid，使阈值口径一致。
        """
        arr = np.asarray(raw, dtype=np.float32)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return np.zeros_like(arr), "empty"
        lo, hi = float(finite.min()), float(finite.max())
        if lo < -1e-6 or hi > 1.0 + 1e-6:
            prob = 1.0 / (1.0 + np.exp(-np.clip(arr, -60.0, 60.0)))
            return prob.astype(np.float32), "sigmoid(logits)"
        return arr, "identity(probability)"

    def predict(self, image: np.ndarray, **params: Any) -> Prediction:
        import torch

        if self._model is None:
            raise ModelError("分割模型尚未加载。")

        threshold = float(params.get("threshold", settings.get("segmentation.default_threshold", 0.30)))
        open_ks = int(params.get("open_ks", settings.get("segmentation.open_ks", 3)))
        close_ks = int(params.get("close_ks", settings.get("segmentation.close_ks", 3)))
        min_area = int(params.get("min_area_px", settings.get("segmentation.min_area_px", 200)))
        in_w = int(params.get("input_width", self._input_w))
        in_h = int(params.get("input_height", self._input_h))

        h0, w0 = image.shape[:2]
        t0 = time.perf_counter()
        resized = cv2.resize(image, (in_w, in_h))
        tensor = torch.from_numpy(resized.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device)
        out = self._forward(tensor)
        prob_small, activation = self._to_probability(out[0, 0].detach().cpu().numpy())
        infer_ms = (time.perf_counter() - t0) * 1000.0

        prob = cv2.resize(prob_small, (w0, h0), interpolation=cv2.INTER_LINEAR)
        mask = (prob > threshold).astype(np.uint8)

        if open_ks > 1:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((open_ks, open_ks), np.uint8))
        if close_ks > 1:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close_ks, close_ks), np.uint8))
        mask = self._filter_small(mask, min_area)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return Prediction(
            task="segmentation", prob=prob, mask=mask, contours=list(contours),
            infer_ms=infer_ms, arch=self.arch, threshold=threshold,
            output_activation=activation,
        )

    @staticmethod
    def _filter_small(mask: np.ndarray, min_area: int) -> np.ndarray:
        if min_area <= 0:
            return mask
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        out = np.zeros_like(mask)
        for idx in range(1, num):
            if stats[idx, cv2.CC_STAT_AREA] >= min_area:
                out[labels == idx] = 1
        return out

    def export(self, fmt: str = "onnx", out_dir: str | None = None) -> str:
        import torch

        if self._model is None:
            raise ModelError("分割模型尚未加载，无法导出。")
        dummy = torch.randn(1, 3, self._input_h, self._input_w, device=self.device)
        target = Path(out_dir) if out_dir else settings.storage_dir("models") / "onnx"
        target.mkdir(parents=True, exist_ok=True)
        out_path = target / f"{self.arch}.onnx"
        torch.onnx.export(self._model, dummy, str(out_path), opset_version=12,
                          input_names=["input"], output_names=["output"], dynamo=False)
        return str(out_path)


def _factory(arch: str):
    def creator(**kwargs: Any) -> UNetSegmenterAdapter:
        kwargs.pop("arch", None)
        return UNetSegmenterAdapter(arch=arch, **kwargs)

    return creator


for _arch in ARCH_LABELS:
    register_plugin("segmentation", _arch, _factory(_arch),
                    meta={"label": ARCH_LABELS[_arch], "framework": "pytorch"})
