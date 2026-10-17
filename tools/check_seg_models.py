"""验证四个分割结构能否构建并加载原项目 B 的权重。

用法：python tools/check_seg_models.py [权重目录]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from algorithms import registry  # noqa: E402
from core.logging_setup import get_cli_logger, setup_logging  # noqa: E402

log = get_cli_logger("tools.check_seg_models")

DEFAULT_WEIGHTS = Path(r"E:\2Project\2026dachuang\Unet\models")
PLAN = [
    ("unet", "Unet_1000_best.pth"),
    ("resunet", "resunet_1000_best.pth"),
    ("att_resunet", "Atteresunet_1000_best.pth"),
    ("unetv2", "UNetV2_800_best.pth"),
]


def try_load(arch: str, weight: Path | None, size=(512, 1024)) -> dict:
    plugin = registry.create("segmentation", arch)
    t0 = time.perf_counter()
    try:
        plugin.load(str(weight) if weight else None)
    except Exception as exc:  # noqa: BLE001
        return {"arch": arch, "ok": False, "stage": "load", "error": str(exc)[:220]}
    load_ms = (time.perf_counter() - t0) * 1000

    image = (np.random.rand(size[0], size[1], 3) * 255).astype("uint8")
    t1 = time.perf_counter()
    try:
        pred = plugin.predict(image, threshold=0.3)
    except Exception as exc:  # noqa: BLE001
        return {"arch": arch, "ok": False, "stage": "predict", "error": str(exc)[:220]}
    infer_ms = (time.perf_counter() - t1) * 1000

    mask = pred.get("mask")
    return {
        "arch": arch, "ok": True, "load_ms": round(load_ms, 1), "infer_ms": round(infer_ms, 1),
        "device": plugin.device,
        "mask_shape": list(mask.shape) if mask is not None else None,
        "mask_ratio": round(float(mask.mean()), 4) if mask is not None else None,
        "activation": pred.get("output_activation"),
        "params_m": round(sum(p.numel() for p in plugin.model.parameters()) / 1e6, 2)
        if plugin.model is not None else None,
        "weight": weight.name if weight else None,
    }


def main() -> int:
    weights_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_WEIGHTS
    setup_logging()
    registry.load_builtin()
    log.info(f"PyTorch {torch.__version__}  CUDA={torch.cuda.is_available()}")
    log.info(f"权重目录: {weights_dir}  (存在={weights_dir.exists()})")
    log.info("=" * 92)

    failures = 0
    for arch, filename in PLAN:
        weight = weights_dir / filename
        if not weight.exists():
            weight = None
        result = try_load(arch, weight)
        if result["ok"]:
            log.info(f"[OK  ] {arch:<12} 权重={result['weight']}  参数={result['params_m']}M  "
                  f"加载={result['load_ms']}ms  推理={result['infer_ms']}ms  "
                  f"掩膜占比={result['mask_ratio']}  阈值语义={result['activation']}")
        else:
            failures += 1
            log.info(f"[FAIL] {arch:<12} 阶段={result['stage']}  {result['error']}")
    log.info("=" * 92)
    log.info("全部通过" if not failures else f"{failures} 个结构未通过")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
