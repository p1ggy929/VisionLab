"""在真实校园道路测试图上跑四个分割模型，输出并排对比图与 IoU 指标。

这是原项目 B `compare_models_test.py` 的平台化版本（批量评测任务的原型实现）。
用法：python tools/check_seg_compare.py [数据根目录] [权重目录]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from algorithms import registry  # noqa: E402
from core import media  # noqa: E402
from core.config import settings  # noqa: E402
from core.logging_setup import get_cli_logger, setup_logging  # noqa: E402

log = get_cli_logger("tools.check_seg_compare")

DEFAULT_DATA = Path(r"E:\2Project\2026dachuang\Unet")
PLAN = [
    ("unet", "Unet_1000_best.pth", "UNet"),
    ("resunet", "resunet_1000_best.pth", "ResUNet"),
    ("att_resunet", "Atteresunet_1000_best.pth", "Attention-ResUNet"),
    ("unetv2", "UNetV2_800_best.pth", "UNetV2 (PVTv2)"),
]


def metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    p = (pred_mask > 0).astype(np.uint8).ravel()
    g = (gt_mask > 0).astype(np.uint8).ravel()
    tp = int(np.logical_and(p == 1, g == 1).sum())
    fp = int(np.logical_and(p == 1, g == 0).sum())
    fn = int(np.logical_and(p == 0, g == 1).sum())
    tn = int(np.logical_and(p == 0, g == 0).sum())
    eps = 1e-7
    return {
        "IoU": round(tp / (tp + fp + fn + eps), 4),
        "Dice": round(2 * tp / (2 * tp + fp + fn + eps), 4),
        "Precision": round(tp / (tp + fp + eps), 4),
        "Recall": round(tp / (tp + fn + eps), 4),
        "Accuracy": round((tp + tn) / (tp + tn + fp + fn + eps), 4),
    }


def load_gt(mask_path: Path) -> np.ndarray | None:
    raw = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    if raw.ndim == 3:
        raw = raw[:, :, 0]
    return (raw > 0).astype(np.uint8)


def main() -> int:
    data_root = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATA
    weights_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else data_root / "models"
    setup_logging()
    registry.load_builtin()

    out_dir = settings.storage_dir("results") / "seg_compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    test_dir = data_root / "dataset" / "test" / "images"
    masks_dir = data_root / "dataset" / "test" / "masks"
    image_path = test_dir / "Campus_road_196.jpg"
    if not image_path.exists():
        candidates = sorted(test_dir.glob("*.jpg")) if test_dir.exists() else []
        if not candidates:
            log.info(f"找不到测试图像，请检查路径：{test_dir}")
            return 1
        image_path = candidates[0]

    image = media.read_image(image_path)
    if image is None:
        log.info(f"图像读取失败：{image_path}")
        return 1
    gt = load_gt(masks_dir / (image_path.stem + ".png"))
    log.info(f"测试图像: {image_path.name}  {image.shape[1]}x{image.shape[0]}")
    if gt is not None:
        log.info(f"真值掩膜前景占比: {gt.mean():.4f}  shape={gt.shape}")

    panels = [image.copy()]
    labels = ["原图"]
    rows = []
    log.info("=" * 92)
    for arch, weight_name, label in PLAN:
        weight = weights_dir / weight_name
        plugin = registry.create("segmentation", arch)
        plugin.load(str(weight) if weight.exists() else None)
        pred = plugin.predict(image, threshold=0.30)
        mask = pred["mask"]
        ratio = float(mask.mean())
        row = {"模型": label, "掩膜占比": round(ratio, 4), "阈值语义": pred.get("output_activation")}
        if gt is not None:
            row.update(metrics(mask, gt))
        rows.append(row)
        vis = media.draw_mask(image.copy(), mask, alpha=0.35)
        vis = media.draw_panel(vis, [f"掩膜占比 {ratio:.3f}"], "top-right", label)
        panels.append(vis)
        labels.append(label)
        log.info("  ".join(f"{k}={v}" for k, v in row.items()))

    h, w = image.shape[:2]
    scale = 620 / w
    tw, th = int(w * scale), int(h * scale)
    tiles = [cv2.resize(p, (tw, th)) for p in panels]
    while len(tiles) < 6:
        tiles.append(np.zeros((th, tw, 3), dtype=np.uint8))
    grid = np.vstack([np.hstack(tiles[0:3]), np.hstack(tiles[3:6])])
    for index, label in enumerate(labels):
        x = (index % 3) * tw + 12
        y = (index // 3) * th + 12
        media.draw_text_items(grid, [{"text": label, "xy": (x, y), "color": (255, 255, 255),
                                      "size": 22, "bg": (40, 40, 40), "padding": 6}])
    out_path = out_dir / "seg_model_comparison.jpg"
    media.write_image(out_path, grid, quality=92)
    log.info("=" * 92)
    log.info("对比图已保存:", out_path)

    header = ["模型", "掩膜占比"] + (["IoU", "Dice", "Precision", "Recall", "Accuracy"]
                                    if gt is not None else [])
    log.info("\n| " + " | ".join(header) + " |")
    log.info("|" + "---|" * len(header))
    for row in rows:
        log.info("| " + " | ".join(str(row.get(k, "-")) for k in header) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
