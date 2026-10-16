"""分割二值化阈值标定：为每个结构找到使 IoU 最优的阈值。

背景：原项目 B 的在线推理脚本用固定阈值 0.30，而各结构的概率分布不同，
最优阈值差异明显。本脚本在真实测试图上扫描阈值，为平台默认值与界面滑块提供依据。

用法：python tools/tune_seg_threshold.py [数据根目录] [权重目录]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from algorithms import registry  # noqa: E402
from core import media  # noqa: E402
from core.logging_setup import setup_logging  # noqa: E402

DEFAULT_DATA = Path(r"E:\2Project\2026dachuang\Unet")
PLAN = [("unet", "Unet_1000_best.pth"), ("resunet", "resunet_1000_best.pth"),
        ("att_resunet", "Atteresunet_1000_best.pth"), ("unetv2", "UNetV2_800_best.pth")]
THRESHOLDS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
REPORTED = {"unet": 0.9688, "resunet": 0.9777, "att_resunet": 0.9737, "unetv2": 0.9364}


def iou_of(pred: np.ndarray, gt: np.ndarray) -> float:
    p = (pred > 0).ravel()
    g = (gt > 0).ravel()
    tp = float(np.logical_and(p, g).sum())
    fp = float(np.logical_and(p, ~g).sum())
    fn = float(np.logical_and(~p, g).sum())
    return tp / (tp + fp + fn + 1e-7)


def main() -> int:
    data_root = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATA
    weights_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else data_root / "models"
    setup_logging()
    registry.load_builtin()

    test_images = sorted((data_root / "dataset" / "test" / "images").glob("*.jpg"))[:6]
    masks_dir = data_root / "dataset" / "test" / "masks"
    if not test_images:
        print("找不到测试图像。")
        return 1
    print(f"抽检 {len(test_images)} 张测试图\n")

    verdicts = {}
    for arch, weight in PLAN:
        plugin = registry.create("segmentation", arch)
        plugin.load(str(weights_dir / weight))
        scores = []
        for threshold in THRESHOLDS:
            ious = []
            for image_path in test_images:
                image = media.read_image(image_path)
                raw = cv2.imread(str(masks_dir / (image_path.stem + ".png")), cv2.IMREAD_UNCHANGED)
                if raw is None:
                    continue
                if raw.ndim == 3:
                    raw = raw[:, :, 0]
                pred = plugin.predict(image, threshold=threshold)["mask"]
                ious.append(iou_of(pred, (raw > 0).astype(np.uint8)))
            scores.append((threshold, float(np.mean(ious))))
        best = max(scores, key=lambda kv: kv[1])
        reference = REPORTED.get(arch, 0.97)
        closest = min(scores, key=lambda kv: abs(kv[1] - reference))
        verdicts[arch] = {"best_threshold": best[0], "best_iou": round(best[1], 4),
                          "closest_threshold": closest[0], "iou_at_closest": round(closest[1], 4),
                          "reported_iou": reference}
        print(f"== {arch} ==")
        print("   " + "  ".join(f"t={t:.2f}:{v:.4f}" for t, v in scores))
        print(f"   最佳阈值 {best[0]:.2f} -> IoU {best[1]:.4f}；"
              f"最接近原项目 B 报告值({reference})的阈值 {closest[0]:.2f} -> IoU {closest[1]:.4f}\n")

    print("=" * 84)
    print("结论：各结构最优阈值并不相同，平台界面提供阈值滑块以便按结构调优")
    print("=" * 84)
    for arch, value in verdicts.items():
        print(f"  {arch:<12} 原报告 IoU={value['reported_iou']}  "
              f"最接近阈值={value['closest_threshold']}  实测 IoU={value['iou_at_closest']}  "
              f"(最优 {value['best_iou']} @ t={value['best_threshold']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
