"""维护工具：按原始划分整体重建两个原项目的数据集索引。

用法：
    python tools/rebuild_datasets.py            # 预览
    python tools/rebuild_datasets.py --apply    # 执行（会清空 dataset/image/annotation/depth_map）

导入后应得到：
    项目 A 检测数据集 : 2,694 张（train 2,301 + val 393）
    项目 B 分割数据集 : 3,120 张（train 2,496 + val 312 + test 312）
    合计              : 5,814 张
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core import db  # noqa: E402
from core.logging_setup import get_cli_logger, setup_logging  # noqa: E402
from services import dataset_service  # noqa: E402

log = get_cli_logger("tools.rebuild_datasets")

YOLO_ROOT = Path(r"E:\2Project\2026dachuang\Yolo_dis_v3源代码\datasets")
UNET_ROOT = Path(r"E:\2Project\2026dachuang\Unet")


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


def _is_split_image(path: Path, parent_name: str) -> bool:
    """判断文件是否为某个划分目录下的图片。"""
    return path.suffix.lower() in IMAGE_SUFFIXES and path.parent.name == parent_name


def _count_split_images(root: Path, parent_name: str) -> int:
    if not root.exists():
        return 0
    return sum(1 for path in root.rglob("*") if _is_split_image(path, parent_name))


def main() -> int:
    parser = argparse.ArgumentParser(description="按原始划分重建数据集索引")
    parser.add_argument("--apply", action="store_true", help="实际执行")
    parser.add_argument("--keep-existing", action="store_true", help="保留已有数据集（追加新版本）")
    args = parser.parse_args()

    setup_logging()
    db.init_db()

    counts = db.table_counts()
    log.info("当前数据：")
    for key in ("dataset", "dataset_version", "image", "annotation", "depth_map"):
        log.info(f"  {key:<18} {counts.get(key, 0)}")

    det_images = YOLO_ROOT / "images"
    seg_root = UNET_ROOT / "dataset"
    plan = [
        ("项目 A 检测数据集", det_images, YOLO_ROOT / "labels",
         _count_split_images(det_images, "train") + _count_split_images(det_images, "val")),
        ("项目 B 分割数据集", seg_root, None, _count_split_images(seg_root, "images")),
    ]
    log.info("\n将导入：")
    total = 0
    for label, source, extra, count in plan:
        total += count
        log.info(f"  {label:<18} 来源 {source}  图像 {count} 张")
    log.info(f"  合计 {total} 张（目标 5,814 张：A 2,694 + B 3,120）")

    if not args.apply:
        log.info("\n以上为预览。加 --apply 参数执行。")
        return 0

    if not args.keep_existing:
        log.info("\n清空已有数据集索引 ...")
        with db.transaction() as conn:
            for table in ("depth_map", "annotation", "image", "dataset_version", "dataset"):
                conn.execute(f"DELETE FROM {table}")
        log.info("  已清空")

    results = {}
    log.info("\n导入项目 A 检测数据集（保留 train / val 原始划分）...")
    results["detection"] = dataset_service.import_legacy_detection_full(
        str(YOLO_ROOT / "images"), str(YOLO_ROOT / "labels"),
        dataset_name="road-target-det", version_tag="v1.0")
    log.info("  ", json.dumps(results["detection"], ensure_ascii=False))

    log.info("\n导入项目 B 分割数据集（保留 train / val / test 原始划分）...")
    results["segmentation"] = dataset_service.import_legacy_segmentation_full(
        str(UNET_ROOT), dataset_name="campus-road-seg", version_tag="v1.0")
    log.info("  ", json.dumps(results["segmentation"], ensure_ascii=False))

    total_images = results["detection"]["images"] + results["segmentation"]["images"]
    log.info("\n" + "=" * 74)
    log.info(f"导入完成：共 {total_images} 张图像")
    log.info("=" * 74)
    for name, item in results.items():
        log.info(f"  {name:<14} 图像 {item['images']:>5}  划分 {item['splits']}  "
              f"校验 {item['validation']}")

    log.info("\n数据库统计：")
    for table, count in db.table_counts().items():
        if count:
            log.info(f"  {table:<22} {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
