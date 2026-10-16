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
    sys.path.insert(0, str(ROOT))

from core import db  # noqa: E402
from core.logging_setup import setup_logging  # noqa: E402
from services import dataset_service  # noqa: E402

YOLO_ROOT = Path(r"E:\2Project\2026dachuang\Yolo_dis_v3源代码\datasets")
UNET_ROOT = Path(r"E:\2Project\2026dachuang\Unet")


def _count_split_images(root: Path, parent_name: str) -> int:
    if not root.exists():
        return 0
    return sum(1 for p in root.rglob("*")
               if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
               and p.parent.name == parent_name)


def main() -> int:
    parser = argparse.ArgumentParser(description="按原始划分重建数据集索引")
    parser.add_argument("--apply", action="store_true", help="实际执行")
    parser.add_argument("--keep-existing", action="store_true", help="保留已有数据集（追加新版本）")
    args = parser.parse_args()

    setup_logging()
    db.init_db()

    counts = db.table_counts()
    print("当前数据：")
    for key in ("dataset", "dataset_version", "image", "annotation", "depth_map"):
        print(f"  {key:<18} {counts.get(key, 0)}")

    det_images = YOLO_ROOT / "images"
    seg_root = UNET_ROOT / "dataset"
    plan = [
        ("项目 A 检测数据集", det_images, YOLO_ROOT / "labels",
         _count_split_images(det_images, "train") + _count_split_images(det_images, "val")),
        ("项目 B 分割数据集", seg_root, None, _count_split_images(seg_root, "images")),
    ]
    print("\n将导入：")
    total = 0
    for label, source, extra, count in plan:
        total += count
        print(f"  {label:<18} 来源 {source}  图像 {count} 张")
    print(f"  合计 {total} 张（目标 5,814 张：A 2,694 + B 3,120）")

    if not args.apply:
        print("\n以上为预览。加 --apply 参数执行。")
        return 0

    if not args.keep_existing:
        print("\n清空已有数据集索引 ...")
        with db.transaction() as conn:
            for table in ("depth_map", "annotation", "image", "dataset_version", "dataset"):
                conn.execute(f"DELETE FROM {table}")
        print("  已清空")

    results = {}
    print("\n导入项目 A 检测数据集（保留 train / val 原始划分）...")
    results["detection"] = dataset_service.import_legacy_detection_full(
        str(YOLO_ROOT / "images"), str(YOLO_ROOT / "labels"),
        dataset_name="road-target-det", version_tag="v1.0")
    print("  ", json.dumps(results["detection"], ensure_ascii=False))

    print("\n导入项目 B 分割数据集（保留 train / val / test 原始划分）...")
    results["segmentation"] = dataset_service.import_legacy_segmentation_full(
        str(UNET_ROOT), dataset_name="campus-road-seg", version_tag="v1.0")
    print("  ", json.dumps(results["segmentation"], ensure_ascii=False))

    total_images = results["detection"]["images"] + results["segmentation"]["images"]
    print("\n" + "=" * 74)
    print(f"导入完成：共 {total_images} 张图像")
    print("=" * 74)
    for name, item in results.items():
        print(f"  {name:<14} 图像 {item['images']:>5}  划分 {item['splits']}  "
              f"校验 {item['validation']}")

    print("\n数据库统计：")
    for table, count in db.table_counts().items():
        if count:
            print(f"  {table:<22} {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
