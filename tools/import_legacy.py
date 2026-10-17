"""命令行工具：把两个原项目的历史数据一次性迁移进平台。

用法（在平台根目录下执行）：
    python tools/import_legacy.py --project "校园道路视觉感知" ^
        --measurements  "E:\\2Project\\2026dachuang\\Yolo_dis_v3源代码\\data\\results\\measurements.csv" ^
        --distance-records "E:\\2Project\\2026dachuang\\Unet\\distance_records.csv" ^
        --det-images    "E:\\2Project\\2026dachuang\\Yolo_dis_v3源代码\\datasets\\images" ^
        --seg-images    "E:\\2Project\\2026dachuang\\Unet\\dataset\\train\\images" ^
        --seg-masks     "E:\\2Project\\2026dachuang\\Unet\\dataset\\train\\masks" ^
        --seg-weights   "E:\\2Project\\2026dachuang\\Unet\\models" ^
        --best-model    "E:\\2Project\\2026dachuang\\Unet\\best_model" ^
        --runs          "E:\\2Project\\2026dachuang\\Yolo_dis_v3源代码\\runs"

参数全部可选；只填存在的路径即可。
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
from services import analysis_service, dataset_service, model_registry, project_service  # noqa: E402

log = get_cli_logger("tools.import_legacy")


def ensure_project(name: str | None) -> str:
    if name:
        row = db.query_one("SELECT id FROM project WHERE name=? AND is_deleted=0", (name,))
        if row:
            return row["id"]
        project = project_service.create_project(
            project_service.ProjectSpec(name, "历史数据迁移目标项目"))
        return project["id"]
    row = db.query_one("SELECT id FROM project WHERE is_deleted=0 ORDER BY created_at LIMIT 1")
    if row:
        return row["id"]
    return project_service.create_project(
        project_service.ProjectSpec("校园道路视觉感知", "平台默认项目"))["id"]


def main() -> int:
    parser = argparse.ArgumentParser(description="导入两个原项目的历史数据")
    parser.add_argument("--project", default=None, help="目标项目名称（不存在则创建）")
    parser.add_argument("--measurements", default=None, help="项目 A 的 measurements.csv")
    parser.add_argument("--distance-records", default=None, help="项目 B 的 distance_records.csv")
    parser.add_argument("--det-images", default=None, help="项目 A 的 datasets/images 目录")
    parser.add_argument("--det-labels", default=None, help="项目 A 的 datasets/labels 目录（可选）")
    parser.add_argument("--seg-images", default=None, help="项目 B 的分割图像目录")
    parser.add_argument("--seg-masks", default=None, help="项目 B 的分割掩膜目录")
    parser.add_argument("--seg-root", default=None,
                        help="项目 B 的数据根目录（含 dataset/train|val|test），按原始划分整体导入")
    parser.add_argument("--seg-weights", default=None, help="项目 B 的 models 目录（*.pth）")
    parser.add_argument("--best-model", default=None, help="项目 B 的 best_model 目录（指标 CSV）")
    parser.add_argument("--runs", default=None, help="项目 A 的 runs 目录")
    parser.add_argument("--det-limit", type=int, default=None, help="检测数据集导入张数上限")
    args = parser.parse_args()

    setup_logging()
    db.init_db()
    project_id = ensure_project(args.project)
    log.info(f"目标项目: {project_id}")
    result: dict = {}

    if args.measurements:
        log.info("导入项目 A 的测距记录 ...")
        result["measurements"] = analysis_service.import_legacy_measurements(
            args.measurements, project_id)
        log.info("  ", json.dumps(result["measurements"], ensure_ascii=False))

    if args.distance_records:
        log.info("导入项目 B 的测距记录 ...")
        result["distance_records"] = analysis_service.import_legacy_distance_records(
            args.distance_records, project_id)
        log.info("  ", json.dumps(result["distance_records"], ensure_ascii=False))

    if args.runs:
        log.info("扫描项目 A 的训练产物（YOLO 权重）...")
        result["yolo_models"] = model_registry.scan_yolo_runs(args.runs, project_id)
        log.info(f"   导入 {result['yolo_models']['imported']} 个，"
              f"跳过 {result['yolo_models']['skipped']} 个")

    if args.seg_weights:
        log.info("扫描项目 B 的分割权重 ...")
        result["seg_models"] = model_registry.scan_segmentation_weights(
            args.seg_weights, project_id, args.best_model)
        log.info(f"   导入 {result['seg_models']['imported']} 个，"
              f"跳过 {result['seg_models']['skipped']} 个")

    if args.seg_root:
        log.info("按原始划分整体导入项目 B 的分割数据集（train + val + test）...")
        result["seg_dataset_full"] = dataset_service.import_legacy_segmentation_full(
            args.seg_root)
        log.info("  ", json.dumps(result["seg_dataset_full"], ensure_ascii=False))
    elif args.seg_images and args.seg_masks:
        log.info("导入项目 B 的分割数据集（单目录）...")
        result["seg_dataset"] = dataset_service.import_legacy_segmentation_dataset(
            args.seg_images, args.seg_masks)
        log.info("  ", json.dumps(result["seg_dataset"], ensure_ascii=False))

    if args.det_images:
        images_root = Path(args.det_images)
        has_splits = any((images_root / s).exists() for s in ("train", "val", "test"))
        if has_splits:
            log.info("按原始划分整体导入项目 A 的检测数据集（train + val）...")
            result["det_dataset_full"] = dataset_service.import_legacy_detection_full(
                args.det_images, args.det_labels)
            log.info("  ", json.dumps(result["det_dataset_full"], ensure_ascii=False))
        else:
            log.info("导入项目 A 的检测数据集（单目录）...")
            result["det_dataset"] = dataset_service.import_legacy_detection_dataset(
                args.det_images, limit=args.det_limit)
            log.info("  ", json.dumps(result["det_dataset"], ensure_ascii=False))

    log.info("\n完成。数据库统计：")
    for table, count in db.table_counts().items():
        if count:
            log.info(f"   {table:<22} {count}")

    if result.get("measurements", {}).get("experiment_id"):
        exp_id = result["measurements"]["experiment_id"]
        log.info("\n历史数据回归验证（复现原申报书口径）：")
        log.info("  ", json.dumps(analysis_service.legacy_summary(exp_id), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
