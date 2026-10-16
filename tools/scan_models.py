"""登记检测权重并核对模型仓库总量。

用法：python tools/scan_models.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import db  # noqa: E402
from core.logging_setup import setup_logging  # noqa: E402
from services import model_registry  # noqa: E402

YOLO_ROOT = Path(r"E:\2Project\2026dachuang\Yolo_dis_v3源代码")
UNET_ROOT = Path(r"E:\2Project\2026dachuang\Unet")


def main() -> int:
    setup_logging()
    db.init_db()
    project = db.query_one("SELECT id FROM project WHERE is_deleted=0 ORDER BY created_at LIMIT 1")
    project_id = project["id"] if project else None

    print("=" * 76)
    print("扫描平台目录 + 原项目权重目录")
    print("=" * 76)
    result = model_registry.scan_platform_models(project_id)
    print(f"  导入 {result['imported']} 个，跳过 {result['skipped']} 个")
    for item in result["details"][:8]:
        print(f"    {item['name']:<22} {item['file']:<26} {item['size_mb']} MB")

    print("\n扫描项目 A 的训练产物（runs/）...")
    if (YOLO_ROOT / "runs").exists():
        runs = model_registry.scan_yolo_runs(str(YOLO_ROOT / "runs"), project_id)
        print(f"  导入 {runs['imported']} 个，跳过 {runs['skipped']} 个")

    print("\n扫描项目 B 的分割权重 ...")
    if (UNET_ROOT / "models").exists():
        seg = model_registry.scan_segmentation_weights(
            str(UNET_ROOT / "models"), project_id,
            str(UNET_ROOT / "best_model") if (UNET_ROOT / "best_model").exists() else None)
        print(f"  导入 {seg['imported']} 个，跳过 {seg['skipped']} 个")

    stats = model_registry.model_stats()
    print("\n" + "=" * 76)
    print("模型仓库总览")
    print("=" * 76)
    print(json.dumps({k: v for k, v in stats.items() if k != "by_arch"},
                     ensure_ascii=False, indent=2))

    rows = db.query_all(
        "SELECT task_type, weight_origin, COUNT(*) AS n FROM model WHERE is_deleted=0 "
        "GROUP BY task_type, weight_origin ORDER BY task_type, weight_origin")
    print("  按任务与来源：")
    for row in rows:
        print(f"    {row['task_type']:<14} {row['weight_origin']:<12} {row['n']:>4}")

    print("\n  检测模型清单（前 12）：")
    for row in db.query_all("SELECT name, arch, variant, weight_origin, size_bytes, notes "
                            "FROM model WHERE task_type='detection' AND is_deleted=0 "
                            "ORDER BY weight_origin, arch LIMIT 12"):
        print(f"    {row['name']:<22} {row['arch']:<10} {row['variant']:<3} "
              f"{row['weight_origin']:<11} {row['size_bytes'] / 1024 / 1024:>6.1f} MB")
    print("\n  目标：40 个检测权重（20 预训练 + 20 自训练）+ 80 个分割权重")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
