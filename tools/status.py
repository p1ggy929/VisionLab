"""数据状态速览：表计数、历史数据回归口径、模型与数据集概况、误差统计。

用法：python tools/status.py
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
from services import analysis_service  # noqa: E402


def main() -> int:
    setup_logging()
    db.init_db()

    print("=" * 74)
    print("元数据库表计数")
    print("=" * 74)
    for table, count in db.table_counts().items():
        if count:
            print(f"  {table:<24} {count}")

    print("\n" + "=" * 74)
    print("历史数据回归（项目 A 的 measurements.csv）")
    print("=" * 74)
    row = db.query_one("SELECT id FROM experiment WHERE name LIKE '历史数据迁移%' "
                       "AND task_type='detection' AND is_deleted=0 LIMIT 1")
    if row:
        summary = analysis_service.legacy_summary(row["id"])
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print("\n对照原申报书 / 分析报告口径：")
        print("  总记录 1460，缺检 254（17.40%），有效 1206，"
              "平均误差 0.446 m，中位误差 0.190 m，最大误差 6.57 m")

    print("\n" + "=" * 74)
    print("模型仓库")
    print("=" * 74)
    from services import model_registry

    stats = model_registry.model_stats()
    print(json.dumps({k: v for k, v in stats.items() if k != "by_arch"},
                     ensure_ascii=False, indent=2))
    print(f"  结构分布（前 8）：{dict(list(stats['by_arch'].items())[:8])}")

    print("\n" + "=" * 74)
    print("数据集")
    print("=" * 74)
    for ds in db.query_all("SELECT id, name, task_type, status, latest_version FROM dataset "
                           "WHERE is_deleted=0"):
        versions = db.query_all("SELECT version, image_count, annotation_count, content_hash, "
                                "frozen_at FROM dataset_version WHERE dataset_id=?", (ds["id"],))
        print(f"  {ds['name']}（{ds['task_type']}，{ds['status']}）")
        for v in versions:
            print(f"     {v['version']}  图像 {v['image_count']}  掩膜/标注 {v['annotation_count']}  "
                  f"哈希 {(v['content_hash'] or '-')[:12]}  "
                  f"{'已冻结' if v['frozen_at'] else '未冻结'}")

    print("\n" + "=" * 74)
    print("校正方案")
    print("=" * 74)
    for plan in db.decode_rows(db.query_all(
            "SELECT name, method, sample_count, rmse_before, rmse_after, is_active "
            "FROM calibration_plan ORDER BY created_at DESC LIMIT 12")):
        print(f"  {'[启用]' if plan['is_active'] else '      '} {plan['name']:<24} "
              f"{plan['method']:<10} 样本 {plan['sample_count']:<5} "
              f"RMSE {plan['rmse_before']} -> {plan['rmse_after']}")

    print("\n" + "=" * 74)
    print("全量误差分析（平台口径）")
    print("=" * 74)
    stats = analysis_service.summary({})
    for key in ("sample_count", "valid_count", "mae_m", "rmse_m", "mre", "median_error_m",
                "max_error_m", "std_error_m", "hit_rate", "miss_count", "miss_rate"):
        print(f"  {key:<18} {stats.get(key)}")
    print("\n  按距离分组：")
    for item in (stats.get("by_distance") or []):
        print(f"    {str(item['key']):>6} m  样本 {item['count']:>4}  有效 {item['valid_count']:>4}  "
              f"MAE {item['mae_m']}  RMSE {item['rmse_m']}")
    print("\n  按类别分组（前 6）：")
    for item in (stats.get("by_category") or [])[:6]:
        print(f"    {str(item['key']):<10} 样本 {item['count']:>4}  有效 {item['valid_count']:>4}  "
              f"MAE {item['mae_m']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
