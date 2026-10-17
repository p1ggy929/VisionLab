"""数据状态速览：表计数、历史数据回归口径、模型与数据集概况、误差统计。

用法：python tools/status.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core import db  # noqa: E402
from core.logging_setup import get_cli_logger, setup_logging  # noqa: E402
from services import analysis_service  # noqa: E402

log = get_cli_logger("tools.status")


def main() -> int:
    setup_logging()
    db.init_db()

    log.info("=" * 74)
    log.info("元数据库表计数")
    log.info("=" * 74)
    for table, count in db.table_counts().items():
        if count:
            log.info(f"  {table:<24} {count}")

    log.info("\n" + "=" * 74)
    log.info("历史数据回归（项目 A 的 measurements.csv）")
    log.info("=" * 74)
    row = db.query_one("SELECT id FROM experiment WHERE name LIKE '历史数据迁移%' "
                       "AND task_type='detection' AND is_deleted=0 LIMIT 1")
    if row:
        summary = analysis_service.legacy_summary(row["id"])
        log.info(json.dumps(summary, ensure_ascii=False, indent=2))
        log.info("\n对照原申报书 / 分析报告口径：")
        log.info("  总记录 1460，缺检 254（17.40%），有效 1206，"
              "平均误差 0.446 m，中位误差 0.190 m，最大误差 6.57 m")

    log.info("\n" + "=" * 74)
    log.info("模型仓库")
    log.info("=" * 74)
    from services import model_registry

    stats = model_registry.model_stats()
    log.info(json.dumps({k: v for k, v in stats.items() if k != "by_arch"},
                     ensure_ascii=False, indent=2))
    log.info(f"  结构分布（前 8）：{dict(list(stats['by_arch'].items())[:8])}")

    log.info("\n" + "=" * 74)
    log.info("数据集")
    log.info("=" * 74)
    for ds in db.query_all("SELECT id, name, task_type, status, latest_version FROM dataset "
                           "WHERE is_deleted=0"):
        versions = db.query_all("SELECT version, image_count, annotation_count, content_hash, "
                                "frozen_at FROM dataset_version WHERE dataset_id=?", (ds["id"],))
        log.info(f"  {ds['name']}（{ds['task_type']}，{ds['status']}）")
        for v in versions:
            log.info(f"     {v['version']}  图像 {v['image_count']}  掩膜/标注 {v['annotation_count']}  "
                  f"哈希 {(v['content_hash'] or '-')[:12]}  "
                  f"{'已冻结' if v['frozen_at'] else '未冻结'}")

    log.info("\n" + "=" * 74)
    log.info("校正方案")
    log.info("=" * 74)
    for plan in db.decode_rows(db.query_all(
            "SELECT name, method, sample_count, rmse_before, rmse_after, is_active "
            "FROM calibration_plan ORDER BY created_at DESC LIMIT 12")):
        log.info(f"  {'[启用]' if plan['is_active'] else '      '} {plan['name']:<24} "
              f"{plan['method']:<10} 样本 {plan['sample_count']:<5} "
              f"RMSE {plan['rmse_before']} -> {plan['rmse_after']}")

    log.info("\n" + "=" * 74)
    log.info("全量误差分析（平台口径）")
    log.info("=" * 74)
    stats = analysis_service.summary({})
    for key in ("sample_count", "valid_count", "mae_m", "rmse_m", "mre", "median_error_m",
                "max_error_m", "std_error_m", "hit_rate", "miss_count", "miss_rate"):
        log.info(f"  {key:<18} {stats.get(key)}")
    log.info("\n  按距离分组：")
    for item in (stats.get("by_distance") or []):
        log.info(f"    {str(item['key']):>6} m  样本 {item['count']:>4}  有效 {item['valid_count']:>4}  "
              f"MAE {item['mae_m']}  RMSE {item['rmse_m']}")
    log.info("\n  按类别分组（前 6）：")
    for item in (stats.get("by_category") or [])[:6]:
        log.info(f"    {str(item['key']):<10} 样本 {item['count']:>4}  有效 {item['valid_count']:>4}  "
              f"MAE {item['mae_m']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
