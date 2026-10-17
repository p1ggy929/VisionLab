"""在真实历史数据上拟合误差校正方案，并把最优方案设为平台当前启用方案。

数据来源：原项目 A 的 1,462 条测距记录（1–20 m、11 类目标、20 个 YOLO 配置）。

用法：python tools/fit_calibration.py
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

log = get_cli_logger("tools.fit_calibration")


def main() -> int:
    setup_logging()
    db.init_db()

    experiment = db.query_one(
        "SELECT id, name FROM experiment WHERE name LIKE '历史数据迁移%' AND task_type='detection' "
        "AND is_deleted=0 LIMIT 1")
    if experiment is None:
        log.info("未找到历史迁移实验，请先运行 tools/import_legacy.py")
        return 1
    filters = {"experiment_id": experiment["id"]}
    log.info(f"数据来源: {experiment['name']}  ({experiment['id']})")

    stats = analysis_service.summary(filters)
    log.info(f"样本 {stats['sample_count']}  有效 {stats['valid_count']}  "
          f"MAE {stats['mae_m']} m  RMSE {stats['rmse_m']} m  缺检率 {stats['miss_rate']}")

    results = []
    log.info("\n" + "=" * 82)
    log.info("全量拟合")
    log.info("=" * 82)
    for method in ("linear", "ratio", "piecewise"):
        plan = analysis_service.calibrate_from_records(analysis_service.CalibrationRequest(
            method=method, filters=filters, name=f"历史数据_{method}",
            experiment_id=experiment["id"]))
        results.append(plan)
        log.info(f"  {method:<10} RMSE {plan['rmse_before']:.4f} -> {plan['rmse_after']:.4f} m  "
              f"样本 {plan['sample_count']}  推荐启用={plan['recommended']}")

    best = min(results, key=lambda p: p["rmse_after"] or 9e9)
    log.info(f"\n最优方案: {best['name']}（RMSE {best['rmse_after']:.4f} m，"
          f"相对未校正下降 {(1 - best['rmse_after'] / best['rmse_before']) * 100:.1f}%）")

    log.info("\n" + "=" * 82)
    log.info("分段模型参数（近 / 中 / 远 三段线性）")
    log.info("=" * 82)
    piecewise = [p for p in results if p["method"] == "piecewise"][0]
    log.info(json.dumps(piecewise["params"], ensure_ascii=False, indent=2))

    log.info("\n" + "=" * 82)
    log.info("按类别拟合并启用（示例：车辆 —— 历史数据中误差最大的类别）")
    log.info("=" * 82)
    category_filters = dict(filters)
    category_filters["category"] = "车辆"
    category_plan = analysis_service.calibrate_from_records(analysis_service.CalibrationRequest(
        method="piecewise", filters=category_filters, name="车辆_piecewise",
        experiment_id=experiment["id"]))
    log.info(f"  车辆: RMSE {category_plan['rmse_before']:.4f} -> {category_plan['rmse_after']:.4f} m  "
          f"推荐={category_plan['recommended']}")

    row = analysis_service.activate_calibration(best["id"])
    plan = analysis_service.build_plan(row)
    log.info(f"\n已启用平台校正方案: {row['name']}（方法 {row['method']}）")
    if plan:
        log.info("  校正曲线抽样：")
        for probe in (1.0, 3.0, 5.0, 10.0, 15.0, 20.0):
            log.info(f"    {probe:>5.1f} m  ->  {plan.apply(probe):.3f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
