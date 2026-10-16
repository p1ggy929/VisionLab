"""维护工具：把校正方案与中间分析记录收敛到最终交付状态。

只保留在真实历史数据上拟合的校正方案，并启用其中 RMSE 最低的一个。
用法：python tools/tidy_state.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import db  # noqa: E402
from core.logging_setup import setup_logging  # noqa: E402

KEEP_PREFIXES = ("历史数据", "车辆")


def main() -> int:
    setup_logging()
    db.init_db()
    patterns = " OR ".join(["name LIKE ?"] * len(KEEP_PREFIXES))
    params = [p + "%" for p in KEEP_PREFIXES]

    with db.transaction() as conn:
        conn.execute(f"DELETE FROM calibration_plan WHERE NOT ({patterns})", params)
        conn.execute("DELETE FROM calibration")
        conn.execute("DELETE FROM error_analysis")
        conn.execute("UPDATE calibration_plan SET is_active=0")
        # 平台默认启用「全量分段校正」；按类别的方案保留为可选，不默认启用
        row = conn.execute(
            "SELECT id FROM calibration_plan WHERE name='历史数据_piecewise' LIMIT 1").fetchone()
        if row is None:
            row = conn.execute(
                "SELECT id FROM calibration_plan WHERE method='piecewise' "
                "ORDER BY sample_count DESC LIMIT 1").fetchone()
        if row:
            conn.execute("UPDATE calibration_plan SET is_active=1 WHERE id=?", (row[0],))

    print("保留的校正方案：")
    for item in db.query_all("SELECT name, method, sample_count, rmse_before, rmse_after, "
                             "is_active FROM calibration_plan ORDER BY created_at"):
        flag = "已启用" if item["is_active"] else "      "
        print(f"  [{flag}] {item['name']:<20} {item['method']:<10} "
              f"样本 {item['sample_count']:<5} RMSE {item['rmse_before']} -> {item['rmse_after']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
