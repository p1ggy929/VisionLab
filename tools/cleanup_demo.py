"""维护工具：清理验收/冒烟测试产生的演示实验，保留历史迁移实验。

用法：
    python tools/cleanup_demo.py            # 预览将删除的内容
    python tools/cleanup_demo.py --apply    # 实际删除
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core import db  # noqa: E402
from core.logging_setup import get_cli_logger, setup_logging  # noqa: E402

log = get_cli_logger("tools.cleanup_demo")

PATTERNS = ("端到端验收实验%", "实时实验%", "冒烟测试实验%", "默认实验%")


def main() -> int:
    parser = argparse.ArgumentParser(description="清理演示实验")
    parser.add_argument("--apply", action="store_true", help="实际执行删除")
    args = parser.parse_args()

    setup_logging()
    db.init_db()

    where = " OR ".join(["name LIKE ?"] * len(PATTERNS))
    rows = db.query_all(f"SELECT id, name, task_type, created_at FROM experiment "
                        f"WHERE is_deleted=0 AND ({where}) ORDER BY created_at",
                        PATTERNS)
    if not rows:
        log.info("没有需要清理的演示实验。")
        return 0

    log.info(f"发现 {len(rows)} 个演示实验：")
    total_records = 0
    for row in rows:
        count = int(db.query_scalar("SELECT COUNT(*) FROM ranging_result WHERE experiment_id=?",
                                     (row["id"],), 0) or 0)
        total_records += count
        log.info(f"  {row['id']}  {row['name']}  ({row['task_type']}, {count} 条测距记录)")

    reports = db.query_all(
        "SELECT id FROM report WHERE experiment_id IN "
        f"(SELECT id FROM experiment WHERE is_deleted=0 AND ({where}))", PATTERNS)
    log.info(f"关联报告 {len(reports)} 份，测距记录合计 {total_records} 条")

    if not args.apply:
        log.info("\n以上为预览。加 --apply 参数执行删除。")
        return 0

    ids = [row["id"] for row in rows]
    marks = ",".join("?" * len(ids))
    with db.transaction() as conn:
        conn.execute(f"DELETE FROM ranging_result WHERE experiment_id IN ({marks})", ids)
        conn.execute(f"DELETE FROM detection_result WHERE experiment_id IN ({marks})", ids)
        conn.execute(f"DELETE FROM segmentation_result WHERE experiment_id IN ({marks})", ids)
        conn.execute(f"DELETE FROM error_analysis WHERE experiment_id IN ({marks})", ids)
        conn.execute(f"DELETE FROM report WHERE experiment_id IN ({marks})", ids)
        conn.execute(f"DELETE FROM task_queue WHERE experiment_id IN ({marks})", ids)
        conn.execute(f"DELETE FROM experiment WHERE id IN ({marks})", ids)
    log.info(f"\n已删除 {len(ids)} 个演示实验及其产物。")
    log.info("剩余数据：")
    for table, count in db.table_counts().items():
        if count:
            log.info(f"  {table:<22} {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
