"""命令行环境自检（等价于 Web 端「系统设置 → 一键自检」）。

设计要点：本脚本的用途是「启动失败时帮助定位原因」，所以它自身必须比被测对象更健壮——
任何一项检查炸掉都不能让整份报告消失。尤其是存储目录不可用时，初始化数据库会直接抛异常；
若在这里崩掉，用户看到的只是一段 Python traceback，反而拿不到「是 storage.root 配错了」
这个本来最该得到的结论。因此数据库/自检调用都做了局部兜底，并把结论翻译成可执行的建议。
"""

from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core.logging_setup import get_cli_logger  # noqa: E402

log = get_cli_logger("tools.self_check")

WIDTH = 74


def _print_header() -> None:
    log.info("=" * WIDTH)
    log.info("  视界智研 · 环境自检")
    log.info("=" * WIDTH)


def _storage_root() -> str:
    try:
        from core.config import get_settings

        return str(get_settings().storage_root)
    except Exception:
        return "storage"


def _report_storage_failure(exc: BaseException) -> int:
    log.info(f"[错误] {'存储目录不可用':<22} {_storage_root()}")
    log.info(f"       {type(exc).__name__}: {exc}")
    log.info("-" * WIDTH)
    log.info("共 1 项，通过 0，错误 1，警告 0")
    log.info("环境未就绪：请先修复存储目录，再重新启动。")
    log.info("=" * WIDTH)
    log.info()
    log.info("排查建议：")
    log.info("  1. 打开 configs/default.yaml，确认 storage.root 指向真实存在的盘符或目录；")
    log.info("  2. 该目录需要当前用户可写（不要指向只读盘、光驱或已断开的网络位置）；")
    log.info("  3. 若曾用环境变量覆盖，清掉后重试：Remove-Item Env:VISIONLAB_STORAGE__ROOT")
    return 2


def main() -> int:
    _print_header()

    # 日志初始化失败不影响后续检查，只提示。
    try:
        from core.logging_setup import setup_logging

        setup_logging()
    except Exception as exc:
        log.info(f"[警告] {'日志初始化失败':<22} {type(exc).__name__}: {exc}")

    try:
        from core import db

        db.init_db()
    except Exception as exc:
        return _report_storage_failure(exc)

    try:
        from services import health_service

        report = health_service.self_check()
    except Exception as exc:
        log.info(f"[错误] {'自检执行失败':<22} {type(exc).__name__}: {exc}")
        log.info("-" * WIDTH)
        log.info("共 1 项，通过 0，错误 1，警告 0")
        log.info("环境未就绪：自检本身未能完成，请根据上面的异常定位。")
        log.info("=" * WIDTH)
        return 2

    for check in report["checks"]:
        mark = "OK  " if check["ok"] else ("警告" if check["level"] == "warning" else "错误")
        log.info(f"[{mark}] {check['name']:<22} {check['detail']}")

    summary = report["summary"]
    log.info("-" * WIDTH)
    log.info(f"共 {summary['total']} 项，通过 {summary['passed']}，"
          f"错误 {summary['errors']}，警告 {summary['warnings']}")
    log.info(report["conclusion"])
    log.info("=" * WIDTH)
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
