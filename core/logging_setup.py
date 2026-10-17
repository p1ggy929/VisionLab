"""日志：控制台 + 滚动文件双通道（沿用项目 A backend/config.py 的做法）。"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .config import settings

_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"
_ROOT_NAME = "visionlab"
_configured = False


def _has_console(logger: logging.Logger) -> bool:
    return any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    )


def _has_file(logger: logging.Logger, path: Path) -> bool:
    target = path.resolve()
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler):
            base = getattr(h, "baseFilename", None)
            if base and Path(base).resolve() == target:
                return True
    return False


def setup_logging() -> logging.Logger:
    global _configured
    logger = logging.getLogger(_ROOT_NAME)
    if _configured:
        return logger

    level = getattr(logging, str(settings.get("logging.level", "INFO")).upper(), logging.INFO)
    logger.setLevel(level)
    logger.propagate = False
    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    if settings.get("logging.console", True) and not _has_console(logger):
        console = logging.StreamHandler(stream=sys.stdout)
        console.setFormatter(formatter)
        logger.addHandler(console)

    log_path = settings.log_path
    if not _has_file(logger, log_path):
        handler = RotatingFileHandler(
            log_path,
            maxBytes=int(settings.get("logging.max_bytes", 5 * 1024 * 1024)),
            backupCount=int(settings.get("logging.backup_count", 5)),
            encoding="utf-8",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # 让 werkzeug / socketio 的日志也进同一通道
    for name in ("werkzeug", "socketio", "engineio"):
        child = logging.getLogger(name)
        child.handlers = list(logger.handlers)
        child.setLevel(level)
        child.propagate = False

    _configured = True
    return logger


def get_logger(name: str = "") -> logging.Logger:
    setup_logging()
    return logging.getLogger(f"{_ROOT_NAME}.{name}" if name else _ROOT_NAME)


def get_cli_logger(name: str = "") -> logging.Logger:
    """命令行脚本专用日志器：只输出消息本身，保持终端排版简洁。

    脚本的用途是给人看进度与结果，若沿用 Web 端的
    ``[时间] [级别] [模块]`` 前缀，表格与分节线会被撑散、难以阅读。
    """
    logger = logging.getLogger(f"{_ROOT_NAME}.cli.{name}" if name else f"{_ROOT_NAME}.cli")
    if not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


class _MemoryHandler(logging.Handler):
    """内存环形缓冲，供监控页读取最近日志。"""

    def __init__(self, capacity: int = 500) -> None:
        super().__init__()
        self.capacity = capacity
        self.records: list[dict] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.records.append(
                {
                    "time": self.formatTime(record, _DATEFMT),
                    "level": record.levelname,
                    "module": record.name,
                    "message": record.getMessage(),
                }
            )
            if len(self.records) > self.capacity:
                del self.records[: len(self.records) - self.capacity]
        except Exception:  # pragma: no cover - 日志失败不得影响主流程
            pass


_memory = _MemoryHandler()


def attach_memory_handler() -> _MemoryHandler:
    logger = setup_logging()
    if _memory not in logger.handlers:
        logger.addHandler(_memory)
    return _memory


def recent_logs(level: str | None = None, keyword: str | None = None, limit: int = 200) -> list[dict]:
    attach_memory_handler()
    items = list(_memory.records)
    if level:
        items = [r for r in items if r["level"] == level.upper()]
    if keyword:
        key = keyword.lower()
        items = [r for r in items if key in r["message"].lower() or key in r["module"].lower()]
    return items[-limit:][::-1]
