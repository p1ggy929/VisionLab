"""SQLite 访问层：连接管理、事务、查询助手、初始化与迁移。

表结构见 core/schema.py；模型层不引入 ORM，保持与原项目一致的轻依赖风格。
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from typing import Any, Iterable, Iterator, Sequence

from .config import settings
from .logging_setup import get_logger
from .schema import DDL, INDEXES, SCHEMA_VERSION

log = get_logger("db")

_local = threading.local()
_write_lock = threading.RLock()

CN_TZ = timezone(timedelta(hours=8))


def now_iso() -> str:
    return datetime.now(CN_TZ).replace(microsecond=0).isoformat()


def new_id(prefix: str = "") -> str:
    token = uuid.uuid4().hex[:16]
    return f"{prefix}_{token}" if prefix else token


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(settings.db_path), timeout=30.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def get_conn() -> sqlite3.Connection:
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = _connect()
        _local.conn = conn
    return conn


def close_conn() -> None:
    conn = getattr(_local, "conn", None)
    if conn is not None:
        conn.close()
        _local.conn = None


@contextmanager
def transaction() -> Iterator[sqlite3.Connection]:
    conn = get_conn()
    with _write_lock:
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise


# ---------------------------------------------------------------- 查询助手
def query_all(sql: str, params: Sequence[Any] | dict = ()) -> list[dict]:
    cur = get_conn().execute(sql, params)
    return [dict(row) for row in cur.fetchall()]


def query_one(sql: str, params: Sequence[Any] | dict = ()) -> dict | None:
    cur = get_conn().execute(sql, params)
    row = cur.fetchone()
    return dict(row) if row else None


def query_scalar(sql: str, params: Sequence[Any] | dict = (), default: Any = None) -> Any:
    cur = get_conn().execute(sql, params)
    row = cur.fetchone()
    if row is None:
        return default
    return row[0]


def execute(sql: str, params: Sequence[Any] | dict = ()) -> int:
    with transaction() as conn:
        cur = conn.execute(sql, params)
        return cur.rowcount


def insert(table: str, data: dict, id_prefix: str = "") -> str:
    payload = dict(data)
    if "id" not in payload or not payload["id"]:
        payload["id"] = new_id(id_prefix)
    stamp = now_iso()
    payload.setdefault("created_at", stamp)
    if _has_column(table, "updated_at"):
        payload.setdefault("updated_at", stamp)
    payload = _encode(payload)
    cols = ", ".join(payload.keys())
    marks = ", ".join(["?"] * len(payload))
    with transaction() as conn:
        conn.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})", list(payload.values()))
    return payload["id"]


def update(table: str, row_id: str, data: dict) -> int:
    payload = _encode(dict(data))
    if _has_column(table, "updated_at"):
        payload["updated_at"] = now_iso()
    if not payload:
        return 0
    assigns = ", ".join(f"{k}=?" for k in payload)
    params = list(payload.values()) + [row_id]
    with transaction() as conn:
        cur = conn.execute(f"UPDATE {table} SET {assigns} WHERE id=?", params)
        return cur.rowcount


def soft_delete(table: str, row_id: str) -> int:
    if _has_column(table, "is_deleted"):
        return update(table, row_id, {"is_deleted": 1})
    with transaction() as conn:
        cur = conn.execute(f"DELETE FROM {table} WHERE id=?", (row_id,))
        return cur.rowcount


def paginate(sql: str, count_sql: str, params: Sequence[Any], page: int, page_size: int) -> dict:
    page = max(1, int(page))
    page_size = min(500, max(1, int(page_size)))
    total = int(query_scalar(count_sql, params, 0) or 0)
    rows = query_all(f"{sql} LIMIT ? OFFSET ?", list(params) + [page_size, (page - 1) * page_size])
    return {"items": rows, "total": total, "page": page, "page_size": page_size}


# ---------------------------------------------------------------- 编解码
_JSON_SUFFIX = (
    "config", "class_names", "foreground_values", "stats", "validation_report",
    "meta", "payload", "quality_flags", "metrics", "resource", "filters",
    "intrinsics", "extrinsics", "depth_check", "config_snapshot", "run_manifest",
    "boxes", "centroid", "sample_points", "group_by", "by_distance", "by_category",
    "by_model", "by_size", "by_illumination", "outliers", "calibration_params",
    "params", "sections", "detail",
)


def _encode(data: dict) -> dict:
    out: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, (dict, list, tuple)) and key in _JSON_SUFFIX:
            out[key] = json.dumps(value, ensure_ascii=False)
        elif isinstance(value, bool):
            out[key] = 1 if value else 0
        else:
            out[key] = value
    return out


def decode_row(row: dict | None) -> dict | None:
    if row is None:
        return None
    out = {}
    for key, value in row.items():
        if key in _JSON_SUFFIX and isinstance(value, str):
            try:
                out[key] = json.loads(value)
                continue
            except json.JSONDecodeError:
                pass
        out[key] = value
    return out


def decode_rows(rows: Iterable[dict]) -> list[dict]:
    return [decode_row(r) or {} for r in rows]


# ---------------------------------------------------------------- 初始化
_column_cache: dict[str, set[str]] = {}


def _has_column(table: str, column: str) -> bool:
    if table not in _column_cache:
        rows = query_all(f"PRAGMA table_info({table})")
        _column_cache[table] = {r["name"] for r in rows}
    return column in _column_cache[table]


def init_db() -> None:
    settings.all_dirs()
    conn = get_conn()
    with _write_lock:
        for statement in DDL:
            conn.execute(statement)
        for statement in INDEXES:
            conn.execute(statement)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_meta (key TEXT PRIMARY KEY, value TEXT)"
        )
        conn.execute(
            "INSERT INTO schema_meta(key, value) VALUES('version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(SCHEMA_VERSION),),
        )
        conn.commit()
    _column_cache.clear()
    log.info("元数据库初始化完成: %s", settings.db_path)


def table_counts() -> dict[str, int]:
    tables = [r["name"] for r in query_all(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    )]
    result = {}
    for name in sorted(tables):
        result[name] = int(query_scalar(f"SELECT COUNT(*) FROM {name}", (), 0) or 0)
    return result
