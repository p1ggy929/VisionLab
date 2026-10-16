"""误差分析与校正：统计、六维归因、异常样本、三类校正方案、历史数据迁移、大屏聚合。"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from algorithms.calibration import CalibrationPlan, calibrate, is_recommended, residual_report
from core import db, storage
from core.config import settings
from core.exceptions import NotFoundError, RangingError, ValidationError
from core.logging_setup import get_logger

log = get_logger("services.analysis")

GROUP_DIMENSIONS = ("task_type", "strategy", "model_name", "category", "size_group",
                    "illumination", "distance_plan_m", "metric_kind", "target_kind")

_JSON_FIELDS = ("by_distance", "by_category", "by_model", "by_size", "by_illumination",
                "outliers", "group_by", "filters", "calibration_params")


def _decode(row: dict | None) -> dict:
    if row is None:
        return {}
    out = dict(row)
    for key in _JSON_FIELDS:
        if isinstance(out.get(key), str):
            try:
                out[key] = json.loads(out[key])
            except json.JSONDecodeError:
                pass
    return out


# ================================================================== 查询
def query_records(filters: dict | None = None, page: int = 1, page_size: int = 50,
                  order: str = "created_at DESC") -> dict:
    filters = filters or {}
    clauses, params = ["1=1"], []
    mapping = {
        "task_type": "task_type", "strategy": "strategy", "model_name": "model_name",
        "category": "category", "size_group": "size_group", "illumination": "illumination",
        "experiment_id": "experiment_id", "project_id": None, "target_kind": "target_kind",
        "metric_kind": "metric_kind", "source": "source",
    }
    for key, column in mapping.items():
        value = filters.get(key)
        if value in (None, "", []):
            continue
        if column is None:
            continue
        if isinstance(value, (list, tuple)):
            marks = ",".join(["?"] * len(value))
            clauses.append(f"{column} IN ({marks})")
            params += list(value)
        else:
            clauses.append(f"{column}=?")
            params.append(value)

    if filters.get("distance_plan_m") not in (None, "", []):
        clauses.append("distance_plan_m=?")
        params.append(float(filters["distance_plan_m"]))
    if filters.get("has_actual"):
        clauses.append("distance_actual_m IS NOT NULL")
    if filters.get("start"):
        clauses.append("created_at >= ?")
        params.append(filters["start"])
    if filters.get("end"):
        clauses.append("created_at <= ?")
        params.append(filters["end"])

    where = " AND ".join(clauses)
    result = db.paginate(
        f"SELECT * FROM ranging_result WHERE {where} ORDER BY {order}",
        f"SELECT COUNT(*) FROM ranging_result WHERE {where}", params, page, page_size)
    for item in result["items"]:
        item["sample_points"] = _decode({"sample_points": item.get("sample_points")}).get("sample_points")
    return result


def get_record(record_id: str) -> dict | None:
    return db.decode_row(db.query_one("SELECT * FROM ranging_result WHERE id=?", (record_id,)))


def update_record(record_id: str, **fields: Any) -> int:
    record = get_record(record_id)
    if record is None:
        raise NotFoundError("测距记录不存在。")
    payload = {k: fields[k] for k in ("distance_actual_m", "actual_source", "notes",
                                      "illumination", "category") if k in fields}
    if "distance_actual_m" in payload:
        actual = payload["distance_actual_m"]
        if actual in (None, ""):
            payload.update({"error_m": None, "abs_error_m": None, "rel_error": None})
        else:
            actual = float(actual)
            depth = float(record.get("depth_m") or 0.0)
            err = depth - actual
            payload.update({"distance_actual_m": actual, "error_m": err,
                            "abs_error_m": abs(err),
                            "rel_error": abs(err) / actual if actual > 0 else None})
    return db.update("ranging_result", record_id, payload)


def delete_record(record_id: str) -> int:
    from core import db as dbmod

    with dbmod.transaction() as conn:
        cur = conn.execute("DELETE FROM ranging_result WHERE id=?", (record_id,))
    return cur.rowcount


# ================================================================== 统计
def summary(filters: dict | None = None, save: bool = False,
            experiment_id: str | None = None) -> dict:
    result = query_records(filters, page=1, page_size=500)
    records = _fetch_all(filters)
    stats = _compute_stats(records)

    # 缺检率的两个口径：
    #   1) ranging_result 中以 depth_m IS NULL 标记的缺检（历史迁移数据的口径）
    #   2) detection_result.is_miss 标记的缺检（平台实时实验的口径）
    # 优先使用有数据的一方，二者都有时以实时实验口径为准并保留两者。
    det_all = int(db.query_scalar("SELECT COUNT(*) FROM detection_result", (), 0) or 0)
    frame_miss = int(db.query_scalar(
        "SELECT COUNT(*) FROM detection_result WHERE is_miss=1", (), 0) or 0)
    stats["frames_detected"] = det_all
    stats["frames_missed"] = frame_miss
    stats["frame_miss_rate"] = round(frame_miss / det_all, 4) if det_all else None
    if det_all == 0:
        # 无实时检测记录时，沿用测距记录自身的缺检统计
        stats["miss_count"] = stats.get("miss_count") or 0
        stats["miss_rate"] = stats.get("miss_rate")
    else:
        stats["miss_count"] = frame_miss
        stats["miss_rate"] = round(frame_miss / det_all, 4)

    if filters:
        stats["filters"] = filters
    if save:
        analysis_id = _persist(stats, experiment_id, filters or {})
        stats["analysis_id"] = analysis_id
    return stats


def _fetch_all(filters: dict | None, limit: int = 200000) -> list[dict]:
    filters = filters or {}
    clauses, params = ["1=1"], []
    for key in ("task_type", "strategy", "model_name", "category", "size_group",
                "illumination", "experiment_id", "target_kind", "metric_kind", "source"):
        value = filters.get(key)
        if value in (None, "", []):
            continue
        if isinstance(value, (list, tuple)):
            marks = ",".join(["?"] * len(value))
            clauses.append(f"{key} IN ({marks})")
            params += list(value)
        else:
            clauses.append(f"{key}=?")
            params.append(value)
    if filters.get("distance_plan_m") not in (None, "", []):
        clauses.append("distance_plan_m=?")
        params.append(float(filters["distance_plan_m"]))
    if filters.get("start"):
        clauses.append("created_at >= ?")
        params.append(filters["start"])
    if filters.get("end"):
        clauses.append("created_at <= ?")
        params.append(filters["end"])
    where = " AND ".join(clauses)
    return db.decode_rows(db.query_all(
        f"SELECT * FROM ranging_result WHERE {where} ORDER BY created_at LIMIT ?",
        params + [limit]))


def _compute_stats(records: list[dict]) -> dict:
    with_truth = [r for r in records
                  if r.get("distance_actual_m") not in (None, "")
                  and r.get("depth_m") not in (None, "")]
    depths = np.asarray([float(r["depth_m"]) for r in with_truth], dtype=float)
    actual = np.asarray([float(r["distance_actual_m"]) for r in with_truth], dtype=float)
    errors = depths - actual
    abs_errors = np.abs(errors)
    threshold = float(settings.get("analysis.hit_rate_threshold", 0.05))
    rel = np.divide(abs_errors, actual, out=np.zeros_like(abs_errors), where=actual > 0)

    missed = sum(1 for r in records if r.get("depth_m") in (None, "", 0))
    stats = {
        "sample_count": len(records),
        "valid_count": len(with_truth),
        "with_truth_count": len(with_truth),
        "miss_count": missed,
        "miss_rate": round(missed / len(records), 4) if records else None,
        "mae_m": round(float(abs_errors.mean()), 4) if abs_errors.size else None,
        "rmse_m": round(float(np.sqrt((errors ** 2).mean())), 4) if errors.size else None,
        "mre": round(float(rel.mean()), 4) if rel.size else None,
        "median_error_m": round(float(np.median(abs_errors)), 4) if abs_errors.size else None,
        "max_error_m": round(float(abs_errors.max()), 4) if abs_errors.size else None,
        "std_error_m": round(float(abs_errors.std()), 4) if abs_errors.size else None,
        "hit_rate": round(float((rel <= threshold).mean()), 4) if rel.size else None,
        "hit_rate_threshold": threshold,
        "depth_min_m": round(float(depths.min()), 3) if depths.size else None,
        "depth_max_m": round(float(depths.max()), 3) if depths.size else None,
    }
    stats["by_distance"] = _group(records, "distance_plan_m")
    stats["by_category"] = _group(records, "category")
    stats["by_model"] = _group(records, "model_name")
    stats["by_size"] = _group(records, "size_group")
    stats["by_illumination"] = _group(records, "illumination")
    stats["by_task"] = _group(records, "task_type")
    stats["by_strategy"] = _group(records, "strategy")
    stats["heatmap"] = _heatmap(records, "category", "distance_plan_m")
    stats["leaderboard"] = _leaderboard(records)
    stats["outliers"] = _outliers(records)
    return stats


def _group(records: list[dict], dimension: str, limit: int = 40) -> list[dict]:
    buckets: dict[Any, list[dict]] = defaultdict(list)
    for row in records:
        key = row.get(dimension)
        if key in (None, ""):
            continue
        buckets[key].append(row)
    out = []
    for key, items in buckets.items():
        # 只有同时具备测量值与真值的记录才参与误差统计（缺检记录不参与）
        paired = [r for r in items
                  if r.get("depth_m") not in (None, "") and r.get("distance_actual_m") not in (None, "")]
        abs_errors = np.asarray([abs(float(r["depth_m"]) - float(r["distance_actual_m"]))
                                 for r in paired], dtype=float)
        rel = np.asarray([abs(float(r["depth_m"]) - float(r["distance_actual_m"]))
                          / float(r["distance_actual_m"]) for r in paired], dtype=float)
        out.append({
            "key": key,
            "count": len(items),
            "valid_count": len(paired),
            "mae_m": round(float(abs_errors.mean()), 4) if abs_errors.size else None,
            "rmse_m": round(float(np.sqrt((abs_errors ** 2).mean())), 4) if abs_errors.size else None,
            "mre": round(float(rel.mean()), 4) if rel.size else None,
            "median_error_m": round(float(np.median(abs_errors)), 4) if abs_errors.size else None,
            "max_error_m": round(float(abs_errors.max()), 4) if abs_errors.size else None,
        })
    out.sort(key=lambda x: (-(x["count"] or 0), str(x["key"])))
    return out[:limit]


def _heatmap(records: list[dict], x_dim: str, y_dim: str) -> dict:
    xs, ys = sorted({str(r.get(x_dim)) for r in records if r.get(x_dim) not in (None, "")},
                    key=_sort_key), sorted(
        {r.get(y_dim) for r in records if r.get(y_dim) not in (None, "")}, key=_sort_key)
    xs = sorted({str(r.get(x_dim)) for r in records if r.get(x_dim) not in (None, "")}, key=_sort_key)
    matrix: list[list[float | None]] = []
    for y in ys:
        row: list[float | None] = []
        for x in xs:
            subset = [r for r in records if str(r.get(x_dim)) == x and r.get(y_dim) == y
                      and r.get("distance_actual_m") not in (None, "")
                      and r.get("depth_m") not in (None, "")]
            if subset:
                errs = [abs(float(r["depth_m"]) - float(r["distance_actual_m"])) for r in subset]
                row.append(round(float(np.mean(errs)), 4))
            else:
                row.append(None)
        matrix.append(row)
    return {"x_label": x_dim, "y_label": y_dim, "x": [str(x) for x in xs],
            "y": [str(y) for y in ys], "z": matrix}


def _sort_key(value: Any):
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, str(value))


def _leaderboard(records: list[dict], top: int = 15) -> list[dict]:
    grouped = _group(records, "model_name", limit=100)
    grouped.sort(key=lambda x: (x["mae_m"] is None, x["mae_m"] or 9e9))
    for item in grouped:
        item["task_type"] = next((r.get("task_type") for r in records
                                  if r.get("model_name") == item["key"]), None)
    return grouped[:top]


def _outliers(records: list[dict], top: int = 20) -> list[dict]:
    sigma = float(settings.get("analysis.outlier_sigma", 3.0))
    with_truth = [r for r in records
                  if r.get("distance_actual_m") not in (None, "")
                  and r.get("depth_m") not in (None, "")]
    if not with_truth:
        return []
    abs_errors = np.asarray([abs(float(r["depth_m"]) - float(r["distance_actual_m"]))
                             for r in with_truth], dtype=float)
    mean, std = float(abs_errors.mean()), float(abs_errors.std())
    threshold = mean + sigma * std if std > 0 else 9e9
    picked = [r for r, e in zip(with_truth, abs_errors) if e >= threshold]
    picked.sort(key=lambda r: -abs(float(r["depth_m"]) - float(r["distance_actual_m"])))
    out = []
    for row in (picked or sorted(with_truth,
                                 key=lambda r: -abs(float(r["depth_m"]) - float(r["distance_actual_m"])))[:top])[:top]:
        depth = float(row["depth_m"])
        actual = float(row["distance_actual_m"])
        out.append({
            "id": row["id"], "category": row.get("category"), "model_name": row.get("model_name"),
            "target_kind": row.get("target_kind"), "metric_kind": row.get("metric_kind"),
            "depth_m": round(depth, 4), "distance_actual_m": round(actual, 4),
            "abs_error_m": round(abs(depth - actual), 4),
            "rel_error": round(abs(depth - actual) / actual, 4) if actual else None,
            "media_uri": row.get("media_uri"), "illumination": row.get("illumination"),
            "created_at": row.get("created_at"),
        })
    return out


def _persist(stats: dict, experiment_id: str | None, filters: dict,
             calibration: CalibrationPlan | None = None) -> str:
    payload = {
        "experiment_id": experiment_id,
        "scope": "experiment" if experiment_id else "project",
        "task_type": (filters or {}).get("task_type"),
        "strategy": (filters or {}).get("strategy"),
        "group_by": list(filters.keys()) if filters else [],
        "filters": filters,
        "sample_count": stats.get("sample_count"),
        "valid_count": stats.get("valid_count"),
        "miss_count": stats.get("miss_count"),
        "miss_rate": stats.get("miss_rate"),
        "mae_m": stats.get("mae_m"), "rmse_m": stats.get("rmse_m"), "mre": stats.get("mre"),
        "median_error_m": stats.get("median_error_m"), "max_error_m": stats.get("max_error_m"),
        "std_error_m": stats.get("std_error_m"), "hit_rate": stats.get("hit_rate"),
        "by_distance": stats.get("by_distance"), "by_category": stats.get("by_category"),
        "by_model": stats.get("by_model"), "by_size": stats.get("by_size"),
        "by_illumination": stats.get("by_illumination"), "outliers": stats.get("outliers"),
    }
    if calibration is not None:
        payload.update({
            "calibration_method": calibration.method,
            "calibration_params": calibration.params,
            "rmse_before": calibration.rmse_before, "rmse_after": calibration.rmse_after,
            "is_recommended": 1 if is_recommended(calibration) else 0,
        })
    return db.insert("error_analysis", payload, id_prefix="ea")


def list_analyses(limit: int = 50) -> list[dict]:
    rows = db.query_all("SELECT * FROM error_analysis ORDER BY created_at DESC LIMIT ?", (limit,))
    return [_decode(r) for r in rows]


def get_analysis(analysis_id: str) -> dict | None:
    return _decode(db.query_one("SELECT * FROM error_analysis WHERE id=?", (analysis_id,)))


# ================================================================== 校正
def calibrate_from_records(method: str = "linear", filters: dict | None = None,
                           bounds: list[float] | None = None,
                           name: str | None = None, persist: bool = True,
                           experiment_id: str | None = None) -> dict:
    records = _fetch_all(filters)
    measured, actual = [], []
    for row in records:
        if row.get("distance_actual_m") in (None, "") or row.get("depth_m") in (None, ""):
            continue
        measured.append(float(row["depth_m"]))
        actual.append(float(row["distance_actual_m"]))
    if len(measured) < 2:
        raise RangingError(f"有效样本仅 {len(measured)} 条，无法拟合校正模型（至少需要 2 条）。")

    plan = calibrate(measured, actual, method=method, bounds=bounds,
                     name=name or f"{method}_{(filters or {}).get('category') or 'all'}")
    plan.pending_values = (measured, actual)  # type: ignore[attr-defined]
    payload = plan.to_dict()
    payload["recommended"] = is_recommended(plan)
    payload["residuals"] = residual_report(measured, actual, plan)[:200]
    payload["before_after"] = {
        "before": [{"measured": round(m, 3), "actual": round(a, 3)}
                   for m, a in list(zip(measured, actual))[:200]],
        "after": [{"measured": round(m, 3), "corrected": round(c, 3), "actual": round(a, 3)}
                  for m, c, a in zip(measured, plan.apply_array(np.asarray(measured)), actual)][:200],
    }

    if persist:
        plan_id = db.insert("calibration_plan", {
            "name": plan.name, "task_type": (filters or {}).get("task_type"),
            "strategy": (filters or {}).get("strategy"),
            "model_id": None, "category": (filters or {}).get("category"),
            "method": plan.method, "params": plan.params,
            "sample_count": plan.sample_count, "rmse_before": plan.rmse_before,
            "rmse_after": plan.rmse_after, "is_active": 0,
        }, id_prefix="cal")
        payload["id"] = plan_id
        _persist({"sample_count": len(measured), "valid_count": len(measured)},
                 experiment_id, filters or {}, plan)
    return payload


def list_calibration_plans() -> list[dict]:
    rows = db.query_all("SELECT * FROM calibration_plan ORDER BY created_at DESC LIMIT 100")
    return db.decode_rows(rows)


def get_calibration_plan(plan_id: str) -> dict | None:
    return db.decode_row(db.query_one("SELECT * FROM calibration_plan WHERE id=?", (plan_id,)))


def activate_calibration(plan_id: str) -> dict:
    """启用指定校正方案（同时停用其他方案），返回方案记录（dict）。"""
    row = get_calibration_plan(plan_id)
    if row is None:
        raise NotFoundError("校正方案不存在。")
    from core import db as dbmod

    with dbmod.transaction() as conn:
        conn.execute("UPDATE calibration_plan SET is_active=0")
        conn.execute("UPDATE calibration_plan SET is_active=1 WHERE id=?", (plan_id,))
    return get_calibration_plan(plan_id) or {}


def active_calibration() -> CalibrationPlan | None:
    row = db.query_one("SELECT * FROM calibration_plan WHERE is_active=1 ORDER BY created_at DESC LIMIT 1")
    return build_plan(db.decode_row(row)) if row else None


def build_plan(row: dict | None) -> CalibrationPlan | None:
    if not row:
        return None
    params = row.get("params")
    if isinstance(params, str):
        params = json.loads(params)
    return CalibrationPlan(method=row["method"], params=params, name=row["name"],
                           plan_id=row["id"], rmse_before=row.get("rmse_before"),
                           rmse_after=row.get("rmse_after"),
                           sample_count=row.get("sample_count") or 0)


def deactivate_calibration() -> None:
    from core import db as dbmod

    with dbmod.transaction() as conn:
        conn.execute("UPDATE calibration_plan SET is_active=0")


# ================================================================== 大屏
def dashboard_kpi() -> dict:
    projects = int(db.query_scalar("SELECT COUNT(*) FROM project WHERE is_deleted=0", (), 0) or 0)
    experiments = int(db.query_scalar("SELECT COUNT(*) FROM experiment WHERE is_deleted=0", (), 0) or 0)
    models = int(db.query_scalar("SELECT COUNT(*) FROM model WHERE is_deleted=0", (), 0) or 0)
    records = int(db.query_scalar("SELECT COUNT(*) FROM ranging_result", (), 0) or 0)
    images = int(db.query_scalar("SELECT COALESCE(SUM(image_count),0) FROM dataset_version", (), 0) or 0)
    mae = db.query_scalar("SELECT AVG(abs_error_m) FROM ranging_result WHERE abs_error_m IS NOT NULL", (), None)
    det_all = int(db.query_scalar("SELECT COUNT(*) FROM detection_result", (), 0) or 0)
    miss = int(db.query_scalar("SELECT COUNT(*) FROM detection_result WHERE is_miss=1", (), 0) or 0)
    return {
        "projects": projects, "experiments": experiments, "models": models,
        "ranging_records": records, "images": images,
        "mean_error_m": round(float(mae), 4) if mae is not None else None,
        "miss_rate": round(miss / det_all, 4) if det_all else None,
        "tasks": _task_counts(),
        "models_by_task": _counts_by("model", "task_type"),
        "records_by_task": _counts_by("ranging_result", "task_type"),
    }


def _counts_by(table: str, column: str) -> dict:
    rows = db.query_all(f"SELECT {column} AS k, COUNT(*) AS n FROM {table} GROUP BY {column}")
    return {str(r["k"]): r["n"] for r in rows}


def _task_counts() -> dict:
    rows = db.query_all("SELECT status, COUNT(*) AS n FROM task_queue GROUP BY status")
    out = {r["status"]: r["n"] for r in rows}
    for key in ("queued", "running", "done", "failed"):
        out.setdefault(key, 0)
    return out


def dashboard_timeseries(metric: str = "mae_m", days: int = 30) -> dict:
    rows = db.query_all(
        "SELECT substr(created_at, 1, 10) AS day, "
        "AVG(abs_error_m) AS mae, COUNT(*) AS n FROM ranging_result "
        "WHERE abs_error_m IS NOT NULL AND created_at >= date('now', ?) "
        "GROUP BY day ORDER BY day", (f"-{int(days)} day",))
    return {"metric": metric,
            "categories": [r["day"] for r in rows],
            "series": [{"name": "平均绝对误差(m)", "data": [round(float(r["mae"]), 4) for r in rows]},
                       {"name": "样本数", "data": [r["n"] for r in rows]}]}


def dashboard_gallery(kind: str = "outlier", limit: int = 12) -> list[dict]:
    if kind == "outlier":
        stats = summary({})
        return stats.get("outliers", [])[:limit]
    rows = db.query_all(
        "SELECT id, frame_uri, boxes, num_boxes, created_at FROM detection_result "
        "WHERE frame_uri IS NOT NULL ORDER BY created_at DESC LIMIT ?", (limit,))
    return db.decode_rows(rows)


def experiment_leaderboard(task_type: str | None = None, limit: int = 15) -> list[dict]:
    filters = {"task_type": task_type} if task_type else {}
    records = _fetch_all(filters)
    board = _leaderboard(records, top=limit)
    if task_type == "segmentation" or task_type is None:
        seg_rows = db.query_all(
            "SELECT COALESCE(m.name, 'segmentation') AS model_name, "
            "AVG(sr.miou) AS miou, AVG(sr.dice) AS dice, COUNT(*) AS n "
            "FROM segmentation_result sr "
            "LEFT JOIN experiment e ON e.id = sr.experiment_id "
            "LEFT JOIN model m ON m.id = e.model_id "
            "WHERE sr.miou IS NOT NULL GROUP BY COALESCE(m.name, 'segmentation') "
            "ORDER BY miou DESC LIMIT ?", (limit,))
        for row in seg_rows:
            board.append({"key": row["model_name"] or "segmentation",
                          "count": row["n"], "miou": round(float(row["miou"]), 4),
                          "dice": round(float(row["dice"] or 0), 4),
                          "task_type": "segmentation", "mae_m": None, "mre": None})
    return board


# ================================================================== 历史数据迁移
def import_legacy_measurements(csv_path: str, project_id: str | None = None,
                               experiment_id: str | None = None,
                               source_tag: str = "legacy_project_a") -> dict:
    """把项目 A 的 measurements.csv 迁移进 ranging_result。"""
    path = Path(csv_path)
    if not path.exists():
        raise ValidationError(f"CSV 不存在: {csv_path}")

    if experiment_id is None:
        experiment_id = _ensure_etl_experiment(project_id, "detection",
                                               "历史数据迁移 · 目标检测边界测距（项目 A）")
    existing = int(db.query_scalar("SELECT COUNT(*) FROM ranging_result WHERE experiment_id=?",
                                    (experiment_id,), 0) or 0)
    if existing:
        log.info("该迁移实验已有 %d 条记录，跳过重复导入", existing)
        return {"imported": 0, "skipped": existing, "experiment_id": experiment_id}

    imported, skipped = 0, 0
    with path.open("r", encoding="utf-8-sig", errors="ignore") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                detected = _to_float(row.get("detected_distance_m"))
                actual = _to_float(row.get("actual_distance_m"))
            except Exception:  # noqa: BLE001
                skipped += 1
                continue
            category = (row.get("category") or "").strip() or None
            error = _to_float(row.get("error_m"))
            if error is None and (actual is not None and detected is not None):
                error = detected - actual
            plan = _nearest_plan(actual if actual is not None else detected)
            model_name = (row.get("model") or "").strip() or None
            # 缺检记录（原始 CSV 中 detected 为空）同样入库，用于统计缺检率
            is_miss = detected is None
            db.insert("ranging_result", {
                "experiment_id": experiment_id,
                "model_name": model_name,
                "task_type": "detection",
                "strategy": "box_boundary",
                "target_kind": "object",
                "metric_kind": "center",
                "category": category,
                "size_group": settings.class_group(category) if category else None,
                "depth_m": detected,
                "raw_depth_m": detected,
                "distance_actual_m": actual,
                "actual_source": "manual",
                "error_m": error,
                "abs_error_m": abs(error) if error is not None else None,
                "rel_error": (abs(error) / actual) if (error is not None and actual) else None,
                "distance_plan_m": plan,
                "illumination": "sunny",
                "media_uri": None,
                "record_time": row.get("timestamp") or db.now_iso(),
                "source": source_tag,
                "notes": ("缺检：原始记录无检测值；" if is_miss else "")
                         + f"原始模型: {row.get('model')} v{row.get('version')}{row.get('size')}",
            }, id_prefix="rr")
            if is_miss:
                skipped += 0
            imported += 1

    _finalize_etl_experiment(experiment_id, imported)
    return {"imported": imported, "skipped": skipped, "experiment_id": experiment_id,
            "including_miss": True}


def import_legacy_distance_records(csv_path: str, project_id: str | None = None,
                                   source_tag: str = "legacy_project_b") -> dict:
    """把项目 B 的 distance_records.csv 迁移进 ranging_result（min/max 各一条）。"""
    path = Path(csv_path)
    if not path.exists():
        raise ValidationError(f"CSV 不存在: {csv_path}")

    experiment_id = _ensure_etl_experiment(project_id, "segmentation",
                                           "历史数据迁移 · 道路边界测距（项目 B）")
    existing = int(db.query_scalar("SELECT COUNT(*) FROM ranging_result WHERE experiment_id=?",
                                    (experiment_id,), 0) or 0)
    if existing:
        return {"imported": 0, "skipped": existing, "experiment_id": experiment_id}

    imported = 0
    with path.open("r", encoding="utf-8-sig", errors="ignore") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for kind, pred_key, actual_key, err_key in (
                    ("min", "min_distance_m", "actual_min_distance_m", "min_error_m"),
                    ("max", "max_distance_m", "actual_max_distance_m", "max_error_m")):
                detected = _to_float(row.get(pred_key))
                if detected is None:
                    continue
                actual = _to_float(row.get(actual_key))
                error = _to_float(row.get(err_key))
                if error is None and actual is not None:
                    error = detected - actual
                db.insert("ranging_result", {
                    "experiment_id": experiment_id,
                    "model_name": "legacy_unet_family",
                    "task_type": "segmentation",
                    "strategy": "mask_boundary",
                    "target_kind": "road_left" if kind == "min" else "road_right",
                    "metric_kind": kind,
                    "depth_m": detected, "raw_depth_m": detected,
                    "valid_points": int(_to_float(row.get("valid_points_count")) or 0),
                    "distance_actual_m": actual,
                    "actual_source": "manual",
                    "error_m": error,
                    "abs_error_m": abs(error) if error is not None else None,
                    "rel_error": (abs(error) / actual) if (error is not None and actual) else None,
                    "distance_plan_m": _nearest_plan(detected),
                    "illumination": "sunny",
                    "record_time": row.get("record_time") or db.now_iso(),
                    "source": source_tag,
                    "notes": f"原始记录: {row.get('record_time') or ''}",
                }, id_prefix="rr")
                imported += 1

    _finalize_etl_experiment(experiment_id, imported)
    return {"imported": imported, "experiment_id": experiment_id}


def _to_float(value: Any) -> float | None:
    if value in (None, "", "nan", "None"):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _nearest_plan(value: float | None) -> float | None:
    plan = list(settings.get("ranging.distance_plan_m", []) or [])
    if value is None or not plan:
        return None
    return float(min(plan, key=lambda p: abs(float(p) - float(value))))


def _ensure_etl_experiment(project_id: str | None, task_type: str, name: str) -> str:
    from services import project_service

    if project_id is None:
        row = db.query_one("SELECT id FROM project WHERE is_deleted=0 ORDER BY created_at LIMIT 1")
        if row:
            project_id = row["id"]
        else:
            project_id = project_service.create_project(
                "历史数据归档", "承载两个原项目迁移进来的实验记录",
                scene="campus_road", app_target="sweeper")["id"]
    existing = db.query_one("SELECT id FROM experiment WHERE name=? AND is_deleted=0", (name,))
    if existing:
        return existing["id"]
    experiment = project_service.create_experiment(
        project_id=project_id, name=name, task_type=task_type, kind="evaluate",
        config_snapshot={"note": "由历史数据迁移脚本创建"})
    return experiment["id"]


def _finalize_etl_experiment(experiment_id: str, imported: int) -> None:
    rows = db.query_all(
        "SELECT COUNT(*) AS n, AVG(abs_error_m) AS mae, "
        "AVG(rel_error) AS mre, MAX(abs_error_m) AS mx FROM ranging_result WHERE experiment_id=?",
        (experiment_id,))
    metrics = {
        "ranging_records": imported,
        "mae_m": round(float(rows[0]["mae"]), 4) if rows and rows[0]["mae"] is not None else None,
        "mre": round(float(rows[0]["mre"]), 4) if rows and rows[0]["mre"] is not None else None,
        "max_error_m": round(float(rows[0]["mx"]), 4) if rows and rows[0]["mx"] is not None else None,
    }
    db.update("experiment", experiment_id, {
        "status": "done", "metrics": metrics, "progress": 100.0,
        "finished_at": db.now_iso(),
    })


def legacy_summary(experiment_id: str) -> dict:
    """对照《申报书》口径复算历史数据（用于证明平台能承接既有成果）。"""
    rows = db.query_all(
        "SELECT COUNT(*) AS total, "
        "SUM(CASE WHEN depth_m IS NULL OR depth_m = 0 THEN 1 ELSE 0 END) AS missing, "
        "COUNT(depth_m) AS valid, "
        "AVG(abs_error_m) AS mae, "
        "MAX(abs_error_m) AS mx, "
        "AVG(rel_error) AS mre "
        "FROM ranging_result WHERE experiment_id=?", (experiment_id,))
    row = rows[0] if rows else {}
    total = int(row.get("total") or 0)
    missing = int(row.get("missing") or 0)
    valid = total - missing
    median = db.query_scalar(
        "SELECT abs_error_m FROM ranging_result WHERE experiment_id=? AND abs_error_m IS NOT NULL "
        "ORDER BY abs_error_m LIMIT 1 OFFSET ?", (experiment_id, max(0, valid // 2 - 1)), None)
    return {
        "total_records": total,
        "missing_records": missing,
        "valid_records": valid,
        "miss_rate": round(missing / total, 4) if total else None,
        "completion_rate": round(valid / total, 4) if total else None,
        "mean_error_m": round(float(row["mae"]), 4) if row.get("mae") is not None else None,
        "mean_rel_error": round(float(row["mre"]), 4) if row.get("mre") is not None else None,
        "median_error_m": round(float(median), 4) if median is not None else None,
        "max_error_m": round(float(row["mx"]), 4) if row.get("mx") is not None else None,
    }
