"""项目与实验管理：项目 CRUD、实验状态机、复现清单（run_manifest）、任务队列。"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from core import db, storage
from core.config import ROOT, settings
from core.exceptions import ConflictError, NotFoundError, ValidationError
from core.logging_setup import get_logger

log = get_logger("services.project")

TASK_TYPES = ("detection", "segmentation", "ranging", "joint")
EXPERIMENT_KINDS = ("train", "infer", "evaluate", "capture")
STATUS_FLOW = ("draft", "configured", "queued", "running", "paused",
               "done", "failed", "canceled", "archived")
EDITABLE_STATUS = ("draft", "configured")
RUNNABLE_STATUS = ("configured", "failed", "canceled")


# ================================================================== 项目
def create_project(name: str, description: str = "", scene: str = "campus_road",
                   app_target: str = "sweeper", owner_id: str | None = None,
                   config: dict | None = None) -> dict:
    name = str(name).strip()
    if not name:
        raise ValidationError("项目名称不能为空。")
    if db.query_one("SELECT id FROM project WHERE name=? AND is_deleted=0", (name,)):
        raise ConflictError(f"项目名称已存在: {name}")
    project_id = db.insert("project", {
        "name": name, "description": description, "scene": scene,
        "app_target": app_target, "owner_id": owner_id, "config": config or {},
        "status": "active",
    }, id_prefix="p")
    if owner_id:
        db.insert("project_member", {"project_id": project_id, "user_id": owner_id,
                                     "role_in_project": "owner"}, id_prefix="pm")
    log.info("创建项目: %s (%s)", name, project_id)
    return get_project(project_id) or {}


def get_project(project_id: str) -> dict | None:
    row = db.query_one("SELECT * FROM project WHERE id=? AND is_deleted=0", (project_id,))
    if row is None:
        return None
    project = db.decode_row(row) or {}
    project["stats"] = project_stats(project_id)
    return project


def project_stats(project_id: str) -> dict:
    return {
        "experiments": int(db.query_scalar(
            "SELECT COUNT(*) FROM experiment WHERE project_id=? AND is_deleted=0", (project_id,), 0) or 0),
        "datasets": int(db.query_scalar(
            "SELECT COUNT(*) FROM dataset WHERE project_id=? AND is_deleted=0", (project_id,), 0) or 0),
        "models": int(db.query_scalar(
            "SELECT COUNT(*) FROM model WHERE project_id=? AND is_deleted=0", (project_id,), 0) or 0),
        "ranging_records": int(db.query_scalar(
            "SELECT COUNT(*) FROM ranging_result r JOIN experiment e ON e.id=r.experiment_id "
            "WHERE e.project_id=?", (project_id,), 0) or 0),
    }


def list_projects(page: int = 1, page_size: int = 20, keyword: str = "") -> dict:
    where, params = "is_deleted=0", []
    if keyword:
        where += " AND (name LIKE ? OR description LIKE ?)"
        params += [f"%{keyword}%", f"%{keyword}%"]
    result = db.paginate(
        f"SELECT * FROM project WHERE {where} ORDER BY updated_at DESC",
        f"SELECT COUNT(*) FROM project WHERE {where}", params, page, page_size)
    for item in result["items"]:
        item["stats"] = project_stats(item["id"])
    return result


def update_project(project_id: str, **fields: Any) -> int:
    allowed = {"name", "description", "scene", "app_target", "status", "config"}
    payload = {k: v for k, v in fields.items() if k in allowed}
    if not get_project(project_id):
        raise NotFoundError("项目不存在。")
    return db.update("project", project_id, payload)


def archive_project(project_id: str) -> int:
    return db.update("project", project_id, {"status": "archived"})


def delete_project(project_id: str) -> int:
    return db.soft_delete("project", project_id)


def add_member(project_id: str, user_id: str, role_in_project: str = "developer") -> str:
    existing = db.query_one("SELECT id FROM project_member WHERE project_id=? AND user_id=?",
                            (project_id, user_id))
    if existing:
        db.update("project_member", existing["id"], {"role_in_project": role_in_project})
        return existing["id"]
    return db.insert("project_member", {"project_id": project_id, "user_id": user_id,
                                        "role_in_project": role_in_project,
                                        "joined_at": db.now_iso()}, id_prefix="pm")


def list_members(project_id: str) -> list[dict]:
    return db.decode_rows(db.query_all(
        "SELECT pm.*, u.username, u.display_name, u.role FROM project_member pm "
        "LEFT JOIN user u ON u.id = pm.user_id WHERE pm.project_id=? ORDER BY pm.joined_at",
        (project_id,)))


# ================================================================== 实验
def create_experiment(project_id: str, name: str, task_type: str, kind: str = "infer",
                      dataset_version_id: str | None = None, model_id: str | None = None,
                      calibration_id: str | None = None, camera_id: str | None = None,
                      config_snapshot: dict | None = None, created_by: str | None = None) -> dict:
    if not get_project(project_id):
        raise NotFoundError("项目不存在。")
    if task_type not in TASK_TYPES:
        raise ValidationError(f"未知任务类型: {task_type}，可选 {TASK_TYPES}")
    if kind not in EXPERIMENT_KINDS:
        raise ValidationError(f"未知实验类型: {kind}，可选 {EXPERIMENT_KINDS}")
    config_snapshot = dict(config_snapshot or {})
    config_snapshot.setdefault("camera", {
        "width": settings.get("camera.width"), "height": settings.get("camera.height"),
        "fps": settings.get("camera.fps"), "filters": settings.get("camera.filters"),
    })
    config_snapshot.setdefault("ranging", {
        "strategy": settings.get("ranging.default_strategy"),
        "kernel": settings.get("ranging.kernel"),
        "sample_points": settings.get("ranging.sample_points"),
        "max_range_m": settings.get("ranging.max_range_m"),
        "min_valid_points": settings.get("ranging.min_valid_points"),
        "road_split": settings.get("ranging.road_split"),
        "reproject": settings.get("ranging.reproject"),
        "corridor_ref_depth_m": settings.get("ranging.corridor_ref_depth_m"),
        "distance_plan_m": settings.get("ranging.distance_plan_m"),
    })
    config_snapshot.setdefault("detection", {
        "conf": settings.get("detection.default_conf"),
        "iou": settings.get("detection.default_iou"),
        "imgsz": settings.get("detection.default_imgsz"),
        "weight_mode": settings.get("detection.weight_mode"),
        "classes": [],
    })
    config_snapshot.setdefault("segmentation", {
        "threshold": settings.get("segmentation.default_threshold"),
        "input_width": settings.get("segmentation.input_width"),
        "input_height": settings.get("segmentation.input_height"),
    })
    experiment_id = db.insert("experiment", {
        "project_id": project_id, "name": str(name).strip() or f"实验 {db.now_iso()}",
        "task_type": task_type, "kind": kind, "status": "configured",
        "dataset_version_id": dataset_version_id, "model_id": model_id,
        "calibration_id": calibration_id, "camera_id": camera_id,
        "config_snapshot": config_snapshot, "created_by": created_by,
        "progress": 0.0,
    }, id_prefix="exp")
    return get_experiment(experiment_id) or {}


def get_experiment(experiment_id: str) -> dict | None:
    row = db.query_one("SELECT * FROM experiment WHERE id=? AND is_deleted=0", (experiment_id,))
    if row is None:
        return None
    exp = db.decode_row(row) or {}
    if exp.get("model_id"):
        exp["model"] = db.decode_row(db.query_one(
            "SELECT id,name,task_type,arch,variant,weights_uri,metrics FROM model WHERE id=?",
            (exp["model_id"],)))
    if exp.get("dataset_version_id"):
        exp["dataset_version"] = db.decode_row(db.query_one(
            "SELECT dv.*, d.name AS dataset_name, d.task_type FROM dataset_version dv "
            "LEFT JOIN dataset d ON d.id=dv.dataset_id WHERE dv.id=?",
            (exp["dataset_version_id"],)))
    exp["counts"] = {
        "detection": int(db.query_scalar("SELECT COUNT(*) FROM detection_result WHERE experiment_id=?",
                                          (experiment_id,), 0) or 0),
        "segmentation": int(db.query_scalar("SELECT COUNT(*) FROM segmentation_result WHERE experiment_id=?",
                                             (experiment_id,), 0) or 0),
        "ranging": int(db.query_scalar("SELECT COUNT(*) FROM ranging_result WHERE experiment_id=?",
                                        (experiment_id,), 0) or 0),
    }
    return exp


def list_experiments(page: int = 1, page_size: int = 20, project_id: str | None = None,
                     task_type: str | None = None, status: str | None = None,
                     kind: str | None = None) -> dict:
    clauses, params = ["is_deleted=0"], []
    if project_id:
        clauses.append("project_id=?")
        params.append(project_id)
    if task_type:
        clauses.append("task_type=?")
        params.append(task_type)
    if status:
        clauses.append("status=?")
        params.append(status)
    if kind:
        clauses.append("kind=?")
        params.append(kind)
    where = " AND ".join(clauses)
    result = db.paginate(
        f"SELECT * FROM experiment WHERE {where} ORDER BY created_at DESC",
        f"SELECT COUNT(*) FROM experiment WHERE {where}", params, page, page_size)
    for item in result["items"]:
        item["metrics"] = db.decode_row({"metrics": item.get("metrics")}).get("metrics", {})
    return result


def update_experiment_config(experiment_id: str, config_snapshot: dict) -> int:
    exp = get_experiment(experiment_id)
    if exp is None:
        raise NotFoundError("实验不存在。")
    if exp["status"] not in EDITABLE_STATUS:
        raise ConflictError(f"实验处于 {exp['status']} 状态，配置不可修改（请先克隆一份）。")
    merged = dict(exp.get("config_snapshot") or {})
    for key, value in (config_snapshot or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return db.update("experiment", experiment_id, {"config_snapshot": merged})


def set_status(experiment_id: str, status: str, **extra: Any) -> int:
    if status not in STATUS_FLOW:
        raise ValidationError(f"未知状态: {status}")
    payload: dict[str, Any] = {"status": status, **extra}
    if status == "running":
        payload.setdefault("started_at", db.now_iso())
    if status in ("done", "failed", "canceled"):
        payload.setdefault("finished_at", db.now_iso())
    return db.update("experiment", experiment_id, payload)


def archive_experiment(experiment_id: str) -> int:
    exp = get_experiment(experiment_id)
    if exp is None:
        raise NotFoundError("实验不存在。")
    if exp["status"] != "done":
        raise ConflictError("只有已完成的实验可以归档。")
    return db.update("experiment", experiment_id, {"status": "archived"})


def clone_experiment(experiment_id: str, name: str | None = None,
                     created_by: str | None = None) -> dict:
    exp = get_experiment(experiment_id)
    if exp is None:
        raise NotFoundError("实验不存在。")
    return create_experiment(
        project_id=exp["project_id"],
        name=name or f"{exp['name']} 副本",
        task_type=exp["task_type"], kind=exp["kind"],
        dataset_version_id=exp.get("dataset_version_id"), model_id=exp.get("model_id"),
        calibration_id=exp.get("calibration_id"), camera_id=exp.get("camera_id"),
        config_snapshot=dict(exp.get("config_snapshot") or {}), created_by=created_by,
    )


def delete_experiment(experiment_id: str) -> int:
    return db.soft_delete("experiment", experiment_id)


def compare_experiments(ids: list[str]) -> dict:
    items = []
    for eid in ids:
        exp = get_experiment(eid)
        if exp:
            items.append({
                "id": exp["id"], "name": exp["name"], "task_type": exp["task_type"],
                "status": exp["status"], "model_id": exp.get("model_id"),
                "model_name": (exp.get("model") or {}).get("name"),
                "metrics": exp.get("metrics") or {}, "counts": exp.get("counts") or {},
                "created_at": exp.get("created_at"),
            })
    dimensions = sorted({k for item in items for k in (item["metrics"] or {})})
    return {"items": items, "dimensions": dimensions}


# ================================================================== 复现清单
def _git_info() -> dict:
    def run(args: list[str]) -> str:
        try:
            out = subprocess.run(args, cwd=str(ROOT), capture_output=True, text=True, timeout=5)
            return out.stdout.strip()
        except Exception:  # noqa: BLE001
            return ""

    commit = run(["git", "rev-parse", "--short", "HEAD"])
    dirty = run(["git", "status", "--porcelain"])
    return {"git_commit": commit or None, "git_dirty": bool(dirty)}


def _env_info() -> dict:
    import importlib.metadata as md

    def version(name: str) -> str | None:
        try:
            return md.version(name)
        except Exception:  # noqa: BLE001
            return None

    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": version("torch"),
        "ultralytics": version("ultralytics"),
        "opencv-python": version("opencv-python"),
        "pyrealsense2": version("pyrealsense2"),
        "numpy": version("numpy"),
        "cuda_available": False,
        "gpu": None,
    }
    try:
        import torch

        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
    except Exception:  # noqa: BLE001
        pass
    return info


def build_run_manifest(experiment_id: str, camera_info: dict | None = None) -> dict:
    """生成实验复现清单：代码 / 环境 / 硬件 / 数据 / 模型 / 超参。"""
    exp = get_experiment(experiment_id)
    if exp is None:
        raise NotFoundError("实验不存在。")

    dataset_info = None
    if exp.get("dataset_version_id"):
        dv = db.decode_row(db.query_one("SELECT * FROM dataset_version WHERE id=?",
                                        (exp["dataset_version_id"],)))
        if dv:
            dataset_info = {
                "dataset_id": dv["dataset_id"], "version": dv["version"],
                "content_hash": dv.get("content_hash"), "split_seed": dv.get("split_seed"),
                "image_count": dv.get("image_count"),
            }

    model_info = None
    if exp.get("model_id"):
        row = db.decode_row(db.query_one("SELECT * FROM model WHERE id=?", (exp["model_id"],)))
        if row:
            model_info = {
                "model_id": row["id"], "name": row["name"], "arch": row["arch"],
                "task_type": row["task_type"], "weights_uri": row.get("weights_uri"),
                "weights_sha256": row.get("weights_sha256"),
                "weight_origin": row.get("weight_origin"),
            }

    calibration_info = None
    if exp.get("calibration_id"):
        row = db.decode_row(db.query_one("SELECT * FROM calibration WHERE id=?",
                                          (exp["calibration_id"],)))
        if row:
            calibration_info = {"id": row["id"], "name": row["name"],
                                "file_sha256": row.get("file_sha256"),
                                "rms_reprojection_px": row.get("rms_reprojection_px")}

    manifest = {
        "experiment_id": experiment_id,
        "project_id": exp.get("project_id"),
        "task_type": exp.get("task_type"),
        "kind": exp.get("kind"),
        "created_at": exp.get("created_at"),
        "operator": exp.get("created_by"),
        "code": {"version": settings.get("app.version"), **_git_info()},
        "env": _env_info(),
        "hardware": {
            "camera": camera_info or {},
            "calibration": calibration_info,
        },
        "dataset": dataset_info,
        "model": model_info,
        "hyperparams": exp.get("config_snapshot") or {},
        "seed": int(settings.get("training.seed", 42)),
    }
    db.update("experiment", experiment_id, {"run_manifest": manifest})
    return manifest


def verify_reproduction(experiment_id: str) -> dict:
    """逐项比对当前环境/数据/权重与清单记录，输出差异清单。"""
    exp = get_experiment(experiment_id)
    if exp is None:
        raise NotFoundError("实验不存在。")
    recorded = exp.get("run_manifest") or {}
    if not recorded:
        return {"ok": False, "differences": [{"item": "run_manifest", "detail": "该实验尚未生成复现清单"}]}

    diffs: list[dict] = []
    current_env = _env_info()
    for key in ("python", "torch", "ultralytics", "opencv-python", "numpy"):
        old = (recorded.get("env") or {}).get(key)
        new = current_env.get(key)
        if old and new and old != new:
            diffs.append({"item": f"env.{key}", "recorded": old, "current": new})

    recorded_code = (recorded.get("code") or {}).get("git_commit")
    current_code = _git_info().get("git_commit")
    if recorded_code and current_code and recorded_code != current_code:
        diffs.append({"item": "code.git_commit", "recorded": recorded_code, "current": current_code})

    rec_model = (recorded.get("model") or {}).get("weights_sha256")
    if rec_model and exp.get("model_id"):
        row = db.query_one("SELECT weights_sha256 FROM model WHERE id=?", (exp["model_id"],))
        current = row["weights_sha256"] if row else None
        if current and current != rec_model:
            diffs.append({"item": "model.weights_sha256", "recorded": rec_model, "current": current})

    rec_dataset = (recorded.get("dataset") or {}).get("content_hash")
    if rec_dataset and exp.get("dataset_version_id"):
        row = db.query_one("SELECT content_hash FROM dataset_version WHERE id=?",
                           (exp["dataset_version_id"],))
        current = row["content_hash"] if row else None
        if current and current != rec_dataset:
            diffs.append({"item": "dataset.content_hash", "recorded": rec_dataset, "current": current})

    return {"ok": not diffs, "differences": diffs,
            "checked_at": db.now_iso(), "manifest": recorded}


# ================================================================== 任务队列
def enqueue(experiment_id: str, task_type: str, payload: dict | None = None,
            priority: int = 5) -> dict:
    job_id = db.insert("task_queue", {
        "experiment_id": experiment_id, "task_type": task_type,
        "payload": payload or {}, "priority": int(priority), "status": "queued",
    }, id_prefix="tq")
    return get_task(job_id) or {}


def get_task(task_id: str) -> dict | None:
    return db.decode_row(db.query_one("SELECT * FROM task_queue WHERE id=?", (task_id,)))


def list_tasks(status: str | None = None, limit: int = 100) -> list[dict]:
    if status:
        rows = db.query_all("SELECT * FROM task_queue WHERE status=? "
                            "ORDER BY priority, enqueued_at DESC LIMIT ?", (status, limit))
    else:
        rows = db.query_all("SELECT * FROM task_queue ORDER BY enqueued_at DESC LIMIT ?", (limit,))
    return db.decode_rows(rows)


def update_task(task_id: str, **fields: Any) -> int:
    allowed = {"status", "worker_id", "gpu_index", "started_at", "finished_at",
               "retry_count", "error", "payload"}
    return db.update("task_queue", task_id, {k: v for k, v in fields.items() if k in allowed})


def task_summary() -> dict:
    rows = db.query_all("SELECT status, COUNT(*) AS n FROM task_queue GROUP BY status")
    summary = {r["status"]: r["n"] for r in rows}
    for key in ("queued", "running", "done", "failed", "canceled"):
        summary.setdefault(key, 0)
    return summary
