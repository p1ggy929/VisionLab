"""误差分析、校正与可视化大屏 API。"""

from __future__ import annotations

from flask import Blueprint

from core.config import settings
from core.exceptions import NotFoundError
from services import analysis_service

from .common import (arg, arg_int, current_user, ok, page_args, paginated, payload,
                     require_perm)

bp = Blueprint("analysis", __name__, url_prefix="/api/v1")


def _filters() -> dict:
    filters = {
        "task_type": arg("task_type"), "strategy": arg("strategy"),
        "model_name": arg("model_name"), "category": arg("category"),
        "size_group": arg("size_group"), "illumination": arg("illumination"),
        "experiment_id": arg("experiment_id"), "target_kind": arg("target_kind"),
        "metric_kind": arg("metric_kind"), "source": arg("source"),
        "start": arg("start"), "end": arg("end"),
    }
    if arg("distance_plan_m"):
        filters["distance_plan_m"] = arg("distance_plan_m")
    return {k: v for k, v in filters.items() if v not in (None, "")}


@bp.get("/analysis/summary")
def summary():
    data = analysis_service.summary(_filters(), save=arg("save") in ("1", "true"),
                                    experiment_id=arg("experiment_id"))
    return ok(data)


@bp.get("/analysis/group")
def group():
    stats = analysis_service.summary(_filters())
    dimension = arg("by", "category")
    key = f"by_{dimension}"
    return ok({"dimension": dimension, "items": stats.get(key, [])})


@bp.get("/analysis/outliers")
def outliers():
    stats = analysis_service.summary(_filters())
    top = arg_int("top", 20)
    return ok({"items": (stats.get("outliers") or [])[:top]})


@bp.get("/analysis/heatmap")
def heatmap():
    stats = analysis_service.summary(_filters())
    return ok(stats.get("heatmap") or {"x": [], "y": [], "z": []})


@bp.get("/analysis/leaderboard")
def leaderboard():
    return ok({"items": analysis_service.experiment_leaderboard(arg("task_type"),
                                                                arg_int("limit", 15))})


@bp.get("/analysis/distributions")
def distributions():
    """距离—误差曲线的原始数据（含 1–20 m 标准距离点）。"""
    stats = analysis_service.summary(_filters())
    return ok({"by_distance": stats.get("by_distance", []),
               "by_size": stats.get("by_size", []),
               "by_category": stats.get("by_category", []),
               "by_illumination": stats.get("by_illumination", []),
               "by_model": stats.get("by_model", []),
               "by_task": stats.get("by_task", []),
               "plan": settings.get("ranging.distance_plan_m", [])})


@bp.get("/analysis/records")
def records():
    page, size = page_args()
    return paginated(analysis_service.query_records(_filters(), page, size,
                                                    order=f"{arg('sort', 'created_at')} DESC"))


@bp.get("/analysis/analyses")
def list_analyses():
    return ok({"items": analysis_service.list_analyses()})


@bp.get("/analysis/analyses/<analysis_id>")
def get_analysis(analysis_id: str):
    data = analysis_service.get_analysis(analysis_id)
    if not data:
        raise NotFoundError("分析记录不存在。")
    return ok(data)


# ================================================================== 校正
@bp.post("/analysis/calibrate")
@require_perm("analysis:calibrate")
def calibrate():
    data = payload()
    filters = dict(_filters())
    for key in ("task_type", "strategy", "model_name", "category", "size_group", "illumination"):
        if data.get(key):
            filters[key] = data[key]
    if data.get("experiment_id"):
        filters["experiment_id"] = data["experiment_id"]
    result = analysis_service.calibrate_from_records(
        method=str(data.get("method", "linear")), filters=filters,
        bounds=data.get("boundaries_m"), name=data.get("name"),
        experiment_id=data.get("experiment_id"))
    return ok(result)


@bp.get("/analysis/calibrations")
def list_calibrations():
    return ok({"items": analysis_service.list_calibration_plans(),
               "active": analysis_service.active_calibration().to_dict()
               if analysis_service.active_calibration() else None,
               "methods": list(settings.get("analysis.calibration.methods", []))})


@bp.post("/analysis/calibrations/<plan_id>/activate")
@require_perm("analysis:calibrate")
def activate(plan_id: str):
    from algorithms.calibration import is_recommended
    from services import pipeline_service

    row = analysis_service.activate_calibration(plan_id)
    plan = analysis_service.build_plan(row)
    pipeline_service.get_pipeline().set_calibration(plan)
    payload = dict(row)
    payload["recommended"] = is_recommended(plan) if plan else False
    return ok(payload)


@bp.post("/analysis/calibrations/deactivate")
@require_perm("analysis:calibrate")
def deactivate():
    analysis_service.deactivate_calibration()
    from services import pipeline_service

    pipeline_service.get_pipeline().set_calibration(None)
    return ok({"active": None})


@bp.get("/analysis/export")
def export_summary():
    data = analysis_service.summary(_filters())
    return ok(data)


# ================================================================== 大屏
@bp.get("/dashboard/kpi")
def dashboard_kpi():
    return ok(analysis_service.dashboard_kpi())


@bp.get("/dashboard/timeseries")
def dashboard_timeseries():
    return ok(analysis_service.dashboard_timeseries(arg("metric", "mae_m"),
                                                    arg_int("days", 30)))


@bp.get("/dashboard/gallery")
def dashboard_gallery():
    return ok({"items": analysis_service.dashboard_gallery(arg("kind", "outlier"),
                                                            arg_int("limit", 12))})


@bp.get("/dashboard/overview")
def dashboard_overview():
    """大屏一次性拉取所需全部数据。"""
    stats = analysis_service.summary({})
    return ok({
        "kpi": analysis_service.dashboard_kpi(),
        "leaderboard": analysis_service.experiment_leaderboard(None, 12),
        "by_distance": stats.get("by_distance", []),
        "by_category": stats.get("by_category", []),
        "by_size": stats.get("by_size", []),
        "by_illumination": stats.get("by_illumination", []),
        "heatmap": stats.get("heatmap", {}),
        "outliers": stats.get("outliers", [])[:12],
        "mae_m": stats.get("mae_m"), "rmse_m": stats.get("rmse_m"),
        "hit_rate": stats.get("hit_rate"),
        "plan": settings.get("ranging.distance_plan_m", []),
    })


# ================================================================== 历史迁移
@bp.post("/analysis/import-legacy")
@require_perm("dataset:import")
def import_legacy():
    from services import analysis_service as svc
    from services import dataset_service, model_registry

    data = payload()
    results: dict = {}
    project_id = data.get("project_id")
    if data.get("measurements_csv"):
        results["measurements"] = svc.import_legacy_measurements(
            data["measurements_csv"], project_id, source_tag="legacy_project_a")
    if data.get("distance_records_csv"):
        results["distance_records"] = svc.import_legacy_distance_records(
            data["distance_records_csv"], project_id, source_tag="legacy_project_b")
    if data.get("runs_dir"):
        results["yolo_models"] = model_registry.scan_yolo_runs(data["runs_dir"], project_id)
    if data.get("seg_models_dir"):
        results["seg_models"] = model_registry.scan_segmentation_weights(
            data["seg_models_dir"], project_id, data.get("best_model_dir"))
    if data.get("seg_root"):
        results["seg_dataset"] = dataset_service.import_legacy_segmentation_full(
            data["seg_root"])
    elif data.get("seg_images_dir") and data.get("seg_masks_dir"):
        results["seg_dataset"] = dataset_service.import_legacy_segmentation_dataset(
            data["seg_images_dir"], data["seg_masks_dir"])
    if data.get("det_images_dir"):
        from pathlib import Path as _Path

        images_root = str(data["det_images_dir"])
        if any((_Path(images_root) / s).exists() for s in ("train", "val", "test")):
            results["det_dataset"] = dataset_service.import_legacy_detection_full(
                images_root, data.get("det_labels_dir"))
        else:
            results["det_dataset"] = dataset_service.import_legacy_detection_dataset(
                images_root, limit=data.get("det_limit"))
    return ok(results)


@bp.get("/analysis/legacy-summary")
def legacy_summary():
    experiment_id = arg("experiment_id")
    if not experiment_id:
        raise NotFoundError("请提供 experiment_id。")
    return ok(analysis_service.legacy_summary(experiment_id))
