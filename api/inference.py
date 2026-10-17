"""推理、实时流水线与测距记录 API。"""

from __future__ import annotations

import io
import time

from flask import Blueprint, Response, request

from algorithms.ranging import STRATEGIES
from core import db, media
from core.config import settings
from core.exceptions import NotFoundError, ValidationError
from services import analysis_service, pipeline_service, project_service

from .common import (arg, arg_int, current_user, ok, page_args, paginated, payload,
                     require_perm)

bp = Blueprint("inference", __name__, url_prefix="/api/v1")

# 允许运行时热更新的流水线参数（与 PipelineService.configure 的形参保持一致）
_PIPELINE_PARAM_KEYS = (
    "sample_points", "kernel", "max_range_m", "road_split", "reproject",
    "corridor_ref_depth_m", "threshold", "conf", "mode", "classes",
)

# 测距记录导出时支持的过滤字段
_RECORD_FILTER_KEYS = (
    "task_type", "strategy", "model_name", "category",
    "size_group", "illumination", "experiment_id",
)


def _pipeline_updates(data: dict) -> dict:
    """从请求体中挑出流水线可热更新的参数。"""
    updates = {}
    for key in _PIPELINE_PARAM_KEYS:
        if key in data:
            updates[key] = data[key]
    return updates


def _record_filters() -> dict:
    """收集导出接口的查询过滤条件（空值不参与过滤）。"""
    filters = {}
    for key in _RECORD_FILTER_KEYS:
        value = arg(key)
        if value:
            filters[key] = value
    return filters


# ================================================================== 流水线
@bp.get("/pipeline/config")
def get_config():
    pipeline = pipeline_service.get_pipeline()
    return ok({
        "config": pipeline.config,
        "options": {
            "modes": [
                {"key": "detection", "label": "目标检测实验"},
                {"key": "segmentation", "label": "道路语义分割实验"},
                {"key": "both", "label": "双策略同跑（推荐）"},
            ],
            "ranging_strategies": [{"key": k, "label": v.name} for k, v in STRATEGIES.items()],
            "classes": settings.detection_class_names(),
            "class_groups": {name: settings.class_group(name)
                             for name in settings.detection_class_names()},
            "distance_plan_m": settings.get("ranging.distance_plan_m", []),
        },
    })


@bp.post("/pipeline/config")
@require_perm("inference:run")
def set_config():
    pipeline = pipeline_service.get_pipeline()
    config = pipeline.configure(**payload())
    return ok({"config": config, "load_errors": config.get("load_errors", [])})


@bp.get("/pipeline/status")
def pipeline_status():
    return ok(pipeline_service.get_pipeline().status())


@bp.post("/pipeline/process")
@require_perm("inference:run")
def process_frame():
    """处理当前帧：返回结构化结果 + 叠加图（base64）。"""
    import base64

    bundle, truth = pipeline_service.current_frame_bundle()
    if bundle is None:
        raise ValidationError("没有可用画面（请先启动相机或合成场景）。")
    result = pipeline_service.get_pipeline().process(bundle, truth)
    overlay = pipeline_service.get_pipeline().overlay_frame()
    if overlay is not None:
        view = media.resize_keep_ratio(overlay, int(settings.get("server.stream.max_width", 960)), 10000)
        result["overlay_base64"] = base64.b64encode(
            media.encode_jpeg(view, int(settings.get("server.stream.jpeg_quality", 75)))).decode()
    depth_view = media.colorize_depth(bundle.depth_m, float(settings.get("ranging.min_range_m", 0.3)), 15.0)
    depth_view = media.resize_keep_ratio(depth_view, int(settings.get("server.stream.max_width", 960)), 10000)
    result["depth_base64"] = base64.b64encode(
        media.encode_jpeg(depth_view, int(settings.get("server.stream.jpeg_quality", 75)))).decode()
    result["ground_truth"] = truth
    return ok(result)


@bp.get("/pipeline/frame")
def pipeline_frame():
    """叠加图的 JPEG 流（用于前端 <img> 直接刷新，无需 base64）。"""
    bundle, truth = pipeline_service.current_frame_bundle()
    if bundle is None:
        return Response(b"", mimetype="image/jpeg")
    pipeline = pipeline_service.get_pipeline()
    pipeline.process(bundle, truth)
    overlay = pipeline.overlay_frame()
    if overlay is None:
        overlay = bundle.color
    view = media.resize_keep_ratio(overlay, int(settings.get("server.stream.max_width", 960)), 10000)
    return Response(media.encode_jpeg(view, int(settings.get("server.stream.jpeg_quality", 75))),
                    mimetype="image/jpeg", headers={"Cache-Control": "no-store"})


@bp.post("/pipeline/session/start")
@require_perm("experiment:run")
def start_session():
    data = payload()
    experiment_id = data.get("experiment_id")
    if not experiment_id:
        project_id = data.get("project_id")
        if not project_id:
            rows = project_service.list_projects(1, 1)["items"]
            if not rows:
                project = project_service.create_project(project_service.ProjectSpec(
                    "默认项目", "平台自动创建", scene="campus_road", app_target="sweeper"))
                project_id = project["id"]
            else:
                project_id = rows[0]["id"]
        experiment = project_service.create_experiment(project_service.ExperimentSpec(
            project_id=project_id,
            name=str(data.get("name", f"实时测距实验 {db.now_iso()}")),
            task_type=str(data.get("task_type", "joint")), kind="infer",
            model_id=data.get("model_id"), dataset_version_id=data.get("dataset_version_id"),
            calibration_id=data.get("calibration_id"),
            config_snapshot={"ranging": pipeline_service.get_pipeline().config},
            created_by=(current_user() or {}).get("username")))
        experiment_id = experiment["id"]
    project_id = data.get("project_id") or (
        project_service.get_experiment(experiment_id) or {}).get("project_id")
    session = pipeline_service.get_pipeline().start_session(experiment_id, project_id)
    return ok({"session": session, "experiment_id": experiment_id})


@bp.post("/pipeline/session/stop")
@require_perm("experiment:run")
def stop_session():
    pipeline = pipeline_service.get_pipeline()
    experiment_id = pipeline.session.get("experiment_id")
    summary = pipeline.stop_session()
    if experiment_id:
        project_service.build_run_manifest(experiment_id)
    return ok({"session": summary})


@bp.post("/pipeline/session/pause")
@require_perm("experiment:run")
def pause_session():
    return ok(pipeline_service.get_pipeline().toggle_pause(True))


@bp.post("/pipeline/session/resume")
@require_perm("experiment:run")
def resume_session():
    return ok(pipeline_service.get_pipeline().toggle_pause(False))


@bp.post("/pipeline/save-frame")
@require_perm("ranging:write")
def save_frame():
    bundle, truth = pipeline_service.current_frame_bundle()
    if bundle is None:
        raise ValidationError("没有可用画面。")
    pipeline = pipeline_service.get_pipeline()
    result = pipeline.last or pipeline.process(bundle, truth)
    created = pipeline.record_frame_result(bundle, result)
    return ok(created)


@bp.post("/pipeline/apply-calibration")
@require_perm("analysis:calibrate")
def apply_calibration():
    data = payload()
    pipeline = pipeline_service.get_pipeline()
    plan_id = data.get("plan_id")
    if not plan_id or data.get("disable"):
        analysis_service.deactivate_calibration()
        pipeline.set_calibration(None)
        return ok({"calibration": None})
    row = analysis_service.activate_calibration(str(plan_id))
    built = analysis_service.build_plan(row)
    pipeline.set_calibration(built)
    return ok(row)


# ================================================================== 测距
@bp.get("/ranging/strategies")
def ranging_strategies():
    return ok({"items": [
        {"key": "box_boundary", "label": "目标边界测距",
         "detail": "检测框中心局部中位数 + 框边界多点采样，输出最近/最远/平均距离与横向偏移",
         "source": "原项目 A"},
        {"key": "mask_boundary", "label": "道路边界测距",
         "detail": "分割掩膜最大外轮廓等距采样 + 按质心左右分离 + 针孔反投影，"
                   "输出左右边界距离与可通行走廊宽度",
         "source": "原项目 B + 平台新增左右分离与反投影"},
    ], "default": settings.get("ranging.default_strategy")})


@bp.post("/ranging/preview")
@require_perm("ranging:run")
def ranging_preview():
    """用当前参数对当前帧试算，不落库。"""
    data = payload()
    pipeline = pipeline_service.get_pipeline()
    pipeline.configure(**_pipeline_updates(data))
    bundle, truth = pipeline_service.current_frame_bundle()
    if bundle is None:
        raise ValidationError("没有可用画面。")
    result = pipeline.process(bundle, truth)
    return ok({"detection": result["detection"], "road": result["road"],
               "objects": result["objects"], "perf": result["perf"],
               "errors": result["errors"]})


@bp.get("/ranging/records")
def list_records():
    page, size = page_args()
    filters = {
        "task_type": arg("task_type"), "strategy": arg("strategy"), "model_name": arg("model_name"),
        "category": arg("category"), "size_group": arg("size_group"),
        "illumination": arg("illumination"), "experiment_id": arg("experiment_id"),
        "target_kind": arg("target_kind"), "metric_kind": arg("metric_kind"),
        "source": arg("source"), "start": arg("start"), "end": arg("end"),
        "has_actual": arg("has_actual") in ("1", "true"),
    }
    if arg("distance_plan_m"):
        filters["distance_plan_m"] = arg("distance_plan_m")
    if arg("category") and "," in str(arg("category")):
        filters["category"] = str(arg("category")).split(",")
    return paginated(analysis_service.query_records(filters, page, size))


@bp.post("/ranging/records")
@require_perm("ranging:write")
def create_record():
    data = payload()
    pipeline = pipeline_service.get_pipeline()
    result = data.get("ranging") or {}
    if not result and data.get("use_last"):
        last = pipeline.last or {}
        kind = data.get("target_kind", "object")
        metric = data.get("metric_kind", "min")
        if kind == "object":
            for item in last.get("objects", []):
                if data.get("category") and item["category"] != data["category"]:
                    continue
                result = item["ranging"].get(metric) or {}
                break
        else:
            result = (last.get("road", {}).get(kind.replace("road_", ""), {}) or {}).get(metric) or {}
    if not result:
        raise ValidationError("缺少测距结果数据（可传 ranging 或 use_last=true）。")

    record = pipeline.record_measurement(pipeline_service.MeasurementRecord(
        target_kind=str(data.get("target_kind", "object")),
        metric_kind=str(data.get("metric_kind", "min")),
        result=result,
        actual_value=data.get("actual_distance_m"),
        actual_source=str(data.get("actual_source", "tape")),
        category=data.get("category"),
        notes=str(data.get("notes", "")),
        illumination=data.get("illumination"),
        media_uri=data.get("media_uri"),
    ))
    return ok(record)


@bp.post("/ranging/records/bulk")
@require_perm("ranging:write")
def bulk_create_records():
    """一次性把当前帧的全部测距结果落库（双策略同跑时最常用）。"""
    data = payload()
    pipeline = pipeline_service.get_pipeline()
    last = pipeline.last
    if not last:
        bundle, truth = pipeline_service.current_frame_bundle()
        if bundle is None:
            raise ValidationError("没有可用画面。")
        last = pipeline.process(bundle, truth)

    actual_map: dict = data.get("actual") or {}
    created = []
    for item in last.get("objects", []):
        for metric in ("min", "max", "center"):
            result = item["ranging"].get(metric)
            if not result:
                continue
            category = item["category"]
            actual = actual_map.get(category)
            created.append(pipeline.record_measurement(pipeline_service.MeasurementRecord(
                target_kind="object", metric_kind=metric, result=result,
                actual_value=actual, category=category,
                illumination=data.get("illumination"),
                actual_source=str(data.get("actual_source", "tape")))))
    for side in ("left", "right", "center"):
        for metric in ("min", "max"):
            result = (last.get("road", {}).get(side, {}) or {}).get(metric)
            if not result:
                continue
            created.append(pipeline.record_measurement(pipeline_service.MeasurementRecord(
                target_kind=f"road_{side}" if side != "center" else "road_center",
                metric_kind=metric, result=result,
                actual_value=actual_map.get(f"road_{side}"),
                illumination=data.get("illumination"))))
    return ok({"created": len(created), "records": created[:50]})


@bp.patch("/ranging/records/<record_id>")
@require_perm("ranging:write")
def update_record(record_id: str):
    analysis_service.update_record(record_id, **payload())
    return ok(analysis_service.get_record(record_id))


@bp.delete("/ranging/records/<record_id>")
@require_perm("ranging:write")
def delete_record(record_id: str):
    analysis_service.delete_record(record_id)
    return ok({"deleted": True})


@bp.get("/ranging/records/export")
def export_records():
    import csv

    rows = analysis_service.fetch_all(_record_filters())
    buffer = io.StringIO()
    fieldnames = ["id", "task_type", "strategy", "target_kind", "metric_kind", "model_name",
                  "category", "size_group", "depth_m", "lateral_m", "euclidean_m",
                  "raw_depth_m", "corrected_depth_m", "valid_points", "total_points",
                  "distance_actual_m", "actual_source", "error_m", "abs_error_m", "rel_error",
                  "illumination", "distance_plan_m", "calibration_applied", "record_time", "source"]
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k) for k in fieldnames})
    payload_bytes = ("\ufeff" + buffer.getvalue()).encode("utf-8")
    fmt = arg("format", "csv")
    if fmt == "json":
        import json

        return ok({"items": rows})
    return Response(payload_bytes, mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=visionlab_ranging.csv"})


# ================================================================== 单图推理
@bp.post("/inference/frame")
@require_perm("inference:run")
def inference_frame():
    """对上传的单张图像（可选深度图）执行推理与测距。"""
    pipeline = pipeline_service.get_pipeline()
    file = request.files.get("image")
    if file is None:
        raise ValidationError("请上传 image 文件。")
    image = media.decode_image(file.read())
    if image is None:
        raise ValidationError("图像解码失败。")
    depth = None
    depth_file = request.files.get("depth")
    if depth_file is not None:
        raw = depth_file.read()
        import numpy as np

        arr = np.frombuffer(raw, dtype=np.uint8)
        import cv2

        decoded = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if decoded is not None:
            scale = float(request.form.get("depth_scale", 0.001))
            depth = decoded.astype("float32") * scale

    from core.camera.base import FrameBundle
    from core.geometry import estimate_intrinsics_from_fov
    import numpy as np

    h, w = image.shape[:2]
    if depth is None:
        depth = np.zeros((h, w), dtype="float32")
    bundle = FrameBundle(color=image, depth_m=depth,
                         depth_raw=(depth / 0.001).astype("uint16"), depth_scale=0.001,
                         timestamp=time.time(),
                         intrinsics=estimate_intrinsics_from_fov(w, h),
                         frame_number=0, serial="upload", source="upload")
    result = pipeline.process(bundle, None)
    return ok({"detection": result["detection"], "road": result["road"],
               "objects": result["objects"], "perf": result["perf"]})
