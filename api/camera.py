"""相机接入、采集与标定 API。"""

from __future__ import annotations

from flask import Blueprint, request

from core.camera.service import get_camera_service
from core.exceptions import NotFoundError
from services import capture_service

from .common import (arg, arg_bool, arg_int, current_user, ok, payload, require_perm)

bp = Blueprint("camera", __name__, url_prefix="/api/v1")


# ================================================================== 设备
@bp.get("/camera/devices")
def devices():
    from core.camera import realsense_backend as rsb

    return ok({"realsense_available": rsb.available(), "devices": rsb.list_devices()})


@bp.get("/camera/status")
def status():
    return ok(get_camera_service().describe())


@bp.get("/camera/backends")
def backends():
    from core.camera import realsense_backend as rsb
    from core.camera.service import get_camera_service as _get

    options = [{"key": "auto", "label": "自动（有设备用 RealSense，否则用合成场景）"}]
    if rsb.available():
        options.append({"key": "realsense", "label": "Intel RealSense D455（真实设备）"})
    options.append({"key": "mock", "label": "合成场景（MockBackend，无设备也可演示）"})
    options.append({"key": "replay", "label": "离线回放（从落盘的真实 RGB-D 序列重放）"})
    return ok({"items": options, "current": _get().describe().get("backend")})


@bp.get("/camera/replay-sources")
def replay_sources():
    """列出可作为离线回放源的数据：采集会话与含深度图的数据集版本。"""
    from core.camera.replay_backend import list_replay_sources

    return ok({"items": list_replay_sources()})


@bp.post("/camera/replay-sample")
@require_perm("capture:run")
def create_replay_sample():
    """用当前相机抓取若干帧，生成一个可用于离线回放的样本。"""
    from core.camera.replay_backend import capture_sample

    data = payload()
    return ok(capture_sample(str(data.get("source_id") or ""), int(data.get("frames", 40))))


@bp.get("/camera/replay-progress")
def replay_progress():
    backend = get_camera_service()._backend  # noqa: SLF001 - 只读探查
    if backend is not None and hasattr(backend, "progress"):
        return ok(backend.progress())  # type: ignore[attr-defined]
    return ok({"active": False})


@bp.post("/camera/stream/start")
@require_perm("camera:control")
def start_stream():
    data = payload()
    source = None
    if str(data.get("backend", "")).lower() == "replay" or data.get("replay_source"):
        source = {
            "type": data.get("replay_type", "capture"),
            "id": data.get("replay_source") or data.get("source_id"),
            "fps": data.get("fps"), "loop": data.get("loop", True),
        }
    info = get_camera_service().start(
        prefer=data.get("backend"), width=data.get("width"),
        height=data.get("height"), fps=data.get("fps"), replay_source=source)
    return ok(info)


@bp.post("/camera/stream/stop")
@require_perm("camera:control")
def stop_stream():
    get_camera_service().stop()
    return ok({"running": False})


@bp.get("/camera/frame")
def camera_frame():
    """返回当前原始帧的 JPEG（用于前端预览轮询降级）。"""
    from core import media
    from flask import Response

    camera = get_camera_service()
    if not camera.running:
        camera.start()
    bundle = camera.latest(timeout=1.0)
    if bundle is None:
        return Response(b"", mimetype="image/jpeg")
    return Response(media.encode_jpeg(bundle.color, 80), mimetype="image/jpeg",
                    headers={"Cache-Control": "no-store"})


@bp.get("/camera/depth_frame")
def camera_depth_frame():
    from core import media
    from flask import Response

    camera = get_camera_service()
    if not camera.running:
        camera.start()
    bundle = camera.latest(timeout=1.0)
    if bundle is None:
        return Response(b"", mimetype="image/jpeg")
    view = media.colorize_depth(bundle.depth_m, settings_min(), settings_max())
    return Response(media.encode_jpeg(view, 80), mimetype="image/jpeg",
                    headers={"Cache-Control": "no-store"})


def settings_min() -> float:
    from core.config import settings

    return float(settings.get("ranging.min_range_m", 0.3))


def settings_max() -> float:
    from core.config import settings

    return min(float(settings.get("ranging.max_range_m", 20.0)), 15.0)


@bp.get("/camera/ground-truth")
def ground_truth():
    return ok({"items": get_camera_service().ground_truth()})


@bp.get("/camera/depth-probe")
def depth_probe():
    return ok(capture_service.depth_probe(arg_int("x", 0) or None,
                                          arg_int("y", 0) or None))


@bp.get("/camera/filters")
def filter_options():
    return ok({"items": [
        {"key": "", "label": "不滤波（保留原始深度，项目 A 的验证结论）"},
        {"key": "spatial", "label": "空间滤波 Spatial"},
        {"key": "temporal", "label": "时间滤波 Temporal"},
        {"key": "hole_filling", "label": "空洞填充 Hole Filling"},
        {"key": "decimation", "label": "降采样 Decimation"},
    ]})


# ================================================================== 标定
@bp.get("/camera/calibrations")
def list_calibrations():
    return ok({"items": capture_service.list_calibrations(),
               "default": capture_service.default_calibration()})


@bp.get("/camera/calibrations/<calibration_id>")
def get_calibration(calibration_id: str):
    calibration = capture_service.get_calibration(calibration_id)
    if calibration is None:
        raise NotFoundError("标定记录不存在。")
    return ok(calibration)


@bp.post("/camera/calibrate")
@require_perm("calibration:run")
def calibrate():
    data = payload()
    user = current_user()
    calibration = capture_service.run_calibration(
        name=data.get("name"), samples=data.get("samples"),
        created_by=user["username"] if user else None,
        target_distances=data.get("target_distances"))
    return ok(calibration)


@bp.post("/camera/calibrations/<calibration_id>/default")
@require_perm("calibration:run")
def set_default(calibration_id: str):
    return ok(capture_service.set_default_calibration(calibration_id))


# ================================================================== 采集
@bp.get("/capture/sessions")
def list_sessions():
    items = capture_service.list_capture_sessions(arg_int("limit", 50))
    for item in items:
        item.pop("items", None)
    return ok({"items": items})


@bp.post("/capture/sessions")
@require_perm("capture:run")
def create_session():
    data = payload()
    user = current_user()
    session = capture_service.start_capture_session(
        name=str(data.get("name", "")), experiment_id=data.get("experiment_id"),
        scene=str(data.get("scene", "campus_road")),
        illumination=str(data.get("illumination", "sunny")),
        operator=user["username"] if user else None)
    return ok(session)


@bp.get("/capture/sessions/<session_id>")
def get_session(session_id: str):
    return ok(capture_service.get_capture_session(session_id))


@bp.post("/capture/sessions/<session_id>/snapshot")
@require_perm("capture:run")
def take_snapshot(session_id: str):
    data = payload()
    return ok(capture_service.snapshot(session_id, illumination=data.get("illumination"),
                                       target_size_group=data.get("target_size_group"),
                                       notes=str(data.get("notes", ""))))


@bp.post("/capture/sessions/<session_id>/truth")
@require_perm("capture:run")
def set_truth(session_id: str):
    data = payload()
    item = capture_service.set_truth(
        session_id, str(data.get("stamp", "")),
        actual_distance_m=data.get("actual_distance_m"),
        actual_min_m=data.get("actual_min_m"), actual_max_m=data.get("actual_max_m"),
        actual_source=str(data.get("actual_source", "tape")),
        category=data.get("category"), illumination=data.get("illumination"),
        notes=str(data.get("notes", "")))
    return ok(item)


@bp.post("/capture/sessions/<session_id>/recording/start")
@require_perm("capture:run")
def start_recording(session_id: str):
    return ok(capture_service.start_recording(session_id))


@bp.post("/capture/sessions/<session_id>/recording/stop")
@require_perm("capture:run")
def stop_recording(session_id: str):
    return ok(capture_service.stop_recording(session_id))


@bp.post("/capture/sessions/<session_id>/finish")
@require_perm("capture:run")
def finish_session(session_id: str):
    data = payload()
    return ok(capture_service.finish_session(session_id, data.get("dataset_version_id")))


@bp.get("/capture/distance-plan")
def distance_plan():
    from core.config import settings

    return ok({"items": settings.get("ranging.distance_plan_m", []),
               "tolerance_m": settings.get("ranging.pause_tolerance_m", 0.05),
               "timeout_sec": settings.get("ranging.pause_timeout_sec", 30)})
