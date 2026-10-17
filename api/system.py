"""系统设置、日志、监控与自检 API。"""

from __future__ import annotations

from flask import Blueprint

from core.config import settings
from services import health_service

from .common import arg, arg_bool, arg_int, current_user, ok, payload, require_perm

bp = Blueprint("system", __name__, url_prefix="/api/v1")

# 允许通过接口读取的配置项（只读白名单）
_CONFIG_WHITELIST = (
    "app", "server.stream", "camera", "ranging", "detection", "segmentation",
    "training", "analysis", "report", "logging", "storage",
)

# 设置页展示的存储子目录
_STORAGE_DIR_NAMES = (
    "datasets", "captures", "models", "runs", "results",
    "calibration", "reports", "logs", "exports",
)


@bp.get("/system/info")
def system_info():
    return ok(health_service.system_info())


@bp.get("/system/health")
def health():
    from core import db

    return ok({"system": health_service.system_info(),
               "metrics": health_service.metrics(),
               "tables": db.table_counts(),
               "camera": health_service.camera_metrics(),  # noqa: SLF001
               "queue": health_service.queue_metrics()})  # noqa: SLF001


@bp.post("/system/self-check")
def self_check():
    return ok(health_service.self_check())


@bp.get("/system/metrics")
def metrics():
    return ok(health_service.metrics())


@bp.get("/system/logs")
def logs():
    return ok({"items": health_service.logs(level=arg("level"), keyword=arg("keyword"),
                                            limit=arg_int("limit", 200))})


@bp.get("/system/logs/tail")
def logs_tail():
    return ok(health_service.log_file_tail(arg_int("lines", 100)))


@bp.post("/system/logs/clear")
@require_perm("system:manage")
def logs_clear():
    health_service.clear_log_buffer()
    return ok({"cleared": True})


@bp.get("/system/settings")
def get_settings_api():
    data = {key: settings.get(key) for key in _CONFIG_WHITELIST}
    all_dirs = {}
    for name in _STORAGE_DIR_NAMES:
        all_dirs[name] = str(settings.storage_dir(name))
    data["all_dirs"] = all_dirs
    data["db_path"] = str(settings.db_path)
    data["readonly"] = True
    return ok(data)


@bp.get("/system/dict")
def dictionary():
    """前端使用的统一枚举字典（前端不硬编码）。"""
    from core.config import settings as cfg
    from services import auth_service, project_service

    return ok({
        "task_types": ["detection", "segmentation", "ranging", "joint"],
        "experiment_kinds": list(project_service.EXPERIMENT_KINDS),
        "experiment_status": list(project_service.STATUS_FLOW),
        "status_labels": {
            "draft": "草稿", "configured": "已配置", "queued": "排队中", "running": "运行中",
            "paused": "已暂停", "done": "已完成", "failed": "失败", "canceled": "已取消",
            "archived": "已归档",
        },
        "task_labels": {"detection": "目标检测", "segmentation": "道路分割",
                        "ranging": "距离测量", "joint": "联合实验"},
        "roles": [{"key": k, "label": v} for k, v in auth_service.ROLE_LABELS.items()],
        "metric_kinds": ["min", "max", "mean", "center"],
        "target_kinds": ["object", "road_left", "road_right", "road_center"],
        "illuminations": [
            {"key": "sunny", "label": "晴天正午"},
            {"key": "shade", "label": "树荫遮挡"},
            {"key": "backlight", "label": "逆光"},
            {"key": "weak_texture", "label": "弱纹理路面"},
            {"key": "overcast", "label": "阴天"},
        ],
        "size_groups": {"large": "大目标", "small": "小目标"},
        "scenes": [
            {"key": "campus_road", "label": "校园道路"},
            {"key": "urban_road", "label": "城市道路"},
            {"key": "indoor", "label": "室内"},
        ],
        "app_targets": [
            {"key": "sweeper", "label": "无人环卫机器人"},
            {"key": "inspection", "label": "校园巡检机器人"},
            {"key": "low_speed_av", "label": "低速无人车"},
        ],
        "distance_plan_m": cfg.get("ranging.distance_plan_m", []),
        "detection_classes": cfg.detection_class_names(),
        "class_groups": {name: cfg.class_group(name) for name in cfg.detection_class_names()},
    })


@bp.get("/system/report-templates")
def report_templates():
    from services import report_service

    return ok({"items": report_service.list_templates()})


@bp.get("/system/health-check-options")
def health_check_options():
    return ok({"checks": ["依赖", "CUDA 加速", "RealSense 设备", "元数据库",
                          "存储目录可写", "磁盘空间", "运行期目录"]})
