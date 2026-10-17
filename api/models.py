"""模型仓库与训练 API。"""

from __future__ import annotations

from flask import Blueprint

from algorithms import registry as plugin_registry
from core.db import now_iso
from core.exceptions import ModelError, NotFoundError
from services import model_registry, project_service

from .common import (arg, arg_int, current_user, ok, page_args, paginated, payload,
                     require_perm)

bp = Blueprint("models", __name__, url_prefix="/api/v1")


# ================================================================== 插件/架构
@bp.get("/models/plugins")
def list_plugins():
    plugin_registry.load_builtin()
    return ok({"items": plugin_registry.list_plugins(),
               "detection_configs": _detection_configs()})


def _detection_configs() -> list[dict]:
    from algorithms.detection import YoloDetectorAdapter

    configs = YoloDetectorAdapter.available_configs()
    return configs


@bp.get("/models/yolo-configs")
def yolo_configs():
    return ok({"items": _detection_configs(),
               "weight_modes": ["pretrained", "trained", "dual"]})


@bp.get("/models/seg-architectures")
def seg_architectures():
    from algorithms.segmentation import ARCH_LABELS

    out = []
    for key, label in ARCH_LABELS.items():
        available, note = True, ""
        if key == "unetv2":
            try:
                import timm  # noqa: F401
            except ImportError:
                available, note = False, "需要 timm：pip install timm==0.9.16"
        out.append({"key": key, "label": label, "available": available, "note": note})
    return ok({"items": out})


# ================================================================== 模型 CRUD
@bp.get("/models")
def list_models():
    page, size = page_args()
    return paginated(model_registry.list_models(page, size, task_type=arg("task_type"),
                                                arch=arg("arch"), keyword=arg("keyword", "")))


@bp.post("/models")
@require_perm("model:create")
def create_model():
    data = payload()
    model = model_registry.register_model(model_registry.ModelRegistration(
        name=str(data.get("name", "")), task_type=str(data.get("task_type", "detection")),
        weights_uri=data.get("weights_uri"), arch=data.get("arch"), variant=data.get("variant"),
        framework=data.get("framework"), project_id=data.get("project_id"),
        num_classes=data.get("num_classes"), class_names=data.get("class_names"),
        metrics=data.get("metrics"), notes=str(data.get("notes", "")),
        weight_origin=str(data.get("weight_origin", "imported"))))
    return ok(model)


@bp.get("/models/compare")
def compare_models():
    ids = [x for x in (arg("ids", "") or "").split(",") if x]
    return ok(model_registry.compare_models(ids))


@bp.get("/models/stats")
def model_stats():
    return ok(model_registry.model_stats())


@bp.get("/models/<model_id>")
def get_model(model_id: str):
    model = model_registry.get_model(model_id)
    if model is None:
        raise NotFoundError("模型不存在。")
    return ok(model)


@bp.patch("/models/<model_id>")
@require_perm("model:update")
def update_model(model_id: str):
    model_registry.update_model(model_id, **payload())
    return ok(model_registry.get_model(model_id))


@bp.delete("/models/<model_id>")
@require_perm("model:delete")
def delete_model(model_id: str):
    model_registry.delete_model(model_id)
    return ok({"deleted": True})


# ================================================================== 产物导入
@bp.post("/models/scan-artifacts")
@require_perm("model:create")
def scan_artifacts():
    data = payload()
    results = {}
    project_id = data.get("project_id")
    if data.get("runs_dir"):
        results["yolo_runs"] = model_registry.scan_yolo_runs(data["runs_dir"], project_id)
    if data.get("seg_models_dir"):
        results["seg_weights"] = model_registry.scan_segmentation_weights(
            data["seg_models_dir"], project_id, data.get("best_model_dir"))
    if data.get("scan_platform", True):
        results["platform_models"] = model_registry.scan_platform_models(project_id)
    return ok(results)


@bp.post("/models/<model_id>/export")
@require_perm("model:export")
def export_model(model_id: str):
    data = payload()
    model = model_registry.get_model(model_id)
    if model is None:
        raise NotFoundError("模型不存在。")
    fmt = str(data.get("format", "onnx"))
    if model["task_type"] == "detection":
        from algorithms.detection import YoloDetectorAdapter

        adapter = YoloDetectorAdapter()
        adapter.load(model.get("weights_uri"))
    else:
        from algorithms.segmentation import UNetSegmenterAdapter

        adapter = UNetSegmenterAdapter(arch=model.get("arch") or "unet")
        adapter.load(model.get("weights_uri"))
    path = adapter.export(fmt)
    return ok({"format": fmt, "path": path})


# ================================================================== 训练
@bp.get("/training/options")
def training_options():
    from core.config import settings

    return ok({
        "epochs": settings.get("training.epochs"),
        "batch": settings.get("training.batch"),
        "imgsz": settings.get("training.imgsz"),
        "lr": settings.get("training.lr"),
        "seed": settings.get("training.seed"),
        "device": settings.get("training.device"),
        "yolo_configs": _detection_configs(),
        "seg_architectures": seg_architectures().get_json()["data"]["items"],
        "note": "训练任务由任务队列串行执行；本平台以实验调度与结果管理为主，"
                "训练实现复用原项目脚本（Ultralytics / 自定义 PyTorch 训练循环）。",
    })


@bp.post("/training/start")
@require_perm("training:run")
def start_training():
    data = payload()
    project_id = data.get("project_id")
    if not project_id:
        row = project_service.list_projects(1, 1)["items"]
        if not row:
            raise ModelError("请先创建项目。")
        project_id = row[0]["id"]
    experiment = project_service.create_experiment(project_service.ExperimentSpec(
        project_id=project_id,
        name=str(data.get("name", f"训练 {data.get('arch') or data.get('yolo')}")),
        task_type=str(data.get("task_type", "detection")),
        kind="train",
        dataset_version_id=data.get("dataset_version_id"),
        config_snapshot={"training": {
            "epochs": data.get("epochs"), "batch": data.get("batch"),
            "imgsz": data.get("imgsz"), "lr": data.get("lr"),
            "arch": data.get("arch"), "yolo": data.get("yolo"),
            "device": data.get("device"),
        }},
        created_by=(current_user() or {}).get("username")))
    project_service.set_status(experiment["id"], "queued")
    task = project_service.enqueue(experiment["id"], "train",
                                   {"task_type": experiment["task_type"], **data})
    return ok({"experiment_id": experiment["id"], "task_id": task["id"], "status": "queued",
               "ws_channel": f"experiment:{experiment['id']}"})


@bp.get("/training/<task_id>/status")
def training_status(task_id: str):
    task = project_service.get_task(task_id)
    if task is None:
        raise NotFoundError("训练任务不存在。")
    experiment = project_service.get_experiment(task["experiment_id"]) if task.get("experiment_id") else None
    return ok({"task": task, "experiment": experiment})


@bp.post("/training/<task_id>/stop")
@require_perm("training:run")
def stop_training(task_id: str):
    project_service.update_task(task_id, status="canceled", finished_at=now_iso())
    return ok({"stopped": True})
