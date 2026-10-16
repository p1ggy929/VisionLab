"""项目、实验、复现清单与任务队列 API。"""

from __future__ import annotations

from flask import Blueprint, request

from core.exceptions import NotFoundError
from services import auth_service, project_service

from .common import (arg, arg_int, current_user, ok, page_args, paginated, payload,
                     require_perm)

bp = Blueprint("projects", __name__, url_prefix="/api/v1")


# ================================================================== 项目
@bp.get("/projects")
def list_projects():
    page, size = page_args()
    return paginated(project_service.list_projects(page, size, keyword=arg("keyword", "")))


@bp.post("/projects")
@require_perm("project:create")
def create_project():
    data = payload()
    user = current_user()
    project = project_service.create_project(
        name=str(data.get("name", "")), description=str(data.get("description", "")),
        scene=str(data.get("scene", "campus_road")),
        app_target=str(data.get("app_target", "sweeper")),
        owner_id=user["id"] if user else None, config=data.get("config"))
    auth_service.audit(user["id"] if user else None, user["username"] if user else None,
                       "project.create", "project", project.get("id"), "success",
                       request.remote_addr or "")
    return ok(project)


@bp.get("/projects/<project_id>")
def get_project(project_id: str):
    project = project_service.get_project(project_id)
    if project is None:
        raise NotFoundError("项目不存在。")
    project["members"] = project_service.list_members(project_id)
    return ok(project)


@bp.patch("/projects/<project_id>")
@require_perm("project:update")
def update_project(project_id: str):
    project_service.update_project(project_id, **payload())
    return ok(project_service.get_project(project_id))


@bp.delete("/projects/<project_id>")
@require_perm("project:update")
def delete_project(project_id: str):
    project_service.delete_project(project_id)
    return ok({"deleted": True})


@bp.post("/projects/<project_id>/archive")
@require_perm("project:update")
def archive_project(project_id: str):
    project_service.archive_project(project_id)
    return ok(project_service.get_project(project_id))


@bp.get("/projects/<project_id>/members")
def list_members(project_id: str):
    return ok({"items": project_service.list_members(project_id)})


@bp.post("/projects/<project_id>/members")
@require_perm("project:update")
def add_member(project_id: str):
    data = payload()
    member_id = project_service.add_member(project_id, str(data.get("user_id", "")),
                                           str(data.get("role_in_project", "developer")))
    return ok({"id": member_id})


# ================================================================== 实验
@bp.get("/experiments")
def list_experiments():
    page, size = page_args()
    return paginated(project_service.list_experiments(
        page, size, project_id=arg("project_id"), task_type=arg("task_type"),
        status=arg("status"), kind=arg("kind")))


@bp.post("/experiments")
@require_perm("experiment:create")
def create_experiment():
    data = payload()
    user = current_user()
    experiment = project_service.create_experiment(
        project_id=str(data.get("project_id", "")), name=str(data.get("name", "")),
        task_type=str(data.get("task_type", "detection")),
        kind=str(data.get("kind", "infer")),
        dataset_version_id=data.get("dataset_version_id"), model_id=data.get("model_id"),
        calibration_id=data.get("calibration_id"), camera_id=data.get("camera_id"),
        config_snapshot=data.get("config_snapshot"),
        created_by=user["username"] if user else None)
    return ok(experiment)


@bp.get("/experiments/compare")
def compare_experiments():
    ids = [x for x in (arg("ids", "") or "").split(",") if x]
    return ok(project_service.compare_experiments(ids))


@bp.get("/experiments/<experiment_id>")
def get_experiment(experiment_id: str):
    experiment = project_service.get_experiment(experiment_id)
    if experiment is None:
        raise NotFoundError("实验不存在。")
    return ok(experiment)


@bp.patch("/experiments/<experiment_id>/config")
@require_perm("experiment:update")
def update_config(experiment_id: str):
    project_service.update_experiment_config(experiment_id, payload().get("config_snapshot") or payload())
    return ok(project_service.get_experiment(experiment_id))


@bp.post("/experiments/<experiment_id>/run")
@require_perm("experiment:run")
def run_experiment(experiment_id: str):
    experiment = project_service.get_experiment(experiment_id)
    if experiment is None:
        raise NotFoundError("实验不存在。")
    project_service.set_status(experiment_id, "queued")
    task = project_service.enqueue(experiment_id, experiment["kind"],
                                   {"task_type": experiment["task_type"]})
    return ok({"task_id": task["id"], "experiment_id": experiment_id,
               "status": "queued", "queue_position": 1,
               "ws_channel": f"experiment:{experiment_id}"})


@bp.post("/experiments/<experiment_id>/pause")
@require_perm("experiment:run")
def pause_experiment(experiment_id: str):
    from services import pipeline_service

    return ok(pipeline_service.get_pipeline().toggle_pause(True))


@bp.post("/experiments/<experiment_id>/resume")
@require_perm("experiment:run")
def resume_experiment(experiment_id: str):
    from services import pipeline_service

    return ok(pipeline_service.get_pipeline().toggle_pause(False))


@bp.post("/experiments/<experiment_id>/cancel")
@require_perm("experiment:run")
def cancel_experiment(experiment_id: str):
    project_service.set_status(experiment_id, "canceled")
    return ok(project_service.get_experiment(experiment_id))


@bp.post("/experiments/<experiment_id>/archive")
@require_perm("experiment:archive")
def archive_experiment(experiment_id: str):
    project_service.archive_experiment(experiment_id)
    return ok(project_service.get_experiment(experiment_id))


@bp.post("/experiments/<experiment_id>/clone")
@require_perm("experiment:create")
def clone_experiment(experiment_id: str):
    data = payload()
    user = current_user()
    return ok(project_service.clone_experiment(experiment_id, data.get("name"),
                                               user["username"] if user else None))


@bp.delete("/experiments/<experiment_id>")
@require_perm("experiment:update")
def delete_experiment(experiment_id: str):
    project_service.delete_experiment(experiment_id)
    return ok({"deleted": True})


# ================================================================== 复现
@bp.get("/experiments/<experiment_id>/manifest")
def get_manifest(experiment_id: str):
    experiment = project_service.get_experiment(experiment_id)
    if experiment is None:
        raise NotFoundError("实验不存在。")
    if not experiment.get("run_manifest"):
        return ok(project_service.build_run_manifest(experiment_id))
    return ok(experiment["run_manifest"])


@bp.post("/experiments/<experiment_id>/manifest")
@require_perm("experiment:run")
def build_manifest(experiment_id: str):
    from core.camera.service import get_camera_service

    camera = get_camera_service().describe()
    return ok(project_service.build_run_manifest(experiment_id, camera))


@bp.post("/experiments/<experiment_id>/reproduce")
def reproduce(experiment_id: str):
    return ok(project_service.verify_reproduction(experiment_id))


# ================================================================== 任务
@bp.get("/tasks")
def list_tasks():
    return ok({"items": project_service.list_tasks(arg("status")),
               "summary": project_service.task_summary()})


@bp.get("/tasks/<task_id>")
def get_task(task_id: str):
    task = project_service.get_task(task_id)
    if task is None:
        raise NotFoundError("任务不存在。")
    return ok(task)
