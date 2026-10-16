"""数据集、版本、标注与图像 API。"""

from __future__ import annotations

from flask import Blueprint

from core.exceptions import NotFoundError
from services import dataset_service

from .common import (arg, arg_int, current_user, ok, page_args, paginated, payload,
                     require_perm)

bp = Blueprint("datasets", __name__, url_prefix="/api/v1")


# ================================================================== 数据集
@bp.get("/datasets")
def list_datasets():
    page, size = page_args()
    return paginated(dataset_service.list_datasets(page, size, task_type=arg("task_type"),
                                                   project_id=arg("project_id")))


@bp.post("/datasets")
@require_perm("dataset:create")
def create_dataset():
    data = payload()
    user = current_user()
    dataset = dataset_service.create_dataset(
        name=str(data.get("name", "")), task_type=str(data.get("task_type", "detection")),
        project_id=data.get("project_id"), scene=str(data.get("scene", "campus_road")),
        class_names=data.get("class_names"), foreground_values=data.get("foreground_values"),
        description=str(data.get("description", "")),
        created_by=user["username"] if user else None)
    return ok(dataset)


@bp.get("/datasets/<dataset_id>")
def get_dataset(dataset_id: str):
    dataset = dataset_service.get_dataset(dataset_id)
    if dataset is None:
        raise NotFoundError("数据集不存在。")
    return ok(dataset)


@bp.patch("/datasets/<dataset_id>")
@require_perm("dataset:update")
def update_dataset(dataset_id: str):
    dataset_service.update_dataset(dataset_id, **payload())
    return ok(dataset_service.get_dataset(dataset_id))


@bp.delete("/datasets/<dataset_id>")
@require_perm("dataset:delete")
def delete_dataset(dataset_id: str):
    dataset_service.delete_dataset(dataset_id)
    return ok({"deleted": True})


# ================================================================== 版本
@bp.get("/datasets/<dataset_id>/versions")
def list_versions(dataset_id: str):
    return ok({"items": dataset_service.list_versions(dataset_id)})


@bp.post("/datasets/<dataset_id>/versions")
@require_perm("dataset:import")
def create_version(dataset_id: str):
    data = payload()
    result = dataset_service.create_version(
        dataset_id, str(data.get("version", "v1.0")), str(data.get("source_root", "")),
        split_ratio=str(data.get("split_ratio", "7:2:1")),
        split_seed=int(data.get("split_seed", 42)))
    return ok(result)


@bp.post("/datasets/versions/<version_id>/import")
@require_perm("dataset:import")
def import_annotations(version_id: str):
    data = payload()
    fmt = str(data.get("format", "yolo"))
    if fmt == "mask":
        result = dataset_service.attach_masks(version_id, str(data.get("source_dir", "")))
    else:
        result = dataset_service.import_annotations(
            version_id, str(data.get("source_dir", "")), fmt=fmt,
            class_names=data.get("class_names"))
    return ok(result)


@bp.post("/datasets/versions/<version_id>/validate")
@require_perm("dataset:update")
def validate_version(version_id: str):
    return ok(dataset_service.validate_version(version_id))


@bp.post("/datasets/versions/<version_id>/unify-foreground")
@require_perm("dataset:update")
def unify_foreground(version_id: str):
    data = payload()
    return ok(dataset_service.unify_foreground(version_id, data.get("target_value")))


@bp.post("/datasets/versions/<version_id>/split")
@require_perm("dataset:update")
def split_version(version_id: str):
    data = payload()
    return ok(dataset_service.split_version(version_id, str(data.get("ratio", "7:2:1")),
                                            int(data.get("seed", 42))))


@bp.post("/datasets/versions/<version_id>/freeze")
@require_perm("dataset:update")
def freeze_version(version_id: str):
    return ok(dataset_service.freeze_version(version_id))


@bp.get("/datasets/versions/<version_id>/stats")
def version_stats(version_id: str):
    return ok(dataset_service.version_stats(version_id))


@bp.get("/datasets/versions/diff")
def version_diff():
    return ok(dataset_service.version_diff(arg("from", ""), arg("to", "")))


@bp.get("/datasets/versions/<version_id>/images")
def list_images(version_id: str):
    page, size = page_args()
    result = dataset_service.list_images(version_id, page, size, split=arg("split"))
    for item in result["items"]:
        item["url"] = f"/api/v1/media/image/{item['id']}"
        if item.get("mask_uri"):
            item["mask_url"] = f"/api/v1/media/image/{item['id']}?kind=mask"
    return paginated(result)


@bp.get("/images/<image_id>")
def get_image(image_id: str):
    from core import db

    row = db.decode_row(db.query_one("SELECT * FROM image WHERE id=?", (image_id,)))
    if row is None:
        raise NotFoundError("图像不存在。")
    row["url"] = f"/api/v1/media/image/{image_id}"
    row["depth_url"] = f"/api/v1/media/image/{image_id}?kind=depth"
    row["annotations"] = dataset_service.list_annotations(image_id)
    return ok(row)


# ================================================================== 历史迁移
@bp.post("/datasets/legacy/detection")
@require_perm("dataset:import")
def import_legacy_detection():
    data = payload()
    source = str(data.get("source_dir", ""))
    from pathlib import Path as _Path

    if any((_Path(source) / s).exists() for s in ("train", "val", "test")):
        return ok(dataset_service.import_legacy_detection_full(
            source, data.get("labels_dir"), dataset_name=str(data.get("name", "road-target-det"))))
    return ok(dataset_service.import_legacy_detection_dataset(
        source, dataset_name=str(data.get("name", "road-target-det-v1")),
        limit=data.get("limit")))


@bp.post("/datasets/legacy/segmentation")
@require_perm("dataset:import")
def import_legacy_segmentation():
    data = payload()
    if data.get("data_root"):
        return ok(dataset_service.import_legacy_segmentation_full(
            str(data["data_root"]), dataset_name=str(data.get("name", "campus-road-seg"))))
    return ok(dataset_service.import_legacy_segmentation_dataset(
        str(data.get("images_dir", "")), str(data.get("masks_dir", "")),
        dataset_name=str(data.get("name", "campus-road-seg"))))
