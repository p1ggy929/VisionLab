"""媒体文件服务：图像、深度、掩膜、报告、通用存储文件。"""

from __future__ import annotations

from pathlib import Path

from flask import Blueprint, Response, send_file

from core import db, media, storage
from core.config import settings
from core.exceptions import NotFoundError, ValidationError

from .common import arg, ok, require_perm

bp = Blueprint("media", __name__, url_prefix="/api/v1")


def _image_row(image_id: str) -> dict:
    row = db.decode_row(db.query_one("SELECT * FROM image WHERE id=?", (image_id,)))
    if row is None:
        raise NotFoundError("图像不存在。")
    return row


@bp.get("/media/image/<image_id>")
def image_file(image_id: str):
    row = _image_row(image_id)
    kind = arg("kind", "rgb")
    if kind == "depth":
        depth_uri = row.get("mask_uri") if arg("as") == "mask" else None
        depth = db.decode_row(db.query_one("SELECT * FROM depth_map WHERE image_id=?", (image_id,)))
        uri = (depth or {}).get("uri") or depth_uri
        if not uri:
            raise NotFoundError("该图像没有关联深度图。")
        path = storage.to_path(uri)
        if not path or not Path(path).exists():
            raise NotFoundError("深度图文件缺失。")
        arr = media.read_depth16(path, (depth or {}).get("depth_scale", 0.001) or 0.001)
        if arr is None:
            raise NotFoundError("深度图读取失败。")
        view = media.colorize_depth(arr, float(settings.get("ranging.min_range_m", 0.3)), 15.0)
        return Response(media.encode_jpeg(view, 85), mimetype="image/jpeg",
                        headers={"Cache-Control": "max-age=60"})
    if kind == "mask":
        path = storage.to_path(row.get("mask_uri"))
        if not path or not Path(path).exists():
            raise NotFoundError("该图像没有掩膜。")
        return send_file(str(path), mimetype="image/png")
    if kind == "overlay":
        path = storage.to_path(row.get("meta", {}).get("overlay_uri") if isinstance(row.get("meta"), dict) else None)
        if path and Path(path).exists():
            return send_file(str(path))
    path = storage.to_path(row.get("uri"))
    if not path or not Path(path).exists():
        raise NotFoundError("图像文件缺失。")
    return send_file(str(path))


@bp.get("/media/frame")
def frame_file():
    """按 URI 读取运行期产物（叠加帧、掩膜、截图）。"""
    uri = arg("uri")
    if not uri:
        raise ValidationError("缺少 uri 参数。")
    path = storage.to_path(uri)
    if not path or not Path(path).exists():
        raise NotFoundError(f"文件不存在: {uri}")
    if path.suffix.lower() in (".jpg", ".jpeg"):
        return send_file(str(path), mimetype="image/jpeg")
    if path.suffix.lower() == ".png":
        return send_file(str(path), mimetype="image/png")
    if path.suffix.lower() in (".mp4", ".avi"):
        return send_file(str(path), mimetype="video/mp4")
    return send_file(str(path), as_attachment=True)


@bp.get("/media/thumb")
def thumbnail():
    """生成缩略图（用于画廊）。"""
    uri = arg("uri")
    width = int(arg("width", 320) or 320)
    if not uri:
        raise ValidationError("缺少 uri 参数。")
    path = storage.to_path(uri)
    if not path or not Path(path).exists():
        raise NotFoundError(f"文件不存在: {uri}")
    image = media.read_image(path)
    if image is None:
        raise NotFoundError("图像读取失败（可能是深度图或视频，请用原文件接口）。")
    view = media.resize_keep_ratio(image, width, width)
    return Response(media.encode_jpeg(view, 78), mimetype="image/jpeg",
                    headers={"Cache-Control": "max-age=300"})


@bp.get("/media/export/<path:rel_path>")
def export_file(rel_path: str):
    path = storage.to_path(rel_path)
    if not path or not Path(path).exists():
        raise NotFoundError(f"文件不存在: {rel_path}")
    return send_file(str(path), as_attachment=True)


@bp.get("/storage/tree")
@require_perm("system:read")
def storage_tree():
    """列出存储根下的目录用量（供监控页与目录说明）。"""
    items = []
    for name in ("datasets", "captures", "models", "runs", "results",
                 "calibration", "reports", "logs", "exports"):
        path = settings.storage_dir(name)
        items.append({
            "key": name, "path": str(path),
            "files": sum(1 for p in path.rglob("*") if p.is_file()),
            "size": storage.human_size(storage.dir_size(path)),
        })
    return ok({"root": str(settings.storage_root), "items": items,
               "free_gb": storage.free_space_gb()})
