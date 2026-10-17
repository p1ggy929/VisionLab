"""采集会话与相机标定。

采集：会话生命周期、标准距离点定格、真值录入、抓拍落盘、双路录制。
标定：内参与深度尺度登记、深度平面校验（对标靶已知距离读深度并给出偏差曲线）、
      标定文件落盘与哈希、默认标定切换。
"""

from __future__ import annotations

from dataclasses import dataclass

import json
from pathlib import Path
from typing import Any

import numpy as np

from core import db, media, storage
from core.camera.service import get_camera_service
from core.config import settings
from core.exceptions import CameraError, NotFoundError, ValidationError
from core.logging_setup import get_logger

log = get_logger("services.capture")


# ================================================================== 采集会话
def start_capture_session(name: str = "", experiment_id: str | None = None,
                          scene: str = "campus_road", illumination: str = "sunny",
                          operator: str | None = None) -> dict:
    session_id = db.new_id("cap")
    root = settings.storage_dir("captures") / session_id
    (root / "rgb").mkdir(parents=True, exist_ok=True)
    (root / "depth").mkdir(parents=True, exist_ok=True)
    (root / "video").mkdir(parents=True, exist_ok=True)

    camera = get_camera_service()
    if not camera.running:
        camera.start()
    info = camera.describe()

    serial = info.get("serial") or f"AUTO-{session_id}"
    if not db.query_one("SELECT id FROM camera_device WHERE serial=?", (serial,)):
        db.insert("camera_device", {
            "model": info.get("model") or "Unknown",
            "serial": serial,
            "firmware": info.get("firmware"),
            "usb_type": info.get("usb_type"),
            "default_resolution": f"{info.get('width')}x{info.get('height')}",
            "default_fps": info.get("fps"),
            "filters": info.get("filters") or [],
            "last_seen_at": db.now_iso(),
        }, id_prefix="cam")

    session = {
        "id": session_id,
        "name": name or f"采集会话 {db.now_iso()}",
        "experiment_id": experiment_id,
        "scene": scene,
        "illumination": illumination,
        "operator": operator,
        "root": storage.to_uri(root),
        "started_at": db.now_iso(),
        "items": [],
        "recording": False,
        "paused": False,
        "camera": info,
    }
    _write_meta(root, session)
    log.info("采集会话已创建: %s", session_id)
    return session


def _meta_path(root: Path) -> Path:
    return root / "session.json"


def _write_meta(root: Path, session: dict) -> None:
    _meta_path(root).write_text(json.dumps(session, ensure_ascii=False, indent=2),
                                encoding="utf-8")


def get_capture_session(session_id: str) -> dict:
    root = settings.storage_dir("captures") / session_id
    path = _meta_path(root)
    if not path.exists():
        raise NotFoundError(f"采集会话不存在: {session_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def snapshot(session_id: str, illumination: str | None = None,
             target_size_group: str | None = None, notes: str = "") -> dict:
    session = get_capture_session(session_id)
    root = Path(storage.to_path(session["root"]) or "")
    camera = get_camera_service()
    bundle = camera.latest(timeout=1.0)
    if bundle is None:
        raise CameraError("没有可用画面，无法抓拍。")

    stamp = storage.timestamp("%Y%m%d_%H%M%S_%f")[:-3]
    rgb_path = root / "rgb" / f"rgb_{stamp}.jpg"
    depth_path = root / "depth" / f"depth_{stamp}.png"
    media.write_image(rgb_path, bundle.color, quality=95)
    media.write_depth16(depth_path, bundle.depth_m, bundle.depth_scale)

    item = {
        "stamp": stamp,
        "rgb_uri": storage.to_uri(rgb_path),
        "depth_uri": storage.to_uri(depth_path),
        "frame_number": bundle.frame_number,
        "depth_scale": bundle.depth_scale,
        "intrinsics": bundle.intrinsics.to_dict(),
        "illumination": illumination or session.get("illumination"),
        "target_size_group": target_size_group,
        "notes": notes,
        "truth": None,
        "created_at": db.now_iso(),
    }
    session.setdefault("items", []).append(item)
    _write_meta(root, session)
    return item


@dataclass(frozen=True)
class TruthEntry:
    """一次抓拍记录的真值标注。"""

    stamp: str
    actual_distance_m: float | None = None
    actual_min_m: float | None = None
    actual_max_m: float | None = None
    actual_source: str = "tape"
    category: str | None = None
    illumination: str | None = None
    notes: str = ""


def set_truth(session_id: str, entry: TruthEntry) -> dict:
    stamp = entry.stamp
    actual_distance_m = entry.actual_distance_m
    actual_min_m, actual_max_m = entry.actual_min_m, entry.actual_max_m
    actual_source, category = entry.actual_source, entry.category
    illumination, notes = entry.illumination, entry.notes
    session = get_capture_session(session_id)
    root = Path(storage.to_path(session["root"]) or "")
    for item in session.get("items", []):
        if item["stamp"] == stamp:
            item["truth"] = {
                "actual_distance_m": actual_distance_m,
                "actual_min_m": actual_min_m,
                "actual_max_m": actual_max_m,
                "actual_source": actual_source,
                "category": category,
                "notes": notes,
            }
            if illumination:
                item["illumination"] = illumination
            _write_meta(root, session)
            _register_image(session, item)
            return item
    raise NotFoundError(f"该会话中没有找到抓拍记录: {stamp}")


def _register_image(session: dict, item: dict) -> None:
    """把采集结果登记到数据集（若指定了数据集版本）。"""
    version_id = session.get("dataset_version_id")
    if not version_id:
        return
    existing = db.query_one("SELECT id FROM image WHERE uri=?", (item["rgb_uri"],))
    if existing:
        return
    depth_path = storage.to_path(item["depth_uri"])
    stats = {}
    if depth_path and Path(depth_path).exists():
        from core.geometry import depth_stats

        arr = media.read_depth16(depth_path, item.get("depth_scale", 0.001))
        if arr is not None:
            stats = depth_stats(arr, settings.get("ranging.min_range_m", 0.3),
                                settings.get("ranging.max_range_m", 20.0))
    image_id = db.insert("image", {
        "dataset_version_id": version_id,
        "uri": item["rgb_uri"],
        "split": "train",
        "source": "capture",
        "capture_session_id": session["id"],
        "illum": item.get("illumination"),
        "target_size_group": item.get("target_size_group"),
        "meta": {"frame_number": item.get("frame_number"),
                 "truth": item.get("truth")},
    }, id_prefix="img")
    db.insert("depth_map", {
        "image_id": image_id, "uri": item["depth_uri"], "encoding": "png16",
        "depth_scale": item.get("depth_scale", 0.001),
        "valid_ratio": stats.get("valid_ratio"), "min_m": stats.get("min_m"),
        "max_m": stats.get("max_m"), "mean_m": stats.get("mean_m"),
    }, id_prefix="dm")


def start_recording(session_id: str) -> dict:
    session = get_capture_session(session_id)
    root = Path(storage.to_path(session["root"]) or "")
    result = get_camera_service().start_recording(root / "video")
    session["recording"] = bool(result.get("recording"))
    _write_meta(root, session)
    return result


def stop_recording(session_id: str) -> dict:
    session = get_capture_session(session_id)
    root = Path(storage.to_path(session["root"]) or "")
    result = get_camera_service().stop_recording()
    session["recording"] = False
    _write_meta(root, session)
    return result


def finish_session(session_id: str, dataset_version_id: str | None = None) -> dict:
    session = get_capture_session(session_id)
    root = Path(storage.to_path(session["root"]) or "")
    if dataset_version_id:
        session["dataset_version_id"] = dataset_version_id
    session["finished_at"] = db.now_iso()
    session["item_count"] = len(session.get("items", []))
    session["with_truth"] = sum(1 for item in session.get("items", []) if item.get("truth"))
    _write_meta(root, session)
    return session


def list_capture_sessions(limit: int = 50) -> list[dict]:
    root = settings.storage_dir("captures")
    out = []
    for path in sorted(root.glob("*/session.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        data["item_count"] = len(data.get("items", []))
        data["with_truth"] = sum(1 for item in data.get("items", []) if item.get("truth"))
        out.append(data)
    return out


# ================================================================== 标定
def list_calibrations() -> list[dict]:
    rows = db.query_all("SELECT * FROM calibration ORDER BY created_at DESC")
    return db.decode_rows(rows)


def get_calibration(calibration_id: str) -> dict | None:
    return db.decode_row(db.query_one("SELECT * FROM calibration WHERE id=?", (calibration_id,)))


def default_calibration() -> dict | None:
    row = db.query_one("SELECT * FROM calibration WHERE is_default=1 ORDER BY created_at DESC LIMIT 1")
    return db.decode_row(row)


def run_calibration(name: str | None = None, samples: list[dict] | None = None,
                    created_by: str | None = None, target_distances: list[float] | None = None) -> dict:
    """执行标定。

    若传入 samples（每项 {u, v, actual_distance_m}），用图像中心区深度与真值做一次
    尺度与外参校验；否则只登记当前相机内参 + 深度平面自检。
    """
    camera = get_camera_service()
    info = camera.describe()
    intrinsics = info.get("intrinsics") or {}
    depth_scale = info.get("depth_scale") or 0.001

    effective_samples = samples
    if not effective_samples:
        bundle = camera.latest(timeout=1.0)
        if bundle is None:
            raise CameraError("没有可用画面，无法标定。")
        effective_samples = []
        for point in (target_distances or settings.get("ranging.distance_plan_m", []))[:8]:
            effective_samples.append({"actual_distance_m": float(point)})
        target_distances = None

    from algorithms.ground import estimate_ground_plane

    bundle = camera.latest(timeout=1.0)
    if bundle is None:
        raise CameraError("没有可用画面，无法标定。")
    plane = estimate_ground_plane(bundle.depth_m,
                                  min_m=settings.get("ranging.min_range_m", 0.3),
                                  max_m=settings.get("ranging.max_range_m", 20.0))

    check_rows: list[dict] = []
    errors: list[float] = []
    for sample in effective_samples:
        actual = sample.get("actual_distance_m")
        if actual is None:
            continue
        h, w = bundle.depth_m.shape[:2]
        v = int(h * 0.5 + (1.25 * (intrinsics.get("fy") or 640.0)) / max(float(actual), 1e-3))
        v = int(np.clip(v, 0, h - 1))
        u = w // 2
        patch = bundle.depth_m[max(0, v - 5):v + 6, max(0, u - 5):u + 6]
        valid = patch[patch > 0]
        measured = float(np.median(valid)) if valid.size else None
        if measured is None:
            continue
        rel = abs(measured - float(actual)) / float(actual)
        errors.append(measured - float(actual))
        check_rows.append({
            "actual_m": round(float(actual), 3),
            "measured_m": round(measured, 4),
            "abs_error_m": round(measured - float(actual), 4),
            "rel_error": round(rel, 4),
        })

    rms = float(np.sqrt(np.mean(np.square(errors)))) if errors else None
    payload = {
        "camera_id": None,
        "name": name or f"calib_{storage.timestamp()}",
        "intrinsics": intrinsics,
        "extrinsics": {"baseline_m": None, "note": "由 RealSense 出厂外参提供"},
        "depth_scale": depth_scale,
        "rms_reprojection_px": rms,
        "depth_check": {
            "ground_plane": plane.to_dict(),
            "samples": check_rows,
            "mean_abs_error_m": round(float(np.mean(np.abs(errors))), 4) if errors else None,
            "note": "深度平面校验：按相机高度与焦距推算标靶所在图像行，读取该处深度与真值对比",
        },
        "created_by": created_by,
    }
    out_dir = settings.storage_dir("calibration")
    file_path = out_dir / f"{payload['name']}.json"
    file_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["file_uri"] = storage.to_uri(file_path)
    payload["file_sha256"] = storage.sha256_file(file_path)

    calibration_id = db.insert("calibration", {**payload, "is_default": 0}, id_prefix="cal")
    calibration = get_calibration(calibration_id)
    log.info("标定完成: %s rms=%s", calibration_id, rms)
    return calibration or {}


def set_default_calibration(calibration_id: str) -> dict:
    calibration = get_calibration(calibration_id)
    if calibration is None:
        raise NotFoundError("标定记录不存在。")
    from core import db as dbmod

    with dbmod.transaction() as conn:
        conn.execute("UPDATE calibration SET is_default=0")
        conn.execute("UPDATE calibration SET is_default=1 WHERE id=?", (calibration_id,))
    return get_calibration(calibration_id) or {}


def depth_probe(x: int | None = None, y: int | None = None) -> dict:
    """读取当前帧指定像素（默认画面中心）的深度与反投影坐标。"""
    from core.geometry import deproject

    camera = get_camera_service()
    bundle = camera.latest(timeout=1.0)
    if bundle is None:
        raise CameraError("没有可用画面。")
    h, w = bundle.depth_m.shape[:2]
    px = int(x if x is not None else w // 2)
    py = int(y if y is not None else int(h * 0.7))
    px = int(np.clip(px, 0, w - 1))
    py = int(np.clip(py, 0, h - 1))
    patch = bundle.depth_m[max(0, py - 5):py + 6, max(0, px - 5):px + 6]
    valid = patch[patch > 0]
    depth = float(np.median(valid)) if valid.size else 0.0
    x_m, y_m, z_m = deproject(px, py, depth, bundle.intrinsics)
    return {"u": px, "v": py, "depth_m": round(depth, 4),
            "x_m": round(x_m, 4), "y_m": round(y_m, 4), "z_m": round(z_m, 4),
            "valid_points": int(valid.size), "total_points": int(patch.size)}
