"""数据集与标注管理：版本、导入、校验、划分、冻结、统计与历史数据迁移。"""

from __future__ import annotations

import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from core import db, media, storage
from core.config import settings
from core.exceptions import ConflictError, DatasetError, NotFoundError, ValidationError
from core.logging_setup import get_logger

log = get_logger("services.dataset")

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
TASK_TYPES = ("detection", "segmentation")


# ================================================================== 数据集
def create_dataset(name: str, task_type: str, project_id: str | None = None,
                   scene: str = "campus_road", class_names: list[str] | None = None,
                   foreground_values: list[int] | None = None,
                   description: str = "", created_by: str | None = None) -> dict:
    name = str(name).strip()
    if not name:
        raise ValidationError("数据集名称不能为空。")
    if task_type not in TASK_TYPES:
        raise ValidationError(f"未知任务类型: {task_type}")
    if db.query_one("SELECT id FROM dataset WHERE name=? AND is_deleted=0", (name,)):
        raise ConflictError(f"数据集名称已存在: {name}")
    dataset_id = db.insert("dataset", {
        "project_id": project_id, "name": name, "task_type": task_type, "scene": scene,
        "class_names": class_names or (
            settings.detection_class_names() if task_type == "detection"
            else ["背景", "可通行区域"]),
        "foreground_values": foreground_values if foreground_values is not None
        else settings.segmentation_foreground_values(),
        "description": description, "created_by": created_by, "status": "draft",
    }, id_prefix="ds")
    return get_dataset(dataset_id) or {}


def get_dataset(dataset_id: str) -> dict | None:
    row = db.query_one("SELECT * FROM dataset WHERE id=? AND is_deleted=0", (dataset_id,))
    if row is None:
        return None
    dataset = db.decode_row(row) or {}
    dataset["versions"] = list_versions(dataset_id)
    return dataset


def list_datasets(page: int = 1, page_size: int = 20, task_type: str | None = None,
                  project_id: str | None = None) -> dict:
    clauses, params = ["is_deleted=0"], []
    if task_type:
        clauses.append("task_type=?")
        params.append(task_type)
    if project_id:
        clauses.append("project_id=?")
        params.append(project_id)
    where = " AND ".join(clauses)
    result = db.paginate(
        f"SELECT * FROM dataset WHERE {where} ORDER BY updated_at DESC",
        f"SELECT COUNT(*) FROM dataset WHERE {where}", params, page, page_size)
    for item in result["items"]:
        item["version_count"] = int(db.query_scalar(
            "SELECT COUNT(*) FROM dataset_version WHERE dataset_id=?", (item["id"],), 0) or 0)
        item["image_count"] = int(db.query_scalar(
            "SELECT COALESCE(SUM(image_count),0) FROM dataset_version WHERE dataset_id=?",
            (item["id"],), 0) or 0)
    return result


def update_dataset(dataset_id: str, **fields: Any) -> int:
    allowed = {"name", "scene", "class_names", "foreground_values", "description", "status"}
    return db.update("dataset", dataset_id, {k: v for k, v in fields.items() if k in allowed})


def delete_dataset(dataset_id: str) -> int:
    return db.soft_delete("dataset", dataset_id)


def list_versions(dataset_id: str) -> list[dict]:
    rows = db.query_all("SELECT * FROM dataset_version WHERE dataset_id=? ORDER BY created_at DESC",
                        (dataset_id,))
    return db.decode_rows(rows)


# ================================================================== 版本
def create_version(dataset_id: str, version: str, source_root: str,
                   split_ratio: str = "7:2:1", split_seed: int = 42,
                   limit: int | None = None) -> dict:
    dataset = get_dataset(dataset_id)
    if dataset is None:
        raise NotFoundError("数据集不存在。")
    if db.query_one("SELECT id FROM dataset_version WHERE dataset_id=? AND version=?",
                    (dataset_id, version)):
        raise ConflictError(f"版本已存在: {version}")
    root = Path(source_root)
    if not root.exists():
        raise DatasetError(f"数据源目录不存在: {source_root}")

    version_id = db.insert("dataset_version", {
        "dataset_id": dataset_id, "version": version, "source_root": str(root),
        "split_ratio": split_ratio, "split_seed": split_seed, "split_manifest_uri": None,
    }, id_prefix="dv")
    db.update("dataset", dataset_id, {"latest_version": version, "status": "draft"})
    result = scan_images(version_id, root, limit=limit)
    return {"version_id": version_id, "version": version, **result}


def _mask_for(image_path: Path, mask_dir: Path | None) -> Path | None:
    if mask_dir is None:
        return None
    stem = image_path.stem
    for ext in (".png", ".jpg", ".jpeg", ".bmp"):
        candidate = mask_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def scan_images(version_id: str, root: Path, mask_dir: Path | None = None,
                limit: int | None = None) -> dict:
    """扫描目录并登记图像（含掩膜匹配），返回统计。"""
    root = Path(root)
    mask_dir = Path(mask_dir) if mask_dir else None
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXT)
    if limit:
        files = files[:limit]
    if not files:
        raise DatasetError(f"目录中没有找到图像文件: {root}")

    rows = []
    for path in files:
        try:
            stat = path.stat()
        except OSError:
            continue
        mask = _mask_for(path, mask_dir)
        rows.append({
            "dataset_version_id": version_id,
            "uri": storage.to_uri(path),
            "mask_uri": storage.to_uri(mask) if mask else None,
            "sha256": None,
            "width": None, "height": None,
            "split": "train",
            "source": "import",
            "meta": {"size_bytes": stat.st_size, "mtime": int(stat.st_mtime)},
        })

    # 批量写入（单事务），避免逐条提交造成的性能损失
    stamp = db.now_iso()
    payload = [{
        "id": db.new_id("img"),
        "dataset_version_id": r["dataset_version_id"],
        "uri": r["uri"], "mask_uri": r["mask_uri"], "sha256": None,
        "width": None, "height": None, "split": "train", "source": "import",
        "capture_session_id": None, "illum": None, "target_size_group": None,
        "meta": __import__("json").dumps(r["meta"], ensure_ascii=False),
        "created_at": stamp,
    } for r in rows]
    with db.transaction() as conn:
        conn.executemany(
            "INSERT INTO image (id, dataset_version_id, uri, mask_uri, sha256, width, height, "
            "split, source, capture_session_id, illum, target_size_group, meta, created_at) "
            "VALUES (:id, :dataset_version_id, :uri, :mask_uri, :sha256, :width, :height, "
            ":split, :source, :capture_session_id, :illum, :target_size_group, :meta, :created_at)",
            payload)

    content_hash = storage.sha256_tree([Path(storage.to_path(r["uri"]) or "") for r in rows])
    db.update("dataset_version", version_id, {
        "image_count": len(rows), "content_hash": content_hash,
    })
    return {"images": len(rows), "masked": sum(1 for r in rows if r.get("mask_uri")),
            "content_hash": content_hash}


def attach_masks(version_id: str, mask_dir: str) -> dict:
    """为已有图像批量匹配掩膜（供分割数据集导入）。"""
    mask_root = Path(mask_dir)
    if not mask_root.exists():
        raise DatasetError(f"掩膜目录不存在: {mask_dir}")
    # 建立掩膜索引，避免对每张图都做文件系统探测
    index: dict[str, Path] = {}
    for ext in (".png", ".jpg", ".jpeg", ".bmp"):
        for path in mask_root.rglob(f"*{ext}"):
            index.setdefault(path.stem, path)

    images = db.query_all("SELECT id, uri FROM image WHERE dataset_version_id=?", (version_id,))
    updates = []
    for row in images:
        path = storage.to_path(row["uri"])
        if path is None:
            continue
        mask = index.get(Path(path).stem)
        if mask is not None:
            updates.append({"id": row["id"], "mask_uri": storage.to_uri(mask)})
    if updates:
        with db.transaction() as conn:
            conn.executemany("UPDATE image SET mask_uri=:mask_uri WHERE id=:id", updates)
    db.update("dataset_version", version_id, {"annotation_count": len(updates)})
    return {"matched": len(updates), "total": len(images)}


def validate_version(version_id: str, sample_limit: int = 400) -> dict:
    """数据完整性校验：掩膜前景值口径、漏标、越界、覆盖率异常。"""
    version = db.query_one("SELECT * FROM dataset_version WHERE id=?", (version_id,))
    if version is None:
        raise NotFoundError("数据集版本不存在。")
    dataset = db.query_one("SELECT * FROM dataset WHERE id=?", (version["dataset_id"],))
    task_type = (dataset or {}).get("task_type", "detection")
    allowed_fg = set(settings.segmentation_foreground_values())

    images = db.query_all("SELECT * FROM image WHERE dataset_version_id=?", (version_id,))
    total = len(images)
    issues: list[dict] = []
    fg_counter: Counter = Counter()
    area_ratios: list[float] = []
    checked = 0

    for row in images[:sample_limit]:
        path = storage.to_path(row["uri"])
        if path is None or not Path(path).exists():
            issues.append({"type": "missing_image", "image_id": row["id"], "detail": row["uri"]})
            continue
        checked += 1
        if task_type == "segmentation":
            if not row.get("mask_uri"):
                issues.append({"type": "missing_mask", "image_id": row["id"], "detail": "无同名掩膜"})
                continue
            mask_path = storage.to_path(row["mask_uri"])
            raw = media.read_depth16(mask_path, 1.0) if mask_path else None
            if raw is None:
                issues.append({"type": "unreadable_mask", "image_id": row["id"]})
                continue
            values = np.unique(raw.astype(np.int32))
            fg_values = [int(v) for v in values if int(v) != 0]
            for v in fg_values:
                fg_counter[v] += 1
            unexpected = [v for v in fg_values if v not in allowed_fg]
            if unexpected:
                issues.append({"type": "fg_value_unexpected", "image_id": row["id"],
                               "detail": f"前景值 {unexpected} 不在白名单 {sorted(allowed_fg)}"})
            if len(fg_values) > 1:
                issues.append({"type": "fg_value_ambiguous", "image_id": row["id"],
                               "detail": f"同一掩膜出现多个前景值 {fg_values}（口径不一致）"})
            ratio = float((raw > 0).mean())
            area_ratios.append(ratio)
            if ratio < 0.02 or ratio > 0.98:
                issues.append({"type": "mask_area_abnormal", "image_id": row["id"],
                               "detail": f"掩膜占比 {ratio:.4f}"})
        else:
            ann = db.query_one("SELECT * FROM annotation WHERE image_id=? AND task_type='detection'",
                               (row["id"],))
            if ann is None:
                issues.append({"type": "missing_label", "image_id": row["id"], "detail": "无检测标注"})

    by_type = Counter(i["type"] for i in issues)
    report = {
        "version_id": version_id,
        "task_type": task_type,
        "total_images": total,
        "checked_images": checked,
        "issue_count": len(issues),
        "issue_types": dict(by_type),
        "foreground_value_histogram": {str(k): v for k, v in sorted(fg_counter.items())},
        "foreground_values_allowed": sorted(allowed_fg),
        "foreground_value_consistent": len(fg_counter) <= 1,
        "mask_area_ratio": {
            "min": round(min(area_ratios), 4) if area_ratios else None,
            "max": round(max(area_ratios), 4) if area_ratios else None,
            "mean": round(float(np.mean(area_ratios)), 4) if area_ratios else None,
        },
        "samples": issues[:50],
        "conclusion": "通过" if not issues else f"发现 {len(issues)} 处问题，建议修复后再冻结",
    }
    db.update("dataset_version", version_id, {"validation_report": report})
    return report


def unify_foreground(version_id: str, target_value: int | None = None,
                     sample_limit: int = 2000) -> dict:
    """把掩膜里的多个前景值统一为目标值（修复项目 B 的 17 / >0 口径冲突）。"""
    target = int(target_value or settings.get("segmentation.primary_foreground_value", 17))
    images = db.query_all(
        "SELECT id, mask_uri FROM image WHERE dataset_version_id=? AND mask_uri IS NOT NULL",
        (version_id,))
    changed = 0
    for row in images[:sample_limit]:
        path = storage.to_path(row["mask_uri"])
        if path is None or not Path(path).exists():
            continue
        raw = media.read_depth16(path, 1.0)
        if raw is None:
            continue
        binary = (raw > 0).astype(np.uint8)
        if not binary.any():
            continue
        unified = (binary * target).astype(np.uint8)
        original = raw.astype(np.uint8)
        if not np.array_equal(original, unified):
            media.write_depth16(path, unified.astype(np.float32), 1.0)
            changed += 1
    db.query_one("SELECT 1")  # 保持连接活跃
    return {"unified": changed, "target_value": target, "total": len(images)}


def split_version(version_id: str, ratio: str = "7:2:1", seed: int = 42) -> dict:
    """划分 train/val/test 并写入 split.json（固定随机种子，保证可复现）。"""
    parts = [float(x) for x in str(ratio).replace("：", ":").split(":")]
    if len(parts) != 3 or sum(parts) <= 0:
        raise ValidationError("划分比例格式应为 '7:2:1'。")
    total_ratio = sum(parts)
    images = db.query_all("SELECT id FROM image WHERE dataset_version_id=?", (version_id,))
    ids = [row["id"] for row in images]
    if not ids:
        raise DatasetError("该版本没有图像，无法划分。")

    rng = random.Random(int(seed))
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * parts[0] / total_ratio)
    n_val = int(n * parts[1] / total_ratio)
    buckets = {
        "train": ids[:n_train],
        "val": ids[n_train:n_train + n_val],
        "test": ids[n_train + n_val:],
    }
    for split, items in buckets.items():
        for image_id in items:
            db.update("image", image_id, {"split": split})

    manifest = {"seed": int(seed), "ratio": ratio,
                "counts": {k: len(v) for k, v in buckets.items()},
                "train": buckets["train"][:50], "val": buckets["val"][:50], "test": buckets["test"][:50]}
    out_dir = settings.storage_dir("datasets") / version_id
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "split.json"
    import json

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    db.update("dataset_version", version_id, {
        "split_ratio": ratio, "split_seed": int(seed),
        "split_manifest_uri": storage.to_uri(manifest_path),
    })
    return manifest


def freeze_version(version_id: str) -> dict:
    version = db.query_one("SELECT * FROM dataset_version WHERE id=?", (version_id,))
    if version is None:
        raise NotFoundError("数据集版本不存在。")
    report = db.decode_row(version).get("validation_report") or {}
    stamp = db.now_iso()
    db.update("dataset_version", version_id, {"frozen_at": stamp})
    db.update("dataset", version["dataset_id"], {"status": "frozen"})
    return {"version_id": version_id, "frozen_at": stamp,
            "validation_passed": report.get("conclusion") == "通过"}


def version_stats(version_id: str, sample_limit: int = 300) -> dict:
    rows = db.query_all("SELECT split, COUNT(*) AS n FROM image WHERE dataset_version_id=? "
                        "GROUP BY split", (version_id,))
    splits = {r["split"] or "train": r["n"] for r in rows}
    version = db.query_one("SELECT * FROM dataset_version WHERE id=?", (version_id,))
    dataset = db.query_one("SELECT * FROM dataset WHERE id=?", (version["dataset_id"],)) if version else None
    task_type = (dataset or {}).get("task_type", "detection")

    class_counter: Counter = Counter()
    size_bins: Counter = Counter()
    samples = db.query_all("SELECT * FROM image WHERE dataset_version_id=? LIMIT ?",
                           (version_id, sample_limit))
    for row in samples:
        ann = db.query_one("SELECT payload FROM annotation WHERE image_id=? LIMIT 1", (row["id"],))
        if ann:
            import json

            try:
                payload = json.loads(ann["payload"]) if isinstance(ann["payload"], str) else ann["payload"]
            except Exception:  # noqa: BLE001
                payload = []
            for item in payload or []:
                class_counter[str(item.get("class_name", "unknown"))] += 1
        if row.get("target_size_group"):
            size_bins[str(row["target_size_group"])] += 1

    return {
        "version_id": version_id,
        "task_type": task_type,
        "splits": splits,
        "total": sum(splits.values()),
        "class_distribution": dict(class_counter.most_common(30)),
        "size_group_distribution": dict(size_bins),
        "content_hash": (version or {}).get("content_hash"),
    }


def list_images(version_id: str, page: int = 1, page_size: int = 60,
                split: str | None = None) -> dict:
    clauses, params = ["dataset_version_id=?"], [version_id]
    if split:
        clauses.append("split=?")
        params.append(split)
    where = " AND ".join(clauses)
    result = db.paginate(
        f"SELECT * FROM image WHERE {where} ORDER BY uri",
        f"SELECT COUNT(*) FROM image WHERE {where}", params, page, page_size)
    for item in result["items"]:
        item["has_mask"] = bool(item.get("mask_uri"))
    return result


def version_diff(from_version: str, to_version: str, limit: int = 200) -> dict:
    def load(vid: str) -> dict[str, str]:
        rows = db.query_all("SELECT uri, mask_uri FROM image WHERE dataset_version_id=?", (vid,))
        return {r["uri"]: r.get("mask_uri") or "" for r in rows}

    left, right = load(from_version), load(to_version)
    added = [u for u in right if u not in left][:limit]
    removed = [u for u in left if u not in right][:limit]
    changed = [u for u in right if u in left and right[u] != left[u]][:limit]
    return {"from": from_version, "to": to_version,
            "added": added, "removed": removed, "changed": changed,
            "counts": {"added": len(added), "removed": len(removed), "changed": len(changed),
                       "from_total": len(left), "to_total": len(right)}}


# ================================================================== 标注
def import_annotations(version_id: str, source_dir: str, fmt: str = "yolo",
                       class_names: list[str] | None = None) -> dict:
    """导入标注。支持 yolo(txt) / labelme(json) / coco(json) / voc(xml) / mask(png)。"""
    import json

    src = Path(source_dir)
    if not src.exists():
        raise DatasetError(f"标注目录不存在: {source_dir}")
    class_names = list(class_names or settings.detection_class_names())
    images = db.query_all("SELECT id, uri FROM image WHERE dataset_version_id=?", (version_id,))
    by_stem = {Path(r["uri"]).stem: r["id"] for r in images}

    imported = 0
    skipped = 0
    fmt = str(fmt).lower()

    if fmt == "coco":
        data_file = next(iter(src.glob("*.json")), None)
        if data_file is None:
            raise DatasetError("COCO 目录下未找到 json 文件。")
        data = json.loads(data_file.read_text(encoding="utf-8"))
        categories = {c["id"]: c["name"] for c in data.get("categories", [])}
        grouping: dict[str, list[dict]] = {}
        for ann in data.get("annotations", []):
            grouping.setdefault(str(ann["image_id"]), []).append(ann)
        for img in data.get("images", []):
            stem = Path(img["file_name"]).stem
            image_id = by_stem.get(stem)
            if image_id is None:
                skipped += 1
                continue
            payload = []
            for ann in grouping.get(str(img["id"]), []):
                x, y, w, h = ann.get("bbox", [0, 0, 0, 0])
                payload.append({"class_name": categories.get(ann.get("category_id"), "unknown"),
                                "bbox": [x, y, x + w, y + h], "conf": 1.0})
            db.insert("annotation", {"image_id": image_id, "task_type": "detection",
                                     "source_format": "coco", "payload": payload,
                                     "is_verified": 1}, id_prefix="ann")
            imported += 1
        return {"imported": imported, "skipped": skipped, "format": fmt}

    files = [p for p in src.rglob("*") if p.suffix.lower() in (".txt", ".json", ".xml", ".png")]
    for path in files:
        stem = path.stem
        image_id = by_stem.get(stem)
        if image_id is None:
            skipped += 1
            continue
        payload: list[dict] = []
        if fmt == "yolo" and path.suffix == ".txt":
            payload = _parse_yolo(path, images, image_id, class_names)
        elif fmt == "labelme" and path.suffix == ".json":
            payload = _parse_labelme(path)
        elif fmt == "voc" and path.suffix == ".xml":
            payload = _parse_voc(path)
        elif fmt == "mask" and path.suffix == ".png":
            payload = [{"mask": True}]
            db.update("image", image_id, {"mask_uri": storage.to_uri(path)})
        else:
            continue
        if not payload:
            skipped += 1
            continue
        db.insert("annotation", {"image_id": image_id, "task_type": "detection",
                                 "source_format": fmt, "payload": payload,
                                 "is_verified": 0}, id_prefix="ann")
        imported += 1

    db.update("dataset_version", version_id, {"annotation_count": imported})
    return {"imported": imported, "skipped": skipped, "format": fmt}


def _parse_yolo(path: Path, images: list[dict], image_id: str, class_names: list[str]) -> list[dict]:
    row = next((r for r in images if r["id"] == image_id), None)
    width = height = 0
    if row:
        p = storage.to_path(row["uri"])
        img = media.read_image(p) if p else None
        if img is not None:
            height, width = img.shape[:2]
    if not width or not height:
        return []
    out = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cid = int(float(parts[0]))
            cx, cy, bw, bh = [float(v) for v in parts[1:5]]
        except ValueError:
            continue
        x1 = (cx - bw / 2) * width
        y1 = (cy - bh / 2) * height
        x2 = (cx + bw / 2) * width
        y2 = (cy + bh / 2) * height
        name = class_names[cid] if 0 <= cid < len(class_names) else f"class_{cid}"
        out.append({"class_id": cid, "class_name": name, "bbox": [x1, y1, x2, y2], "conf": 1.0})
    return out


def _translate_class(raw: str) -> str:
    """把英文别名（如 crosswalk / vehicle）映射回平台的中文类别名。"""
    text = str(raw or "").strip()
    if not text:
        return "unknown"
    if text in settings.detection_class_names():
        return text
    low = text.lower()
    aliases = settings.get("detection.aliases", {}) or {}
    for cn, names in aliases.items():
        candidates = names if isinstance(names, list) else [names]
        for candidate in candidates:
            if str(candidate).lower() == low:
                return str(cn)
    return text


def _parse_labelme(path: Path) -> list[dict]:
    import json

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []
    out = []
    for shape in data.get("shapes", []):
        points = shape.get("points") or []
        if len(points) < 2:
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        out.append({"class_name": _translate_class(shape.get("label", "unknown")),
                    "raw_label": shape.get("label"),
                    "bbox": [min(xs), min(ys), max(xs), max(ys)], "conf": 1.0})
    return out


def _parse_voc(path: Path) -> list[dict]:
    import xml.etree.ElementTree as ET

    try:
        root = ET.parse(str(path)).getroot()
    except Exception:  # noqa: BLE001
        return []
    out = []
    for obj in root.findall("object"):
        name = obj.findtext("name") or "unknown"
        box = obj.find("bndbox")
        if box is None:
            continue
        try:
            out.append({"class_name": name, "conf": 1.0, "bbox": [
                float(box.findtext("xmin")), float(box.findtext("ymin")),
                float(box.findtext("xmax")), float(box.findtext("ymax"))]})
        except (TypeError, ValueError):
            continue
    return out


def list_annotations(image_id: str) -> list[dict]:
    rows = db.query_all("SELECT * FROM annotation WHERE image_id=?", (image_id,))
    return db.decode_rows(rows)


# ================================================================== 历史数据迁移
def _resolve_labels_dir(images_dir: Path) -> Path | None:
    """从图像目录推断标注目录。

    原项目 A 的目录结构是 `<dataset>/images/train` 与 `<dataset>/labels/train`，
    因此需要向上寻找同级的 labels 目录，而不是在图像目录内部查找。
    """
    candidates: list[Path] = []
    # 同一数据集下的 labels/<split>
    if images_dir.parent.name in ("images", "image", "imgs"):
        base = images_dir.parent.parent
        candidates.append(base / "labels" / images_dir.name)
        candidates.append(base / "labels")
        candidates.append(base / "annotations" / images_dir.name)
        candidates.append(base / "annotations")
    candidates.append(images_dir / "labels")
    candidates.append(images_dir.parent / "labels")
    candidates.append(images_dir.parent.parent / "labels")
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def import_legacy_detection_dataset(source_dir: str, dataset_name: str = "road-target-det-v1",
                                    limit: int | None = None, fmt: str = "labelme",
                                    labels_dir: str | None = None) -> dict:
    """导入项目 A 的图像与标注（datasets/images + datasets/labels）。

    原项目 A 的标签目录同时含 LabelMe JSON 与 YOLO txt，默认优先使用 JSON（含原始框坐标）。
    """
    src = Path(source_dir)
    images_dir = src / "images" if (src / "images").exists() else src
    detected_labels = Path(labels_dir) if labels_dir else _resolve_labels_dir(images_dir)

    dataset = _ensure_dataset(dataset_name, "detection", f"来自原项目 A 的 {images_dir}")
    version = f"v{len(list_versions(dataset['id'])) + 1}.0"
    created = create_version(dataset["id"], version, str(images_dir), limit=limit)
    version_id = created["version_id"]

    imported, use_fmt = 0, None
    if detected_labels and detected_labels.exists():
        has_json = any(detected_labels.rglob("*.json"))
        has_txt = any(detected_labels.rglob("*.txt"))
        if fmt == "labelme" and has_json:
            use_fmt = "labelme"
        elif has_txt:
            use_fmt = "yolo"
        elif has_json:
            use_fmt = "labelme"
        if use_fmt:
            res = import_annotations(version_id, str(detected_labels), fmt=use_fmt,
                                     class_names=settings.detection_class_names())
            imported = res.get("imported", 0)

    images = db.query_all("SELECT id, uri FROM image WHERE dataset_version_id=?", (version_id,))
    for row in images:
        stem = Path(row["uri"]).stem
        cls = _guess_category_from_name(stem)
        db.update("image", row["id"], {
            "target_size_group": settings.class_group(cls) if cls else None,
            "source": "legacy_project_a",
            "illum": "sunny",
        })
    return {"dataset_id": dataset["id"], "version_id": version_id, "version": version,
            "images": created["images"], "annotations": imported,
            "labels_dir": str(detected_labels) if detected_labels else None,
            "annotation_format": use_fmt}


def _ensure_dataset(name: str, task_type: str, description: str = "",
                    scene: str = "campus_road") -> dict:
    existing = db.query_one("SELECT * FROM dataset WHERE name=? AND is_deleted=0", (name,))
    if existing:
        return db.decode_row(existing)
    return create_dataset(name, task_type, scene=scene, description=description)


def _guess_category_from_name(stem: str) -> str | None:
    """从文件名推断类别（原项目 A 的截图命名含类别英文别名）。"""
    low = stem.lower()
    aliases = settings.get("detection.aliases", {}) or {}
    for cn, names in aliases.items():
        for alias in (names if isinstance(names, list) else [names]):
            if str(alias).lower() in low:
                return str(cn)
    return None


def import_legacy_segmentation_dataset(images_dir: str, masks_dir: str,
                                       dataset_name: str = "campus-road-seg") -> dict:
    """导入项目 B 的道路分割数据集（images + masks 同名）。"""
    images_path, masks_path = Path(images_dir), Path(masks_dir)
    if not images_path.exists():
        raise DatasetError(f"图像目录不存在: {images_dir}")
    dataset = _ensure_dataset(dataset_name, "segmentation",
                              "来自原项目 B 的校园道路分割数据集", scene="campus_road")
    version = f"v{len(list_versions(dataset['id'])) + 1}.0"
    created = create_version(dataset["id"], version, str(images_path))
    version_id = created["version_id"]
    masked = 0
    if masks_path.exists():
        masked = attach_masks(version_id, str(masks_path))["matched"]
    report = validate_version(version_id)
    split_version(version_id, "8:1:1", seed=42)
    return {"dataset_id": dataset["id"], "version_id": version_id, "version": version,
            "images": created["images"], "masks": masked, "validation": report["conclusion"],
            "foreground_histogram": report["foreground_value_histogram"]}


# ---------------------------------------------------------------- 保留原始划分的导入
SPLIT_NAMES = ("train", "val", "test")


def _split_of(path: Path, root: Path) -> str:
    """从相对路径推断原项目的数据划分（train / val / test）。"""
    try:
        parts = [p.lower() for p in path.relative_to(root).parts]
    except ValueError:
        parts = [p.lower() for p in path.parts]
    for part in parts:
        for name in SPLIT_NAMES:
            if part == name or part.startswith(name + "_"):
                return name
    return "train"


def import_legacy_segmentation_full(data_root: str, dataset_name: str = "campus-road-seg",
                                    version_tag: str = "v1.0") -> dict:
    """导入项目 B 的完整分割数据集（train + val + test），**保留原始划分**。

    目录约定：`<data_root>/dataset/<split>/images/*.jpg` 与 `<data_root>/dataset/<split>/masks/*.png`。
    """
    root = Path(data_root)
    dataset_dir = root / "dataset" if (root / "dataset").exists() else root
    if not dataset_dir.exists():
        raise DatasetError(f"数据根目录不存在: {data_root}")

    image_files = sorted(p for p in dataset_dir.rglob("*")
                         if p.suffix.lower() in IMAGE_EXT and p.parent.name == "images")
    if not image_files:
        raise DatasetError(f"未在 {dataset_dir} 下找到 <split>/images 结构。")

    dataset = _ensure_dataset(dataset_name, "segmentation",
                              f"来自原项目 B 的校园道路分割数据集（{dataset_dir}）",
                              scene="campus_road")
    existing = db.query_one("SELECT id FROM dataset_version WHERE dataset_id=? AND version=?",
                            (dataset["id"], version_tag))
    if existing:
        raise ConflictError(f"版本已存在: {version_tag}（如需重导请先删除该版本）")

    version_id = db.insert("dataset_version", {
        "dataset_id": dataset["id"], "version": version_tag,
        "source_root": str(dataset_dir), "split_ratio": "原始划分",
        "split_seed": 0, "split_manifest_uri": None,
    }, id_prefix="dv")
    db.update("dataset", dataset["id"], {"latest_version": version_tag, "status": "draft"})

    # 掩膜索引（只取 <split>/masks 下的文件，按文件名匹配）
    mask_index: dict[str, Path] = {}
    for ext in (".png", ".jpg", ".jpeg", ".bmp"):
        for path in dataset_dir.rglob(f"*{ext}"):
            if path.parent.name == "masks":
                mask_index.setdefault(path.stem, path)

    stamp = db.now_iso()
    payload, split_counts = [], {name: 0 for name in SPLIT_NAMES}
    for path in image_files:
        split = _split_of(path, dataset_dir)
        split_counts[split] = split_counts.get(split, 0) + 1
        try:
            stat = path.stat()
        except OSError:
            continue
        mask = mask_index.get(path.stem)
        payload.append({
            "id": db.new_id("img"), "dataset_version_id": version_id,
            "uri": storage.to_uri(path), "mask_uri": storage.to_uri(mask) if mask else None,
            "sha256": None, "width": None, "height": None, "split": split,
            "source": "legacy_project_b", "capture_session_id": None, "illum": "sunny",
            "target_size_group": None,
            "meta": __import__("json").dumps({"size_bytes": stat.st_size,
                                              "orig_split": split}, ensure_ascii=False),
            "created_at": stamp,
        })
    with db.transaction() as conn:
        conn.executemany(
            "INSERT INTO image (id, dataset_version_id, uri, mask_uri, sha256, width, height, "
            "split, source, capture_session_id, illum, target_size_group, meta, created_at) "
            "VALUES (:id, :dataset_version_id, :uri, :mask_uri, :sha256, :width, :height, "
            ":split, :source, :capture_session_id, :illum, :target_size_group, :meta, :created_at)",
            payload)

    content_hash = storage.sha256_tree([Path(storage.to_path(r["uri"]) or "") for r in payload])
    masked = sum(1 for r in payload if r["mask_uri"])
    db.update("dataset_version", version_id, {
        "image_count": len(payload), "annotation_count": masked, "content_hash": content_hash,
    })

    manifest = {"source": "legacy_project_b", "seed": None, "ratio": "原始划分",
                "counts": split_counts, "content_hash": content_hash}
    out_dir = settings.storage_dir("datasets") / version_id
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "split.json"
    import json

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    db.update("dataset_version", version_id, {"split_manifest_uri": storage.to_uri(manifest_path)})

    report = validate_version(version_id)
    return {"dataset_id": dataset["id"], "version_id": version_id, "version": version_tag,
            "images": len(payload), "masks": masked, "splits": split_counts,
            "content_hash": content_hash, "validation": report["conclusion"],
            "foreground_histogram": report["foreground_value_histogram"]}


def import_legacy_detection_full(images_root: str, labels_root: str | None = None,
                                 dataset_name: str = "road-target-det",
                                 version_tag: str = "v1.0") -> dict:
    """导入项目 A 的完整检测数据集（train + val），**保留原始划分**。

    目录约定：`<images_root>/<split>/*.jpg` 与 `<labels_root>/<split>/*.json|*.txt`。
    """
    images_dir = Path(images_root)
    if not images_dir.exists():
        raise DatasetError(f"图像根目录不存在: {images_root}")
    labels_dir = Path(labels_root) if labels_root else _resolve_labels_dir(images_dir)

    image_files = sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXT)
    if not image_files:
        raise DatasetError(f"未在 {images_dir} 下找到图像。")

    dataset = _ensure_dataset(dataset_name, "detection",
                              f"来自原项目 A 的道路目标检测数据集（{images_dir}）")
    existing = db.query_one("SELECT id FROM dataset_version WHERE dataset_id=? AND version=?",
                            (dataset["id"], version_tag))
    if existing:
        raise ConflictError(f"版本已存在: {version_tag}（如需重导请先删除该版本）")

    version_id = db.insert("dataset_version", {
        "dataset_id": dataset["id"], "version": version_tag,
        "source_root": str(images_dir), "split_ratio": "原始划分",
        "split_seed": 0, "split_manifest_uri": None,
    }, id_prefix="dv")
    db.update("dataset", dataset["id"], {"latest_version": version_tag, "status": "draft"})

    stamp = db.now_iso()
    payload, split_counts = [], {name: 0 for name in SPLIT_NAMES}
    for path in image_files:
        split = _split_of(path, images_dir)
        split_counts[split] = split_counts.get(split, 0) + 1
        try:
            stat = path.stat()
        except OSError:
            continue
        payload.append({
            "id": db.new_id("img"), "dataset_version_id": version_id,
            "uri": storage.to_uri(path), "mask_uri": None, "sha256": None,
            "width": None, "height": None, "split": split, "source": "legacy_project_a",
            "capture_session_id": None, "illum": "sunny", "target_size_group": None,
            "meta": __import__("json").dumps({"size_bytes": stat.st_size,
                                              "orig_split": split}, ensure_ascii=False),
            "created_at": stamp,
        })
    with db.transaction() as conn:
        conn.executemany(
            "INSERT INTO image (id, dataset_version_id, uri, mask_uri, sha256, width, height, "
            "split, source, capture_session_id, illum, target_size_group, meta, created_at) "
            "VALUES (:id, :dataset_version_id, :uri, :mask_uri, :sha256, :width, :height, "
            ":split, :source, :capture_session_id, :illum, :target_size_group, :meta, :created_at)",
            payload)

    content_hash = storage.sha256_tree([Path(storage.to_path(r["uri"]) or "") for r in payload])
    db.update("dataset_version", version_id, {
        "image_count": len(payload), "content_hash": content_hash,
    })

    imported = 0
    if labels_dir and labels_dir.exists():
        imported = _import_annotations_by_split(version_id, labels_dir, images_dir)

    manifest = {"source": "legacy_project_a", "seed": None, "ratio": "原始划分",
                "counts": split_counts, "content_hash": content_hash}
    out_dir = settings.storage_dir("datasets") / version_id
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "split.json"
    import json

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    db.update("dataset_version", version_id, {"split_manifest_uri": storage.to_uri(manifest_path)})

    report = validate_version(version_id)
    return {"dataset_id": dataset["id"], "version_id": version_id, "version": version_tag,
            "images": len(payload), "annotations": imported, "splits": split_counts,
            "labels_dir": str(labels_dir) if labels_dir else None,
            "content_hash": content_hash, "validation": report["conclusion"]}


def _import_annotations_by_split(version_id: str, labels_dir: Path, images_dir: Path) -> int:
    """按 split 分别导入标注（LabelMe JSON 优先，回退 YOLO txt）。"""
    total = 0
    for split in SPLIT_NAMES:
        for candidate in (labels_dir / split, labels_dir):
            if not candidate.exists():
                continue
            has_json = any(candidate.glob("*.json"))
            has_txt = any(candidate.glob("*.txt"))
            fmt = "labelme" if has_json else ("yolo" if has_txt else None)
            if fmt is None:
                continue
            result = import_annotations(version_id, str(candidate), fmt=fmt,
                                        class_names=settings.detection_class_names())
            total += result.get("imported", 0)
            break
    db.update("dataset_version", version_id, {"annotation_count": total})
    return total
