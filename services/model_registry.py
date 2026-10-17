"""模型仓库：模型登记、权重哈希、模型卡片、历史训练产物导入。"""

from __future__ import annotations

from dataclasses import dataclass

import csv
import re
from pathlib import Path
from typing import Any

from core import db, storage
from core.config import settings
from core.exceptions import ConflictError, ModelError, NotFoundError, ValidationError
from core.logging_setup import get_logger

log = get_logger("services.models")

YOLO_SIZE_RE = re.compile(r"^yolo(v?)(\d+)([nstmlcxbc])$", re.IGNORECASE)
UNET_SCALE_RE = re.compile(r"^(Unet|resunet|Atteresunet|UNetV2)_(\d+)_(best|last)$", re.IGNORECASE)


def normalize_yolo_stem(stem: str) -> str:
    """把权重文件名归一化后交给 YOLO_SIZE_RE 匹配。

    覆盖项目 A 的命名实况：
      · `yolov11n_cloud.pt`  -> yolov11n
      · `yolov8n_1780651176` -> yolov8n
      · `yolo12l_best`       -> yolo12l
    """
    text = str(stem).strip()
    for suffix in ("_best", "_last", "_cloud"):
        if text.lower().endswith(suffix):
            text = text[: -len(suffix)]
    # 去掉结尾的纯数字时间戳后缀
    text = re.sub(r"_\d{6,}$", "", text)
    return text


@dataclass(frozen=True)
class ModelRegistration:
    """登记一个模型卡片所需的完整信息。"""

    name: str
    task_type: str
    weights_uri: str | None = None
    arch: str | None = None
    variant: str | None = None
    framework: str | None = None
    project_id: str | None = None
    num_classes: int | None = None
    class_names: list[str] | None = None
    metrics: dict | None = None
    notes: str = ""
    weight_origin: str = "imported"
    train_experiment_id: str | None = None
    source_filename: str | None = None


def register_model(registration: ModelRegistration) -> dict:
    name = str(registration.name).strip()
    task_type = registration.task_type
    weights_uri, arch = registration.weights_uri, registration.arch
    variant, framework = registration.variant, registration.framework
    project_id, num_classes = registration.project_id, registration.num_classes
    class_names, metrics = registration.class_names, registration.metrics
    notes, weight_origin = registration.notes, registration.weight_origin
    train_experiment_id = registration.train_experiment_id
    source_filename = registration.source_filename
    if not name:
        raise ValidationError("模型名称不能为空。")
    if task_type not in ("detection", "segmentation"):
        raise ValidationError(f"未知任务类型: {task_type}")
    if db.query_one("SELECT id FROM model WHERE name=? AND is_deleted=0", (name,)):
        raise ConflictError(f"模型名称已存在: {name}")

    path = storage.to_path(weights_uri) if weights_uri else None
    sha = None
    size_bytes = 0
    if path is not None and Path(path).exists():
        sha = storage.sha256_file(Path(path))
        size_bytes = Path(path).stat().st_size
        source_filename = source_filename or Path(path).name

    if framework is None:
        framework = "ultralytics" if task_type == "detection" else "pytorch"
    if num_classes is None:
        num_classes = settings.detection_num_classes() if task_type == "detection" else 1
    if class_names is None:
        class_names = (settings.detection_class_names() if task_type == "detection"
                       else ["背景", "可通行区域"])

    model_id = db.insert("model", {
        "project_id": project_id, "name": name, "task_type": task_type,
        "framework": framework, "arch": arch or name, "variant": variant,
        "input_size": ("640x640" if task_type == "detection"
                       else f"{settings.get('segmentation.input_width')}x{settings.get('segmentation.input_height')}"),
        "num_classes": num_classes, "class_names": class_names,
        "weights_uri": weights_uri, "weights_sha256": sha,
        "source_filename": source_filename, "weight_origin": weight_origin,
        "train_experiment_id": train_experiment_id, "metrics": metrics or {},
        "resource": {}, "size_bytes": size_bytes, "notes": notes,
    }, id_prefix="m")
    return get_model(model_id) or {}


def get_model(model_id: str) -> dict | None:
    row = db.query_one("SELECT * FROM model WHERE id=? AND is_deleted=0", (model_id,))
    if row is None:
        return None
    model = db.decode_row(row) or {}
    model["exists"] = bool(model.get("weights_uri")
                           and Path(storage.to_path(model["weights_uri"]) or "").exists())
    model["size_human"] = storage.human_size(model.get("size_bytes") or 0)
    model["usage"] = int(db.query_scalar("SELECT COUNT(*) FROM experiment WHERE model_id=? AND is_deleted=0",
                                          (model_id,), 0) or 0)
    return model


def list_models(page: int = 1, page_size: int = 30, task_type: str | None = None,
                arch: str | None = None, keyword: str = "") -> dict:
    clauses, params = ["is_deleted=0"], []
    if task_type:
        clauses.append("task_type=?")
        params.append(task_type)
    if arch:
        clauses.append("arch=?")
        params.append(arch)
    if keyword:
        clauses.append("(name LIKE ? OR arch LIKE ? OR source_filename LIKE ?)")
        params += [f"%{keyword}%"] * 3
    where = " AND ".join(clauses)
    result = db.paginate(
        f"SELECT * FROM model WHERE {where} ORDER BY task_type, arch, variant",
        f"SELECT COUNT(*) FROM model WHERE {where}", params, page, page_size)
    for item in result["items"]:
        item["metrics"] = db.decode_row({"metrics": item.get("metrics")}).get("metrics", {})
        item["size_human"] = storage.human_size(item.get("size_bytes") or 0)
    return result


def update_model(model_id: str, **fields: Any) -> int:
    allowed = {"name", "notes", "metrics", "resource", "variant", "weight_origin",
               "train_experiment_id", "project_id", "class_names", "num_classes"}
    payload = {k: v for k, v in fields.items() if k in allowed}
    return db.update("model", model_id, payload)


def delete_model(model_id: str) -> int:
    usage = int(db.query_scalar("SELECT COUNT(*) FROM experiment WHERE model_id=? AND is_deleted=0",
                                 (model_id,), 0) or 0)
    if usage:
        raise ConflictError(f"该模型被 {usage} 个实验引用，禁止删除（可先归档相关实验）。")
    return db.soft_delete("model", model_id)


def compare_models(model_ids: list[str]) -> dict:
    items = []
    for mid in model_ids:
        model = get_model(mid)
        if model:
            items.append(model)
    dimensions = sorted({k for item in items for k in (item.get("metrics") or {})})
    return {"items": items, "dimensions": dimensions}


# ================================================================== 产物导入
def scan_yolo_runs(runs_dir: str, project_id: str | None = None) -> dict:
    """扫描项目 A 的 runs/* 目录，提取指标并登记模型卡片。"""
    root = Path(runs_dir)
    if not root.exists():
        raise ModelError(f"runs 目录不存在: {runs_dir}")
    imported, skipped = 0, 0
    details: list[dict] = []

    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        name = run_dir.name
        base = name.replace("_cloud", "")
        match = YOLO_SIZE_RE.match(base)
        if not match:
            skipped += 1
            continue
        version, size = match.group(2), match.group(3)
        arch = f"yolo{version}{size}" if version in ("11", "12") else f"yolov{version}{size}"
        model_name = f"{arch}_trained"
        if db.query_one("SELECT id FROM model WHERE name=? AND is_deleted=0", (model_name,)):
            skipped += 1
            continue

        metrics = _read_yolo_metrics(run_dir / "results.csv")
        weights = _find_weight(run_dir)
        register_model(ModelRegistration(
            name=model_name, task_type="detection", arch=arch,
            variant=size, framework="ultralytics", project_id=project_id,
            weights_uri=storage.to_uri(weights) if weights else None,
            metrics=metrics, weight_origin="trained",
            source_filename=weights.name if weights else None,
            notes=f"由历史训练产物导入（{name}）",
        ))
        imported += 1
        details.append({"run": name, "arch": arch, "metrics": metrics,
                        "weights": weights.name if weights else None})
    return {"imported": imported, "skipped": skipped, "details": details}


def _read_yolo_metrics(results_csv: Path) -> dict:
    if not results_csv.exists():
        return {}
    try:
        with results_csv.open("r", encoding="utf-8", errors="ignore") as handle:
            rows = list(csv.DictReader(handle))
    except Exception:  # noqa: BLE001
        return {}
    if not rows:
        return {}
    best = {}
    best_map = -1.0
    for row in rows:
        cleaned = {str(k).strip(): v for k, v in row.items() if k}
        try:
            value = float(cleaned.get("metrics/mAP50(B)") or 0.0)
        except (TypeError, ValueError):
            value = 0.0
        if value >= best_map:
            best_map = value
            best = cleaned
    if not best:
        return {}

    def num(key: str, digits: int = 4):
        try:
            return round(float(best.get(key) or 0.0), digits)
        except (TypeError, ValueError):
            return None

    return {
        "mAP50": num("metrics/mAP50(B)"),
        "mAP50_95": num("metrics/mAP50-95(B)"),
        "precision": num("metrics/precision(B)"),
        "recall": num("metrics/recall(B)"),
        "epochs": int(float(best.get("epoch") or 0)) if best.get("epoch") else None,
    }


def _find_weight(run_dir: Path) -> Path | None:
    for candidate in (run_dir / "weights" / "best.pt", run_dir / "weights" / "last.pt"):
        if candidate.exists():
            return candidate
    found = sorted(run_dir.rglob("*.pt"))
    return found[0] if found else None


def scan_segmentation_weights(models_dir: str, project_id: str | None = None,
                              best_model_dir: str | None = None) -> dict:
    """扫描项目 B 的 models/*.pth（4 类结构 × 10 个训练规模 × best/last）。"""
    root = Path(models_dir)
    if not root.exists():
        raise ModelError(f"分割权重目录不存在: {models_dir}")
    metrics_index = _load_best_model_metrics(Path(best_model_dir) if best_model_dir else None)

    imported, skipped = 0, 0
    details: list[dict] = []
    for path in sorted(root.glob("*.pth")):
        match = UNET_SCALE_RE.match(path.stem)
        if not match:
            skipped += 1
            continue
        raw_arch, scale, tag = match.group(1), int(match.group(2)), match.group(3).lower()
        arch = {"unet": "unet", "resunet": "resunet",
                "atteresunet": "att_resunet", "unetv2": "unetv2"}.get(raw_arch.lower())
        if arch is None:
            skipped += 1
            continue
        model_name = f"{arch}_{scale}_{tag}"
        if db.query_one("SELECT id FROM model WHERE name=? AND is_deleted=0", (model_name,)):
            skipped += 1
            continue
        metrics = metrics_index.get((raw_arch.lower(), scale), {})
        register_model(ModelRegistration(
            name=model_name, task_type="segmentation", arch=arch, variant=str(scale),
            framework="pytorch", project_id=project_id,
            weights_uri=storage.to_uri(path), metrics=metrics, weight_origin="trained",
            source_filename=path.name,
            notes=f"由原项目 B 权重导入（训练样本数 {scale}，{tag}）",
        ))
        imported += 1
        details.append({"file": path.name, "arch": arch, "scale": scale,
                        "tag": tag, "metrics": metrics})
    return {"imported": imported, "skipped": skipped, "details": details[:40]}


def _metric_reader(row: dict):
    """返回绑定到指定 CSV 行的指标读取闭包（避免循环内闭包捕获到错误行）。"""

    def read(name: str):
        try:
            return round(float(row.get(name) or 0.0), 4)
        except (TypeError, ValueError):
            return None
    return read


def _load_best_model_metrics(best_model_dir: Path | None) -> dict:
    """读取项目 B best_model/*/**_val_results.csv 中的最优 IoU。"""
    index: dict[tuple, dict] = {}
    if best_model_dir is None or not best_model_dir.exists():
        return index
    for csv_path in best_model_dir.rglob("*val_results*.csv"):
        try:
            with csv_path.open("r", encoding="utf-8", errors="ignore") as handle:
                rows = list(csv.DictReader(handle))
        except Exception:  # noqa: BLE001
            continue
        if not rows:
            continue
        # 文件名形如 Unet_1000_val_results.csv
        stem = csv_path.stem.replace("_val_results", "").replace("_sorted", "")
        match = re.match(r"^([A-Za-z]+)_(\d+)", stem)
        if not match:
            continue
        key = (match.group(1).lower(), int(match.group(2)))
        best, best_iou = {}, -1.0
        for row in rows:
            try:
                iou = float(row.get("IoU") or row.get("iou") or 0.0)
            except (TypeError, ValueError):
                continue
            if iou > best_iou:
                best_iou, best = iou, row
        if not best:
            continue

        read_metric = _metric_reader(best)
        index[key] = {"miou": read_metric("IoU"), "dice": read_metric("Dice"),
                      "accuracy": read_metric("Accuracy"), "precision": read_metric("Precision"),
                      "recall": read_metric("Recall")}
    return index


def scan_platform_models(project_id: str | None = None) -> dict:
    """扫描平台模型目录 **以及 models.yaml 中登记的原始项目权重目录**。

    原项目 A 的 `models/pretrained/`（20 个官方权重）与 `models/trained/`（20 个自训练权重）
    会被按命名约定登记为模型卡片，供检测适配器直接查找与加载。
    """
    imported, skipped, details = 0, 0, []
    sources: list[tuple[str, Path]] = [
        ("pretrained", settings.storage_dir("models") / "pretrained"),
        ("trained", settings.storage_dir("models") / "trained"),
    ]
    for origin, key in (("pretrained", "detection_pretrained"), ("trained", "detection_trained")):
        external = settings.external_source(key)
        if external:
            sources.append((origin, external))

    for origin, folder in sources:
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.pt")):
            stem = normalize_yolo_stem(path.stem)
            match = YOLO_SIZE_RE.match(stem)
            if not match:
                skipped += 1
                continue
            version, size = match.group(2), match.group(3)
            arch = f"yolo{version}{size}" if version in ("11", "12") else f"yolov{version}{size}"
            model_name = f"{arch}_{origin}"
            if db.query_one("SELECT id FROM model WHERE name=? AND is_deleted=0", (model_name,)):
                skipped += 1
                continue
            note = (f"来源目录 {folder}"
                    if origin == "trained"
                    else f"官方预训练权重（COCO 80 类），仅作对照基线；"
                         f"用于本项目 11 类检测前需微调。来源目录 {folder}")
            register_model(ModelRegistration(
                name=model_name, task_type="detection", arch=arch, variant=size,
                project_id=project_id, weights_uri=storage.to_uri(path),
                weight_origin=origin, source_filename=path.name, notes=note))
            imported += 1
            details.append({"name": model_name, "file": path.name,
                            "size_mb": round(path.stat().st_size / 1024 / 1024, 1)})
    return {"imported": imported, "skipped": skipped, "details": details[:40]}


def model_stats() -> dict:
    rows = db.query_all("SELECT task_type, COUNT(*) AS n, "
                        "COALESCE(SUM(size_bytes),0) AS bytes FROM model "
                        "WHERE is_deleted=0 GROUP BY task_type")
    by_task = {r["task_type"]: {"count": r["n"], "bytes": r["bytes"]} for r in rows}
    arch_rows = db.query_all("SELECT arch, COUNT(*) AS n FROM model WHERE is_deleted=0 "
                             "GROUP BY arch ORDER BY n DESC LIMIT 30")
    return {
        "by_task": by_task,
        "total": sum(v["count"] for v in by_task.values()),
        "total_size": storage.human_size(sum(v["bytes"] for v in by_task.values())),
        "by_arch": {r["arch"]: r["n"] for r in arch_rows},
        "with_weights": int(db.query_scalar(
            "SELECT COUNT(*) FROM model WHERE is_deleted=0 AND weights_uri IS NOT NULL", (), 0) or 0),
    }
