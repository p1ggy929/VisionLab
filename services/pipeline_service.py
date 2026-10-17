"""运行期流水线：相机帧 → 检测/分割 → 统一测距 → 叠加渲染 → 落库。

这是平台的运行时核心，把两条算法路线、统一测距内核与实验记录串起来。
"""

from __future__ import annotations

from dataclasses import dataclass

import threading
import time
from typing import Any

import numpy as np

from algorithms import registry
from algorithms.calibration import CalibrationPlan
from algorithms.ranging import (BoxBoundaryStrategy, DepthSampler, MaskBoundaryStrategy,
                                road_summary, summarize_per_detection)
from algorithms.ranging.base import attach_visual_hints
from core import db, media
from core.camera.base import FrameBundle
from core.camera.service import get_camera_service
from core.config import settings
from core.exceptions import ModelError, ValidationError
from core.logging_setup import get_logger

log = get_logger("services.pipeline")

MODES = ("detection", "segmentation", "both")


@dataclass(frozen=True)
class MeasurementRecord:
    """一条待落库的测距结果（含人工真值）。"""

    target_kind: str
    metric_kind: str
    result: dict
    actual_value: float | None = None
    actual_source: str = "tape"
    category: str | None = None
    notes: str = ""
    illumination: str | None = None
    media_uri: str | None = None
    detection_result_id: str | None = None
    segmentation_result_id: str | None = None


class PipelineService:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._detector = None
        self._segmenter = None
        self._detector_key = "depth_obstacle"
        self._segmenter_key = "classical_road"
        self._calibration: CalibrationPlan | None = None

        self.config: dict[str, Any] = {
            "mode": "both",
            "detector_key": self._detector_key,
            "segmenter_key": self._segmenter_key,
            "classes": [],
            "conf": float(settings.get("detection.default_conf", 0.25)),
            "iou": float(settings.get("detection.default_iou", 0.45)),
            "imgsz": int(settings.get("detection.default_imgsz", 640)),
            "weight_mode": str(settings.get("detection.weight_mode", "dual")),
            "threshold": float(settings.get("segmentation.default_threshold", 0.30)),
            "open_ks": int(settings.get("segmentation.open_ks", 3)),
            "close_ks": int(settings.get("segmentation.close_ks", 3)),
            "sample_points": int(settings.get("ranging.sample_points", 20)),
            "kernel": int(settings.get("ranging.kernel", 21)),
            "max_range_m": float(settings.get("ranging.max_range_m", 20.0)),
            "road_split": bool(settings.get("ranging.road_split", True)),
            "reproject": bool(settings.get("ranging.reproject", True)),
            "corridor_ref_depth_m": float(settings.get("ranging.corridor_ref_depth_m", 5.0)),
            "record_mode": "manual",       # manual | auto_plan
            "auto_record": False,
            "detector_model_id": None,
            "segmenter_model_id": None,
        }

        self.session: dict[str, Any] = {
            "active": False, "experiment_id": None, "project_id": None,
            "started_at": None, "frames": 0, "recorded": 0,
            "plan_index": 0, "paused": False, "paused_at": None,
            "illumination": "sunny", "scene": "campus_road",
        }
        self.last: dict[str, Any] = {}
        self._last_overlay: np.ndarray | None = None

    # ============================================================== 配置
    def configure(self, **kwargs: Any) -> dict:
        with self._lock:
            reload_needed = False
            for key, value in kwargs.items():
                if value is None or key not in self.config:
                    continue
                if key in ("detector_key", "segmenter_key", "weight_mode",
                           "detector_model_id", "segmenter_model_id"):
                    if self.config.get(key) != value:
                        reload_needed = True
                self.config[key] = value
            if reload_needed:
                self.reload_plugins()
            return dict(self.config)

    @staticmethod
    def _detector_class_names() -> list[str]:
        return settings.detection_class_names()

    def reload_plugins(self) -> None:
        """按配置重建插件实例（模型可插拔）。"""
        mode = self.config["mode"]
        errors: list[str] = []
        if mode in ("detection", "both"):
            try:
                key = self.config["detector_key"]
                plugin = registry.create("detection", key, **self._plugin_kwargs("detection", key))
                weights = self._weights_uri(self.config.get("detector_model_id"))
                plugin.load(weights, device="auto")
                self._detector = plugin
                self._detector_key = key
                log.info("检测插件已加载: %s (weights=%s)", key, weights)
            except Exception as exc:  # noqa: BLE001
                self._detector = None
                errors.append(f"检测插件加载失败: {exc}")
                log.warning("检测插件加载失败: %s", exc)
        if mode in ("segmentation", "both"):
            try:
                key = self.config["segmenter_key"]
                plugin = registry.create("segmentation", key, **self._plugin_kwargs("segmentation", key))
                weights = self._weights_uri(self.config.get("segmenter_model_id"))
                plugin.load(weights, device="auto")
                self._segmenter = plugin
                self._segmenter_key = key
                log.info("分割插件已加载: %s (weights=%s)", key, weights)
            except Exception as exc:  # noqa: BLE001
                self._segmenter = None
                errors.append(f"分割插件加载失败: {exc}")
                log.warning("分割插件加载失败: %s", exc)
        self.config["load_errors"] = errors

    def _plugin_kwargs(self, task: str, key: str) -> dict:
        if task == "detection" and key == "yolo":
            arch = self._model_arch()
            version, size = ("v8", "n")
            if arch and arch.startswith("yolo"):
                tail = arch[4:] if arch.startswith("yolov") else arch[4:]
                if arch.startswith("yolov"):
                    version, size = tail[: len(tail) - 1] or "8", tail[-1]
                else:
                    version, size = (tail[:-1] or "11"), tail[-1]
            return {"version": version, "size": size,
                    "weight_mode": self.config["weight_mode"],
                    "class_names": self._detector_class_names()}
        if task == "segmentation" and key in ("unet", "resunet", "att_resunet", "unetv2"):
            return {"arch": key, "n_classes": 1}
        return {}

    def _model_arch(self) -> str | None:
        model_id = self.config.get("detector_model_id")
        if not model_id:
            return None
        row = db.query_one("SELECT arch FROM model WHERE id=?", (model_id,))
        return row["arch"] if row else None

    @staticmethod
    def _weights_uri(model_id: str | None) -> str | None:
        if not model_id:
            return None
        row = db.query_one("SELECT weights_uri FROM model WHERE id=?", (model_id,))
        return row["weights_uri"] if row else None

    def set_calibration(self, plan: CalibrationPlan | None) -> None:
        self._calibration = plan

    # ============================================================== 会话
    def start_session(self, experiment_id: str, project_id: str | None = None) -> dict:
        with self._lock:
            self.session.update({
                "active": True, "experiment_id": experiment_id, "project_id": project_id,
                "started_at": db.now_iso(), "frames": 0, "recorded": 0,
                "plan_index": 0, "paused": False, "paused_at": None,
            })
            db.update("experiment", experiment_id, {"status": "running",
                                                    "started_at": db.now_iso()})
            log.info("实验会话开始: %s", experiment_id)
            return dict(self.session)

    def stop_session(self) -> dict:
        with self._lock:
            experiment_id = self.session.get("experiment_id")
            summary = dict(self.session)
            if experiment_id:
                stats = self._experiment_metrics(experiment_id)
                db.update("experiment", experiment_id, {
                    "status": "done", "finished_at": db.now_iso(),
                    "metrics": stats, "progress": 100.0,
                })
            self.session.update({"active": False, "paused": False})
            log.info("实验会话结束: %s", experiment_id)
            return summary

    @staticmethod
    def _experiment_metrics(experiment_id: str) -> dict:
        rows = db.query_all(
            "SELECT COUNT(*) AS n, AVG(abs_error_m) AS mae, "
            "AVG(rel_error) AS mre FROM ranging_result WHERE experiment_id=?",
            (experiment_id,))
        detected = db.query_scalar(
            "SELECT COUNT(*) FROM detection_result WHERE experiment_id=?", (experiment_id,), 0)
        missed = db.query_scalar(
            "SELECT COUNT(*) FROM detection_result WHERE experiment_id=? AND is_miss=1",
            (experiment_id,), 0)
        metrics: dict[str, Any] = {
            "ranging_records": int(rows[0]["n"] or 0) if rows else 0,
            "mae_m": round(float(rows[0]["mae"]), 4) if rows and rows[0]["mae"] is not None else None,
            "mre": round(float(rows[0]["mre"]), 4) if rows and rows[0]["mre"] is not None else None,
            "frames_detected": int(detected or 0),
            "frames_missed": int(missed or 0),
        }
        if detected:
            metrics["miss_rate"] = round(float(missed or 0) / float(detected), 4)
        return metrics

    def toggle_pause(self, paused: bool | None = None) -> dict:
        with self._lock:
            self.session["paused"] = (not self.session["paused"]) if paused is None else bool(paused)
            self.session["paused_at"] = db.now_iso() if self.session["paused"] else None
            if self.session.get("experiment_id"):
                db.update("experiment", self.session["experiment_id"],
                          {"status": "paused" if self.session["paused"] else "running"})
            return {"paused": self.session["paused"], "paused_at": self.session["paused_at"]}

    # ============================================================== 处理
    def process(self, bundle: FrameBundle, ground_truth: list[dict] | None = None) -> dict:
        """对一帧执行完整流水线，返回渲染结果与结构化数据。"""
        with self._lock:
            cfg = dict(self.config)
            detector, segmenter = self._detector, self._segmenter
            calibration = self._calibration

        mode = cfg["mode"]
        overlay = bundle.color.copy()
        detection_payload: dict[str, Any] = {"boxes": [], "detections": [], "infer_ms": 0.0}
        road_payload: dict[str, Any] = {"left": {}, "right": {}, "center": {}, "corridor_width_m": None}
        object_results: list[dict] = []
        target_results: list[dict] = []
        errors: list[str] = list(cfg.get("load_errors") or [])

        sampler = DepthSampler(kernel=int(cfg["kernel"]),
                               min_m=float(settings.get("ranging.min_range_m", 0.3)),
                               max_m=float(cfg["max_range_m"]))

        t_start = time.perf_counter()

        # ---------------- 检测 + 目标边界测距 ----------------
        if mode in ("detection", "both") and detector is not None:
            params: dict[str, Any] = {
                "conf": cfg["conf"], "iou": cfg["iou"], "imgsz": cfg["imgsz"],
                "classes": cfg["classes"] or None,
                "depth_m": bundle.depth_m, "intrinsics": bundle.intrinsics,
            }
            if cfg["detector_key"] == "sim_ground_truth":
                params["ground_truth"] = ground_truth or []
            try:
                pred = detector.predict(bundle.color, **params)
                strategy = BoxBoundaryStrategy(sampler)
                results = strategy.measure(bundle.depth_m, bundle.intrinsics, pred,
                                           sample_points=cfg["sample_points"],
                                           calibration=calibration)
                compact = summarize_per_detection(results)
                detection_payload = {
                    "boxes": pred.get("boxes") or [],
                    "detections": pred.detections(),
                    "infer_ms": float(pred.get("infer_ms") or 0.0),
                    "arch": pred.get("arch"), "weight_mode": pred.get("weight_mode"),
                    "reason": pred.get("reason"),
                }
                object_results = compact
                for item in compact:
                    if item["ranging"].get("min"):
                        target_results.append(("object", item["category"], item["ranging"]["min"]))
                    if item["ranging"].get("center"):
                        target_results.append(("object", item["category"], item["ranging"]["center"]))
                overlay = media.draw_detections(overlay, _attach_ranging(pred.detections(), compact))
                for item in compact:
                    hint = item["ranging"].get("min")
                    if hint:
                        overlay = media.draw_ranging_points(overlay, hint)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"检测/测距失败: {exc}")
                log.warning("检测/测距失败: %s", exc)

        # ---------------- 分割 + 道路边界测距 ----------------
        if mode in ("segmentation", "both") and segmenter is not None:
            try:
                seg_pred = segmenter.predict(
                    bundle.color, depth_m=bundle.depth_m, intr=bundle.intrinsics,
                    threshold=cfg["threshold"], open_ks=cfg["open_ks"],
                    close_ks=cfg["close_ks"], intrinsics=bundle.intrinsics,
                )
                strategy = MaskBoundaryStrategy(sampler)
                results = strategy.measure(
                    bundle.depth_m, bundle.intrinsics, seg_pred,
                    sample_points=cfg["sample_points"], calibration=calibration,
                    road_split=cfg["road_split"], reproject=cfg["reproject"],
                    corridor_ref_depth_m=cfg["corridor_ref_depth_m"],
                )
                road_payload = road_summary(results)
                road_payload["infer_ms"] = float(seg_pred.get("infer_ms") or 0.0)
                mask_ratio = float(np.asarray(seg_pred.get("mask")).mean()) if seg_pred.get("mask") is not None else 0.0
                road_payload["mask_area_ratio"] = round(mask_ratio, 4)
                road_payload["arch"] = seg_pred.get("arch")
                for item in results:
                    if item.metric_kind == "min" and item.target_kind.startswith("road"):
                        target_results.append((item.target_kind, None, item.to_dict()))
                overlay = media.draw_mask(overlay, seg_pred.get("mask"), alpha=0.30)
                for item in results:
                    if item.metric_kind in ("min", "max"):
                        overlay = media.draw_ranging_points(overlay, attach_visual_hints(item))
            except Exception as exc:  # noqa: BLE001
                errors.append(f"分割/测距失败: {exc}")
                log.warning("分割/测距失败: %s", exc)

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        # ---------------- 叠加信息面板 ----------------
        if cfg["mode"] in ("segmentation", "both") and road_payload.get("left"):
            panel = []
            for side, label in (("left", "道路左边界"), ("right", "道路右边界")):
                for kind, kl in (("min", "Min"), ("max", "Max")):
                    block = road_payload.get(side, {}).get(kind)
                    if block and block.get("depth_m") is not None:
                        panel.append(f"{label} {kl} {block['depth_m']:.2f} m")
            if road_payload.get("corridor_width_m"):
                panel.append(f"可通行走廊 {road_payload['corridor_width_m']:.2f} m "
                             f"@ {road_payload.get('left', {}).get('min', {}).get('depth_m', 0):.1f} m 附近")
            overlay = media.draw_panel(overlay, panel, "top-right", "道路边界测距")

        if cfg["mode"] in ("detection", "both") and detection_payload["detections"]:
            hud = [f"检出 {len(detection_payload['detections'])} 个目标",
                   f"推理 {detection_payload['infer_ms']:.1f} ms"]
            if calibration is not None:
                hud.append(f"已启用校正: {calibration.name}")
            overlay = media.draw_hud(overlay, hud, (12, 10))

        # 暂停/标准距离点提示
        paused = bool(self.session.get("paused"))
        plan_hint = self._plan_hint(object_results)
        if paused:
            overlay = media.draw_hud(overlay, ["⏸ 已暂停：可录入真值"], (12, 700),
                                     media.HudStyle(color=media.COLOR_WARN, size=18))

        self._last_overlay = overlay
        result = {
            "frame_number": bundle.frame_number,
            "timestamp": bundle.timestamp,
            "source": bundle.source,
            "mode": mode,
            "detection": detection_payload,
            "road": road_payload,
            "objects": object_results,
            "errors": errors,
            "paused": paused,
            "plan_hint": plan_hint,
            "perf": {
                "pipeline_ms": round(elapsed_ms, 2),
                "infer_ms": round(detection_payload.get("infer_ms", 0.0), 2),
                "seg_ms": round(road_payload.get("infer_ms", 0.0), 2),
            },
            "intrinsics": bundle.intrinsics.to_dict(),
            "depth_scale": bundle.depth_scale,
        }
        self.last = result
        self.session["frames"] = int(self.session.get("frames", 0)) + 1
        return result

    @staticmethod
    def _plan_hint(object_results: list[dict]) -> dict | None:
        """判断当前最近目标是否落在某个标准距离点附近，提示可以定格记录。"""
        plan = list(settings.get("ranging.distance_plan_m", []) or [])
        if not plan or not object_results:
            return None
        depths = [item["ranging"]["center"]["depth_m"] for item in object_results
                  if item["ranging"].get("center")]
        if not depths:
            return None
        nearest = min(depths)
        tol = float(settings.get("ranging.pause_tolerance_m", 0.05))
        for point in plan:
            if abs(nearest - float(point)) <= max(tol, 0.35):
                return {"plan_distance_m": float(point), "observed_m": round(nearest, 3),
                        "within_tolerance": abs(nearest - float(point)) <= tol}
        return None

    def overlay_frame(self) -> np.ndarray | None:
        return self._last_overlay

    # ============================================================== 落库
    def record_frame_result(self, bundle: FrameBundle, result: dict, save_media: bool = True) -> dict:
        """把一帧的检测/分割结果落库（返回记录 id）。"""
        experiment_id = self.session.get("experiment_id")
        if not experiment_id:
            raise ValidationError("当前没有进行中的实验会话，无法落库。")

        created: dict[str, Any] = {"detection_result_id": None, "segmentation_result_id": None,
                                   "frame_uri": None}
        if save_media and self._last_overlay is not None:
            out_dir = settings.storage_dir("results") / experiment_id / "frames"
            out_dir.mkdir(parents=True, exist_ok=True)
            name = f"frame_{bundle.frame_number:06d}.jpg"
            media.write_image(out_dir / name, self._last_overlay, quality=88)
            created["frame_uri"] = f"results/{experiment_id}/frames/{name}"

        if result.get("detection", {}).get("detections") is not None:
            dets = result["detection"]["detections"]
            created["detection_result_id"] = db.insert("detection_result", {
                "experiment_id": experiment_id,
                "frame_number": bundle.frame_number,
                "frame_uri": created["frame_uri"],
                "boxes": dets,
                "num_boxes": len(dets),
                "is_miss": 1 if not dets else 0,
                "weight_mode": result["detection"].get("weight_mode"),
                "infer_ms": result["detection"].get("infer_ms"),
            }, id_prefix="det")

        road = result.get("road") or {}
        if road.get("mask_area_ratio") is not None:
            created["segmentation_result_id"] = db.insert("segmentation_result", {
                "experiment_id": experiment_id,
                "frame_number": bundle.frame_number,
                "overlay_uri": created["frame_uri"],
                "threshold": self.config["threshold"],
                "open_ks": self.config["open_ks"],
                "close_ks": self.config["close_ks"],
                "mask_area_ratio": road.get("mask_area_ratio"),
                "is_empty": 1 if not road.get("mask_area_ratio") else 0,
                "is_suspicious": 1 if (road.get("mask_area_ratio") or 0) < 0.02 else 0,
                "infer_ms": road.get("infer_ms"),
            }, id_prefix="seg")
        return created

    def record_measurement(self, record: MeasurementRecord) -> dict:
        """写入一条测距记录（含真值与误差）。"""
        target_kind, metric_kind, result = record.target_kind, record.metric_kind, record.result
        actual_value, actual_source = record.actual_value, record.actual_source
        category, notes = record.category, record.notes
        illumination = record.illumination
        media_uri = record.media_uri
        detection_result_id = record.detection_result_id
        segmentation_result_id = record.segmentation_result_id
        experiment_id = self.session.get("experiment_id")
        if not experiment_id:
            raise ValidationError("当前没有进行中的实验会话，无法记录测距结果。")

        depth = result.get("depth_m")
        if depth is None:
            raise ValidationError("该测距结果没有有效深度值。")

        actual = None if actual_value in (None, "") else float(actual_value)
        error = abs_error = rel = None
        if actual is not None and actual > 0:
            error = float(depth) - actual
            abs_error = abs(error)
            rel = abs_error / actual

        exp = db.query_one("SELECT task_type FROM experiment WHERE id=?", (experiment_id,))
        task_type = (exp or {}).get("task_type") or self._infer_task_type(target_kind)
        model_id = (self.config.get("detector_model_id")
                    if target_kind == "object" else self.config.get("segmenter_model_id"))
        model_name = None
        if model_id:
            row = db.query_one("SELECT name FROM model WHERE id=?", (model_id,))
            model_name = row["name"] if row else None
        if not model_name:
            model_name = self._detector_key if target_kind == "object" else self._segmenter_key

        plan_distance = None
        plan = list(settings.get("ranging.distance_plan_m", []) or [])
        if plan:
            plan_distance = min(plan, key=lambda p: abs(float(p) - float(depth)))

        record_id = db.insert("ranging_result", {
            "experiment_id": experiment_id,
            "detection_result_id": detection_result_id,
            "segmentation_result_id": segmentation_result_id,
            "model_id": model_id,
            "model_name": model_name,
            "task_type": task_type,
            "strategy": self._strategy_of(target_kind),
            "target_kind": target_kind,
            "metric_kind": metric_kind,
            "category": category,
            "size_group": settings.class_group(category) if category else None,
            "depth_m": depth,
            "lateral_m": result.get("lateral_m"),
            "euclidean_m": result.get("euclidean_m"),
            "raw_depth_m": result.get("raw_depth_m"),
            "corrected_depth_m": result.get("corrected_depth_m"),
            "valid_points": result.get("valid_points"),
            "total_points": result.get("total_points"),
            "sample_points": result.get("sample_points") or [],
            "distance_actual_m": actual,
            "actual_source": actual_source if actual is not None else None,
            "error_m": error,
            "abs_error_m": abs_error,
            "rel_error": rel,
            "illumination": illumination if illumination is not None else self.session.get("illumination"),
            "distance_plan_m": plan_distance,
            "calibration_applied": (self._calibration.name if self._calibration else None),
            "frame_number": self.last.get("frame_number"),
            "record_time": db.now_iso(),
            "media_uri": media_uri,
            "notes": notes,
            "source": "platform",
        }, id_prefix="rr")
        self.session["recorded"] = int(self.session.get("recorded", 0)) + 1
        return db.decode_row(db.query_one("SELECT * FROM ranging_result WHERE id=?", (record_id,))) or {}

    @staticmethod
    def _strategy_of(target_kind: str) -> str:
        return "mask_boundary" if str(target_kind).startswith("road") else "box_boundary"

    @staticmethod
    def _infer_task_type(target_kind: str) -> str:
        return "segmentation" if str(target_kind).startswith("road") else "detection"

    def status(self) -> dict:
        return {
            "config": dict(self.config),
            "session": dict(self.session),
            "detector": self._detector.describe() if self._detector else None,
            "segmenter": self._segmenter.describe() if self._segmenter else None,
            "calibration": self._calibration.to_dict() if self._calibration else None,
            "last": {k: v for k, v in self.last.items() if k != "sample_points"},
        }


def _attach_ranging(detections: list[dict], compact: list[dict]) -> list[dict]:
    """把测距结果按 bbox 挂回检测目标，供叠加渲染显示距离。"""
    by_box: dict[tuple, dict] = {}
    for item in compact:
        bbox = item.get("bbox")
        if not bbox:
            continue
        by_box[tuple(int(round(v)) for v in bbox)] = item["ranging"]
    out = []
    for det in detections:
        entry = dict(det)
        key = tuple(int(round(v)) for v in det.get("bbox", []))
        ranging = by_box.get(key)
        if ranging is None:
            # 容忍 1 像素误差的匹配
            for cand_key, cand_val in by_box.items():
                if len(cand_key) == len(key) and all(abs(a - b) <= 1 for a, b in zip(cand_key, key)):
                    ranging = cand_val
                    break
        if ranging:
            center = ranging.get("center") or ranging.get("min") or {}
            entry["ranging"] = {"depth_m": center.get("depth_m"),
                                "lateral_m": center.get("lateral_m")}
        out.append(entry)
    return out


_service: PipelineService | None = None
_service_lock = threading.Lock()


def get_pipeline() -> PipelineService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = PipelineService()
                try:
                    _service.reload_plugins()  # noqa: SLF001 - 首次构建即加载默认插件
                except Exception as exc:  # noqa: BLE001
                    log.warning("初始插件加载失败: %s", exc)
    return _service


def current_frame_bundle():
    """取当前相机最新帧（含仿真真值）。"""
    camera = get_camera_service()
    if not camera.running:
        camera.start()
    bundle = camera.latest(timeout=0.5)
    truth = camera.ground_truth()
    return bundle, truth
