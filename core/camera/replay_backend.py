"""离线回放后端：从落盘的 RGB-D 序列按时间戳回放。

数据来源两类：
  1. 采集会话目录（`storage/captures/<id>/rgb/*.jpg` + `depth/*.png` + `session.json`）
  2. 数据集版本（`image.uri` + `depth_map.uri` 配对）

用途：
  · 采集现场没有相机时，用历史数据重跑整条流水线（推理 → 测距 → 误差 → 报告）；
  · 复现验证时从原始数据重放，确认结果可重复；
  · 教学场景下把"某一次真实采集"分发给多个学生做同样的实验。

与 MockBackend（合成场景）的区别：回放用的是**真实采集到的数据**，
深度值来自真实相机，因此测距结果具备真实误差特征。
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import settings
from ..exceptions import CameraError
from ..geometry import Intrinsics, estimate_intrinsics_from_fov
from ..logging_setup import get_logger
from .base import CameraBackend, FrameBundle

log = get_logger("camera.replay")

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp")


class ReplayBackend(CameraBackend):
    """按顺序回放落盘的 RGB-D 帧对。"""

    name = "replay"

    def __init__(self, source: str | None = None, fps: int = 15, loop: bool = True,
                 session_id: str | None = None, dataset_version_id: str | None = None) -> None:
        self.fps = max(1, int(fps))
        self.loop = bool(loop)
        self.source_label = ""
        self.session_id = session_id
        self.dataset_version_id = dataset_version_id
        self._pairs: list[dict] = []
        self._index = 0
        self._running = False
        self._lock = threading.RLock()
        self._next_at = 0.0
        self._frames = 0
        self._truth: list[dict] = []
        self.intrinsics = Intrinsics()
        self.depth_scale = 0.001
        self.width = 0
        self.height = 0

        if dataset_version_id:
            self._load_from_dataset(dataset_version_id)
        else:
            self._load_from_session(source or session_id or "")
        if not self._pairs:
            raise CameraError(
                "回放数据源为空：请先在「数据采集」中完成一次带深度的采集，"
                "或指定一个含深度图的数据集版本。")

    # ------------------------------------------------------------ 装载
    def _load_from_session(self, source: str) -> None:
        base = Path(source)
        if not base.is_absolute():
            base = settings.storage_dir("captures") / source
        if not base.exists():
            raise CameraError(f"回放目录不存在: {base}")

        meta_path = base / "session.json"
        meta: dict = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                meta = {}

        rgb_dir = base / "rgb"
        depth_dir = base / "depth"
        if not rgb_dir.exists() or not depth_dir.exists():
            raise CameraError(f"回放目录缺少 rgb/ 或 depth/ 子目录: {base}")

        depths = {p.stem.replace("depth_", ""): p for p in depth_dir.iterdir()
                  if p.suffix.lower() in IMAGE_EXT}
        pairs: list[dict] = []
        for rgb in sorted(rgb_dir.iterdir()):
            if rgb.suffix.lower() not in IMAGE_EXT:
                continue
            stamp = rgb.stem.replace("rgb_", "")
            depth = depths.get(stamp)
            if depth is None:
                continue
            pairs.append({"rgb": rgb, "depth": depth, "stamp": stamp})

        self._pairs = pairs
        self.source_label = meta.get("name") or base.name
        self.session_id = meta.get("id") or base.name
        self.depth_scale = float((meta.get("camera") or {}).get("depth_scale") or 0.001)
        self.intrinsics = self._intrinsics_from_session(meta, pairs)

        items = {item.get("stamp"): item for item in (meta.get("items") or [])}
        self._truth = []
        for pair in pairs:
            item = items.get(pair["stamp"]) or {}
            truth = item.get("truth") or {}
            self._truth.append({
                "stamp": pair["stamp"],
                "category": truth.get("category"),
                "actual_distance_m": truth.get("actual_distance_m"),
                "actual_min_m": truth.get("actual_min_m"),
                "actual_max_m": truth.get("actual_max_m"),
                "illumination": item.get("illumination"),
                "target_size_group": item.get("target_size_group"),
            })
        log.info("回放源已装载: %s（%d 帧）", self.source_label, len(self._pairs))

    @staticmethod
    def _both_files_exist(rgb: str | None, depth: str | None) -> bool:
        """判断 RGB 与深度图路径是否都可用（任一路径缺失即视为不可用）。"""
        if not rgb or not depth:
            return False
        return Path(rgb).exists() and Path(depth).exists()

    def _load_from_dataset(self, version_id: str) -> None:
        from core import db

        rows = db.query_all(
            "SELECT i.id, i.uri, i.illum, i.target_size_group, i.meta, "
            "d.uri AS depth_uri, d.depth_scale AS depth_scale "
            "FROM image i LEFT JOIN depth_map d ON d.image_id = i.id "
            "WHERE i.dataset_version_id=? AND d.uri IS NOT NULL ORDER BY i.uri", (version_id,))
        from core.storage import to_path

        pairs = []
        truths = []
        for row in rows:
            rgb = to_path(row["uri"])
            depth = to_path(row["depth_uri"])
            if not self._both_files_exist(rgb, depth):
                continue
            pairs.append({"rgb": Path(rgb), "depth": Path(depth), "stamp": Path(rgb).stem})
            meta = row.get("meta")
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except json.JSONDecodeError:
                    meta = {}
            truth = (meta or {}).get("truth") or {}
            truths.append({
                "stamp": Path(rgb).stem,
                "category": truth.get("category"),
                "actual_distance_m": truth.get("actual_distance_m"),
                "illumination": row.get("illum"),
                "target_size_group": row.get("target_size_group"),
            })
        self._pairs = pairs
        self._truth = truths
        version = db.query_one("SELECT dv.version, d.name FROM dataset_version dv "
                               "LEFT JOIN dataset d ON d.id=dv.dataset_id WHERE dv.id=?",
                               (version_id,))
        self.source_label = (f"{version['name']} {version['version']}" if version
                             else f"数据集版本 {version_id}")
        self.depth_scale = float((rows[0].get("depth_scale") if rows else None) or 0.001)
        self.intrinsics = self._intrinsics_from_session({}, pairs)

    def _intrinsics_from_session(self, meta: dict, pairs: list[dict]) -> Intrinsics:
        """优先用会话记录的内参；否则读首帧尺寸估算（D455 彩色视场角 87°）。"""
        intr = (meta.get("camera") or {}).get("intrinsics")
        if intr and intr.get("fx"):
            return Intrinsics.from_dict(intr)
        if pairs:
            import cv2

            image = cv2.imread(str(pairs[0]["rgb"]), cv2.IMREAD_COLOR)
            if image is not None:
                self.height, self.width = image.shape[:2]
                return estimate_intrinsics_from_fov(self.width, self.height, hfov_deg=87.0)
        return Intrinsics()

    # ------------------------------------------------------------ 接口
    def describe(self) -> dict:
        return {
            "backend": self.name,
            "model": f"离线回放 · {self.source_label}",
            "serial": self.session_id or "-",
            "firmware": "replay",
            "usb_type": "n/a",
            "width": self.width or self.intrinsics.width,
            "height": self.height or self.intrinsics.height,
            "fps": self.fps,
            "depth_scale": self.depth_scale,
            "intrinsics": self.intrinsics.to_dict(),
            "frames_total": len(self._pairs),
            "loop": self.loop,
            "note": "从落盘的真实 RGB-D 序列回放；用于无相机环境下的复现、教学与答辩演示",
        }

    def start(self) -> None:
        self._running = True
        self._index = 0
        self._next_at = 0.0

    def stop(self) -> None:
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    def get_latest(self, timeout: float = 1.0) -> Optional[FrameBundle]:
        """按 fps 节流返回下一帧；到末尾后按 loop 决定循环或停住。"""
        from .. import media

        if not self._running:
            self.start()
        with self._lock:
            if self._index >= len(self._pairs):
                if not self.loop:
                    return None
                self._index = 0
            pair = self._pairs[self._index]

            now = time.time()
            if self._next_at > now:
                time.sleep(min(self._next_at - now, max(timeout, 0.0)))
            self._next_at = max(self._next_at, now) + 1.0 / self.fps
            self._index += 1
            self._frames += 1
            frame_number = self._index

        color = media.read_image(pair["rgb"])
        depth_raw = media.read_depth16(pair["depth"], self.depth_scale)
        if color is None or depth_raw is None:
            return None
        if depth_raw.shape[:2] != color.shape[:2]:
            import cv2

            depth_raw = cv2.resize(depth_raw, (color.shape[1], color.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
        self.height, self.width = color.shape[:2]
        raw = np.clip(depth_raw / self.depth_scale, 0, 65535).astype(np.uint16)

        return FrameBundle(
            color=color, depth_m=depth_raw.astype(np.float32), depth_raw=raw,
            depth_scale=self.depth_scale, timestamp=time.time(),
            intrinsics=self.intrinsics, frame_number=frame_number,
            serial=self.session_id or "replay", source="replay",
        )

    # ------------------------------------------------------------ 真值
    def ground_truth(self) -> list[dict]:
        """返回**当前帧**记录的真值（若采集时录入过）。"""
        index = max(0, min(self._index - 1, len(self._truth) - 1))
        if not self._truth or index < 0:
            return []
        item = self._truth[index]
        if not item:
            return []
        out = []
        for key, label in (("actual_distance_m", None), ("actual_min_m", "road_left"),
                           ("actual_max_m", "road_right")):
            value = item.get(key)
            if value in (None, ""):
                continue
            out.append({
                "category": item.get("category") or (label or "目标"),
                "world_z": float(value), "source": "recorded_truth",
                "illumination": item.get("illumination"),
                "size_group": item.get("target_size_group"),
                "stamp": item.get("stamp"),
            })
        return out

    def progress(self) -> dict:
        return {"frame_index": self._index, "frames_total": len(self._pairs),
                "frames_served": self._frames, "looping": self.loop,
                "source": self.source_label}


def list_replay_sources() -> list[dict]:
    """列出可作为回放源的数据：采集会话 + 含深度图的数据集版本。"""
    from core import db
    from core.storage import to_path

    out: list[dict] = []
    capture_root = settings.storage_dir("captures")
    for path in sorted(capture_root.glob("*/session.json"),
                       key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        base = path.parent
        rgb_count = sum(1 for p in (base / "rgb").glob("*") if p.suffix.lower() in IMAGE_EXT)
        depth_count = sum(1 for p in (base / "depth").glob("*") if p.suffix.lower() in IMAGE_EXT)
        if min(rgb_count, depth_count) == 0:
            continue
        out.append({
            "type": "capture", "id": meta.get("id") or base.name,
            "label": meta.get("name") or base.name,
            "frames": min(rgb_count, depth_count),
            "illumination": meta.get("illumination"),
            "started_at": meta.get("started_at"),
            "with_truth": sum(1 for item in (meta.get("items") or []) if item.get("truth")),
        })

    rows = db.query_all(
        "SELECT i.dataset_version_id AS vid, COUNT(*) AS n, "
        "d.name AS dataset_name, dv.version AS version "
        "FROM image i JOIN depth_map dm ON dm.image_id = i.id "
        "LEFT JOIN dataset_version dv ON dv.id = i.dataset_version_id "
        "LEFT JOIN dataset d ON d.id = dv.dataset_id "
        "GROUP BY i.dataset_version_id ORDER BY n DESC")
    for row in rows:
        out.append({
            "type": "dataset", "id": row["vid"],
            "label": f"{row['dataset_name'] or '数据集'} {row['version'] or ''}".strip(),
            "frames": row["n"], "illumination": None, "started_at": None, "with_truth": 0,
        })
    return out


def capture_sample(source_id: str, limit: int = 40, with_truth: bool = True) -> dict:
    """为演示准备一个回放样本：用当前相机（合成或真实）抓取若干帧落盘。

    当相机后端能提供仿真真值（MockBackend）时，一并把最近目标的真值距离写入该帧，
    使回放数据带真值——这样回放时的测距误差分析与校正链路同样可用。

    生成的数据是**真实落盘的 RGB-D 序列**，回放路径与真实采集完全一致。
    """
    from core.camera.service import get_camera_service
    from services import capture_service

    session = capture_service.start_capture_session(
        name=f"回放样本 {time.strftime('%Y-%m-%d %H:%M')}", scene="campus_road",
        illumination="sunny", operator="system")
    camera = get_camera_service()
    saved, with_truth_count = 0, 0
    for _ in range(int(limit)):
        bundle = camera.latest(timeout=0.4)
        if bundle is None:
            time.sleep(0.05)
            continue
        try:
            item = capture_service.snapshot(session["id"])
        except Exception as exc:  # noqa: BLE001
            log.warning("抓帧失败: %s", exc)
            time.sleep(0.05)
            continue
        saved += 1

        if with_truth:
            truth = camera.ground_truth()
            if truth:
                nearest = min(truth, key=lambda t: float(t.get("world_z") or 1e9))
                try:
                    capture_service.set_truth(session["id"], capture_service.TruthEntry(
                        stamp=item["stamp"],
                        actual_distance_m=float(nearest["world_z"]),
                        actual_source="manual",
                        category=nearest.get("category"),
                        notes=f"由 {bundle.source} 后端在抓帧时同步记录的仿真真值",
                        illumination="sunny"))
                    with_truth_count += 1
                except Exception as exc:  # noqa: BLE001
                    log.warning("写入真值失败: %s", exc)
        time.sleep(0.03)

    capture_service.finish_session(session["id"])
    log.info("已生成回放样本 %s：%d 帧（含真值 %d 帧）", session["id"], saved, with_truth_count)
    return {"session_id": session["id"], "frames": saved, "with_truth": with_truth_count,
            "source_backend": camera.describe().get("backend")}
