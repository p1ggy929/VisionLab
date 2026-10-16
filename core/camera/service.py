"""相机服务：全平台唯一的相机访问入口（RealSense 设备独占）。"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..config import settings
from ..exceptions import CameraError
from ..logging_setup import get_logger
from ..storage import ensure_dir, timestamp
from .base import CameraBackend, FrameBundle
from .mock_backend import MockBackend

log = get_logger("camera.service")


class VideoRecorder:
    """双路录制（原始流 + 叠加流）——合并项目 A data_saver 与项目 B Main.py 的重复实现。"""

    def __init__(self, out_dir: Path, width: int, height: int, fps: int = 25) -> None:
        self.out_dir = ensure_dir(Path(out_dir))
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.recording = False
        self._writers: dict[str, cv2.VideoWriter] = {}
        self._stamp = ""
        self.frames = 0

    def start(self) -> dict:
        if self.recording:
            return {"recording": True, "stamp": self._stamp}
        self._stamp = timestamp()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        for kind in ("original", "overlay"):
            path = self.out_dir / f"{kind}_{self._stamp}.mp4"
            writer = cv2.VideoWriter(str(path), fourcc, self.fps, (self.width, self.height))
            if not writer.isOpened():
                log.warning("视频写入器打开失败: %s", path)
                continue
            self._writers[kind] = writer
        self.recording = bool(self._writers)
        self.frames = 0
        log.info("开始录制: %s", self._stamp)
        return {"recording": self.recording, "stamp": self._stamp}

    def write(self, original: np.ndarray, overlay: np.ndarray | None) -> None:
        if not self.recording:
            return
        for kind, frame in (("original", original), ("overlay", overlay)):
            writer = self._writers.get(kind)
            if writer is None or frame is None:
                continue
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))
            writer.write(frame)
        self.frames += 1

    def stop(self) -> dict:
        if not self.recording:
            return {"recording": False, "frames": self.frames}
        for writer in self._writers.values():
            writer.release()
        self._writers.clear()
        self.recording = False
        files = sorted(str(p.name) for p in self.out_dir.glob(f"*_{self._stamp}.mp4"))
        log.info("停止录制: %s (%d 帧)", self._stamp, self.frames)
        return {"recording": False, "stamp": self._stamp, "frames": self.frames, "files": files}


class CameraService:
    """管理相机后端、最新帧、健康指标与录制。线程安全。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._backend: Optional[CameraBackend] = None
        self._bundle: Optional[FrameBundle] = None
        self._started_at = 0.0
        self._frames = 0
        self._fps_window: list[float] = []
        self._recorder: Optional[VideoRecorder] = None
        self._capture_dir: Optional[Path] = None

    # ------------------------------------------------------- 后端选择
    def _build_backend(self, prefer: str | None = None,
                       replay_source: dict | None = None) -> CameraBackend:
        """按优先级构建相机后端。

        auto      : 有 RealSense 设备则用真实相机，否则用合成场景
        realsense : 强制真实相机（无设备则报错）
        mock      : 强制合成场景（无设备也能完整演示）
        replay    : 离线回放（从落盘的真实 RGB-D 序列重放）
        """
        mode = (prefer or str(settings.get("camera.backend", "auto"))).lower()

        if mode == "replay":
            from .replay_backend import ReplayBackend

            source = replay_source or {}
            return ReplayBackend(
                source=source.get("id") or source.get("path") or source.get("session_id"),
                fps=int(source.get("fps") or settings.get("camera.fps", 15)),
                loop=bool(source.get("loop", True)),
                dataset_version_id=(source.get("id")
                                    if source.get("type") == "dataset" else None),
            )

        if mode in ("realsense", "auto"):
            from . import realsense_backend as rsb

            if rsb.available() and rsb.list_devices():
                return rsb.RealSenseBackend()
            if mode == "realsense":
                raise CameraError("未检测到可用的 RealSense 设备（pyrealsense2 已安装但无设备）。")
            log.warning("未检测到 RealSense 设备，切换到离线合成后端（MockBackend）")
        return MockBackend()

    def start(self, prefer: str | None = None, width: int | None = None,
              height: int | None = None, fps: int | None = None,
              replay_source: dict | None = None) -> dict:
        with self._lock:
            self.stop()
            backend = self._build_backend(prefer, replay_source)
            if width or height or fps:
                setattr(backend, "width", int(width or getattr(backend, "width", 1280)))
                setattr(backend, "height", int(height or getattr(backend, "height", 720)))
                setattr(backend, "fps", int(fps or getattr(backend, "fps", 30)))
            backend.start()
            self._backend = backend
            self._started_at = time.time()
            self._frames = 0
            self._fps_window.clear()
            info = backend.describe()
            log.info("相机已启动: backend=%s %sx%s", info.get("backend"), info.get("width"),
                     info.get("height"))
            return info

    def stop(self) -> None:
        with self._lock:
            if self._backend is not None:
                try:
                    self._backend.stop()
                except Exception as exc:  # noqa: BLE001
                    log.warning("停止相机异常: %s", exc)
            self._backend = None
            self._bundle = None

    @property
    def running(self) -> bool:
        backend = self._backend
        return bool(backend and backend.is_running)

    def latest(self, timeout: float = 0.5, refresh: bool = True) -> Optional[FrameBundle]:
        """取最新帧；MockBackend 每次调用即渲染一帧。"""
        with self._lock:
            backend = self._backend
            if backend is None:
                return None
            if refresh or self._bundle is None:
                bundle = backend.get_latest(timeout=timeout)
                if bundle is not None:
                    self._bundle = bundle
                    self._frames += 1
                    self._fps_window.append(time.time())
                    cutoff = time.time() - 2.0
                    self._fps_window = [t for t in self._fps_window if t >= cutoff]
        return self._bundle

    def describe(self) -> dict:
        with self._lock:
            backend = self._backend
            if backend is None:
                from . import realsense_backend as rsb

                return {
                    "running": False,
                    "backend": None,
                    "realsense_available": rsb.available(),
                    "devices": rsb.list_devices(),
                    "storage_free_gb": None,
                }
            info = backend.describe()
            info["running"] = backend.is_running
            info["frames"] = self._frames
            info["fps_actual"] = round(len(self._fps_window) / 2.0, 1)
            info["uptime_s"] = round(time.time() - self._started_at, 1) if self._started_at else 0
            if hasattr(backend, "health"):
                info["health"] = backend.health()  # type: ignore[attr-defined]
            return info

    def ground_truth(self) -> list[dict]:
        backend = self._backend
        if backend is not None and hasattr(backend, "ground_truth"):
            return backend.ground_truth()  # type: ignore[attr-defined]
        return []

    # ------------------------------------------------------- 录制
    def start_recording(self, session_dir: Path | None = None) -> dict:
        bundle = self._bundle
        if bundle is None:
            raise CameraError("当前没有可用画面，无法开始录制。")
        out = Path(session_dir) if session_dir else settings.storage_dir("captures") / f"rec_{timestamp()}"
        ensure_dir(out)
        h, w = bundle.color.shape[:2]
        self._recorder = VideoRecorder(out, w, h, fps=int(settings.get("camera.fps", 30)))
        return self._recorder.start()

    def record_frame(self, overlay: np.ndarray | None = None) -> None:
        if self._recorder is None or not self._recorder.recording:
            return
        bundle = self._bundle
        if bundle is None:
            return
        self._recorder.write(bundle.color, overlay)

    def stop_recording(self) -> dict:
        if self._recorder is None:
            return {"recording": False}
        result = self._recorder.stop()
        self._recorder = None
        return result

    @property
    def recording(self) -> bool:
        return bool(self._recorder and self._recorder.recording)

    # ------------------------------------------------------- 抓拍
    def snapshot(self, session_dir: Path) -> dict:
        bundle = self._bundle
        if bundle is None:
            raise CameraError("当前没有可用画面，无法抓拍。")
        from .. import media as media_mod

        ensure_dir(session_dir)
        stamp = timestamp()
        rgb_path = session_dir / f"rgb_{stamp}.jpg"
        depth_path = session_dir / f"depth_{stamp}.png"
        media_mod.write_image(rgb_path, bundle.color, quality=95)
        media_mod.write_depth16(depth_path, bundle.depth_m, bundle.depth_scale)
        return {
            "stamp": stamp,
            "rgb": str(rgb_path),
            "depth": str(depth_path),
            "frame_number": bundle.frame_number,
            "intrinsics": bundle.intrinsics.to_dict(),
            "depth_scale": bundle.depth_scale,
        }


_service: CameraService | None = None
_service_lock = threading.Lock()


def get_camera_service() -> CameraService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = CameraService()
    return _service
