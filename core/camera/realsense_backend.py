"""Intel RealSense D455 后端（沿用项目 A backend/camera.py 的工程经验）。

关键点：
  - 唯一允许 import pyrealsense2 的模块；
  - 采集线程 + 长度为 1 的队列（取最新帧，不堆积）；
  - wait_for_frames 带 timeout，拔线后不会死等；
  - 默认关闭官方 spatial/temporal/hole-filling 滤波，保留原始深度细节；
  - 读取 depth_scale 与彩色流内参写入 FrameBundle，供测距与标定使用。
"""

from __future__ import annotations

import threading
import time
from queue import Empty, Queue
from typing import Optional

import numpy as np

from ..config import settings
from ..exceptions import CameraError
from ..geometry import Intrinsics
from ..logging_setup import get_logger
from .base import CameraBackend, FrameBundle

log = get_logger("camera.realsense")

try:  # pragma: no cover - 依赖硬件与驱动
    import pyrealsense2 as rs
except Exception:  # noqa: BLE001
    rs = None


def available() -> bool:
    return rs is not None


def list_devices() -> list[dict]:
    if rs is None:
        return []
    devices = []
    try:
        ctx = rs.context()
        for dev in ctx.query_devices():
            info = dev.get_info
            devices.append({
                "name": info(rs.camera_info.name),
                "serial": info(rs.camera_info.serial_number),
                "firmware": info(rs.camera_info.firmware_version),
                "usb": info(rs.camera_info.usb_type_descriptor),
                "product_line": info(rs.camera_info.product_line),
            })
    except Exception as exc:  # noqa: BLE001
        log.warning("枚举 RealSense 设备失败: %s", exc)
    return devices


class RealSenseBackend(CameraBackend):
    name = "realsense"

    def __init__(self, width: int | None = None, height: int | None = None,
                 fps: int | None = None, filters: list[str] | None = None) -> None:
        if rs is None:
            raise CameraError("未检测到 pyrealsense2，无法使用 RealSense 相机。")
        cfg = settings.section("camera")
        self.width = int(width or cfg.get("width", 1280))
        self.height = int(height or cfg.get("height", 720))
        self.fps = int(fps or cfg.get("fps", 30))
        self.filter_names = list(filters if filters is not None else (cfg.get("filters") or []))

        self._pipeline = None
        self._profile = None
        self._align = None
        self._filters: list = []
        self._queue: "Queue[FrameBundle]" = Queue(maxsize=1)
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._frames = 0
        self.depth_scale = 0.001
        self.intrinsics = Intrinsics(width=self.width, height=self.height,
                                     fx=self.width / 2, fy=self.width / 2,
                                     ppx=self.width / 2, ppy=self.height / 2)
        self.serial = ""
        self.firmware = ""
        self.usb = ""
        self.model = "Intel RealSense D455"
        self._drops = 0
        self._last_frame_ts = 0.0

    # ------------------------------------------------------------ 描述
    def describe(self) -> dict:
        return {
            "backend": self.name,
            "model": self.model,
            "serial": self.serial,
            "firmware": self.firmware,
            "usb_type": self.usb,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "depth_scale": self.depth_scale,
            "filters": self.filter_names,
            "intrinsics": self.intrinsics.to_dict(),
            "available_devices": list_devices(),
        }

    # ------------------------------------------------------------ 生命周期
    def _build_filters(self) -> None:
        mapping = {
            "decimation": lambda: rs.decimation_filter(),
            "spatial": lambda: rs.spatial_filter(),
            "temporal": lambda: rs.temporal_filter(),
            "hole_filling": lambda: rs.hole_filling_filter(),
        }
        filters = []
        for name in self.filter_names:
            factory = mapping.get(name)
            if factory is not None:
                filters.append(factory())
        self._filters = filters

    def start(self) -> None:
        if self._running.is_set():
            return
        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        try:
            self._profile = self._pipeline.start(config)
        except Exception as exc:  # noqa: BLE001
            raise CameraError(f"启动 RealSense 失败: {exc}") from exc

        device = self._profile.get_device()
        try:
            self.model = device.get_info(rs.camera_info.name)
            self.serial = device.get_info(rs.camera_info.serial_number)
            self.firmware = device.get_info(rs.camera_info.firmware_version)
            self.usb = device.get_info(rs.camera_info.usb_type_descriptor)
        except Exception:  # noqa: BLE001
            pass

        depth_sensor = device.first_depth_sensor()
        self.depth_scale = float(depth_sensor.get_depth_scale())

        try:
            color_profile = self._profile.get_stream(rs.stream.color).as_video_stream_profile()
            intr = color_profile.get_intrinsics()
            self.intrinsics = Intrinsics(
                fx=float(intr.fx), fy=float(intr.fy), ppx=float(intr.ppx), ppy=float(intr.ppy),
                width=int(intr.width), height=int(intr.height),
                model=str(getattr(intr, "model", "brown_conrady")),
                coeffs=tuple(float(c) for c in getattr(intr, "coeffs", (0,) * 5)),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("读取内参失败，使用估算值: %s", exc)

        self._align = rs.align(rs.stream.color)
        self._build_filters()

        self._running.set()
        self._thread = threading.Thread(target=self._loop, name="RealSenseCapture", daemon=True)
        self._thread.start()
        log.info("RealSense 已启动: %s SN=%s depth_scale=%.6f", self.model, self.serial, self.depth_scale)

    def stop(self) -> None:
        if not self._running.is_set():
            return
        self._running.clear()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        try:
            if self._pipeline is not None:
                self._pipeline.stop()
        except Exception:  # noqa: BLE001
            pass
        log.info("RealSense 已停止")

    @property
    def is_running(self) -> bool:
        return self._running.is_set()

    # ------------------------------------------------------------ 采集循环
    def _loop(self) -> None:  # pragma: no cover - 依赖硬件
        while self._running.is_set():
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=1000)
            except RuntimeError:
                log.warning("等待帧超时，重试中…")
                continue
            except Exception as exc:  # noqa: BLE001
                log.error("采集异常: %s", exc)
                time.sleep(0.2)
                continue

            aligned = self._align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            for filt in self._filters:
                depth_frame = filt.process(depth_frame)

            raw = np.asanyarray(depth_frame.get_data())
            color = np.asanyarray(color_frame.get_data())
            self._frames += 1
            self._last_frame_ts = time.time()
            bundle = FrameBundle(
                color=color, depth_m=raw.astype(np.float32) * self.depth_scale,
                depth_raw=raw, depth_scale=self.depth_scale, timestamp=self._last_frame_ts,
                intrinsics=self.intrinsics, frame_number=self._frames,
                serial=self.serial, source="realsense",
            )
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                    self._drops += 1
                except Empty:
                    pass
            self._queue.put(bundle)

    def get_latest(self, timeout: float = 1.0) -> Optional[FrameBundle]:
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            return None

    def health(self) -> dict:
        return {
            "running": self.is_running,
            "frames": self._frames,
            "drops": self._drops,
            "last_frame_age_s": round(time.time() - self._last_frame_ts, 2) if self._last_frame_ts else None,
        }
