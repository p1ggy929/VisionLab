"""离线合成 RGB-D 后端（MockBackend）。

用途：
  1. 无 RealSense 设备时开发、教学与答辩演示的兜底（保证全流程可跑通）；
  2. 生成"已知真值"的合成场景，用于验证测距/误差分析/校正链路。

场景模型（相机高 H、俯仰固定，地平线在 v_h）：
  地面深度      Z(u,v) = H * fy / (v - v_h)          （v > v_h）
  横向物理坐标  X(u,v) = (u - ppx) * Z / fx
  可通行区域    |X| <= road_half_width
  目标          以世界坐标 (Xc, Zc) + 物理尺寸 (w,h) 投影为图像框

深度观测模型（模拟真实双目"误差随距离非线性放大"）：
  Z_obs = Z + bias_coef * Z^2 + N(0, noise)
"""

from __future__ import annotations

import math
import time
from queue import Empty, Queue
from typing import Optional

import cv2
import numpy as np

from ..config import settings
from ..geometry import Intrinsics, estimate_intrinsics_from_fov
from .base import CameraBackend, FrameBundle

SKY_TOP = (150, 120, 90)
SKY_BOTTOM = (220, 200, 175)
ROAD_NEAR = (95, 95, 98)
ROAD_FAR = (135, 135, 138)


class _SceneObject:
    """合成场景中的一个目标。"""

    def __init__(self, name: str, xc: float, zc: float, w: float, h: float,
                 color: tuple[int, int, int], kind: str = "box") -> None:
        self.name = name
        self.xc = xc
        self.zc = zc
        self.w = w
        self.h = h
        self.color = color
        self.kind = kind


def default_scene() -> list[_SceneObject]:
    """校园道路日间场景：11 类目标中的代表性样本，距离覆盖 1–20 m。"""
    return [
        _SceneObject("车辆", -1.35, 12.5, 1.9, 1.55, (60, 70, 200)),
        _SceneObject("行人", 1.05, 7.2, 0.55, 1.70, (40, 40, 210)),
        _SceneObject("交通标志", 2.55, 9.5, 0.60, 0.85, (30, 30, 240), kind="pole"),
        _SceneObject("减速带", 0.00, 5.6, 3.40, 0.10, (90, 90, 230), kind="strip"),
        _SceneObject("井盖", 1.85, 3.4, 0.70, 0.06, (110, 110, 120), kind="flat"),
        _SceneObject("路标", -2.35, 15.5, 0.45, 0.70, (200, 200, 210), kind="pole"),
        _SceneObject("废瓶子", 0.55, 2.1, 0.10, 0.24, (120, 200, 160), kind="small"),
        _SceneObject("落叶", -0.85, 1.6, 0.35, 0.06, (70, 150, 190), kind="flat"),
        _SceneObject("废纸", 1.15, 2.9, 0.28, 0.05, (215, 215, 210), kind="flat"),
        _SceneObject("大理石球", -1.75, 4.6, 0.42, 0.42, (170, 170, 175), kind="small"),
        _SceneObject("斑马线", 0.00, 8.2, 3.60, 0.04, (235, 235, 235), kind="zebra"),
    ]


class MockBackend(CameraBackend):
    """合成 RGB-D 相机后端。"""

    name = "mock"

    def __init__(
        self,
        width: int | None = None,
        height: int | None = None,
        fps: int | None = None,
        moving: bool | None = None,
        noise: float | None = None,
        drop_rate: float | None = None,
        bias_coef: float = 0.0038,
        camera_height_m: float = 1.25,
        road_half_width_m: float = 3.1,
        seed: int = 20260512,
    ) -> None:
        cfg = settings.section("camera")
        mock_cfg = cfg.get("mock", {}) or {}
        self.width = int(width or cfg.get("width", 1280))
        self.height = int(height or cfg.get("height", 720))
        self.fps = int(fps or cfg.get("fps", 30))
        self.moving = bool(mock_cfg.get("moving", True) if moving is None else moving)
        self.noise = float(mock_cfg.get("noise", 0.004) if noise is None else noise)
        self.drop_rate = float(mock_cfg.get("drop_rate", 0.03) if drop_rate is None else drop_rate)
        self.bias_coef = float(bias_coef)
        self.camera_height_m = float(camera_height_m)
        self.road_half_width_m = float(road_half_width_m)

        self.intrinsics = estimate_intrinsics_from_fov(self.width, self.height, hfov_deg=87.0)
        self.depth_scale = 0.001
        self._rng = np.random.default_rng(seed)
        self._queue: "Queue[FrameBundle]" = Queue(maxsize=1)
        self._running = False
        self._frames = 0
        self._objects = default_scene()
        self._t0 = time.time()
        self._horizon_ratio = 0.44
        self._last_meta: list[dict] = []

    # ------------------------------------------------------------ 接口
    def describe(self) -> dict:
        return {
            "backend": self.name,
            "model": "Synthetic RGB-D Scene (MockBackend)",
            "serial": "MOCK-0000",
            "firmware": "sim-1.0",
            "usb_type": "n/a",
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "depth_scale": self.depth_scale,
            "intrinsics": self.intrinsics.to_dict(),
            "note": "无 RealSense 设备时使用的合成场景，深度含模拟的随距离放大偏置",
            "bias_coef": self.bias_coef,
            "road_half_width_m": self.road_half_width_m,
            "camera_height_m": self.camera_height_m,
        }

    def start(self) -> None:
        self._running = True
        self._t0 = time.time()

    def stop(self) -> None:
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    def get_latest(self, timeout: float = 1.0) -> Optional[FrameBundle]:
        if not self._running:
            self.start()
        bundle = self._render()
        try:
            while True:
                self._queue.get_nowait()
        except Empty:
            pass
        self._queue.put(bundle)
        return bundle

    # ------------------------------------------------------------ 场景渲染
    @property
    def horizon_v(self) -> float:
        return self.height * self._horizon_ratio

    def _render(self) -> FrameBundle:
        self._frames += 1
        t = time.time() - self._t0
        w, h = self.width, self.height
        fx, fy, ppx, ppy = (self.intrinsics.fx, self.intrinsics.fy,
                            self.intrinsics.ppx, self.intrinsics.ppy)
        v_h = self.horizon_v

        # ---------- 深度基底：地面平面 ----------
        vv, uu = np.mgrid[0:h, 0:w].astype(np.float32)
        denom = np.maximum(vv - v_h, 1.0)
        ground_depth = (self.camera_height_m * fy) / denom
        ground_depth[vv <= v_h] = 0.0
        ground_depth = np.clip(ground_depth, 0.0, 60.0)

        lateral = (uu - ppx) * ground_depth / max(fx, 1e-6)
        road_mask = (np.abs(lateral) <= self.road_half_width_m) & (vv > v_h)

        # ---------- RGB 基底 ----------
        color = np.zeros((h, w, 3), dtype=np.uint8)
        sky_grad = np.linspace(0, 1, h, dtype=np.float32).reshape(-1, 1, 1)
        sky = (np.array(SKY_TOP, dtype=np.float32).reshape(1, 1, 3) * (1 - sky_grad)
               + np.array(SKY_BOTTOM, dtype=np.float32).reshape(1, 1, 3) * sky_grad)
        color[:, :] = sky.astype(np.uint8)

        road_grad = np.clip((vv - v_h) / max(h - v_h, 1.0), 0, 1).reshape(h, w, 1)
        road_color = (np.array(ROAD_FAR, dtype=np.float32).reshape(1, 1, 3) * (1 - road_grad)
                      + np.array(ROAD_NEAR, dtype=np.float32).reshape(1, 1, 3) * road_grad)
        color[road_mask] = road_color[road_mask].astype(np.uint8)

        # 路面纹理（细颗粒噪点，避免深度/分割过于"干净"）
        texture = self._rng.normal(0, 5.0, size=(h, w, 1)).astype(np.float32)
        color = np.clip(color.astype(np.float32) + texture * road_mask[..., None], 0, 255).astype(np.uint8)

        # 道路两侧路面边缘白线
        for sign in (-1, 1):
            edge_x = ppx + sign * self.road_half_width_m * fx / np.maximum(ground_depth, 1e-3)
            for v in range(int(v_h) + 2, h):
                z = (self.camera_height_m * fy) / max(v - v_h, 1e-3)
                if z > 28:
                    continue
                x = int(round(ppx + sign * self.road_half_width_m * fx / max(z, 1e-3)))
                if 0 <= x < w:
                    cv2.circle(color, (x, v), max(1, int(3 - v / h * 2)), (235, 235, 235), -1)

        # 树荫斑块（模拟复杂日间光照）
        shade = np.zeros((h, w), dtype=np.float32)
        for i in range(3):
            cx = int(w * (0.2 + 0.3 * i) + 40 * math.sin(t * 0.3 + i))
            cy = int(h * 0.62 + 20 * i)
            cv2.ellipse(shade, (cx, cy), (int(w * 0.13), int(h * 0.05)), 15 * i, 0, 360, 0.35, -1)
        color = np.clip(color.astype(np.float32) * (1 - shade[..., None]), 0, 255).astype(np.uint8)

        # ---------- 目标与深度叠加 ----------
        depth = ground_depth.copy()
        objects_meta: list[dict] = []
        for idx, obj in enumerate(self._objects):
            zc = obj.zc + (math.sin(t * 0.5 + idx) * 0.25 if self.moving else 0.0)
            if zc <= 1.0:
                continue
            xc = obj.xc + (math.sin(t * 0.35 + idx * 1.7) * 0.12 if self.moving else 0.0)

            v_bottom = v_h + (self.camera_height_m * fy) / zc
            v_top = v_h + ((self.camera_height_m - obj.h) * fy) / zc
            u_l = ppx + (xc - obj.w / 2.0) * fx / zc
            u_r = ppx + (xc + obj.w / 2.0) * fx / zc
            x1, x2 = int(max(0, min(u_l, u_r))), int(min(w - 1, max(u_l, u_r)))
            y1, y2 = int(max(0, min(v_top, v_bottom))), int(min(h - 1, max(v_top, v_bottom)))
            if x2 <= x1 or y2 <= y1:
                continue

            self._paint(color, obj, x1, y1, x2, y2, t)
            depth[y1:y2 + 1, x1:x2 + 1] = zc
            objects_meta.append({
                "index": idx, "name": obj.name, "category": obj.name,
                "world_x": round(xc, 3), "world_z": round(zc, 3),
                "bbox": [x1, y1, x2, y2], "size_m": [obj.w, obj.h],
                "size_group": "small" if max(obj.w, obj.h) < 0.75 else "large",
            })
        self._last_meta = objects_meta

        # ---------- 深度观测模型：偏置 + 噪声 + 缺失 ----------
        valid = depth > 0
        observed = depth + self.bias_coef * depth * depth
        observed += self._rng.normal(0, self.noise, size=depth.shape).astype(np.float32) * np.maximum(depth, 1.0)
        observed[~valid] = 0.0
        if self.drop_rate > 0:
            holes = self._rng.random(size=depth.shape) < self.drop_rate
            observed[holes & (self._rng.random(size=depth.shape) > 0.35)] = 0.0
        observed = np.clip(observed, 0.0, 65.535)

        raw = np.clip(observed / self.depth_scale, 0, 65535).astype(np.uint16)

        return FrameBundle(
            color=color, depth_m=observed.astype(np.float32), depth_raw=raw,
            depth_scale=self.depth_scale, timestamp=time.time(),
            intrinsics=self.intrinsics, frame_number=self._frames,
            serial="MOCK-0000", source="mock",
        )

    @staticmethod
    def _paint(color: np.ndarray, obj: _SceneObject, x1: int, y1: int, x2: int, y2: int, t: float) -> None:
        """把目标画到 RGB 图上（不同 kind 用不同画法，便于检测/分割演示）。"""
        base = obj.color
        if obj.kind == "flat":
            cv2.rectangle(color, (x1, y1 + (y2 - y1) // 2), (x2, y2), base, -1)
        elif obj.kind == "strip":
            cv2.rectangle(color, (x1, y1), (x2, y2), base, -1)
            for x in range(x1, x2, 26):
                cv2.line(color, (x, y1), (x + 12, y2), (250, 250, 250), 2)
        elif obj.kind == "zebra":
            for x in range(x1, x2, 34):
                cv2.rectangle(color, (x, y1 + 2), (min(x + 18, x2), y2), (250, 250, 250), -1)
        elif obj.kind == "pole":
            cx = (x1 + x2) // 2
            cv2.line(color, (cx, y2), (cx, y1 + int((y2 - y1) * 0.4)), (70, 70, 75), 2)
            cv2.rectangle(color, (x1, y1), (x2, y1 + int((y2 - y1) * 0.45)), base, -1)
        elif obj.kind == "small":
            cv2.ellipse(color, ((x1 + x2) // 2, (y1 + y2) // 2),
                        (max(2, (x2 - x1) // 2), max(2, (y2 - y1) // 2)), 0, 0, 360, base, -1)
        else:  # box
            cv2.rectangle(color, (x1, y1), (x2, y2), base, -1)
            cv2.rectangle(color, (x1 + 2, y1 + 2), (x2 - 2, y2 - 2), tuple(min(255, c + 35) for c in base), 2)

    # ------------------------------------------------------------ 真值
    def ground_truth(self) -> list[dict]:
        """返回当前帧目标的仿真真值（世界坐标 + 投影包围框 + 物理尺寸）。

        仿真器提供真值标注是常规做法（如 CARLA 的 ground truth），
        平台在界面上明确标识「仿真真值」，不冒充真实检测结果。
        """
        if self._last_meta:
            return [dict(item) for item in self._last_meta]
        out = []
        for obj in self._objects:
            out.append({
                "category": obj.name, "name": obj.name,
                "world_x": obj.xc, "world_z": obj.zc,
                "size_m": [obj.w, obj.h],
                "size_group": "small" if max(obj.w, obj.h) < 0.75 else "large",
                "bbox": None,
            })
        return out
