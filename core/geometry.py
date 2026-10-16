"""相机几何：针孔模型、反投影、像素与物理量换算。

平台相对既有实现新增的能力：把"沿光轴深度"扩展为
（沿光轴深度 depth_m，横向偏移 lateral_m，欧氏距离 euclidean_m）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict


@dataclass
class Intrinsics:
    """相机内参。"""

    fx: float = 640.0
    fy: float = 640.0
    ppx: float = 640.0
    ppy: float = 360.0
    width: int = 1280
    height: int = 720
    model: str = "brown_conrady"
    coeffs: tuple = (0.0, 0.0, 0.0, 0.0, 0.0)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["coeffs"] = list(self.coeffs)
        return data

    @classmethod
    def from_dict(cls, data: dict | None) -> "Intrinsics":
        data = data or {}
        return cls(
            fx=float(data.get("fx", 640.0)),
            fy=float(data.get("fy", 640.0)),
            ppx=float(data.get("ppx", 640.0)),
            ppy=float(data.get("ppy", 360.0)),
            width=int(data.get("width", 1280)),
            height=int(data.get("height", 720)),
            model=str(data.get("model", "brown_conrady")),
            coeffs=tuple(data.get("coeffs", (0.0,) * 5)),
        )


def deproject(u: float, v: float, depth_m: float, intr: Intrinsics) -> tuple[float, float, float]:
    """像素坐标 + 沿光轴深度 -> 相机坐标系下的 (X, Y, Z)，单位米。

    X = (u - ppx) * Z / fx        （横向，右为正）
    Y = (v - ppy) * Z / fy        （纵向，下为正）
    Z = depth_m                   （沿光轴）
    """
    if depth_m is None or depth_m <= 0:
        return (0.0, 0.0, 0.0)
    x = (float(u) - intr.ppx) * depth_m / max(intr.fx, 1e-6)
    y = (float(v) - intr.ppy) * depth_m / max(intr.fy, 1e-6)
    return (x, y, depth_m)


def euclidean(x: float, y: float, z: float) -> float:
    return math.sqrt(x * x + y * y + z * z)


def pixel_to_meter_x(pixels: float, depth_m: float, intr: Intrinsics) -> float:
    """像素横向差 -> 米。"""
    return abs(pixels) * depth_m / max(intr.fx, 1e-6)


def estimate_intrinsics_from_fov(width: int, height: int, hfov_deg: float = 87.0) -> Intrinsics:
    """在无相机时按 D455 彩色视场角估算内参，保证反投影量级合理。"""
    fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return Intrinsics(fx=fx, fy=fx, ppx=width / 2.0, ppy=height / 2.0, width=width, height=height)


def depth_stats(depth_m, valid_min: float = 0.3, valid_max: float = 20.0) -> dict:
    """深度图统计（用 numpy，调用方保证传入 ndarray）。"""
    import numpy as np

    arr = np.asarray(depth_m, dtype="float32")
    valid = arr[(arr >= valid_min) & (arr <= valid_max)]
    if valid.size == 0:
        return {"valid_ratio": 0.0, "min_m": 0.0, "max_m": 0.0, "mean_m": 0.0}
    return {
        "valid_ratio": float(valid.size / max(arr.size, 1)),
        "min_m": float(valid.min()),
        "max_m": float(valid.max()),
        "mean_m": float(valid.mean()),
    }
