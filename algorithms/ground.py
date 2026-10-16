"""地面平面模型：从深度图稳健估计"地面深度—图像行"关系。

用于两条无权重兜底算法：
  - 深度基线检测器（找显著近于地面的区域 = 障碍/目标）
  - 传统道路分割器（找贴合地面的连续区域 = 可通行区域）

模型：地面深度与图像行近似满足 Z(v) = 1 / (alpha * v + beta)
（等价于 Z = H*fy/(v - v_h)，取倒数后对 v 线性）。
"""

from __future__ import annotations

import numpy as np


class GroundPlane:
    def __init__(self, alpha: float, beta: float, valid: bool, horizon_v: float) -> None:
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.valid = bool(valid)
        self.horizon_v = float(horizon_v)

    def depth_at(self, rows: np.ndarray) -> np.ndarray:
        denom = self.alpha * rows + self.beta
        denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
        return 1.0 / denom

    def expected_map(self, height: int, width: int) -> np.ndarray:
        rows = np.arange(height, dtype=np.float32).reshape(-1, 1)
        return np.repeat(self.depth_at(rows), width, axis=1)

    def to_dict(self) -> dict:
        return {"alpha": self.alpha, "beta": self.beta, "valid": self.valid,
                "horizon_v": self.horizon_v}


def estimate_ground_plane(depth_m: np.ndarray, min_m: float = 0.3, max_m: float = 20.0,
                          row_percentile: float = 50.0, band: float = 0.6) -> GroundPlane:
    """从单帧深度图估计地面平面。

    做法：对每一行取中央 band 区域内有效深度的中位数，得到 (v, Z) 样本；
    再对 1/Z 与 v 做一次稳健线性拟合（去掉明显离群行）。
    """
    h, w = depth_m.shape[:2]
    x0 = int(w * (0.5 - band / 2))
    x1 = int(w * (0.5 + band / 2))
    rows: list[float] = []
    inv_z: list[float] = []
    for v in range(int(h * 0.5), h, 2):
        line = depth_m[v, x0:x1]
        valid = line[(line >= min_m) & (line <= max_m)]
        if valid.size < max(8, (x1 - x0) * 0.2):
            continue
        z = float(np.percentile(valid, row_percentile))
        if z <= 0:
            continue
        rows.append(float(v))
        inv_z.append(1.0 / z)

    if len(rows) < 8:
        return GroundPlane(0.0, 0.0, False, h * 0.5)

    v_arr = np.asarray(rows, dtype=np.float64)
    y_arr = np.asarray(inv_z, dtype=np.float64)

    alpha, beta = np.polyfit(v_arr, y_arr, 1)
    residual = np.abs(alpha * v_arr + beta - y_arr)
    keep = residual <= max(np.percentile(residual, 75) * 2.0, 1e-4)
    if np.count_nonzero(keep) >= 6:
        alpha, beta = np.polyfit(v_arr[keep], y_arr[keep], 1)

    horizon = float(-beta / alpha) if abs(alpha) > 1e-9 else h * 0.5
    horizon = float(np.clip(horizon, 0.15 * h, 0.95 * h))
    return GroundPlane(alpha, beta, True, horizon)


def closer_than_ground_mask(depth_m: np.ndarray, plane: GroundPlane, ratio: float = 0.85,
                            min_m: float = 0.3, max_m: float = 20.0) -> np.ndarray:
    """返回"显著近于地面"的二值掩膜（候选目标区域）。"""
    h, w = depth_m.shape[:2]
    expected = plane.expected_map(h, w) if plane.valid else np.full((h, w), np.inf, dtype=np.float32)
    valid = (depth_m >= min_m) & (depth_m <= max_m)
    rows = np.arange(h, dtype=np.float32).reshape(-1, 1)
    below_horizon = rows > plane.horizon_v if plane.valid else np.ones((h, 1), dtype=bool)
    mask = valid & (depth_m < expected * ratio) & below_horizon
    return mask.astype(np.uint8)


def on_ground_mask(depth_m: np.ndarray, plane: GroundPlane, tol_abs: float = 0.20,
                   tol_rel: float = 0.10, min_m: float = 0.3, max_m: float = 20.0) -> np.ndarray:
    """返回"贴合地面"的二值掩膜（候选可通行区域）。"""
    h, w = depth_m.shape[:2]
    if not plane.valid:
        return np.zeros((h, w), dtype=np.uint8)
    expected = plane.expected_map(h, w)
    valid = (depth_m >= min_m) & (depth_m <= max_m)
    tol = np.maximum(tol_abs, tol_rel * expected)
    rows = np.arange(h, dtype=np.float32).reshape(-1, 1)
    below_horizon = rows > plane.horizon_v
    mask = valid & (np.abs(depth_m - expected) <= tol) & below_horizon
    return mask.astype(np.uint8)
