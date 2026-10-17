"""深度采样器：目标边界测距与道路边界测距共用的采样与过滤内核。

沿用项目 A 的经验：局部中位数（核 21×21）比单像素更抗噪；
沿用项目 B 的有效点定义：>0 且 ≤ 20 m。
"""

from __future__ import annotations

import numpy as np

from core.config import settings
from core.geometry import Intrinsics, deproject


class DepthSampler:
    """在指定像素点处采样深度，并做有效性过滤。"""

    def __init__(self, kernel: int | None = None, mode: str | None = None,
                 min_m: float | None = None, max_m: float | None = None) -> None:
        kernel = int(kernel or settings.get("ranging.kernel", 21))
        self.kernel = kernel if kernel % 2 == 1 else kernel + 1
        self.mode = str(mode or settings.get("ranging.sampler_mode", "median"))
        self.min_m = float(min_m if min_m is not None else settings.get("ranging.min_range_m", 0.3))
        self.max_m = float(max_m if max_m is not None else settings.get("ranging.max_range_m", 20.0))

    # ------------------------------------------------------------------
    def patch_value(self, depth_m: np.ndarray, x: int, y: int) -> float | None:
        if self.mode == "point":
            value = float(depth_m[y, x])
            return value if self.min_m <= value <= self.max_m else None

        half = self.kernel // 2
        h, w = depth_m.shape[:2]
        y0, y1 = max(0, y - half), min(h, y + half + 1)
        x0, x1 = max(0, x - half), min(w, x + half + 1)
        patch = depth_m[y0:y1, x0:x1]
        if patch.size == 0:
            return None
        valid = patch[(patch >= self.min_m) & (patch <= self.max_m)]
        if valid.size > 0:
            return float(np.median(valid))
        value = float(depth_m[y, x])
        return value if self.min_m <= value <= self.max_m else None

    def sample_points(self, depth_m: np.ndarray, points) -> tuple[list[float], list[tuple[int, int]], list[float]]:
        """返回 (depth_values, valid_points, lateral_offsets)。"""
        h, w = depth_m.shape[:2]
        values: list[float] = []
        valid_points: list[tuple[int, int]] = []
        laterals: list[float] = []
        for point in points:
            if point is None or len(point) < 2:
                continue
            x, y = int(point[0]), int(point[1])
            if not (0 <= x < w and 0 <= y < h):
                continue
            value = self.patch_value(depth_m, x, y)
            if value is None:
                continue
            values.append(value)
            valid_points.append((x, y))
            laterals.append(x)  # 先放像素 x，稍后由策略换算为米
        return values, valid_points, laterals

    # ------------------------------------------------------------------
    def box_center_distance(self, depth_m: np.ndarray, box,
                            intr: Intrinsics) -> tuple[float | None, tuple[int, int] | None]:
        """检测框中心的局部中位数距离（项目 A 的 compute_box_distance）。"""
        if box is None or len(box) < 4:
            return None, None
        h, w = depth_m.shape[:2]
        cx = int((float(box[0]) + float(box[2])) / 2.0)
        cy = int((float(box[1]) + float(box[3])) / 2.0)
        if not (0 <= cx < w and 0 <= cy < h):
            return None, None
        value = self.patch_value(depth_m, cx, cy)
        return value, (cx, cy)

    @staticmethod
    def contour_points(contour: np.ndarray, num_points: int | None = None) -> np.ndarray:
        """轮廓等距采样（项目 B 的 np.linspace 做法）。"""
        num_points = int(num_points or settings.get("ranging.sample_points", 20))
        if contour is None or len(contour) == 0:
            return np.empty((0, 2), dtype=np.int32)
        pts = contour.reshape(-1, 2)
        if len(pts) <= num_points:
            return pts.astype(np.int32)
        idx = np.linspace(0, len(pts) - 1, num_points).astype(int)
        return pts[idx].astype(np.int32)

    @staticmethod
    def box_edge_points(box, num_points: int | None = None) -> np.ndarray:
        """沿检测框边界等距采样（用于目标边界距离）。"""
        if box is None or len(box) < 4:
            return np.empty((0, 2), dtype=np.int32)
        num_points = int(num_points or settings.get("ranging.sample_points", 20))
        side = max(2, num_points // 4)
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        xs = np.linspace(x1, x2, side)
        ys = np.linspace(y1, y2, side)
        pts = []
        pts.extend(zip(xs, np.full_like(xs, y1)))
        pts.extend(zip(np.full_like(ys, x2), ys))
        pts.extend(zip(xs[::-1], np.full_like(xs, y2)))
        pts.extend(zip(np.full_like(ys, x1), ys[::-1]))
        return np.asarray([(int(round(a)), int(round(b))) for a, b in pts], dtype=np.int32)

    @staticmethod
    def to_lateral_meters(pixel_x: np.ndarray, depth_values: list[float], intr: Intrinsics) -> list[float]:
        """像素 x + 深度 -> 横向物理偏移（米）。"""
        out = []
        for px, z in zip(pixel_x, depth_values):
            x, _, _ = deproject(float(px), 0.0, float(z), intr)
            out.append(x)
        return out
