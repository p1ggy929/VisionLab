"""双目深度误差校正策略。

沿用项目 A `backend/analysis.py` 的三种拟合：
  linear     actual = a * measured + b
  ratio      actual = measured / (1 + k * measured)
  piecewise  近/中/远三段线性（**分段边界可配置**，修复既有实现写死 5/15 m 的问题）

平台新增：校正参数版本化 + 可回灌测距流程；校正后 RMSE 未改善时标记为不推荐。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.config import settings
from core.exceptions import RangingError

METHODS = ("linear", "ratio", "piecewise")


# --------------------------------------------------------------------- 应用
@dataclass
class CalibrationPlan:
    """一个可应用的校正方案。"""

    method: str
    params: Any
    name: str = ""
    plan_id: str | None = None
    rmse_before: float | None = None
    rmse_after: float | None = None
    sample_count: int = 0

    def apply(self, measured: float) -> float:
        if measured is None or not math.isfinite(float(measured)):
            return float("nan")
        x = float(measured)
        try:
            if self.method == "linear":
                a, b = self.params
                return a * x + b
            if self.method == "ratio":
                k = self.params[0] if isinstance(self.params, (list, tuple)) else float(self.params)
                return x / (1.0 + k * x)
            if self.method == "piecewise":
                return self._apply_piecewise(x)
        except Exception:  # noqa: BLE001
            return x
        return x

    def _apply_piecewise(self, x: float) -> float:
        params = self.params or {}
        bounds = params.get("bounds") or list(settings.get("analysis.piecewise_bounds_m", [5.0, 15.0]))
        low, high = float(bounds[0]), float(bounds[1])
        if x < low and "near" in params:
            key = "near"
        elif x < high and "mid" in params:
            key = "mid"
        elif "far" in params:
            key = "far"
        elif "mid" in params:
            key = "mid"
        elif "near" in params:
            key = "near"
        else:
            return x
        a, b = params[key]
        return a * x + b

    def apply_array(self, values: np.ndarray) -> np.ndarray:
        return np.asarray([self.apply(v) for v in np.asarray(values, dtype=float)], dtype=float)

    def to_dict(self) -> dict:
        return {
            "id": self.plan_id, "name": self.name, "method": self.method,
            "params": self.params, "sample_count": self.sample_count,
            "rmse_before": self.rmse_before, "rmse_after": self.rmse_after,
        }


# --------------------------------------------------------------------- 拟合
def _clean(measured, actual) -> tuple[np.ndarray, np.ndarray]:
    m = np.asarray(measured, dtype=float).ravel()
    a = np.asarray(actual, dtype=float).ravel()
    mask = np.isfinite(m) & np.isfinite(a) & (a > 0) & (m > 0)
    return m[mask], a[mask]


def fit_linear(measured, actual) -> list[float]:
    m, a = _clean(measured, actual)
    if m.size < 2:
        raise RangingError("样本不足，无法拟合线性校正（至少需要 2 个有效样本）。")
    coeffs = np.polyfit(m, a, deg=1)
    return [float(coeffs[0]), float(coeffs[1])]


def fit_ratio(measured, actual) -> list[float]:
    m, a = _clean(measured, actual)
    if m.size < 2:
        raise RangingError("样本不足，无法拟合比率校正（至少需要 2 个有效样本）。")

    def model(x, k):
        return x / (1.0 + k * x)

    try:
        from scipy.optimize import curve_fit

        params, _ = curve_fit(model, m, a, p0=[0.01], maxfev=8000)
        return [float(params[0])]
    except Exception:  # noqa: BLE001 - scipy 缺失或拟合失败时退化为网格搜索
        best_k, best_err = 0.0, float("inf")
        for k in np.linspace(-0.02, 0.20, 441):
            err = float(np.mean((model(m, k) - a) ** 2))
            if err < best_err:
                best_k, best_err = float(k), err
        return [best_k]


def fit_piecewise(measured, actual, bounds: list[float] | None = None) -> dict:
    m, a = _clean(measured, actual)
    if m.size < 3:
        raise RangingError("样本不足，无法拟合分段校正（至少需要 3 个有效样本）。")
    bounds = list(bounds or settings.get("analysis.piecewise_bounds_m", [5.0, 15.0]))
    low, high = float(bounds[0]), float(bounds[1])
    buckets = {"near": (0.0, low), "mid": (low, high), "far": (high, float("inf"))}
    params: dict[str, Any] = {"bounds": [low, high]}
    for key, (lo, hi) in buckets.items():
        mask = (a >= lo) & (a < hi)
        if np.count_nonzero(mask) < 2:
            # 样本不足的段用全局线性兜底，保证方案完整可用
            coeffs = np.polyfit(m, a, deg=1)
        else:
            coeffs = np.polyfit(m[mask], a[mask], deg=1)
        params[key] = [float(coeffs[0]), float(coeffs[1])]
    return params


FITTERS = {"linear": fit_linear, "ratio": fit_ratio, "piecewise": fit_piecewise}


def calibrate(measured, actual, method: str = "linear",
              bounds: list[float] | None = None, name: str = "") -> CalibrationPlan:
    """拟合并返回校正方案，同时给出校正前后 RMSE。"""
    method = str(method).lower()
    if method not in FITTERS:
        raise RangingError(f"未知校正方法: {method}，可选 {list(FITTERS)}")

    m, a = _clean(measured, actual)
    if m.size < int(settings.get("analysis.calibration.min_samples", 2)):
        raise RangingError(f"有效样本仅 {m.size} 条，不足以拟合校正模型。")

    if method == "piecewise":
        params = fit_piecewise(m, a, bounds)
    else:
        params = FITTERS[method](m, a)

    plan = CalibrationPlan(method=method, params=params, name=name or method, sample_count=int(m.size))
    corrected = plan.apply_array(m)
    before = float(np.sqrt(np.mean((a - m) ** 2)))
    after = float(np.sqrt(np.mean((a - corrected) ** 2)))
    plan.rmse_before = round(before, 4)
    plan.rmse_after = round(after, 4)
    return plan


def is_recommended(plan: CalibrationPlan, min_gain: float = 0.02) -> bool:
    """校正后 RMSE 未下降（或下降不足）则不建议启用——避免"为了好看而校正"。"""
    if plan.rmse_before is None or plan.rmse_after is None:
        return False
    return plan.rmse_after < plan.rmse_before * (1.0 - min_gain)


def residual_report(measured, actual, plan: CalibrationPlan) -> list[dict]:
    m, a = _clean(measured, actual)
    corrected = plan.apply_array(m)
    out = []
    for mv, av, cv in zip(m, a, corrected):
        out.append({
            "measured": round(float(mv), 4),
            "actual": round(float(av), 4),
            "corrected": round(float(cv), 4),
            "residual_before": round(float(mv - av), 4),
            "residual_after": round(float(cv - av), 4),
        })
    return out
