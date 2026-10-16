"""误差校正策略。"""

from .strategies import (  # noqa: F401
    METHODS,
    CalibrationPlan,
    calibrate,
    fit_linear,
    fit_piecewise,
    fit_ratio,
    is_recommended,
    residual_report,
)

__all__ = ["CalibrationPlan", "calibrate", "fit_linear", "fit_ratio", "fit_piecewise",
           "is_recommended", "residual_report", "METHODS"]
