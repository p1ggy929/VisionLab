"""平台统一异常。第三方异常一律转为本模块异常，避免泄漏到 API 层。"""

from __future__ import annotations


class VisionLabError(Exception):
    """平台基础异常。"""

    code = 50000
    http_status = 500

    def __init__(self, message: str, detail: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail or {}

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "detail": self.detail}


class ConfigError(VisionLabError):
    code = 50010


class CameraError(VisionLabError):
    code = 50020
    http_status = 503


class DatasetError(VisionLabError):
    code = 50030
    http_status = 400


class ModelError(VisionLabError):
    code = 50040
    http_status = 400


class RangingError(VisionLabError):
    code = 50050
    http_status = 400


class NotFoundError(VisionLabError):
    code = 40400
    http_status = 404


class ValidationError(VisionLabError):
    code = 40000
    http_status = 400


class ConflictError(VisionLabError):
    code = 40900
    http_status = 409


class AuthError(VisionLabError):
    code = 40100
    http_status = 401


class PermissionError_(VisionLabError):
    code = 40300
    http_status = 403


class TaskError(VisionLabError):
    code = 50060
