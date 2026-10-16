"""API 公共设施：统一响应、认证装饰器、参数解析、错误处理。"""

from __future__ import annotations

import functools
import traceback
from typing import Any, Callable

from flask import g, jsonify, request

from core.exceptions import AuthError, PermissionError_, VisionLabError
from core.logging_setup import get_logger
from services import auth_service

log = get_logger("api")


# ------------------------------------------------------------------ 响应
def ok(data: Any = None, message: str = "ok") -> Any:
    return jsonify({"code": 0, "message": message, "data": data})


def fail(message: str, code: int = 40000, http_status: int = 400, detail: dict | None = None) -> Any:
    return jsonify({"code": code, "message": message, "detail": detail or {}}), http_status


def paginated(result: dict, message: str = "ok") -> Any:
    return jsonify({"code": 0, "message": message, "data": result.get("items", []),
                    "total": result.get("total", 0), "page": result.get("page", 1),
                    "page_size": result.get("page_size", 20)})


# ------------------------------------------------------------------ 认证
def current_user() -> dict | None:
    return getattr(g, "user", None)


def _extract_token() -> str | None:
    header = request.headers.get("Authorization", "")
    if header.startswith("Bearer "):
        return header[7:].strip()
    return (request.args.get("token")
            or request.cookies.get("vl_token")
            or (request.get_json(silent=True) or {}).get("token"))


def load_user() -> dict | None:
    if getattr(g, "user", None) is not None:
        return g.user
    g.user = None
    token = _extract_token()
    if not token:
        return None
    try:
        payload = auth_service.parse_token(token)
    except AuthError:
        return None
    user = auth_service.get_user(payload["id"])
    if user and int(user.get("is_active", 1)):
        g.user = {"id": user["id"], "username": user["username"],
                  "role": user["role"], "display_name": user.get("display_name")}
    return g.user


def login_required(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any):
        user = load_user()
        if user is None:
            return fail("请先登录。", 40100, 401)
        return func(*args, **kwargs)

    return wrapper


def require_perm(action: str) -> Callable:
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any):
            user = load_user()
            if user is None:
                return fail("请先登录。", 40100, 401)
            if not auth_service.check_permission(user, action):
                auth_service.audit(user["id"], user["username"], action,
                                   "permission", None, "denied", request.remote_addr or "",
                                   {"path": request.path})
                return fail(f"当前角色没有权限执行 [{action}]。", 40300, 403)
            return func(*args, **kwargs)

        return wrapper

    return decorator


# ------------------------------------------------------------------ 参数
def payload() -> dict:
    data = request.get_json(silent=True)
    if isinstance(data, dict):
        return data
    if request.form:
        return {k: v for k, v in request.form.items()}
    return {}


def arg(name: str, default: Any = None, cast: Callable | None = None) -> Any:
    value = request.args.get(name, default)
    if value is None or value == "":
        return default
    if cast is None:
        return value
    try:
        return cast(value)
    except (TypeError, ValueError):
        return default


def arg_int(name: str, default: int) -> int:
    return arg(name, default, int) or default


def arg_float(name: str, default: float | None = None) -> float | None:
    value = arg(name, default)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def arg_bool(name: str, default: bool = False) -> bool:
    value = request.args.get(name)
    if value is None:
        return default
    return str(value).lower() in ("1", "true", "yes", "on")


def page_args() -> tuple[int, int]:
    return arg_int("page", 1), min(int(arg("page_size", 20) or 20), 200)


# ------------------------------------------------------------------ 错误
def register_error_handlers(app) -> None:
    @app.errorhandler(VisionLabError)
    def handle_visionlab(exc: VisionLabError):
        log.warning("业务异常 %s: %s", type(exc).__name__, exc.message)
        return fail(exc.message, exc.code, exc.http_status, exc.detail)

    @app.errorhandler(AuthError)
    def handle_auth(exc: AuthError):
        return fail(exc.message, exc.code, exc.http_status)

    @app.errorhandler(PermissionError_)
    def handle_perm(exc: PermissionError_):
        return fail(exc.message, exc.code, exc.http_status)

    @app.errorhandler(404)
    def handle_404(_):
        if request.path.startswith("/api/"):
            return fail(f"接口不存在: {request.path}", 40400, 404)
        from flask import render_template

        return render_template("error.html", code=404, message="页面不存在"), 404

    @app.errorhandler(500)
    def handle_500(exc):
        log.error("服务端异常: %s\n%s", exc, traceback.format_exc())
        if request.path.startswith("/api/"):
            return fail("服务端内部错误，详情见系统日志。", 50000, 500)
        from flask import render_template

        return render_template("error.html", code=500, message="服务端内部错误"), 500

    @app.errorhandler(Exception)
    def handle_any(exc):
        if isinstance(exc, VisionLabError):
            return handle_visionlab(exc)
        log.error("未处理异常: %s\n%s", exc, traceback.format_exc())
        if request.path.startswith("/api/"):
            return fail(f"服务端异常: {exc}", 50000, 500)
        raise exc
