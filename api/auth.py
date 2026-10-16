"""认证与用户 API。"""

from __future__ import annotations

from flask import Blueprint, request

from core.exceptions import ValidationError
from services import auth_service

from .common import arg_bool, current_user, fail, login_required, ok, payload, require_perm

bp = Blueprint("auth", __name__, url_prefix="/api/v1")


@bp.post("/auth/login")
def login():
    data = payload()
    username = str(data.get("username", "")).strip()
    password = str(data.get("password", ""))
    if not username or not password:
        return fail("请输入用户名和密码。", 40000, 400)
    result = auth_service.login(username, password, request.remote_addr or "")
    response = ok(result)
    # 同时写入 Cookie，便于直接用浏览器访问页面路由
    response.set_cookie("vl_token", result["token"], httponly=False, samesite="Lax",
                        max_age=int(12 * 3600))
    return response


@bp.post("/auth/logout")
@login_required
def logout():
    user = current_user()
    auth_service.audit(user["id"], user["username"], "logout", "user", user["id"],
                       "success", request.remote_addr or "")
    response = ok({"logged_out": True})
    response.delete_cookie("vl_token")
    return response


@bp.get("/auth/me")
@login_required
def me():
    user = current_user()
    return ok({"user": {**user, "role_label": auth_service.ROLE_LABELS.get(user["role"], user["role"])},
               "permissions": sorted(auth_service.permissions_of(user["role"]))})


@bp.post("/auth/change-password")
@login_required
def change_password():
    user = current_user()
    data = payload()
    auth_service.change_password(user["id"], str(data.get("old_password", "")),
                                 str(data.get("new_password", "")))
    return ok({"changed": True})


@bp.get("/users")
@login_required
def list_users():
    actor = current_user()
    if not auth_service.check_permission(actor, "system:read") and actor["role"] != "admin":
        return fail("无权查看用户列表。", 40300, 403)
    return ok({"items": auth_service.list_users(),
               "roles": [{"key": k, "label": v} for k, v in auth_service.ROLE_LABELS.items()]})


@bp.post("/users")
@require_perm("system:manage")
def create_user():
    data = payload()
    user_id = auth_service.create_user(
        str(data.get("username", "")), str(data.get("password", "")),
        role=str(data.get("role", "researcher")),
        display_name=data.get("display_name"), email=data.get("email"))
    auth_service.audit(current_user()["id"], current_user()["username"], "user.create",
                       "user", user_id, "success", request.remote_addr or "")
    return ok(auth_service.get_user(user_id))


@bp.patch("/users/<user_id>")
@require_perm("system:manage")
def update_user(user_id: str):
    data = payload()
    auth_service.update_user(user_id, **data)
    return ok(auth_service.get_user(user_id))


@bp.post("/users/<user_id>/reset-password")
@require_perm("system:manage")
def reset_password(user_id: str):
    data = payload()
    auth_service.reset_password(user_id, str(data.get("new_password", "")))
    return ok({"reset": True})


@bp.get("/audit")
@require_perm("audit:read")
def audit_log():
    return ok({"items": auth_service.list_audit(limit=200)})
