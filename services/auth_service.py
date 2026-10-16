"""用户、认证、RBAC 权限与审计。"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta
from typing import Any

from core import db
from core.config import settings
from core.exceptions import AuthError, PermissionError_, ValidationError
from core.logging_setup import get_logger

log = get_logger("services.auth")

# 权限点定义：角色 -> 允许的动作集合
ROLE_PERMISSIONS: dict[str, set[str]] = {
    "admin": {"*"},
    "teacher": {
        "project:read", "experiment:read", "experiment:export", "model:read",
        "dataset:read", "analysis:read", "report:read", "report:create",
        "dashboard:read", "system:read", "monitor:read", "audit:read",
    },
    "researcher": {
        "project:read", "project:create", "project:update",
        "experiment:read", "experiment:create", "experiment:update", "experiment:run",
        "experiment:archive", "experiment:export",
        "dataset:read", "dataset:create", "dataset:update", "dataset:import", "dataset:delete",
        "model:read", "model:create", "model:update", "model:delete", "model:export",
        "training:run", "inference:run", "ranging:run", "ranging:write",
        "analysis:read", "analysis:calibrate",
        "camera:read", "camera:control", "capture:run", "calibration:run",
        "report:read", "report:create", "dashboard:read", "monitor:read",
    },
    "viewer": {"dashboard:read", "report:read", "experiment:read", "model:read",
               "dataset:read", "analysis:read"},
}

ROLE_LABELS = {"admin": "系统管理员", "teacher": "指导教师",
               "researcher": "算法研究者", "viewer": "只读访客"}

_ITERATIONS = 120_000


# ------------------------------------------------------------------ 口令
def hash_password(password: str, salt: str | None = None) -> str:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), _ITERATIONS)
    return f"pbkdf2_sha256${_ITERATIONS}${salt}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iterations, salt, digest = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        calc = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"),
                                   salt.encode("utf-8"), int(iterations))
        return hmac.compare_digest(calc.hex(), digest)
    except Exception:  # noqa: BLE001
        return False


# ------------------------------------------------------------------ 令牌
def _secret() -> bytes:
    raw = str(settings.get("server.secret_key", "visionlab"))
    return raw.encode("utf-8")


def issue_token(user: dict) -> str:
    expires = datetime.now() + timedelta(hours=int(settings.get("server.session_hours", 12)))
    payload = f"{user['id']}|{user['username']}|{user['role']}|{int(expires.timestamp())}"
    signature = hmac.new(_secret(), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{payload}|{signature}"


def parse_token(token: str) -> dict:
    try:
        user_id, username, role, exp, signature = token.split("|")
    except ValueError as exc:
        raise AuthError("令牌格式无效，请重新登录。") from exc
    payload = f"{user_id}|{username}|{role}|{exp}"
    expected = hmac.new(_secret(), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature):
        raise AuthError("令牌签名校验失败，请重新登录。")
    if int(exp) < int(datetime.now().timestamp()):
        raise AuthError("登录已过期，请重新登录。")
    return {"id": user_id, "username": username, "role": role}


# ------------------------------------------------------------------ 用户
def ensure_default_admin() -> dict | None:
    """首次启动创建管理员账号。"""
    if db.query_scalar("SELECT COUNT(*) FROM user", (), 0):
        return None
    cfg = settings.section("security").get("first_run_admin", {}) or {}
    username = str(cfg.get("username", "admin"))
    password = str(cfg.get("password", "admin123"))
    user_id = create_user(username, password, role="admin", display_name="系统管理员")
    log.warning("已创建默认管理员账号 %s / %s，请登录后立即修改密码", username, password)
    return get_user(user_id)


def create_user(username: str, password: str, role: str = "researcher",
                display_name: str | None = None, email: str | None = None) -> str:
    username = str(username).strip()
    if not username or len(username) < 3:
        raise ValidationError("用户名至少 3 个字符。")
    if len(str(password)) < 6:
        raise ValidationError("密码至少 6 位。")
    if role not in ROLE_PERMISSIONS:
        raise ValidationError(f"未知角色: {role}")
    if db.query_one("SELECT id FROM user WHERE username=?", (username,)):
        raise ValidationError(f"用户名已存在: {username}")
    return db.insert("user", {
        "username": username, "display_name": display_name or username,
        "password_hash": hash_password(password), "role": role, "email": email,
        "is_active": 1,
    }, id_prefix="u")


def get_user(user_id: str) -> dict | None:
    return db.decode_row(db.query_one("SELECT * FROM user WHERE id=?", (user_id,)))


def get_user_by_name(username: str) -> dict | None:
    return db.decode_row(db.query_one("SELECT * FROM user WHERE username=?", (username,)))


def list_users() -> list[dict]:
    rows = db.query_all("SELECT * FROM user ORDER BY created_at")
    return [{k: v for k, v in r.items() if k != "password_hash"} for r in rows]


def update_user(user_id: str, **fields: Any) -> int:
    allowed = {"display_name", "role", "email", "is_active"}
    payload = {k: v for k, v in fields.items() if k in allowed}
    if "role" in payload and payload["role"] not in ROLE_PERMISSIONS:
        raise ValidationError(f"未知角色: {payload['role']}")
    return db.update("user", user_id, payload)


def reset_password(user_id: str, new_password: str) -> int:
    if len(str(new_password)) < 6:
        raise ValidationError("密码至少 6 位。")
    return db.update("user", user_id, {"password_hash": hash_password(new_password),
                                       "failed_attempts": 0, "locked_until": None})


def change_password(user_id: str, old_password: str, new_password: str) -> int:
    user = get_user(user_id)
    if user is None:
        raise AuthError("用户不存在。")
    if not verify_password(old_password, user["password_hash"]):
        raise AuthError("原密码不正确。")
    return reset_password(user_id, new_password)


# ------------------------------------------------------------------ 登录
def login(username: str, password: str, ip: str = "") -> dict:
    user = get_user_by_name(username)
    if user is None:
        audit(None, username, "login", "user", None, "failed", ip, {"reason": "no_such_user"})
        raise AuthError("用户名或密码错误。")
    if not int(user.get("is_active", 1)):
        raise AuthError("账号已被禁用，请联系管理员。")

    locked_until = user.get("locked_until")
    if locked_until and datetime.fromisoformat(locked_until) > datetime.now():
        remaining = int((datetime.fromisoformat(locked_until) - datetime.now()).total_seconds())
        raise AuthError(f"账号已锁定，请 {remaining} 秒后重试。")

    if not verify_password(password, user["password_hash"]):
        failures = int(user.get("failed_attempts", 0)) + 1
        payload = {"failed_attempts": failures}
        max_fail = int(settings.get("security.max_login_failures", 5))
        if failures >= max_fail:
            lock_seconds = int(settings.get("security.lock_seconds", 60))
            payload["locked_until"] = (datetime.now() + timedelta(seconds=lock_seconds)).isoformat()
        db.update("user", user["id"], payload)
        audit(user["id"], username, "login", "user", user["id"], "failed", ip,
              {"reason": "bad_password", "failures": failures})
        raise AuthError("用户名或密码错误。")

    db.update("user", user["id"], {"failed_attempts": 0, "locked_until": None,
                                   "last_login_at": db.now_iso()})
    audit(user["id"], username, "login", "user", user["id"], "success", ip)
    return {
        "token": issue_token(user),
        "user": {"id": user["id"], "username": user["username"],
                 "display_name": user.get("display_name"), "role": user["role"],
                 "role_label": ROLE_LABELS.get(user["role"], user["role"])},
        "permissions": sorted(permissions_of(user["role"])),
    }


def permissions_of(role: str) -> set[str]:
    perms = ROLE_PERMISSIONS.get(role, set())
    if "*" in perms:
        merged: set[str] = set()
        for value in ROLE_PERMISSIONS.values():
            merged |= value
        return merged
    return set(perms)


def check_permission(user: dict | None, action: str) -> bool:
    if user is None:
        return False
    perms = ROLE_PERMISSIONS.get(user.get("role", ""), set())
    return "*" in perms or action in perms


def require(user: dict | None, action: str) -> None:
    if user is None:
        raise AuthError("请先登录。")
    if not check_permission(user, action):
        raise PermissionError_(f"当前角色（{ROLE_LABELS.get(user.get('role'), user.get('role'))}）"
                               f"没有执行 [{action}] 的权限。")


# ------------------------------------------------------------------ 审计
def audit(user_id: str | None, username: str | None, action: str, resource_type: str = "",
          resource_id: str | None = None, result: str = "success", ip: str = "",
          detail: dict | None = None) -> None:
    try:
        db.insert("audit_log", {
            "user_id": user_id, "username": username, "action": action,
            "resource_type": resource_type, "resource_id": resource_id,
            "result": result, "ip": ip, "detail": detail or {},
        }, id_prefix="aud")
    except Exception as exc:  # noqa: BLE001 - 审计失败不得影响主流程
        log.warning("写入审计日志失败: %s", exc)


def list_audit(limit: int = 200, action: str | None = None) -> list[dict]:
    if action:
        rows = db.query_all("SELECT * FROM audit_log WHERE action LIKE ? ORDER BY created_at DESC LIMIT ?",
                            (f"%{action}%", limit))
    else:
        rows = db.query_all("SELECT * FROM audit_log ORDER BY created_at DESC LIMIT ?", (limit,))
    return db.decode_rows(rows)
