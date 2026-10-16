"""系统健康：环境自检、运行指标、日志查询（对应 15 号模块 M15）。"""

from __future__ import annotations

import importlib.metadata as md
import platform
import shutil
import sys
import time
from pathlib import Path

from core import db, storage
from core.camera.service import get_camera_service
from core.config import ROOT, settings
from core.exceptions import NotFoundError
from core.logging_setup import get_logger, recent_logs

log = get_logger("services.health")


def _version(name: str) -> str | None:
    try:
        return md.version(name)
    except Exception:  # noqa: BLE001
        return None


def system_info() -> dict:
    info = {
        "app": settings.get("app.name"),
        "version": settings.get("app.version"),
        "code_name": settings.get("app.code_name"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "root": str(ROOT),
        "storage_root": str(settings.storage_root),
        "db_path": str(settings.db_path),
        "db_size": storage.human_size(settings.db_path.stat().st_size) if settings.db_path.exists() else "0 B",
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            free, total = torch.cuda.mem_get_info()
            info["gpu_mem_free_gb"] = round(free / 1024 ** 3, 2)
            info["gpu_mem_total_gb"] = round(total / 1024 ** 3, 2)
    except Exception as exc:  # noqa: BLE001
        info["torch"] = None
        info["torch_error"] = str(exc)
    for name in ("ultralytics", "opencv-python", "numpy", "pandas", "scipy",
                 "pyrealsense2", "flask", "flask-socketio", "plotly", "reportlab", "pillow", "timm"):
        info[name] = _version(name)
    return info


def self_check() -> dict:
    """一键自检：依赖 / CUDA / RealSense / 数据库 / 磁盘 / 存储可写。"""
    checks: list[dict] = []

    def add(name: str, ok: bool, detail: str, level: str = "error") -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail,
                       "level": "ok" if ok else level})

    # 依赖
    required = {"flask": "Web 框架", "numpy": "数值计算", "opencv-python": "图像处理",
                "pyyaml": "配置解析", "pillow": "中文文字渲染"}
    optional = {"ultralytics": "YOLO 检测（缺失则只能用深度基线检测器）",
                "torch": "PyTorch 分割/训练（缺失则只能用传统道路分割器）",
                "pyrealsense2": "RealSense 采集（缺失则只能用合成场景）",
                "reportlab": "PDF 报告（缺失则只能导出 Markdown/CSV）",
                "timm": "UNetV2/PVTv2（缺失则该结构不可用）",
                "scipy": "比率校正拟合（缺失则退化为网格搜索）",
                "plotly": "前端交互图表（缺失则大屏降级为静态图）"}
    for name, purpose in required.items():
        version = _version(name)
        add(f"依赖 {name}", version is not None, f"{purpose}：{version or '未安装'}")
    for name, purpose in optional.items():
        version = _version(name)
        add(f"可选依赖 {name}", version is not None,
            f"{purpose}：{version or '未安装'}", level="warning")

    # CUDA
    try:
        import torch

        if torch.cuda.is_available():
            add("CUDA 加速", True,
                f"{torch.cuda.get_device_name(0)} / CUDA {torch.version.cuda}")
        else:
            add("CUDA 加速", False, "未检测到可用 GPU，训练与推理将使用 CPU（较慢）", "warning")
    except Exception as exc:  # noqa: BLE001
        add("CUDA 加速", False, f"PyTorch 不可用：{exc}", "warning")

    # RealSense
    try:
        from core.camera import realsense_backend as rsb

        if rsb.available():
            devices = rsb.list_devices()
            if devices:
                names = ", ".join(f"{d['name']}({d['serial']})" for d in devices)
                add("RealSense 设备", True, names)
            else:
                add("RealSense 设备", False,
                    "pyrealsense2 已安装但未发现设备，将使用合成场景（MockBackend）", "warning")
        else:
            add("RealSense 设备", False,
                "未安装 pyrealsense2，将使用合成场景（MockBackend）", "warning")
    except Exception as exc:  # noqa: BLE001
        add("RealSense 设备", False, f"检测失败：{exc}", "warning")

    # 数据库
    try:
        counts = db.table_counts()
        add("元数据库", True, f"{len(counts)} 张表，"
                              f"实验 {counts.get('experiment', 0)} 条，"
                              f"测距记录 {counts.get('ranging_result', 0)} 条")
    except Exception as exc:  # noqa: BLE001
        add("元数据库", False, f"访问失败：{exc}")

    # 存储可写
    try:
        probe = settings.storage_root / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        add("存储目录可写", True, str(settings.storage_root))
    except Exception as exc:  # noqa: BLE001
        add("存储目录可写", False, f"写入失败：{exc}")

    # 磁盘
    free_gb = storage.free_space_gb()
    add("磁盘空间", free_gb < 0 or free_gb > 5,
        f"剩余 {free_gb} GB" if free_gb >= 0 else "无法读取", "warning" if free_gb < 5 else "error")

    # 目录
    missing = [name for name in ("datasets", "captures", "models", "results", "reports", "logs")
               if not settings.storage_dir(name).exists()]
    add("运行期目录", not missing, "全部就绪" if not missing else f"缺失：{missing}")

    failed = [c for c in checks if not c["ok"] and c["level"] == "error"]
    warned = [c for c in checks if not c["ok"] and c["level"] == "warning"]
    return {
        "checked_at": db.now_iso(),
        "checks": checks,
        "summary": {
            "total": len(checks),
            "passed": sum(1 for c in checks if c["ok"]),
            "errors": len(failed),
            "warnings": len(warned),
        },
        "conclusion": ("环境就绪，可以开始实验" if not failed
                       else f"存在 {len(failed)} 项必须处理的问题"),
    }


def metrics() -> dict:
    data: dict = {"timestamp": time.time(), "queue": _queue_metrics(), "camera": _camera_metrics(),
                  "disk": _disk_metrics()}
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            data["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "mem_used_gb": round((total - free) / 1024 ** 3, 2),
                "mem_total_gb": round(total / 1024 ** 3, 2),
                "utilization": round((total - free) / max(total, 1) * 100, 1),
            }
    except Exception:  # noqa: BLE001
        pass
    try:
        import os

        if hasattr(os, "getloadavg"):
            data["loadavg"] = [round(x, 2) for x in os.getloadavg()]
    except Exception:  # noqa: BLE001
        pass
    return data


def _queue_metrics() -> dict:
    rows = db.query_all("SELECT status, COUNT(*) AS n FROM task_queue GROUP BY status")
    out = {r["status"]: r["n"] for r in rows}
    running = db.query_all("SELECT id, task_type, started_at FROM task_queue "
                           "WHERE status='running' LIMIT 10")
    return {"counts": out, "running": running}


def _camera_metrics() -> dict:
    camera = get_camera_service()
    info = camera.describe()
    return {
        "running": info.get("running"),
        "backend": info.get("backend"),
        "model": info.get("model"),
        "serial": info.get("serial"),
        "usb": info.get("usb_type"),
        "fps_actual": info.get("fps_actual"),
        "frames": info.get("frames"),
        "recording": camera.recording,
        "health": info.get("health"),
    }


def _disk_metrics() -> dict:
    out = {"free_gb": storage.free_space_gb(), "root": str(settings.storage_root), "dirs": {}}
    for name in ("datasets", "captures", "models", "runs", "results", "reports", "logs"):
        path = settings.storage_dir(name)
        out["dirs"][name] = {"files": sum(1 for _ in path.rglob("*") if _.is_file()),
                             "size": storage.human_size(storage.dir_size(path))}
    return out


def logs(level: str | None = None, keyword: str | None = None, limit: int = 200) -> list[dict]:
    items = recent_logs(level=level, keyword=keyword, limit=limit)
    if items:
        return items
    # 进程重启后内存缓冲为空，退化为读取日志文件尾部
    path = settings.log_path
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()[-limit:]
    out = []
    for line in reversed(lines):
        if level and f"[{level.upper()}]" not in line:
            continue
        if keyword and keyword.lower() not in line.lower():
            continue
        out.append({"time": line[1:20] if line.startswith("[") else "", "level": "INFO",
                    "module": "file", "message": line})
    return out


def log_file_tail(lines: int = 100) -> dict:
    path = settings.log_path
    if not path.exists():
        return {"path": str(path), "exists": False, "lines": []}
    content = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return {"path": str(path), "exists": True, "size": storage.human_size(path.stat().st_size),
            "lines": content[-lines:]}


def clear_log_buffer() -> None:
    from core import logging_setup

    logging_setup.attach_memory_handler().records.clear()
    log.info("内存日志缓冲已清空")


def get_log(log_id: str) -> dict:
    rows = [r for r in recent_logs(limit=1000) if str(r.get("time")) == log_id]
    if not rows:
        raise NotFoundError("日志不存在。")
    return rows[0]
