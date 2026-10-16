"""实时通道：Socket.IO 事件与后台推流线程。

频道约定：
  stream:<session>      推理叠加帧 + 结构化测距结果（约 10–15 fps）
  experiment:<id>       训练/推理进度与状态变更
  log                   日志追加
  system:metrics        运行指标
"""

from __future__ import annotations

import threading
import time
from typing import Any

from core.config import settings
from core.logging_setup import get_logger
from services import pipeline_service

log = get_logger("api.ws")

_socketio = None
_thread: threading.Thread | None = None
_stop = threading.Event()
_subscribers: set[str] = set()
_lock = threading.Lock()
_stats = {"frames": 0, "errors": 0, "last_latency_ms": 0.0}


def init(socketio) -> None:
    global _socketio
    _socketio = socketio

    @socketio.on("connect")
    def on_connect():
        sid = _current_sid()
        log.info("实时通道连接: %s", sid)
        socketio.emit("server_ready", {"message": "已连接", "version": settings.get("app.version")},
                      to=sid)

    @socketio.on("disconnect")
    def on_disconnect():
        sid = _current_sid()
        with _lock:
            _subscribers.discard(sid)
        log.info("实时通道断开: %s", sid)

    @socketio.on("subscribe")
    def on_subscribe(data):
        sid = _current_sid()
        channel = str((data or {}).get("channel", "stream"))
        with _lock:
            _subscribers.add(sid)
        socketio.emit("subscribed", {"channel": channel, "sid": sid}, to=sid)

    @socketio.on("unsubscribe")
    def on_unsubscribe(_data=None):
        sid = _current_sid()
        with _lock:
            _subscribers.discard(sid)
        socketio.emit("unsubscribed", {"sid": sid}, to=sid)

    @socketio.on("ping_server")
    def on_ping(_data=None):
        socketio.emit("pong_server", {"ts": time.time()}, to=_current_sid())


def _current_sid() -> str:
    try:
        from flask import request

        return str(getattr(request, "sid", "") or "")
    except Exception:  # noqa: BLE001
        return ""


def start_streamer() -> None:
    """启动后台推流线程（无订阅者时不消耗算力）。"""
    global _thread
    if _thread is not None and _thread.is_alive():
        return
    _stop.clear()
    _thread = threading.Thread(target=_loop, name="VisionLabStreamer", daemon=True)
    _thread.start()
    log.info("实时推流线程已启动")


def stop_streamer() -> None:
    _stop.set()
    if _thread is not None:
        _thread.join(timeout=2)


def _loop() -> None:  # pragma: no cover - 依赖运行期
    interval = 1.0 / max(1, int(settings.get("server.stream.fps", 15)))
    while not _stop.is_set():
        try:
            with _lock:
                has_sub = bool(_subscribers)
            if not has_sub or _socketio is None:
                time.sleep(0.3)
                continue
            payload = _render_payload()
            if payload is not None:
                _socketio.emit("frame", payload)
                _stats["frames"] += 1
            time.sleep(interval)
        except Exception as exc:  # noqa: BLE001
            _stats["errors"] += 1
            log.warning("推流异常: %s", exc)
            time.sleep(0.5)


def _render_payload() -> dict | None:
    import base64

    from core import media

    bundle, truth = pipeline_service.current_frame_bundle()
    if bundle is None:
        return None
    pipeline = pipeline_service.get_pipeline()
    t0 = time.perf_counter()
    result = pipeline.process(bundle, truth)
    overlay = pipeline.overlay_frame()
    if overlay is None:
        return None
    max_w = int(settings.get("server.stream.max_width", 960))
    quality = int(settings.get("server.stream.jpeg_quality", 75))
    view = media.resize_keep_ratio(overlay, max_w, 10000)
    depth_view = media.colorize_depth(bundle.depth_m,
                                      float(settings.get("ranging.min_range_m", 0.3)), 15.0)
    depth_view = media.resize_keep_ratio(depth_view, max_w, 10000)
    latency = (time.perf_counter() - t0) * 1000.0
    _stats["last_latency_ms"] = round(latency, 2)

    return {
        "type": "frame",
        "ts": time.time(),
        "frame_number": bundle.frame_number,
        "source": bundle.source,
        "mode": result["mode"],
        "image": base64.b64encode(media.encode_jpeg(view, quality)).decode(),
        "depth_image": base64.b64encode(media.encode_jpeg(depth_view, quality)).decode(),
        "detection": {
            "count": len(result["detection"]["detections"]),
            "infer_ms": result["detection"]["infer_ms"],
            "arch": result["detection"].get("arch"),
            "reason": result["detection"].get("reason"),
        },
        "road": result["road"],
        "objects": result["objects"],
        "errors": result["errors"],
        "paused": result["paused"],
        "plan_hint": result["plan_hint"],
        "session": dict(pipeline.session),
        "perf": {**result["perf"], "latency_ms": _stats["last_latency_ms"]},
    }


def emit_experiment(experiment_id: str, event: str, data: dict) -> None:
    if _socketio is None:
        return
    _socketio.emit("experiment_event", {"experiment_id": experiment_id, "event": event, **data})


def emit_log(record: dict) -> None:
    if _socketio is None:
        return
    _socketio.emit("log_entry", record)


def stats() -> dict:
    with _lock:
        subs = len(_subscribers)
    return {**_stats, "subscribers": subs,
            "running": bool(_thread and _thread.is_alive()),
            "target_fps": settings.get("server.stream.fps")}


def broadcast_metrics(payload: dict) -> None:
    if _socketio is None:
        return
    _socketio.emit("metrics", payload)
