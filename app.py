"""视界智研 —— 基于双目视觉测距与目标检测算法实验平台。

启动：
    python app.py
或：
    python app.py --port 5000 --host 0.0.0.0

技术栈沿用两个原项目：Flask + Flask-SocketIO + Jinja2 + 原生 JS + Plotly +
PyTorch + Ultralytics + OpenCV + pyrealsense2 + NumPy/Pandas/SciPy + ReportLab；
元数据使用 Python 标准库 sqlite3（零额外依赖）。
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core import db  # noqa: E402
from core.config import settings  # noqa: E402
from core.logging_setup import get_logger, setup_logging  # noqa: E402

log = get_logger("app")


def create_app():
    from flask import Flask

    from api import register_blueprints
    from api import ws as ws_module
    from api.common import register_error_handlers
    from api.pages import bp as pages_bp
    from services import auth_service

    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["SECRET_KEY"] = settings.get("server.secret_key", "visionlab")
    app.config["JSON_AS_ASCII"] = False
    app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

    register_error_handlers(app)
    register_blueprints(app)
    app.register_blueprint(pages_bp)

    socketio = None
    if settings.get("server.realtime", "auto") != "off":
        try:
            from flask_socketio import SocketIO

            socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading",
                                ping_timeout=30, ping_interval=10)
            ws_module.init(socketio)
        except ImportError:
            log.warning("未安装 flask-socketio，实时推流降级为轮询模式")
            socketio = None
    app.extensions["socketio"] = socketio

    @app.context_processor
    def inject_globals():
        return {"app_name": settings.get("app.name", "视界智研"),
                "app_short": settings.get("app.short_name", "视界智研"),
                "app_version": settings.get("app.version", "1.0.0"),
                "has_socketio": socketio is not None}

    from api.pages import nav_icon, nav_url

    app.jinja_env.globals["nav_url"] = nav_url
    app.jinja_env.globals["nav_icon"] = nav_icon

    @app.before_request
    def guard_pages():
        """页面请求未登录时跳转登录页；API 请求仍返回标准 401。"""
        from flask import redirect, request

        path = request.path
        if path.startswith("/api/") or path.startswith("/static/"):
            return None
        if path in ("/login", "/favicon.ico"):
            return None
        from api.common import load_user

        if load_user() is None:
            return redirect(f"/login?next={path}")
        return None

    @app.after_request
    def add_headers(response):
        response.headers.setdefault("X-Platform", "VisionLab-D455")
        if response.mimetype == "application/json":
            response.headers["Cache-Control"] = "no-store"
        return response

    with app.app_context():
        db.init_db()
        created = auth_service.ensure_default_admin()
        if created:
            log.info("默认管理员已创建: %s", created["username"])
        from algorithms import registry

        registry.load_builtin()
    return app


def bootstrap_data() -> None:
    """首次启动的准备：默认项目、相机自检、必要时启动合成场景。"""
    from core.camera.service import get_camera_service
    from services import model_registry, project_service

    if not db.query_scalar("SELECT COUNT(*) FROM project WHERE is_deleted=0", (), 0):
        project = project_service.create_project(project_service.ProjectSpec(
            "校园道路视觉感知", "视界智研平台的默认项目，用于承载检测/分割/测距实验",
            scene="campus_road", app_target="sweeper"))
        log.info("已创建默认项目: %s", project["name"])

    if not db.query_scalar("SELECT COUNT(*) FROM model WHERE is_deleted=0", (), 0):
        try:
            result = model_registry.scan_platform_models()
            if result.get("imported"):
                log.info("已登记平台模型: %d 个", result["imported"])
        except Exception as exc:  # noqa: BLE001
            log.warning("扫描平台模型失败: %s", exc)

    camera = get_camera_service()
    if not camera.running:
        try:
            info = camera.start()
            log.info("相机已就绪: backend=%s %sx%s", info.get("backend"),
                     info.get("width"), info.get("height"))
        except Exception as exc:  # noqa: BLE001
            log.warning("相机启动失败: %s", exc)

    from services import pipeline_service

    pipeline_service.get_pipeline()
    try:
        from services import analysis_service

        plan = analysis_service.active_calibration()
        if plan:
            pipeline_service.get_pipeline().set_calibration(plan)
            log.info("已启用校正方案: %s", plan.name)
    except Exception as exc:  # noqa: BLE001
        log.warning("加载校正方案失败: %s", exc)


def main() -> int:
    parser = argparse.ArgumentParser(description="视界智研算法实验平台")
    parser.add_argument("--host", default=None, help="监听地址（默认取配置）")
    parser.add_argument("--port", type=int, default=None, help="端口（默认取配置）")
    parser.add_argument("--debug", action="store_true", help="开启调试模式")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    parser.add_argument("--no-camera", action="store_true", help="强制使用合成场景（离线演示）")
    args = parser.parse_args()

    setup_logging()
    if args.no_camera:
        import os

        os.environ["VISIONLAB_CAMERA__BACKEND"] = "mock"

    app = create_app()
    bootstrap_data()

    host = args.host or settings.get("server.host", "0.0.0.0")
    port = int(args.port or settings.get("server.port", 5000))
    debug = bool(args.debug or settings.get("server.debug", False))

    from api import ws as ws_module

    ws_module.start_streamer()

    url = f"http://127.0.0.1:{port}/"
    banner = f"""
============================================================
  {settings.get('app.name')}
  版本 {settings.get('app.version')}   代号 {settings.get('app.code_name')}
------------------------------------------------------------
  访问地址   {url}
  存储根目录 {settings.storage_root}
  元数据库   {settings.db_path}
  日志文件   {settings.log_path}
------------------------------------------------------------
  首次登录   admin / admin123   （请登录后立即修改密码）
  离线演示   无 RealSense 设备时自动使用合成场景
============================================================
"""
    log.info(banner)
    log.info("平台启动: %s", url)

    if not args.no_browser and not debug:
        def _open() -> None:
            time.sleep(1.5)
            try:
                webbrowser.open(url)
            except Exception:  # noqa: BLE001
                pass

        threading.Thread(target=_open, daemon=True).start()

    socketio = app.extensions.get("socketio")
    try:
        if socketio is not None:
            socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True,
                         use_reloader=False)
        else:
            app.run(host=host, port=port, debug=debug, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        log.info("收到中断信号，正在停止…")
    finally:
        ws_module.stop_streamer()
        from core.camera.service import get_camera_service

        get_camera_service().stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
