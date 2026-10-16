"""HTML 页面路由（18 个页面）。"""

from __future__ import annotations

from flask import Blueprint, redirect, render_template, url_for

from .common import current_user, load_user

bp = Blueprint("pages", __name__)

PAGES = [
    ("dashboard", "工作台", "dashboard", "工作台"),
    ("projects", "项目管理", "projects", "实验中心"),
    ("experiments", "实验管理", "experiments", "实验中心"),
    ("training", "训练实验", "training", "实验中心"),
    ("experiment_detection", "检测实验", "detection", "实验中心"),
    ("experiment_segmentation", "分割实验", "segmentation", "实验中心"),
    ("ranging", "测距实验", "ranging", "实验中心"),
    ("calibration", "相机标定", "calibration", "数据资产"),
    ("capture", "数据采集", "capture", "数据资产"),
    ("datasets", "数据集管理", "datasets", "数据资产"),
    ("annotations", "标注管理", "annotations", "数据资产"),
    ("models", "模型管理", "models", "模型资产"),
    ("analysis", "误差分析", "analysis", "分析演示"),
    ("bigscreen", "可视化大屏", "bigscreen", "分析演示"),
    ("reports", "报告中心", "reports", "分析演示"),
    ("settings", "系统设置", "settings", "系统管理"),
    ("monitor", "日志与监控", "monitor", "系统管理"),
]


def _ctx(page: str, title: str, **extra) -> dict:
    return {
        "page": page, "page_title": title, "nav": _nav(), "user": current_user(),
        "app_name": "视界智研", "app_version": "1.0.0", **extra,
    }


def _nav() -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for page, label, group, _ in PAGES:
        if page == "dashboard":
            continue
        groups.setdefault(group, []).append({"page": page, "label": label})
    return [{"group": name, "items": items} for name, items in groups.items()]


#: 页面 key -> 路由
NAV_URLS: dict[str, str] = {
    "dashboard": "/",
    "projects": "/projects",
    "project_detail": "/projects",
    "experiments": "/experiments",
    "experiment_detail": "/experiments",
    "training": "/training",
    "experiment_detection": "/experiments/detection",
    "experiment_segmentation": "/experiments/segmentation",
    "ranging": "/experiments/ranging",
    "calibration": "/camera/calibration",
    "capture": "/camera/capture",
    "datasets": "/datasets",
    "dataset_detail": "/datasets",
    "annotations": "/annotations",
    "models": "/models",
    "analysis": "/analysis",
    "bigscreen": "/bigscreen",
    "reports": "/reports",
    "settings": "/settings",
    "monitor": "/monitor",
}

#: 页面 key -> 图标字符
NAV_ICONS: dict[str, str] = {
    "projects": "▣", "experiments": "◈", "training": "⚙", "experiment_detection": "▢",
    "experiment_segmentation": "◧", "ranging": "↔", "calibration": "⌖", "capture": "◉",
    "datasets": "▤", "dataset_detail": "▤", "annotations": "✎", "models": "◫",
    "analysis": "◔", "bigscreen": "▩", "reports": "▦", "settings": "⚒", "monitor": "◍",
}


def nav_url(page: str) -> str:
    return NAV_URLS.get(page, "/")


def nav_icon(page: str) -> str:
    return NAV_ICONS.get(page, "•")


@bp.get("/login")
def login_page():
    if load_user():
        return redirect(url_for("pages.dashboard"))
    return render_template("login.html", page="login", page_title="登录", app_name="视界智研",
                           app_version="1.0.0", nav=[], user=None)


@bp.get("/")
def dashboard():
    return render_template("dashboard.html", **_ctx("dashboard", "工作台"))


@bp.get("/projects")
def projects():
    return render_template("projects.html", **_ctx("projects", "项目管理"))


@bp.get("/projects/<project_id>")
def project_detail(project_id: str):
    return render_template("project_detail.html", **_ctx("project_detail", "项目详情",
                                                         project_id=project_id))


@bp.get("/experiments")
def experiments():
    return render_template("experiments.html", **_ctx("experiments", "实验管理"))


@bp.get("/experiments/detection")
def experiment_detection():
    return render_template("experiment_detection.html",
                           **_ctx("experiment_detection", "目标检测实验"))


@bp.get("/experiments/segmentation")
def experiment_segmentation():
    return render_template("experiment_segmentation.html",
                           **_ctx("experiment_segmentation", "道路分割实验"))


@bp.get("/experiments/ranging")
def experiment_ranging():
    return render_template("ranging.html", **_ctx("ranging", "测距实验"))


@bp.get("/experiments/<experiment_id>")
def experiment_detail(experiment_id: str):
    return render_template("experiment_detail.html",
                           **_ctx("experiment_detail", "实验详情", experiment_id=experiment_id))


@bp.get("/camera/calibration")
def calibration():
    return render_template("calibration.html", **_ctx("calibration", "相机标定"))


@bp.get("/camera/capture")
def capture():
    return render_template("capture.html", **_ctx("capture", "数据采集"))


@bp.get("/datasets")
def datasets():
    return render_template("datasets.html", **_ctx("datasets", "数据集管理"))


@bp.get("/datasets/<dataset_id>")
def dataset_detail(dataset_id: str):
    return render_template("dataset_detail.html", **_ctx("dataset_detail", "数据集详情",
                                                          dataset_id=dataset_id))


@bp.get("/annotations")
def annotations():
    return render_template("annotations.html", **_ctx("annotations", "标注管理"))


@bp.get("/models")
def models():
    return render_template("models.html", **_ctx("models", "模型管理"))


@bp.get("/training")
def training():
    return render_template("training.html", **_ctx("training", "训练实验"))


@bp.get("/analysis")
def analysis():
    return render_template("analysis.html", **_ctx("analysis", "误差分析"))


@bp.get("/bigscreen")
def bigscreen():
    return render_template("bigscreen.html", **_ctx("bigscreen", "可视化大屏"))


@bp.get("/reports")
def reports():
    return render_template("reports.html", **_ctx("reports", "报告中心"))


@bp.get("/settings")
def settings_page():
    return render_template("settings.html", **_ctx("settings", "系统设置"))


@bp.get("/monitor")
def monitor():
    return render_template("monitor.html", **_ctx("monitor", "日志与监控"))
