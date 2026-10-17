"""报告中心 API。"""

from __future__ import annotations

from pathlib import Path

from flask import Blueprint, Response, send_file

from core.exceptions import NotFoundError
from core.storage import to_path
from services import report_service

from .common import arg, current_user, ok, page_args, paginated, payload, require_perm

bp = Blueprint("reports", __name__, url_prefix="/api/v1")


@bp.get("/reports/templates")
def templates():
    return ok({"items": report_service.list_templates()})


@bp.post("/reports")
@require_perm("report:create")
def create_report():
    data = payload()
    user = current_user()
    report = report_service.create_report(report_service.ReportRequest(
        name=str(data.get("name", "")), template=str(data.get("template", "experiment")),
        fmt=str(data.get("format", "pdf")), experiment_id=data.get("experiment_id"),
        project_id=data.get("project_id"), sections=data.get("sections"),
        params=data.get("params"), created_by=user["username"] if user else None))
    return ok(report)


@bp.get("/reports")
def list_reports():
    page, size = page_args()
    return paginated(report_service.list_reports(page, size, template=arg("template")))


@bp.get("/reports/<report_id>")
def get_report(report_id: str):
    report = report_service.get_report(report_id)
    if report is None:
        raise NotFoundError("报告不存在。")
    return ok(report)


@bp.get("/reports/<report_id>/download")
def download(report_id: str):
    report = report_service.get_report(report_id)
    if report is None or not report.get("file_uri"):
        raise NotFoundError("报告文件不存在。")
    path = to_path(report["file_uri"])
    if not path or not Path(path).exists():
        raise NotFoundError("报告文件已被移除。")
    return send_file(str(path), as_attachment=True, download_name=Path(path).name)


@bp.get("/reports/<report_id>/preview")
def preview(report_id: str):
    """以纯文本返回 Markdown 预览（PDF 无法内嵌时使用）。"""
    from services import analysis_service, project_service

    report = report_service.get_report(report_id)
    if report is None:
        raise NotFoundError("报告不存在。")
    payload_data = report_service.collect(  # noqa: SLF001
        report["template"], report.get("experiment_id"), report.get("project_id"),
        report.get("params") or {}, None)
    text = report_service.render_markdown(payload_data)  # noqa: SLF001
    return Response(text, mimetype="text/markdown; charset=utf-8")


@bp.post("/reports/<report_id>/share")
@require_perm("report:create")
def share(report_id: str):
    report = report_service.get_report(report_id)
    if report is None:
        raise NotFoundError("报告不存在。")
    return ok({"share_token": report.get("share_token"),
               "url": f"/api/v1/share/{report.get('share_token')}"})


@bp.get("/share/<token>")
def shared(token: str):
    report = report_service.get_by_token(token)
    if report is None:
        raise NotFoundError("分享链接无效或已失效。")
    if not report.get("file_uri"):
        raise NotFoundError("报告文件不存在。")
    path = to_path(report["file_uri"])
    if not path or not Path(path).exists():
        raise NotFoundError("报告文件已被移除。")
    return send_file(str(path), as_attachment=False, download_name=Path(path).name)


@bp.delete("/reports/<report_id>")
@require_perm("report:create")
def delete_report(report_id: str):
    report_service.delete_report(report_id)
    return ok({"deleted": True})
