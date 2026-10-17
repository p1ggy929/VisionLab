"""报告中心：模板、渲染（PDF / Markdown / CSV / JSON）、归档与分享。"""

from __future__ import annotations

import csv
import io
import json
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core import db, storage
from core.config import settings
from core.exceptions import NotFoundError, ValidationError
from core.logging_setup import get_logger

log = get_logger("services.report")

TEMPLATES: dict[str, dict] = {
    "experiment": {
        "label": "实验报告",
        "sections": ["overview", "config", "metrics", "ranging", "figures", "manifest"],
        "description": "单个实验的完整记录：配置、指标、测距明细与复现清单",
    },
    "compare": {
        "label": "对比报告",
        "sections": ["overview", "metrics", "leaderboard", "figures"],
        "description": "多个实验/模型的横向指标对比",
    },
    "error": {
        "label": "误差分析报告",
        "sections": ["overview", "metrics", "by_distance", "by_category", "calibration", "outliers"],
        "description": "测距误差统计、六维归因、校正效果与异常样本",
    },
    "dataset": {
        "label": "数据集报告",
        "sections": ["overview", "dataset_stats", "validation"],
        "description": "数据集版本、划分、类别分布与校验结论",
    },
    "summary": {
        "label": "项目总结报告",
        "sections": ["overview", "kpi", "metrics", "leaderboard", "by_distance"],
        "description": "全平台或全项目的总体情况汇总",
    },
}


def list_templates() -> list[dict]:
    return [{"key": k, **v} for k, v in TEMPLATES.items()]


@dataclass(frozen=True)
class ReportRequest:
    """报告生成请求：模板、格式、关联对象与章节。"""

    name: str
    template: str
    fmt: str = "pdf"
    experiment_id: str | None = None
    project_id: str | None = None
    sections: list[str] | None = None
    params: dict | None = None
    created_by: str | None = None


def create_report(request: ReportRequest) -> dict:
    name, template, fmt = request.name, request.template, request.fmt
    experiment_id, project_id = request.experiment_id, request.project_id
    sections, params = request.sections, request.params
    created_by = request.created_by
    if template not in TEMPLATES:
        raise ValidationError(f"未知报告模板: {template}，可选 {list(TEMPLATES)}")
    fmt = str(fmt).lower()
    if fmt not in ("pdf", "md", "csv", "json"):
        raise ValidationError(f"不支持的报告格式: {fmt}")

    payload = collect(template, experiment_id, project_id, params or {}, sections)
    report_id = db.insert("report", {
        "experiment_id": experiment_id, "project_id": project_id,
        "name": name or f"{TEMPLATES[template]['label']} {db.now_iso()}",
        "template": template, "sections": sections or TEMPLATES[template]["sections"],
        "params": params or {}, "format": fmt, "status": "generating",
        "share_token": secrets.token_urlsafe(12), "created_by": created_by,
    }, id_prefix="rep")

    out_dir = settings.storage_dir("reports") / report_id
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = storage.safe_name(name or template, template)
    path = out_dir / f"{safe}.{fmt}"

    try:
        if fmt == "pdf":
            _render_pdf(path, payload)
        elif fmt == "md":
            path.write_text(render_markdown(payload), encoding="utf-8")
        elif fmt == "csv":
            _render_csv(path, payload)
        else:
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                            encoding="utf-8")
        meta = path.stat()
        db.update("report", report_id, {
            "status": "done", "file_uri": storage.to_uri(path), "file_size": meta.st_size,
        })
    except Exception as exc:  # noqa: BLE001
        log.error("报告生成失败: %s", exc)
        db.update("report", report_id, {"status": "failed"})
        raise
    log.info("报告已生成: %s", path)
    return get_report(report_id) or {}


def get_report(report_id: str) -> dict | None:
    row = db.query_one("SELECT * FROM report WHERE id=?", (report_id,))
    if row is None:
        return None
    report = db.decode_row(row) or {}
    report["file_human"] = storage.human_size(report.get("file_size") or 0)
    report["exists"] = bool(report.get("file_uri")
                            and Path(storage.to_path(report["file_uri"]) or "").exists())
    return report


def get_by_token(token: str) -> dict | None:
    return db.decode_row(db.query_one("SELECT * FROM report WHERE share_token=?", (token,)))


def list_reports(page: int = 1, page_size: int = 20, template: str | None = None) -> dict:
    clauses, params = ["1=1"], []
    if template:
        clauses.append("template=?")
        params.append(template)
    where = " AND ".join(clauses)
    result = db.paginate(
        f"SELECT * FROM report WHERE {where} ORDER BY created_at DESC",
        f"SELECT COUNT(*) FROM report WHERE {where}", params, page, page_size)
    for item in result["items"]:
        item["file_human"] = storage.human_size(item.get("file_size") or 0)
    return result


def delete_report(report_id: str) -> int:
    report = get_report(report_id)
    if report and report.get("file_uri"):
        path = storage.to_path(report["file_uri"])
        if path and Path(path).exists():
            try:
                Path(path).unlink()
            except OSError:
                pass
    from core import db as dbmod

    with dbmod.transaction() as conn:
        cur = conn.execute("DELETE FROM report WHERE id=?", (report_id,))
    return cur.rowcount


# ================================================================== 数据收集
def collect(template: str, experiment_id: str | None, project_id: str | None,
             params: dict, sections: list[str] | None) -> dict:
    from services import analysis_service, dataset_service, project_service

    payload: dict[str, Any] = {
        "meta": {
            "title": TEMPLATES[template]["label"],
            "platform": settings.get("app.name"),
            "version": settings.get("app.version"),
            "generated_at": datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S"),
            "template": template,
            "sections": sections or TEMPLATES[template]["sections"],
        },
        "overview": {},
        "kpi": analysis_service.dashboard_kpi(),
    }

    if experiment_id:
        exp = project_service.get_experiment(experiment_id)
        if exp is None:
            raise NotFoundError("实验不存在。")
        payload["experiment"] = exp
        payload["overview"] = {
            "实验名称": exp["name"], "任务类型": exp["task_type"],
            "实验类型": exp["kind"], "状态": exp["status"],
            "创建时间": exp.get("created_at"),
            "完成时间": exp.get("finished_at"),
            "模型": (exp.get("model") or {}).get("name"),
            "数据集": (exp.get("dataset_version") or {}).get("version"),
            "记录数": exp.get("counts"),
        }
        filters = {"experiment_id": experiment_id}
        payload["metrics"] = analysis_service.summary(filters)
        payload["ranging"] = analysis_service.query_records(filters, page=1, page_size=200)["items"]
        payload["manifest"] = exp.get("run_manifest") or {}

    if project_id and not experiment_id:
        project = project_service.get_project(project_id)
        if project:
            payload["project"] = project
            payload["overview"] = {"项目名称": project["name"], "场景": project.get("scene"),
                                   "应用目标": project.get("app_target"),
                                   "统计": project.get("stats")}
            payload["metrics"] = analysis_service.summary({})

    if template == "dataset" and params.get("version_id"):
        payload["dataset_stats"] = dataset_service.version_stats(params["version_id"])
        version = db.query_one("SELECT * FROM dataset_version WHERE id=?", (params["version_id"],))
        payload["validation"] = db.decode_row(version).get("validation_report") if version else {}

    if template == "compare" and params.get("experiment_ids"):
        payload["compare"] = project_service.compare_experiments(params["experiment_ids"])

    payload.setdefault("metrics", analysis_service.summary({}))
    payload["leaderboard"] = analysis_service.experiment_leaderboard(params.get("task_type"))
    return payload


# ================================================================== 渲染
def render_markdown(payload: dict) -> str:
    meta = payload.get("meta", {})
    lines = [f"# {meta.get('platform', '实验平台')} · {meta.get('title', '报告')}", ""]
    lines.append(f"- 生成时间：{meta.get('generated_at')}")
    lines.append(f"- 平台版本：{meta.get('version')}")
    lines.append("")

    overview = payload.get("overview") or {}
    if overview:
        lines += ["## 一、基本信息", "", "| 项 | 值 |", "| --- | --- |"]
        for key, value in overview.items():
            lines.append(f"| {key} | {value} |")
        lines.append("")

    kpi = payload.get("kpi") or {}
    if kpi:
        lines += ["## 二、总体指标", "", "| 指标 | 数值 |", "| --- | --- |",
                  f"| 项目数 | {kpi.get('projects')} |",
                  f"| 实验数 | {kpi.get('experiments')} |",
                  f"| 模型数 | {kpi.get('models')} |",
                  f"| 图像样本 | {kpi.get('images')} |",
                  f"| 测距记录 | {kpi.get('ranging_records')} |",
                  f"| 平均绝对误差 (m) | {kpi.get('mean_error_m')} |",
                  f"| 缺检率 | {kpi.get('miss_rate')} |", ""]

    metrics = payload.get("metrics") or {}
    if metrics:
        lines += ["## 三、误差统计", "", "| 指标 | 数值 |", "| --- | --- |"]
        for key in ("sample_count", "valid_count", "mae_m", "rmse_m", "mre",
                    "median_error_m", "max_error_m", "std_error_m", "hit_rate",
                    "miss_count", "miss_rate"):
            if key in metrics:
                lines.append(f"| {key} | {metrics.get(key)} |")
        lines.append("")
        for dim, title in (("by_distance", "按距离"), ("by_category", "按类别"),
                           ("by_size", "按目标尺寸组"), ("by_illumination", "按光照条件"),
                           ("by_model", "按模型")):
            rows = metrics.get(dim) or []
            if not rows:
                continue
            lines += [f"### 3.x {title}分组", "", "| 分组 | 样本数 | 有效数 | MAE(m) | RMSE(m) | 相对误差 | 最大误差(m) |",
                      "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
            for row in rows[:30]:
                lines.append(f"| {row.get('key')} | {row.get('count')} | {row.get('valid_count')} | "
                             f"{row.get('mae_m')} | {row.get('rmse_m')} | {row.get('mre')} | "
                             f"{row.get('max_error_m')} |")
            lines.append("")

    compare = payload.get("compare")
    if compare:
        lines += ["## 四、实验对比", "", "| 实验 | 任务 | 模型 | 指标 |", "| --- | --- | --- | --- |"]
        for item in compare.get("items", []):
            metrics_text = ", ".join(f"{k}={v}" for k, v in (item.get("metrics") or {}).items())
            lines.append(f"| {item.get('name')} | {item.get('task_type')} | "
                         f"{item.get('model_name')} | {metrics_text} |")
        lines.append("")

    board = payload.get("leaderboard") or []
    if board:
        lines += ["## 五、模型排行榜", "", "| 排名 | 模型 | 样本数 | MAE(m) | 相对误差 |",
                  "| ---: | --- | ---: | ---: | ---: |"]
        for i, item in enumerate(board[:15], 1):
            lines.append(f"| {i} | {item.get('key')} | {item.get('count')} | "
                         f"{item.get('mae_m')} | {item.get('mre')} |")
        lines.append("")

    manifest = payload.get("manifest")
    if manifest:
        lines += ["## 六、复现清单", "", "```json",
                  json.dumps(manifest, ensure_ascii=False, indent=2)[:4000], "```", ""]

    lines += ["---", "", "> 本报告由「视界智研」算法实验平台自动生成。"]
    return "\n".join(lines)


def _render_csv(path: Path, payload: dict) -> None:
    records = payload.get("ranging") or []
    buffer = io.StringIO()
    if records:
        fieldnames = ["id", "task_type", "strategy", "target_kind", "metric_kind", "model_name",
                      "category", "size_group", "depth_m", "lateral_m", "valid_points",
                      "distance_actual_m", "error_m", "abs_error_m", "rel_error",
                      "illumination", "distance_plan_m", "record_time"]
        writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in records:
            writer.writerow({k: row.get(k) for k in fieldnames})
    else:
        metrics = payload.get("metrics") or {}
        writer = csv.writer(buffer)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            if not isinstance(value, (list, dict)):
                writer.writerow([key, value])
        for dim in ("by_distance", "by_category", "by_model", "by_size", "by_illumination"):
            for row in metrics.get(dim) or []:
                writer.writerow([f"{dim}.{row.get('key')}", row.get("mae_m")])
    path.write_text(buffer.getvalue(), encoding="utf-8-sig")


def _render_pdf(path: Path, payload: dict) -> None:
    """用 ReportLab 渲染 PDF；中文字体缺失时降级为 Markdown 输出。"""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.platypus import (PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                        TableStyle)
    except ImportError:
        path.with_suffix(".md").write_text(render_markdown(payload), encoding="utf-8")
        return

    font_name = "Helvetica"
    for candidate in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf",
                      "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"):
        if Path(candidate).exists():
            try:
                pdfmetrics.registerFont(TTFont("CJK", candidate))
                font_name = "CJK"
                break
            except Exception:  # noqa: BLE001
                continue

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("VLTitle", parent=styles["Title"], fontName=font_name, fontSize=20,
                                 leading=26)
    h_style = ParagraphStyle("VLH", parent=styles["Heading2"], fontName=font_name, fontSize=14,
                             leading=20, spaceBefore=10, spaceAfter=6)
    body = ParagraphStyle("VLBody", parent=styles["BodyText"], fontName=font_name, fontSize=10,
                          leading=16)
    small = ParagraphStyle("VLSmall", parent=body, fontSize=8.5, leading=12,
                           textColor=colors.HexColor("#555555"))

    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm,
                            title=payload.get("meta", {}).get("title", "报告"))
    flow: list[Any] = []
    meta = payload.get("meta", {})
    flow.append(Paragraph(str(meta.get("platform", "")), title_style))
    flow.append(Paragraph(str(meta.get("title", "报告")), h_style))
    flow.append(Paragraph(f"生成时间：{meta.get('generated_at')}　版本：{meta.get('version')}", small))
    flow.append(Spacer(1, 8))

    def add_table(rows: list[list[Any]], header: bool = True) -> None:
        if not rows:
            return
        table = Table(rows, hAlign="LEFT")
        style = [("FONTNAME", (0, 0), (-1, -1), font_name),
                 ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                 ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")),
                 ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                 ("LEFTPADDING", (0, 0), (-1, -1), 4),
                 ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
        if header:
            style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f7")))
            style.append(("FONTNAME", (0, 0), (-1, 0), font_name))
        table.setStyle(TableStyle(style))
        flow.append(table)
        flow.append(Spacer(1, 8))

    overview = payload.get("overview") or {}
    if overview:
        flow.append(Paragraph("一、基本信息", h_style))
        add_table([[Paragraph("项", body), Paragraph("值", body)]] +
                  [[Paragraph(str(k), body), Paragraph(str(v), body)] for k, v in overview.items()])

    kpi = payload.get("kpi") or {}
    if kpi:
        flow.append(Paragraph("二、总体指标", h_style))
        kpi_rows = [[Paragraph("指标", body), Paragraph("数值", body)]]
        for key in ("projects", "experiments", "models", "images",
                    "ranging_records", "mean_error_m", "miss_rate"):
            kpi_rows.append([Paragraph(key, body), Paragraph(str(kpi.get(key)), body)])
        add_table(kpi_rows)

    metrics = payload.get("metrics") or {}
    if metrics:
        flow.append(Paragraph("三、误差统计", h_style))
        rows = [[Paragraph("指标", body), Paragraph("数值", body)]]
        for key in ("sample_count", "valid_count", "mae_m", "rmse_m", "mre", "median_error_m",
                    "max_error_m", "std_error_m", "hit_rate", "miss_count", "miss_rate"):
            if key in metrics:
                rows.append([Paragraph(key, body), Paragraph(str(metrics.get(key)), body)])
        add_table(rows)

        for dim, title in (("by_distance", "按距离"), ("by_category", "按类别"),
                           ("by_size", "按目标尺寸组"), ("by_illumination", "按光照条件"),
                           ("by_model", "按模型")):
            data = metrics.get(dim) or []
            if not data:
                continue
            flow.append(Paragraph(f"3.x {title}分组", body))
            headers = ("分组", "样本数", "有效数", "MAE(m)", "RMSE(m)", "相对误差", "最大误差(m)")
            table_rows = [[Paragraph(title, body) for title in headers]]
            for row in data[:25]:
                table_rows.append([Paragraph(str(row.get("key")), body),
                                   Paragraph(str(row.get("count")), body),
                                   Paragraph(str(row.get("valid_count")), body),
                                   Paragraph(str(row.get("mae_m")), body),
                                   Paragraph(str(row.get("rmse_m")), body),
                                   Paragraph(str(row.get("mre")), body),
                                   Paragraph(str(row.get("max_error_m")), body)])
            add_table(table_rows)
            if len(flow) > 40:
                flow.append(PageBreak())

    board = payload.get("leaderboard") or []
    if board:
        flow.append(Paragraph("四、模型排行榜", h_style))
        table_rows = [[Paragraph(x, body) for x in ("排名", "模型", "样本数", "MAE(m)", "相对误差")]]
        for i, item in enumerate(board[:15], 1):
            table_rows.append([Paragraph(str(i), body), Paragraph(str(item.get("key")), body),
                               Paragraph(str(item.get("count")), body),
                               Paragraph(str(item.get("mae_m")), body),
                               Paragraph(str(item.get("mre")), body)])
        add_table(table_rows)

    manifest = payload.get("manifest")
    if manifest:
        flow.append(Paragraph("五、复现清单（节选）", h_style))
        text = json.dumps(manifest, ensure_ascii=False, indent=2)[:2200].replace("\n", "<br/>")
        flow.append(Paragraph(text, small))

    flow.append(Spacer(1, 12))
    flow.append(Paragraph("本报告由「视界智研 —— 基于双目视觉测距与目标检测算法实验平台」自动生成。", small))
    doc.build(flow)
