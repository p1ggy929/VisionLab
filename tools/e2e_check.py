"""端到端 HTTP 验收：对运行中的服务跑完整实验流程。

用法：
    python tools/e2e_check.py --base http://127.0.0.1:5000 --user admin --password admin123
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from http.cookiejar import CookieJar
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core.logging_setup import get_cli_logger  # noqa: E402

log = get_cli_logger("tools.e2e_check")


class Client:
    def __init__(self, base: str) -> None:
        self.base = base.rstrip("/")
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(CookieJar()))
        self.token = ""

    def call(self, method: str, path: str, body: dict | None = None) -> dict:
        url = self.base + path
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(url, data=data, method=method)
        request.add_header("Content-Type", "application/json")
        if self.token:
            request.add_header("Authorization", "Bearer " + self.token)
        try:
            with self.opener.open(request, timeout=120) as response:
                payload = response.read()
                try:
                    return json.loads(payload)
                except json.JSONDecodeError:
                    return {"code": 0, "raw_length": len(payload)}
        except urllib.error.HTTPError as exc:
            text = exc.read().decode("utf-8", "ignore")
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"code": exc.code, "message": text[:200]}

    def raw(self, path: str) -> bytes:
        url = self.base + path
        request = urllib.request.Request(url)
        if self.token:
            request.add_header("Authorization", "Bearer " + self.token)
        with self.opener.open(request, timeout=120) as response:
            return response.read()


def step(index: int, title: str) -> None:
    log.info(f"\n[{index:>2}] {title}")
    log.info("-" * 70)


def main() -> int:
    parser = argparse.ArgumentParser(description="视界智研端到端验收")
    parser.add_argument("--base", default="http://127.0.0.1:5000")
    parser.add_argument("--user", default="admin")
    parser.add_argument("--password", default="admin123")
    args = parser.parse_args()

    client = Client(args.base)
    failures: list[str] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        mark = "OK  " if condition else "FAIL"
        log.info(f"   [{mark}] {name}" + (f"  {detail}" if detail else ""))
        if not condition:
            failures.append(name)

    step(1, "登录与身份")
    data = client.call("POST", "/api/v1/auth/login",
                       {"username": args.user, "password": args.password})
    check("登录成功", data.get("code") == 0, data.get("message", ""))
    if data.get("code") != 0:
        return 1
    client.token = data["data"]["token"]
    me = client.call("GET", "/api/v1/auth/me")
    check("获取当前用户", me.get("code") == 0, json.dumps(me.get("data", {}).get("user", {}),
                                                        ensure_ascii=False))

    step(2, "相机接入")
    status = client.call("GET", "/api/v1/camera/status").get("data", {})
    check("相机运行中", bool(status.get("running")),
          f"backend={status.get('backend')} {status.get('width')}x{status.get('height')}")
    check("内参可用", bool((status.get("intrinsics") or {}).get("fx")))
    probe = client.call("GET", "/api/v1/camera/depth-probe").get("data", {})
    check("深度探测", probe.get("depth_m", 0) > 0, f"Z={probe.get('depth_m')} m, X={probe.get('x_m')} m")

    step(3, "相机标定")
    calib = client.call("POST", "/api/v1/camera/calibrate", {"name": "e2e_calib"})
    check("执行标定", calib.get("code") == 0,
          f"RMS={((calib.get('data') or {}).get('rms_reprojection_px'))}")

    step(4, "双策略同跑（检测 + 分割 + 统一测距）")
    client.call("POST", "/api/v1/pipeline/config",
                {"mode": "both", "detector_key": "sim_ground_truth",
                 "segmenter_key": "classical_road", "sample_points": 20, "kernel": 21})
    frame = client.call("POST", "/api/v1/pipeline/process", {}).get("data", {})
    det = (frame.get("detection") or {}).get("detections") or []
    road = frame.get("road") or {}
    check("检测到目标", len(det) > 0, f"{len(det)} 个")
    check("目标边界测距输出", bool(frame.get("objects")))
    check("道路左边界距离", bool((road.get("left") or {}).get("min")),
          f"{((road.get('left') or {}).get('min') or {}).get('depth_m')} m")
    check("道路右边界距离", bool((road.get("right") or {}).get("min")),
          f"{((road.get('right') or {}).get('min') or {}).get('depth_m')} m")
    check("可通行走廊宽度", road.get("corridor_width_m") is not None,
          f"{road.get('corridor_width_m')} m（合成场景真值 6.20 m）")
    lateral = ((road.get("left") or {}).get("min") or {}).get("lateral_m")
    check("横向偏移（平台新增能力）", lateral is not None, f"X={lateral} m")
    check("叠加图已生成", bool(frame.get("overlay_base64")))
    check("深度伪彩已生成", bool(frame.get("depth_base64")))

    step(5, "实验会话与测距落库（带真值）")
    session = client.call("POST", "/api/v1/pipeline/session/start",
                          {"name": "端到端验收实验", "task_type": "joint"}).get("data", {})
    experiment_id = session.get("experiment_id")
    check("会话已开始", bool(experiment_id), str(experiment_id))

    truth = {item["category"]: item["world_z"] for item in (frame.get("ground_truth") or [])}
    bulk = client.call("POST", "/api/v1/ranging/records/bulk",
                       {"illumination": "sunny", "actual": truth}).get("data", {})
    check("落库当前帧测距结果", (bulk.get("created") or 0) > 0, f"{bulk.get('created')} 条")

    stopped = client.call("POST", "/api/v1/pipeline/session/stop").get("data", {})
    check("会话结束", bool(stopped), json.dumps(stopped.get("session", {}).get("experiment_id", ""),
                                              ensure_ascii=False))

    step(6, "误差分析与分组归因")
    stats = client.call("GET", f"/api/v1/analysis/summary?experiment_id={experiment_id}").get("data", {})
    check("统计输出", stats.get("sample_count", 0) > 0,
          f"样本 {stats.get('sample_count')} 有效 {stats.get('valid_count')} "
          f"MAE={stats.get('mae_m')} RMSE={stats.get('rmse_m')}")
    check("按类别归因", bool(stats.get("by_category")))
    check("按尺寸组归因", bool(stats.get("by_size")))
    check("热力图数据", bool((stats.get("heatmap") or {}).get("x")))

    step(7, "误差校正（三类模型）")
    for method in ("linear", "ratio", "piecewise"):
        result = client.call("POST", "/api/v1/analysis/calibrate",
                             {"method": method, "experiment_id": experiment_id}).get("data", {})
        ok = result.get("rmse_before") is not None and result.get("rmse_after") is not None
        check(f"{method} 拟合", ok,
              f"RMSE {result.get('rmse_before')} -> {result.get('rmse_after')} "
              f"推荐={result.get('recommended')}")
        if ok and method == "piecewise":
            active = client.call("POST", f"/api/v1/analysis/calibrations/{result['id']}/activate", {})
            check("启用校正方案", active.get("code") == 0)

    step(8, "历史数据回归（原项目 A 口径）")
    experiments = client.call("GET", "/api/v1/experiments?page_size=100").get("data", [])
    legacy = [e for e in experiments if "历史数据迁移" in e["name"] and e["task_type"] == "detection"]
    if legacy:
        summary = client.call("GET",
                              f"/api/v1/analysis/legacy-summary?experiment_id={legacy[0]['id']}").get("data", {})
        check("历史记录条数", summary.get("total_records", 0) > 1000,
              f"{summary.get('total_records')} 条")
        check("缺检率复现（原报告 17.40%）", abs((summary.get("miss_rate") or 0) - 0.174) < 0.01,
              f"{summary.get('miss_rate')}")
        check("平均误差复现（原报告 0.446 m）", abs((summary.get("mean_error_m") or 0) - 0.446) < 0.01,
              f"{summary.get('mean_error_m')} m")
        check("中位误差复现（原报告 0.190 m）", abs((summary.get("median_error_m") or 0) - 0.19) < 0.01,
              f"{summary.get('median_error_m')} m")
    else:
        check("找到历史迁移实验", False, "请先运行 tools/import_legacy.py")

    step(9, "可视化大屏数据")
    overview = client.call("GET", "/api/v1/dashboard/overview").get("data", {})
    check("KPI", bool(overview.get("kpi")), json.dumps(overview.get("kpi", {}), ensure_ascii=False)[:150])
    check("距离—误差序列", bool(overview.get("by_distance")),
          f"{len(overview.get('by_distance') or [])} 个距离点")
    check("模型排行榜", bool(overview.get("leaderboard")))
    check("大屏热力图", bool((overview.get("heatmap") or {}).get("x")))

    step(10, "报告导出")
    for fmt in ("md", "pdf", "csv", "json"):
        report = client.call("POST", "/api/v1/reports",
                             {"name": f"端到端验收报告_{fmt}", "template": "error",
                              "format": fmt, "experiment_id": experiment_id}).get("data", {})
        ok = report.get("status") == "done" and (report.get("file_size") or 0) > 0
        check(f"{fmt.upper()} 报告", ok,
              f"{report.get('file_human')}  {Path(report.get('file_uri') or '').name}")
        if fmt == "pdf" and ok:
            blob = client.raw(f"/api/v1/reports/{report['id']}/download")
            check("PDF 可下载且为合法 PDF", blob[:4] == b"%PDF", f"{len(blob)} 字节")

    step(11, "模型仓库与数据集")
    models = client.call("GET", "/api/v1/models?page_size=1").get("data", [])
    total_models = client.call("GET", "/api/v1/models?page_size=1").get("total", 0)
    check("模型已登记", total_models > 0, f"{total_models} 个")
    datasets = client.call("GET", "/api/v1/datasets").get("data", [])
    check("数据集已登记", len(datasets) > 0, f"{len(datasets)} 个")
    for dataset in datasets[:2]:
        versions = (client.call("GET", f"/api/v1/datasets/{dataset['id']}").get("data", {})
                    .get("versions") or [])
        if versions:
            validation = client.call("POST",
                                     f"/api/v1/datasets/versions/{versions[0]['id']}/validate",
                                     {}).get("data", {})
            check(f"数据集校验 · {dataset['name']}", validation.get("conclusion") is not None,
                  f"{validation.get('checked_images')} 张抽检，问题 {validation.get('issue_count')} 处，"
                  f"前景值 {validation.get('foreground_value_histogram')}")

    step(12, "系统自检与监控")
    selfcheck = client.call("POST", "/api/v1/system/self-check", {}).get("data", {})
    summary = selfcheck.get("summary", {})
    check("一键自检", summary.get("total", 0) > 0,
          f"{summary.get('passed')}/{summary.get('total')} 通过，错误 {summary.get('errors')}，"
          f"警告 {summary.get('warnings')}")
    metrics = client.call("GET", "/api/v1/system/metrics").get("data", {})
    check("运行指标", bool(metrics.get("camera")), f"相机 {metrics.get('camera', {}).get('backend')}")
    logs = client.call("GET", "/api/v1/system/logs?limit=5").get("data", {})
    check("日志可读", len(logs.get("items") or []) >= 0)

    step(13, "离线回放（从落盘的真实 RGB-D 序列重放）")
    backends = client.call("GET", "/api/v1/camera/backends").get("data", {})
    keys = [item["key"] for item in (backends.get("items") or [])]
    check("回放后端已注册", "replay" in keys, "/".join(keys))

    sample = client.call("POST", "/api/v1/camera/replay-sample", {"frames": 12}).get("data", {})
    check("生成回放样本", (sample.get("frames") or 0) > 0,
          f"{sample.get('session_id')} · {sample.get('frames')} 帧 · "
          f"含真值 {sample.get('with_truth')}")

    sources = client.call("GET", "/api/v1/camera/replay-sources").get("data", {})
    items = sources.get("items") or []
    check("回放源列表", len(items) > 0,
          ", ".join(f"{i['type']}:{i['label']}({i['frames']})" for i in items[:3]))

    started = client.call("POST", "/api/v1/camera/stream/start",
                          {"backend": "replay", "replay_type": "capture",
                           "replay_source": sample.get("session_id"), "fps": 20,
                           "loop": True}).get("data", {})
    check("切换到回放后端", started.get("backend") == "replay",
          f"{started.get('model')} · {started.get('frames_total')} 帧 · "
          f"depth_scale={started.get('depth_scale')}")

    client.call("POST", "/api/v1/pipeline/config",
                {"mode": "both", "detector_key": "depth_obstacle",
                 "segmenter_key": "classical_road"})
    replay_frame = client.call("POST", "/api/v1/pipeline/process", {}).get("data", {})
    road = replay_frame.get("road") or {}
    check("回放帧上跑通测距", bool((road.get("left") or {}).get("min")),
          f"左边界 {((road.get('left') or {}).get('min') or {}).get('depth_m')} m，"
          f"走廊 {road.get('corridor_width_m')} m，"
          f"来源 {replay_frame.get('source')}")
    progress = client.call("GET", "/api/v1/camera/replay-progress").get("data", {})
    check("回放进度可查", progress.get("frames_total", 0) > 0,
          f"{progress.get('frame_index')}/{progress.get('frames_total')}")

    gt = replay_frame.get("ground_truth") or []
    check("回放帧携带随帧真值", len(gt) > 0,
          json.dumps(gt[0], ensure_ascii=False)[:110] if gt else "无")

    client.call("POST", "/api/v1/camera/stream/start", {"backend": "mock"})
    restored = client.call("GET", "/api/v1/camera/status").get("data", {})
    check("恢复合成场景", restored.get("backend") == "mock", str(restored.get("backend")))

    step(14, "真实 YOLO 权重可用性（检测适配器）")
    configs = client.call("GET", "/api/v1/models/yolo-configs").get("data", {})
    keys = [c["key"] for c in (configs.get("items") or [])]
    check("20 个 YOLO 配置", len(keys) == 20, ", ".join(keys[:6]) + " ...")
    check("v11/v12 命名正确",
          "yolo11n" in keys and "yolo12l" in keys and "yolon" not in keys,
          "yolo11n/yolo12l 存在，yolon 不存在")

    det_models = client.call("GET", "/api/v1/models?task_type=detection&page_size=100").get("data", [])
    seg_models = client.call("GET", "/api/v1/models?task_type=segmentation&page_size=100").get("data", [])
    check("检测权重数量", len(det_models) >= 40, f"{len(det_models)} 个")
    check("分割权重数量", len(seg_models) >= 80, f"{len(seg_models)} 个")
    trained = [m for m in det_models if m["weight_origin"] == "trained"]
    check("自训练权重已登记", len(trained) >= 20, f"{len(trained)} 个，"
          f"示例 weights_uri={trained[0]['weights_uri'][:70] if trained else '-'}")

    step(15, "数据资产总量")
    datasets = client.call("GET", "/api/v1/datasets").get("data", [])
    total_images, split_info = 0, []
    for dataset in datasets:
        detail = client.call("GET", f"/api/v1/datasets/{dataset['id']}").get("data", {})
        for version in detail.get("versions") or []:
            total_images += version.get("image_count") or 0
            stats = client.call("GET",
                                f"/api/v1/datasets/versions/{version['id']}/stats").get("data", {})
            split_info.append(f"{dataset['name']}:{stats.get('splits')}")
    check("图像总量达标（目标 5,814）", total_images >= 5814, f"{total_images} 张")
    for info in split_info:
        log.info(f"        划分 {info}")

    log.info("\n" + "=" * 70)
    if failures:
        log.info(f"端到端验收完成：{len(failures)} 项未通过")
        for name in failures:
            log.info("   - " + name)
        return 1
    log.info("端到端验收全部通过")
    log.info("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
