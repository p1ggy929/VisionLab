"""API 离线冒烟测试：用 Flask 测试客户端跑通全部接口，无需启动 HTTP 服务。

用法：python tools/check_api.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app import bootstrap_data, create_app  # noqa: E402
from core.logging_setup import get_cli_logger  # noqa: E402

log = get_cli_logger("tools.check_api")

FAILURES: list[str] = []


def show(tag: str, resp, keys=None, expect: int = 200) -> dict:
    status = resp.status_code
    try:
        data = resp.get_json()
    except Exception:  # noqa: BLE001
        data = None
    ok = status == expect
    if not ok:
        FAILURES.append(tag)
    log.info(f"[{'OK ' if ok else 'ERR'}] {tag:<46} HTTP {status}")
    if not ok and data:
        log.info("      ", json.dumps(data, ensure_ascii=False)[:300])
    if ok and data and keys:
        payload = data.get("data")
        for key in keys:
            source = payload if isinstance(payload, dict) else (data if isinstance(data, dict) else {})
            log.info(f"        {key} = {json.dumps(source.get(key), ensure_ascii=False)[:120]}")
    return data or {}


def main() -> int:
    app = create_app()
    bootstrap_data()
    client = app.test_client()

    log.info("=" * 72)
    log.info("1) 认证")
    data = show("POST /auth/login",
                client.post("/api/v1/auth/login",
                            json={"username": "admin", "password": "admin123"}))
    token = (data.get("data") or {}).get("token")
    if not token:
        log.info("登录失败，终止")
        return 1
    headers = {"Authorization": f"Bearer {token}"}
    show("GET  /auth/me", client.get("/api/v1/auth/me", headers=headers))
    show("GET  /system/info", client.get("/api/v1/system/info", headers=headers))

    log.info("=" * 72)
    log.info("2) 系统与相机")
    show("POST /system/self-check", client.post("/api/v1/system/self-check", headers=headers))
    show("GET  /camera/status", client.get("/api/v1/camera/status", headers=headers),
         ["backend", "running"])
    show("POST /camera/stream/start",
         client.post("/api/v1/camera/stream/start", json={"backend": "mock"}, headers=headers),
         ["backend", "width", "height"])
    show("GET  /camera/frame", client.get("/api/v1/camera/frame", headers=headers))
    show("GET  /camera/depth-probe", client.get("/api/v1/camera/depth-probe", headers=headers))
    show("POST /camera/calibrate", client.post("/api/v1/camera/calibrate", json={}, headers=headers))

    log.info("=" * 72)
    log.info("3) 双策略同跑流水线")
    show("GET  /pipeline/config", client.get("/api/v1/pipeline/config", headers=headers))
    show("POST /pipeline/config",
         client.post("/api/v1/pipeline/config",
                     json={"mode": "both", "detector_key": "sim_ground_truth",
                           "segmenter_key": "classical_road"}, headers=headers))
    result = show("POST /pipeline/process",
                  client.post("/api/v1/pipeline/process", headers=headers))
    payload = result.get("data") or {}
    detection = payload.get("detection") or {}
    road = payload.get("road") or {}
    log.info(f"        检出 {len(detection.get('detections') or [])} 个目标；"
          f"走廊宽度 {road.get('corridor_width_m')} m；"
          f"叠加图 {len(payload.get('overlay_base64') or '')} 字节(base64)")

    log.info("=" * 72)
    log.info("4) 实验会话与测距记录")
    session = show("POST /pipeline/session/start",
                   client.post("/api/v1/pipeline/session/start",
                               json={"name": "离线冒烟实验", "task_type": "joint"}, headers=headers))
    experiment_id = (session.get("data") or {}).get("experiment_id")
    truth = {item["category"]: item["world_z"] for item in (payload.get("ground_truth") or [])}
    bulk = show("POST /ranging/records/bulk",
                client.post("/api/v1/ranging/records/bulk",
                            json={"illumination": "sunny", "actual": truth}, headers=headers))
    log.info("        落库记录数 =", (bulk.get("data") or {}).get("created"))
    show("POST /pipeline/save-frame", client.post("/api/v1/pipeline/save-frame", headers=headers))
    show("POST /pipeline/session/stop", client.post("/api/v1/pipeline/session/stop", headers=headers))

    log.info("=" * 72)
    log.info("5) 误差分析与校正")
    stats = show("GET  /analysis/summary", client.get("/api/v1/analysis/summary", headers=headers))
    summary = stats.get("data") or {}
    log.info(f"        样本 {summary.get('sample_count')} 有效 {summary.get('valid_count')} "
          f"MAE={summary.get('mae_m')} RMSE={summary.get('rmse_m')}")
    show("GET  /analysis/distributions", client.get("/api/v1/analysis/distributions", headers=headers))
    show("GET  /analysis/heatmap", client.get("/api/v1/analysis/heatmap", headers=headers))
    show("GET  /analysis/leaderboard", client.get("/api/v1/analysis/leaderboard", headers=headers))
    show("POST /analysis/calibrate",
         client.post("/api/v1/analysis/calibrate", json={"method": "piecewise"}, headers=headers))

    log.info("=" * 72)
    log.info("6) 大屏与报告")
    show("GET  /dashboard/overview", client.get("/api/v1/dashboard/overview", headers=headers))
    report = show("POST /reports",
                  client.post("/api/v1/reports",
                              json={"name": "离线冒烟报告", "template": "error", "format": "md",
                                    "experiment_id": experiment_id}, headers=headers))
    report_id = (report.get("data") or {}).get("id")
    show("GET  /reports/<id>", client.get(f"/api/v1/reports/{report_id}", headers=headers))
    show("GET  /reports/<id>/download",
         client.get(f"/api/v1/reports/{report_id}/download", headers=headers))

    log.info("=" * 72)
    log.info("7) 模型与数据集")
    show("GET  /models", client.get("/api/v1/models", headers=headers))
    show("POST /models/scan-artifacts",
         client.post("/api/v1/models/scan-artifacts", json={}, headers=headers))
    show("GET  /datasets", client.get("/api/v1/datasets", headers=headers))
    show("GET  /system/dict", client.get("/api/v1/system/dict", headers=headers))

    log.info("=" * 72)
    log.info("8) 页面路由")
    for path in ("/", "/projects", "/experiments", "/experiments/ranging", "/camera/capture",
                 "/camera/calibration", "/datasets", "/annotations", "/models", "/training",
                 "/experiments/detection", "/experiments/segmentation", "/analysis",
                 "/bigscreen", "/reports", "/settings", "/monitor"):
        response = client.get(path, headers=headers)
        ok = response.status_code == 200
        if not ok:
            FAILURES.append("GET " + path)
        log.info(f"[{'OK ' if ok else 'ERR'}] GET  {path:<42} HTTP {response.status_code}")

    log.info("=" * 72)
    if FAILURES:
        log.info(f"API 冒烟测试完成：{len(FAILURES)} 项未通过")
        for name in FAILURES:
            log.info("   - " + name)
        return 1
    log.info("API 冒烟测试全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
