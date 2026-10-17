"""通过 HTTP 接口用真实 U-Net 权重跑一次分割，验证 PyTorch 插件在服务内可用。

用法：python tools/check_real_seg_http.py --base http://127.0.0.1:5000
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

log = get_cli_logger("tools.check_real_seg_http")

ARCH_WEIGHTS = {
    "unet": "Unet_1000_best.pth",
    "resunet": "resunet_1000_best.pth",
    "att_resunet": "Atteresunet_1000_best.pth",
    "unetv2": "UNetV2_800_best.pth",
}


class Client:
    def __init__(self, base: str) -> None:
        self.base = base.rstrip("/")
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CookieJar()))
        self.token = ""

    def call(self, method: str, path: str, body: dict | None = None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(self.base + path, data=data, method=method)
        request.add_header("Content-Type", "application/json")
        if self.token:
            request.add_header("Authorization", "Bearer " + self.token)
        try:
            with self.opener.open(request, timeout=300) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            text = exc.read().decode("utf-8", "ignore")
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"code": exc.code, "message": text[:200]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:5000")
    parser.add_argument("--arch", default=None, help="只测某个结构，默认全测")
    args = parser.parse_args()

    client = Client(args.base)
    login = client.call("POST", "/api/v1/auth/login",
                        {"username": "admin", "password": "admin123"})
    if login.get("code") != 0:
        log.info("登录失败:", login.get("message"))
        return 1
    client.token = login["data"]["token"]

    found: dict[str, dict] = {}
    for page in (1, 2, 3):
        listing = client.call("GET", f"/api/v1/models?task_type=segmentation&page_size=100&page={page}")
        for model in listing.get("data") or []:
            if model["variant"] == "1000" or model["name"].endswith("_best"):
                found.setdefault(model["arch"], model)
            found.setdefault(model["arch"], model)
    log.info("已登记的分割结构:", sorted(set(found)))

    targets = [args.arch] if args.arch else list(ARCH_WEIGHTS)
    failures = 0
    for arch in targets:
        model = found.get(arch)
        if model is None:
            log.info(f"[SKIP] {arch}: 模型仓库中没有该结构")
            continue
        config = client.call("POST", "/api/v1/pipeline/config",
                             {"mode": "segmentation", "segmenter_key": arch,
                              "segmenter_model_id": model["id"], "threshold": 0.30})
        errors = ((config.get("data") or {}).get("load_errors")) or []
        result = client.call("POST", "/api/v1/pipeline/process", {})
        road = ((result.get("data") or {}).get("road")) or {}
        ok = not errors and road.get("mask_area_ratio") is not None
        if not ok:
            failures += 1
        log.info(f"[{'OK ' if ok else 'FAIL'}] {arch:<12} 权重={model['name']:<22} "
              f"掩膜占比={road.get('mask_area_ratio')}  分割耗时={road.get('infer_ms')} ms  "
              f"左边界={((road.get('left') or {}).get('min') or {}).get('depth_m')} m  "
              f"走廊={road.get('corridor_width_m')} m")
        if errors:
            log.info("        加载错误:", errors)
        if road.get("arch"):
            log.info(f"        实际使用结构: {road.get('arch')}")

    client.call("POST", "/api/v1/pipeline/config",
                {"mode": "both", "segmenter_key": "classical_road", "segmenter_model_id": None})
    log.info("\n" + ("全部结构通过服务端验证" if not failures else f"{failures} 个结构未通过"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
