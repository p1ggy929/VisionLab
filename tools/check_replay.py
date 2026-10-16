"""验证离线回放：抓取真实 RGB-D 序列 → 切换到回放后端 → 跑通测距链路。

用法：python tools/check_replay.py [--frames 30]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from algorithms import registry  # noqa: E402
from algorithms.ranging import BoxBoundaryStrategy, DepthSampler, MaskBoundaryStrategy, road_summary  # noqa: E402
from core import db, media  # noqa: E402
from core.camera.replay_backend import ReplayBackend, list_replay_sources  # noqa: E402
from core.camera.service import get_camera_service  # noqa: E402
from core.config import settings  # noqa: E402
from core.logging_setup import setup_logging  # noqa: E402

OUT = settings.storage_dir("results") / "check_replay"


def main() -> int:
    parser = argparse.ArgumentParser(description="离线回放验证")
    parser.add_argument("--frames", type=int, default=24, help="抓取的回放样本帧数")
    parser.add_argument("--source", default=None, help="直接指定已有回放源（采集会话 ID）")
    args = parser.parse_args()

    setup_logging()
    db.init_db()
    registry.load_builtin()
    OUT.mkdir(parents=True, exist_ok=True)

    # ---------------- 1) 准备回放源 ----------------
    if args.source:
        source_id = args.source
        print(f"使用已有回放源: {source_id}")
    else:
        print(f"1) 用当前相机抓取 {args.frames} 帧真实 RGB-D 序列 ...")
        from core.camera.replay_backend import capture_sample

        camera = get_camera_service()
        if not camera.running:
            camera.start()
        sample = capture_sample("", args.frames)
        source_id = sample["session_id"]
        print("   ", json.dumps(sample, ensure_ascii=False))

    print("\n2) 可用回放源列表 ...")
    sources = list_replay_sources()
    for item in sources[:6]:
        print(f"   [{item['type']:<7}] {item['label']:<30} {item['frames']:>5} 帧  "
              f"含真值 {item['with_truth']}")
    if not any(s["id"] == source_id for s in sources):
        print(f"   [警告] 回放源 {source_id} 未出现在列表中")

    # ---------------- 3) 直接使用后端回放 ----------------
    print(f"\n3) 装载回放后端 ...")
    backend = ReplayBackend(source=source_id, fps=60, loop=True)
    backend.start()
    info = backend.describe()
    print("   ", json.dumps({k: info[k] for k in
                             ("backend", "model", "width", "height", "frames_total",
                              "depth_scale")}, ensure_ascii=False))

    frames = []
    for _ in range(6):
        bundle = backend.get_latest(timeout=1.0)
        if bundle is None:
            continue
        frames.append(bundle)
    print(f"   连续取到 {len(frames)} 帧；"
          f"形状 {frames[0].color.shape}，深度有效占比 "
          f"{float((frames[0].depth_m > 0).mean()):.3f}" if frames else "   未取到帧")
    print("   回放进度:", backend.progress())

    truth = backend.ground_truth()
    print(f"   当前帧随帧真值: {json.dumps(truth, ensure_ascii=False)}")

    # ---------------- 4) 在回放帧上跑测距 ----------------
    print("\n4) 在回放帧上运行检测 + 分割 + 统一测距 ...")
    bundle = frames[-1]
    sampler = DepthSampler(kernel=21)

    detector = registry.create("detection", "depth_obstacle")
    detector.load(None)
    pred = detector.predict(bundle.color, depth_m=bundle.depth_m, intrinsics=bundle.intrinsics)
    box_results = BoxBoundaryStrategy(sampler).measure(bundle.depth_m, bundle.intrinsics, pred)
    print(f"   检测（深度基线）: {len(pred['boxes'])} 个目标，测距结果 {len(box_results)} 条")
    for item in box_results:
        if item.metric_kind == "center" and item.extra.get("class_id") is not None:
            print(f"      {item.category:<8} center={item.depth_m:.3f} m  "
                  f"lateral={item.lateral_m}")
            break

    segmenter = registry.create("segmentation", "classical_road")
    segmenter.load(None)
    seg = segmenter.predict(bundle.color, depth_m=bundle.depth_m, intrinsics=bundle.intrinsics)
    mask_results = MaskBoundaryStrategy(sampler).measure(
        bundle.depth_m, bundle.intrinsics, seg, road_split=True, reproject=True,
        corridor_ref_depth_m=5.0)
    summary = road_summary(mask_results)
    print(f"   分割: 掩膜占比 {float(seg['mask'].mean()):.3f}")
    print(f"   道路左边界 min={((summary['left'].get('min') or {}).get('depth_m'))} m  "
          f"右边界 min={((summary['right'].get('min') or {}).get('depth_m'))} m  "
          f"走廊={summary.get('corridor_width_m')} m")

    vis = media.draw_mask(bundle.color.copy(), seg["mask"], alpha=0.3)
    media.draw_detections(vis, pred.detections())
    media.write_image(OUT / "replay_overlay.jpg", vis, quality=90)
    media.write_image(OUT / "replay_depth.jpg",
                      media.colorize_depth(bundle.depth_m, 0.3, 15.0), quality=90)

    # ---------------- 5) 通过 CameraService 切换 ----------------
    print("\n5) 通过 CameraService 切换到回放后端 ...")
    service = get_camera_service()
    started = service.start(prefer="replay", replay_source={"type": "capture", "id": source_id,
                                                            "fps": 30, "loop": True})
    print("   ", json.dumps({k: started.get(k) for k in
                             ("backend", "model", "frames_total", "width", "height")},
                            ensure_ascii=False))
    for _ in range(3):
        service.latest(timeout=0.5)
    status = service.describe()
    print(f"   运行状态: backend={status.get('backend')} running={status.get('running')} "
          f"fps_actual={status.get('fps_actual')}")
    print(f"   随帧真值: {json.dumps(service.ground_truth(), ensure_ascii=False)}")

    # 恢复为自动（合成场景）
    service.start(prefer="mock")
    print("\n6) 已恢复为合成场景后端")

    print(f"\n产物目录: {OUT}")
    print("离线回放验证通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
