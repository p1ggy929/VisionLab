"""算法层离线验证：相机 → 检测 → 分割 → 双策略测距 → 误差校正 全链路。

不依赖 Web 服务，可在无浏览器环境下快速验证算法层是否可用。
用法：python tools/check_algorithms.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from algorithms import registry  # noqa: E402
from algorithms.calibration import calibrate, is_recommended  # noqa: E402
from algorithms.ranging import (BoxBoundaryStrategy, DepthSampler, MaskBoundaryStrategy,  # noqa: E402
                                road_summary)
from algorithms.ranging.box_boundary import summarize_per_detection  # noqa: E402
from core import db, media  # noqa: E402
from core.camera.mock_backend import MockBackend  # noqa: E402
from core.config import settings  # noqa: E402
from core.geometry import depth_stats  # noqa: E402

OUT = settings.storage_dir("results") / "check_algorithms"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> int:
    db.init_db()
    registry.load_builtin()

    print("== 已注册插件 ==")
    for item in registry.list_plugins():
        print(f"    {item['task']:<13} {item['key']}")

    camera = MockBackend(width=1280, height=720, moving=False, drop_rate=0.0)
    camera.start()
    frame = camera.get_latest()
    truth = camera.ground_truth()
    assert frame is not None

    print(f"\n== 合成 RGB-D 场景 == shape={frame.color.shape} depth_scale={frame.depth_scale}")
    print("   深度统计:", {k: round(v, 3) for k, v in depth_stats(frame.depth_m).items()})
    print("   仿真真值:", [(t["category"], t["world_z"]) for t in truth])
    media.write_image(OUT / "scene_rgb.jpg", frame.color)
    media.write_depth16(OUT / "scene_depth.png", frame.depth_m, frame.depth_scale)
    media.write_image(OUT / "scene_depth_vis.jpg", media.colorize_depth(frame.depth_m))

    sampler = DepthSampler(kernel=21, mode="median")
    box_strategy = BoxBoundaryStrategy(sampler)
    rows = []

    for key in ("sim_ground_truth", "depth_obstacle"):
        plugin = registry.create("detection", key)
        plugin.load(None)
        params = {"depth_m": frame.depth_m, "intrinsics": frame.intrinsics, "conf": 0.30}
        if key == "sim_ground_truth":
            params.update({"ground_truth": truth, "miss_rate": 0.0})
        pred = plugin.predict(frame.color, **params)
        results = box_strategy.measure(frame.depth_m, frame.intrinsics, pred)
        compact = summarize_per_detection(results)
        print(f"\n== 检测[{key}] == {len(pred['boxes'])} 个目标, {pred['infer_ms']:.1f} ms")
        for item in compact:
            ranging = item["ranging"]
            print("    {:<8} center={:>6} min={:>6} max={:>6} lateral={}".format(
                item["category"],
                f"{ranging['center']['depth_m']:.2f}" if ranging.get("center") else "  -  ",
                f"{ranging['min']['depth_m']:.2f}", f"{ranging['max']['depth_m']:.2f}",
                f"{ranging['min']['lateral_m']:+.2f}" if ranging["min"].get("lateral_m") is not None else "-"))
            rows.append((key, item))
        media.write_image(OUT / f"detect_{key}.jpg",
                          media.draw_detections(frame.color.copy(), pred.detections()))

    truth_by_cat: dict[str, list[float]] = {}
    for item in truth:
        truth_by_cat.setdefault(item["category"], []).append(item["world_z"])
    measured, actual = [], []
    for key, item in rows:
        if key != "sim_ground_truth":
            continue
        category = item["category"]
        center = item["ranging"].get("center")
        if category in truth_by_cat and center:
            measured.append(center["depth_m"])
            actual.append(min(truth_by_cat[category],
                              key=lambda z: abs(z - center["depth_m"])))
    errors = np.asarray(measured) - np.asarray(actual)
    print("\n== 目标边界测距误差（对仿真真值） ==")
    if errors.size:
        print(f"    样本 {errors.size}  MAE={np.abs(errors).mean():.3f} m  "
              f"RMSE={np.sqrt((errors ** 2).mean()):.3f} m  最大={np.abs(errors).max():.3f} m")

    segmenter = registry.create("segmentation", "classical_road")
    segmenter.load(None)
    seg_pred = segmenter.predict(frame.color, depth_m=frame.depth_m)
    print(f"\n== 传统道路分割 == 掩膜占比 {float(seg_pred['mask'].mean()):.3f}, "
          f"{seg_pred['infer_ms']:.1f} ms")
    media.write_image(OUT / "segment.jpg", media.draw_mask(frame.color, seg_pred["mask"]))

    mask_strategy = MaskBoundaryStrategy(sampler)
    mask_results = mask_strategy.measure(frame.depth_m, frame.intrinsics, seg_pred)
    summary = road_summary(mask_results)
    print("== 道路边界测距 ==")
    for side in ("left", "right", "center"):
        block = summary[side]
        if block.get("min"):
            print(f"    {side:<6} min={block['min']['depth_m']:.2f}m "
                  f"max={block['max']['depth_m']:.2f}m "
                  f"lateral_min={block['min'].get('lateral_m')}")
    print(f"    可通行走廊宽度: {summary.get('corridor_width_m')} m"
          f"（合成道路真实全宽 6.20 m，参考深度 5.0 m）")

    road_vis = media.draw_mask(frame.color, seg_pred["mask"])
    for item in mask_results:
        if item.metric_kind in ("min", "max"):
            from algorithms.ranging.base import attach_visual_hints

            road_vis = media.draw_ranging_points(road_vis, attach_visual_hints(item))
    panel = []
    for side, label in (("left", "道路左边界"), ("right", "道路右边界")):
        for kind, kl in (("min", "Min"), ("max", "Max")):
            block = summary[side].get(kind)
            if block:
                panel.append(f"{label} {kl} {block['depth_m']:.2f} m")
    if summary.get("corridor_width_m"):
        panel.append(f"可通行走廊 {summary['corridor_width_m']:.2f} m")
    media.write_image(OUT / "road_ranging.jpg",
                      media.draw_panel(road_vis, panel, "top-right", "道路边界测距"))

    print("\n== 误差校正（三类模型） ==")
    if len(measured) >= 3:
        for method in ("linear", "ratio", "piecewise"):
            plan = calibrate(measured, actual, method=method)
            print(f"    {method:<10} rmse_before={plan.rmse_before:.4f} "
                  f"rmse_after={plan.rmse_after:.4f} 推荐={is_recommended(plan)}")
    else:
        print("    样本不足，跳过")

    print(f"\n产物目录: {OUT}")
    print("算法层验证通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
