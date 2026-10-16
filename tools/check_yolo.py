"""验证真实 YOLO 权重能否加载并在真实道路图像上推理。

用法：
    python tools/check_yolo.py                  # 抽查若干配置（默认含训练权重）
    python tools/check_yolo.py --all            # 遍历 20 个配置
    python tools/check_yolo.py --arch yolo12m   # 指定单一结构
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms import registry  # noqa: E402
from algorithms.detection.yolo_adapter import YoloDetectorAdapter  # noqa: E402
from core import media  # noqa: E402
from core.config import settings  # noqa: E402
from core.logging_setup import setup_logging  # noqa: E402

DATASET = Path(r"E:\2Project\2026dachuang\Yolo_dis_v3源代码\datasets\images\val")


def pick_images(limit: int = 3) -> list[Path]:
    if DATASET.exists():
        files = sorted(p for p in DATASET.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if files:
            step = max(1, len(files) // limit)
            return files[::step][:limit]
    out = settings.storage_dir("results") / "check_algorithms" / "scene_rgb.jpg"
    return [out] if out.exists() else []


def main() -> int:
    parser = argparse.ArgumentParser(description="验证 YOLO 权重加载与推理")
    parser.add_argument("--arch", default=None, help="只测一个结构，如 yolo12m")
    parser.add_argument("--all", action="store_true", help="遍历全部 20 个配置")
    parser.add_argument("--mode", default="trained",
                        choices=["trained", "pretrained", "dual"],
                        help="权重模式（默认用自训练权重，类别与平台一致）")
    args = parser.parse_args()

    setup_logging()
    registry.load_builtin()

    images = pick_images(3)
    if not images:
        print("找不到可用于推理的测试图像。")
        return 1
    print(f"测试图像 {len(images)} 张，来自 {images[0].parent}")
    print(f"权重模式: {args.mode}")
    print("权重搜索目录：")
    for folder in settings.weight_search_dirs("detection"):
        count = len(list(folder.glob('*.pt')))
        print(f"  {folder}  ({count} 个 .pt)")

    configs = YoloDetectorAdapter.available_configs()
    if args.arch:
        configs = [c for c in configs if c["key"] == args.arch]
    elif not args.all:
        picks = {"yolov8n", "yolov9t", "yolov10m", "yolo11s", "yolo12m"}
        configs = [c for c in configs if c["key"] in picks]
    print(f"\n将测试 {len(configs)} 个配置\n" + "=" * 96)

    ok_count, fail_count = 0, 0
    total_classes, translated, rows = set(), set(), []
    for cfg in configs:
        version, size = cfg["version"].lstrip("v"), cfg["size"]
        key = cfg["key"]
        try:
            plugin = registry.create("detection", "yolo", version=version, size=size,
                                     weight_mode=args.mode)
            t0 = time.perf_counter()
            plugin.load(None)
            load_ms = (time.perf_counter() - t0) * 1000

            image = media.read_image(images[0])
            t1 = time.perf_counter()
            pred = plugin.predict(image, conf=0.25, iou=0.45, imgsz=640)
            infer_ms = (time.perf_counter() - t1) * 1000

            desc = plugin.describe()
            names = set(str(n) for n in (desc.get("class_names") or []))
            total_classes |= names
            sources = list((desc.get("sources") or {}).keys())
            labels = [str(x) for x in (pred.get("labels") or [])]
            translated |= set(labels)
            rows.append({"配置": key, "权重来源": "+".join(sources) or "-",
                         "类别数": len(names), "检出": len(pred.get("boxes") or []),
                         "加载(ms)": round(load_ms), "推理(ms)": round(infer_ms)})
            detail = ("  检出类别=" + ",".join(sorted(set(labels)))) if labels else ""
            print(f"[OK ] {key:<10} 权重={list((desc.get('sources') or {}).values())}  "
                  f"类别 {len(names)}  检出 {len(pred.get('boxes') or [])}  "
                  f"加载 {load_ms:.0f}ms 推理 {infer_ms:.0f}ms{detail}")
            ok_count += 1
        except Exception as exc:  # noqa: BLE001
            print(f"[ERR] {key:<10} {str(exc)[:150]}")
            fail_count += 1

    print("=" * 96)
    print(f"通过 {ok_count} 个，失败 {fail_count} 个")
    if rows:
        header = list(rows[0].keys())
        print("\n| " + " | ".join(header) + " |")
        print("|" + "---|" * len(header))
        for row in rows:
            print("| " + " | ".join(str(row[h]) for h in header) + " |")
    if total_classes:
        platform_classes = set(settings.detection_class_names())
        print(f"\n平台 11 类（中文）：{sorted(platform_classes)}")
        print(f"权重内置类别名（英文）：{sorted(total_classes)}")
        print(f"经平台别名映射后的检出类别：{sorted(translated) or '（本次抽检未检出目标）'}")
        unmapped = {t for t in translated if t not in platform_classes}
        if unmapped:
            print(f"[警告] 未被映射到平台类别体系的检出：{sorted(unmapped)}")
        else:
            print("[OK ] 所有权重类别名都能映射到平台的中文类别体系")
    return 1 if fail_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
