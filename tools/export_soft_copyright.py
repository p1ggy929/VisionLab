"""导出软件著作权登记所需的源代码文档（前 30 页 + 后 30 页，每页 50 行）。

用法：
    python tools/export_soft_copyright.py                 # 默认前后各 30 页
    python tools/export_soft_copyright.py --pages 30 --lines 50

输出：
    storage/exports/软著源代码_前30页.txt
    storage/exports/软著源代码_后30页.txt
    storage/exports/软著源代码_统计.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import settings  # noqa: E402

# 优先级：入口与公共层在前，算法层在后
FILE_ORDER = [
    "app.py",
    "core/config.py", "core/db.py", "core/schema.py", "core/exceptions.py",
    "core/logging_setup.py", "core/storage.py", "core/geometry.py", "core/media.py",
    "core/camera/base.py", "core/camera/service.py", "core/camera/mock_backend.py",
    "core/camera/realsense_backend.py",
    "algorithms/base.py", "algorithms/registry.py", "algorithms/ground.py",
    "algorithms/detection/yolo_adapter.py", "algorithms/detection/depth_adapter.py",
    "algorithms/detection/sim_adapter.py",
    "algorithms/segmentation/unet_adapter.py", "algorithms/segmentation/classical_adapter.py",
    "algorithms/segmentation/models/blocks.py", "algorithms/segmentation/models/__init__.py",
    "algorithms/segmentation/models/unet.py", "algorithms/segmentation/models/resunet.py",
    "algorithms/segmentation/models/att_resunet.py", "algorithms/segmentation/models/unetv2.py",
    "algorithms/ranging/base.py", "algorithms/ranging/depth_sampler.py",
    "algorithms/ranging/box_boundary.py", "algorithms/ranging/mask_boundary.py",
    "algorithms/calibration/strategies.py",
    "services/auth_service.py", "services/project_service.py", "services/dataset_service.py",
    "services/model_registry.py", "services/pipeline_service.py", "services/capture_service.py",
    "services/analysis_service.py", "services/report_service.py", "services/health_service.py",
    "api/common.py", "api/auth.py", "api/projects.py", "api/camera.py", "api/datasets.py",
    "api/models.py", "api/inference.py", "api/analysis.py", "api/reports.py",
    "api/system.py", "api/media.py", "api/ws.py", "api/pages.py",
    "static/js/api.js", "static/js/ui.js", "static/js/pages.js",
    "templates/base.html", "templates/dashboard.html", "templates/ranging.html",
    "tools/import_legacy.py", "tools/self_check.py",
]

SKIP_PARTS = (".venv", "venv", "__pycache__", "storage", "node_modules", ".git")


def collect() -> list[Path]:
    files: list[Path] = []
    for rel in FILE_ORDER:
        path = ROOT / rel
        if path.exists() and path.is_file():
            files.append(path)
    # 补充未被显式列出的源码文件
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        if any(part in path.parts for part in SKIP_PARTS):
            continue
        if path.suffix.lower() not in (".py", ".js", ".html", ".css", ".yaml"):
            continue
        if path.name.startswith("_"):
            continue
        if path not in files:
            files.append(path)
    return files


def build_lines(files: list[Path]) -> tuple[list[str], dict]:
    lines: list[str] = []
    per_file: dict[str, int] = {}
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        content = [line.rstrip("\n") for line in text.splitlines()]
        # 去掉纯空行以符合「每页不少于 50 行」的排版要求
        content = [line for line in content if line.strip()]
        if not content:
            continue
        rel = path.relative_to(ROOT).as_posix()
        per_file[rel] = len(content)
        lines.append(f"// ==================== {rel} ====================")
        lines.extend(content)
        lines.append("")
    return lines, per_file


def main() -> int:
    parser = argparse.ArgumentParser(description="导出软著源代码文档")
    parser.add_argument("--pages", type=int, default=30, help="前后各取多少页（默认 30）")
    parser.add_argument("--lines", type=int, default=50, help="每页行数（默认 50）")
    args = parser.parse_args()

    files = collect()
    lines, per_file = build_lines(files)
    page_size = int(args.lines)
    needed = int(args.pages) * page_size

    out_dir = settings.storage_dir("exports")
    header = (f"视界智研算法实验平台 V1.0  源代码\n"
              f"导出时间：{datetime.now():%Y-%m-%d %H:%M:%S}\n"
              f"文件总数：{len(files)}   有效代码行：{len(lines)}\n")

    head_lines = lines[:needed]
    tail_lines = lines[-needed:] if len(lines) > needed else lines

    (out_dir / "软著源代码_前30页.txt").write_text(
        header + "\n".join(head_lines), encoding="utf-8")
    (out_dir / "软著源代码_后30页.txt").write_text(
        header + "\n".join(tail_lines), encoding="utf-8")

    stats = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "file_count": len(files),
        "code_lines": len(lines),
        "page_size": page_size,
        "pages_each_side": args.pages,
        "top_files": sorted(per_file.items(), key=lambda kv: -kv[1])[:20],
    }
    (out_dir / "软著源代码_统计.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 68)
    print(f"源码文件 {len(files)} 个，有效代码行 {len(lines)} 行")
    print(f"前 {args.pages} 页 / 后 {args.pages} 页已导出至：{out_dir}")
    print("=" * 68)
    for rel, count in stats["top_files"][:10]:
        print(f"  {count:>6} 行  {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
