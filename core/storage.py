"""存储层：相对 URI 解析、哈希、目录组织。禁止在业务代码里出现绝对路径。"""

from __future__ import annotations

import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .config import settings
from .exceptions import ValidationError


def to_uri(path: Path | str) -> str:
    """绝对路径 -> 相对存储根的 URI（统一用 / 分隔）。"""
    p = Path(path).resolve()
    try:
        rel = p.relative_to(settings.storage_root)
    except ValueError:
        return str(p).replace("\\", "/")
    return str(rel).replace("\\", "/")


def to_path(uri: str | None) -> Path | None:
    """相对 URI -> 绝对路径（兼容历史遗留的绝对路径）。"""
    if not uri:
        return None
    raw = str(uri)
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    return settings.storage_root / raw


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_name(name: str, fallback: str = "item") -> str:
    keep = []
    for ch in str(name):
        if ch.isalnum() or ch in "-_.（）()【】[]":
            keep.append(ch)
        elif ch.isspace():
            keep.append("_")
    result = "".join(keep).strip("_")
    return result or fallback


def timestamp(fmt: str = "%Y%m%d_%H%M%S") -> str:
    """本地时间戳字符串（显式取 UTC 再转本地时区，避免依赖隐式时区）。"""
    return datetime.now(timezone.utc).astimezone().strftime(fmt)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_tree(paths: list[Path], fast_above: int = 300) -> str:
    """对文件名 + 大小 + 内容做联合哈希，用于数据集内容指纹。

    文件数超过 fast_above 时退化为「文件名 + 大小 + 修改时间」的元数据指纹，
    避免对数千张图像逐个读取内容（导入大目录时会慢几个数量级）。
    """
    items = sorted(p for p in paths if p and Path(p).exists())
    digest = hashlib.sha256()
    fast = len(items) > fast_above
    for path in items:
        digest.update(path.name.encode("utf-8", "ignore"))
        try:
            stat = path.stat()
        except OSError:
            continue
        digest.update(str(stat.st_size).encode())
        if fast:
            digest.update(str(int(stat.st_mtime)).encode())
        elif path.is_file() and stat.st_size < 8 * 1024 * 1024:
            digest.update(sha256_file(path).encode())
    digest.update(f"|n={len(items)}|fast={int(fast)}".encode())
    return digest.hexdigest()


def copy_into(src: Path, dest_dir: Path, name: str | None = None) -> Path:
    if not Path(src).exists():
        raise ValidationError(f"源文件不存在: {src}")
    ensure_dir(dest_dir)
    target = dest_dir / (name or Path(src).name)
    if Path(src).resolve() != target.resolve():
        shutil.copy2(src, target)
    return target


def human_size(num_bytes: int | float) -> str:
    value = float(num_bytes or 0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} TB"


def dir_size(path: Path) -> int:
    total = 0
    if Path(path).exists():
        for item in Path(path).rglob("*"):
            if item.is_file():
                try:
                    total += item.stat().st_size
                except OSError:
                    pass
    return total


def free_space_gb(path: Path | None = None) -> float:
    target = Path(path) if path else settings.storage_root
    try:
        usage = shutil.disk_usage(str(target if target.exists() else target.parent))
        return round(usage.free / (1024 ** 3), 2)
    except OSError:
        return -1.0
