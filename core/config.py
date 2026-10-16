"""配置加载器：YAML + 环境变量覆盖，全平台唯一配置入口。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

from .exceptions import ConfigError

# 仓库根目录：本文件位于 <root>/core/config.py
ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"

_ENV_PREFIX = "VISIONLAB_"


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"配置文件不存在: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"配置文件格式错误（顶层必须是映射）: {path}")
    return data


def _apply_env_overrides(data: Dict[str, Any]) -> Dict[str, Any]:
    """支持 VISIONLAB_SECTION__KEY=value 形式覆盖，值按 YAML 解析。"""
    for raw_key, raw_value in os.environ.items():
        if not raw_key.startswith(_ENV_PREFIX):
            continue
        path = raw_key[len(_ENV_PREFIX):].lower().split("__")
        if not path or not path[0]:
            continue
        cursor = data
        for part in path[:-1]:
            cursor = cursor.setdefault(part, {})
            if not isinstance(cursor, dict):
                break
        else:
            try:
                parsed = yaml.safe_load(raw_value)
            except yaml.YAMLError:
                parsed = raw_value
            cursor[path[-1]] = parsed
    return data


class Settings:
    """平台配置对象。点号访问：settings.get("ranging.kernel")。"""

    def __init__(self, root: Path = ROOT, config_dir: Path = CONFIG_DIR) -> None:
        self.root = Path(root)
        self.config_dir = Path(config_dir)
        merged: Dict[str, Any] = {}
        for name in ("default.yaml", "classes.yaml", "models.yaml"):
            merged = _deep_merge(merged, _read_yaml(self.config_dir / name))
        merged = _apply_env_overrides(merged)
        self._data = merged
        # 存储根目录解析为绝对路径
        storage_root = Path(str(self.get("storage.root", "storage")))
        if not storage_root.is_absolute():
            storage_root = self.root / storage_root
        self.storage_root = storage_root.resolve()

    # ---------- 读取 ----------
    def get(self, dotted: str, default: Any = None) -> Any:
        cursor: Any = self._data
        for part in dotted.split("."):
            if not isinstance(cursor, dict) or part not in cursor:
                return default
            cursor = cursor[part]
        return cursor

    def require(self, dotted: str) -> Any:
        value = self.get(dotted, None)
        if value is None:
            raise ConfigError(f"缺少必填配置项: {dotted}")
        return value

    def section(self, name: str) -> Dict[str, Any]:
        value = self.get(name, {})
        return value if isinstance(value, dict) else {}

    def as_dict(self) -> Dict[str, Any]:
        return self._data

    # ---------- 派生路径 ----------
    def storage_dir(self, name: str) -> Path:
        rel = self.get(f"storage.dirs.{name}", name)
        path = self.storage_root / str(rel)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def db_path(self) -> Path:
        path = self.storage_root / "visionlab.db"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def log_path(self) -> Path:
        raw = str(self.get("logging.file", "storage/logs/visionlab.log"))
        path = Path(raw)
        if not path.is_absolute():
            path = self.root / path
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    # ---------- 类别体系 ----------
    def detection_classes(self) -> list[dict]:
        return list(self.get("detection.classes", []) or [])

    def detection_class_names(self) -> list[str]:
        return [c["name"] for c in self.detection_classes()]

    def class_group(self, name: str) -> str:
        for item in self.detection_classes():
            if item.get("name") == name:
                return str(item.get("group", "large"))
        return "large"

    def class_alias(self, name: str) -> str:
        aliases = self.get("detection.aliases", {}) or {}
        value = aliases.get(name)
        if isinstance(value, list) and value:
            return str(value[0])
        if isinstance(value, str):
            return value
        return name

    def detection_num_classes(self) -> int:
        return int(self.get("detection.num_classes", len(self.detection_classes())))

    def segmentation_foreground_values(self) -> list[int]:
        values = self.get("segmentation.foreground_values", [1, 17, 255]) or [1, 17, 255]
        return [int(v) for v in values]

    # ---------- 外部权重/数据来源 ----------
    def external_source(self, key: str) -> Path | None:
        """解析 models.yaml 中登记的外部权重/数据目录（相对路径按仓库根解析）。"""
        raw = self.get(f"external_sources.{key}")
        if not raw:
            return None
        path = Path(str(raw))
        if not path.is_absolute():
            path = self.root / path
        return path if path.exists() else None

    def weight_search_dirs(self, task: str) -> list[Path]:
        """返回某任务可用的权重搜索目录（平台自身目录优先，其次外部原项目目录）。"""
        dirs: list[Path] = []
        if task == "detection":
            dirs += [self.storage_dir("models") / "pretrained",
                     self.storage_dir("models") / "trained"]
            for key in ("detection_pretrained", "detection_trained"):
                path = self.external_source(key)
                if path:
                    dirs.append(path)
        else:
            dirs.append(self.storage_dir("models") / "trained")
            path = self.external_source("segmentation")
            if path:
                dirs.append(path)
        seen: set[str] = set()
        result = []
        for item in dirs:
            key = str(item).lower()
            if key not in seen and item.exists():
                seen.add(key)
                result.append(item)
        return result

    def all_dirs(self) -> None:
        for name in ("datasets", "captures", "models", "runs", "results",
                     "calibration", "reports", "logs", "exports"):
            self.storage_dir(name)


_settings: Settings | None = None


def get_settings(reload: bool = False) -> Settings:
    global _settings
    if _settings is None or reload:
        _settings = Settings()
    return _settings


settings = get_settings()
