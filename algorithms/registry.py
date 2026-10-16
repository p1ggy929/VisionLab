"""插件注册中心：模型可插拔的落点。"""

from __future__ import annotations

from typing import Any, Callable

from core.exceptions import ModelError
from core.logging_setup import get_logger
from .base import AlgorithmPlugin, BasePlugin

log = get_logger("algorithms.registry")

_registry: dict[str, dict[str, Callable[..., BasePlugin]]] = {"detection": {}, "segmentation": {}}
_meta: dict[str, dict] = {}


def register_plugin(task: str, key: str, factory: Callable[..., BasePlugin], meta: dict | None = None,
                    override: bool = False) -> Callable[..., BasePlugin]:
    """注册插件。可直接调用，也可作为装饰器使用（返回 factory 本身）。"""
    task = str(task).lower()
    if task not in _registry:
        raise ModelError(f"未知任务类型: {task}")
    if key in _registry[task] and not override:
        raise ModelError(f"插件已注册: {task}/{key}")
    _registry[task][key] = factory
    _meta[f"{task}/{key}"] = {"task": task, "key": key, **(meta or {})}
    log.info("注册插件: %s/%s", task, key)
    return factory


def plugin(task: str, key: str, meta: dict | None = None) -> Callable:
    """装饰器形式注册。"""

    def wrapper(cls):
        register_plugin(task, key, cls, meta)
        return cls

    return wrapper


def list_plugins(task: str | None = None) -> list[dict]:
    if task:
        keys = _registry.get(str(task).lower(), {})
        return [_meta.get(f"{task}/{k}", {"task": task, "key": k}) for k in keys]
    out = []
    for t, keys in _registry.items():
        out.extend(_meta.get(f"{t}/{k}", {"task": t, "key": k}) for k in keys)
    return out


def create(task: str, key: str, **kwargs: Any) -> BasePlugin:
    task = str(task).lower()
    factory = _registry.get(task, {}).get(key)
    if factory is None:
        raise ModelError(f"未注册的插件: {task}/{key}；可用: {list(_registry.get(task, {}))}")
    return factory(**kwargs)


def has(task: str, key: str) -> bool:
    return key in _registry.get(str(task).lower(), {})


def load_builtin() -> None:
    """导入内置插件模块，触发注册。"""
    from .detection import depth_adapter, sim_adapter, yolo_adapter  # noqa: F401
    from .segmentation import classical_adapter, unet_adapter  # noqa: F401
