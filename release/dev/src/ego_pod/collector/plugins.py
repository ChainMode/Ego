"""可插拔数据源注册表与 Python entry_points 发现机制（ADR 0006 §5）。

允许通过 ``importlib.metadata.entry_points(group="ego_pod.collectors")`` 或编程式
:func:`register_collector_plugin` 注入自定义数据源适配器，并在注册与加载阶段对必需的
``collect(entry, category)`` 方法执行严格前置校验。
"""

from __future__ import annotations

import importlib.metadata
from typing import Any

from ego_pod.collector.base import SourceAdapter
from ego_pod.exit_codes import CollectorError

__all__ = [
    "ENTRY_POINT_GROUP",
    "validate_plugin_adapter",
    "register_collector_plugin",
    "unregister_collector_plugin",
    "clear_collector_plugins",
    "discover_entry_point_plugins",
    "is_registered_collector_plugin",
    "get_collector_plugin",
]

ENTRY_POINT_GROUP = "ego_pod.collectors"

_PROGRAMMATIC_PLUGINS: dict[str, SourceAdapter] = {}


def validate_plugin_adapter(plugin_name: str, candidate: Any) -> SourceAdapter:
    """实例化（若传入类）并前置校验插件是否实现可调用的 ``collect`` 方法。"""

    instance = candidate
    if isinstance(candidate, type):
        try:
            instance = candidate()
        except Exception as exc:  # noqa: BLE001
            raise CollectorError(
                f"[COLLECTOR] 无法实例化数据源插件 {plugin_name!r}：{exc}"
            ) from exc

    collect_fn = getattr(instance, "collect", None)
    if not callable(collect_fn):
        raise CollectorError(
            f"[COLLECTOR] 数据源插件 {plugin_name!r} 接口校验失败：缺少必需的可调用方法 'collect'"
        )
    return instance  # type: ignore[return-value]


def register_collector_plugin(source_type: str, plugin: Any) -> SourceAdapter:
    """在当前进程内注册自定义数据源插件并立即校验其 ``collect`` 接口。"""

    if not isinstance(source_type, str) or not source_type.strip():
        raise CollectorError("[COLLECTOR] 插件 source_type 必须为非空字符串")
    key = source_type.strip()
    validated = validate_plugin_adapter(key, plugin)
    _PROGRAMMATIC_PLUGINS[key] = validated
    return validated


def unregister_collector_plugin(source_type: str) -> None:
    _PROGRAMMATIC_PLUGINS.pop(source_type, None)


def clear_collector_plugins() -> None:
    _PROGRAMMATIC_PLUGINS.clear()


def discover_entry_point_plugins() -> dict[str, Any]:
    """扫描环境中声明于 ``ego_pod.collectors`` 组的 entry_points。"""

    discovered: dict[str, Any] = {}
    try:
        eps = importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)
    except TypeError:  # pragma: no cover
        eps = importlib.metadata.entry_points().get(ENTRY_POINT_GROUP, ())  # type: ignore[assignment]
    for ep in eps:
        discovered[ep.name] = ep
    return discovered


def is_registered_collector_plugin(source_type: str) -> bool:
    """判断给定 ``source_type`` 是否为已注册或可通过 entry_points 发现的插件。"""

    if source_type in _PROGRAMMATIC_PLUGINS:
        return True
    eps = discover_entry_point_plugins()
    return source_type in eps


def get_collector_plugin(source_type: str) -> SourceAdapter | None:
    """按 ``source_type`` 获取已校验的插件适配器实例；未注册时返回 ``None``。"""

    if source_type in _PROGRAMMATIC_PLUGINS:
        return _PROGRAMMATIC_PLUGINS[source_type]

    eps = discover_entry_point_plugins()
    ep = eps.get(source_type)
    if ep is None:
        return None
    loaded = ep.load()
    return validate_plugin_adapter(source_type, loaded)
