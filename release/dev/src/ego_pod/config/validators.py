"""Ego_Config 的结构、安全红线与多规格范围校验规则（ADR 0005 / ADR 0006）。

校验顺序严格保持短路语义（检测到首个违规即抛出 :class:`~ego_pod.exit_codes.ConfigError`，
退出码 2，不打印成功确认信息）：

1. 顶层必需项（``sources`` → ``training`` → ``editions``）；
2. ``editions`` 非空、规格名称匹配 ``^[a-z0-9][a-z0-9-]{0,31}$``、各规格 ``base`` 属于
   :data:`~ego_pod.config.schema.SUPPORTED_BASE_MODELS` 白名单及可选覆盖参数合法；
3. ``sources`` 条目拒绝任何明文凭证字段，并校验 ``local`` / ``url`` / ``s3`` / ``oss``
   或已注册插件类型及其必填参数；
4. ``training.method == "qlora"``；
5. 五个 LoRA 训练参数完备性；
6. 可选 ``zones`` 块与 ``output.max_history`` 合法性。
"""

from __future__ import annotations

import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TextIO
from urllib.parse import urlparse

from ego_pod.config.schema import SUPPORTED_BASE_MODELS
from ego_pod.exit_codes import ConfigError

__all__ = ["validate"]

_REQUIRED_TOP_LEVEL: tuple[str, ...] = ("sources", "training", "editions")
_REQUIRED_TRAINING_METHOD = "qlora"

_TRAINING_PARAM_SPECS: tuple[tuple[str, str], ...] = (
    ("lora_rank", "pos_int"),
    ("lora_alpha", "pos_int"),
    ("learning_rate", "pos_num"),
    ("epochs", "pos_int"),
    ("batch_size", "pos_int"),
)

_CONSTRAINT_TEXT = {
    "pos_int": "必须为正整数",
    "pos_num": "必须为大于 0 的数值",
}

_EDITION_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,31}$")
_ZONE_NAME_PATTERN = re.compile(r"^[a-z0-9-]+$")

_DATA_CATEGORIES: frozenset[str] = frozenset(
    {"sense", "action", "thought", "skill", "credentials"}
)

#: 严禁写入 ego-config.yaml 的明文云端/API 凭证字段名集合（ADR 0006 §5）。
_FORBIDDEN_CREDENTIAL_KEYS: frozenset[str] = frozenset(
    {
        "access_key",
        "secret_key",
        "access_key_id",
        "secret_access_key",
        "token",
        "session_token",
        "api_key",
        "password",
        "credentials_json",
    }
)

_LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})


def _mapping_get(obj: Any, key: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key)
    return None


def _is_present(data: Any, key: str) -> bool:
    if not isinstance(data, Mapping):
        return False
    return key in data and data[key] is not None


def _is_pos_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_non_neg_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_pos_num(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value > 0
    )


def _is_iterable_seq(value: Any) -> bool:
    return isinstance(value, (list, tuple))


def _is_registered_plugin_type(source_type: str) -> bool:
    """惰性检查自定义数据源插件是否已在插件注册表中注册。"""

    try:
        from ego_pod.collector.plugins import is_registered_collector_plugin  # noqa: PLC0415
    except ImportError:
        return False
    return is_registered_collector_plugin(source_type)


def _check_top_level_present(data: Any) -> None:
    for field in _REQUIRED_TOP_LEVEL:
        if not _is_present(data, field):
            raise ConfigError(
                f"ego-config.yaml 顶层缺少必需配置项 '{field}'"
            )


def _check_editions_and_bases(data: Mapping[str, Any]) -> None:
    """校验多规格 editions 字典、规格命名、受控基座白名单与可选单规格覆盖项。"""

    editions = data["editions"]
    if not isinstance(editions, Mapping) or not editions:
        raise ConfigError(
            "editions 不能为空映射：请至少声明一个有效 Edition（如 normal / lite / plus / super 或自定义规格）"
        )

    supported_str = ", ".join(SUPPORTED_BASE_MODELS)
    for raw_name, edition_def in editions.items():
        if not isinstance(raw_name, str) or not _EDITION_NAME_PATTERN.fullmatch(raw_name):
            raise ConfigError(
                f"非法的 Edition 名称 {raw_name!r}：必须匹配 ^[a-z0-9][a-z0-9-]{{0,31}}$"
            )
        if not isinstance(edition_def, Mapping):
            raise ConfigError(
                f"editions.{raw_name} 必须为映射对象并包含合法 base 字段（受支持基座：{supported_str}）"
            )

        base = _mapping_get(edition_def, "base")
        if not isinstance(base, str) or base not in SUPPORTED_BASE_MODELS:
            raise ConfigError(
                f"editions.{raw_name}.base 非法（当前为 {base!r}）；"
                f"受支持基座模型白名单为：{supported_str}"
            )

        for int_override in ("lora_rank", "lora_alpha", "max_level"):
            if int_override in edition_def and edition_def[int_override] is not None:
                if not _is_pos_int(edition_def[int_override]):
                    raise ConfigError(
                        f"editions.{raw_name}.{int_override} 必须为正整数"
                    )

        if "max_size_mb" in edition_def and edition_def["max_size_mb"] is not None:
            if not _is_pos_num(edition_def["max_size_mb"]):
                raise ConfigError(
                    f"editions.{raw_name}.max_size_mb 必须为大于 0 的数值"
                )

        if "zones" in edition_def and edition_def["zones"] is not None:
            raw_zones = edition_def["zones"]
            if not _is_iterable_seq(raw_zones):
                raise ConfigError(
                    f"editions.{raw_name}.zones 必须为分区名称列表"
                )
            for z_name in raw_zones:
                if not isinstance(z_name, str) or not _ZONE_NAME_PATTERN.fullmatch(z_name):
                    raise ConfigError(
                        f"editions.{raw_name}.zones 含非法 Zone_Name {z_name!r}"
                    )


def _validate_url_field(name: str, raw_url: Any) -> None:
    if not isinstance(raw_url, str) or not raw_url.strip():
        raise ConfigError(
            f"数据源条目 '{name}' 的 type 为 'url' 但缺少合法 url 字段"
            "（若使用本地目录请设为 local 数据源）"
        )
    parsed = urlparse(raw_url.strip())
    scheme = (parsed.scheme or "").lower()
    host = (parsed.hostname or "").lower()
    if scheme == "https" and host:
        return
    if scheme == "http" and host in _LOOPBACK_HOSTS:
        return
    raise ConfigError(
        f"数据源条目 '{name}' 的 url {raw_url!r} 不满足传输安全要求："
        "仅允许 https:// 或本地环回 http://127.0.0.1|localhost（若使用本地目录请设为 local 数据源）"
    )


def _check_sources(data: Mapping[str, Any]) -> None:
    """校验数据源条目：拦截明文凭证并校验内置/插件数据源必填字段。"""

    sources = data["sources"]
    if not isinstance(sources, Mapping):
        return

    for name, entry in sources.items():
        if isinstance(entry, Mapping):
            for key in entry.keys():
                if str(key).lower() in _FORBIDDEN_CREDENTIAL_KEYS:
                    raise ConfigError(
                        f"数据源条目 '{name}' 包含禁止写入配置文件的明文凭证字段 {key!r}；"
                        "请移除该字段并改用环境变量 EGO_CLOUD_ACCESS_KEY_ID / EGO_CLOUD_SECRET_ACCESS_KEY"
                    )
            nested_opts = entry.get("options")
            if isinstance(nested_opts, Mapping):
                for key in nested_opts.keys():
                    if str(key).lower() in _FORBIDDEN_CREDENTIAL_KEYS:
                        raise ConfigError(
                            f"数据源条目 '{name}.options' 包含禁止落盘的明文凭证字段 {key!r}；"
                            "请改用环境变量 EGO_CLOUD_ACCESS_KEY_ID / EGO_CLOUD_SECRET_ACCESS_KEY"
                        )

        entry_type = _mapping_get(entry, "type")
        if entry_type == "local":
            continue
        if entry_type == "url":
            _validate_url_field(str(name), _mapping_get(entry, "url"))
            continue
        if entry_type in ("s3", "oss"):
            bucket = _mapping_get(entry, "bucket")
            raw_url = _mapping_get(entry, "url")
            has_bucket = isinstance(bucket, str) and bool(bucket.strip())
            has_cloud_uri = isinstance(raw_url, str) and raw_url.strip().startswith(
                ("s3://", "oss://")
            )
            if not (has_bucket or has_cloud_uri):
                raise ConfigError(
                    f"数据源条目 '{name}' 的 type 为 {entry_type!r} 但缺少 bucket 或 {entry_type}:// URI"
                    "（若使用本地目录请设为 local 数据源）"
                )
            continue
        if isinstance(entry_type, str) and entry_type and _is_registered_plugin_type(entry_type):
            continue

        raise ConfigError(
            f"数据源条目 '{name}' 的 type 为 {entry_type!r}："
            "不支持的数据源类型（内置支持 local 数据源 / url / s3 / oss 或已注册插件）"
        )


def _check_training_method(data: Mapping[str, Any]) -> None:
    method = _mapping_get(data["training"], "method")
    if method != _REQUIRED_TRAINING_METHOD:
        raise ConfigError(
            "仅支持 qlora 训练方法："
            f"training.method 当前为 {method!r}"
        )


def _check_training_params(data: Mapping[str, Any]) -> None:
    training = data["training"]
    for name, kind in _TRAINING_PARAM_SPECS:
        value = _mapping_get(training, name)
        valid = _is_pos_int(value) if kind == "pos_int" else _is_pos_num(value)
        if not valid:
            raise ConfigError(
                f"训练参数 '{name}' 缺失或非法（{_CONSTRAINT_TEXT[kind]}）"
            )


def _check_zone_config(data: Any) -> None:
    zones_block = _mapping_get(data, "zones")
    if zones_block is None:
        return

    if isinstance(zones_block, Mapping) and "max_level" in zones_block and zones_block["max_level"] is not None:
        if not _is_pos_int(zones_block["max_level"]):
            raise ConfigError("zones.max_level 必须为正整数")

    declared_raw = _mapping_get(zones_block, "zones")
    declared: list[Any] = list(declared_raw) if _is_iterable_seq(declared_raw) else []

    seen: set[str] = set()
    for name in declared:
        if not isinstance(name, str) or not _ZONE_NAME_PATTERN.fullmatch(name):
            raise ConfigError(
                f"非法的 Zone_Name {name!r}："
                "分区名必须由小写字母、数字与连字符（-）组成且非空"
            )
        if name in seen:
            raise ConfigError(
                f"重复的 Zone_Name '{name}'：Zone_Config 中分区名必须唯一"
            )
        seen.add(name)

    routing = _mapping_get(zones_block, "routing")
    if isinstance(routing, Mapping):
        for category, target in routing.items():
            if category not in _DATA_CATEGORIES:
                print(
                    f"警告：routing 中的键 {category!r} 不属于 Data_Category 五元集"
                    "（sense/action/thought/skill/credentials），该规则永不命中样本",
                    file=sys.stderr,
                    flush=True,
                )
            if target not in seen:
                raise ConfigError(
                    f"routing 中键 {category!r} 指向未声明的 Functional_Zone "
                    f"{target!r}：routing 的每个值必须是已声明的 Zone_Name"
                )


def _check_output_config(data: Any) -> None:
    output_block = _mapping_get(data, "output")
    if not isinstance(output_block, Mapping):
        return
    if "max_history" in output_block and output_block["max_history"] is not None:
        if not _is_non_neg_int(output_block["max_history"]):
            raise ConfigError("output.max_history 必须为非负整数")


def validate(
    data: Any,
    config_path: str | Path,
    *,
    stream: TextIO | None = None,
) -> None:
    """对原始配置映射执行结构、基座白名单与安全红线校验。"""

    _check_top_level_present(data)
    _check_editions_and_bases(data)
    _check_sources(data)
    _check_training_method(data)
    _check_training_params(data)
    _check_zone_config(data)
    _check_output_config(data)

    abs_path = Path(config_path).resolve()
    target = sys.stdout if stream is None else stream
    print(f"Config validated: {abs_path}", file=target, flush=True)
