"""``ego-config.yaml`` 的加载与解析错误定位。

本模块只负责"把磁盘上的 YAML 文件读进来、解析成内存中的 :class:`EgoConfig`
数据结构"这一件事，对应需求 R3 中的加载相关条款（R3-AC1/AC2/AC3/AC4）与
R11-AC6（``ego train`` 配置加载失败）。

职责边界（重要）：

- **本模块负责**：解析路径为绝对路径、检测文件是否存在、读取文件、解析
  YAML、把原始字典转换为 :class:`EgoConfig` / :class:`SourceEntry` /
  :class:`TrainingParams` / :class:`EditionDef` 等 dataclass，并返回
  ``(EgoConfig, absolute_path)`` 二元组。下列四类错误均映射为
  :class:`~ego_pod.exit_codes.ConfigError`（退出码 2）：

  1. 文件不存在（R3-AC2）
  2. 文件存在但无读权限，或存在性检查与读取之间 / 读取过程中发生任何
     I/O 错误（文件被删除、被替换、变为不可读等，R3-AC3）
  3. YAML 解析错误（R3-AC4），错误信息附带解析错误所在的行号与列号

- **本模块不负责**：结构与取值校验（顶层字段缺失、normal-only、
  qwen3.5-0.8B-only、local-only、qlora-only、五个 LoRA 参数完备性等）。
  这些由 ``validators.py``（任务 2.4）在拿到 :class:`EgoConfig` 之后完成。

为了把"字段缺失 / 取值非法"的判定权完整交给 ``validators.py``，本模块在
把原始字典转换为 dataclass 时采取**宽松构造**策略：对缺失的字段填入
``None``（Python dataclass 在运行期不强制类型注解，因此 ``frozen=True`` 与
``Literal`` 注解都不会阻止以 ``None`` 构造）。这样 ``validators.py`` 既能区分
"字段缺失"（值为 ``None``）与"取值非法"（值存在但不合法），又无需访问原始
字典。本模块自身绝不因字段缺失或取值非法而抛错——它只对加载/解析失败抛错。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from ego_pod.config.schema import (
    EditionDef,
    EgoConfig,
    SourceEntry,
    TrainingParams,
    ZoneConfig,
)
from ego_pod.exit_codes import ConfigError

__all__ = ["load"]

#: 文件不存在时的固定提示（对应 R3-AC2）。
_NOT_FOUND_MESSAGE = "未找到 `ego-config.yaml`，请先执行 `ego init`"


def load(path: str | os.PathLike[str]) -> tuple[EgoConfig, str]:
    """加载并解析 ``ego-config.yaml``，返回 ``(EgoConfig, 绝对路径)``。

    :param path: 配置文件路径，可为相对或绝对路径。
    :returns: 二元组 ``(config, absolute_path)``，其中 ``config`` 为宽松构造的
        :class:`EgoConfig`（缺失字段以 ``None`` 占位，交由 ``validators.py``
        校验），``absolute_path`` 为配置文件的绝对路径字符串。
    :raises ConfigError: 当文件不存在（R3-AC2）、无读权限或读取期间发生 I/O
        错误（R3-AC3）、或 YAML 解析失败（R3-AC4）时抛出，``exit_code`` 为 2。
    """
    abs_path = os.path.abspath(os.fspath(path))

    # --- R3-AC2：存在性检查 -------------------------------------------------
    # 注意：此处的 not-found 判定与下方读取阶段的 FileNotFoundError 分属两类。
    # 启动时文件本就不存在 → R3-AC2（提示用户先 ego init）；存在性检查通过后
    # 文件在读取前被删除 / 替换 → 落入下方 OSError 分支按 R3-AC3 的 I/O 错误处理。
    if not Path(abs_path).exists():
        raise ConfigError(_NOT_FOUND_MESSAGE)

    # --- R3-AC3：读取（含权限错误与读取期间 I/O 错误）-----------------------
    # 以二进制方式读取后交给 PyYAML 自行检测编码（含 BOM），从而把编码异常
    # 归入 YAML 解析错误（ReaderError ⊂ YAMLError），而把真正的文件系统错误
    # 归入此处的 OSError。
    try:
        raw_bytes = Path(abs_path).read_bytes()
    except PermissionError as exc:
        raise ConfigError(
            f"无法读取配置文件 {abs_path}：没有读权限（{exc.strerror or exc}）"
        ) from exc
    except OSError as exc:
        # 涵盖：存在性检查后文件被删除（FileNotFoundError）、被替换为目录
        # （IsADirectoryError）、设备 I/O 错误等一切文件系统层面的读取失败。
        reason = exc.strerror or str(exc)
        raise ConfigError(
            f"读取配置文件 {abs_path} 时发生 I/O 错误：{reason}"
        ) from exc

    # --- R3-AC4：YAML 解析（解析失败时附带行列号）---------------------------
    try:
        data = yaml.safe_load(raw_bytes)
    except yaml.YAMLError as exc:
        raise ConfigError(_format_yaml_error(abs_path, exc)) from exc

    # --- 宽松构造 dataclass（取值/结构校验交给 validators.py）---------------
    config = _build_config(data)
    return config, abs_path


# ---------------------------------------------------------------------------
# YAML 解析错误的行列号提取
# ---------------------------------------------------------------------------


def _format_yaml_error(abs_path: str, exc: yaml.YAMLError) -> str:
    """把 :class:`yaml.YAMLError` 格式化为带行列号的错误信息（R3-AC4）。

    PyYAML 的标记型错误（:class:`yaml.MarkedYAMLError`，如 ScannerError /
    ParserError / ComposerError）携带 ``problem_mark`` / ``context_mark``，
    其 ``line`` / ``column`` 均为 0 基；此处转换为更符合用户直觉的 1 基行列号。
    对不含位置标记的错误（如 ReaderError 编码错误）则省略行列号。
    """
    mark = getattr(exc, "problem_mark", None) or getattr(exc, "context_mark", None)
    if mark is not None:
        line = mark.line + 1
        column = mark.column + 1
        return (
            f"配置文件 {abs_path} 不是合法的 YAML 文档"
            f"（行 {line}，列 {column}）：{exc}"
        )
    return f"配置文件 {abs_path} 不是合法的 YAML 文档：{exc}"


# ---------------------------------------------------------------------------
# 原始字典 → dataclass 的宽松转换
# ---------------------------------------------------------------------------


def _build_config(data: Any) -> EgoConfig:
    """把 YAML 解析出的原始对象宽松地转换为 :class:`EgoConfig`。

    空文件解析为 ``None``，非映射的顶层文档（如 YAML 列表/标量）同样被视为
    "各顶层字段均缺失"，以便 ``validators.py`` 按 R3-AC5 报告首个缺失字段。
    缺失的顶层字段以 ``None`` 占位；``output`` 不在 MVP 校验范围内，缺失时
    回退为空字典。
    """
    if not isinstance(data, dict):
        data = {}

    output = data.get("output")
    if not isinstance(output, dict):
        output = {}

    return EgoConfig(
        sources=_build_sources(data.get("sources")),
        training=_build_training(data.get("training")),
        editions=_build_editions(data.get("editions")),
        output=output,
        zones=_build_zone_config(data.get("zones")),
    )


def _build_sources(raw: Any) -> dict[str, SourceEntry] | None:
    """转换 ``sources`` 段；缺失或非映射时返回 ``None`` 以表示"字段缺失"。"""
    if not isinstance(raw, dict):
        return None
    return {str(name): _build_source_entry(entry) for name, entry in raw.items()}


def _build_source_entry(entry: Any) -> SourceEntry:
    """转换单个数据源条目；缺失的 ``type`` / ``path`` 以 ``None`` 占位。"""
    if not isinstance(entry, dict):
        return SourceEntry(type=entry, path=None)  # 非映射条目：交给校验器拒绝
    return SourceEntry(type=entry.get("type"), path=entry.get("path"))


def _build_training(raw: Any) -> TrainingParams | None:
    """转换 ``training`` 段；缺失或非映射时返回 ``None`` 表示"字段缺失"。

    五个 LoRA 数值参数与 ``method`` / ``quantization`` 缺失时均以 ``None``
    占位，由 ``validators.py``（R3-AC9 / R8-AC9）判定缺失或非法。
    """
    if not isinstance(raw, dict):
        return None
    return TrainingParams(
        method=raw.get("method"),
        quantization=raw.get("quantization"),
        lora_rank=raw.get("lora_rank"),
        lora_alpha=raw.get("lora_alpha"),
        learning_rate=raw.get("learning_rate"),
        epochs=raw.get("epochs"),
        batch_size=raw.get("batch_size"),
    )


def _build_editions(raw: Any) -> dict[str, EditionDef] | None:
    """转换 ``editions`` 段；缺失或非映射时返回 ``None`` 表示"字段缺失"。"""
    if not isinstance(raw, dict):
        return None
    return {str(name): _build_edition_def(edition) for name, edition in raw.items()}


def _build_edition_def(edition: Any) -> EditionDef:
    """转换单个 Edition 定义；缺失的 ``base`` 以 ``None`` 占位。"""
    if not isinstance(edition, dict):
        return EditionDef(base=edition)  # 非映射定义：交给校验器拒绝
    description = edition.get("description", "")
    return EditionDef(base=edition.get("base"), description=description)


def _build_zone_config(raw: Any) -> ZoneConfig | None:
    """转换可选的顶层 ``zones`` 段为 :class:`ZoneConfig`，沿用宽松构造风格。

    与既有 ``_build_*`` 一致地不在此处做合法性判定（Zone_Name 正则、去重、
    ``routing`` 取值是否指向已声明分区等均由 ``validators.py`` 的
    ``_check_zone_config`` 负责）：

    - ``zones`` 整块缺省（键不存在或值为 ``None``）⇒ 返回 ``None``，使
      ``EgoConfig.zones`` 为 ``None``，下游据此回退到内置默认基线（R1-AC8/AC9）。
    - ``zones`` 块存在但子字段缺失/类型异常 ⇒ 以空 ``tuple`` / 空 ``dict`` 宽松
      占位，把合法性判定完整交给校验器。
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        # 非映射的 zones 块（如标量/列表）：视为各子字段缺失，交给校验器拒绝。
        return ZoneConfig(zones=(), routing={})
    zones_raw = raw.get("zones") or []
    routing_raw = raw.get("routing") or {}
    zones = tuple(zones_raw) if isinstance(zones_raw, (list, tuple)) else ()
    routing = dict(routing_raw) if isinstance(routing_raw, dict) else {}
    return ZoneConfig(zones=zones, routing=routing)
