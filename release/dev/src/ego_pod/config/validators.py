"""Ego_Config 的 MVP 范围校验规则。

本模块在 ``loader`` 完成 YAML 解析（对应 Requirements 3.1–3.4 的文件加载与
语法解析）之后，对解析得到的**原始映射**执行 Requirements 3.5–3.9 所定义的
结构与取值校验，并在全部通过时打印 Requirements 3.10 要求的成功确认信息。

为什么校验作用于"原始映射"而非已构造的 :class:`~ego_pod.config.schema.EgoConfig`：
R3-AC5 要求在顶层缺失 ``sources`` / ``training`` / ``editions`` 任一字段时给出
报错，而带有必填字段的 ``EgoConfig`` dataclass 在字段缺失时根本无法被构造。
因此本模块以 ``loader`` 解析出的 :class:`~collections.abc.Mapping`（PyYAML 的
解析产物）作为输入；``loader``（任务 2.3）应当在调用 :func:`validate` 通过后，
再据此构造 ``EgoConfig`` 并返回。

校验顺序严格遵循 Requirements 3.11 规定的 AC 编号升序短路（本模块负责
AC5 → AC6 → AC7 → AC8 → AC9，AC3/AC4 的 I/O 与语法错误由 ``loader`` 处理），
并在其后追加 Requirements 8.9 的五个 LoRA 训练参数完备性/合法性预检：

================  ============================================================
检查步骤          语义（对应 Requirement）
================  ============================================================
顶层缺失          R3-AC5：按 sources → training → editions 顺序检测，仅报告首个缺失
normal-only       R3-AC6：``editions`` 必须声明 ``normal`` Edition
qwen3.5-0.8B-only R3-AC7：``editions.normal.base`` 必须为 ``qwen3.5-0.8B``
local-only        R3-AC8 / R9-AC2：``sources`` 中不得存在 ``type`` 非 ``local`` 的条目
qlora-only        R3-AC9：``training.method`` 必须为 ``qlora``
五参数完备性      R8-AC9：lora_rank/lora_alpha/epochs/batch_size 为正整数、learning_rate > 0
Zone_Config       R1-AC3/AC10：可选 ``zones`` 块的 Zone_Name 正则/去重/routing 取值合法性
================  ============================================================

任一步骤检测到违规即抛出 :class:`~ego_pod.exit_codes.ConfigError`（退出码 2）
并立即停止，不继续后续检查，也不收集或合并多条错误（Requirements 3.11）。
仅当上述全部检查通过时，:func:`validate` 才向标准输出打印
``Config validated: <绝对路径>``（Requirements 3.10）；任一校验失败分支均不
打印该确认信息。
"""

from __future__ import annotations

import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TextIO

from ego_pod.exit_codes import ConfigError

__all__ = ["validate"]


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

#: R3-AC5 顶层必需配置项的固定检测顺序（``output`` 不在 R3-AC5 检测范围内）。
_REQUIRED_TOP_LEVEL: tuple[str, ...] = ("sources", "training", "editions")

#: MVP 唯一支持的 Edition 名称。
_REQUIRED_EDITION = "normal"

#: MVP 唯一支持的基座模型标识。
_REQUIRED_BASE_MODEL = "qwen3.5-0.8B"

#: MVP 唯一支持的数据源类型。
_REQUIRED_SOURCE_TYPE = "local"

#: MVP 唯一支持的训练方法。
_REQUIRED_TRAINING_METHOD = "qlora"

#: R8-AC9 五个 LoRA 训练参数的检测顺序与各自约束类型。
#: ``"pos_int"`` 表示正整数；``"pos_num"`` 表示大于 0 的数值。
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

#: R1-AC3 Zone_Name 取值规则：仅由小写字母、数字与连字符组成且非空。
_ZONE_NAME_PATTERN = re.compile(r"^[a-z0-9-]+$")

#: 固定五元 Data_Category 集合；``routing`` 的键理想上属于该集合，但落在集合外
#: 的键永不命中任何样本，故仅视为无害（不致命），见 :func:`_check_zone_config`。
_DATA_CATEGORIES: frozenset[str] = frozenset(
    {"sense", "action", "thought", "skill", "credentials"}
)


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------


def _mapping_get(obj: Any, key: str) -> Any:
    """从 ``obj`` 取键 ``key``；若 ``obj`` 不是映射则返回 ``None``。"""

    if isinstance(obj, Mapping):
        return obj.get(key)
    return None


def _is_present(data: Any, key: str) -> bool:
    """判断顶层字段是否"存在"。

    一个字段被视为缺失，当且仅当其键不存在，或其值为 ``None``
    （例如 YAML 中写成空的 ``sources:``）。空映射等其他取值视为存在，
    由后续语义检查处理。
    """

    if not isinstance(data, Mapping):
        return False
    return key in data and data[key] is not None


def _is_pos_int(value: Any) -> bool:
    """是否为正整数（排除 ``bool``，因其在 Python 中是 ``int`` 的子类）。"""

    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_pos_num(value: Any) -> bool:
    """是否为大于 0 的数值（``int`` 或 ``float``，排除 ``bool``）。"""

    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value > 0
    )


def _is_iterable_seq(value: Any) -> bool:
    """是否为可按声明顺序遍历的序列（``list`` / ``tuple``）。

    用于 Zone_Config 的 ``zones.zones`` 声明列表：YAML 序列解析为 ``list``。
    字符串虽可迭代但语义上不是分区名列表，故排除；其他非序列取值（``None`` /
    映射 / 标量）一律视为"无声明分区"。
    """

    return isinstance(value, (list, tuple))


# ---------------------------------------------------------------------------
# 各校验步骤（按 AC 编号升序）
# ---------------------------------------------------------------------------


def _check_top_level_present(data: Any) -> None:
    """R3-AC5：按 sources → training → editions 顺序检测顶层缺失。

    仅报告首个被检测到的缺失字段并立即抛错，不合并多个缺失字段。
    """

    for field in _REQUIRED_TOP_LEVEL:
        if not _is_present(data, field):
            raise ConfigError(
                f"ego-config.yaml 顶层缺少必需配置项 '{field}'"
            )


def _check_normal_edition(data: Mapping[str, Any]) -> None:
    """R3-AC6：``editions`` 必须声明 ``normal`` Edition。"""

    editions = data["editions"]
    if _mapping_get(editions, _REQUIRED_EDITION) is None:
        raise ConfigError("MVP 仅支持 normal Edition：editions 中未声明 normal")


def _check_base_model(data: Mapping[str, Any]) -> None:
    """R3-AC7：``editions.normal.base`` 必须为 ``qwen3.5-0.8B``。"""

    normal = data["editions"][_REQUIRED_EDITION]
    base = _mapping_get(normal, "base")
    if base != _REQUIRED_BASE_MODEL:
        raise ConfigError(
            "MVP 仅支持 qwen3.5-0.8B 基座："
            f"editions.normal.base 当前为 {base!r}"
        )


def _check_local_sources(data: Mapping[str, Any]) -> None:
    """R3-AC8 / R9-AC2：``sources`` 中不得存在 ``type`` 非 ``local`` 的条目。

    按 ``sources`` 中的条目顺序检测，报告首个违规条目的名称及统一文案
    "MVP 仅支持 local 数据源"。该检查同时满足 R9-AC2 在配置阶段对非 local
    数据源的拒绝语义。
    """

    sources = data["sources"]
    if not isinstance(sources, Mapping):
        # 结构异常（如 sources 为列表/标量）不在 R3-AC8 的逐条目检测范围内，
        # 留待 loader 构造 EgoConfig 时按结构错误处理。
        return

    for name, entry in sources.items():
        entry_type = _mapping_get(entry, "type")
        if entry_type != _REQUIRED_SOURCE_TYPE:
            raise ConfigError(
                f"数据源条目 '{name}' 的 type 为 {entry_type!r}："
                "MVP 仅支持 local 数据源"
            )


def _check_training_method(data: Mapping[str, Any]) -> None:
    """R3-AC9：``training.method`` 必须为 ``qlora``。"""

    method = _mapping_get(data["training"], "method")
    if method != _REQUIRED_TRAINING_METHOD:
        raise ConfigError(
            "MVP 仅支持 qlora 训练方法："
            f"training.method 当前为 {method!r}"
        )


def _check_training_params(data: Mapping[str, Any]) -> None:
    """R8-AC9：五个 LoRA 训练参数的完备性与合法性预检。

    按 lora_rank → lora_alpha → learning_rate → epochs → batch_size 顺序检测，
    报告首个缺失或非法的参数名。lora_rank/lora_alpha/epochs/batch_size 必须为
    正整数，learning_rate 必须为大于 0 的数值。
    """

    training = data["training"]
    for name, kind in _TRAINING_PARAM_SPECS:
        value = _mapping_get(training, name)
        valid = _is_pos_int(value) if kind == "pos_int" else _is_pos_num(value)
        if not valid:
            raise ConfigError(
                f"训练参数 '{name}' 缺失或非法（{_CONSTRAINT_TEXT[kind]}）"
            )


def _check_zone_config(data: Any) -> None:
    """R1-AC3 / AC10 / R7-AC1：可选 ``zones`` 块的 Zone_Config 校验。

    校验作用于 ``loader`` 解析得到的原始映射中的 ``zones`` 子块，其形如::

        zones:
          zones: [coding, social, knowledge]   # 声明的 Functional_Zone 名称集合
          routing:                               # Data_Category → Zone_Name 路由基线
            skill: coding
            action: coding

    校验规则（沿用既有短路风格：检测到首个错误即抛 :class:`ConfigError`、
    退出码 2，不合并多错）：

    - **整体缺省**：``zones`` 块缺失（键不存在或值为 ``None``）⇒ 跳过全部校验，
      下游回退到内置默认基线（R1-AC8）。
    - **Zone_Name 正则**（R1-AC3）：``zones.zones`` 中每个声明的分区名须匹配
      ``^[a-z0-9-]+$``。首个违规 → ``ConfigError``，消息含该违规 Zone_Name。
    - **去重**（R1-AC10）：``zones.zones`` 内出现重复 Zone_Name → ``ConfigError``，
      消息含首个重复名。
    - **routing 取值合法性**：``routing`` 的每个值必须指向已声明的 Zone_Name；
      指向未声明分区的值 → ``ConfigError``（消息含该 routing 键与未声明的值）。
      ``routing`` 的键理想上属于 Data_Category 五元集，但落在集合外的键永不命中
      任何样本，因而无害——此处仅在标准错误打印告警，不致命。
    """

    zones_block = _mapping_get(data, "zones")
    if zones_block is None:
        # R1-AC8：未声明 Zone_Config ⇒ 跳过校验，下游采用默认基线。
        return

    declared_raw = _mapping_get(zones_block, "zones")
    declared: list[Any] = list(declared_raw) if _is_iterable_seq(declared_raw) else []

    # R1-AC3 + R1-AC10：按声明顺序逐个校验正则合法性与重复。
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

    # routing 取值合法性：每个值必须指向已声明的 Zone_Name。
    routing = _mapping_get(zones_block, "routing")
    if isinstance(routing, Mapping):
        for category, target in routing.items():
            # 键不属于 Data_Category 五元集 ⇒ 永不命中样本，仅告警不致命。
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


# ---------------------------------------------------------------------------
# 对外入口
# ---------------------------------------------------------------------------


def validate(
    data: Any,
    config_path: str | Path,
    *,
    stream: TextIO | None = None,
) -> None:
    """对原始配置映射执行 MVP 范围校验。

    校验通过时向标准输出（或 ``stream``）打印一条形如
    ``Config validated: <绝对路径>`` 的确认信息（Requirements 3.10）；任一
    校验步骤检测到违规时抛出 :class:`~ego_pod.exit_codes.ConfigError`（退出码 2）
    并**不**打印该确认信息。

    :param data: ``loader`` 由 YAML 解析得到的原始对象（正常情况下为
        :class:`~collections.abc.Mapping`）。
    :param config_path: ``ego-config.yaml`` 的路径，用于成功确认信息中的
        绝对路径展示；内部会解析为绝对路径。
    :param stream: 成功确认信息的目标输出流，默认 ``sys.stdout``。
    :raises ConfigError: 当任一校验步骤检测到违规（退出码 2）。
    """

    # 按 AC 编号升序短路：任一步骤抛出 ConfigError 即立即终止。
    _check_top_level_present(data)   # R3-AC5
    _check_normal_edition(data)      # R3-AC6
    _check_base_model(data)          # R3-AC7
    _check_local_sources(data)       # R3-AC8 / R9-AC2
    _check_training_method(data)     # R3-AC9
    _check_training_params(data)     # R8-AC9
    _check_zone_config(data)         # R1-AC3 / AC10 / R7-AC1

    # 仅在全部校验通过后打印成功确认信息（Requirements 3.10）。
    abs_path = Path(config_path).resolve()
    target = sys.stdout if stream is None else stream
    print(f"Config validated: {abs_path}", file=target, flush=True)
