"""Ego_Config 的内存数据结构与多规格预设定义。

本模块定义 Ego_Config 的核心不可变 dataclass，以及 v0.2 修炼场（Dojo）支持的
受控基座模型白名单 :data:`SUPPORTED_BASE_MODELS` 与四档内置规格预设
:data:`EDITION_PRESETS`（ADR 0006）。

所有 dataclass 均为 ``frozen=True``，保证配置在加载、校验、流水线传递过程中的
不可变性。加载、默认渲染、范围校验分别由 ``loader.py``、``defaults.py``、
``validators.py`` 实现。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

#: 受控基座模型白名单（ADR 0006 §2）：覆盖 Qwen 2.5/3.5、Gemma 4 与 Phi-4 家族。
SUPPORTED_BASE_MODELS: tuple[str, ...] = (
    "qwen2.5-0.5B",
    "qwen3.5-0.8B",
    "qwen3.5-2B",
    "qwen3.5-4B",
    "qwen3.5-9B",
    "qwen3.5-27B",
    "gemma4-2B",
    "gemma4-9B",
    "gemma4-27B",
    "phi4-mini",
    "phi4-14B",
)


@dataclass(frozen=True)
class EditionPreset:
    """内置 Edition 规格预设（ADR 0006 §1）。"""

    name: str
    default_base: str
    lora_rank: int
    lora_alpha: int
    max_level: int
    max_size_mb: float


#: 四档标准 Edition 预设：lite / normal / plus / super。
EDITION_PRESETS: dict[str, EditionPreset] = {
    "lite": EditionPreset(
        name="lite",
        default_base="qwen3.5-0.8B",
        lora_rank=8,
        lora_alpha=16,
        max_level=1,
        max_size_mb=64.0,
    ),
    "normal": EditionPreset(
        name="normal",
        default_base="qwen3.5-2B",
        lora_rank=16,
        lora_alpha=32,
        max_level=2,
        max_size_mb=256.0,
    ),
    "plus": EditionPreset(
        name="plus",
        default_base="qwen3.5-4B",
        lora_rank=32,
        lora_alpha=64,
        max_level=3,
        max_size_mb=1024.0,
    ),
    "super": EditionPreset(
        name="super",
        default_base="qwen3.5-27B",
        lora_rank=64,
        lora_alpha=128,
        max_level=3,
        max_size_mb=4096.0,
    ),
}

#: 自定义 Edition 未显式指定时的默认网格最大深度与体积上限（MB）。
CUSTOM_EDITION_DEFAULT_MAX_LEVEL = 3
CUSTOM_EDITION_DEFAULT_MAX_SIZE_MB = 4096.0


@dataclass(frozen=True)
class SourceEntry:
    """单个 Data_Category 的数据源条目。

    支持内置 ``local`` / ``url`` / ``s3`` / ``oss`` 数据源以及通过插件注册表扩展的
    自定义数据源类型。云端凭证严禁出现在本结构或 YAML 中（仅从环境变量读取）。
    """

    type: str
    path: str = ""
    url: str | None = None
    bucket: str | None = None
    prefix: str = ""
    allowed_hosts: tuple[str, ...] = ()
    options: dict[str, Any] | None = None


@dataclass(frozen=True)
class TrainingParams:
    """训练参数。

    固定 ``method=qlora``、``quantization=4bit``；五个数值参数均需满足各自的
    取值约束（lora_rank / lora_alpha / epochs / batch_size 为正整数，
    learning_rate 大于 0），具体校验由 ``validators.py`` 负责。
    """

    method: Literal["qlora"]
    quantization: Literal["4bit"]
    lora_rank: int
    lora_alpha: int
    learning_rate: float
    epochs: int
    batch_size: int


@dataclass(frozen=True)
class EditionDef:
    """单个 Edition 的定义（支持预设规格与自定义规格覆盖）。"""

    base: str
    description: str = ""
    lora_rank: int | None = None
    lora_alpha: int | None = None
    max_level: int | None = None
    max_size_mb: float | None = None
    zones: tuple[str, ...] | None = None


@dataclass(frozen=True)
class ResolvedEditionSpec:
    """合并预设默认值、全局训练参数与单规格覆盖后的完整 Edition 规格。"""

    name: str
    base: str
    description: str
    training: TrainingParams
    max_level: int
    max_size_mb: float
    zones: tuple[str, ...] | None = None


def resolve_edition_spec(
    name: str,
    edition_def: EditionDef,
    global_training: TrainingParams,
) -> ResolvedEditionSpec:
    """将 ``EditionDef`` 与内置预设及全局 ``TrainingParams`` 合并为可执行规格。

    优先级：``edition_def`` 显式声明字段 > 全局 ``training``（针对 rank/alpha，若非预设覆盖）
    / ``EDITION_PRESETS[name]`` > 自定义规格兜底默认值。
    注：对于非 ``normal`` 的内置预设（如 ``lite`` / ``plus`` / ``super``），若单规格未显式
    指定 ``lora_rank`` / ``lora_alpha``，优先使用该档位的预设秩以体现档位差异；而 ``normal``
    及自定义规格在未显式指定单规格 ``lora_rank`` / ``lora_alpha`` 时沿用全局 ``training``
    中的 ``lora_rank`` / ``lora_alpha``，保证既有配置行为 100% 向后兼容。
    """

    preset = EDITION_PRESETS.get(name)
    if edition_def.lora_rank is not None:
        rank = edition_def.lora_rank
    elif preset is not None and name != "normal":
        rank = preset.lora_rank
    else:
        rank = global_training.lora_rank

    if edition_def.lora_alpha is not None:
        alpha = edition_def.lora_alpha
    elif preset is not None and name != "normal":
        alpha = preset.lora_alpha
    else:
        alpha = global_training.lora_alpha

    if edition_def.max_level is not None:
        max_level = edition_def.max_level
    elif preset is not None:
        max_level = preset.max_level
    else:
        max_level = CUSTOM_EDITION_DEFAULT_MAX_LEVEL

    if edition_def.max_size_mb is not None:
        max_size_mb = float(edition_def.max_size_mb)
    elif preset is not None:
        max_size_mb = preset.max_size_mb
    else:
        max_size_mb = CUSTOM_EDITION_DEFAULT_MAX_SIZE_MB

    eff_training = replace(global_training, lora_rank=rank, lora_alpha=alpha)
    return ResolvedEditionSpec(
        name=name,
        base=edition_def.base,
        description=edition_def.description,
        training=eff_training,
        max_level=max_level,
        max_size_mb=max_size_mb,
        zones=edition_def.zones,
    )


@dataclass(frozen=True)
class ZoneConfig:
    """Zone_Config：功能分区分类法与 Data_Category → Functional_Zone 路由基线。

    ``zones`` 是人格功能分区（Functional_Zone）的命名集合，与五元 Data_Category
    集合相互独立；未在 ``routing`` 中出现的类别一律路由到 Kernel（Level 0）。
    ``max_level`` 控制直接调用路由时的最大深度上限（Python 直接构造默认 1 以保持
    向后兼容；``ego train`` 管线按所配置 Edition 的最大深度驱动深层 ``L1..L3`` 路由）。
    """

    zones: tuple[str, ...]
    routing: dict[str, str]
    max_level: int = 1


@dataclass(frozen=True)
class EgoConfig:
    """Ego_Config 的完整内存表示，对应工作目录下的 ``ego-config.yaml``。"""

    sources: dict[str, SourceEntry]
    training: TrainingParams
    editions: dict[str, EditionDef]
    output: dict[str, Any]
    zones: ZoneConfig | None = None
