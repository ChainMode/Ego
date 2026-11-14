"""Ego_Config 的内存数据结构定义。

本模块仅定义 Ego_Config 的核心 dataclass，与设计文档 §2（Config 模块）
保持字段一一对应。所有 dataclass 均为 ``frozen=True``，因此实例创建后
其字段不可被重新赋值，保证配置在加载、校验、流水线传递过程中的不可变性。

加载（YAML → dataclass）、默认配置渲染、范围校验分别由 ``loader.py``、
``defaults.py``、``validators.py`` 实现，不在本模块职责范围内。
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SourceEntry:
    """单个 Data_Category 的数据源条目。

    MVP 仅支持 ``type: local`` 的本地目录数据源。``path`` 为绝对路径或
    相对于 Ego_Pod 进程工作目录的路径。
    """

    type: Literal["local"]
    path: str


@dataclass(frozen=True)
class TrainingParams:
    """训练参数。

    MVP 固定 ``method=qlora``、``quantization=4bit``；五个数值参数均需满足
    各自的取值约束（lora_rank / lora_alpha / epochs / batch_size 为正整数，
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
    """单个 Edition 的定义。

    MVP 仅支持名为 ``normal`` 的 Edition，其 ``base`` 固定为 ``qwen3.5-0.8B``。
    """

    base: Literal["qwen3.5-0.8B"]
    description: str = ""


@dataclass(frozen=True)
class ZoneConfig:
    """Zone_Config：功能分区分类法与 Data_Category → Functional_Zone 路由基线。

    Zone_Config：functional-zone taxonomy and the Data_Category → Zone routing baseline.

    -------------------------------------------------------------------------
    与 Data_Category 集合解耦（DECOUPLED from the Data_Category set）
    -------------------------------------------------------------------------
    ``zones`` 是人格功能分区（Functional_Zone）的命名集合，与五元 Data_Category
    集合（sense / action / thought / skill / credentials）**相互独立**：分区名由
    本配置自由声明，并非从 Data_Category 派生。``routing`` 仅把语料类别映射到分区名，
    起到"桥接"作用——**未在 ``routing`` 中出现的类别一律路由到 Kernel**（Level 0，
    与分区无关的共享基础）。

    ``zones`` is the named set of personality Functional_Zones; it is decoupled
    from the fixed Data_Category set. ``routing`` only bridges corpus categories
    to zone names; any category absent from ``routing`` routes to the Kernel.
    """

    #: 声明的 Functional_Zone 名称集合，保持声明顺序；每个须匹配 Zone_Name 正则且唯一
    #: （合法性/去重由 validators.py 校验）。
    #: Declared Functional_Zone names (declaration order preserved; each must be unique).
    zones: tuple[str, ...]
    #: Data_Category → Zone_Name 的路由基线；缺省键的类别路由到 Kernel。
    #: Data_Category → Zone_Name routing baseline; categories not present route to Kernel.
    routing: dict[str, str]


@dataclass(frozen=True)
class EgoConfig:
    """Ego_Config 的完整内存表示，对应工作目录下的 ``ego-config.yaml``。

    - ``sources``：键限定为 sense / action / thought / skill / credentials。
    - ``training``：训练参数。
    - ``editions``：键必须包含 ``normal``。
    - ``output``：MVP 暂仅占位。
    - ``zones``：可选的全局 Zone_Config，缺省（``None``）⇒ 下游回退到默认基线。
    """

    sources: dict[str, SourceEntry]
    training: TrainingParams
    editions: dict[str, EditionDef]
    output: dict
    zones: "ZoneConfig | None" = None
