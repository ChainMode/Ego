"""Compiler 按 Cell 路由的数据结构与 Zone_Config 归一（design.md §2.1）。

Cell data structures and Zone_Config resolution for the Compiler (design.md §2.1).

-----------------------------------------------------------------------------
本模块职责（What this module provides）
-----------------------------------------------------------------------------
本模块定义把人格表示为**二维稀疏网格（Sparse_Grid）**所需的纯 Python 数据结构，
并实现把 :class:`~ego_pod.config.schema.ZoneConfig`（或 ``None``）归一为可路由的
:class:`ResolvedZoneConfig` 的 :func:`resolve_zone_config`。网格的最小可寻址单元是
**Cell = (Functional_Zone, Level)**，其中 **Kernel** 是与分区无关的特殊 Level-0 Cell
（编码为 ``CellKey(zone=None, level=0)``），在任何一次加载中恒被加载。

This module defines the pure-Python data structures for the two-dimensional
Sparse_Grid personality model and resolves a :class:`ZoneConfig` (or ``None``)
into a routable :class:`ResolvedZoneConfig`. The smallest addressable unit is a
**Cell = (Functional_Zone, Level)**; the **Kernel** is the zone-agnostic Level-0
Cell, encoded as ``CellKey(zone=None, level=0)``.

实际的逐样本路由函数 ``route_cells`` 由任务 2.2 在本文件中追加实现；本文件仅提供
数据结构与 :func:`resolve_zone_config`。

The per-sample ``route_cells`` function is appended by task 2.2; this file only
provides the data structures and :func:`resolve_zone_config`.

-----------------------------------------------------------------------------
导入安全（Import safety, R6-AC8 / R7-AC2）
-----------------------------------------------------------------------------
本模块为**纯 Python、无任何机器学习依赖**，可在仅安装默认依赖的环境中安全导入。
:class:`~ego_pod.config.schema.ZoneConfig` 仅用于类型注解，故在
:data:`typing.TYPE_CHECKING` 下导入，运行时不产生依赖（``schema`` 本身亦为
import-safe，直接导入也无副作用）。

This module is pure Python with no ML dependencies and is import-safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ego_pod.compiler.classify import SampleCandidate

if TYPE_CHECKING:  # pragma: no cover - 仅供类型检查，避免运行时硬依赖。
    from ego_pod.config.schema import ZoneConfig

__all__ = [
    "CellKey",
    "KERNEL",
    "ResolvedZoneConfig",
    "CellSubDataset",
    "DEFAULT_ZONES",
    "DEFAULT_ROUTING",
    "resolve_zone_config",
    "route_cells",
]


@dataclass(frozen=True, order=True)
class CellKey:
    """Cell 的可寻址键 / Addressable key of a Cell.

    Kernel 编码为 ``(zone=None, level=0)``；分区 Cell 为 ``(zone, level>=1)``。
    ``frozen=True`` 使其可哈希（可作 dict 键）、``order=True`` 使其可稳定排序
    （Kernel 因 ``zone is None`` 排序时需借助 :data:`KERNEL` 显式置顶，本类型仅
    保证同构键之间的确定性排序）。

    Kernel is encoded as ``(zone=None, level=0)``; a zone Cell is
    ``(zone, level>=1)``. ``frozen=True`` makes it hashable (usable as a dict
    key); ``order=True`` gives a deterministic ordering between like-typed keys.
    """

    #: 功能分区名（Zone_Name）；``None`` 表示 Kernel。/ Zone name; ``None`` ⇒ Kernel.
    zone: str | None
    #: 分区内层级；Kernel 恒为 0，分区 Cell ∈ [1, max_level]。
    #: Level within the zone; Kernel is always 0, zone Cells are in [1, max_level].
    level: int

    @property
    def is_kernel(self) -> bool:
        """该 Cell 是否为 Kernel（zone 为 None 且 level == 0）。

        Whether this Cell is the Kernel (zone is None and level == 0).
        """

        return self.zone is None and self.level == 0


#: 唯一的 Kernel Cell 键（Level-0，与分区无关）。/ The unique Kernel Cell key (Level-0).
KERNEL = CellKey(zone=None, level=0)


@dataclass(frozen=True)
class ResolvedZoneConfig:
    """已解析、可路由的 Zone_Config（默认基线或用户声明归一后）。

    Resolved, routable Zone_Config (default baseline or normalized user config).

    与 :class:`~ego_pod.config.schema.ZoneConfig` 的区别：后者是配置层的声明（可能
    含指向未声明分区的冗余 routing 项），本类型是归一后的产物——``routing`` 仅保留
    **命中已声明分区**的类别，未出现的 Data_Category 在路由时一律归入 Kernel。
    """

    #: 已声明的 Functional_Zone 名称集合，保持声明顺序。
    #: Declared Functional_Zone names (declaration order preserved).
    zones: tuple[str, ...]
    #: Data_Category → Zone_Name 路由基线，仅含命中某已声明分区的类别。
    #: Data_Category → Zone_Name routing baseline; only categories that hit a zone.
    routing: dict[str, str]


@dataclass(frozen=True)
class CellSubDataset:
    """一个非空 Cell 的子数据集（供 Trainer 与网格摘要消费）。

    The sub-dataset of one non-empty Cell (consumed by the Trainer and summary).

    ``samples`` 复用 Compiler 既有的语料候选样本类型
    （:class:`~ego_pod.compiler.classify.SampleCandidate`），故 Cell_Sub_Dataset
    的 JSONL 字段与编码约定与 ego-pod-mvp Conversation_Corpus 完全一致（R2-AC5）。
    """

    #: 该子数据集所属的 Cell。/ The Cell this sub-dataset belongs to.
    cell: CellKey
    #: 被路由到该 Cell 的样本（保持输入顺序）。/ Samples routed to this Cell.
    samples: list[SampleCandidate]
    #: 相对 ``.ego/cells/`` 的 JSONL 写出路径（Kernel → ``kernel.jsonl``，
    #: 分区 → ``<zone>/L<level>.jsonl``）。
    #: JSONL write path relative to ``.ego/cells/``.
    relpath: str


#: 默认 Functional_Zone 集合（当 Ego_Config 未声明 Zone_Config 时生效）。
#: Default Functional_Zone set (used when Ego_Config declares no Zone_Config).
DEFAULT_ZONES: tuple[str, ...] = ("coding", "knowledge")

#: 默认 Data_Category → Zone_Name 路由基线。``sense`` / ``credentials`` 不在表中
#: ⇒ 归入 Kernel（R2-AC3），这天然保证 Kernel 非空（决策 B 的 kernel-always 约束）。
#: Default Data_Category → Zone_Name routing baseline. ``sense`` / ``credentials``
#: are absent ⇒ routed to the Kernel, guaranteeing a non-empty Kernel.
DEFAULT_ROUTING: dict[str, str] = {
    "skill": "coding",
    "action": "coding",
    "thought": "knowledge",
}


def resolve_zone_config(zone_config: "ZoneConfig | None") -> ResolvedZoneConfig:
    """把 Ego_Config 的 ``zones``（或 ``None``）归一为 :class:`ResolvedZoneConfig`。

    Normalize Ego_Config's ``zones`` (or ``None``) into a :class:`ResolvedZoneConfig`.

    * ``zone_config is None`` ⇒ 采用默认基线（:data:`DEFAULT_ZONES` /
      :data:`DEFAULT_ROUTING`），即用户未声明 Zone_Config 时的回退（R1-AC8）。
    * 否则 ⇒ 采用用户声明的 :class:`~ego_pod.config.schema.ZoneConfig`（R1-AC9）：
      ``zones`` 保持声明顺序；``routing`` 仅保留**值指向某已声明分区**的类别项，
      指向未声明分区的冗余项被丢弃（此类配置应已被 ``validators`` 以退出码 2 拦截，
      此处的过滤是归一阶段的防御性兜底，确保下游路由只见到合法映射）。未出现在
      归一后 ``routing`` 中的 Data_Category 在路由时一律归入 Kernel（R2-AC3）。

    本函数为纯函数，不读写 I/O、无 ML 依赖。

    :param zone_config: Ego_Config 的可选 Zone_Config 声明；``None`` ⇒ 默认基线。
    :returns: 归一后的、可供 ``route_cells`` 直接消费的 :class:`ResolvedZoneConfig`。
    """

    if zone_config is None:
        return ResolvedZoneConfig(
            zones=DEFAULT_ZONES,
            routing=dict(DEFAULT_ROUTING),
        )

    declared = tuple(zone_config.zones)
    declared_set = set(declared)
    routing = {
        category: zone
        for category, zone in zone_config.routing.items()
        if zone in declared_set
    }
    return ResolvedZoneConfig(zones=declared, routing=routing)


#: MVP 路由策略下被填充分区的唯一 Level（决策 F）。/ The only Level MVP routing
#: produces for a populated zone (decision F): Level 1.
_MVP_LEVEL: int = 1


def route_cells(
    samples: list[SampleCandidate],
    zone_config: ResolvedZoneConfig,
) -> dict[CellKey, list[SampleCandidate]]:
    """把候选样本路由到恰好一个 Cell（划分性纯函数，design.md §2.2）。

    Route each candidate sample to exactly one Cell (pure, partitioning function).

    对每条样本（**保持输入顺序**）：

    1. 依据 ``zone_config.routing`` 查 ``sample.category`` → 目标 Zone_Name；
       类别**未命中** routing（不在表中，或 ``routing`` 为空 / 未声明任何分区）
       ⇒ 归入 Kernel（R2-AC3）。
    2. 命中某已声明分区 ⇒ MVP 路由策略对每个被填充分区只产 **Level 1**
       （决策 F），故目标 Cell = ``CellKey(zone, level=1)``。
    3. 未命中 ⇒ 目标 Cell = :data:`KERNEL`。

    把样本按输入顺序追加到目标 Cell 的列表中。

    For each sample (input order preserved): look up ``sample.category`` in
    ``zone_config.routing``. A hit ⇒ target Cell ``CellKey(zone, level=1)``; a
    miss (category absent, empty routing, or no zones declared) ⇒ the
    :data:`KERNEL` Cell. The sample is appended to the target Cell's list.

    划分性不变量（R2-AC1/AC2/AC7、R7-AC3）：返回值各 Cell 列表样本数之和
    ``== len(samples)``，且每条样本**恰好**出现在一个 Cell 中——本函数不丢弃、
    不复制任何样本。

    Partitioning invariant (R2-AC1/AC2/AC7, R7-AC3): the lengths of all returned
    Cell lists sum to ``len(samples)`` and every sample appears in exactly one
    Cell — nothing is dropped or duplicated.

    当 Zone_Config 未声明任何分区（``zone_config.routing`` 为空）时，所有样本
    都归 Kernel，网格退化为仅含 Kernel 的单 Cell，其子数据集等于输入候选集合
    （R2-AC9）。

    本函数为纯函数：不读写 I/O、不产生副作用、无 ML 依赖 ⇒ 划分性可被属性测试
    直接行使（Property 1）。

    :param samples: 经 Compiler 加工后的最终候选样本列表（输入顺序有意义）。
    :param zone_config: 已归一、可路由的 :class:`ResolvedZoneConfig`。
    :returns: ``CellKey → list[SampleCandidate]`` 映射；仅含**非空** Cell
        （未被填充的 Cell 不作为键出现）。
    """

    routed: dict[CellKey, list[SampleCandidate]] = {}
    for sample in samples:
        zone = zone_config.routing.get(sample.category)
        if zone is None:
            target = KERNEL
        else:
            target = CellKey(zone=zone, level=_MVP_LEVEL)
        routed.setdefault(target, []).append(sample)
    return routed
