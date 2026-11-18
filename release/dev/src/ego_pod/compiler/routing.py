"""Compiler 按 Cell 路由的数据结构、深层专精度分层与按 Edition 网格裁剪（ADR 0005）。

本模块定义二维稀疏网格（Sparse_Grid）的纯 Python 数据结构、:func:`resolve_zone_config`、
支持 ``L1..L3+`` 深层专精度分层的 :func:`route_cells`，以及供多 Edition 构建复用的
确定性网格裁剪函数 :func:`prune_cells_for_edition`。

导入安全（Import safety, R6-AC8 / R7-AC2）：
本模块为纯 Python、无任何机器学习依赖，可在仅安装默认依赖的环境中安全导入。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ego_pod.compiler.classify import SampleCandidate
from ego_pod.compiler.specificity import (
    NullFieldStats,
    SpecificityModel,
    assign_zone_levels,
    fit_specificity_model,
)

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
    "prune_cells_for_edition",
]


@dataclass(frozen=True, order=True)
class CellKey:
    """Cell 的可寻址键 / Addressable key of a Cell.

    Kernel 编码为 ``(zone=None, level=0)``；分区 Cell 为 ``(zone, level>=1)``。
    """

    zone: str | None
    level: int

    @property
    def is_kernel(self) -> bool:
        """该 Cell 是否为 Kernel（zone 为 None 且 level == 0）。"""

        return self.zone is None and self.level == 0


#: 唯一的 Kernel Cell 键（Level-0，与分区无关）。
KERNEL = CellKey(zone=None, level=0)


@dataclass(frozen=True)
class ResolvedZoneConfig:
    """已解析、可路由的 Zone_Config（默认基线或用户声明归一后）。"""

    zones: tuple[str, ...]
    routing: dict[str, str]
    max_level: int = 1


@dataclass(frozen=True)
class CellSubDataset:
    """一个非空 Cell 的子数据集（供 Trainer 与网格摘要消费）。"""

    cell: CellKey
    samples: list[SampleCandidate]
    relpath: str


#: 默认 Functional_Zone 集合（当 Ego_Config 未声明 Zone_Config 时生效）。
DEFAULT_ZONES: tuple[str, ...] = ("coding", "knowledge")

#: 默认 Data_Category → Zone_Name 路由基线。``sense`` / ``credentials`` 不在表中
#: ⇒ 归入 Kernel（R2-AC3）。
DEFAULT_ROUTING: dict[str, str] = {
    "skill": "coding",
    "action": "coding",
    "thought": "knowledge",
}


def resolve_zone_config(
    zone_config: "ZoneConfig | None",
    *,
    default_max_level: int = 1,
) -> ResolvedZoneConfig:
    """把 Ego_Config 的 ``zones``（或 ``None``）归一为 :class:`ResolvedZoneConfig`。

    * ``zone_config is None`` ⇒ 采用默认基线（:data:`DEFAULT_ZONES` /
      :data:`DEFAULT_ROUTING`），``max_level`` 取 ``max(1, default_max_level)``。
    * ``zone_config`` 非空 ⇒ 保留用户声明顺序与有效映射；若用户在 ``zone_config`` 上显式
      设定了 ``max_level > 1`` 则采用用户值，否则回退到 ``max(zone_config.max_level, default_max_level)``。
    """

    eff_default = max(1, int(default_max_level))
    if zone_config is None:
        return ResolvedZoneConfig(
            zones=DEFAULT_ZONES,
            routing=dict(DEFAULT_ROUTING),
            max_level=eff_default,
        )

    declared = tuple(zone_config.zones)
    declared_set = set(declared)
    routing = {
        category: zone
        for category, zone in zone_config.routing.items()
        if zone in declared_set
    }
    cfg_max = getattr(zone_config, "max_level", 1)
    resolved_max = max(1, int(cfg_max) if cfg_max != 1 else eff_default)
    return ResolvedZoneConfig(
        zones=declared,
        routing=routing,
        max_level=resolved_max,
    )


def route_cells(
    samples: list[SampleCandidate],
    zone_config: ResolvedZoneConfig,
    *,
    max_level: int | None = None,
    fit_samples: Sequence[SampleCandidate] | None = None,
    specificity_model: SpecificityModel | None = None,
    null_stats: NullFieldStats | None = None,
) -> dict[CellKey, list[SampleCandidate]]:
    """把候选样本路由到恰好一个 Cell（划分性纯函数，支持 ``L1..max_level`` 深层分层）。

    - 当有效 ``max_level == 1`` 时，保持单层快速路径（所有命中分区的样本落入 ``L1``），
      与 MVP 基线行为 100% 兼容。
    - 当有效 ``max_level >= 2`` 时，调用 :mod:`~ego_pod.compiler.specificity` 计算分区内
      专精度得分、分位数分层并经 ``compact_levels`` 保序压缩为无跳号的连续 ``1..K``。
      若提供 ``specificity_model`` 或 ``fit_samples``（例如留出集切分后的纯训练子集），
      统计特征严格仅在训练子集上拟合，杜绝验证集数据泄漏。
    """

    eff_max_level = max(
        1, zone_config.max_level if max_level is None else int(max_level)
    )
    if not samples:
        return {}

    if eff_max_level == 1:
        routed_flat: dict[CellKey, list[SampleCandidate]] = {}
        for sample in samples:
            zone = zone_config.routing.get(sample.category)
            target = KERNEL if zone is None else CellKey(zone=zone, level=1)
            routed_flat.setdefault(target, []).append(sample)
        return routed_flat

    model = specificity_model
    if model is None:
        training_basis = samples if fit_samples is None else fit_samples
        model = fit_specificity_model(
            training_basis,
            zone_config.routing,
            max_level=eff_max_level,
            stats=null_stats if fit_samples is not None else None,
        )

    kernel_bucket: list[SampleCandidate] = []
    zone_groups: dict[str, list[SampleCandidate]] = {}
    zone_order: list[str] = []

    for sample in samples:
        zone = zone_config.routing.get(sample.category)
        if zone is None:
            kernel_bucket.append(sample)
        else:
            if zone not in zone_groups:
                zone_groups[zone] = []
                zone_order.append(zone)
            zone_groups[zone].append(sample)

    routed: dict[CellKey, list[SampleCandidate]] = {}
    if kernel_bucket:
        routed[KERNEL] = kernel_bucket

    for zone in zone_order:
        grp = zone_groups[zone]
        levels = assign_zone_levels(
            grp,
            zone,
            model,
            max_level=eff_max_level,
            stats=null_stats if fit_samples is None else None,
        )
        for sample, lvl in zip(grp, levels):
            routed.setdefault(CellKey(zone=zone, level=lvl), []).append(sample)

    return routed


def prune_cells_for_edition(
    routed: Mapping[CellKey, list[SampleCandidate]],
    *,
    max_level: int,
    allowed_zones: Sequence[str] | None = None,
) -> dict[CellKey, list[SampleCandidate]]:
    """按目标 Edition 的 ``max_level`` 与可选 ``allowed_zones`` 子集确定性裁剪网格。

    1. 未包含在 ``allowed_zones`` 中的分区样本安全折叠回 :data:`KERNEL`，确保样本不丢失；
    2. 超过 ``max_level`` 的深层样本确定性折叠至 ``max_level``（例如 ``L3 → L2``）；
    3. 每个保留分区的出现层级保序重编号为严格连续的 ``1..K``（``K <= max_level``）。
    """

    eff_max = max(1, int(max_level))
    allowed_set = set(allowed_zones) if allowed_zones is not None else None

    kernel_samples: list[SampleCandidate] = []
    zone_buckets: dict[str, dict[int, list[SampleCandidate]]] = {}
    zone_order: list[str] = []

    for cell_key, cell_samples in routed.items():
        if not cell_samples:
            continue
        if cell_key.is_kernel or cell_key.zone is None or (
            allowed_set is not None and cell_key.zone not in allowed_set
        ):
            kernel_samples.extend(cell_samples)
        else:
            zone = cell_key.zone
            if zone not in zone_buckets:
                zone_buckets[zone] = {}
                zone_order.append(zone)
            capped = min(eff_max, max(1, cell_key.level))
            zone_buckets[zone].setdefault(capped, []).extend(cell_samples)

    pruned: dict[CellKey, list[SampleCandidate]] = {}
    if kernel_samples:
        pruned[KERNEL] = kernel_samples

    for zone in zone_order:
        lvl_map = zone_buckets[zone]
        for new_idx, old_lvl in enumerate(sorted(lvl_map.keys()), start=1):
            pruned[CellKey(zone=zone, level=new_idx)] = lvl_map[old_lvl]

    return pruned
