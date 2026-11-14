"""``route_cells`` 路由划分性的属性测试（hypothesis）。

Feature: ego-layered-adapter, Property 1

本文件实现 design.md「Correctness Properties」中的 **Property 1: route_cells
路由划分性**，对应 tasks.md 任务 2.3：

    *For any* 候选样本列表与任意 ``ResolvedZoneConfig``，``route_cells`` 返回的
    各 Cell 样本列表样本数之和等于输入样本数，且每条样本恰好出现在一个 Cell 中；
    进一步，每条样本所在的 Cell 与其 ``category`` 一致——``category`` 命中
    ``zone_config.routing`` 时落在对应分区的 Cell（``CellKey(zone, 1)``），未命中
    （或 Zone_Config 未声明任何分区）时落在 Kernel。

测试在**无任何机器学习依赖**的环境下运行：:func:`~ego_pod.compiler.routing.route_cells`
是纯函数，仅由内存中的候选样本列表与 :class:`~ego_pod.compiler.routing.ResolvedZoneConfig`
推导，导入安全（R6-AC8 / R7-AC2）。每个属性最少运行 **100 次迭代**。

策略：生成一个 :class:`~ego_pod.compiler.classify.SampleCandidate` 列表，其
``category`` 取自五元 Data_Category 集
``{sense, action, thought, skill, credentials}``，并**偶尔**取一个集合外的
类别串以行使"未命中 ⇒ Kernel"分支（R2-AC3）；其余字段为最小合法串。
:class:`ResolvedZoneConfig` 取**默认基线**（``resolve_zone_config(None)``）或一个
随机配置（``zones`` 元组 + 取值指向已声明分区的 ``routing`` 字典）。

身份追踪：``route_cells`` 不复制样本对象（仅按输入顺序 ``append`` 同一引用），
因此以 ``id(sample)`` 作为每条样本的唯一标识来核对"无丢弃、无复制、顺序保持"
（:class:`SampleCandidate` 为可变 dataclass、不可哈希，故不能直接入集合）。

**Validates: Requirements 2.1, 2.2, 2.3, 2.7, 2.9, 1.7, 7.3**
"""

from __future__ import annotations

import string
from collections import Counter

import hypothesis.strategies as st
from hypothesis import given, settings

from ego_pod.compiler.classify import SampleCandidate
from ego_pod.compiler.routing import (
    KERNEL,
    CellKey,
    ResolvedZoneConfig,
    resolve_zone_config,
    route_cells,
)

# ---------------------------------------------------------------------------
# 输入生成策略
# ---------------------------------------------------------------------------

#: 五元 Data_Category 合法集（与 classify.VALID_CATEGORIES 一致）。
_DATA_CATEGORIES = ["sense", "action", "thought", "skill", "credentials"]

#: Zone_Name：小写字母、数字、连字符（R1-AC3 的 ``^[a-z0-9-]+$``）。
_ZONE_NAME = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789-", min_size=1, max_size=8
)

#: 集合外类别串：任意小写字母串但不属于五元集——必走"未命中 ⇒ Kernel"分支。
_out_of_set_category = st.text(
    alphabet=string.ascii_lowercase, min_size=1, max_size=8
).filter(lambda s: s not in _DATA_CATEGORIES)

#: 单个样本的 ``category``：偏向取自五元集，偶尔取集合外类别串（重复 in-set 分支
#: 以提高其权重，使集合外类别仅"偶尔"出现，符合任务对生成器的要求）。
_category = st.one_of(
    st.sampled_from(_DATA_CATEGORIES),
    st.sampled_from(_DATA_CATEGORIES),
    st.sampled_from(_DATA_CATEGORIES),
    _out_of_set_category,
)


@st.composite
def sample_candidates(draw: st.DrawFn) -> SampleCandidate:
    """生成一条候选样本：随机 ``category`` + 最小合法字段串。

    ``route_cells`` 仅读取 ``sample.category``，其余字段对路由无影响，故取最小
    合法串即可；``category`` 可能是集合外类别串（类型上不严格属于 ``Category``），
    这是有意为之，用于行使"未命中 ⇒ Kernel"分支（R2-AC3）。
    """

    return SampleCandidate(
        instruction=draw(st.text(min_size=1, max_size=8)),
        input_field=draw(st.text(max_size=4)),
        output=draw(st.text(min_size=1, max_size=8)),
        category=draw(_category),  # type: ignore[arg-type]
    )


@st.composite
def random_resolved_config(draw: st.DrawFn) -> ResolvedZoneConfig:
    """合成一个随机 :class:`ResolvedZoneConfig`：``zones`` 元组 + ``routing`` 字典。

    ``routing`` 的键取自五元 Data_Category、取值取自已声明分区（保证归一后的
    不变量：routing 仅指向已声明分区）。当未声明任何分区时 ``routing`` 为空，
    用以行使 R2-AC9 的退化路径。
    """

    zones = tuple(draw(st.lists(_ZONE_NAME, min_size=0, max_size=4, unique=True)))
    if zones:
        routing = draw(
            st.dictionaries(
                keys=st.sampled_from(_DATA_CATEGORIES),
                values=st.sampled_from(zones),
                max_size=5,
            )
        )
    else:
        routing = {}
    return ResolvedZoneConfig(zones=zones, routing=routing)


#: 完整配置输入空间：默认基线（``resolve_zone_config(None)``）或随机配置。
_resolved_configs = st.one_of(
    st.builds(resolve_zone_config, st.none()),
    random_resolved_config(),
)

#: 候选样本列表（含空列表，覆盖边界）。
_sample_lists = st.lists(sample_candidates(), max_size=30)


# ===========================================================================
# Property 1a：划分性 —— 无丢弃、无复制、每条样本恰好出现一次（R2-AC7 / 7.3）
# ===========================================================================


@given(samples=_sample_lists, config=_resolved_configs)
@settings(max_examples=200, deadline=None)
def test_route_cells_partitions_samples(
    samples: list[SampleCandidate], config: ResolvedZoneConfig
) -> None:
    """各 Cell 样本数之和 == 输入样本数，且各 Cell 列表的并集（按身份计）与输入
    构成相等的多重集合——每条样本恰好出现在一个 Cell 中，既不丢弃也不复制。

    **Feature: ego-layered-adapter, Property 1**

    **Validates: Requirements 2.1, 2.2, 2.7, 7.3**
    """

    routed = route_cells(samples, config)

    # 样本数守恒：各 Cell 列表长度之和 == 输入长度（R2-AC7）。
    total = sum(len(cell_samples) for cell_samples in routed.values())
    assert total == len(samples)

    # 按对象身份核对多重集合相等：无丢弃、无复制（每个输入对象恰好出现一次）。
    input_ids = Counter(id(s) for s in samples)
    output_ids = Counter(
        id(s) for cell_samples in routed.values() for s in cell_samples
    )
    assert output_ids == input_ids

    # 仅出现非空 Cell（route_cells 仅在追加时建键）。
    for cell_samples in routed.values():
        assert len(cell_samples) >= 1


# ===========================================================================
# Property 1b：路由正确性 —— 命中分区 ⇒ CellKey(zone, 1)，未命中 ⇒ Kernel
# ===========================================================================


@given(samples=_sample_lists, config=_resolved_configs)
@settings(max_examples=200, deadline=None)
def test_route_cells_routes_by_category(
    samples: list[SampleCandidate], config: ResolvedZoneConfig
) -> None:
    """每条样本所在的 Cell 与其 ``category`` 经 ``routing`` 的查表结果一致：
    命中已声明分区 ⇒ ``CellKey(zone, 1)``（R2-AC1/AC2），未命中（不在 routing
    或 routing 为空）⇒ :data:`KERNEL`（R2-AC3）。

    **Feature: ego-layered-adapter, Property 1**

    **Validates: Requirements 2.1, 2.2, 2.3, 1.7**
    """

    routed = route_cells(samples, config)

    for cell, cell_samples in routed.items():
        for sample in cell_samples:
            zone = config.routing.get(sample.category)
            if zone is None:
                # 未命中 routing（含集合外类别 / routing 为空）⇒ Kernel。
                assert cell == KERNEL
                assert cell.is_kernel
            else:
                # 命中已声明分区 ⇒ MVP 只产 Level 1 的分区 Cell（决策 F）。
                assert cell == CellKey(zone=zone, level=1)
                assert not cell.is_kernel


# ===========================================================================
# Property 1c：无分区退化 —— 空 zones/routing ⇒ 全部归 Kernel（R2-AC9）
# ===========================================================================


@given(samples=_sample_lists)
@settings(max_examples=200, deadline=None)
def test_route_cells_degenerates_to_kernel_when_no_zones(
    samples: list[SampleCandidate],
) -> None:
    """当 :class:`ResolvedZoneConfig` 未声明任何分区（空 ``zones``/``routing``）时，
    所有样本归入 Kernel，且该 Cell 的列表（按身份与顺序）等于输入候选集合
    （空输入 ⇒ 返回空映射）。

    **Feature: ego-layered-adapter, Property 1**

    **Validates: Requirements 2.9, 1.7**
    """

    config = ResolvedZoneConfig(zones=(), routing={})
    routed = route_cells(samples, config)

    if not samples:
        # 空输入：无任何 Cell 被填充。
        assert routed == {}
        return

    # 仅有 Kernel 一个 Cell，其样本列表逐对象、逐顺序等于输入。
    assert set(routed.keys()) == {KERNEL}
    assert [id(s) for s in routed[KERNEL]] == [id(s) for s in samples]


# ===========================================================================
# Property 1d：Cell 内顺序保持 —— 同一 Cell 内样本相对顺序与输入一致
# ===========================================================================


@given(samples=_sample_lists, config=_resolved_configs)
@settings(max_examples=200, deadline=None)
def test_route_cells_preserves_order_within_cell(
    samples: list[SampleCandidate], config: ResolvedZoneConfig
) -> None:
    """每个 Cell 内的样本按其在输入中的相对顺序出现（``route_cells`` 按输入顺序
    追加）。以输入位置索引核对：每个 Cell 列表对应的输入索引序列严格递增。

    **Feature: ego-layered-adapter, Property 1**

    **Validates: Requirements 2.7, 7.3**
    """

    routed = route_cells(samples, config)

    # 输入对象 ⇒ 其在输入列表中的位置（对象身份唯一，故索引唯一）。
    input_index = {id(sample): i for i, sample in enumerate(samples)}

    for cell_samples in routed.values():
        indices = [input_index[id(sample)] for sample in cell_samples]
        assert indices == sorted(indices)
