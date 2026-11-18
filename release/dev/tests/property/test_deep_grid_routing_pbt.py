"""深层多维稀疏网格路由与按 Edition 裁剪的 Hypothesis 属性测试（ADR 0005）。

验证以下核心数学不变量：
1. 多层级划分性（Partitioning）：对任意样本序列与任意 max_level ∈ [1, 5]，
   各 Cell 样本数之和恒等于输入样本总数，且对象身份多重集合完全相等（不丢不重）；
2. 层级连续无跳号（Contiguous Levels 1..K）：任意非空分区的出现层级集合恒等于
   {1, 2, ..., K}（其中 1 <= K <= max_level）；
3. 路由确定性（Determinism）：相同输入与拟合集重复调用产出完全一致的划分；
4. 按 Edition 裁剪守恒与连续性：对任意已路由网格按任意 edition_max_level 裁剪后，
   样本计数恒守恒、Kernel 恒保留、各分区层级恒连续且 <= edition_max_level。
"""

from __future__ import annotations

from collections import Counter

import hypothesis.strategies as st
from hypothesis import given, settings

from ego_pod.compiler.classify import SampleCandidate
from ego_pod.compiler.routing import (
    KERNEL,
    ResolvedZoneConfig,
    prune_cells_for_edition,
    route_cells,
)

_CATEGORIES = ["sense", "action", "thought", "skill", "credentials"]
_ZONE_NAME = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789-", min_size=1, max_size=8
)


@st.composite
def _varied_samples(draw: st.DrawFn) -> SampleCandidate:
    cat = draw(st.sampled_from(_CATEGORIES))
    tier = draw(st.integers(min_value=0, max_value=3))
    if tier == 0:
        out = draw(st.text(max_size=12))
    elif tier == 1:
        out = "日常简短记录 " + draw(st.text(min_size=4, max_size=25))
    elif tier == 2:
        out = (
            "因为模块边界需要解耦，因此采用配置驱动校验：param="
            + draw(st.text(min_size=5, max_size=40))
        )
    else:
        out = (
            "```python\ndef fn_x(a: int) -> int:\n    return a * 2\n```\n"
            "复杂度分析与不变量证明："
            + draw(st.text(min_size=10, max_size=60))
        )
    return SampleCandidate(
        instruction="指令",
        input_field="",
        output=out,
        category=cat,  # type: ignore[arg-type]
    )


@st.composite
def _deep_configs(draw: st.DrawFn) -> ResolvedZoneConfig:
    zones = tuple(draw(st.lists(_ZONE_NAME, min_size=0, max_size=4, unique=True)))
    max_level = draw(st.integers(min_value=1, max_value=5))
    if zones:
        routing = draw(
            st.dictionaries(
                keys=st.sampled_from(_CATEGORIES),
                values=st.sampled_from(zones),
                max_size=5,
            )
        )
    else:
        routing = {}
    return ResolvedZoneConfig(zones=zones, routing=routing, max_level=max_level)


@given(
    samples=st.lists(_varied_samples(), max_size=35),
    config=_deep_configs(),
)
@settings(max_examples=150, deadline=None)
def test_deep_route_cells_partition_and_contiguous_levels(
    samples: list[SampleCandidate],
    config: ResolvedZoneConfig,
) -> None:
    routed = route_cells(samples, config)

    # 1. 样本计数守恒与身份多重集一致
    assert sum(len(v) for v in routed.values()) == len(samples)
    assert Counter(id(s) for v in routed.values() for s in v) == Counter(
        id(s) for s in samples
    )

    # 2. 每个非空分区的 Level 必须从 1 起严格连续且不超过 config.max_level
    by_zone: dict[str, list[int]] = {}
    for cell_key, cell_samples in routed.items():
        assert len(cell_samples) >= 1
        if cell_key.is_kernel:
            assert cell_key == KERNEL
        else:
            assert cell_key.zone in config.zones
            by_zone.setdefault(cell_key.zone, []).append(cell_key.level)

    for _zone, levels in by_zone.items():
        ordered = sorted(levels)
        assert ordered == list(range(1, len(ordered) + 1))
        assert ordered[-1] <= config.max_level

    # 3. 路由确定性：二次调用完全一致
    routed_again = route_cells(samples, config)
    assert routed == routed_again


@given(
    samples=st.lists(_varied_samples(), min_size=1, max_size=30),
    config=_deep_configs(),
    edition_max_level=st.integers(min_value=1, max_value=4),
)
@settings(max_examples=120, deadline=None)
def test_prune_cells_for_edition_invariants(
    samples: list[SampleCandidate],
    config: ResolvedZoneConfig,
    edition_max_level: int,
) -> None:
    routed = route_cells(samples, config)
    subset_zones = config.zones[:1] if len(config.zones) >= 2 else None

    pruned = prune_cells_for_edition(
        routed,
        max_level=edition_max_level,
        allowed_zones=subset_zones,
    )

    assert sum(len(v) for v in pruned.values()) == len(samples)
    assert Counter(id(s) for v in pruned.values() for s in v) == Counter(
        id(s) for s in samples
    )

    by_zone: dict[str, list[int]] = {}
    for key, vals in pruned.items():
        assert len(vals) >= 1
        if not key.is_kernel:
            if subset_zones is not None:
                assert key.zone in subset_zones
            by_zone.setdefault(key.zone, []).append(key.level)

    for _zone, levels in by_zone.items():
        ordered = sorted(levels)
        assert ordered == list(range(1, len(ordered) + 1))
        assert ordered[-1] <= edition_max_level
