"""``select_cells`` 层级前缀属性的属性测试。

本文件实现 design.md「Correctness Properties」中的 **Property 3**，对应
tasks.md 任务 5.2：

* **Property 3: select_cells 层级前缀属性** —— *For any* 内存 :class:`Grid`
  与任意合法 :class:`LoadSelection`，:func:`~ego_pod.runtime.selection.select_cells`
  满足：(a) 结果恒包含 Kernel 且排在最前（kernel-always）；(b) 对每个被选分区
  选定深度 N，选出的恰为该分区 Level ``1..N`` 的连续前缀；(c) 当
  ``selection is None`` 或每个分区都选定其最大 Level 时，结果等于 Full_In_Load
  的 Cell 集合；(d) 结果顺序为 Kernel 优先、再各分区（按分区名稳定排序）Level
  由低到高。此外，*for any* 越界深度（``<1`` 或 ``>max_level``）或未知分区名，
  ``select_cells`` 抛出 :class:`~ego_pod.exit_codes.RuntimeError_`（退出码 1）。

  **Feature: ego-layered-adapter, Property 3**

  **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.6, 7.5**

测试在**无任何机器学习依赖**的环境下运行：:func:`select_cells` 是纯函数，仅
由内存 :class:`Grid` 与 :class:`LoadSelection` 推导，导入安全（决策 D /
R6-AC8 / R7-AC2）。每个属性最少运行 **100 次迭代**。

策略：合成一张 :class:`Grid` —— 一个 Kernel :class:`GridCell` 加上一个随机的
分区字典，每个分区含**连续 ``1..max``**（``max`` 为小的随机数）的
:class:`GridCell`，相对路径为合成串
``adapters/<zone>/L<level>.safetensors``，Cell 键经 ``CellKey(zone, level)``
构造。随机的合法 :class:`LoadSelection` 为每个被选分区取 ``1..max_level`` 的
深度；越界/未知分区生成器覆盖 R5-AC6 的失败分支。
"""

from __future__ import annotations

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings

from ego_pod.exit_codes import EXIT_RUNTIME_FAILURE, RuntimeError_
from ego_pod.runtime.selection import (
    KERNEL,
    CellKey,
    Grid,
    GridCell,
    LoadSelection,
    select_cells,
)

# ---------------------------------------------------------------------------
# 输入生成策略
# ---------------------------------------------------------------------------

#: Zone_Name：小写字母、数字、连字符（R1-AC3 的 ``^[a-z0-9-]+$``）。
_ZONE_NAME = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789-", min_size=1, max_size=8
)

#: 每个分区的最大 Level（小的随机数，保证连续 ``1..max`` 且分区非空）。
_MAX_LEVEL = st.integers(min_value=1, max_value=5)


def _kernel_cell() -> GridCell:
    """规范 Kernel GridCell（相对路径 ``adapters/kernel.safetensors``）。"""
    return GridCell(KERNEL, "adapters/kernel.safetensors")


def _build_grid(zone_levels: dict[str, int]) -> Grid:
    """据 ``{zone: max_level}`` 构造一张内存 :class:`Grid`。

    每个分区含 Level ``1..max_level`` 连续的 :class:`GridCell`，相对路径为合成串
    ``adapters/<zone>/L<level>.safetensors``，与设计的磁盘命名约定一致。
    """
    zones: dict[str, list[GridCell]] = {}
    for name, max_level in zone_levels.items():
        zones[name] = [
            GridCell(
                CellKey(name, level),
                f"adapters/{name}/L{level}.safetensors",
            )
            for level in range(1, max_level + 1)
        ]
    return Grid(kernel=_kernel_cell(), zones=zones)


@st.composite
def grids(draw, *, min_zones: int = 0, max_zones: int = 4) -> Grid:
    """合成一张 :class:`Grid`：Kernel + 随机分区（每分区连续 ``1..max`` Level）。"""
    zone_levels = draw(
        st.dictionaries(
            _ZONE_NAME, _MAX_LEVEL, min_size=min_zones, max_size=max_zones
        )
    )
    return _build_grid(zone_levels)


@st.composite
def grid_and_valid_selection(draw) -> tuple[Grid, LoadSelection]:
    """合成 (Grid, 合法 LoadSelection)：被选分区子集，每个深度取 ``1..max_level``。"""
    grid = draw(grids(min_zones=0, max_zones=4))
    zone_names = list(grid.zones)
    if zone_names:
        chosen = draw(st.lists(st.sampled_from(zone_names), unique=True))
    else:
        chosen = []
    depths = {zone: draw(st.integers(min_value=1, max_value=grid.max_level(zone))) for zone in chosen}
    return grid, LoadSelection(depths=depths)


@st.composite
def grid_and_low_depth(draw) -> tuple[Grid, LoadSelection]:
    """合成 (Grid, 含某分区深度 ``<1`` 的非法 LoadSelection)。"""
    grid = draw(grids(min_zones=1, max_zones=4))
    zone = draw(st.sampled_from(list(grid.zones)))
    depth = draw(st.integers(max_value=0))  # 深度 <1，非法（R5-AC6）。
    return grid, LoadSelection(depths={zone: depth})


@st.composite
def grid_and_high_depth(draw) -> tuple[Grid, LoadSelection]:
    """合成 (Grid, 含某分区深度 ``>max_level`` 的非法 LoadSelection)。"""
    grid = draw(grids(min_zones=1, max_zones=4))
    zone = draw(st.sampled_from(list(grid.zones)))
    max_level = grid.max_level(zone)
    depth = draw(st.integers(min_value=max_level + 1, max_value=max_level + 20))
    return grid, LoadSelection(depths={zone: depth})


@st.composite
def grid_and_unknown_zone(draw) -> tuple[Grid, LoadSelection]:
    """合成 (Grid, 命名了网格未列出分区的非法 LoadSelection)。"""
    grid = draw(grids(min_zones=0, max_zones=4))
    unknown = draw(_ZONE_NAME.filter(lambda n: n not in grid.zones))
    depth = draw(st.integers(min_value=1, max_value=5))
    return grid, LoadSelection(depths={unknown: depth})


def _full_in_load_expected(grid: Grid) -> list[CellKey]:
    """Full_In_Load 的期望有序 Cell 列表：Kernel + 各分区（名序）Level 升序全部。"""
    expected: list[CellKey] = [KERNEL]
    for zone in sorted(grid.zones):
        for grid_cell in grid.zones[zone]:
            expected.append(grid_cell.cell)
    return expected


# ===========================================================================
# Property 3a：Full_In_Load（selection is None） —— Kernel 在前 + 全部 Cell
# ===========================================================================


@given(grid=grids(min_zones=0, max_zones=4))
@settings(max_examples=200, deadline=None)
def test_full_in_load_returns_kernel_first_then_all_cells(grid: Grid) -> None:
    """``select_cells(grid, None)`` 返回 Kernel 在首位，再每分区全部 Level。

    断言：(a) 结果首元素为 Kernel 且 ``is_kernel``（R5-AC1/AC3）；(b) Kernel
    恒在结果中；(c) 结果恰等于 ``Kernel + 每个分区（名序）Level 升序的全部
    Cell``，即 Full_In_Load 包含网格中每一个 Cell（R5-AC1/AC4）。

    **Feature: ego-layered-adapter, Property 3**

    **Validates: Requirements 5.1, 5.3, 5.4, 7.5**
    """
    result = select_cells(grid, None)

    # (a) Kernel 恒含且排最前（R5-AC3）。
    assert result[0] == KERNEL
    assert result[0].is_kernel
    assert KERNEL in result
    assert result.count(KERNEL) == 1

    # (c) result == kernel + 网格中每一个 Cell（按名序、Level 升序，R5-AC1/AC4）。
    assert result == _full_in_load_expected(grid)

    # 集合等价：选出的 == Kernel ∪ 网格全部 Cell（无遗漏、无多余）。
    all_cells = {KERNEL}
    for cells in grid.zones.values():
        all_cells.update(grid_cell.cell for grid_cell in cells)
    assert set(result) == all_cells


# ===========================================================================
# Property 3b：Nested_In_Load —— 每分区 Level 1..depth 连续前缀 + 顺序稳定
# ===========================================================================


@given(data=grid_and_valid_selection())
@settings(max_examples=200, deadline=None)
def test_valid_selection_is_level_prefix_per_zone(
    data: tuple[Grid, LoadSelection],
) -> None:
    """合法 selection ⇒ 每个被选分区选出 Level ``1..depth`` 的连续前缀。

    断言：(a) Kernel 在首位且恰出现一次（R5-AC3）；(b) 对每个被选分区深度 N，
    选出的恰为 Level ``1..N`` 连续前缀（R5-AC2、R7-AC5）；(c) 结果顺序为 Kernel
    优先、再各分区按名序、分区内 Level 升序（R5-AC4）；(d) 未被选中的分区不贡献
    任何 Cell（R5-AC2）。

    **Feature: ego-layered-adapter, Property 3**

    **Validates: Requirements 5.2, 5.3, 5.4, 7.5**
    """
    grid, selection = data

    result = select_cells(grid, selection)

    # (a) Kernel 恒含且排最前、恰一次（R5-AC3）。
    assert result[0] == KERNEL
    assert result.count(KERNEL) == 1

    # (b)+(c) 结果恰等于 Kernel + 各被选分区（名序）Level 1..depth 前缀。
    expected: list[CellKey] = [KERNEL]
    for zone in sorted(selection.depths):
        depth = selection.depths[zone]
        for level in range(1, depth + 1):
            expected.append(CellKey(zone, level))
    assert result == expected

    # (b) 逐分区前缀属性：该分区被选出的 Level 序列恰为 1..depth（R7-AC5）。
    for zone, depth in selection.depths.items():
        zone_levels = [c.level for c in result if c.zone == zone]
        assert zone_levels == list(range(1, depth + 1))

    # (d) 未被选中的分区不贡献任何 Cell（R5-AC2）。
    selected_zones = set(selection.depths)
    for cell in result:
        if not cell.is_kernel:
            assert cell.zone in selected_zones

    # (c) 非 Kernel 部分的分区出现顺序为分区名稳定（字典序）排序（R5-AC4）。
    seen_zones: list[str] = []
    for cell in result:
        if not cell.is_kernel and cell.zone not in seen_zones:
            seen_zones.append(cell.zone)
    assert seen_zones == sorted(z for z in selection.depths if selection.depths[z] >= 1)


# ===========================================================================
# Property 3c：每分区取 max_level ⇒ 结果等于 Full_In_Load（R7-AC5）
# ===========================================================================


@given(grid=grids(min_zones=0, max_zones=4))
@settings(max_examples=200, deadline=None)
def test_all_zones_at_max_level_equals_full_in_load(grid: Grid) -> None:
    """每个分区都选定其 ``max_level`` ⇒ 结果等于 ``select_cells(grid, None)``。

    Full_In_Load 是"每分区都取最深"的特例：显式构造一个为每个分区取
    ``max_level`` 的 :class:`LoadSelection`，其选出的有序 Cell 列表应与
    ``selection is None``（Full_In_Load）逐一相等（R5-AC1、R7-AC5）。

    **Feature: ego-layered-adapter, Property 3**

    **Validates: Requirements 5.1, 5.2, 7.5**
    """
    full_selection = LoadSelection(
        depths={zone: grid.max_level(zone) for zone in grid.zones}
    )

    explicit = select_cells(grid, full_selection)
    full_in_load = select_cells(grid, None)

    assert explicit == full_in_load


# ===========================================================================
# Property 3d：非法 selection ⇒ RuntimeError_（退出码 1，R5-AC6）
# ===========================================================================


@given(data=grid_and_low_depth())
@settings(max_examples=100, deadline=None)
def test_depth_below_one_raises_runtime_error(
    data: tuple[Grid, LoadSelection],
) -> None:
    """某分区深度 ``<1`` ⇒ ``select_cells`` 抛 ``RuntimeError_``（退出码 1）。

    **Feature: ego-layered-adapter, Property 3**

    **Validates: Requirements 5.6, 7.5**
    """
    grid, selection = data
    with pytest.raises(RuntimeError_) as exc_info:
        select_cells(grid, selection)
    assert exc_info.value.exit_code == EXIT_RUNTIME_FAILURE


@given(data=grid_and_high_depth())
@settings(max_examples=100, deadline=None)
def test_depth_above_max_level_raises_runtime_error(
    data: tuple[Grid, LoadSelection],
) -> None:
    """某分区深度 ``>max_level`` ⇒ ``select_cells`` 抛 ``RuntimeError_``（退出码 1）。

    **Feature: ego-layered-adapter, Property 3**

    **Validates: Requirements 5.6, 7.5**
    """
    grid, selection = data
    with pytest.raises(RuntimeError_) as exc_info:
        select_cells(grid, selection)
    assert exc_info.value.exit_code == EXIT_RUNTIME_FAILURE


@given(data=grid_and_unknown_zone())
@settings(max_examples=100, deadline=None)
def test_unknown_zone_raises_runtime_error(
    data: tuple[Grid, LoadSelection],
) -> None:
    """命名网格未列出的分区 ⇒ ``select_cells`` 抛 ``RuntimeError_``（退出码 1）。

    **Feature: ego-layered-adapter, Property 3**

    **Validates: Requirements 5.6, 7.5**
    """
    grid, selection = data
    with pytest.raises(RuntimeError_) as exc_info:
        select_cells(grid, selection)
    assert exc_info.value.exit_code == EXIT_RUNTIME_FAILURE
