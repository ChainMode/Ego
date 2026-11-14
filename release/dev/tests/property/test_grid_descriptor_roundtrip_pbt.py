"""Grid_Descriptor 构造—解析往返一致的属性测试。

本文件实现 design.md「Correctness Properties」中的 **Property 2**，对应
tasks.md 任务 4.2：

* **Property 2: Grid_Descriptor 构造—解析往返一致** —— 对任意已训练 Cell 列表
  （含恰好一个 Kernel 与若干分区/层级，每分区 Level 连续 ``1..max``），
  ``parse_grid_descriptor(build_grid_descriptor(cells))`` 还原出的内存
  :class:`~ego_pod.runtime.selection.Grid` 与输入语义等价：

  - Kernel 相对路径为 ``adapters/kernel.safetensors``；
  - 各分区的 :class:`~ego_pod.runtime.selection.GridCell` 列表按 Level **升序**
    排列且连续 ``1..max``；
  - 每个分区 Cell 的相对路径为 ``adapters/<zone>/L<level>.safetensors``；
  - 还原出的分区集合与各分区最大 Level 与输入一致；
  - 加载方仅凭该描述即可枚举全部 Cell（用 ``select_cells(grid, None)`` 行使
    Full_In_Load 的语义往返等价），无需扫描 ``adapters/`` 目录。

  **Validates: Requirements 4.1, 4.2, 4.3, 4.5, 1.6, 7.4**

测试在**无机器学习依赖**的环境下运行：``build_grid_descriptor`` /
``parse_grid_descriptor`` / ``select_cells`` 均为纯 Python 纯函数（设计「决策
D」），故 hypothesis 直接以合成 :class:`~ego_pod.trainer.cells.TrainedCell`
列表（随机分区名集合 + 每分区连续层级）行使往返一致性，每个属性测试最少运行
100 次迭代。

``build_grid_descriptor`` **不**读取 :class:`TrainedCell.output_path` 的内容，
仅依据每个 Cell 的 ``(zone, level)`` 推导相对路径（见其实现注记），因此本测试把
``output_path`` 设为明显的 dummy 路径以凸显这一点。

**Feature: ego-layered-adapter, Property 2**

**Validates: Requirements 4.1, 4.2, 4.3, 4.5, 1.6, 7.4**
"""

from __future__ import annotations

from pathlib import Path

import hypothesis.strategies as st
from hypothesis import given, settings

from ego_pod.runtime.selection import KERNEL, CellKey, Grid, select_cells
from ego_pod.trainer.cells import TrainedCell
from ego_pod.vault.grid_descriptor import (
    build_grid_descriptor,
    parse_grid_descriptor,
)
from ego_pod.vault.package import EDITION_NAME

# ---------------------------------------------------------------------------
# 输入生成策略
# ---------------------------------------------------------------------------

#: Zone_Name：由小写字母、数字、连字符组成（匹配 ^[a-z0-9-]+$，R1-AC3），可直接
#: 安全用作目录名。保持短小以加速 hypothesis 收缩。
_zone_name = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789-",
    min_size=1,
    max_size=6,
)

#: 每个分区的最大 Level（小而随机；契约支持任意深度，MVP 实际只产 Level 1）。
_max_level = st.integers(min_value=1, max_value=4)

#: 分区 → 最大 Level 的映射（dict 的键天然去重，保证 Zone_Name 唯一）；允许空集
#: 以覆盖「仅 Kernel 的网格包」退化情形（zones 为空数组）。
_zone_maxlevels = st.dictionaries(
    keys=_zone_name,
    values=_max_level,
    max_size=4,
)


def _build_cells(zone_maxlevels: dict[str, int]) -> list[TrainedCell]:
    """据 ``{zone: max_level}`` 构造已训练 Cell 列表（含恰好一个 Kernel）。

    刻意以**非自然顺序**追加（分区 Cell 按 Level 降序、Kernel 置于末尾），以确保
    ``build_grid_descriptor`` / ``parse_grid_descriptor`` 的「升序且连续」保证来自
    其内部稳定排序而非输入顺序。``output_path`` 设为明显的 dummy 路径（被
    ``build_grid_descriptor`` 忽略）。
    """

    cells: list[TrainedCell] = []
    for zone, max_level in zone_maxlevels.items():
        # 连续 1..max，但以降序追加以行使 build/parse 的排序。
        for level in range(max_level, 0, -1):
            cells.append(
                TrainedCell(
                    zone=zone,
                    level=level,
                    output_path=Path(f"/dummy/ignored/{zone}/L{level}.safetensors"),
                )
            )
    # Kernel（zone=None, level=0）置于末尾，验证 build 不依赖 Kernel 在首位。
    cells.append(
        TrainedCell(zone=None, level=0, output_path=Path("/dummy/ignored/kernel.safetensors"))
    )
    return cells


# ===========================================================================
# Property 2: Grid_Descriptor 构造—解析往返一致（任务 4.2）
# ===========================================================================


@given(zone_maxlevels=_zone_maxlevels)
@settings(max_examples=200, deadline=None)
def test_grid_descriptor_build_parse_roundtrip(
    zone_maxlevels: dict[str, int],
) -> None:
    """``parse_grid_descriptor(build_grid_descriptor(cells))`` 与输入语义等价。

    断言还原出的内存 :class:`Grid`：

    * Kernel 相对路径为 ``adapters/kernel.safetensors``（R4-AC1/AC3）；
    * 分区集合与各分区最大 Level 与输入一致；
    * 每个分区的 Cell 列表按 Level 升序且连续 ``1..max``（R4-AC5、R1-AC6）；
    * 每个分区 Cell 的相对路径为 ``adapters/<zone>/L<level>.safetensors``（R4-AC1/AC3）。

    **Feature: ego-layered-adapter, Property 2**

    **Validates: Requirements 4.1, 4.2, 4.3, 4.5, 1.6, 7.4**
    """
    cells = _build_cells(zone_maxlevels)

    descriptor = build_grid_descriptor(cells)
    grid = parse_grid_descriptor(descriptor)

    assert isinstance(grid, Grid)

    # Kernel 恒还原为 (None, 0)，相对路径为网格包 Kernel 路径（R4-AC1/AC3）。
    assert grid.kernel.cell == KERNEL
    assert grid.kernel.relpath == "adapters/kernel.safetensors"

    # 还原出的分区集合与输入一致（R4-AC2：仅凭描述即可确定全部分区）。
    assert set(grid.zones) == set(zone_maxlevels)

    for zone, max_level in zone_maxlevels.items():
        grid_cells = grid.zones[zone]
        levels = [gc.cell.level for gc in grid_cells]

        # 升序且连续 1..max（R4-AC5、R1-AC6）。
        assert levels == list(range(1, max_level + 1))
        # 该分区最大 Level 与输入一致（仅凭描述可确定可用 Level，R4-AC2）。
        assert grid.max_level(zone) == max_level

        for grid_cell, level in zip(grid_cells, range(1, max_level + 1)):
            # 每个 Cell 的键与相对路径逐一一致（R4-AC1/AC3）。
            assert grid_cell.cell == CellKey(zone=zone, level=level)
            assert grid_cell.relpath == f"adapters/{zone}/L{level}.safetensors"


@given(zone_maxlevels=_zone_maxlevels)
@settings(max_examples=200, deadline=None)
def test_loader_enumerates_all_cells_from_descriptor_alone(
    zone_maxlevels: dict[str, int],
) -> None:
    """加载方仅凭 Grid_Descriptor 即可枚举全部 Cell（语义往返等价）。

    用 ``select_cells(grid, None)`` 行使 Full_In_Load：断言其选出的有序 Cell 集合
    恰为「Kernel + 每个分区 Level 1..max」——即仅凭解析出的内存网格即可枚举出
    与输入已训练 Cell 列表语义等价的全部 Cell，无需扫描 ``adapters/`` 目录
    （R4-AC2、R7-AC4）。Kernel 恒排在首位（R1-AC6 派生 / kernel-always）。

    **Feature: ego-layered-adapter, Property 2**

    **Validates: Requirements 4.1, 4.2, 4.3, 4.5, 1.6, 7.4**
    """
    cells = _build_cells(zone_maxlevels)

    grid = parse_grid_descriptor(build_grid_descriptor(cells))
    selected = select_cells(grid, None)

    # 期望：Kernel 优先，再按分区名字典序、各分区 Level 由低到高（Full_In_Load）。
    expected: list[CellKey] = [KERNEL]
    for zone in sorted(zone_maxlevels):
        for level in range(1, zone_maxlevels[zone] + 1):
            expected.append(CellKey(zone=zone, level=level))

    assert selected == expected
    # Kernel 恒被包含且排最前。
    assert selected[0] == KERNEL
    assert KERNEL in selected
    # 划分性：选出的 Cell 总数 == Kernel(1) + 各分区层级数之和（无遗漏无重复）。
    assert len(selected) == 1 + sum(zone_maxlevels.values())
    assert len(set(selected)) == len(selected)


# ===========================================================================
# build_grid_descriptor 输出形状与设计 JSON 一致（任务 4.2 附加用例）
# ===========================================================================


@given(zone_maxlevels=_zone_maxlevels)
@settings(max_examples=200, deadline=None)
def test_build_grid_descriptor_matches_design_json_shape(
    zone_maxlevels: dict[str, int],
) -> None:
    """``build_grid_descriptor`` 的输出结构与 design.md 的磁盘 JSON 形状一致。

    顶层 ``{<edition>: {"kernel": {"path": ...}, "zones": {<zone>: [{"level", "path"}, ...]}}}``：

    * 顶层恰含单一 Edition 键；
    * Edition 字典恰含 ``"kernel"`` 与 ``"zones"`` 两键；
    * ``kernel`` 字典恰含 ``"path"``，取值为 ``adapters/kernel.safetensors``；
    * 每个分区为按 Level 升序、连续 ``1..max`` 的条目数组，每条目恰含
      ``"level"`` 与 ``"path"`` 两键。

    **Feature: ego-layered-adapter, Property 2**

    **Validates: Requirements 4.1, 4.2, 4.3, 4.5, 1.6, 7.4**
    """
    cells = _build_cells(zone_maxlevels)

    descriptor = build_grid_descriptor(cells)

    # 顶层：恰含单一 Edition 键（默认 "normal"）。
    assert set(descriptor) == {EDITION_NAME}

    edition_grid = descriptor[EDITION_NAME]
    # Edition 字典：恰含 "kernel" 与 "zones"。
    assert set(edition_grid) == {"kernel", "zones"}

    # kernel：恰含 "path"，取值为网格包 Kernel 相对路径（R4-AC1/AC3）。
    assert edition_grid["kernel"] == {"path": "adapters/kernel.safetensors"}

    # zones：分区集合与输入一致。
    zones = edition_grid["zones"]
    assert set(zones) == set(zone_maxlevels)

    for zone, max_level in zone_maxlevels.items():
        entries = zones[zone]
        # 升序且连续 1..max（R4-AC5）。
        assert [e["level"] for e in entries] == list(range(1, max_level + 1))
        for entry, level in zip(entries, range(1, max_level + 1)):
            # 每条目恰含 "level" 与 "path"，且路径符合磁盘命名约定（R4-AC1/AC3）。
            assert set(entry) == {"level", "path"}
            assert entry["level"] == level
            assert entry["path"] == f"adapters/{zone}/L{level}.safetensors"
