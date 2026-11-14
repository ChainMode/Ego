"""Runtime 网格内存结构与 ``select_cells`` 纯函数（二维稀疏网格的加载选择）。

本模块实现 Ego Layered Adapter 设计文档 §"Components and Interfaces → 5.
Runtime → 5.1 数据结构与 select_cells" 所定义的**纯函数**加载选择语义：把
一张解析自 ``manifest.json`` 的内存网格（:class:`Grid`）与一次加载选择
（:class:`LoadSelection`，或 ``None``）映射为一个**有序的 Cell 列表**，供
``runtime.inference`` 按序以 Coexist_Activate 加载并激活各 Cell 适配器。

二维网格（Sparse_Grid）的内存表示：

* :class:`CellKey` —— Cell 的可寻址键。Kernel 编码为 ``(zone=None, level=0)``；
  分区 Cell 编码为 ``(zone, level>=1)``。该结构与 :mod:`ego_pod.compiler.routing`
  的 ``CellKey`` 同构，但在本模块内**自包含**地重新定义，以保持 Runtime 不
  依赖 Compiler（避免跨模块耦合）。
* :class:`GridCell` —— 网格中一个 Cell 及其相对 ``<edition>`` 目录的适配器
  文件相对路径。
* :class:`Grid` —— Kernel + 各 Functional_Zone 的有序 Level 列表（每个分区
  按 Level 升序、且连续 ``1..max``）。
* :class:`LoadSelection` —— 一次加载选择；``depths`` 为每个被选分区的加载
  深度，空 ``depths`` ⇒ 仅 Kernel。``None``（未指定 selection）由
  :func:`select_cells` 解释为 Full_In_Load。

:func:`select_cells` 满足以下关键正确性（对应需求 R5-AC1/AC2/AC3/AC4/AC6、
R7-AC5）：

* **Kernel 恒含且排最前**（R5-AC3）：任何一次加载恒包含 Kernel，且 Kernel 是
  结果列表的第一个元素，与是否选定某分区无关。
* **Full_In_Load**（R5-AC1）：``selection is None`` ⇒ Kernel + 每个分区
  Level ``1..max``。
* **Nested_In_Load**（R5-AC2）：否则 Kernel + 对 ``selection.depths`` 中每个
  分区取 Level ``1..depth``。
* **有序**（R5-AC4）：Kernel 优先，再按分区名稳定（字典序）排序，各分区内
  Level 由低到高。
* **层级前缀属性**（R7-AC5）：对某分区选定深度 ``N``，选出的恰为该分区 Level
  ``1..N`` 的前缀；当每个分区都取 ``max_level`` 时，结果集合等于 Full_In_Load。
* **校验**（R5-AC6）：深度 ``<1`` 或 ``>max_level(zone)`` ⇒ 抛
  :class:`~ego_pod.exit_codes.RuntimeError_`（含无效深度）；``selection`` 命名
  网格未列出的分区 ⇒ 抛 :class:`~ego_pod.exit_codes.RuntimeError_`（含未知
  分区名）。两者均映射到退出码 1。

本模块为**纯 Python**：无任何机器学习依赖、无 I/O，导入安全（满足决策 D /
R6-AC8 / R7-AC2），因此层级前缀属性可被属性测试直接行使（Property 3）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ego_pod.exit_codes import RuntimeError_

__all__ = [
    "CellKey",
    "KERNEL",
    "GridCell",
    "Grid",
    "LoadSelection",
    "select_cells",
]


# ---------------------------------------------------------------------------
# 内存网格数据结构
# ---------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class CellKey:
    """Cell 的可寻址键。

    Kernel 编码为 ``(zone=None, level=0)``；分区 Cell 编码为
    ``(zone, level>=1)``。``frozen=True, order=True`` 使其可作为 dict 键并支持
    稳定排序（同一分区内按 Level 比较）。本结构与
    :mod:`ego_pod.compiler.routing` 的 ``CellKey`` 同构，但在 Runtime 内自
    包含定义以避免跨模块依赖。

    .. note::
        由于 ``zone`` 可为 ``None``，跨 Kernel 与分区 Cell 直接排序
        （``None`` 与 ``str`` 比较）会抛 ``TypeError``。:func:`select_cells`
        从不对混合列表排序——它显式把 Kernel 置首，并对分区名（``str``）排序，
        故不触发该比较。
    """

    #: Functional_Zone 名称；``None`` ⇒ Kernel。
    zone: str | None
    #: 层级：Kernel 恒为 ``0``；分区 Cell ∈ ``[1, max_level]``。
    level: int

    @property
    def is_kernel(self) -> bool:
        """该 Cell 是否为 Kernel（与分区无关的 Level-0 Cell）。"""
        return self.zone is None and self.level == 0


#: Kernel 的规范 Cell 键（与分区无关的 Level-0 Cell），任何一次加载恒含。
KERNEL = CellKey(None, 0)


@dataclass(frozen=True)
class GridCell:
    """网格中一个 Cell 及其适配器文件位置。

    ``relpath`` 为该 Cell 的 LoRA_Adapter 文件相对其 ``<edition>`` 目录的
    相对路径（如 ``adapters/kernel.safetensors`` 或
    ``adapters/<zone>/L<level>.safetensors``）。
    """

    #: 该 Cell 的可寻址键。
    cell: CellKey
    #: 适配器文件相对 ``<edition>`` 目录的相对路径。
    relpath: str


@dataclass(frozen=True)
class Grid:
    """解析自 ``manifest.json`` 的内存网格：Kernel + 各分区有序 Level 列表。

    ``zones`` 把每个 Functional_Zone 的 Zone_Name 映射到其 Cell 列表；每个
    列表按 Level **升序**排列且 Level **连续 ``1..max``**（每分区层级前缀，
    见需求 R1-AC6 / R4-AC5）。含 0 个 Cell 的分区表达为空列表（R4-AC6）。
    """

    #: Kernel Cell（与分区无关的 Level-0 Cell）。
    kernel: GridCell
    #: Zone_Name → 该分区按 Level 升序、连续 ``1..max`` 的 Cell 列表。
    zones: dict[str, list[GridCell]] = field(default_factory=dict)

    def max_level(self, zone: str) -> int:
        """返回 ``zone`` 分区的最大 Level。

        分区存在且非空 ⇒ 返回其最后一个（最高）Cell 的 Level；分区存在但为
        空列表 ⇒ 返回 ``0``（无任何可加载 Level，R4-AC6）；分区不存在 ⇒ 抛
        :class:`~ego_pod.exit_codes.RuntimeError_`（含未知分区名）。
        """
        cells = self.zones.get(zone)
        if cells is None:
            raise RuntimeError_(
                f"加载选择命名了网格未列出的功能分区「{zone}」"
            )
        if not cells:
            return 0
        return cells[-1].cell.level


@dataclass(frozen=True)
class LoadSelection:
    """一次加载选择（R5）。

    ``depths`` 为每个被选 Functional_Zone 的加载深度（Zone_Name → 选定深度
    ``1..max_level``）。空 ``depths`` ⇒ 仅加载 Kernel。

    .. note::
        ``None``（未向 :func:`select_cells` 指定 selection）被解释为
        Full_In_Load（R5-AC1）；空 :class:`LoadSelection`（``depths={}``）则
        表示**显式**的 kernel-only 选择，二者语义不同。
    """

    #: Zone_Name → 选定深度（``1..max_level``）。
    depths: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 纯函数：从网格选出有序 Cell 列表
# ---------------------------------------------------------------------------


def select_cells(grid: Grid, selection: LoadSelection | None) -> list[CellKey]:
    """从 ``grid`` 选出一个有序的 Cell 列表（纯函数）。

    语义（R5-AC1/AC2/AC3/AC4、R7-AC5）：

    * ``selection is None`` ⇒ **Full_In_Load**：Kernel + 每个分区 Level
      ``1..max``。
    * 否则 ⇒ **Nested_In_Load**：Kernel + 对 ``selection.depths`` 中每个分区
      取 Level ``1..depth``。
    * Kernel 恒被包含且置于列表**最前**（R5-AC3）。
    * 顺序：Kernel 优先，再按分区名字典序稳定排序，各分区内 Level 由低到高
      （R5-AC4）。

    校验（R5-AC6）：

    * 若 ``selection`` 命名了 ``grid.zones`` 未列出的分区 ⇒ 抛
      :class:`~ego_pod.exit_codes.RuntimeError_`（消息含未知分区名）。
    * 若某分区请求的深度 ``<1`` 或 ``>grid.max_level(zone)`` ⇒ 抛
      :class:`~ego_pod.exit_codes.RuntimeError_`（消息含无效深度）。

    层级前缀属性（R7-AC5、Property 3）：对某分区选定深度 ``N``，选出的恰为该
    分区 Level ``1..N`` 的前缀；当每个分区都取其 ``max_level`` 时，结果等于
    Full_In_Load 的 Cell 集合。

    本函数无 I/O、无机器学习依赖，纯粹由内存数据推导。

    :param grid: 解析自 manifest 的内存网格。
    :param selection: 加载选择；``None`` ⇒ Full_In_Load。
    :returns: 有序的 :class:`CellKey` 列表，Kernel 在首位。
    :raises RuntimeError_: 选择命名了未知分区或请求了非法深度（退出码 1）。
    """
    # Kernel 恒含且排最前（R5-AC3）。
    selected: list[CellKey] = [grid.kernel.cell]

    if selection is None:
        # Full_In_Load（R5-AC1）：每个分区按 Level 升序取全部 Cell。
        for zone in sorted(grid.zones):
            for grid_cell in grid.zones[zone]:
                selected.append(grid_cell.cell)
        return selected

    # Nested_In_Load（R5-AC2）：对每个被选分区取 Level 1..depth 前缀。
    # 按分区名字典序处理以保证稳定顺序（R5-AC4）。
    for zone in sorted(selection.depths):
        if zone not in grid.zones:
            # R5-AC6：未知分区名。
            raise RuntimeError_(
                f"加载选择命名了网格未列出的功能分区「{zone}」"
            )
        depth = selection.depths[zone]
        max_level = grid.max_level(zone)
        if depth < 1 or depth > max_level:
            # R5-AC6：无效深度（小于 1 或超过该分区最大 Level）。
            raise RuntimeError_(
                f"功能分区「{zone}」的加载深度 {depth} 非法："
                f"须在 1 到该分区最大 Level {max_level} 之间"
            )
        # 取该分区 Level 1..depth 的前缀（zones[zone] 已按 Level 升序、连续）。
        for grid_cell in grid.zones[zone]:
            if grid_cell.cell.level <= depth:
                selected.append(grid_cell.cell)

    return selected
