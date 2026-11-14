"""Grid_Descriptor 的构造与解析（往返对称）。

Build and parse the ``manifest.json`` Grid_Descriptor (round-trip symmetric).

本模块实现 Ego Layered Adapter 设计文档 §"Components and Interfaces → 4. Vault
→ 4.1 grid_descriptor.py（往返对称）" 所定义的一对**互逆纯函数**，把人格二维稀疏
网格（Sparse_Grid）的产物契约在两种表示之间转换：

* :func:`build_grid_descriptor` —— 把 Trainer 产出的已训练 Cell 列表
  (:class:`~ego_pod.trainer.cells.TrainedCell`) 构造为 ``manifest.json`` 顶层
  ``grid`` 键的磁盘 JSON 形状（单 Edition）。
* :func:`parse_grid_descriptor` —— ``build`` 的逆：把 ``grid`` 描述（或 ``None``）
  解析回 Runtime 的内存网格 (:class:`~ego_pod.runtime.selection.Grid`)。

This module implements a pair of **mutually inverse pure functions** that convert
the two-dimensional Sparse_Grid artifact contract between two representations:
:func:`build_grid_descriptor` turns the trainer's list of trained Cells into the
on-disk JSON shape of the ``grid`` key in ``manifest.json``, and
:func:`parse_grid_descriptor` is its inverse, parsing that ``grid`` description
(or ``None``) back into the in-memory :class:`Grid`.

Grid_Descriptor 的磁盘 JSON 形状（设计文档 §"Data Models"，开放问题 1 的确认决策）::

    "grid": {
      "<edition>": {
        "kernel": { "path": "adapters/kernel.safetensors" },
        "zones": {
          "<zone>": [
            { "level": 1, "path": "adapters/<zone>/L1.safetensors" },
            { "level": 2, "path": "adapters/<zone>/L2.safetensors" }
          ]
        }
      }
    }

关键正确性（对应需求 R4-AC1/AC2/AC5/AC6、R6-AC1、R7-AC4）：

* **相对路径**（R4-AC1）：Kernel 落 ``adapters/kernel.safetensors``，分区 Cell 落
  ``adapters/<zone>/L<level>.safetensors``，均相对该 ``<edition>`` 目录。
* **升序且连续**（R4-AC5）：每个分区的 Level 列表按 Level 升序排列，且连续
  ``1..max``（输入契约保证连续，构造时稳定排序）。
* **空分区**（R4-AC6）：含 0 个 Cell 的分区在描述中表达为空数组 ``[]``，
  :func:`parse_grid_descriptor` 据此还原为空 Cell 列表（MVP 不产生空分区，但
  schema 与解析均支持）。
* **退化即扁平**（R6-AC1）：``grid is None``（manifest 无 ``grid`` 键）⇒
  :func:`parse_grid_descriptor` 返回单 Kernel Cell 的 Flat 网格，Kernel 相对路径为
  ``adapters/ego-lora.safetensors``，无任何分区。
* **往返一致**（R7-AC4、Property 2）：``parse_grid_descriptor(build_grid_descriptor(cells))``
  还原出的内存 :class:`Grid` 与输入语义等价（Kernel、各分区、各 Level 与文件
  相对路径逐一一致），使加载方仅凭描述即可枚举全部 Cell 而无需扫描目录。

本模块为**纯 Python**：无任何机器学习依赖、无 I/O，导入安全（设计「决策 D」/
R6-AC8 / R7-AC2），因此往返一致性可被属性测试直接行使（Property 2）。
"""

from __future__ import annotations

from ego_pod.runtime.selection import KERNEL, CellKey, Grid, GridCell
from ego_pod.trainer.cells import TrainedCell
from ego_pod.vault.package import EDITION_NAME

__all__ = [
    "ADAPTERS_DIRNAME",
    "KERNEL_RELPATH",
    "FLAT_KERNEL_RELPATH",
    "kernel_relpath",
    "zone_cell_relpath",
    "build_grid_descriptor",
    "parse_grid_descriptor",
]


# ---------------------------------------------------------------------------
# 相对路径约定（设计文档 §"Data Models → 磁盘文件命名"）
# ---------------------------------------------------------------------------

#: ``<edition>`` 目录下存放全部 Cell 适配器的子目录名。
#: Name of the sub-directory under ``<edition>`` that holds all Cell adapters.
ADAPTERS_DIRNAME = "adapters"

#: 网格包中 Kernel 适配器相对 ``<edition>`` 目录的相对路径（R4-AC1/AC3）。
#: Relative path (to ``<edition>``) of the Kernel adapter in a Grid_Package.
KERNEL_RELPATH = f"{ADAPTERS_DIRNAME}/kernel.safetensors"

#: 退化扁平包（Flat_Package）中单 Kernel Cell 适配器的相对路径（R6-AC1）。
#: Relative path of the single Kernel Cell adapter in a degenerate Flat_Package.
FLAT_KERNEL_RELPATH = f"{ADAPTERS_DIRNAME}/ego-lora.safetensors"


def kernel_relpath() -> str:
    """返回网格包 Kernel 适配器的相对路径（Return the Grid_Package Kernel relpath）。

    恒为 :data:`KERNEL_RELPATH`（``adapters/kernel.safetensors``），相对该 Edition
    目录（R4-AC1/AC3）。

    :returns: Kernel 适配器相对 ``<edition>`` 目录的相对路径（POSIX 正斜杠）。
    """

    return KERNEL_RELPATH


def zone_cell_relpath(zone: str, level: int) -> str:
    """返回某分区 Cell 适配器的相对路径（Return a zone Cell adapter relpath）。

    形如 ``adapters/<zone>/L<level>.safetensors``，相对该 Edition 目录
    （R4-AC1/AC3）。``Zone_Name`` 已被约束为 ``[a-z0-9-]+``（R1-AC3），可直接安全
    用作目录名，无需转义。

    :param zone: Functional_Zone 名称（Zone_Name）。
    :param level: 该 Cell 在分区内的径向层级（``>= 1``）。
    :returns: 该分区 Cell 适配器相对 ``<edition>`` 目录的相对路径（POSIX 正斜杠）。
    """

    return f"{ADAPTERS_DIRNAME}/{zone}/L{level}.safetensors"


# ---------------------------------------------------------------------------
# 构造：已训练 Cell 列表 → manifest 的 grid 描述
# ---------------------------------------------------------------------------


def build_grid_descriptor(
    cells: list[TrainedCell],
    *,
    edition: str = EDITION_NAME,
) -> dict:
    """把已训练 Cell 列表构造为 ``manifest.json`` 的 ``grid`` 描述（单 Edition）。

    Build the ``grid`` description for one Edition from a list of trained Cells.

    输出结构（设计文档 §"Data Models"）::

        {
          "<edition>": {
            "kernel": {"path": "adapters/kernel.safetensors"},
            "zones": {
              "<zone>": [{"level": 1, "path": "adapters/<zone>/L1.safetensors"}, ...]
            }
          }
        }

    语义保证：

    * Kernel（``zone=None, level=0``）落 :data:`KERNEL_RELPATH`（R4-AC1/AC3）。
    * 每个分区的 Level 列表按 Level **升序**排列，且连续 ``1..max``（R4-AC5）；
      输入契约保证每分区 Level 连续，构造时稳定排序。
    * 路径为相对该 ``<edition>`` 目录的相对路径（R4-AC1）。
    * 分区按 Zone_Name 字典序写出，使产物确定（便于逐字节比较与回归）。

    .. note::
        本函数仅依据每个 :class:`~ego_pod.trainer.cells.TrainedCell` 的
        ``(zone, level)`` 计算**相对路径**，**不**使用其 ``output_path``（后者是
        训练阶段的临时落点，由 Vault 移动到 Ego_Package）。

    含 0 个 Cell 的分区无法从 ``cells`` 列表表达（MVP 也不产生空分区）；空数组
    ``[]`` 的语义由 :func:`parse_grid_descriptor` 支持（R4-AC6）。

    :param cells: Trainer 产出的已训练 Cell 列表，须含恰好一个 Kernel Cell
        （``zone=None, level=0``，kernel-always 由 Vault 在构建前保证）。
    :param edition: 目标 Edition 名称，默认 :data:`~ego_pod.vault.package.EDITION_NAME`。
    :returns: 可直接作为 ``manifest.json`` 顶层 ``grid`` 键值的字典。
    """

    # 按 Zone_Name 聚合分区 Cell；Kernel（zone is None, level 0）单独处理。
    zone_cells: dict[str, list[TrainedCell]] = {}
    for cell in cells:
        if cell.zone is None and cell.level == 0:
            # Kernel：路径恒为 KERNEL_RELPATH，无需记录。
            continue
        zone_cells.setdefault(cell.zone, []).append(cell)

    # 每个分区按 Level 升序构造其 Cell 数组（R4-AC5）。
    zones: dict[str, list[dict]] = {}
    for zone in sorted(zone_cells):
        ordered = sorted(zone_cells[zone], key=lambda c: c.level)
        zones[zone] = [
            {"level": c.level, "path": zone_cell_relpath(zone, c.level)}
            for c in ordered
        ]

    return {
        edition: {
            "kernel": {"path": KERNEL_RELPATH},
            "zones": zones,
        }
    }


# ---------------------------------------------------------------------------
# 解析：manifest 的 grid 描述 → 内存 Grid（build 的逆）
# ---------------------------------------------------------------------------


def parse_grid_descriptor(
    grid: dict | None,
    edition: str = EDITION_NAME,
) -> Grid:
    """把 ``grid`` 描述（或 ``None``）解析回内存网格（``build`` 的逆，往返一致）。

    Parse the ``grid`` description (or ``None``) back into an in-memory
    :class:`Grid`; the inverse of :func:`build_grid_descriptor` (R7-AC4).

    * ``grid is None``（manifest 无 ``grid`` 键）⇒ 返回单 Kernel Cell 的 **Flat 网格**：
      Kernel 相对路径为 :data:`FLAT_KERNEL_RELPATH`（``adapters/ego-lora.safetensors``），
      无任何分区（R6-AC1）。加载方据此把无 ``grid`` 的包视为「单个 Kernel Cell」。
    * 否则 ⇒ 解析该 Edition 的 Kernel 与各分区：Kernel 还原为
      :data:`~ego_pod.runtime.selection.KERNEL` 加其描述路径；每个分区的 Cell 数组
      还原为按 Level **升序**的 :class:`~ego_pod.runtime.selection.GridCell` 列表
      （解析时稳定排序，保证 :class:`Grid` 的「升序且连续」不变量）；空数组 ``[]``
      还原为空 Cell 列表（R4-AC6）。

    :param grid: ``manifest.json`` 顶层 ``grid`` 键的值；``None`` 表示 Flat_Package。
    :param edition: 目标 Edition 名称，默认 :data:`~ego_pod.vault.package.EDITION_NAME`。
    :returns: 还原出的内存 :class:`Grid`（Kernel + 各分区有序 Level 列表）。
    """

    if grid is None:
        # 退化即扁平（R6-AC1）：单 Kernel Cell，Kernel 落 adapters/ego-lora.safetensors。
        return Grid(
            kernel=GridCell(cell=KERNEL, relpath=FLAT_KERNEL_RELPATH),
            zones={},
        )

    edition_grid = grid[edition]

    kernel = GridCell(cell=KERNEL, relpath=edition_grid["kernel"]["path"])

    zones: dict[str, list[GridCell]] = {}
    for zone, entries in edition_grid.get("zones", {}).items():
        # 按 Level 升序还原（保证 Grid 的「升序且连续 1..max」不变量，R4-AC5）。
        ordered = sorted(entries, key=lambda e: e["level"])
        zones[zone] = [
            GridCell(cell=CellKey(zone=zone, level=e["level"]), relpath=e["path"])
            for e in ordered
        ]

    return Grid(kernel=kernel, zones=zones)
