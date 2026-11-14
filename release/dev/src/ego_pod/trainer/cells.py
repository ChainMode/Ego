"""网格训练（Grid_Training）的数据结构。

本模块定义按 Cell 训练编排（``trainer.grid.run_grid_training``，见设计文档
§「Components and Interfaces → 3. Trainer」）所需的三个**纯数据结构**，把人格从
**单个扁平 LoRA 适配器**演进为**二维稀疏网格（Sparse_Grid）**时的最小可寻址单元
建模出来：

- :class:`CellSpec`：一个**待训练** Cell 的规格——网格中最小可寻址单元
  ``Cell = (Functional_Zone, Level)`` 加上其 Cell_Sub_Dataset（JSONL）路径。
- :class:`TrainedCell`：一个**已训练** Cell 的产出（导出的 LoRA_Adapter 临时
  路径），供 Vault 构建 ``manifest.json`` 的 Grid_Descriptor（需求 3.8）。
- :class:`GridTrainingResult`：``run_grid_training`` 的整体结果（已训练 Cell 集合
  与所选后端的名称/硬件/Backend_Identifier）。

**Kernel 的编码约定**：Kernel 是与分区无关、恒被加载的特殊 Level-0 Cell，统一编码
为 ``zone=None, level=0``（设计「决策 B」kernel-always）；外围 Functional_Zone 的
Cell 则为 ``zone=<name>, level>=1``。

本模块为**纯 Python、无 ML 依赖、导入安全**（设计「决策 D」）：只从
:mod:`ego_pod.trainer.base` 引入类型别名 ``Hardware`` / ``BackendIdentifier``，不导入
``torch`` / ``peft`` 等训练依赖，保证 ``import ego_pod`` 在默认安装下永不抛错
（需求 6.8、7.2）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ego_pod.trainer.base import BackendIdentifier, Hardware

__all__ = [
    "CellSpec",
    "TrainedCell",
    "GridTrainingResult",
]


@dataclass(frozen=True)
class CellSpec:
    """一个**待训练** Cell 的规格（A specification of a Cell to be trained）。

    描述网格中一个最小可寻址单元 ``Cell = (Functional_Zone, Level)`` 及其
    Cell_Sub_Dataset 的写出路径，作为 ``run_grid_training`` 的逐 Cell 训练输入。
    Kernel 是特殊的 Level-0 Cell，统一编码为 ``zone=None, level=0``。

    :ivar zone: 所属 Functional_Zone 名称；``None`` 表示 Kernel（与分区无关）。
    :ivar level: 该 Cell 在分区内的径向层级；Kernel 恒为 ``0``，分区 Cell ``>= 1``。
    :ivar sub_dataset_path: 该 Cell 的 Cell_Sub_Dataset（JSONL）中间文件路径，由
        Compiler 的 ``route_cells`` 写出（Kernel → ``.ego/cells/kernel.jsonl``、
        分区 → ``.ego/cells/<zone>/L<level>.jsonl``）。
    """

    zone: str | None
    level: int
    sub_dataset_path: Path

    @property
    def is_kernel(self) -> bool:
        """该 Cell 是否为 Kernel（Is this Cell the Kernel）。

        Kernel 编码为 ``zone=None`` 且 ``level=0``；任一不满足即为分区 Cell。
        """

        return self.zone is None and self.level == 0


@dataclass(frozen=True)
class TrainedCell:
    """一个**已训练** Cell 的产出（The output of a trained Cell）。

    记录一个 Cell 经训练并导出后的 LoRA_Adapter 临时落点，供 Vault 据此构建
    ``manifest.json`` 的 Grid_Descriptor 并事务性写入 Ego_Package（需求 3.8）。
    Kernel 同样编码为 ``zone=None, level=0``。

    :ivar zone: 所属 Functional_Zone 名称；``None`` 表示 Kernel（Level 0）。
    :ivar level: 该 Cell 在分区内的径向层级；Kernel 恒为 ``0``，分区 Cell ``>= 1``。
    :ivar output_path: 导出的 LoRA_Adapter（``.safetensors``）临时路径，供 Vault
        移动/重命名到 Ego_Package（如 ``adapters/kernel.safetensors`` 或
        ``adapters/<zone>/L<level>.safetensors``）。
    """

    zone: str | None
    level: int
    output_path: Path


@dataclass(frozen=True)
class GridTrainingResult:
    """``run_grid_training`` 的结果（The result of grid training）。

    汇总一次网格训练中全部**已训练** Cell 的产出与所选 Training_Backend 的信息，供
    ``ego train`` 编排层把各 Cell 适配器交给 Vault，并由 Vault 把 Backend_Identifier
    写入 ``manifest.json`` 的 ``training_backend`` 标记（需求 3.8、7.1、7.4）。

    :ivar cells: 已训练 Cell 的有序集合（Kernel 优先，再各分区 Level 由低到高）；
        空 Cell 已被跳过、不出现在此处。
    :ivar backend_name: 所选 Training_Backend 的可读名称（需求 8.2）。
    :ivar backend_hardware: 所选后端的硬件类型（CUDA / Apple Silicon / CPU / STUB）。
    :ivar backend_identifier: 所选后端的稳定 Backend_Identifier（取值集合
        ``{unsloth, peft-cuda, mlx, peft-cpu, stub}``），默认 ``"stub"`` 以保持
        对外构造的向后兼容（需求 7.1、7.4）。
    """

    cells: tuple[TrainedCell, ...]
    backend_name: str
    backend_hardware: Hardware
    backend_identifier: BackendIdentifier = "stub"
