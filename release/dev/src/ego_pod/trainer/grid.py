"""按 Cell 训练编排（Grid_Training，需求 3.1–3.9）。

本模块实现 :func:`run_grid_training`：把一张二维稀疏网格（Sparse_Grid）的若干
**待训练 Cell**（:class:`~ego_pod.trainer.cells.CellSpec`）编排为按 Cell 训练并导出
多个独立 LoRA_Adapter 的流水线，对应设计文档 §「Components and Interfaces →
3. Trainer → 3.2 run_grid_training」。

它把人格从 ego-pod-mvp / ego-pod-real-trainer 的**单个扁平 LoRA 适配器**
（``run_training`` 单适配器路径）扩展为**按 Cell 多适配器**，同时**完整复用**
``run_training`` 既有的装配逻辑（探测后端、打印所选后端、校验五个 LoRA 参数、
离线守卫），保证对外契约（``detect_backend`` 优先级、五参数校验、隐私/离线规则）
与单适配器路径**逐字保持一致**（R6-AC5、R8-AC9、R9-AC7）。

流水线步骤（见设计 §3.2）：

1. **装配**（复用 ``run_training`` 既有逻辑）：``detect_backend()`` 探测首个可用
   后端（优先级不变，R6-AC5）→ 打印 ``Selected backend: <name> (<hardware>)`` →
   :func:`~ego_pod.trainer.pipeline._validate_training_params` 校验五个 LoRA 参数
   （缺失/非法 → :class:`~ego_pod.exit_codes.TrainerError`，退出码 1，R8-AC9）→
   离线模式下经 ``privacy.offline`` 在加载基座前检查本地权重可用性（缺失由
   ``privacy.offline`` 抛 ``OfflineMissingWeightsError``，R9-AC7，本模块不拦截）。
2. **训练排序**（R3-AC2）：Kernel（Level 0）优先，再逐个 Functional_Zone，同一分区
   内 Level 由 1 到该分区最大 Level 升序。
3. **逐 Cell 训练**：对每个 Cell——

   - **空 Cell**（Cell_Sub_Dataset 为空/缺失）⇒ **跳过**，在标准输出记录被跳过
     Cell 的标识（``Kernel`` 或 ``(<zone>, L<level>)``）与原因，**不产出**任何
     LoRA_Adapter 文件（R3-AC6）。
   - **非空 Cell** ⇒ ``set_deterministic_seed(_SEED)`` 重置随机源（每 Cell 独立
     确定性，R3-AC5）→ **fresh** ``backend.load_base_model(model_id, quantization)``
     （fresh-load-per-cell 隔离，保证各 Cell 仅从其子数据集学习、互不污染，
     R3-AC1/AC3，见设计「决策 fresh-load-per-cell」）→ ``backend.train(...)`` →
     ``backend.export_lora_safetensors(model, cell_tmp_path)``（仅含适配器张量，
     R3-AC4）。计算失败（OOM / 损失 NaN/Inf 等）⇒ ``TrainerError``，消息含失败
     Cell 标识（R3-AC7）。
4. **汇总**：返回 :class:`~ego_pod.trainer.cells.GridTrainingResult`，携带每个**已
   训练** Cell 的 ``(zone, level, output_path)`` 与所选后端的名称/硬件/
   Backend_Identifier，供 Vault 构建 ``manifest.json`` 的 Grid_Descriptor（R3-AC8）。

**单 Cell 退化（R3-AC9）**：当 ``cells`` 仅含 Kernel（无任何分区）时，本函数训练并
导出**恰好一个** LoRA_Adapter，其训练行为等同于 MVP / real-trainer 的 ``run_training``
单适配器路径。

**Cell 适配器临时落点**（供 Vault 移动/重命名到 Ego_Package）：

- Kernel ⇒ ``{cwd}/.ego/cells_out/kernel.safetensors``
- 分区 Cell ⇒ ``{cwd}/.ego/cells_out/<zone>/L<level>.safetensors``

**导入安全（R7-AC2、R6-AC8）**：本模块顶层只依赖标准库与 import-safe 的 ``ego_pod``
子模块（``trainer.cells`` / ``trainer.base`` / ``trainer.detect`` / ``trainer.loop`` /
``trainer.pipeline`` / ``privacy.offline`` / ``logging_io`` / ``config.schema`` /
``exit_codes``），全部重型 ML 依赖（``torch`` / ``peft`` 等）经所选后端的方法在其
内部惰性导入，故 ``import ego_pod.trainer.grid`` 在仅安装默认依赖的环境下永不抛错。

隐私约束（R9-AC4/AC5）：与 ``run_training`` 一致，本模块只在本地把 Cell_Sub_Dataset
路径与张量交给后端计算，不发送任何原始训练数据；产物仅落于工作目录内。
"""

from __future__ import annotations

from pathlib import Path
from typing import TextIO

from ego_pod.config.schema import TrainingParams
from ego_pod.exit_codes import EgoError, TrainerError
from ego_pod.logging_io import print_summary
from ego_pod.privacy.offline import OfflineMode, get_active_offline_mode
from ego_pod.trainer.base import TrainingBackend, resolve_identifier
from ego_pod.trainer.cells import CellSpec, GridTrainingResult, TrainedCell
from ego_pod.trainer.detect import detect_backend
from ego_pod.trainer.loop import set_deterministic_seed

# 复用 run_training 既有的装配逻辑/常量，避免在网格路径重复实现（R6-AC5、R8-AC9、
# R9-AC7 与单适配器路径逐字一致）。
from ego_pod.trainer.pipeline import (
    DEFAULT_MODEL_ID,
    _base_model_available,
    _validate_training_params,
    make_progress_callback,
)

__all__ = [
    "run_grid_training",
]

#: Trainer 中间产物目录名，位于进程工作目录下（与 ``trainer.pipeline`` 一致，R9-AC8）。
EGO_DIR_NAME = ".ego"

#: 各 Cell 适配器的临时导出子目录名，落于 ``{cwd}/.ego/cells_out/`` 下供 Vault 取用。
CELLS_OUT_DIR_NAME = "cells_out"

#: 各 Cell 训练前重置的确定性种子（沿用 real-trainer 的 ``_SEED`` 约定，固定为 0）。
#:
#: 每个 Cell 训练前以 :func:`~ego_pod.trainer.loop.set_deterministic_seed` 重置随机源，
#: fresh base + 固定种子 ⇒ 相同 Base_Model/子数据集/参数/种子下重复训练某 Cell 产出
#: 语义等价的适配器（R3-AC5）；各 Cell 数据集不同、训练相互独立，自然得到不同适配器。
_SEED = 0


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------


def _cell_label(cell: CellSpec) -> str:
    """构造一个 Cell 的人类可读标识，用于日志与错误消息（R3-AC6/AC7）。

    Kernel ⇒ ``"Kernel"``；分区 Cell ⇒ ``"(<zone>, L<level>)"``。
    """

    if cell.is_kernel:
        return "Kernel"
    return f"({cell.zone}, L{cell.level})"


def _training_order_key(cell: CellSpec) -> tuple[int, str, int]:
    """训练顺序排序键（R3-AC2）。

    Kernel（Level 0）优先（首元 ``0``），再各分区（首元 ``1``）按 Zone_Name 稳定
    升序、同一分区内按 Level 升序。Kernel 的 ``zone`` 为 ``None``，以空串占位避免
    与分区名做 ``None`` 比较。
    """

    if cell.is_kernel:
        return (0, "", 0)
    return (1, cell.zone or "", cell.level)


def _cell_output_path(root: Path, cell: CellSpec) -> Path:
    """计算某 Cell 的适配器临时导出路径。

    Kernel ⇒ ``<root>/kernel.safetensors``；分区 Cell ⇒
    ``<root>/<zone>/L<level>.safetensors``（``<zone>`` 已被 Config 约束为
    ``[a-z0-9-]+``，可安全用作目录名）。
    """

    if cell.is_kernel:
        return root / "kernel.safetensors"
    return root / str(cell.zone) / f"L{cell.level}.safetensors"


def _sub_dataset_is_empty(path: Path | None) -> bool:
    """判定一个 Cell 的 Cell_Sub_Dataset 是否为空（R3-AC6）。

    Compiler 仅为**非空** Cell 写出 Cell_Sub_Dataset，因此"文件缺失"等价于"该 Cell
    为空"。文件存在但无任何非空白行（无 JSONL 记录）同样视为空。读取过程中遭遇
    I/O 错误时保守视为空（跳过），不在此处中断整次网格训练。

    本判定只读文本、无 ML 依赖，保证导入安全与无依赖环境可运行。
    """

    if path is None:
        return True
    p = Path(path)
    if not p.exists():
        return True
    try:
        with p.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    return False
    except OSError:
        return True
    return True


# ---------------------------------------------------------------------------
# 编排入口
# ---------------------------------------------------------------------------


def run_grid_training(
    cells: list[CellSpec],
    params: TrainingParams,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    cwd: Path | None = None,
    backend: TrainingBackend | None = None,
    offline: OfflineMode | None = None,
    base_model_available_check=None,
    stdout: TextIO | None = None,
) -> GridTrainingResult:
    """为每个非空 Cell 训练并导出一个独立 LoRA_Adapter（网格训练，R3）。

    步骤（见模块文档）：装配（探测后端 → 打印所选后端 → 校验五参数 → 离线守卫，
    全部复用 ``run_training`` 既有逻辑）→ 按训练顺序排序 Cell（Kernel 优先、分区内
    Level 升序）→ 逐 Cell 训练（空 Cell 跳过并记录、非空 Cell fresh-load →
    train → export）→ 汇总为 :class:`GridTrainingResult`。

    :param cells: 待训练 Cell 规格列表；Kernel 编码为 ``zone=None, level=0``。
        仅含 Kernel 时退化为单适配器训练（R3-AC9）。
    :param params: 来自 Ego_Config 的训练参数，提供 ``quantization`` 与五个 LoRA 参数。
    :param model_id: 基座模型标识，MVP 固定为 ``qwen3.5-0.8B``（R3-AC1）。
    :param cwd: 进程工作目录，默认 :meth:`pathlib.Path.cwd`；各 Cell 适配器临时落于
        ``{cwd}/.ego/cells_out/`` 下（隐私约束 R9-AC4：产物仅落于工作目录内）。
    :param backend: 显式指定的 Training_Backend；默认由 :func:`detect_backend` 探测
        （优先级不变，R6-AC5）。
    :param offline: 离线模式状态；默认回退到 :func:`get_active_offline_mode` 注册的
        进程内活动模式（``None`` 等价于在线模式）。
    :param base_model_available_check: 离线模式下判定基座权重是否可用的回调
        （签名 ``check(model_id) -> bool``）；默认扫描本地 HuggingFace 缓存。
    :param stdout: 后端选择记录、跳过记录与进度记录的目标输出流，默认 ``sys.stdout``。
    :returns: :class:`GridTrainingResult`，含每个已训练 Cell 的 ``(zone, level,
        output_path)`` 与所选后端名称/硬件/Backend_Identifier（R3-AC8）。
    :raises TrainerError: 五个 LoRA 参数缺失或非法（R8-AC9）；或某 Cell 训练时基座
        加载失败/计算错误（OOM/NaN/Inf 等），消息含失败 Cell 标识（R3-AC7）。
    :raises OfflineMissingWeightsError: 离线模式下基座权重缓存缺失（R9-AC7，由
        ``privacy.offline`` 抛出，本模块不拦截）。
    """

    base_cwd = Path.cwd() if cwd is None else Path(cwd)
    cells_out_root = base_cwd / EGO_DIR_NAME / CELLS_OUT_DIR_NAME

    # 1. 探测后端并打印所选后端（复用 run_training 逻辑，R6-AC5 / R8-AC2）。
    selected = backend if backend is not None else detect_backend()
    print_summary(
        f"Selected backend: {selected.name} ({selected.hardware})",
        stream=stdout,
    )

    # 2. 校验五个 LoRA 参数（缺失/非法 → 退出码 1，R8-AC9）——复用 run_training 校验。
    _validate_training_params(params)

    # 3. 离线模式下加载基座前先检查权重可用性；缺失由 privacy.offline 抛
    #    OfflineMissingWeightsError，交由 CLI 顶层按 R9-AC7 处理（本模块不拦截）。
    active_offline = offline if offline is not None else get_active_offline_mode()
    if active_offline is not None and active_offline.enabled:
        available = _base_model_available(
            selected, model_id, check=base_model_available_check
        )
        active_offline.require_base_model_available(available)

    # 4. 排序为训练顺序：Kernel 优先，再各分区 Level 升序（R3-AC2）。
    ordered_cells = sorted(cells, key=_training_order_key)

    # 5. 逐 Cell 训练；进度回调复用 run_training 的固定格式记录（R8-AC8）。
    progress_callback = make_progress_callback(stream=stdout)
    trained: list[TrainedCell] = []

    for cell in ordered_cells:
        label = _cell_label(cell)

        # 空 Cell：跳过并记录标识与原因，不产出文件（R3-AC6）。
        if _sub_dataset_is_empty(cell.sub_dataset_path):
            print_summary(
                f"Skipping empty cell {label}: Cell_Sub_Dataset 为空，"
                f"不为该 Cell 产出 LoRA_Adapter。",
                stream=stdout,
            )
            continue

        cell_output = _cell_output_path(cells_out_root, cell)

        # fresh-load-per-cell 隔离（R3-AC1/AC3）：每 Cell 独立加载冻结 Base_Model，
        # 训练前以固定种子重置随机源（R3-AC5），训练并仅导出适配器张量（R3-AC4）。
        try:
            set_deterministic_seed(_SEED)
            model = selected.load_base_model(model_id, params.quantization)
            selected.train(model, cell.sub_dataset_path, params, progress_callback)
            cell_output.parent.mkdir(parents=True, exist_ok=True)
            selected.export_lora_safetensors(model, cell_output)
        except TrainerError as exc:
            # 计算错误（OOM/NaN/Inf）或基座加载失败已是受控 TrainerError：重抛并
            # 在消息中附上失败 Cell 标识（R3-AC7），保持退出码 1 语义不变。
            raise TrainerError(f"Cell {label} 训练失败：{exc}") from exc
        except EgoError:
            # 其它受控错误（如离线缺权重）原样上抛，保持各自退出码语义不被覆盖。
            raise
        except Exception as exc:  # noqa: BLE001 - 归一化未受控底层异常
            raise TrainerError(f"Cell {label} 训练失败：{exc}") from exc

        trained.append(
            TrainedCell(zone=cell.zone, level=cell.level, output_path=cell_output)
        )

    # 6. 汇总：每个已训练 Cell 的 (zone, level, output_path) + 所选后端信息（R3-AC8）。
    return GridTrainingResult(
        cells=tuple(trained),
        backend_name=selected.name,
        backend_hardware=selected.hardware,
        backend_identifier=resolve_identifier(selected),
    )
