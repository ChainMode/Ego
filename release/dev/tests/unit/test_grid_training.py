"""`ego_pod.trainer.grid.run_grid_training` 的单元测试（任务 3.3）。

本文件为「按 Cell 训练编排」（Grid_Training，见 design.md §「Components and
Interfaces → 3. Trainer → 3.2 run_grid_training」）提供无 ML 依赖的示例/边界单元
测试，覆盖以下验收标准：

* **R3-AC2（训练顺序）**：``run_grid_training`` 先训练 Kernel（Level 0），再逐个
  Functional_Zone，同一分区内 Level 由低到高升序——与传入 ``cells`` 的顺序无关。
* **R3-AC6（空 Cell 跳过）**：Cell_Sub_Dataset 缺失/为空的 Cell 被跳过——不产出
  ``TrainedCell``、在标准输出记录被跳过 Cell 的标识与原因、且不为其写出适配器文件。
* **R3-AC8（结果字段与向 Vault 报告）**：``GridTrainingResult.cells`` 为每个已训练
  （非空）Cell 携带 ``TrainedCell(zone, level, output_path)``，坐标正确、输出路径落
  于 ``.ego/cells_out/...``；``backend_name`` / ``backend_hardware`` /
  ``backend_identifier`` 取自所选后端。
* **R3-AC9（单 Cell 退化）**：仅含一个 Kernel CellSpec 时恰好产出一个
  ``TrainedCell``（Kernel），行为等同于单适配器训练。

为保持无 ML 依赖（设计「决策 D」），本测试不导入 ``torch`` / ``peft``，而是注入一个
**记录调用序列的假 Training_Backend**（结构上满足 ``TrainingBackend`` Protocol），其
``load_base_model`` / ``train`` / ``export_lora_safetensors`` 均为不触碰任何 ML 依赖
的轻量操作——``export`` 仅写出一个小占位文件以验证落点。该假后端的模式改编自
``tests/unit/test_pipeline_compat.py`` 的 ``_MemoryBackend``。

**Feature: ego-layered-adapter**
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from ego_pod.config.schema import TrainingParams
from ego_pod.privacy.offline import OfflineMode
from ego_pod.trainer.base import BackendIdentifier, Hardware
from ego_pod.trainer.cells import CellSpec, GridTrainingResult, TrainedCell
from ego_pod.trainer.grid import run_grid_training

# ---------------------------------------------------------------------------
# 公共辅助
# ---------------------------------------------------------------------------

#: Backend_Identifier 的固定取值集合（R6-AC5、需求 7.1/7.4）。
_VALID_IDENTIFIERS = {"unsloth", "peft-cuda", "mlx", "peft-cpu", "stub"}


def _make_params() -> TrainingParams:
    """构造一组合法的训练参数（五个 LoRA 参数均合法，避免 ``_validate_*`` 短路）。"""

    return TrainingParams(
        method="qlora",
        quantization="4bit",
        lora_rank=16,
        lora_alpha=32,
        learning_rate=2e-4,
        epochs=1,
        batch_size=4,
    )


def _write_sub_dataset(path: Path, *lines: str) -> Path:
    """写出一个非空 Cell_Sub_Dataset（JSONL，UTF-8、LF），返回其路径。

    未显式给出语料行时，写入两条最小合法语料行（字段沿用 MVP Conversation_Corpus
    约定 ``instruction`` / ``input`` / ``output`` / ``category``）。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    if not lines:
        lines = (
            '{"instruction":"问候","input":"","output":"你好","category":"skill"}',
            '{"instruction":"应答","input":"","output":"在的","category":"action"}',
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class _RecordingBackend:
    """记录调用序列的假 Training_Backend（无 torch/peft 依赖）。

    结构上满足 ``TrainingBackend`` Protocol。按调用先后记录：

    * :attr:`load_calls`：每次 ``load_base_model`` 的 ``model_id``；
    * :attr:`train_paths`：每次 ``train`` 的 ``corpus_path``（即该 Cell 的
      Cell_Sub_Dataset 路径）——其**顺序即训练顺序**，供 R3-AC2 断言；
    * :attr:`export_paths`：每次 ``export_lora_safetensors`` 的目标路径。

    ``export`` 写出一个小占位文件（非空字节），使"非空 Cell 产出适配器文件、空
    Cell 不产文件"可被文件系统断言。
    """

    name = "HuggingFace PEFT"
    hardware: Hardware = "CPU"
    identifier: BackendIdentifier = "peft-cpu"

    def __init__(self) -> None:
        self.load_calls: list[str] = []
        self.train_paths: list[Path] = []
        self.export_paths: list[Path] = []

    @classmethod
    def is_available(cls) -> bool:
        return True

    def load_base_model(self, model_id: str, quantization: str) -> Any:
        self.load_calls.append(model_id)
        return {"model_id": model_id, "quantization": quantization}

    def train(self, model: Any, corpus_path: Path, params: TrainingParams, progress_callback) -> None:
        self.train_paths.append(Path(corpus_path))
        # 至少触发一次进度回调（与 StubBackend / _MemoryBackend 的调用约定一致）。
        progress_callback(epoch=params.epochs, step=params.epochs, loss=0.0)

    def export_lora_safetensors(self, model: Any, output_path: Path) -> None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"placeholder-adapter")  # 占位，仅验证落点。
        self.export_paths.append(out)


def _cells_out_root(cwd: Path) -> Path:
    """各 Cell 适配器的临时导出根目录（与 ``trainer.grid`` 约定一致）。"""

    return cwd / ".ego" / "cells_out"


# ===========================================================================
# R3-AC2：训练顺序为 Kernel → 分区（Level 升序）
# ===========================================================================


def test_training_order_kernel_first_then_zones_level_ascending(tmp_path: Path) -> None:
    """传入非排序的 Cell 列表，训练顺序仍为 Kernel → 分区按名/Level 升序。

    **Validates: Requirements 3.2**
    """
    cells_dir = tmp_path / "cells"
    kernel_path = _write_sub_dataset(cells_dir / "kernel.jsonl")
    coding_path = _write_sub_dataset(cells_dir / "coding" / "L1.jsonl")
    knowledge_path = _write_sub_dataset(cells_dir / "knowledge" / "L1.jsonl")

    # 故意以非排序顺序传入（knowledge 在前、kernel 居中、coding 在后）。
    cells = [
        CellSpec(zone="knowledge", level=1, sub_dataset_path=knowledge_path),
        CellSpec(zone=None, level=0, sub_dataset_path=kernel_path),
        CellSpec(zone="coding", level=1, sub_dataset_path=coding_path),
    ]

    backend = _RecordingBackend()
    result = run_grid_training(
        cells,
        _make_params(),
        cwd=tmp_path,
        backend=backend,
        offline=OfflineMode(enabled=False),
        stdout=io.StringIO(),
    )

    # 训练调用顺序（按 train 的 corpus_path）= Kernel → coding(L1) → knowledge(L1)。
    assert backend.train_paths == [kernel_path, coding_path, knowledge_path]

    # 已训练 Cell 的产出顺序同样为 Kernel 优先、分区按名升序（R3-AC2 / R3-AC8）。
    assert [(c.zone, c.level) for c in result.cells] == [
        (None, 0),
        ("coding", 1),
        ("knowledge", 1),
    ]


# ===========================================================================
# R3-AC6：空 Cell 跳过、不产 TrainedCell、不写文件、打印跳过原因
# ===========================================================================


def test_empty_cell_is_skipped_with_message_and_no_file(tmp_path: Path) -> None:
    """Cell_Sub_Dataset 缺失/为空的 Cell 被跳过：无 TrainedCell、有跳过记录、无文件。

    **Validates: Requirements 3.6**
    """
    cells_dir = tmp_path / "cells"
    kernel_path = _write_sub_dataset(cells_dir / "kernel.jsonl")
    # 缺失文件（从未写出）与空文件（仅空白）两种"空 Cell"形态各取其一。
    missing_path = cells_dir / "coding" / "L1.jsonl"  # 不存在
    empty_path = cells_dir / "knowledge" / "L1.jsonl"
    empty_path.parent.mkdir(parents=True, exist_ok=True)
    empty_path.write_text("   \n\n", encoding="utf-8")  # 仅空白 → 视为空

    cells = [
        CellSpec(zone=None, level=0, sub_dataset_path=kernel_path),
        CellSpec(zone="coding", level=1, sub_dataset_path=missing_path),
        CellSpec(zone="knowledge", level=1, sub_dataset_path=empty_path),
    ]

    backend = _RecordingBackend()
    out = io.StringIO()
    result = run_grid_training(
        cells,
        _make_params(),
        cwd=tmp_path,
        backend=backend,
        offline=OfflineMode(enabled=False),
        stdout=out,
    )

    # 仅 Kernel 被训练；两个空分区 Cell 不出现在结果中（R3-AC6）。
    assert [(c.zone, c.level) for c in result.cells] == [(None, 0)]

    # 后端未对空 Cell 执行训练/导出（仅 Kernel 一次）。
    assert backend.train_paths == [kernel_path]
    assert len(backend.export_paths) == 1

    # 标准输出记录两个被跳过 Cell 的标识与原因（R3-AC6）。
    printed = out.getvalue()
    assert "Skipping empty cell (coding, L1)" in printed
    assert "Skipping empty cell (knowledge, L1)" in printed

    # 不为被跳过 Cell 写出任何适配器文件。
    root = _cells_out_root(tmp_path)
    assert not (root / "coding" / "L1.safetensors").exists()
    assert not (root / "knowledge" / "L1.safetensors").exists()


# ===========================================================================
# R3-AC8：GridTrainingResult 字段、Cell 坐标/输出路径、后端信息
# ===========================================================================


def test_result_fields_and_cell_output_paths(tmp_path: Path) -> None:
    """``GridTrainingResult`` 携带各已训练 Cell 的坐标/输出路径与所选后端信息。

    **Validates: Requirements 3.8**
    """
    cells_dir = tmp_path / "cells"
    kernel_path = _write_sub_dataset(cells_dir / "kernel.jsonl")
    coding_path = _write_sub_dataset(cells_dir / "coding" / "L1.jsonl")

    cells = [
        CellSpec(zone=None, level=0, sub_dataset_path=kernel_path),
        CellSpec(zone="coding", level=1, sub_dataset_path=coding_path),
    ]

    backend = _RecordingBackend()
    result = run_grid_training(
        cells,
        _make_params(),
        cwd=tmp_path,
        backend=backend,
        offline=OfflineMode(enabled=False),
        stdout=io.StringIO(),
    )

    assert isinstance(result, GridTrainingResult)

    # 后端信息取自所选后端（R3-AC8）。
    assert result.backend_name == "HuggingFace PEFT"
    assert result.backend_hardware == "CPU"
    assert result.backend_identifier == "peft-cpu"
    assert result.backend_identifier in _VALID_IDENTIFIERS

    # 每个已训练 Cell 为 TrainedCell，坐标正确、输出路径落于 .ego/cells_out/...。
    root = _cells_out_root(tmp_path)
    by_coord = {(c.zone, c.level): c for c in result.cells}
    assert set(by_coord) == {(None, 0), ("coding", 1)}

    kernel_cell = by_coord[(None, 0)]
    coding_cell = by_coord[("coding", 1)]
    assert isinstance(kernel_cell, TrainedCell)
    assert kernel_cell.output_path == root / "kernel.safetensors"
    assert coding_cell.output_path == root / "coding" / "L1.safetensors"

    # 报告的输出路径确实是 export 写出的非空适配器文件。
    assert kernel_cell.output_path.exists()
    assert kernel_cell.output_path.stat().st_size > 0
    assert coding_cell.output_path.exists()
    assert coding_cell.output_path.stat().st_size > 0


# ===========================================================================
# R3-AC9：仅含 Kernel ⇒ 恰好一个 TrainedCell（单适配器退化）
# ===========================================================================


def test_single_kernel_cell_parity(tmp_path: Path) -> None:
    """仅含一个 Kernel CellSpec 时恰好产出一个 TrainedCell（Kernel）。

    **Validates: Requirements 3.9**
    """
    kernel_path = _write_sub_dataset(tmp_path / "cells" / "kernel.jsonl")
    cells = [CellSpec(zone=None, level=0, sub_dataset_path=kernel_path)]

    backend = _RecordingBackend()
    result = run_grid_training(
        cells,
        _make_params(),
        cwd=tmp_path,
        backend=backend,
        offline=OfflineMode(enabled=False),
        stdout=io.StringIO(),
    )

    # 恰好一个已训练 Cell，且为 Kernel（zone=None, level=0）。
    assert len(result.cells) == 1
    only = result.cells[0]
    assert (only.zone, only.level) == (None, 0)
    assert only.output_path == _cells_out_root(tmp_path) / "kernel.safetensors"

    # 后端只被驱动了一次完整的 load → train → export 序列（单适配器路径）。
    assert backend.load_calls == ["qwen3.5-0.8B"]
    assert backend.train_paths == [kernel_path]
    assert len(backend.export_paths) == 1
