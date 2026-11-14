"""`ego_pod.trainer.pipeline` 对外兼容性单元测试（任务 9.5）。

本文件验证「为 ``TrainingResult`` 新增 ``backend_identifier`` 字段、并由
``run_training`` 经 ``resolve_identifier`` 回传」这一改动对 MVP 既有调用方保持
**向后兼容**（design.md §8 决策、tasks.md 任务 9.5）：

* **R10-AC2 / R7-AC6**：``TrainingResult`` 仍可只用 MVP 的三个原始字段
  （``output_path`` / ``backend_name`` / ``backend_hardware``）构造，新增的
  ``backend_identifier`` 带默认值 ``"stub"``，使既有构造点透明不破坏；同时新字段
  可被显式赋值并取自固定标识集合。
* **R10-AC2**：``run_training`` 的对外签名兼容——既有必填参数仅 ``corpus_path``
  与 ``params`` 两个位置参数，本特性**未**新增任何必填参数（所有新增/既有可选项
  均为带默认值的关键字参数）。

无需任何 ML 依赖：仅检视 ``TrainingResult`` dataclass 字段与 ``run_training`` 的
``inspect.signature``，并以一个最小内存后端（无 torch/peft）驱动一次端到端
``run_training`` 验证返回的 ``backend_identifier``。

**Feature: ego-pod-real-trainer**
"""

from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path
from typing import Any

from ego_pod.config.schema import TrainingParams
from ego_pod.privacy.offline import OfflineMode
from ego_pod.trainer.base import BackendIdentifier, Hardware
from ego_pod.trainer.pipeline import TrainingResult, run_training

# ---------------------------------------------------------------------------
# 公共辅助
# ---------------------------------------------------------------------------

#: Backend_Identifier 的固定取值集合（R7-AC1、R7-AC4）。
_VALID_IDENTIFIERS = {"unsloth", "peft-cuda", "mlx", "peft-cpu", "stub"}


def _make_params() -> TrainingParams:
    return TrainingParams(
        method="qlora",
        quantization="4bit",
        lora_rank=16,
        lora_alpha=32,
        learning_rate=2e-4,
        epochs=2,
        batch_size=4,
    )


class _MemoryBackend:
    """最小内存后端：结构上满足 ``TrainingBackend``，不触碰任何 ML 依赖。

    用于驱动一次完整 ``run_training``（探测注入、参数校验、加载、训练、导出），
    以验证返回的 :class:`TrainingResult` 字段。``identifier`` 缺省为 ``"peft-cpu"``，
    用于断言 ``run_training`` 经 ``resolve_identifier`` 正确回填。
    """

    name = "HuggingFace PEFT"
    hardware: Hardware = "CPU"
    identifier: BackendIdentifier = "peft-cpu"

    @classmethod
    def is_available(cls) -> bool:
        return True

    def load_base_model(self, model_id: str, quantization: str) -> Any:
        return {"model_id": model_id, "quantization": quantization}

    def train(self, model: Any, corpus_path: Path, params: TrainingParams, progress_callback) -> None:
        progress_callback(epoch=params.epochs, step=params.epochs, loss=0.0)

    def export_lora_safetensors(self, model: Any, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"")  # 占位，仅验证落点；内容与本测试无关。


# ===========================================================================
# TrainingResult 字段兼容性（R7-AC6、R10-AC2）
# ===========================================================================


def test_training_result_constructs_with_three_original_fields() -> None:
    """``TrainingResult`` 仍可只用三个原始字段构造，``backend_identifier`` 默认 ``"stub"``。

    **Validates: Requirements 7.6, 10.2**
    """
    result = TrainingResult(
        output_path=Path("/tmp/.ego/lora.safetensors"),
        backend_name="stub",
        backend_hardware="STUB",
    )

    assert result.output_path == Path("/tmp/.ego/lora.safetensors")
    assert result.backend_name == "stub"
    assert result.backend_hardware == "STUB"
    # 新增字段带默认值，使既有三字段构造点透明不破坏（R10-AC2）。
    assert result.backend_identifier == "stub"


def test_training_result_accepts_explicit_backend_identifier() -> None:
    """``backend_identifier`` 可被显式赋值，取自固定标识集合。

    **Validates: Requirements 7.6, 10.2**
    """
    result = TrainingResult(
        output_path=Path("/tmp/.ego/lora.safetensors"),
        backend_name="HuggingFace PEFT",
        backend_hardware="CUDA",
        backend_identifier="peft-cuda",
    )

    assert result.backend_identifier == "peft-cuda"
    assert result.backend_identifier in _VALID_IDENTIFIERS


def test_training_result_original_fields_unchanged() -> None:
    """三个原始字段名仍存在且顺序在前，``backend_identifier`` 为唯一新增字段。

    钉死「对外字段兼容」：既有字段名/相对顺序不变，新增字段附带默认值置于末尾。

    **Validates: Requirements 7.6, 10.2**
    """
    field_names = [f.name for f in dataclasses.fields(TrainingResult)]

    # 三个原始字段仍存在且为前三位（既有按位置/关键字构造均不破坏）。
    assert field_names[:3] == [
        "output_path",
        "backend_name",
        "backend_hardware",
    ]
    # 唯一新增字段为 backend_identifier。
    assert set(field_names) == {
        "output_path",
        "backend_name",
        "backend_hardware",
        "backend_identifier",
    }

    # 新增字段带默认值（确保旧的三参数构造点仍可用）。
    identifier_field = next(
        f for f in dataclasses.fields(TrainingResult) if f.name == "backend_identifier"
    )
    assert identifier_field.default == "stub"


# ===========================================================================
# run_training 签名兼容性（R10-AC2）
# ===========================================================================


def test_run_training_has_no_new_required_params() -> None:
    """``run_training`` 仅 ``corpus_path`` / ``params`` 为必填，无新增必填参数。

    本特性新增的接线（offline / base_model_available_check / 返回 identifier 等）
    均以带默认值的关键字参数引入，既有 ``run_training(corpus_path, params)`` 调用
    方式不被破坏。

    **Validates: Requirements 10.2**
    """
    sig = inspect.signature(run_training)
    required = [
        name
        for name, p in sig.parameters.items()
        if p.default is inspect.Parameter.empty
        and p.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]

    assert required == ["corpus_path", "params"]


def test_run_training_optional_params_have_defaults() -> None:
    """本特性相关的可选参数均带默认值（不破坏既有调用方式）。

    **Validates: Requirements 10.2**
    """
    sig = inspect.signature(run_training)
    params = sig.parameters

    for name in ("model_id", "cwd", "output_path", "backend", "offline",
                 "base_model_available_check", "stdout"):
        assert name in params, f"缺失预期可选参数：{name}"
        assert params[name].default is not inspect.Parameter.empty, (
            f"参数 {name} 应带默认值"
        )


def test_run_training_returns_resolved_backend_identifier(tmp_path: Path) -> None:
    """``run_training`` 经 ``resolve_identifier`` 把所选后端标识回填到 ``TrainingResult``。

    用最小内存后端（无 ML 依赖）注入，断言返回的 ``backend_identifier`` 等于后端
    文档化标识，且原始三字段语义不变（R7-AC1/AC6、R10-AC2）。

    **Validates: Requirements 7.6, 10.2**
    """
    backend = _MemoryBackend()

    result = run_training(
        tmp_path / "corpus.jsonl",
        _make_params(),
        cwd=tmp_path,
        backend=backend,
        offline=OfflineMode(enabled=False),
    )

    assert isinstance(result, TrainingResult)
    assert result.backend_identifier == "peft-cpu"
    assert result.backend_identifier in _VALID_IDENTIFIERS
    assert result.backend_name == "HuggingFace PEFT"
    assert result.backend_hardware == "CPU"
    # 导出落点为默认 {cwd}/.ego/lora.safetensors。
    assert result.output_path == tmp_path / ".ego" / "lora.safetensors"
    assert result.output_path.exists()
