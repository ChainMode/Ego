"""`ego_pod.trainer.peft_backend` 的加载精度回退与摘要格式单元测试（任务 5.8）。

聚焦属性测试之外的固定样例与边界（design.md §Testing Strategy「单元测试聚焦
固定样例与边界」），覆盖以下三类行为：

* **加载摘要单行格式与 CPU 精度回退**（R1-AC3、R1-AC4）：``PeftCpuBackend``
  的 ``hardware == "CPU"`` 使 ``use_cuda`` / ``use_4bit`` 恒为 ``False``，故
  :meth:`~ego_pod.trainer.peft_backend._PeftBackendBase.load_base_model` 走"以
  CPU 受支持精度加载、不使用 4-bit"的回退分支，并在标准输出记录一行含
  ``model_id`` / 目标设备 ``CPU`` / 实际加载精度（``precision=...``）的摘要。
* **``_dtype_name`` 映射**：torch dtype → 简洁可读名
  （``float32`` → ``fp32`` / ``float16`` → ``fp16`` / ``bfloat16`` → ``bf16``），
  供加载摘要的人类可读精度输出使用（R1-AC4）。
* **训练完成摘要单行格式**（R4-AC4）：以 :class:`FakeModel` / :class:`FakeTokenizer`
  驱动 :meth:`~ego_pod.trainer.peft_backend._PeftBackendBase.train`，断言训练完成
  后在标准输出记录一行含已完成 epoch 数、累计步数与最终损失的摘要。

依赖门控：本文件**无需安装** ``transformers`` / ``peft`` / ``bitsandbytes``。
``torch`` 为默认依赖（已安装），故 ``_cpu_bf16_supported`` / ``_dtype_name``
按真实 torch 工作；加载路径通过在 ``sys.modules`` 临时注入一个**假 transformers
模块**（其 ``from_pretrained`` 返回轻量假对象）来确定性地驱动，不触达任何网络。
训练路径用 ``tests/_doubles.py`` 的无依赖假对象驱动，并给假模型挂上
``peft_config`` 属性使 ``_is_peft_model`` 为真，从而跳过对未安装的 ``peft`` 的导入。

**Feature: ego-pod-real-trainer**
"""

from __future__ import annotations

import contextlib
import sys
import types
from pathlib import Path
from typing import Iterator

import pytest

from _doubles import FakeModel, FakeTokenizer

from ego_pod.config.schema import TrainingParams
from ego_pod.trainer.peft_backend import (
    PeftCpuBackend,
    _PeftBackendBase,
    _dtype_name,
)


# ---------------------------------------------------------------------------
# 注入"返回轻量假对象"的假 transformers 模块
# ---------------------------------------------------------------------------


class _FakeLoadedModel:
    """``AutoModelForCausalLM.from_pretrained`` 返回的轻量假模型。

    加载摘要测试只验证打印行为，不使用该返回对象的能力，故仅记录构造入参。
    """

    def __init__(self, model_id: str, **kwargs: object) -> None:
        self.model_id = model_id
        self.kwargs = kwargs


class _FakeLoadedTokenizer:
    """``AutoTokenizer.from_pretrained`` 返回的轻量假 tokenizer。"""

    def __init__(self, model_id: str, **kwargs: object) -> None:
        self.model_id = model_id
        self.kwargs = kwargs


def _make_fake_transformers() -> types.ModuleType:
    """构造一个假 ``transformers`` 模块，其加载入口返回轻量假对象。

    ``AutoTokenizer`` / ``AutoModelForCausalLM`` 暴露 ``from_pretrained`` 静态方法
    并分别返回 :class:`_FakeLoadedTokenizer` / :class:`_FakeLoadedModel`；
    ``BitsAndBytesConfig`` 一并提供以兼容 CUDA 4-bit 分支（CPU 路径不会触达）。
    """

    module = types.ModuleType("transformers")

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> _FakeLoadedModel:
            return _FakeLoadedModel(model_id, **kwargs)

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> _FakeLoadedTokenizer:
            return _FakeLoadedTokenizer(model_id, **kwargs)

    module.AutoModelForCausalLM = _FakeAutoModel  # type: ignore[attr-defined]
    module.AutoTokenizer = _FakeAutoTokenizer  # type: ignore[attr-defined]
    module.BitsAndBytesConfig = object  # type: ignore[attr-defined]
    return module


@contextlib.contextmanager
def _inject_fake_transformers() -> Iterator[None]:
    """在 ``sys.modules`` 临时注入返回轻量假对象的假 ``transformers`` 模块。

    退出时精确还原 ``sys.modules['transformers']`` 的原状态（原本缺失则删除，
    原本存在则恢复），避免污染其它测试对"transformers 是否可用"的判定。
    """

    fake = _make_fake_transformers()
    had_previous = "transformers" in sys.modules
    previous = sys.modules.get("transformers")
    sys.modules["transformers"] = fake
    try:
        yield
    finally:
        if had_previous:
            sys.modules["transformers"] = previous  # type: ignore[assignment]
        else:
            sys.modules.pop("transformers", None)


# ===========================================================================
# 加载摘要单行格式与 CPU 精度回退（R1-AC3、R1-AC4）
# ===========================================================================


def test_cpu_load_summary_line_format(capsys: pytest.CaptureFixture[str]) -> None:
    """CPU 后端加载基座 → 打印含 model_id / CPU / precision= 的单行摘要（R1-AC3、R1-AC4）。

    ``PeftCpuBackend`` 的 ``hardware == "CPU"`` 使 ``use_4bit`` 恒为 ``False``，
    故走精度回退分支（不使用 bitsandbytes 4-bit 量化），摘要记录实际加载精度。
    """
    backend = PeftCpuBackend()
    model_id = "qwen3.5-0.8B"

    with _inject_fake_transformers():
        model, tokenizer = backend.load_base_model(model_id, "4bit")

    # 返回 (model, tokenizer) 复合对象，供 train/export 使用。
    assert isinstance(model, _FakeLoadedModel)
    assert isinstance(tokenizer, _FakeLoadedTokenizer)

    out = capsys.readouterr().out
    summary_lines = [
        line for line in out.splitlines() if line.startswith("Loaded base model:")
    ]
    # 恰好一行加载摘要。
    assert len(summary_lines) == 1
    line = summary_lines[0]

    # 摘要单行包含 model_id、目标设备 CPU、与实际加载精度（precision=）。
    assert f"model_id={model_id}" in line
    assert "device=CPU" in line
    assert "precision=" in line
    # CPU 回退不使用 4-bit 量化。
    assert "4bit" not in line.split("quantization=", 1)[1]


def test_cpu_load_precision_token_is_fp32_or_bf16(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CPU 回退记录的实际精度令牌为 ``fp32`` 或 ``bf16``（R1-AC3）。

    torch 已安装，故 ``_cpu_bf16_supported`` / ``_dtype_name`` 按真实 torch 工作：
    精度回退优先 bf16（若 CPU 支持），否则 fp32——二者之一必然出现在摘要里。
    """
    backend = PeftCpuBackend()

    with _inject_fake_transformers():
        backend.load_base_model("qwen3.5-0.8B", "4bit")

    out = capsys.readouterr().out
    line = next(
        line for line in out.splitlines() if line.startswith("Loaded base model:")
    )
    precision_token = line.split("precision=", 1)[1].rstrip(") ")
    assert precision_token in {"fp32", "bf16"}


# ===========================================================================
# _dtype_name 映射（R1-AC4）
# ===========================================================================


def test_dtype_name_maps_known_torch_dtypes() -> None:
    """torch dtype → 简洁可读名：float32→fp32 / float16→fp16 / bfloat16→bf16。"""
    import torch

    assert _dtype_name(torch.float32) == "fp32"
    assert _dtype_name(torch.float16) == "fp16"
    assert _dtype_name(torch.bfloat16) == "bf16"


def test_dtype_name_falls_back_to_short_name_for_unknown() -> None:
    """未识别 dtype 回退到去掉 ``torch.`` 前缀的简洁名（如 ``int8``）。"""
    import torch

    # int8 不在已知映射内，回退到去前缀的短名。
    assert _dtype_name(torch.int8) == "int8"


# ===========================================================================
# 训练完成摘要单行格式（R4-AC4）
# ===========================================================================


def _make_params(*, epochs: int = 2, batch_size: int = 2) -> TrainingParams:
    """构造一组合法的训练参数（仅 epochs / batch_size 影响本测试的步数）。"""
    return TrainingParams(
        method="qlora",
        quantization="4bit",
        lora_rank=8,
        lora_alpha=16,
        learning_rate=1e-4,
        epochs=epochs,
        batch_size=batch_size,
    )


def _write_corpus(tmp_path: Path) -> Path:
    """写入一个小型合法 Conversation_Corpus（JSONL）。"""
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text(
        '{"instruction": "指令一", "input": "", "output": "回复一", "category": "thought"}\n'
        '{"instruction": "指令二", "input": "上下文", "output": "回复二", "category": "skill"}\n'
        '{"instruction": "指令三", "input": "", "output": "回复三", "category": "action"}\n',
        encoding="utf-8",
    )
    return corpus_path


class _ProgressRecorder:
    """以关键字参数记录每次 ``progress_callback`` 调用（贴合 loop 的调用约定）。"""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int, float]] = []

    def __call__(self, *, epoch: int, step: int, loss: float) -> None:
        self.calls.append((epoch, step, loss))


def test_training_complete_summary_line_format(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """train 完成后打印含 epochs / steps / final_loss 的单行训练完成摘要（R4-AC4）。

    用 :class:`FakeModel` / :class:`FakeTokenizer` 驱动训练循环：给假模型挂上
    ``peft_config`` 属性使 ``_is_peft_model`` 为真，从而跳过对未安装的 ``peft``
    的导入（``get_peft_model`` 分支）。假模型每步返回固定有限 loss=1.0，循环
    逐 epoch 回调并在完成后打印摘要。
    """
    backend = _PeftBackendBase()

    tokenizer = FakeTokenizer(mode="char")
    model = FakeModel(loss_value=1.0)
    # 挂上 peft_config 使 _is_peft_model 为真 → 跳过 peft 导入分支。
    model.peft_config = {"default": object()}

    corpus_path = _write_corpus(tmp_path)
    params = _make_params(epochs=2, batch_size=2)
    recorder = _ProgressRecorder()

    backend.train((model, tokenizer), corpus_path, params, recorder)

    # 逐 epoch 回调：恰好 epochs 次，epoch 编号 1..epochs。
    assert [call[0] for call in recorder.calls] == [1, 2]

    out = capsys.readouterr().out
    summary_lines = [
        line for line in out.splitlines() if line.startswith("Training complete:")
    ]
    # 恰好一行训练完成摘要。
    assert len(summary_lines) == 1
    line = summary_lines[0]

    # 单行包含已完成 epoch 数、累计步数与最终损失。
    assert "epochs=2" in line
    assert "steps=" in line
    assert "final_loss=" in line
    # 最终损失等于假模型的固定有限 loss（格式化为 6 位小数）。
    assert "final_loss=1.000000" in line


def test_training_complete_summary_step_count_matches_callback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """训练完成摘要里的累计步数与最后一次进度回调上报的 step 一致。"""
    backend = _PeftBackendBase()

    tokenizer = FakeTokenizer(mode="char")
    model = FakeModel(loss_value=0.5)
    model.peft_config = {"default": object()}

    corpus_path = _write_corpus(tmp_path)
    params = _make_params(epochs=1, batch_size=2)
    recorder = _ProgressRecorder()

    backend.train((model, tokenizer), corpus_path, params, recorder)

    last_step = recorder.calls[-1][1]
    out = capsys.readouterr().out
    line = next(
        line for line in out.splitlines() if line.startswith("Training complete:")
    )
    assert f"steps={last_step}" in line
