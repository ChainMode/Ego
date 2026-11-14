"""`ego_pod.trainer.stub.StubBackend` 的单元测试。

覆盖设计文档「多后端 Trainer 抽象 / Stub_Backend 行为」一节与需求：

* R8-AC1：``is_available()`` 始终返回 ``True``（最终回退）；
* R8-AC5：``train`` 跳过权重更新，``load_base_model`` 返回占位对象；
* R8-AC6：``export_lora_safetensors`` 产出可被 SafeTensors 库读回的合法权重文件；
* R8-AC8：``train`` 至少每 epoch 触发一次进度回调；
* R12-AC9：``StubBackend.name == "stub"``，作为 ``manifest.json`` 中
  ``training_backend`` stub 标记的来源。

测试使用 ``tmp_path`` 写入临时文件，不依赖任何真实 ML 依赖或网络。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import load_file

from ego_pod.config.schema import TrainingParams
from ego_pod.trainer.base import TrainingBackend
from ego_pod.trainer.stub import StubBackend


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------


def _params(epochs: int = 3) -> TrainingParams:
    """构造一组合法的训练参数，仅 ``epochs`` 对 stub 行为有影响。"""

    return TrainingParams(
        method="qlora",
        quantization="4bit",
        lora_rank=16,
        lora_alpha=32,
        learning_rate=2e-4,
        epochs=epochs,
        batch_size=4,
    )


class _CallRecorder:
    """记录 ``progress_callback`` 的每次调用参数。"""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, epoch: int, step: int, loss: float) -> None:
        self.calls.append({"epoch": epoch, "step": step, "loss": loss})


# ---------------------------------------------------------------------------
# R8-AC1：is_available 始终为真，结构上满足 TrainingBackend Protocol
# ---------------------------------------------------------------------------


def test_is_available_always_true():
    assert StubBackend.is_available() is True


def test_satisfies_training_backend_protocol():
    # Protocol 为 runtime_checkable，实例应通过结构化类型检查
    backend = StubBackend()
    assert isinstance(backend, TrainingBackend)


# ---------------------------------------------------------------------------
# R12-AC9：name == "stub"（manifest training_backend stub 标记来源）
# ---------------------------------------------------------------------------


def test_name_is_stub_marker():
    assert StubBackend.name == "stub"
    assert StubBackend().name == "stub"


def test_hardware_is_stub():
    assert StubBackend.hardware == "STUB"
    assert StubBackend().hardware == "STUB"


# ---------------------------------------------------------------------------
# R8-AC5：load_base_model 返回占位对象，不加载真实权重
# ---------------------------------------------------------------------------


def test_load_base_model_returns_placeholder():
    backend = StubBackend()
    model = backend.load_base_model("qwen3.5-0.8B", "4bit")

    assert isinstance(model, dict)
    assert model["model_id"] == "qwen3.5-0.8B"
    assert model["quantization"] == "4bit"

    placeholder = model["placeholder"]
    assert isinstance(placeholder, torch.Tensor)
    assert tuple(placeholder.shape) == (1, 1)
    assert placeholder.dtype == torch.float32


# ---------------------------------------------------------------------------
# R8-AC8：train 至少每 epoch 触发一次进度回调
# ---------------------------------------------------------------------------


def test_train_invokes_progress_callback(tmp_path: Path):
    backend = StubBackend()
    model = backend.load_base_model("qwen3.5-0.8B", "4bit")
    recorder = _CallRecorder()
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("", encoding="utf-8")

    backend.train(model, corpus, _params(epochs=3), recorder)

    # 至少调用一次进度回调
    assert len(recorder.calls) >= 1
    last = recorder.calls[-1]
    # epoch 取 params.epochs，满足"至少每完成一个 epoch 一行"
    assert last["epoch"] == 3
    assert last["step"] == 3
    assert last["loss"] == 0.0


def test_train_callback_epoch_tracks_params(tmp_path: Path):
    backend = StubBackend()
    model = backend.load_base_model("qwen3.5-0.8B", "4bit")
    recorder = _CallRecorder()
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("", encoding="utf-8")

    backend.train(model, corpus, _params(epochs=7), recorder)

    assert recorder.calls[-1]["epoch"] == 7


def test_train_does_not_require_corpus_contents(tmp_path: Path):
    # stub 不读取语料内容；即便语料文件不存在也不应抛错
    backend = StubBackend()
    model = backend.load_base_model("qwen3.5-0.8B", "4bit")
    recorder = _CallRecorder()
    missing_corpus = tmp_path / "absent.jsonl"

    backend.train(model, missing_corpus, _params(), recorder)

    assert len(recorder.calls) == 1


# ---------------------------------------------------------------------------
# R8-AC6：export 产出可被 SafeTensors 读回的 1×1 占位张量
# ---------------------------------------------------------------------------


def test_export_writes_safetensors_readable_file(tmp_path: Path):
    backend = StubBackend()
    model = backend.load_base_model("qwen3.5-0.8B", "4bit")
    out = tmp_path / "lora.safetensors"

    backend.export_lora_safetensors(model, out)

    assert out.exists()
    assert out.stat().st_size > 0

    # 可被 safetensors.numpy.load_file 正常读回
    tensors = load_file(str(out))
    assert "placeholder" in tensors
    placeholder = tensors["placeholder"]
    assert placeholder.shape == (1, 1)
    assert placeholder.dtype == np.float32
    assert np.array_equal(placeholder, np.zeros((1, 1), dtype=np.float32))


def test_export_creates_missing_parent_dirs(tmp_path: Path):
    backend = StubBackend()
    model = backend.load_base_model("qwen3.5-0.8B", "4bit")
    # 父目录尚不存在，export 应自动创建
    out = tmp_path / "nested" / "deep" / "ego-lora.safetensors"

    backend.export_lora_safetensors(model, out)

    assert out.exists()
    tensors = load_file(str(out))
    assert tensors["placeholder"].shape == (1, 1)


def test_export_overwrites_existing_file(tmp_path: Path):
    backend = StubBackend()
    model = backend.load_base_model("qwen3.5-0.8B", "4bit")
    out = tmp_path / "lora.safetensors"
    out.write_bytes(b"stale-bytes")

    backend.export_lora_safetensors(model, out)

    tensors = load_file(str(out))
    assert set(tensors.keys()) == {"placeholder"}
    assert tensors["placeholder"].shape == (1, 1)


# ---------------------------------------------------------------------------
# 端到端 stub 训练序列：load → train → export 串联可跑通
# ---------------------------------------------------------------------------


def test_full_stub_training_sequence(tmp_path: Path):
    backend = StubBackend()
    recorder = _CallRecorder()
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"instruction":"hi","output":"hello"}\n', encoding="utf-8")
    out = tmp_path / "out" / "ego-lora.safetensors"

    model = backend.load_base_model("qwen3.5-0.8B", "4bit")
    backend.train(model, corpus, _params(epochs=2), recorder)
    backend.export_lora_safetensors(model, out)

    assert len(recorder.calls) >= 1
    assert out.exists()
    assert load_file(str(out))["placeholder"].shape == (1, 1)
