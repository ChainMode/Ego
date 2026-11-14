"""`runtime/inference.py` 的会话级单元测试（任务 10.5）。

聚焦属性测试之外的两组固定样例，覆盖 Requirement 6 中两条会话生命周期相关的
验收标准，且**完全无需安装任何 ML 依赖**（``torch`` / ``transformers`` /
``peft`` / ``unsloth`` / ``mlx`` 均不要求存在）：

* **``close()`` 资源释放（R6-AC5）**：用一个注入的假后端
  （``backend=`` 构造参数）驱动 :class:`~ego_pod.runtime.inference.BackendInferenceSession`
  的构造——其 ``load_base_model`` 返回 ``(fake_model, fake_tokenizer)``。为避免
  ``__init__`` 触达需要 ``peft`` / ``safetensors.torch`` 的
  :func:`~ego_pod.runtime.inference._apply_lora_adapter`，将该函数 monkeypatch
  为"原样返回 base 模型"。构造成功后断言 ``model`` / ``tokenizer`` 已设置；
  调用 ``close()`` 后断言 ``model`` / ``tokenizer`` / ``adapter_tensors`` 三者
  均被置空，且 ``close()`` 不抛错——后者顺带验证
  :func:`~ego_pod.runtime.inference._release_torch_memory` 在 ``torch`` 缺失时
  仍安全（其 ``gc.collect()`` 与 ``torch.cuda.empty_cache()`` 均裹在 try/except）。

* **Stub 路径回归（R6-AC6）**：在临时目录下构造一个 ``training_backend == "stub"``
  的 Ego_Package（``manifest.json`` + ``normal/config.json`` + 合法的
  ``normal/adapters/ego-lora.safetensors``，后者用 :func:`safetensors.numpy.save_file`
  写入一个极小数组——``numpy`` + ``safetensors`` 为已装依赖）。调用
  :func:`~ego_pod.runtime.inference.load_inference_session` 后断言：返回的是
  :class:`~ego_pod.runtime.inference.StubInferenceSession`；``generate`` 对任意
  输入返回固定占位回复 :data:`~ego_pod.runtime.inference.STUB_PLACEHOLDER_REPLY`；
  启动时把 :data:`~ego_pod.runtime.inference.STUB_MODE_STARTUP_NOTICE` 打印到
  传入的 stdout；``close()` 为空操作（不抛错）。这与 MVP 既有 Stub 行为一致。

**Feature: ego-pod-real-trainer**
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from safetensors.numpy import save_file

import ego_pod.runtime.inference as inference
from ego_pod.runtime.inference import (
    ADAPTER_RELPATH,
    CONFIG_FILENAME,
    DEFAULT_EDITION,
    MANIFEST_FILENAME,
    OUTPUT_DIRNAME,
    STUB_MODE_STARTUP_NOTICE,
    STUB_PLACEHOLDER_REPLY,
    BackendInferenceSession,
    StubInferenceSession,
    load_inference_session,
)


# ===========================================================================
# 测试替身：注入的假后端
# ===========================================================================


class _FakeBackend:
    """注入到 ``BackendInferenceSession(backend=...)`` 的假后端。

    仅实现构造所需的 ``load_base_model``，返回 ``(model, tokenizer)`` 二元组；
    记录调用入参以便断言。不依赖任何真实 ML 库。
    """

    def __init__(self, model: Any, tokenizer: Any) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self.calls: list[tuple[str, str]] = []

    def load_base_model(self, model_id: str, quantization: str) -> tuple[Any, Any]:
        self.calls.append((model_id, quantization))
        return self._model, self._tokenizer


class _Sentinel:
    """轻量占位对象，作为假 base 模型 / tokenizer 的可辨识引用。"""

    def __init__(self, label: str) -> None:
        self.label = label

    def __repr__(self) -> str:  # pragma: no cover - 仅调试用
        return f"_Sentinel({self.label!r})"


# ===========================================================================
# R6-AC5：close() 释放 base + adapter 资源（model/tokenizer/adapter_tensors 置空）
# ===========================================================================


def _make_backend_session(monkeypatch, tmp_path: Path) -> tuple[
    BackendInferenceSession, _Sentinel, _Sentinel
]:
    """构造一个由假后端驱动的 :class:`BackendInferenceSession`。

    把 ``_apply_lora_adapter`` monkeypatch 为"原样返回 base 模型"，从而绕开对
    ``peft`` / ``safetensors.torch`` 的导入；返回会话与其期望的
    ``(model, tokenizer)`` 哨兵以便断言。
    """
    fake_model = _Sentinel("base-model")
    fake_tokenizer = _Sentinel("tokenizer")
    backend = _FakeBackend(fake_model, fake_tokenizer)

    # 绕开真实 LoRA 应用（需 peft/safetensors.torch）：原样返回传入的 base。
    monkeypatch.setattr(
        inference,
        "_apply_lora_adapter",
        lambda model, adapter_path, lora_rank, lora_alpha: model,
    )

    session = BackendInferenceSession(
        base_model="qwen3.5-0.8B",
        quantization="4bit",
        lora_rank=16,
        lora_alpha=32,
        adapter_path=tmp_path / "adapters" / "ego-lora.safetensors",
        backend=backend,
    )
    return session, fake_model, fake_tokenizer


def test_backend_session_construction_sets_model_and_tokenizer(monkeypatch, tmp_path):
    # 构造成功后，base 模型与 tokenizer 引用应被设置（来自假后端的返回）。
    session, fake_model, fake_tokenizer = _make_backend_session(monkeypatch, tmp_path)

    assert session.model is fake_model
    assert session.tokenizer is fake_tokenizer
    # 假后端的 load_base_model 恰好以 config 提供的入参被调用一次。
    assert session.backend.calls == [("qwen3.5-0.8B", "4bit")]


def test_backend_session_close_releases_references(monkeypatch, tmp_path):
    # R6-AC5：close() 后 model / tokenizer / adapter_tensors 三者均被置空。
    session, _fake_model, _fake_tokenizer = _make_backend_session(monkeypatch, tmp_path)

    # close 前持有引用。
    assert session.model is not None
    assert session.tokenizer is not None

    session.close()

    assert session.model is None
    assert session.tokenizer is None
    assert session.adapter_tensors is None


def test_backend_session_close_does_not_raise_without_torch(monkeypatch, tmp_path):
    # close() 调用 _release_torch_memory（gc.collect + 可用时 empty_cache），
    # 全部裹在 try/except 中——torch 缺失时亦不应抛错；重复 close 仍幂等安全。
    session, _m, _t = _make_backend_session(monkeypatch, tmp_path)

    session.close()
    # 二次 close 不应抛错（引用已为 None，资源释放为尽力而为）。
    session.close()

    assert session.model is None


def test_release_torch_memory_is_safe_to_call_directly():
    # 直接调用资源释放辅助：无论 torch 是否存在都不应抛错（需求 6.5 / 8.4）。
    inference._release_torch_memory()


# ===========================================================================
# R6-AC6：stub 分流路径——固定占位回复、启动提示、close() 空操作均不变
# ===========================================================================


def _write_stub_package(cwd: Path, *, edition: str = DEFAULT_EDITION) -> Path:
    """在 ``cwd`` 下构造一个 ``training_backend == "stub"`` 的合法 Ego_Package。

    写入根部 ``manifest.json``（标记 stub）、``<edition>/config.json`` 与合法的
    ``<edition>/adapters/ego-lora.safetensors``（用 safetensors.numpy 写入 1×1
    占位张量，使 :func:`load_inference_session` 的适配器完整性校验通过）。
    """
    root = cwd / OUTPUT_DIRNAME
    edition_dir = root / edition
    edition_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "version": "1.0.0",
        "created_at": "2025-11-25T08:00:00Z",
        "base_model": "qwen3.5-0.8B",
        "editions": [edition],
        "training_backend": "stub",
    }
    (root / MANIFEST_FILENAME).write_text(json.dumps(manifest), encoding="utf-8")

    config = {
        "base_model": "qwen3.5-0.8B",
        "quantization": "4bit",
        "lora_rank": 16,
        "lora_alpha": 32,
    }
    (edition_dir / CONFIG_FILENAME).write_text(json.dumps(config), encoding="utf-8")

    adapter_path = edition_dir / ADAPTER_RELPATH
    adapter_path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"placeholder": np.zeros((1, 1), dtype=np.float32)}, str(adapter_path))

    return root


def test_stub_manifest_routes_to_stub_session(tmp_path):
    # R6-AC6：training_backend == "stub" → 返回 StubInferenceSession。
    _write_stub_package(tmp_path)
    stdout = io.StringIO()

    session = load_inference_session(cwd=tmp_path, stdout=stdout)

    assert isinstance(session, StubInferenceSession)


def test_stub_session_prints_startup_notice(tmp_path):
    # R6-AC6：进入 Stub 模式时把启动提示打印到提供的 stdout。
    _write_stub_package(tmp_path)
    stdout = io.StringIO()

    load_inference_session(cwd=tmp_path, stdout=stdout)

    assert STUB_MODE_STARTUP_NOTICE in stdout.getvalue()


def test_stub_session_generate_returns_fixed_reply(tmp_path):
    # R6-AC6：对任意非空输入恒返回同一条固定占位回复。
    _write_stub_package(tmp_path)
    session = load_inference_session(cwd=tmp_path, stdout=io.StringIO())

    assert session.generate("anything") == STUB_PLACEHOLDER_REPLY
    # 不同输入仍得到同一固定回复（与输入内容无关）。
    assert session.generate("另一个完全不同的输入") == STUB_PLACEHOLDER_REPLY


def test_stub_session_close_is_noop(tmp_path):
    # R6-AC6：Stub 会话不持有真实资源，close() 为空操作且不抛错。
    _write_stub_package(tmp_path)
    session = load_inference_session(cwd=tmp_path, stdout=io.StringIO())

    assert session.close() is None
    # close 后仍可重复调用，保持空操作语义。
    assert session.close() is None
