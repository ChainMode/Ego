"""Runtime 据 Backend_Identifier 分流到匹配推理路径的属性测试。

本文件实现 design.md「Correctness Properties」中的 **Property 15**，对应
tasks.md 任务 10.3：

* **Property 15: Runtime 据 Backend_Identifier 分流到匹配推理路径** —— 对
  ``manifest.json`` 的 ``training_backend`` 取值：为 ``stub`` 时
  :func:`~ego_pod.runtime.inference.load_inference_session` 返回
  :class:`~ego_pod.runtime.inference.StubInferenceSession`（行为不变）；为
  ``{unsloth, peft-cuda, mlx, peft-cpu}`` 之一时，
  :func:`~ego_pod.runtime.inference._resolve_inference_backend` 返回与该标识
  匹配的后端类并据此走真实后端推理路径（构造
  :class:`~ego_pod.runtime.inference.BackendInferenceSession`）。
  **Validates: Requirements 6.1, 6.6, 7.5**

测试在**无 ML 依赖**的环境下运行：

1. **路由正确性（往返）**：对四个真实标识，``_resolve_inference_backend(id)``
   返回匹配后端类的实例，且该实例 ``.identifier`` 等于输入标识（R6-AC1、
   R7-AC5）。后端类实例化不触碰任何 ML 依赖（``load_base_model`` 等才惰性
   导入），故无需安装 torch/transformers/peft/mlx。
2. **未知标识失败**：对不属于固定集合（且非 ``stub``）的任意字符串，
   ``_resolve_inference_backend`` 抛出 :class:`~ego_pod.exit_codes.RuntimeError_`
   且退出码为 1，避免对未知标识静默走错路径。
3. **Stub 分流端到端**：构造一个 ``training_backend == "stub"`` 的 Ego_Package
   临时目录树（用 :func:`safetensors.numpy.save_file` 写入合法的占位适配器，
   safetensors+numpy 可用），断言 ``load_inference_session`` 返回
   :class:`StubInferenceSession` 并打印 Stub 模式启动提示，且不加载任何 ML
   依赖（R6-AC6）。
4. **真实分流到后端路径**：对真实标识的 manifest，注入一个假后端（经
   ``backend=`` 参数），其 ``load_base_model`` 返回 ``(fake_model, fake_tokenizer)``；
   断言 ``load_inference_session`` 走的是真实后端路径（调用了注入后端、不打印
   Stub 提示、不返回 Stub 会话），从而验证 Runtime 据 Backend_Identifier 分流。
   适配器的真实应用（``_apply_lora_adapter`` 惰性导入 ``peft``）在无 ML 依赖时
   被包装为 :class:`RuntimeError_`，但这发生在路由抵达真实路径**之后**，不影响
   分流正确性的断言（保持依赖无关）。

**Feature: ego-pod-real-trainer**
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import hypothesis.strategies as st
import numpy as np
from hypothesis import given, settings
from safetensors.numpy import save_file

from ego_pod.exit_codes import EXIT_RUNTIME_FAILURE, RuntimeError_
from ego_pod.runtime.inference import (
    ADAPTER_RELPATH,
    BackendInferenceSession,
    CONFIG_FILENAME,
    DEFAULT_EDITION,
    MANIFEST_FILENAME,
    STUB_MODE_STARTUP_NOTICE,
    STUB_TRAINING_BACKEND,
    StubInferenceSession,
    _resolve_inference_backend,
    load_inference_session,
)
from ego_pod.trainer.mlx_backend import MlxBackend
from ego_pod.trainer.peft_backend import PeftCpuBackend, PeftCudaBackend
from ego_pod.trainer.unsloth_backend import UnslothBackend

# ---------------------------------------------------------------------------
# 标识 → 匹配后端类 映射（四个真实推理后端，R6-AC1、R7-AC5）
# ---------------------------------------------------------------------------

#: ``Backend_Identifier`` → 与之匹配的真实后端类。
_IDENTIFIER_TO_BACKEND: dict[str, type] = {
    "peft-cuda": PeftCudaBackend,
    "peft-cpu": PeftCpuBackend,
    "unsloth": UnslothBackend,
    "mlx": MlxBackend,
}

#: 四个真实推理后端标识（Stub 走独立的 Stub 推理路径，不经本解析函数）。
_REAL_IDENTIFIERS: tuple[str, ...] = tuple(_IDENTIFIER_TO_BACKEND)

#: 不应被 ``_resolve_inference_backend`` 接受的标识：四个真实标识 + ``stub``。
#: ``stub`` 在语义上由 ``load_inference_session`` 的 Stub 分流处理（R6-AC6），
#: 不应作为真实后端标识传入解析函数。
_EXCLUDED_IDENTIFIERS: frozenset[str] = frozenset(
    _REAL_IDENTIFIERS + (STUB_TRAINING_BACKEND,)
)


# ---------------------------------------------------------------------------
# Ego_Package 临时夹具（manifest.json + normal/config.json + 合法占位适配器）
# ---------------------------------------------------------------------------


def _build_package(root: Path, *, training_backend: str) -> None:
    """在 ``root``（Ego_Package 根，即 ``ego-output/``）下构造完整三件产物。

    写入 ``manifest.json``（携带给定 ``training_backend``）、
    ``<normal>/config.json``（合法 LoRA 参数）与
    ``<normal>/adapters/ego-lora.safetensors``（用 numpy 后端写入的合法占位张量，
    可被 ``safetensors`` 正常打开，从而通过 ``load_inference_session`` 的完整性
    校验）。不依赖任何 ML 框架（仅 numpy + safetensors）。
    """
    edition_dir = root / DEFAULT_EDITION
    adapter_path = edition_dir / ADAPTER_RELPATH
    adapter_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "version": "1.0.0",
        "created_at": "2025-11-25T08:00:00Z",
        "base_model": "qwen3.5-0.8B",
        "editions": [DEFAULT_EDITION],
        "training_backend": training_backend,
    }
    (root / MANIFEST_FILENAME).write_text(json.dumps(manifest), encoding="utf-8")

    config = {
        "base_model": "qwen3.5-0.8B",
        "quantization": "4bit",
        "lora_rank": 16,
        "lora_alpha": 32,
    }
    (edition_dir / CONFIG_FILENAME).write_text(json.dumps(config), encoding="utf-8")

    save_file({"placeholder": np.zeros((1, 1), dtype=np.float32)}, str(adapter_path))


class _RecordingBackend:
    """记录 ``load_base_model`` 调用的假后端，用于验证路由抵达真实推理路径。

    其 ``load_base_model`` 返回 ``(fake_model, fake_tokenizer)`` 复合对象（均为
    无 ML 依赖的 :class:`types.SimpleNamespace`），不加载任何真实权重。
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def load_base_model(self, model_id: str, quantization: str):
        self.calls.append((model_id, quantization))
        return SimpleNamespace(name="fake-model"), SimpleNamespace(name="fake-tokenizer")


# ===========================================================================
# Property 15a：真实标识 → 匹配后端类（往返路由正确性，R6-AC1、R7-AC5）
# ===========================================================================


@given(identifier=st.sampled_from(_REAL_IDENTIFIERS))
@settings(max_examples=200, deadline=None)
def test_resolve_routes_identifier_to_matching_backend(identifier: str) -> None:
    """``_resolve_inference_backend(id)`` 返回匹配后端实例且其 ``.identifier`` 等于 id。

    覆盖「Runtime 据 Backend_Identifier 选择与训练时匹配的推理路径」的核心
    往返：标识 → 后端类 → 后端实例的 ``.identifier`` 回到原标识（R6-AC1、
    R7-AC5）。后端实例化不触碰任何 ML 依赖。

    **Feature: ego-pod-real-trainer, Property 15**

    **Validates: Requirements 6.1, 7.5**
    """
    expected_cls = _IDENTIFIER_TO_BACKEND[identifier]

    backend = _resolve_inference_backend(identifier)

    # 返回与该标识匹配的后端类实例（R6-AC1、R7-AC5）。
    assert isinstance(backend, expected_cls)
    # 往返一致：该后端实例文档化的 identifier 等于输入标识（R7-AC4/AC5）。
    assert backend.identifier == identifier


# ===========================================================================
# Property 15b：未知标识 → RuntimeError_（退出码 1），不静默走错路径
# ===========================================================================


@given(
    bad_identifier=st.text().filter(lambda s: s not in _EXCLUDED_IDENTIFIERS),
)
@settings(max_examples=200, deadline=None)
def test_resolve_unknown_identifier_raises_runtime_error(bad_identifier: str) -> None:
    """非固定集合（且非 ``stub``）的标识 → ``RuntimeError_``（退出码 1）。

    避免对未知 ``training_backend`` 标识静默选择错误的推理路径。

    **Feature: ego-pod-real-trainer, Property 15**

    **Validates: Requirements 6.1, 7.5**
    """
    try:
        _resolve_inference_backend(bad_identifier)
    except RuntimeError_ as exc:
        assert exc.exit_code == EXIT_RUNTIME_FAILURE
    else:  # pragma: no cover - 未抛错即属性失败
        raise AssertionError(
            f"未知标识 {bad_identifier!r} 应抛 RuntimeError_，但未抛出"
        )


# ===========================================================================
# Property 15c：manifest training_backend == "stub" → Stub 推理路径（端到端）
# ===========================================================================


def test_load_inference_session_routes_stub_to_stub_session() -> None:
    """``training_backend == "stub"`` → 返回 StubInferenceSession 并打印 Stub 提示。

    端到端验证 Stub 分流路径行为不变（R6-AC6）：在临时 Ego_Package 树上调用
    ``load_inference_session``，断言返回 :class:`StubInferenceSession` 且向注入的
    stdout 打印 Stub 模式启动提示，全程不加载任何 ML 依赖。

    **Feature: ego-pod-real-trainer, Property 15**

    **Validates: Requirements 6.6, 7.5**
    """
    import io

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "ego-output"
        _build_package(root, training_backend=STUB_TRAINING_BACKEND)

        stdout = io.StringIO()
        session = load_inference_session(output_root=root, stdout=stdout)

        # Stub 分流：返回 Stub 推理会话（行为不变，R6-AC6）。
        assert isinstance(session, StubInferenceSession)
        # 启动时打印 Stub 模式提示（R6-AC6）。
        assert STUB_MODE_STARTUP_NOTICE in stdout.getvalue()
        session.close()


# ===========================================================================
# Property 15d：manifest 为真实标识 → 路由抵达真实后端推理路径（非 Stub）
# ===========================================================================


@given(identifier=st.sampled_from(_REAL_IDENTIFIERS))
@settings(max_examples=100, deadline=None)
def test_load_inference_session_routes_real_identifier_to_backend_path(
    identifier: str,
) -> None:
    """真实标识 → 走真实后端路径（调用注入后端、不打印 Stub 提示、非 Stub 会话）。

    注入一个记录调用的假后端（``backend=`` 参数），其 ``load_base_model`` 返回
    无 ML 依赖的假模型/假分词器。断言 ``load_inference_session`` 把
    ``manifest.training_backend`` 作为 Backend_Identifier 路由到真实后端推理路径
    （R6-AC1、R7-AC5）：注入后端的 ``load_base_model`` 被调用、未打印 Stub 提示、
    结果不是 Stub 会话。

    适配器的真实应用经 ``peft`` 惰性导入；无 ML 依赖时被包装为
    :class:`RuntimeError_`，但这发生在路由抵达真实路径**之后**（``load_base_model``
    已被调用），故不影响分流正确性断言，测试保持依赖无关。

    **Feature: ego-pod-real-trainer, Property 15**

    **Validates: Requirements 6.1, 6.6, 7.5**
    """
    import io

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "ego-output"
        _build_package(root, training_backend=identifier)

        fake_backend = _RecordingBackend()
        stdout = io.StringIO()

        session = None
        try:
            session = load_inference_session(
                output_root=root, backend=fake_backend, stdout=stdout
            )
        except RuntimeError_:
            # 无 ML 依赖时，适配器应用阶段（peft 惰性导入）失败被包装为
            # RuntimeError_——这发生在路由抵达真实路径之后，符合预期。
            session = None

        # 路由抵达真实后端路径：注入后端的 load_base_model 被调用（非 Stub 分流）。
        assert fake_backend.calls == [("qwen3.5-0.8B", "4bit")]
        # 真实路径不打印 Stub 模式提示（区别于 Stub 分流，R6-AC6）。
        assert STUB_MODE_STARTUP_NOTICE not in stdout.getvalue()
        # 若成功构造会话，则必为真实后端会话而非 Stub 会话。
        if session is not None:
            assert isinstance(session, BackendInferenceSession)
            assert not isinstance(session, StubInferenceSession)
            session.close()
