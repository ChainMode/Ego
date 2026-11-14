"""``ego_pod.runtime.inference.BackendInferenceSession`` 加载失败归一化的属性测试。

本文件实现 design.md「Correctness Properties」中的 **Property 16**，对应
tasks.md 任务 10.4：

* **Property 16: 推理加载失败包装为 RuntimeError_** —— 对任意在推理加载阶段
  （``backend.load_base_model``）注入的失败（依赖缺失 ``ImportError``、
  ``NotImplementedError``、硬件不可用 ``RuntimeError``、权重/IO 错误 ``OSError``
  等），:class:`~ego_pod.runtime.inference.BackendInferenceSession` 的构造抛出
  :class:`~ego_pod.exit_codes.RuntimeError_`（退出码 1），其消息包含失败原因。
  **Validates: Requirements 6.4**

依赖门控：本属性面向构造阶段的 **错误归一化控制流**，**无需安装** ``torch`` /
``transformers`` / ``peft`` / ``bitsandbytes`` / ``mlx`` 等 Training_Dependencies。

注入手法：``BackendInferenceSession.__init__`` 在显式传入 ``backend`` 时优先使用
该后端，并调用 ``backend.load_base_model(base_model, quantization)`` 加载 base。
本文件构造一个 **假后端**，其 ``load_base_model`` 在被调用时立即抛出 hypothesis
生成的各类异常——由于异常在 ``_apply_lora_adapter`` 之前抛出，``peft`` /
``safetensors`` 永不被导入，``adapter_path`` 也不会被触达（传入一个虚构 Path 即可）。

设计对齐（见 ``runtime/inference.py`` 的 ``BackendInferenceSession.__init__``）::

    except EgoError:
        raise
    except (ImportError, NotImplementedError, RuntimeError, OSError) as exc:
        raise RuntimeError_(
            "加载真实推理后端失败（缺失依赖或硬件不可用）：" f"{exc}"
        ) from exc

因此：

* 被包装类别（``ImportError`` / ``NotImplementedError`` / ``RuntimeError`` /
  ``OSError`` 及其子类）→ 归一化为 :class:`RuntimeError_`，``exit_code == 1``，
  消息含 ``str(exc)``（原始失败原因），且 ``__cause__`` 为注入的原异常实例；
* :class:`~ego_pod.exit_codes.EgoError` 子类（如 :class:`TrainerError`）→ 原样
  透传，不被二次包装为 :class:`RuntimeError_`。

**Feature: ego-pod-real-trainer, Property 16**

**Validates: Requirements 6.4**
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ego_pod.exit_codes import (
    EXIT_RUNTIME_FAILURE,
    CollectorError,
    CompilerError,
    ConfigError,
    EgoError,
    RuntimeError_,
    TrainerError,
    VaultError,
)
from ego_pod.runtime.inference import BackendInferenceSession


# ---------------------------------------------------------------------------
# 假后端：load_base_model 立即抛出注入异常
# ---------------------------------------------------------------------------


class _RaisingBackend:
    """假推理后端：``load_base_model`` 在被调用时立即抛出注入的异常实例。

    异常在 ``_apply_lora_adapter`` 之前抛出，故 ``peft`` / ``safetensors`` 永不
    被导入，``adapter_path`` 也不会被读取——无需任何 ML 依赖即可覆盖加载失败路径。
    """

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    def load_base_model(self, model_id: str, quantization: str) -> Any:
        raise self._exc


def _make_session(exc: BaseException) -> BackendInferenceSession:
    """以抛 ``exc`` 的假后端构造 :class:`BackendInferenceSession`。

    ``adapter_path`` 传入一个虚构路径（加载在其被触达前即失败，故不会被读取）。
    """
    return BackendInferenceSession(
        base_model="qwen3.5-0.8B",
        quantization="4bit",
        lora_rank=1,
        lora_alpha=1,
        adapter_path=Path("does-not-exist") / "adapters" / "ego-lora.safetensors",
        backend=_RaisingBackend(exc),
    )


# ---------------------------------------------------------------------------
# 生成策略
# ---------------------------------------------------------------------------

#: 失败原因文本（非空，排除代理项 "Cs" 类以避免无法编码的字符）。
_reasons = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",)),
    min_size=1,
    max_size=80,
)


@st.composite
def _wrapped_exceptions(draw: st.DrawFn) -> BaseException:
    """生成被 ``BackendInferenceSession`` 归一化的异常实例。

    覆盖 ``__init__`` 的 ``except (ImportError, NotImplementedError,
    RuntimeError, OSError)`` 子句所列各类别（含子类），原因文本由 hypothesis 生成：

    * 依赖缺失：:class:`ImportError` / :class:`ModuleNotFoundError`
    * 未实现：:class:`NotImplementedError`
    * 硬件不可用 / OOM：:class:`RuntimeError`（含 ``CUDA out of memory`` 变体）
    * 权重/IO 错误：:class:`OSError` / :class:`FileNotFoundError`
    """
    reason = draw(_reasons)
    kind = draw(
        st.sampled_from(
            [
                "import",
                "modulenotfound",
                "notimplemented",
                "runtime",
                "runtime_oom",
                "oserror",
                "filenotfound",
            ]
        )
    )
    if kind == "import":
        return ImportError(reason)
    if kind == "modulenotfound":
        return ModuleNotFoundError(reason)
    if kind == "notimplemented":
        return NotImplementedError(reason)
    if kind == "runtime":
        return RuntimeError(reason)
    if kind == "runtime_oom":
        return RuntimeError(f"CUDA out of memory: {reason}")
    if kind == "oserror":
        return OSError(reason)
    return FileNotFoundError(reason)


#: 固定的"代表性"被包装异常实例集合，覆盖任务点名的四个类别 + 任意消息。
_REPRESENTATIVE_WRAPPED = [
    ImportError("No module named 'peft'"),
    NotImplementedError("真实后端推理尚未实现"),
    RuntimeError("CUDA out of memory"),
    OSError("safetensors header is corrupt"),
]

#: ``EgoError`` 家族类（应被原样透传，不二次包装）。
_EGO_ERROR_CLASSES = [
    TrainerError,
    RuntimeError_,
    ConfigError,
    CollectorError,
    CompilerError,
    VaultError,
]


# ===========================================================================
# Property 16: 推理加载失败包装为 RuntimeError_（任务 10.4）
# ===========================================================================


@given(exc=_wrapped_exceptions())
@settings(max_examples=200, deadline=None)
def test_load_failure_wrapped_as_runtime_error(exc: BaseException) -> None:
    """任意被包装类别的加载异常 → ``RuntimeError_``（退出码 1，消息含原因）。

    对任意注入的 ``ImportError`` / ``NotImplementedError`` / ``RuntimeError`` /
    ``OSError``（及其子类），``BackendInferenceSession`` 的构造把其归一化为
    :class:`RuntimeError_`，且：

    * ``exit_code == 1``（``EXIT_RUNTIME_FAILURE``，不引入新退出码）；
    * 消息包含原始失败原因 ``str(exc)``（实现以 ``f"...{exc}"`` 拼接）；
    * ``__cause__`` 为注入的原异常实例（``raise ... from exc`` 链保留）。

    **Feature: ego-pod-real-trainer, Property 16**

    **Validates: Requirements 6.4**
    """
    reason = str(exc)

    with pytest.raises(RuntimeError_) as excinfo:
        _make_session(exc)

    wrapped = excinfo.value
    # 复用既有退出码语义，映射到 1（不引入新退出码取值）。
    assert wrapped.exit_code == EXIT_RUNTIME_FAILURE
    # 消息携带原始失败原因，供 CLI 顶层打印到标准错误（R6-AC4）。
    assert reason in str(wrapped)
    # from exc 链保留原始异常，便于诊断。
    assert wrapped.__cause__ is exc


@pytest.mark.parametrize(
    "exc", _REPRESENTATIVE_WRAPPED, ids=lambda e: type(e).__name__
)
def test_representative_load_failures_wrapped(exc: BaseException) -> None:
    """点名覆盖四个被包装类别的代表实例，均归一化为 ``RuntimeError_``。

    显式覆盖任务列举的具体场景（``peft`` 缺失 / 未实现 / 显存不足 / 权重损坏），
    断言归一化包装行为稳定。

    **Feature: ego-pod-real-trainer, Property 16**

    **Validates: Requirements 6.4**
    """
    with pytest.raises(RuntimeError_) as excinfo:
        _make_session(exc)

    assert excinfo.value.exit_code == EXIT_RUNTIME_FAILURE
    assert str(exc) in str(excinfo.value)
    assert excinfo.value.__cause__ is exc


# ===========================================================================
# 透传不变式：EgoError 子类原样上抛，不被二次包装为 RuntimeError_
# ===========================================================================


@given(
    error_cls=st.sampled_from(_EGO_ERROR_CLASSES),
    reason=_reasons,
)
@settings(max_examples=100, deadline=None)
def test_ego_error_subclasses_passed_through(
    error_cls: type[EgoError], reason: str
) -> None:
    """加载阶段抛出的任意 ``EgoError`` 子类被原样透传（不包装为 ``RuntimeError_``）。

    若 ``load_base_model`` 抛出 :class:`EgoError` 家族异常（如后端的
    :class:`TrainerError`），``BackendInferenceSession`` 应原样重抛 **同一实例**，
    保持其退出码语义不被 ``except (ImportError, ...)`` 子句二次包装。

    特别地：``RuntimeError_`` 本身是 ``EgoError`` 子类而非内置 ``RuntimeError``，
    故注入的 ``RuntimeError_`` 也应被原样透传（验证 ``except EgoError: raise``
    优先于 ``except RuntimeError`` 子句）。

    **Feature: ego-pod-real-trainer, Property 16**

    **Validates: Requirements 6.4**
    """
    injected = error_cls(reason)

    with pytest.raises(EgoError) as excinfo:
        _make_session(injected)

    # 原样透传：异常类型与实例均保持不变，未被包装为 RuntimeError_。
    assert excinfo.value is injected
    assert type(excinfo.value) is error_cls
