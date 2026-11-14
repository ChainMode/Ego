"""`ego_pod.trainer.peft_backend` 基座加载失败包装的属性测试（hypothesis）。

本文件实现 design.md「Correctness Properties」中的 **Property 12**，对应
tasks.md 任务 5.2：

* **Property 12: 基座加载失败包装为带 model_id 的 TrainerError** —— 对任意底层
  加载器抛出的异常类型（权重缺失、依赖缺失、显存/内存不足等，经 mock 注入），
  :meth:`~ego_pod.trainer.peft_backend._PeftBackendBase.load_base_model` 抛出
  :class:`~ego_pod.exit_codes.TrainerError`，其消息同时包含 ``model_id`` 与原始
  失败原因。**Validates: Requirements 1.5**

依赖门控：本属性面向 ``load_base_model`` 的 **错误归一化控制流**，**无需安装**
``transformers`` / ``peft`` / ``bitsandbytes``。

注入手法：``_PeftBackendBase.load_base_model`` 在 CPU 路径（``PeftCpuBackend``，
``hardware == "CPU"``，故 ``use_cuda`` / ``use_4bit`` 恒为 ``False``）内以
``from transformers import AutoModelForCausalLM, AutoTokenizer`` 惰性导入并调用
``AutoTokenizer.from_pretrained(model_id)``。本文件在 ``sys.modules`` 中临时注入一个
**假 transformers 模块**，其 ``from_pretrained`` 抛出 hypothesis 生成的各类异常，
从而无需真实下载/真实依赖即可覆盖"权重缺失 / 依赖缺失 / OOM / 其它"等失败类型。
``torch`` 为默认依赖（已安装），CPU 路径在 tokenizer 处即失败，不触达任何网络。

设计对齐（见 ``trainer/peft_backend.py``）：

    except EgoError:
        raise
    except Exception as exc:
        raise TrainerError(f"Base_Model 加载失败（model_id={model_id!r}）：{exc}") from exc

因此包装后的消息必同时含 ``model_id``（经 ``!r`` 渲染，原串为其子串）与
``str(exc)``（原始失败原因），且 ``TrainerError.exit_code == 1``。

**Feature: ego-pod-real-trainer**
"""

from __future__ import annotations

import contextlib
import sys
import types
from typing import Iterator

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ego_pod.exit_codes import EXIT_RUNTIME_FAILURE, TrainerError
from ego_pod.trainer.peft_backend import PeftCpuBackend

# ---------------------------------------------------------------------------
# 注入假 transformers 模块的工具
# ---------------------------------------------------------------------------


def _make_fake_transformers(exc: BaseException) -> types.ModuleType:
    """构造一个假 ``transformers`` 模块，其加载入口在调用时抛出 ``exc``。

    ``AutoTokenizer`` 与 ``AutoModelForCausalLM`` 均暴露 ``from_pretrained``
    静态方法并抛出同一注入异常实例；``BitsAndBytesConfig`` 一并提供以兼容
    CUDA 4-bit 分支（CPU 路径不会触达）。
    """

    module = types.ModuleType("transformers")

    class _RaisingAuto:
        @staticmethod
        def from_pretrained(*_args: object, **_kwargs: object) -> object:
            raise exc

    module.AutoTokenizer = _RaisingAuto  # type: ignore[attr-defined]
    module.AutoModelForCausalLM = _RaisingAuto  # type: ignore[attr-defined]
    module.BitsAndBytesConfig = object  # type: ignore[attr-defined]
    return module


@contextlib.contextmanager
def _inject_transformers(exc: BaseException) -> Iterator[None]:
    """在 ``sys.modules`` 中临时注入抛 ``exc`` 的假 ``transformers`` 模块。

    退出时精确还原 ``sys.modules['transformers']`` 的原状态（原本缺失则删除，
    原本存在则恢复），避免污染其它测试对"transformers 是否可用"的判定。
    """

    fake = _make_fake_transformers(exc)
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


# ---------------------------------------------------------------------------
# 生成策略
# ---------------------------------------------------------------------------

#: model_id 取自不含引号/反斜杠的安全字符集，保证原串是其 ``repr`` 的子串
#: （从而"消息包含 model_id"的断言可靠成立）。
_MODEL_ID_ALPHABET = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_/"
)
_model_ids = st.text(alphabet=_MODEL_ID_ALPHABET, min_size=1, max_size=32)

#: 注入异常的原因文本（非空，使"消息包含原因"的断言非平凡）。
_reasons = st.text(min_size=1, max_size=80)


@st.composite
def _injected_exceptions(draw: st.DrawFn) -> BaseException:
    """生成覆盖各失败类型的底层异常实例（均为非 ``EgoError`` 的标准异常）。

    覆盖设计文档列举的失败来源：

    * 权重缺失：:class:`FileNotFoundError` / :class:`OSError`
    * 依赖缺失：:class:`ImportError` / :class:`ModuleNotFoundError`
    * 显存/内存不足（OOM）：消息含 ``out of memory`` 的 :class:`RuntimeError` /
      :class:`MemoryError`
    * 其它通用失败：:class:`ValueError` / :class:`RuntimeError`
    """

    reason = draw(_reasons)
    kind = draw(
        st.sampled_from(
            [
                "weights_filenotfound",
                "weights_oserror",
                "deps_importerror",
                "deps_modulenotfound",
                "oom_runtime",
                "oom_memoryerror",
                "value",
                "runtime",
            ]
        )
    )
    if kind == "weights_filenotfound":
        return FileNotFoundError(reason)
    if kind == "weights_oserror":
        return OSError(reason)
    if kind == "deps_importerror":
        return ImportError(reason)
    if kind == "deps_modulenotfound":
        return ModuleNotFoundError(reason)
    if kind == "oom_runtime":
        return RuntimeError(f"CUDA out of memory: {reason}")
    if kind == "oom_memoryerror":
        return MemoryError(reason)
    if kind == "value":
        return ValueError(reason)
    return RuntimeError(reason)


# ===========================================================================
# Property 12: 基座加载失败包装为带 model_id 的 TrainerError（任务 5.2）
# ===========================================================================


@given(model_id=_model_ids, exc=_injected_exceptions())
@settings(max_examples=200, deadline=None)
def test_load_base_model_wraps_failures_with_model_id_and_reason(
    model_id: str, exc: BaseException
) -> None:
    """任意底层加载异常 → ``TrainerError``，消息含 ``model_id`` 与原始原因。

    对任意注入的底层异常类型（权重缺失 / 依赖缺失 / OOM / 通用），
    ``PeftCpuBackend.load_base_model`` 把其归一化为 :class:`TrainerError`，且：

    * 消息包含 ``model_id``（设计中以 ``model_id={model_id!r}`` 渲染，原串为子串）；
    * 消息包含原始失败原因 ``str(exc)``；
    * ``exit_code == 1``（``EXIT_RUNTIME_FAILURE``，不引入新退出码）。

    **Feature: ego-pod-real-trainer, Property 12**

    **Validates: Requirements 1.5**
    """
    backend = PeftCpuBackend()
    reason = str(exc)

    with _inject_transformers(exc):
        with pytest.raises(TrainerError) as excinfo:
            backend.load_base_model(model_id, "4bit")

    message = str(excinfo.value)
    # 同时包含 model_id 与原始失败原因（R1-AC5）。
    assert model_id in message
    assert reason in message
    # 复用既有退出码语义，映射到 1（不引入新退出码取值）。
    assert excinfo.value.exit_code == EXIT_RUNTIME_FAILURE


def test_missing_transformers_dependency_wraps_as_trainer_error() -> None:
    """依赖缺失（``import transformers`` 抛 ImportError）同样包装为带 model_id 的 TrainerError。

    显式覆盖"加载路径的依赖导入本身失败"这一具体场景（R1-AC5 的"依赖缺失"），
    以固定的 :class:`ModuleNotFoundError` 注入，断言归一化包装行为稳定。

    **Feature: ego-pod-real-trainer, Property 12**

    **Validates: Requirements 1.5**
    """
    backend = PeftCpuBackend()
    model_id = "qwen3.5-0.8B"
    exc = ModuleNotFoundError("No module named 'transformers'")

    with _inject_transformers(exc):
        with pytest.raises(TrainerError) as excinfo:
            backend.load_base_model(model_id, "4bit")

    message = str(excinfo.value)
    assert model_id in message
    assert "No module named 'transformers'" in message
    assert excinfo.value.exit_code == EXIT_RUNTIME_FAILURE
