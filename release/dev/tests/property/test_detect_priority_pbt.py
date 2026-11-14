"""`ego_pod.trainer.detect.detect_backend` 的探测优先级属性测试。

本文件实现 design.md「Correctness Properties」中的 **Property 18**，对应
tasks.md 任务 1.5：

* **Property 18: 探测优先级与最终回退** —— 对任意各 Real_Backend 的可用性组合
  （Stub 恒可用），:func:`~ego_pod.trainer.detect.detect_backend` 返回固定优先级
  顺序（Unsloth → Peft_Cuda → Mlx → Peft_Cpu → Stub）中第一个可用的后端；当四个
  Real_Backend 全部不可用时返回 ``StubBackend``。
  **Validates: Requirements 8.1, 8.2, 8.3**

测试用 hypothesis 生成四个 Real_Backend 的可用性组合（每个后端独立的可用/不可用
布尔位，共 ``2**4 = 16`` 种组合），并在每个示例内用 :func:`unittest.mock.patch.object`
临时把各后端类的 ``is_available`` classmethod 覆盖为固定返回值（等价于模拟"依赖
导入成功/失败 + 目标硬件可用/不可用"，R8-AC2 的跳过语义）。``StubBackend`` 保持其
真实实现（``is_available`` 恒为 ``True``，R8-AC1 末位回退）。每个示例通过
:class:`contextlib.ExitStack` 在用例结束后干净地还原所有 patch，避免 hypothesis
跨示例的状态泄漏。

测试不依赖任何真实 GPU / ML 依赖或网络。

**Feature: ego-pod-real-trainer**
"""

from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import patch

import hypothesis.strategies as st
from hypothesis import given, settings

from ego_pod.trainer import detect
from ego_pod.trainer.mlx_backend import MlxBackend
from ego_pod.trainer.peft_backend import PeftCpuBackend, PeftCudaBackend
from ego_pod.trainer.stub import StubBackend
from ego_pod.trainer.unsloth_backend import UnslothBackend

# ---------------------------------------------------------------------------
# 固定优先级顺序（Unsloth → PeftCuda → Mlx → PeftCpu → Stub，R8-AC1）
# ---------------------------------------------------------------------------

#: 四个真实后端，按探测优先级从高到低排列（不含始终可用的 Stub 末位回退）。
_REAL_BACKENDS: tuple[type, ...] = (
    UnslothBackend,
    PeftCudaBackend,
    MlxBackend,
    PeftCpuBackend,
)


def _set_availability(stack: ExitStack, available: dict[type, bool]) -> None:
    """在 ``stack`` 内把每个真实后端类的 ``is_available`` 覆盖为固定布尔值。

    ``StubBackend`` **不**被 patch，保持其真实实现（恒为 ``True``），以验证它作为
    探测链的最终回退（R8-AC1、R8-AC3）。各 patch 随 ``stack`` 退出而还原，使
    hypothesis 的每个示例互不干扰。

    :param stack: 管理 patch 生命周期的 :class:`contextlib.ExitStack`。
    :param available: ``后端类 → 是否可用`` 的映射（覆盖全部四个真实后端）。
    """

    for backend_cls in _REAL_BACKENDS:
        is_avail = available[backend_cls]
        stack.enter_context(
            patch.object(
                backend_cls,
                "is_available",
                classmethod(lambda cls, _v=is_avail: _v),
            )
        )


def _expected_backend_cls(available: dict[type, bool]) -> type:
    """按固定优先级返回第一个可用的后端类；全不可用时返回 ``StubBackend``。"""

    for backend_cls in _REAL_BACKENDS:
        if available[backend_cls]:
            return backend_cls
    return StubBackend


# ===========================================================================
# Property 18: 探测优先级与最终回退（任务 1.5）
# ===========================================================================


@given(
    # 为四个真实后端各生成一个独立的可用/不可用布尔位（共 16 种组合）。
    flags=st.fixed_dictionaries(
        {backend_cls: st.booleans() for backend_cls in _REAL_BACKENDS}
    )
)
@settings(max_examples=200, deadline=None)
def test_detect_returns_first_available_in_priority(flags: dict[type, bool]) -> None:
    """``detect_backend`` 返回固定优先级中第一个可用的后端（Stub 兜底）。

    对任意四个 Real_Backend 的可用性组合：

    * 选中者恰为优先级顺序（Unsloth → PeftCuda → Mlx → PeftCpu）中第一个可用者
      （R8-AC1、R8-AC2：不可用者被跳过，继续探测下一候选）；
    * 当四个 Real_Backend 全部不可用时，回退到始终可用的 ``StubBackend``
      （R8-AC1 末位回退、R8-AC3）。

    **Feature: ego-pod-real-trainer, Property 18**

    **Validates: Requirements 8.1, 8.2, 8.3**
    """
    expected_cls = _expected_backend_cls(flags)

    with ExitStack() as stack:
        _set_availability(stack, flags)
        selected = detect.detect_backend()

    assert type(selected) is expected_cls


@given(
    flags=st.fixed_dictionaries(
        {backend_cls: st.booleans() for backend_cls in _REAL_BACKENDS}
    )
)
@settings(max_examples=200, deadline=None)
def test_unavailable_higher_priority_is_skipped(flags: dict[type, bool]) -> None:
    """优先级更高但不可用的后端必被跳过，不会被选中（R8-AC2）。

    选中后端在优先级序列中的位置之前的所有真实后端均不可用——即"第一个可用者"
    的定义被严格满足。

    **Feature: ego-pod-real-trainer, Property 18**

    **Validates: Requirements 8.1, 8.2**
    """
    with ExitStack() as stack:
        _set_availability(stack, flags)
        selected = detect.detect_backend()

    selected_cls = type(selected)
    if selected_cls is StubBackend:
        # 选中 Stub ⇒ 四个真实后端必全部不可用（R8-AC3）。
        assert all(not flags[cls] for cls in _REAL_BACKENDS)
    else:
        # 选中某真实后端 ⇒ 其优先级之前的真实后端全部不可用，且它自身可用。
        index = _REAL_BACKENDS.index(selected_cls)
        assert flags[selected_cls] is True
        assert all(not flags[cls] for cls in _REAL_BACKENDS[:index])


def test_all_real_backends_unavailable_falls_back_to_stub() -> None:
    """四个 Real_Backend 全部不可用时回退到 Stub_Backend（R8-AC3 显式样例）。

    **Feature: ego-pod-real-trainer, Property 18**

    **Validates: Requirements 8.1, 8.3**
    """
    none_available = {backend_cls: False for backend_cls in _REAL_BACKENDS}

    with ExitStack() as stack:
        _set_availability(stack, none_available)
        selected = detect.detect_backend()

    assert type(selected) is StubBackend
    assert StubBackend.is_available() is True
