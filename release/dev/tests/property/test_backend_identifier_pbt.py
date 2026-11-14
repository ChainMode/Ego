"""`ego_pod.trainer.base.resolve_identifier` 的 Backend_Identifier 映射属性测试。

本文件实现 design.md「Correctness Properties」中的 **Property 13**，对应
tasks.md 任务 1.4：

* **Property 13: 后端到 Backend_Identifier 的映射** —— 对任意 Training_Backend
  实例，:func:`~ego_pod.trainer.base.resolve_identifier` 返回的标识属于固定集合
  ``{unsloth, peft-cuda, mlx, peft-cpu, stub}`` 且等于该后端文档化的取值；特别
  地，Peft_Cuda_Backend → ``peft-cuda`` 与 Peft_Cpu_Backend → ``peft-cpu`` 为两个
  不同取值。**Validates: Requirements 7.1, 7.2, 7.4, 9.5**

测试覆盖两条路径：

1. **真实后端类**（``UnslothBackend`` / ``PeftCudaBackend`` / ``MlxBackend`` /
   ``PeftCpuBackend`` / ``StubBackend``）：实例化每个后端类并断言
   ``resolve_identifier(instance)`` 等于其文档化标识，且五个标识恰好覆盖固定集合
   （从而 ``peft-cuda`` 与 ``peft-cpu`` 必为不同取值，R7-AC4；``unsloth`` 与
   ``mlx`` 各自独立，R9-AC5）。后端类的实例化不触碰任何 ML 依赖（``load_base_model``
   等才会惰性导入），故无需安装 torch/transformers/peft/mlx。
2. **缺失 ``identifier`` 成员的回退推断**：对仅暴露 ``name`` / ``hardware`` 而无
   ``identifier`` 属性的"后端样对象"，断言 ``resolve_identifier`` 按 ``name`` /
   ``hardware`` 正确回退推断（PEFT 两变体靠 hardware 区分 cuda/cpu；其余靠 name
   映射；未知 name 兜底 ``stub``），稳健满足「Protocol 形状不变」的兼容性要求。

**Feature: ego-pod-real-trainer**
"""

from __future__ import annotations

from types import SimpleNamespace

import hypothesis.strategies as st
from hypothesis import given, settings

from ego_pod.trainer.base import resolve_identifier
from ego_pod.trainer.mlx_backend import MlxBackend
from ego_pod.trainer.peft_backend import PeftCpuBackend, PeftCudaBackend
from ego_pod.trainer.stub import StubBackend
from ego_pod.trainer.unsloth_backend import UnslothBackend

# ---------------------------------------------------------------------------
# 固定取值集合与后端 → 文档化标识映射
# ---------------------------------------------------------------------------

#: Backend_Identifier 的固定取值集合（R7-AC1、R7-AC4）。
_VALID_IDENTIFIERS: frozenset[str] = frozenset(
    {"unsloth", "peft-cuda", "mlx", "peft-cpu", "stub"}
)

#: ``(后端类, 文档化标识)`` 对，覆盖全部五个 Training_Backend。
#:
#: ``UnslothBackend`` 在当前实现中尚未声明 ``identifier`` 类属性（待任务 7.1
#: 落地），其文档化标识 ``unsloth`` 由 ``resolve_identifier`` 经 ``name`` 回退
#: 推断得到——这正是 Property 13 要保证的稳定映射。
_BACKEND_EXPECTED: list[tuple[type, str]] = [
    (UnslothBackend, "unsloth"),
    (PeftCudaBackend, "peft-cuda"),
    (MlxBackend, "mlx"),
    (PeftCpuBackend, "peft-cpu"),
    (StubBackend, "stub"),
]


# ===========================================================================
# Property 13: 后端到 Backend_Identifier 的映射（任务 1.4）
# ===========================================================================


@given(pair=st.sampled_from(_BACKEND_EXPECTED))
@settings(max_examples=200, deadline=None)
def test_resolve_identifier_matches_documented_value(pair: tuple[type, str]) -> None:
    """对任意后端实例，``resolve_identifier`` 返回其文档化标识且属于固定集合。

    **Feature: ego-pod-real-trainer, Property 13**

    **Validates: Requirements 7.1, 7.2, 7.4, 9.5**
    """
    backend_cls, expected = pair
    instance = backend_cls()

    ident = resolve_identifier(instance)

    # 返回值等于该后端文档化的取值（R7-AC1、R7-AC2、R9-AC5）。
    assert ident == expected
    # 返回值属于固定集合（R7-AC1、R7-AC4）。
    assert ident in _VALID_IDENTIFIERS


def test_all_backends_cover_fixed_identifier_set() -> None:
    """五个后端的标识恰好覆盖固定集合（彼此两两不同）。

    由此自然得出 ``peft-cuda`` 与 ``peft-cpu`` 为两个不同取值（R7-AC4），且
    ``unsloth`` / ``mlx`` 各自为独立取值（R9-AC5）。

    **Feature: ego-pod-real-trainer, Property 13**

    **Validates: Requirements 7.1, 7.4, 9.5**
    """
    produced = {resolve_identifier(cls()) for cls, _ in _BACKEND_EXPECTED}
    assert produced == _VALID_IDENTIFIERS


def test_peft_cuda_and_peft_cpu_are_distinct() -> None:
    """Peft_Cuda_Backend → ``peft-cuda``、Peft_Cpu_Backend → ``peft-cpu`` 且两者不同。

    **Feature: ego-pod-real-trainer, Property 13**

    **Validates: Requirements 7.1, 7.4**
    """
    cuda_ident = resolve_identifier(PeftCudaBackend())
    cpu_ident = resolve_identifier(PeftCpuBackend())

    assert cuda_ident == "peft-cuda"
    assert cpu_ident == "peft-cpu"
    assert cuda_ident != cpu_ident


# ---------------------------------------------------------------------------
# 缺失 identifier 成员时的回退推断（兼容性兜底）
# ---------------------------------------------------------------------------

#: 回退推断中用于区分 PEFT 两变体的 hardware 取值（仅 ``CUDA`` 判为 cuda 变体）。
_hardware_values = st.sampled_from(
    ["CUDA", "CPU", "Apple Silicon", "STUB", "unknown", ""]
)

#: 不属于已知映射、也非 PEFT 的"未知 name"（兜底应为 ``stub``）。
_KNOWN_NAMES = frozenset({"HuggingFace PEFT", "Unsloth", "Apple MLX", "stub"})
_unknown_names = st.text(max_size=24).filter(lambda s: s not in _KNOWN_NAMES)


@st.composite
def _identifier_less_backends(draw: st.DrawFn) -> tuple[object, str]:
    """生成「无 ``identifier`` 属性、仅含 ``name`` / ``hardware``」的后端样对象。

    :returns: ``(obj, expected)``，``obj`` 为 :class:`types.SimpleNamespace`
        （不具 ``identifier`` 属性，故 ``resolve_identifier`` 走 name/hardware 回退），
        ``expected`` 为按回退规则应得的标识。
    """

    kind = draw(st.sampled_from(["peft", "unsloth", "mlx", "stub", "unknown"]))
    hardware = draw(_hardware_values)

    if kind == "peft":
        name = "HuggingFace PEFT"
        # PEFT 两变体靠 hardware 区分：仅 CUDA → peft-cuda，其余 → peft-cpu。
        expected = "peft-cuda" if hardware == "CUDA" else "peft-cpu"
    elif kind == "unsloth":
        name = "Unsloth"
        expected = "unsloth"
    elif kind == "mlx":
        name = "Apple MLX"
        expected = "mlx"
    elif kind == "stub":
        name = "stub"
        expected = "stub"
    else:  # unknown name → 兜底 stub
        name = draw(_unknown_names)
        expected = "stub"

    return SimpleNamespace(name=name, hardware=hardware), expected


@given(case=_identifier_less_backends())
@settings(max_examples=200, deadline=None)
def test_resolve_identifier_falls_back_by_name_and_hardware(
    case: tuple[object, str],
) -> None:
    """``identifier`` 缺失时，按 ``name`` / ``hardware`` 回退推断且结果在固定集合内。

    覆盖「Protocol 形状不变」的兼容性兜底：即便历史/外部后端实现未声明
    ``identifier``，``resolve_identifier`` 仍得到正确且合法的标识。

    **Feature: ego-pod-real-trainer, Property 13**

    **Validates: Requirements 7.1, 7.2, 7.4, 9.5**
    """
    obj, expected = case
    # 前置确认：该样对象确实没有 identifier 属性，强制走回退路径。
    assert not hasattr(obj, "identifier")

    ident = resolve_identifier(obj)

    assert ident == expected
    assert ident in _VALID_IDENTIFIERS
