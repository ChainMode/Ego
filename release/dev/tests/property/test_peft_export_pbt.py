"""`ego_pod.trainer.peft_backend.export_lora_safetensors` 的属性测试（hypothesis）。

本文件实现 design.md「Correctness Properties」中与 PEFT 适配器导出直接相关的
三条属性，对应 tasks.md 任务 5.5、5.6、5.7：

* **Property 9: 导出文件仅含适配器张量**（任务 5.5）——
  对任意 LoRA 适配器 state dict，:meth:`~ego_pod.trainer.peft_backend._PeftBackendBase.export_lora_safetensors`
  写出的文件的张量键集合等于该适配器键集合，且不包含任何 Base_Model 骨干权重键。
  **Validates: Requirements 5.2**

* **Property 10: 适配器导出—加载往返**（任务 5.6）——
  对任意 LoRA 适配器 state dict，先 ``export_lora_safetensors`` 写出再用
  ``safetensors`` 打开并加载，能成功枚举张量键并还原出与原适配器逐张量相等的
  权重（``load(export(x)) == x``）。**Validates: Requirements 5.1, 5.3, 5.4, 9.3**

* **Property 11: 写入 I/O 错误触发带路径的 TrainerError**（任务 5.7）——
  对任意会在写入时抛出 I/O 错误的目标（此处 patch ``safetensors.torch.save_file``
  抛 ``OSError``），``export_lora_safetensors`` 抛出
  :class:`~ego_pod.exit_codes.TrainerError`，其消息包含目标 ``output_path`` 与
  失败原因。**Validates: Requirements 5.5**

依赖门控与注入手法（见 design.md「Testing Strategy」）：

``export_lora_safetensors`` 在函数体内惰性执行
``from peft import get_peft_model_state_dict`` 与
``from safetensors.torch import save_file``。本仓库默认安装 **包含** ``torch`` 与
``safetensors``、但 **不包含** ``peft``。因此本文件在 ``sys.modules`` 中临时注入一个
**假 ``peft`` 模块**，其 ``get_peft_model_state_dict(model)`` 直接返回我们生成的
适配器 state dict（由真实 ``torch`` 张量组成）；``save_file`` / ``load_file`` 则使用
**真实** 的 ``safetensors.torch``，使导出—加载往返（Property 10）是货真价实的。

模型对象按 :func:`~ego_pod.trainer.peft_backend._unpack_model` 期望的
``(model_obj, tokenizer)`` 复合形态传入；``model_obj`` 为占位 dummy，因为假
``get_peft_model_state_dict`` 忽略其入参、恒返回闭包捕获的适配器 state dict
（故"导出文件仅含适配器键"按构造成立、不含任何骨干键）。

**Feature: ego-pod-real-trainer**
"""

from __future__ import annotations

import contextlib
import sys
import types
from pathlib import Path
from typing import Iterator
from unittest import mock

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from safetensors.torch import load_file

from ego_pod.exit_codes import EXIT_RUNTIME_FAILURE, TrainerError
from ego_pod.trainer.peft_backend import PeftCpuBackend

# ---------------------------------------------------------------------------
# 注入假 peft 模块的工具
# ---------------------------------------------------------------------------


def _make_fake_peft(state_dict: dict[str, torch.Tensor]) -> types.ModuleType:
    """构造一个假 ``peft`` 模块，其 ``get_peft_model_state_dict`` 恒返回 ``state_dict``。

    ``export_lora_safetensors`` 内以 ``from peft import get_peft_model_state_dict``
    惰性导入并调用 ``get_peft_model_state_dict(model_obj)``；本假实现忽略其入参
    （``model_obj`` 为 dummy），恒返回闭包捕获的、由真实 ``torch`` 张量组成的适配器
    state dict——从而"导出文件键集合 == 适配器键集合、且不含 base 骨干键"按构造成立。
    """

    module = types.ModuleType("peft")

    def _get_peft_model_state_dict(_model: object, *_args: object, **_kwargs: object):
        return state_dict

    module.get_peft_model_state_dict = _get_peft_model_state_dict  # type: ignore[attr-defined]
    return module


@contextlib.contextmanager
def _inject_peft(state_dict: dict[str, torch.Tensor]) -> Iterator[None]:
    """在 ``sys.modules`` 中临时注入返回 ``state_dict`` 的假 ``peft`` 模块。

    退出时精确还原 ``sys.modules['peft']`` 的原状态（原本缺失则删除，原本存在则
    恢复），避免污染其它测试对"peft 是否可用"的判定。
    """

    fake = _make_fake_peft(state_dict)
    had_previous = "peft" in sys.modules
    previous = sys.modules.get("peft")
    sys.modules["peft"] = fake
    try:
        yield
    finally:
        if had_previous:
            sys.modules["peft"] = previous  # type: ignore[assignment]
        else:
            sys.modules.pop("peft", None)


# ---------------------------------------------------------------------------
# 生成策略：LoRA 适配器 state dict（真实 torch 张量）
# ---------------------------------------------------------------------------

#: LoRA 适配器常见的投影模块名（仅用于构造形如真实 PEFT 的张量键）。
_PROJECTIONS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

#: float32 可精确表示的有限浮点值（``width=32`` 保证写入/读回逐位相等）。
_finite_f32 = st.floats(allow_nan=False, allow_infinity=False, width=32)


#: LoRA 风格张量键，形如
#: ``base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight``，
#: 必含 ``lora_A`` / ``lora_B`` 子串（区别于 base 骨干权重键）。
_lora_keys = st.builds(
    lambda layer, proj, ab: (
        f"base_model.model.model.layers.{layer}.self_attn.{proj}.lora_{ab}.weight"
    ),
    st.integers(min_value=0, max_value=31),
    st.sampled_from(_PROJECTIONS),
    st.sampled_from(["A", "B"]),
)


@st.composite
def _tensor_spec(draw: st.DrawFn) -> tuple[tuple[int, int], list[float]]:
    """生成一个 2D 张量规格 ``((rows, cols), flat_values)``（LoRA 张量为二维）。"""

    rows = draw(st.integers(min_value=1, max_value=6))
    cols = draw(st.integers(min_value=1, max_value=6))
    values = draw(
        st.lists(_finite_f32, min_size=rows * cols, max_size=rows * cols)
    )
    return (rows, cols), values


#: 适配器规格：键（LoRA 风格）→ 张量规格；至少 1 个张量，至多 8 个（控制文件大小）。
_adapter_specs = st.dictionaries(
    keys=_lora_keys,
    values=_tensor_spec(),
    min_size=1,
    max_size=8,
)


def _build_state_dict(
    specs: dict[str, tuple[tuple[int, int], list[float]]],
) -> dict[str, torch.Tensor]:
    """把张量规格物化为真实 ``torch.float32`` 张量构成的适配器 state dict。"""

    state: dict[str, torch.Tensor] = {}
    for key, ((rows, cols), values) in specs.items():
        tensor = torch.tensor(values, dtype=torch.float32).reshape(rows, cols)
        state[key] = tensor
    return state


def _assert_no_backbone_keys(keys: set[str]) -> None:
    """断言键集合中每个键都是 LoRA 适配器键（含 ``lora_A`` / ``lora_B``）、无骨干键。"""

    for key in keys:
        assert "lora_A" in key or "lora_B" in key, (
            f"导出文件包含疑似 Base_Model 骨干权重键：{key!r}"
        )


# ===========================================================================
# Property 9: 导出文件仅含适配器张量（任务 5.5）
# ===========================================================================


@given(specs=_adapter_specs)
@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_export_file_contains_only_adapter_tensors(
    specs: dict[str, tuple[tuple[int, int], list[float]]],
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """导出文件的张量键集合等于适配器键集合，且不含任何 base 骨干键。

    对任意 LoRA 适配器 state dict，``export_lora_safetensors`` 写出的
    ``.safetensors`` 文件经 ``safetensors`` 重新打开后：

    * 其键集合**恰好等于**输入适配器的键集合（不增不减）；
    * 每个键都是 LoRA 适配器键（含 ``lora_A`` / ``lora_B``），不含 Base_Model
      骨干权重键（R5-AC2）。

    **Feature: ego-pod-real-trainer, Property 9**

    **Validates: Requirements 5.2**
    """
    state_dict = _build_state_dict(specs)
    output_path = tmp_path_factory.mktemp("export_only_adapter") / "ego-lora.safetensors"
    backend = PeftCpuBackend()

    with _inject_peft(state_dict):
        backend.export_lora_safetensors((object(), None), output_path)

    assert output_path.is_file()
    loaded = load_file(str(output_path))
    file_keys = set(loaded.keys())

    # 文件键集合恰为适配器键集合（R5-AC2：仅含适配器张量）。
    assert file_keys == set(state_dict.keys())
    # 不含任何 Base_Model 骨干权重键。
    _assert_no_backbone_keys(file_keys)


# ===========================================================================
# Property 10: 适配器导出—加载往返（任务 5.6）
# ===========================================================================


@given(specs=_adapter_specs)
@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_export_then_load_roundtrip_preserves_tensors(
    specs: dict[str, tuple[tuple[int, int], list[float]]],
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """``load(export(x)) == x``：导出后用 safetensors 加载，逐张量与原适配器相等。

    对任意 LoRA 适配器 state dict，``export_lora_safetensors`` 写出可被
    ``safetensors.torch.load_file`` 正常打开并枚举键的合法文件（R5-AC1、R5-AC3），
    且：

    * 加载得到的键集合与原适配器键集合相等；
    * 每个键对应的张量与原张量逐位相等（``torch.equal``，含 dtype 与 shape 一致）；
      该往返产出的文件即可被同后端推理路径加载应用（R5-AC4、R9-AC3）。

    **Feature: ego-pod-real-trainer, Property 10**

    **Validates: Requirements 5.1, 5.3, 5.4, 9.3**
    """
    state_dict = _build_state_dict(specs)
    output_path = tmp_path_factory.mktemp("export_roundtrip") / "ego-lora.safetensors"
    backend = PeftCpuBackend()

    with _inject_peft(state_dict):
        backend.export_lora_safetensors((object(), None), output_path)

    # 文件合法、可被 safetensors 打开并枚举键（R5-AC3）。
    assert output_path.is_file()
    loaded = load_file(str(output_path))

    # 键集合往返一致（R5-AC1）。
    assert set(loaded.keys()) == set(state_dict.keys())

    # 每个张量逐位相等（load(export(x)) == x，R5-AC4、R9-AC3）。
    for key, original in state_dict.items():
        restored = loaded[key]
        assert restored.shape == original.shape
        assert restored.dtype == original.dtype
        assert torch.equal(restored, original), (
            f"键 {key!r} 的张量往返后不相等"
        )


# ===========================================================================
# Property 11: 写入 I/O 错误触发带路径的 TrainerError（任务 5.7）
# ===========================================================================

#: 注入的 I/O 错误类型（均为 ``OSError`` 子类，模拟磁盘满 / 权限拒绝 / 目标是目录等）。
_io_errors = st.sampled_from(
    [
        OSError,
        PermissionError,
        IsADirectoryError,
        FileNotFoundError,
    ]
)
#: I/O 错误的原因文本（非空，使"消息包含原因"的断言非平凡）。
_io_reasons = st.text(min_size=1, max_size=60)


@given(specs=_adapter_specs, error_cls=_io_errors, reason=_io_reasons)
@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_export_io_error_raises_trainer_error_with_path(
    specs: dict[str, tuple[tuple[int, int], list[float]]],
    error_cls: type[OSError],
    reason: str,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """写入时遭遇 I/O 错误 → ``TrainerError``，消息含目标 ``output_path`` 与原因，且不留文件。

    把 ``safetensors.torch.save_file`` patch 为抛出生成的 ``OSError``（及其子类），
    断言 ``export_lora_safetensors``：

    * 抛出 :class:`TrainerError`（``exit_code == 1``，不引入新退出码）；
    * 消息同时包含目标 ``output_path`` 字符串与失败原因（R5-AC5）；
    * 未写出任何 ``.safetensors`` 文件（写入失败不残留半成品）。

    **Feature: ego-pod-real-trainer, Property 11**

    **Validates: Requirements 5.5**
    """
    state_dict = _build_state_dict(specs)
    output_path = tmp_path_factory.mktemp("export_io_error") / "ego-lora.safetensors"
    backend = PeftCpuBackend()
    injected = error_cls(reason)

    with _inject_peft(state_dict):
        with mock.patch(
            "safetensors.torch.save_file", side_effect=injected
        ):
            with pytest.raises(TrainerError) as excinfo:
                backend.export_lora_safetensors((object(), None), output_path)

    message = str(excinfo.value)
    # 消息包含目标 output_path（R5-AC5）。
    assert str(output_path) in message
    # 消息包含失败原因。
    assert reason in message
    # 复用既有退出码语义，映射到 1（不引入新退出码取值）。
    assert excinfo.value.exit_code == EXIT_RUNTIME_FAILURE
    # 写入失败不应残留任何文件。
    assert not output_path.exists()
