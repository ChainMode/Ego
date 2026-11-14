"""`ego_pod.trainer.mlx_backend` 显式失败的属性测试（hypothesis）。

本文件实现 design.md「Correctness Properties」中的 **Property 19**，对应
tasks.md 任务 8.2：

* **Property 19: MLX 被选中即显式失败且不产出文件** —— 对任意对
  :class:`~ego_pod.trainer.mlx_backend.MlxBackend` 的 ``load_base_model`` /
  ``train`` / ``export_lora_safetensors`` 的调用，均抛出携带明确「未实现」原因的
  :class:`~ego_pod.exit_codes.TrainerError`，且不写出任何适配器文件（不静默产出
  无效适配器）。**Validates: Requirements 9.4**

依赖门控：本属性面向 MLX 后端的 **显式失败控制流**，三个方法在任何调用下都直接
抛 ``TrainerError``，**无需安装** ``mlx`` / ``mlx-lm`` 或任何 ML 依赖，可在最小
环境（仅 ``pytest`` + ``hypothesis``）下运行。

关键不变量（R9-AC4）：``export_lora_safetensors`` 即便给定一个可写的目标路径，
也必须在不创建该文件的前提下抛错——杜绝"探测可用 → 调用静默产出无效/空适配器"
的隐患。本文件对每个生成的目标路径断言 ``not target.exists()``。

**Feature: ego-pod-real-trainer**
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ego_pod.config.schema import TrainingParams
from ego_pod.exit_codes import EXIT_RUNTIME_FAILURE, TrainerError
from ego_pod.trainer.mlx_backend import MlxBackend

# ---------------------------------------------------------------------------
# 生成策略
# ---------------------------------------------------------------------------

#: model_id / quantization 取值（含 MVP 固定值与任意其它字符串）。
_model_ids = st.text(min_size=0, max_size=32)
_quantizations = st.sampled_from(["4bit", "8bit", "none", ""]) | st.text(max_size=8)

#: 语料路径（相对/绝对、含多级、含中文等），仅作参数传入，不会被读取。
_corpus_names = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789-_/.", min_size=1, max_size=24
)

#: 安全的相对文件名片段（用于在 tmp_path 下构造导出目标路径）。
_path_segments = st.lists(
    st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz0123456789-_", min_size=1, max_size=12
    ),
    min_size=1,
    max_size=3,
)


def _make_params(draw: st.DrawFn) -> TrainingParams:
    """构造一个合法的 :class:`TrainingParams`（MLX 不读取其内容，仅满足类型）。"""

    return TrainingParams(
        method="qlora",
        quantization="4bit",
        lora_rank=draw(st.integers(min_value=1, max_value=64)),
        lora_alpha=draw(st.integers(min_value=1, max_value=128)),
        learning_rate=draw(
            st.floats(
                min_value=1e-6, max_value=1.0, allow_nan=False, allow_infinity=False
            )
        ),
        epochs=draw(st.integers(min_value=1, max_value=10)),
        batch_size=draw(st.integers(min_value=1, max_value=16)),
    )


def _assert_not_implemented_trainer_error(exc: TrainerError) -> None:
    """断言异常为携带明确「未实现」原因、退出码 1 的 :class:`TrainerError`。"""

    message = str(exc)
    assert message  # 非空，携带明确原因。
    assert "未实现" in message or "未落地" in message
    assert exc.exit_code == EXIT_RUNTIME_FAILURE


# ===========================================================================
# Property 19: MLX 被选中即显式失败且不产出文件（任务 8.2）
# ===========================================================================


@given(model_id=_model_ids, quantization=_quantizations)
@settings(max_examples=200, deadline=None)
def test_mlx_load_base_model_always_raises(model_id: str, quantization: str) -> None:
    """``MlxBackend.load_base_model`` 对任意入参均抛带原因的 ``TrainerError``。

    **Feature: ego-pod-real-trainer, Property 19**

    **Validates: Requirements 9.4**
    """
    backend = MlxBackend()
    with pytest.raises(TrainerError) as excinfo:
        backend.load_base_model(model_id, quantization)
    _assert_not_implemented_trainer_error(excinfo.value)


@given(data=st.data(), corpus_name=_corpus_names)
@settings(max_examples=200, deadline=None)
def test_mlx_train_always_raises(data: st.DataObject, corpus_name: str) -> None:
    """``MlxBackend.train`` 对任意入参均抛带原因的 ``TrainerError``。

    **Feature: ego-pod-real-trainer, Property 19**

    **Validates: Requirements 9.4**
    """
    backend = MlxBackend()
    params = _make_params(data.draw)
    calls: list[tuple] = []

    def _progress(*args: object, **kwargs: object) -> None:  # 不应被调用
        calls.append((args, kwargs))

    with pytest.raises(TrainerError) as excinfo:
        backend.train(object(), Path(corpus_name), params, _progress)

    _assert_not_implemented_trainer_error(excinfo.value)
    # 显式失败：不进入训练循环，故进度回调从未被调用。
    assert calls == []


@given(segments=_path_segments)
@settings(max_examples=200, deadline=None)
def test_mlx_export_raises_and_writes_no_file(
    segments: list[str], tmp_path_factory: pytest.TempPathFactory
) -> None:
    """``MlxBackend.export_lora_safetensors`` 抛 ``TrainerError`` 且不创建任何文件。

    即便给定一个其父目录可写的目标路径，导出也必须在不写出文件的前提下失败
    （R9-AC4：不静默产出无效适配器）。

    **Feature: ego-pod-real-trainer, Property 19**

    **Validates: Requirements 9.4**
    """
    backend = MlxBackend()
    base_dir = tmp_path_factory.mktemp("mlx_export")
    target = base_dir.joinpath(*segments).with_suffix(".safetensors")

    with pytest.raises(TrainerError) as excinfo:
        backend.export_lora_safetensors(object(), target)

    _assert_not_implemented_trainer_error(excinfo.value)
    # 关键不变量：未写出任何适配器文件（不静默产出无效适配器）。
    assert not target.exists()
