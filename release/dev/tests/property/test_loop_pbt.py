"""`ego_pod.trainer.loop` QLoRA 训练循环的属性测试（hypothesis）。

本文件实现 design.md「Correctness Properties」中与 ``trainer/loop.py`` 直接相关
的两条属性，对应 tasks.md 任务 3.2 与 3.3：

* **Property 5: 训练循环逐 epoch 进度回调**（任务 3.2）——
  对任意合法 ``params``（``epochs >= 1``，小规模）与假模型/假数据集，
  :func:`~ego_pod.trainer.loop.train_qlora` 恰好调用 ``progress_callback``
  ``epochs`` 次，传入的 epoch 编号依次为 ``1..epochs``，且累计步数参数随调用
  单调不减。**Validates: Requirements 4.1, 4.2**

* **Property 8: 数值发散与 OOM 触发 TrainerError**（任务 3.3）——
  对任意注入到训练步中、于任意位置出现 ``NaN`` / ``Inf`` 的损失序列，
  ``train_qlora`` 抛出 :class:`~ego_pod.exit_codes.TrainerError`；同理对任意
  注入的 OOM 错误亦抛出携带原因的 ``TrainerError``（不静默继续）。
  **Validates: Requirements 4.7**

测试全程使用 ``tests/_doubles.py`` 提供的 **无 ML 依赖** 假模型
(:class:`FakeModel` / :func:`make_finite_loss_model` /
:func:`make_diverging_loss_model`)，因此可在最小环境（仅 ``pytest`` +
``hypothesis``）下运行，无需安装 ``torch`` / ``peft``：``train_qlora`` 对假模型
不会构造真实优化器（无真实可训练张量），循环仍完整执行前向、损失有限性检查、
OOM 归一化与逐 epoch 回调（见 ``trainer/loop.py`` 模块文档的「鸭子类型契约」）。

数据集采用「小整数列表」：``FakeModel.forward`` 忽略输入内容，且
``loop._collate`` 对不具 ``input_ids`` 的元素原样以 ``list`` 传入模型，故整数
列表是驱动循环控制流的最轻量数据集（每个元素经分批后构成一个训练步的批数据）。

**Feature: ego-pod-real-trainer**
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ego_pod.config.schema import TrainingParams
from ego_pod.exit_codes import TrainerError
from ego_pod.trainer.loop import TrainingMetrics, train_qlora

# ``tests/_doubles.py`` 经 conftest 把 tests/ 目录加入 sys.path，可作顶层导入。
from _doubles import FakeModel, make_diverging_loss_model, make_finite_loss_model

# ---------------------------------------------------------------------------
# 公共辅助
# ---------------------------------------------------------------------------


def _make_params(
    *,
    epochs: int,
    batch_size: int,
    lora_rank: int,
    lora_alpha: int,
    learning_rate: float,
) -> TrainingParams:
    """构造一个固定 ``method=qlora`` / ``quantization=4bit`` 的 :class:`TrainingParams`。

    只填充 ``train_qlora`` 真正使用的三个超参（``epochs`` / ``batch_size`` /
    ``learning_rate``）以及为满足 dataclass 完备性所需的 ``lora_rank`` /
    ``lora_alpha``。
    """

    return TrainingParams(
        method="qlora",
        quantization="4bit",
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
    )


class _ProgressRecorder:
    """记录每次 ``progress_callback`` 调用的 ``(epoch, step, loss)`` 三元组。

    ``train_qlora`` 以关键字参数 ``epoch=`` / ``step=`` / ``loss=`` 调用回调
    （见 ``trainer/loop.py``），故本类以同名关键字参数接收，贴合
    :data:`ego_pod.trainer.base.ProgressCallback` 的语义约定。
    """

    def __init__(self) -> None:
        self.calls: list[tuple[int, int, float]] = []

    def __call__(self, *, epoch: int, step: int, loss: float) -> None:
        self.calls.append((epoch, step, loss))

    @property
    def epochs(self) -> list[int]:
        return [epoch for epoch, _step, _loss in self.calls]

    @property
    def steps(self) -> list[int]:
        return [step for _epoch, step, _loss in self.calls]


# 合法超参取值（小规模，使属性测试在最少 100 次迭代下仍快速）。
_epochs = st.integers(min_value=1, max_value=6)
_batch_sizes = st.integers(min_value=1, max_value=6)
_dataset_sizes = st.integers(min_value=1, max_value=20)
_lora_rank = st.integers(min_value=1, max_value=64)
_lora_alpha = st.integers(min_value=1, max_value=128)
_learning_rate = st.floats(
    min_value=1e-6, max_value=1.0, allow_nan=False, allow_infinity=False
)
_finite_loss = st.floats(
    min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False
)
_seeds = st.integers(min_value=0, max_value=2**16)

#: 注入数值发散的取值：NaN / +Inf / -Inf 三种非有限值。
_diverge_values = st.sampled_from([float("nan"), float("inf"), float("-inf")])


# ===========================================================================
# Property 5: 训练循环逐 epoch 进度回调（任务 3.2）
# ===========================================================================


@st.composite
def _progress_scenarios(draw: st.DrawFn):
    """生成 (params, dataset, loss_value, seed)：合法小规模超参 + 有限损失模型。"""

    params = _make_params(
        epochs=draw(_epochs),
        batch_size=draw(_batch_sizes),
        lora_rank=draw(_lora_rank),
        lora_alpha=draw(_lora_alpha),
        learning_rate=draw(_learning_rate),
    )
    dataset = list(range(draw(_dataset_sizes)))
    loss_value = draw(_finite_loss)
    seed = draw(_seeds)
    return params, dataset, loss_value, seed


@given(scenario=_progress_scenarios())
@settings(max_examples=200, deadline=None)
def test_progress_callback_once_per_epoch(scenario) -> None:
    """``train_qlora`` 恰好每 epoch 回调一次，epoch 编号为 ``1..epochs``，步数单调不减。

    对任意合法 ``params``（``epochs >= 1``）与有限损失的假模型/数据集：

    * 回调次数恰为 ``params.epochs``；
    * 传入的 epoch 编号依次为 ``1, 2, ..., epochs``（连续递增、无重复无遗漏）；
    * 传入的累计步数参数随调用单调不减（``step`` 非递减）；
    * 返回的 :class:`TrainingMetrics` 的 ``completed_epochs`` 等于 ``params.epochs``，
      且 ``total_steps`` 与最后一次回调的步数一致。

    **Feature: ego-pod-real-trainer, Property 5**

    **Validates: Requirements 4.1, 4.2**
    """
    params, dataset, loss_value, seed = scenario
    model = make_finite_loss_model(loss_value=loss_value)
    recorder = _ProgressRecorder()

    metrics = train_qlora(
        model,
        dataset,
        params,
        seed=seed,
        progress_callback=recorder,
    )

    # 回调次数恰为 epochs（R4-AC2：至少每 epoch 一次；本循环实现为恰好一次）。
    assert len(recorder.calls) == params.epochs

    # epoch 编号依次为 1..epochs。
    assert recorder.epochs == list(range(1, params.epochs + 1))

    # 累计步数随调用单调不减。
    steps = recorder.steps
    assert all(earlier <= later for earlier, later in zip(steps, steps[1:]))

    # 返回度量与回调观测一致。
    assert isinstance(metrics, TrainingMetrics)
    assert metrics.completed_epochs == params.epochs
    assert metrics.total_steps == steps[-1]


@given(scenario=_progress_scenarios())
@settings(max_examples=120, deadline=None)
def test_progress_step_matches_batch_count(scenario) -> None:
    """累计步数严格等于已处理批次数，且每 epoch 步数为 ``ceil(len/batch_size)``。

    进一步钉死 ``step`` 的语义（R4-AC2「累计训练步数」）：每个 epoch 切分出
    ``ceil(dataset_size / batch_size)`` 个批次（每批一步），故第 ``e`` 次回调的
    ``step`` 恰为 ``e * batches_per_epoch``，且每次前向调用对应一步。

    **Feature: ego-pod-real-trainer, Property 5**

    **Validates: Requirements 4.1, 4.2**
    """
    params, dataset, loss_value, seed = scenario
    model = make_finite_loss_model(loss_value=loss_value)
    recorder = _ProgressRecorder()

    train_qlora(model, dataset, params, seed=seed, progress_callback=recorder)

    batches_per_epoch = math.ceil(len(dataset) / params.batch_size)
    expected_steps = [
        epoch * batches_per_epoch for epoch in range(1, params.epochs + 1)
    ]
    assert recorder.steps == expected_steps
    # 每个批次恰对应一次前向调用：总步数 == 模型前向调用次数。
    assert recorder.steps[-1] == model.call_count


# ===========================================================================
# Property 8: 数值发散与 OOM 触发 TrainerError（任务 3.3）
# ===========================================================================


@st.composite
def _failure_scenarios(draw: st.DrawFn):
    """生成 (params, dataset, fail_at)：fail_at 落在 [1, 总前向调用数] 内。

    总前向调用数 = ``epochs * ceil(dataset_size / batch_size)``；把失败位置约束
    在该范围内，确保注入的发散/OOM 在训练正常结束前一定被触达（覆盖「任意位置」）。
    """

    epochs = draw(_epochs)
    batch_size = draw(_batch_sizes)
    dataset_size = draw(_dataset_sizes)
    params = _make_params(
        epochs=epochs,
        batch_size=batch_size,
        lora_rank=draw(_lora_rank),
        lora_alpha=draw(_lora_alpha),
        learning_rate=draw(_learning_rate),
    )
    dataset = list(range(dataset_size))
    total_calls = epochs * math.ceil(dataset_size / batch_size)
    fail_at = draw(st.integers(min_value=1, max_value=total_calls))
    seed = draw(_seeds)
    return params, dataset, fail_at, seed


@given(scenario=_failure_scenarios(), diverge=_diverge_values)
@settings(max_examples=200, deadline=None)
def test_nonfinite_loss_raises_trainer_error(scenario, diverge: float) -> None:
    """训练步出现 ``NaN`` / ``Inf`` 损失时 ``train_qlora`` 抛 :class:`TrainerError`。

    在第 ``fail_at`` 次前向（1-based，落在总步数内）注入非有限损失，断言训练
    中止并抛 ``TrainerError``，且错误消息携带定位信息（epoch / step）与损失值，
    而非静默继续（R4-AC7）。

    **Feature: ego-pod-real-trainer, Property 8**

    **Validates: Requirements 4.7**
    """
    params, dataset, fail_at, seed = scenario
    # finite_steps = fail_at - 1 → 第 fail_at 次前向返回发散损失。
    model = make_diverging_loss_model(finite_steps=fail_at - 1, diverge=diverge)
    recorder = _ProgressRecorder()

    with pytest.raises(TrainerError) as excinfo:
        train_qlora(model, dataset, params, seed=seed, progress_callback=recorder)

    message = str(excinfo.value)
    assert message  # 非空，携带原因。
    # 错误消息包含步数定位（R4-AC7：消息含具体失败原因）。
    assert f"step={fail_at}" in message


@given(scenario=_failure_scenarios())
@settings(max_examples=200, deadline=None)
def test_oom_raises_trainer_error(scenario) -> None:
    """训练步发生 OOM 时 ``train_qlora`` 抛携带原因的 :class:`TrainerError`。

    在第 ``fail_at`` 次前向注入「消息含 out of memory 的 RuntimeError」模拟显存/
    内存不足，断言被归一化包装为 ``TrainerError``（不向上透出裸 ``RuntimeError``、
    不静默继续），且消息携带定位信息与原始 OOM 原因（R4-AC7）。

    **Feature: ego-pod-real-trainer, Property 8**

    **Validates: Requirements 4.7**
    """
    params, dataset, fail_at, seed = scenario
    model = FakeModel(oom_at_call=fail_at)
    recorder = _ProgressRecorder()

    with pytest.raises(TrainerError) as excinfo:
        train_qlora(model, dataset, params, seed=seed, progress_callback=recorder)

    message = str(excinfo.value)
    assert message  # 非空，携带原因。
    assert f"step={fail_at}" in message
    # 保留原始 OOM 原因（FakeModel 注入的消息含 "out of memory"）。
    assert "out of memory" in message.lower()
