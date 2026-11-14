"""`ego_pod.trainer.pipeline.run_training` 离线缺权重短路的属性测试（hypothesis）。

本文件实现 design.md「Correctness Properties」中的 **Property 20**，对应
tasks.md 任务 9.2：

* **Property 20: 离线模式缺失权重 → 退出码 1** —— 对任意「离线模式启用 且 本地
  权重不可用」的组合，训练流水线抛出
  :class:`~ego_pod.privacy.offline.OfflineMissingWeightsError`（固定 stderr 文案、
  退出码 1）；当权重可用时不触发该短路而继续加载。
  **Validates: Requirements 2.3**

依赖门控：本属性面向 ``run_training`` 的 **离线门控控制流**，**无需安装** 任何
ML 依赖，可在最小环境（仅 ``pytest`` + ``hypothesis``）下运行。

测试设计：

* 必须使用 ``hardware != "STUB"`` 的"真实"后端，因为
  :func:`~ego_pod.trainer.pipeline._base_model_available` 对 Stub_Backend
  （``hardware == "STUB"``）恒视权重为"可用"而绕过 ``base_model_available_check``。
  故采用 :class:`~ego_pod.trainer.peft_backend.PeftCpuBackend`（``hardware ==
  "CPU"``，``identifier == "peft-cpu"``）。
* 为使「权重可用」分支也完全确定、且**绝不触达网络/真实权重**，本文件用一个
  :class:`PeftCpuBackend` 的轻量子类 :class:`_TrackingPeftCpuBackend`：它保留真实
  后端的 ``name`` / ``hardware`` / ``identifier`` / ``is_available``，仅把
  ``load_base_model`` 覆盖为"记录调用并抛出一个**非** ``OfflineMissingWeightsError``
  的哨兵错误"。由此可精确断言：
    - 「离线 + 不可用」→ 在加载**之前**短路（``load`` 从未被调用）并抛
      ``OfflineMissingWeightsError``；
    - 「离线 + 可用」或「在线」→ **不**短路，流程进入 ``load``（``load`` 被调用
      恰一次），且抛出的不是 ``OfflineMissingWeightsError``。
* 另含一个使用 **未改写的** :class:`PeftCpuBackend` 的直接用例，钉死真实后端在
  「离线 + 不可用」下抛 ``OfflineMissingWeightsError``（加载从未被触达，故无网络）。

**Feature: ego-pod-real-trainer**
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ego_pod.config.schema import TrainingParams
from ego_pod.exit_codes import EXIT_RUNTIME_FAILURE
from ego_pod.privacy.offline import (
    OFFLINE_MISSING_WEIGHTS_MESSAGE,
    OfflineMissingWeightsError,
    OfflineMode,
)
from ego_pod.trainer.peft_backend import PeftCpuBackend
from ego_pod.trainer.pipeline import run_training

# ---------------------------------------------------------------------------
# 测试用后端：真实 PeftCpuBackend 的子类，仅覆盖 load_base_model 使其确定失败
# ---------------------------------------------------------------------------


class _LoadReached(RuntimeError):
    """哨兵错误：标识 ``load_base_model`` 已被触达（证明未发生离线短路）。

    继承 :class:`RuntimeError` 而非 ``EgoError``，从而被 ``run_training`` 归一化
    包装为 ``TrainerError``——总之**不是** ``OfflineMissingWeightsError``。
    """


class _TrackingPeftCpuBackend(PeftCpuBackend):
    """真实 ``PeftCpuBackend`` 子类：记录 ``load`` 调用并以哨兵错误确定失败。

    保留真实后端的 ``name`` / ``hardware == "CPU"`` / ``identifier == "peft-cpu"``
    与 ``is_available``（即 ``_base_model_available`` 会真正调用注入的
    ``base_model_available_check``，而非因 STUB 而绕过）。``load_base_model`` 被
    覆盖为"记录调用次数并抛 :class:`_LoadReached`"，使「权重可用」分支无需任何
    ML 依赖或网络即可确定地走过离线门控并进入加载。
    """

    def __init__(self) -> None:
        self.load_calls = 0

    def load_base_model(self, model_id: str, quantization: str) -> Any:
        self.load_calls += 1
        raise _LoadReached("load_base_model reached (no real weights in test)")


# ---------------------------------------------------------------------------
# 生成策略
# ---------------------------------------------------------------------------

_model_ids = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_/",
    min_size=1,
    max_size=32,
)


def _valid_params() -> TrainingParams:
    """构造一组合法的 :class:`TrainingParams`（先于离线门控的参数校验需通过）。"""

    return TrainingParams(
        method="qlora",
        quantization="4bit",
        lora_rank=16,
        lora_alpha=32,
        learning_rate=2e-4,
        epochs=3,
        batch_size=4,
    )


def _run(
    backend: Any,
    *,
    model_id: str,
    enabled: bool,
    available: bool,
    tmp_path: Path,
):
    """以给定离线状态与权重可用性调用 ``run_training``。

    ``corpus_path`` 指向一个不存在的路径：在「短路」与「加载即失败」两种路径下
    它都不会被读取，故无需真实语料。
    """

    return run_training(
        tmp_path / "corpus.jsonl",
        _valid_params(),
        model_id=model_id,
        cwd=tmp_path,
        backend=backend,
        offline=OfflineMode(enabled=enabled),
        base_model_available_check=lambda _mid: available,
        stdout=None,
    )


# ===========================================================================
# Property 20: 离线模式缺失权重 → 退出码 1（任务 9.2）
# ===========================================================================


@given(model_id=_model_ids)
@settings(max_examples=200, deadline=None)
def test_offline_missing_weights_raises_before_load(
    model_id: str, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """离线启用 且 权重不可用 → 在加载前短路并抛 ``OfflineMissingWeightsError``。

    断言：
    * 抛出 :class:`OfflineMissingWeightsError`，消息为固定文案、退出码为 1；
    * 短路发生在 ``load_base_model`` **之前**（``load_calls == 0``）。

    **Feature: ego-pod-real-trainer, Property 20**

    **Validates: Requirements 2.3**
    """
    tmp_path = tmp_path_factory.mktemp("offline_missing")
    backend = _TrackingPeftCpuBackend()

    with pytest.raises(OfflineMissingWeightsError) as excinfo:
        _run(backend, model_id=model_id, enabled=True, available=False, tmp_path=tmp_path)

    assert str(excinfo.value) == OFFLINE_MISSING_WEIGHTS_MESSAGE
    assert excinfo.value.exit_code == EXIT_RUNTIME_FAILURE
    # 关键：在加载之前短路，load_base_model 从未被调用。
    assert backend.load_calls == 0


@given(model_id=_model_ids)
@settings(max_examples=200, deadline=None)
def test_offline_with_available_weights_does_not_short_circuit(
    model_id: str, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """离线启用 且 权重可用 → 不短路，继续进入加载（且非 OfflineMissingWeightsError）。

    断言：
    * **不**抛 :class:`OfflineMissingWeightsError`；
    * 流程进入 ``load_base_model``（``load_calls == 1``），证明未发生离线短路。

    **Feature: ego-pod-real-trainer, Property 20**

    **Validates: Requirements 2.3**
    """
    tmp_path = tmp_path_factory.mktemp("offline_available")
    backend = _TrackingPeftCpuBackend()

    with pytest.raises(Exception) as excinfo:  # noqa: PT011 - 下方精确断言类型
        _run(backend, model_id=model_id, enabled=True, available=True, tmp_path=tmp_path)

    # 不是离线缺权重短路（而是后续加载阶段的失败）。
    assert not isinstance(excinfo.value, OfflineMissingWeightsError)
    # 确实进入了加载阶段（未短路）。
    assert backend.load_calls == 1


@given(model_id=_model_ids, available=st.booleans())
@settings(max_examples=200, deadline=None)
def test_online_mode_never_short_circuits(
    model_id: str, available: bool, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """在线模式（offline 未启用）下无论权重是否可用都不触发离线短路。

    断言：流程总是进入 ``load_base_model``（``load_calls == 1``），且抛出的不是
    ``OfflineMissingWeightsError``——离线短路仅在 ``enabled`` 为真时生效。

    **Feature: ego-pod-real-trainer, Property 20**

    **Validates: Requirements 2.3**
    """
    tmp_path = tmp_path_factory.mktemp("online_mode")
    backend = _TrackingPeftCpuBackend()

    with pytest.raises(Exception) as excinfo:  # noqa: PT011 - 下方精确断言类型
        _run(
            backend,
            model_id=model_id,
            enabled=False,
            available=available,
            tmp_path=tmp_path,
        )

    assert not isinstance(excinfo.value, OfflineMissingWeightsError)
    assert backend.load_calls == 1


def test_real_peft_cpu_backend_offline_missing_weights(tmp_path: Path) -> None:
    """未改写的真实 ``PeftCpuBackend`` 在「离线 + 不可用」下抛 ``OfflineMissingWeightsError``。

    使用真实后端类（``hardware == "CPU"``，非 STUB），断言离线门控在加载之前生效；
    由于短路先于加载发生，``load_base_model`` 从不被触达，故无任何网络/权重依赖。

    **Feature: ego-pod-real-trainer, Property 20**

    **Validates: Requirements 2.3**
    """
    backend = PeftCpuBackend()

    with pytest.raises(OfflineMissingWeightsError) as excinfo:
        run_training(
            tmp_path / "corpus.jsonl",
            _valid_params(),
            model_id="qwen3.5-0.8B",
            cwd=tmp_path,
            backend=backend,
            offline=OfflineMode(enabled=True),
            base_model_available_check=lambda _mid: False,
        )

    assert str(excinfo.value) == OFFLINE_MISSING_WEIGHTS_MESSAGE
    assert excinfo.value.exit_code == EXIT_RUNTIME_FAILURE
