"""`ego_pod.privacy.offline` 离线模式开关的单元测试。

覆盖设计文档 §8 与需求 R9-AC6 / R9-AC7：

- ``--offline`` 与 ``EGO_OFFLINE=1`` 任一启用即进入离线模式；非显式启用
  （含 ``EGO_OFFLINE`` 取非 ``"1"`` 值、缺省、空串）时不进入离线行为。
- 离线模式启用时设置 ``HF_HUB_OFFLINE=1`` 与 ``TRANSFORMERS_OFFLINE=1``，并在
  标准错误（或注入流）打印离线提示；未启用时不变更环境、不打印。
- ``OfflineMissingWeightsError`` 携带正确 stderr 文案与退出码 1。

所有用例通过注入的 ``env`` 字典与 ``stream`` 完成，不修改真实 ``os.environ``，
也不触达真实网络。
"""

from __future__ import annotations

import io

import pytest

from ego_pod.exit_codes import EXIT_RUNTIME_FAILURE, EgoError
from ego_pod.privacy.offline import (
    ENV_EGO_OFFLINE,
    ENV_HF_HUB_OFFLINE,
    ENV_TRANSFORMERS_OFFLINE,
    OFFLINE_ENABLED_MESSAGE,
    OFFLINE_MISSING_WEIGHTS_MESSAGE,
    OfflineMissingWeightsError,
    OfflineMode,
    init_offline_mode,
    is_offline_requested,
    set_active_offline_mode,
)


@pytest.fixture(autouse=True)
def _reset_active_offline_mode():
    """隔离进程内"活动离线模式"全局状态，避免跨用例污染。"""
    set_active_offline_mode(None)
    yield
    set_active_offline_mode(None)


# ---------------------------------------------------------------------------
# is_offline_requested：启用判定与 EGO_OFFLINE 取值精确性（R9-AC6）
# ---------------------------------------------------------------------------


def test_cli_offline_flag_enables():
    """``--offline`` 单独启用即生效，与环境变量无关。"""
    assert is_offline_requested(True, {}) is True


def test_env_var_value_1_enables():
    """``EGO_OFFLINE=1`` 单独启用即生效。"""
    assert is_offline_requested(False, {ENV_EGO_OFFLINE: "1"}) is True


def test_cli_and_env_both_enabled():
    """两者同时启用仍为启用。"""
    assert is_offline_requested(True, {ENV_EGO_OFFLINE: "1"}) is True


def test_neither_enabled_is_offline_false():
    """两者均未启用时不进入离线行为。"""
    assert is_offline_requested(False, {}) is False


@pytest.mark.parametrize("value", ["0", "true", "TRUE", "yes", "", "01", " 1", "1 "])
def test_env_var_non_1_values_do_not_enable(value):
    """``EGO_OFFLINE`` 取非严格 ``"1"`` 值时不启用离线模式（R9-AC6）。

    避免因 ``"0"`` / ``"true"`` / 空串 / 带空白等取值被误判为启用。
    """
    assert is_offline_requested(False, {ENV_EGO_OFFLINE: value}) is False


# ---------------------------------------------------------------------------
# init_offline_mode：启用时设置环境变量并打印提示（R9-AC6）
# ---------------------------------------------------------------------------


def test_init_offline_via_cli_flag_sets_env_and_prints():
    env: dict[str, str] = {}
    stream = io.StringIO()

    mode = init_offline_mode(cli_offline=True, env=env, stream=stream)

    assert mode.enabled is True
    assert env[ENV_HF_HUB_OFFLINE] == "1"
    assert env[ENV_TRANSFORMERS_OFFLINE] == "1"
    assert OFFLINE_ENABLED_MESSAGE in stream.getvalue()


def test_init_offline_via_env_var_sets_env_and_prints():
    env: dict[str, str] = {ENV_EGO_OFFLINE: "1"}
    stream = io.StringIO()

    mode = init_offline_mode(cli_offline=False, env=env, stream=stream)

    assert mode.enabled is True
    assert env[ENV_HF_HUB_OFFLINE] == "1"
    assert env[ENV_TRANSFORMERS_OFFLINE] == "1"
    assert OFFLINE_ENABLED_MESSAGE in stream.getvalue()


# ---------------------------------------------------------------------------
# init_offline_mode：未启用时不变更环境、不打印（R9-AC6 不自动进入离线）
# ---------------------------------------------------------------------------


def test_init_online_does_not_mutate_env_or_print():
    env: dict[str, str] = {}
    stream = io.StringIO()

    mode = init_offline_mode(cli_offline=False, env=env, stream=stream)

    assert mode.enabled is False
    assert ENV_HF_HUB_OFFLINE not in env
    assert ENV_TRANSFORMERS_OFFLINE not in env
    assert stream.getvalue() == ""


def test_init_online_with_non_1_env_value_does_not_enable():
    """``EGO_OFFLINE`` 取非 ``"1"`` 值时不进入离线行为，环境不被变更。"""
    env: dict[str, str] = {ENV_EGO_OFFLINE: "0"}
    stream = io.StringIO()

    mode = init_offline_mode(cli_offline=False, env=env, stream=stream)

    assert mode.enabled is False
    assert ENV_HF_HUB_OFFLINE not in env
    assert ENV_TRANSFORMERS_OFFLINE not in env
    assert stream.getvalue() == ""


def test_init_offline_registers_active_mode():
    """init_offline_mode 将解析结果注册为进程内活动离线模式。"""
    from ego_pod.privacy.offline import get_active_offline_mode

    env: dict[str, str] = {}
    mode = init_offline_mode(cli_offline=True, env=env, stream=io.StringIO())
    assert get_active_offline_mode() is mode


# ---------------------------------------------------------------------------
# OfflineMissingWeightsError：文案与退出码（R9-AC7）
# ---------------------------------------------------------------------------


def test_missing_weights_error_default_message():
    err = OfflineMissingWeightsError()
    assert str(err) == OFFLINE_MISSING_WEIGHTS_MESSAGE
    assert OFFLINE_MISSING_WEIGHTS_MESSAGE == "离线模式下缺失 Base_Model 权重"


def test_missing_weights_error_exit_code_is_1():
    err = OfflineMissingWeightsError()
    assert err.exit_code == 1
    assert err.exit_code == EXIT_RUNTIME_FAILURE


def test_missing_weights_error_is_ego_error():
    """可被 CLI 顶层 ``except EgoError`` 统一捕获。"""
    assert isinstance(OfflineMissingWeightsError(), EgoError)


def test_missing_weights_error_custom_message_keeps_exit_code():
    err = OfflineMissingWeightsError("自定义文案")
    assert str(err) == "自定义文案"
    assert err.exit_code == EXIT_RUNTIME_FAILURE


# ---------------------------------------------------------------------------
# OfflineMode.require_base_model_available：R9-AC7 强制点
# ---------------------------------------------------------------------------


def test_require_base_model_raises_when_offline_and_missing():
    """离线模式 + 权重不可用 → 抛 OfflineMissingWeightsError。"""
    mode = OfflineMode(enabled=True)
    with pytest.raises(OfflineMissingWeightsError) as exc_info:
        mode.require_base_model_available(available=False)
    assert str(exc_info.value) == OFFLINE_MISSING_WEIGHTS_MESSAGE
    assert exc_info.value.exit_code == 1


def test_require_base_model_ok_when_offline_and_available():
    """离线模式 + 权重可用 → 不抛异常。"""
    mode = OfflineMode(enabled=True)
    assert mode.require_base_model_available(available=True) is None


def test_require_base_model_noop_when_online_even_if_missing():
    """在线模式下即使权重不可用也不抛（由 R8-AC7 单独处理，不走本路径）。"""
    mode = OfflineMode(enabled=False)
    assert mode.require_base_model_available(available=False) is None
