"""本特性新增/复用错误类型统一映射到退出码 1 的属性测试（hypothesis）。

本文件实现 design.md「Correctness Properties」中的 **Property 21**，对应
tasks.md 任务 12.3：

* **Property 21: 特性新增错误统一映射到退出码 1** —— 对本特性新增/复用的训练与
  推理错误类型（:class:`~ego_pod.exit_codes.TrainerError`、
  :class:`~ego_pod.exit_codes.RuntimeError_`、
  :class:`~ego_pod.privacy.offline.OfflineMissingWeightsError`），其均为
  :class:`~ego_pod.exit_codes.EgoError` 子类且 ``exit_code == 1``
  （:data:`~ego_pod.exit_codes.EXIT_RUNTIME_FAILURE`），**不引入新的退出码取值**
  （退出码恒属于 ``{EXIT_OK, EXIT_RUNTIME_FAILURE, EXIT_USAGE_OR_CONFIG}``）。

本特性完全复用 MVP 的 ``exit_codes.py`` 与 ``EgoError`` 层级，不新增退出码常量
（design.md §Error Handling，R10-AC5）。本文件刻意只覆盖**本特性所依赖的三个**
错误类型，与既有 ``test_exit_code_pbt.py``（MVP Property 14，覆盖完整 ``EgoError``
族与 CLI 顶层映射）互补而不重叠。

依赖门控：本属性仅断言异常类的类属性与继承关系，**无需安装任何 ML 依赖**，可在
最小环境（仅 ``pytest`` + ``hypothesis``）下运行。错误消息文本不影响退出码，仅作为
构造参数施加，以验证退出码与消息内容无关。

**Feature: ego-pod-real-trainer, Property 21**

**Validates: Requirements 10.5**
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from ego_pod.exit_codes import (
    EXIT_OK,
    EXIT_RUNTIME_FAILURE,
    EXIT_USAGE_OR_CONFIG,
    EgoError,
    RuntimeError_,
    TrainerError,
)
from ego_pod.privacy.offline import OfflineMissingWeightsError

# ---------------------------------------------------------------------------
# 公共常量与策略
# ---------------------------------------------------------------------------

#: 退出码合法取值集合（本特性不引入新取值，恒属于 {0, 1, 2}）。
_VALID_EXIT_CODES = {EXIT_OK, EXIT_RUNTIME_FAILURE, EXIT_USAGE_OR_CONFIG}

#: 本特性新增/复用的三个错误类型（design.md Property 21）。
_FEATURE_ERROR_CLASSES: list[type[EgoError]] = [
    TrainerError,
    RuntimeError_,
    OfflineMissingWeightsError,
]

#: 在三个错误类型上取值。
_error_classes = st.sampled_from(_FEATURE_ERROR_CLASSES)

#: 错误消息字符集：排除代理项（Unicode 类别 ``Cs``），避免在构造/打印消息时触发
#: ``UnicodeEncodeError``。消息内容不影响退出码，仅作为构造参数。
_safe_message = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",)), max_size=40
)


# ===========================================================================
# Property 21：三个错误类型均为 EgoError 子类且 exit_code == 1，不引入新取值
# ===========================================================================


@given(exc_cls=_error_classes, msg=_safe_message)
@settings(max_examples=200, deadline=None)
def test_feature_errors_map_to_exit_code_one(
    exc_cls: type[EgoError], msg: str
) -> None:
    """任意特性错误实例均为 ``EgoError`` 且 ``exit_code == 1``，不引入新退出码。

    对 ``TrainerError`` / ``RuntimeError_`` / ``OfflineMissingWeightsError`` 三者，
    在任意消息文本下断言：

    1. 实例是 :class:`~ego_pod.exit_codes.EgoError` 的子类实例（受 CLI 顶层
       ``except EgoError`` 统一捕获）；
    2. ``exit_code == EXIT_RUNTIME_FAILURE``（== 1）；
    3. ``exit_code`` 仍落在 ``{EXIT_OK, EXIT_RUNTIME_FAILURE,
       EXIT_USAGE_OR_CONFIG}`` 内——本特性不引入新的退出码取值（R10-AC5）。

    **Validates: Requirements 10.5**
    """
    err = exc_cls(msg)

    # 1) 受控错误：EgoError 子类与实例。
    assert issubclass(exc_cls, EgoError)
    assert isinstance(err, EgoError)
    # 2) 统一映射到退出码 1。
    assert err.exit_code == EXIT_RUNTIME_FAILURE
    assert err.exit_code == 1
    # 3) 不引入新的退出码取值。
    assert err.exit_code in _VALID_EXIT_CODES


@given(exc_cls=_error_classes)
@settings(max_examples=50, deadline=None)
def test_feature_errors_default_construct_to_exit_code_one(
    exc_cls: type[EgoError],
) -> None:
    """三个错误类型在「无显式消息」下构造也固定映射到退出码 1。

    ``OfflineMissingWeightsError`` 的构造函数带默认消息参数；
    ``TrainerError`` / ``RuntimeError_`` 继承 ``Exception`` 可零参构造。无论是否
    提供消息，``exit_code`` 都恒为 1 且不引入新取值。

    **Validates: Requirements 10.5**
    """
    err = exc_cls()

    assert isinstance(err, EgoError)
    assert err.exit_code == EXIT_RUNTIME_FAILURE
    assert err.exit_code in _VALID_EXIT_CODES
