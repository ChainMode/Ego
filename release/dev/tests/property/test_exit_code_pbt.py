"""`ego_pod.cli` 退出码映射的属性测试（hypothesis）。

本文件实现 design.md「Correctness Properties」中的 **Property 14: exit_code
正确映射**，对应 tasks.md 任务 12.4：

    CLI 顶层退出码总属于 {0, 1, 2}，且退出码 0 当且仅当 train/run/init 的所有
    阶段均成功。

围绕这一不变量，从两个互补的输入空间施加属性约束：

1. **异常→退出码映射**：用 hypothesis 在 :class:`~ego_pod.exit_codes.EgoError`
   族的全部子类上取值，断言每个异常实例的 ``exit_code`` 类属性符合设计
   §Error Handling 的映射表（``ConfigError`` → 2；其余运行期错误与基类
   ``EgoError`` → 1），且当且仅当为 ``ConfigError`` 时退出码为 2。进一步把这些
   异常注入 CLI 顶层 :func:`ego_pod.cli.main`（通过替换 ``_dispatch``），断言
   ``main`` 将受控异常映射为其 ``exit_code``、把未预期 ``Exception`` 兜底映射为
   1、把成功返回码原样透传，且"返回 0 当且仅当被分发的命令返回 0"。

2. **argparse 用法错误**：用 hypothesis 生成未知子命令与未知/缺失取值参数等
   非法 argv，断言 ``main`` 经由 argparse 以 ``SystemExit(2)`` 结束
   （R1-AC4 / R1-AC7）；并以采样方式覆盖 ``--help`` / ``--version`` / 无子命令
   等成功路径（退出码 0，R1-AC1 / R1-AC2 / R1-AC3 / R1-AC6）。

所有注入都通过 :func:`unittest.mock.patch` 替换 ``ego_pod.cli._dispatch``
（位于 argparse 解析之后），因此既不触达任何真实子命令实现，也不进行任何文件
或网络 I/O；argparse 自身的 ``SystemExit``（派生自 ``BaseException``）不受
``main`` 内 ``except Exception`` 兜底影响，退出码 2 行为得以保留。

**Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.6, 1.7, 11.2, 11.3**
"""

from __future__ import annotations

import string
from unittest import mock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ego_pod import __version__
from ego_pod.cli import main
from ego_pod.exit_codes import (
    EXIT_OK,
    EXIT_RUNTIME_FAILURE,
    EXIT_USAGE_OR_CONFIG,
    CollectorError,
    CompilerError,
    ConfigError,
    EgoError,
    RuntimeError_,
    TrainerError,
    VaultError,
)

# ---------------------------------------------------------------------------
# 公共常量与辅助
# ---------------------------------------------------------------------------

#: 顶层退出码合法取值集合（Property 14：退出码总属于 {0, 1, 2}）。
_VALID_EXIT_CODES = {EXIT_OK, EXIT_RUNTIME_FAILURE, EXIT_USAGE_OR_CONFIG}

#: 三个合法子命令名。提供合法 ``args.command`` 以便 ``main`` 进入 ``_dispatch``
#: 分支（无子命令会在 ``_dispatch`` 之前直接以退出码 0 返回）。
_SUBCOMMANDS = ["init", "train", "run"]

#: ``EgoError`` 族子类 → 期望退出码（设计 §Error Handling 映射表）。
#: ``ConfigError`` 表达"用法/配置错误"→ 2；其余运行期错误与基类 → 1。
_ERROR_EXIT_MAP: list[tuple[type[EgoError], int]] = [
    (EgoError, EXIT_RUNTIME_FAILURE),
    (ConfigError, EXIT_USAGE_OR_CONFIG),
    (CollectorError, EXIT_RUNTIME_FAILURE),
    (CompilerError, EXIT_RUNTIME_FAILURE),
    (TrainerError, EXIT_RUNTIME_FAILURE),
    (VaultError, EXIT_RUNTIME_FAILURE),
    (RuntimeError_, EXIT_RUNTIME_FAILURE),
]

#: 未预期（非 EgoError）异常类型，应被 ``main`` 顶层兜底映射为退出码 1。
_UNEXPECTED_EXCEPTIONS: list[type[Exception]] = [
    ValueError,
    RuntimeError,
    KeyError,
    TypeError,
    OSError,
    Exception,
]

#: argparse 顶层长选项；生成"未知参数"时需避开它们及其唯一前缀（argparse 默认
#: 开启缩写匹配，例如 ``--v`` 会匹配 ``--version``）。
_KNOWN_LONG_OPTS = ("--help", "--version", "--offline")

#: 错误消息字符集：排除代理项（Unicode 类别 ``Cs``），避免 ``print`` 到被捕获的
#: stderr 时触发 ``UnicodeEncodeError``。消息内容不影响退出码，仅作为构造参数。
_safe_message = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",)), max_size=40
)

#: 在 ``EgoError`` 族子类与期望退出码上取值。
_error_cases = st.sampled_from(_ERROR_EXIT_MAP)

#: 合法子命令取值。
_subcommands = st.sampled_from(_SUBCOMMANDS)

#: 透传退出码取值（覆盖成功 0 与失败 1 / 2 三种）。
_passthrough_codes = st.sampled_from(sorted(_VALID_EXIT_CODES))


def _exit_code(excinfo: pytest.ExceptionInfo[SystemExit]) -> int:
    """把捕获到的 :class:`SystemExit` 退出码归一化为 ``int``。

    ``SystemExit.code`` 可能为 ``None``（视为 0）、``int`` 或字符串。
    """
    code = excinfo.value.code
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    return int(code)


# ===========================================================================
# 异常族 exit_code 类属性映射
# ===========================================================================


@given(case=_error_cases, msg=_safe_message)
@settings(max_examples=200, deadline=None)
def test_ego_error_exit_code_attribute(
    case: tuple[type[EgoError], int], msg: str
) -> None:
    """任意 ``EgoError`` 子类实例的 ``exit_code`` 符合映射表且总属于 {0,1,2}。

    并验证"退出码为 2 当且仅当该异常是 ``ConfigError``"——即只有配置/用法类错误
    映射到 2，其余运行期错误与基类一律映射到 1。

    **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.6, 1.7, 11.2, 11.3**
    """
    exc_cls, expected = case
    err = exc_cls(msg)

    assert err.exit_code == expected
    assert err.exit_code in _VALID_EXIT_CODES
    # 退出码 2 ⇔ ConfigError；其余（含基类 EgoError）→ 1。
    assert (err.exit_code == EXIT_USAGE_OR_CONFIG) == isinstance(err, ConfigError)


# ===========================================================================
# CLI 顶层：受控 EgoError → 其 exit_code
# ===========================================================================


@given(case=_error_cases, command=_subcommands, msg=_safe_message)
@settings(max_examples=200, deadline=None)
def test_main_maps_ego_error_to_its_exit_code(
    case: tuple[type[EgoError], int], command: str, msg: str
) -> None:
    """``main`` 把分发阶段抛出的 ``EgoError`` 映射为该异常的 ``exit_code``。

    覆盖 R11-AC2（流水线任一阶段以非零退出码结束时，``ego train`` 以相同非零退出码
    结束——各阶段错误经 ``EgoError`` 族携带退出码上抛）与 R1-AC4/AC7、R12-AC5 的
    顶层映射语义。退出码恒属于 {0, 1, 2}。

    **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.6, 1.7, 11.2, 11.3**
    """
    exc_cls, expected = case
    # 在 argparse 解析之后注入异常：替换 _dispatch 使其抛出受控异常。
    with mock.patch("ego_pod.cli._dispatch", side_effect=exc_cls(msg)):
        rc = main([command])

    assert rc == expected
    assert rc in _VALID_EXIT_CODES
    # 非零结束：受控错误一定不返回 0（仅成功路径返回 0）。
    assert rc != EXIT_OK


# ===========================================================================
# CLI 顶层：未预期 Exception → 兜底退出码 1
# ===========================================================================


@given(
    exc_cls=st.sampled_from(_UNEXPECTED_EXCEPTIONS),
    command=_subcommands,
    msg=_safe_message,
)
@settings(max_examples=200, deadline=None)
def test_main_maps_unexpected_exception_to_one(
    exc_cls: type[Exception], command: str, msg: str
) -> None:
    """``main`` 把分发阶段抛出的非 ``EgoError`` 异常兜底映射为退出码 1。

    顶层 ``except Exception`` 不透出 traceback，统一返回
    :data:`EXIT_RUNTIME_FAILURE`（1）。

    **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.6, 1.7, 11.2, 11.3**
    """
    with mock.patch("ego_pod.cli._dispatch", side_effect=exc_cls(msg)):
        rc = main([command])

    assert rc == EXIT_RUNTIME_FAILURE
    assert rc in _VALID_EXIT_CODES


# ===========================================================================
# CLI 顶层：成功返回码透传，且 0 当且仅当分发成功
# ===========================================================================


@given(command=_subcommands, code=_passthrough_codes)
@settings(max_examples=150, deadline=None)
def test_main_passes_through_dispatch_code(command: str, code: int) -> None:
    """``main`` 原样透传 ``_dispatch`` 返回的退出码；返回 0 ⇔ 分发返回 0。

    这正是 Property 14 的核心："退出码 0 当且仅当所有阶段成功"——CLI 不会无中生有
    地把失败码改写为 0，也不会把成功码改写为非 0。覆盖 R11-AC3（四阶段均以退出码 0
    结束时 ``ego train`` 以退出码 0 结束）与 R11-AC2（失败码透传）。

    **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.6, 1.7, 11.2, 11.3**
    """
    with mock.patch("ego_pod.cli._dispatch", return_value=code):
        rc = main([command])

    assert rc == code
    assert rc in _VALID_EXIT_CODES
    # 返回 0 当且仅当被分发命令返回 0。
    assert (rc == EXIT_OK) == (code == EXIT_OK)


# ===========================================================================
# argparse 用法错误：未知子命令 / 未知参数 → SystemExit(2)
# ===========================================================================

#: 未知子命令：小写字母串，排除三个合法子命令名。
_unknown_subcommand = st.text(
    alphabet=string.ascii_lowercase, min_size=1, max_size=15
).filter(lambda s: s not in set(_SUBCOMMANDS))


@given(sub=_unknown_subcommand)
@settings(max_examples=200, deadline=None)
def test_unknown_subcommand_exits_two(sub: str) -> None:
    """任意未知子命令使 ``main`` 经 argparse 以 ``SystemExit(2)`` 结束（R1-AC4）。

    **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.6, 1.7, 11.2, 11.3**
    """
    with pytest.raises(SystemExit) as excinfo:
        main([sub])
    code = _exit_code(excinfo)
    assert code == EXIT_USAGE_OR_CONFIG
    assert code in _VALID_EXIT_CODES


def _is_problematic_flag(flag: str) -> bool:
    """判断生成的长选项是否可能被 argparse 解析为合法/缩写选项。

    argparse 默认开启缩写匹配：若 ``flag`` 是某个已知长选项的唯一前缀（或与其相等），
    则不会触发解析错误，应从"未知参数"输入空间中过滤掉。
    """
    return any(opt == flag or opt.startswith(flag) for opt in _KNOWN_LONG_OPTS)


#: 未知长选项：``--`` + 小写字母串，排除合法选项及其缩写前缀。
_unknown_flag = (
    st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=12)
    .map(lambda s: "--" + s)
    .filter(lambda f: not _is_problematic_flag(f))
)


@given(flag=_unknown_flag)
@settings(max_examples=200, deadline=None)
def test_unknown_global_flag_exits_two(flag: str) -> None:
    """任意未知顶层长选项使 ``main`` 以 ``SystemExit(2)`` 结束（R1-AC7）。

    **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.6, 1.7, 11.2, 11.3**
    """
    with pytest.raises(SystemExit) as excinfo:
        main([flag])
    code = _exit_code(excinfo)
    assert code == EXIT_USAGE_OR_CONFIG
    assert code in _VALID_EXIT_CODES


#: 需要取值但缺失取值的参数：``run --edition`` / ``train --config``（R1-AC7）。
_missing_value_argv = st.sampled_from(
    [
        ["run", "--edition"],
        ["train", "--config"],
    ]
)


@given(argv=_missing_value_argv)
@settings(max_examples=50, deadline=None)
def test_missing_argument_value_exits_two(argv: list[str]) -> None:
    """子命令参数缺失取值使 ``main`` 以 ``SystemExit(2)`` 结束（R1-AC7）。

    **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.6, 1.7, 11.2, 11.3**
    """
    with pytest.raises(SystemExit) as excinfo:
        main(list(argv))
    code = _exit_code(excinfo)
    assert code == EXIT_USAGE_OR_CONFIG
    assert code in _VALID_EXIT_CODES


# ===========================================================================
# 成功路径：--help / 子命令 --help / --version → 0；无子命令 → 0
# ===========================================================================

#: 以 ``SystemExit(0)`` 结束的成功 argv 形态：
#: ``--help`` / ``-h`` / ``--version`` 以及 ``<subcommand> --help``。
_help_version_argv = st.one_of(
    st.sampled_from([["--help"], ["-h"], ["--version"]]),
    st.sampled_from(_SUBCOMMANDS).map(lambda c: [c, "--help"]),
)


@given(argv=_help_version_argv)
@settings(max_examples=100, deadline=None)
def test_help_and_version_exit_zero(argv: list[str]) -> None:
    """``--help`` / ``<sub> --help`` / ``--version`` 均以 ``SystemExit(0)`` 结束。

    覆盖 R1-AC1（``ego --help``）、R1-AC3（``ego <sub> --help``）、R1-AC6
    （``ego --version``）。

    **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.6, 1.7, 11.2, 11.3**
    """
    with pytest.raises(SystemExit) as excinfo:
        main(list(argv))
    code = _exit_code(excinfo)
    assert code == EXIT_OK
    assert code in _VALID_EXIT_CODES


def test_no_subcommand_returns_zero() -> None:
    """未提供任何子命令时 ``main`` 返回退出码 0（R1-AC2）。

    无子命令路径在 ``_dispatch`` 之前直接返回，不抛 ``SystemExit``。

    **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.6, 1.7, 11.2, 11.3**
    """
    rc = main([])
    assert rc == EXIT_OK
    assert rc in _VALID_EXIT_CODES


def test_version_string_is_printed(capsys) -> None:
    """``--version`` 在标准输出打印当前版本号（R1-AC6 的可观测确认）。

    **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.6, 1.7, 11.2, 11.3**
    """
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert _exit_code(excinfo) == EXIT_OK
    assert __version__ in capsys.readouterr().out
