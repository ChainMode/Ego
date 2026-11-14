"""`ego_pod.cli` 命令分发骨架的单元测试（任务 4.4）。

本文件覆盖 Requirement 1（CLI 安装与命令入口）的五类用例：

- **R1-AC1**：``ego --help`` 在标准输出列出含 ``init`` / ``train`` / ``run`` 的
  帮助信息并以退出码 0 结束。
- **R1-AC2**：未提供任何子命令时输出与 ``--help`` 等价的帮助并以退出码 0 结束。
- **R1-AC3**：``ego <subcommand> --help`` 输出该子命令用法、参数与默认值并以
  退出码 0 结束。
- **R1-AC4**：不存在的子命令在标准错误输出含该名称的错误信息并以退出码 2 结束。
- **R1-AC5**：``ego`` 入口（``[project.scripts] ego = "ego_pod.cli:main"``）指向
  一个可调用的 :func:`ego_pod.cli.main`。
- **R1-AC6**：``ego --version`` 在标准输出打印语义化版本号并以退出码 0 结束。
- **R1-AC7**：对子命令传入缺失必填值或非法参数时在标准错误输出错误信息并以
  退出码 2 结束。

测试约定：``argparse`` 对 ``--help`` / ``--version`` / 解析错误一律抛
:class:`SystemExit`（派生自 ``BaseException``），因此需用
``pytest.raises(SystemExit)`` 包裹并断言 ``.code``。``--help`` / ``--version``
的退出码为 0，解析错误（未知子命令、缺失/非法参数）的退出码为 2。未提供子命令的
情形不抛 ``SystemExit``，而是由 :func:`ego_pod.cli.main` 返回 ``int`` 退出码 0。
"""

from __future__ import annotations

import pytest

from ego_pod import __version__
from ego_pod.cli import build_parser, main


def _exit_code(excinfo: pytest.ExceptionInfo[SystemExit]) -> int:
    """从捕获的 :class:`SystemExit` 中取出归一化为 ``int`` 的退出码。

    ``SystemExit.code`` 可能是 ``None``（视为 0）或字符串；统一转为整数以便断言。
    """
    code = excinfo.value.code
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    # argparse 不会以字符串退出码结束解析错误，但稳妥起见仍做转换。
    return int(code)


# ===========================================================================
# R1-AC1：ego --help → 退出码 0，stdout 含三个子命令
# ===========================================================================


def test_help_exits_zero_and_lists_subcommands(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert _exit_code(excinfo) == 0

    out = capsys.readouterr().out
    # 帮助信息应在标准输出列出三个子命令名称。
    for sub in ("init", "train", "run"):
        assert sub in out


def test_help_short_flag_also_exits_zero(capsys):
    # -h 等价于 --help。
    with pytest.raises(SystemExit) as excinfo:
        main(["-h"])
    assert _exit_code(excinfo) == 0
    assert "init" in capsys.readouterr().out


# ===========================================================================
# R1-AC2：无子命令 → 返回退出码 0，stdout 输出等价帮助
# ===========================================================================


def test_no_subcommand_prints_help_and_returns_zero(capsys):
    # 未提供子命令不抛 SystemExit，由 main 返回退出码 0。
    rc = main([])
    assert rc == 0

    out = capsys.readouterr().out
    # 输出应与 --help 等价：含三个子命令名称。
    for sub in ("init", "train", "run"):
        assert sub in out


def test_no_subcommand_output_matches_help(capsys):
    # R1-AC2 要求"等价于 ego --help"的帮助信息。
    main([])
    no_cmd_out = capsys.readouterr().out

    with pytest.raises(SystemExit):
        main(["--help"])
    help_out = capsys.readouterr().out

    assert no_cmd_out == help_out


# ===========================================================================
# R1-AC3：ego <subcommand> --help → 退出码 0，展示用法/参数/默认值
# ===========================================================================


@pytest.mark.parametrize("subcommand", ["init", "train", "run"])
def test_subcommand_help_exits_zero(subcommand, capsys):
    with pytest.raises(SystemExit) as excinfo:
        main([subcommand, "--help"])
    assert _exit_code(excinfo) == 0
    out = capsys.readouterr().out
    # 子命令帮助应至少包含其用法行（以程序名与子命令开头）。
    assert subcommand in out


def test_init_help_shows_force_option(capsys):
    with pytest.raises(SystemExit):
        main(["init", "--help"])
    assert "--force" in capsys.readouterr().out


def test_train_help_shows_config_option(capsys):
    with pytest.raises(SystemExit):
        main(["train", "--help"])
    assert "--config" in capsys.readouterr().out


def test_run_help_shows_edition_option_and_default(capsys):
    with pytest.raises(SystemExit):
        main(["run", "--help"])
    out = capsys.readouterr().out
    assert "--edition" in out
    # R1-AC3 要求展示默认值：run 的 --edition 默认 normal。
    assert "normal" in out


# ===========================================================================
# R1-AC4：未知子命令 → 退出码 2，stderr 含该名称
# ===========================================================================


def test_unknown_subcommand_exits_two(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["bogus"])
    assert _exit_code(excinfo) == 2
    err = capsys.readouterr().err
    # 错误信息应包含未知子命令名称。
    assert "bogus" in err


# ===========================================================================
# R1-AC6：ego --version → 退出码 0，stdout 打印版本号
# ===========================================================================


def test_version_prints_version_and_exits_zero(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert _exit_code(excinfo) == 0
    out = capsys.readouterr().out
    # 标准输出应包含当前已安装版本号。
    assert __version__ in out


# ===========================================================================
# R1-AC7：缺失必填值 / 非法参数 → 退出码 2，stderr 含错误信息
# ===========================================================================


def test_unknown_global_flag_exits_two(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--definitely-not-a-flag"])
    assert _exit_code(excinfo) == 2
    assert capsys.readouterr().err  # 应有错误信息输出到标准错误


def test_run_edition_missing_value_exits_two(capsys):
    # --edition 需要一个取值；缺失值是 argparse 解析错误 → 退出码 2。
    with pytest.raises(SystemExit) as excinfo:
        main(["run", "--edition"])
    assert _exit_code(excinfo) == 2
    err = capsys.readouterr().err
    assert "--edition" in err


def test_train_config_missing_value_exits_two(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["train", "--config"])
    assert _exit_code(excinfo) == 2
    err = capsys.readouterr().err
    assert "--config" in err


def test_init_unrecognized_argument_exits_two(capsys):
    # init 只接受 --force；多余的未知参数应触发解析错误退出码 2。
    with pytest.raises(SystemExit) as excinfo:
        main(["init", "--bogus-option"])
    assert _exit_code(excinfo) == 2
    assert capsys.readouterr().err


# ===========================================================================
# R1-AC5：ego 入口为可调用的 main（pip 注册的可执行命令）
# ===========================================================================


def test_main_is_callable_entry_point():
    # [project.scripts] ego = "ego_pod.cli:main" 指向本可调用对象。
    assert callable(main)


def test_build_parser_prog_is_ego():
    parser = build_parser()
    assert parser.prog == "ego"
