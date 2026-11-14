"""``ego`` 命令行入口：argparse 解析、子命令分发与离线模式装配。

本模块实现 Requirement 1（CLI 安装与命令入口）的命令分发骨架：

* 顶层全局参数 ``--version`` / ``--offline`` / ``--help``；
* 三个子命令 ``init`` / ``train`` / ``run``，分别支持 ``--force`` / ``--config``
  / ``--edition``；
* 未知子命令、缺失或非法参数由 argparse 默认行为以退出码 2 结束
  （R1-AC4 / R1-AC7）；
* ``--help`` 与无子命令调用打印帮助并以退出码 0 结束（R1-AC1 / R1-AC2）；
* ``--version`` 打印 :data:`ego_pod.__version__` 并以退出码 0 结束（R1-AC6）。

入口函数 :func:`main` 在解析参数后通过
:func:`ego_pod.privacy.offline.init_offline_mode` 装配离线模式，再分发到各子命令
实现模块。子命令实现模块按需惰性导入（见 :func:`_dispatch`），以便在尚未实现
``train`` / ``run`` 等模块时本模块仍可被安全导入。

顶层异常→退出码映射在 :func:`main` 中实现：``_dispatch`` 调用被
``try/except`` 包裹，受控的 :class:`~ego_pod.exit_codes.EgoError` 族异常按其
``exit_code`` 退出（ConfigError→2，其余→1），其它未预期 ``Exception`` 兜底映射到
:data:`~ego_pod.exit_codes.EXIT_RUNTIME_FAILURE`（1），且一律不向用户透出
traceback。argparse 自身的 ``SystemExit``（``--help`` / ``--version`` 退出码 0、
未知子命令与非法参数退出码 2）派生自 ``BaseException``，不会被该兜底吞掉。
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from ego_pod import __version__
from ego_pod.exit_codes import EXIT_OK, EXIT_RUNTIME_FAILURE, EgoError

__all__ = ["build_parser", "main"]

#: 程序名，统一用于 ``argparse`` 的 ``prog``，使帮助与版本输出均以 ``ego`` 起头。
PROG = "ego"

#: ``ego run --edition`` 的默认 Edition。MVP 仅支持 ``normal``（R12-AC2）；在此声明
#: 默认值以便 ``ego run --help`` 能展示默认值（R1-AC3）。实际取值校验由
#: ``commands/run.py``（任务 10.3）负责。
DEFAULT_EDITION = "normal"

#: 默认配置文件名。``ego train`` 未提供 ``--config`` 时回退到当前工作目录下的该文件
#: （R11-AC4）；实际加载逻辑由 ``commands/train.py``（任务 11.1）负责。
DEFAULT_CONFIG_FILENAME = "ego-config.yaml"


def build_parser() -> argparse.ArgumentParser:
    """构建并返回 ``ego`` 的顶层 argparse 解析器（含三个子命令）。

    解析器配置要点：

    * ``prog="ego"``；``--version`` 打印 :data:`ego_pod.__version__`；
    * 顶层 ``--offline`` 为全局开关（store_true）；
    * 子命令通过 ``add_subparsers(dest="command")`` 注册，且**非必需**——未提供子
      命令时 ``args.command`` 为 ``None``，由 :func:`main` 打印帮助并以退出码 0 结束
      （R1-AC2）；
    * 未知子命令与缺失/非法参数沿用 argparse 默认行为：``parser.error()`` 打印到
      标准错误并以退出码 2 结束（R1-AC4 / R1-AC7）。

    :returns: 配置完成的 :class:`argparse.ArgumentParser`。
    """
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Ego Pod —— 本地、隐私优先的个性化 AI 模型训练管线。"
            "提供 init / train / run 三条命令。"
        ),
    )
    # R1-AC6：打印语义化版本号并以退出码 0 结束（argparse version action 默认行为）。
    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
        help="打印 Ego Pod 版本号并退出",
    )
    # R9-AC6：全局离线模式开关；实际生效由 privacy.init_offline_mode 在 main 中装配。
    parser.add_argument(
        "--offline",
        action="store_true",
        help="启用离线模式，拒绝任何外发网络请求",
    )

    # 子命令注册。required=False（默认）：无子命令时 args.command 为 None。
    subparsers = parser.add_subparsers(
        dest="command",
        title="子命令",
        metavar="{init,train,run}",
        help="要执行的操作",
    )

    # --- ego init ---------------------------------------------------------
    init_parser = subparsers.add_parser(
        "init",
        help="在当前目录生成默认 ego-config.yaml",
        description="在当前工作目录初始化默认 Ego_Config（ego-config.yaml）。",
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="若 ego-config.yaml 已存在则覆盖它",
    )

    # --- ego train --------------------------------------------------------
    train_parser = subparsers.add_parser(
        "train",
        help="运行 采集→构建→训练→产出 端到端训练管线",
        description=(
            "依次执行 Collector → Compiler → Trainer → Vault，"
            "产出 normal edition 的 Ego_Package。"
        ),
    )
    train_parser.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help=(
            "Ego_Config 文件路径；省略时使用当前工作目录下的 "
            f"{DEFAULT_CONFIG_FILENAME}"
        ),
    )

    # --- ego run ----------------------------------------------------------
    run_parser = subparsers.add_parser(
        "run",
        help="加载训练产出并进入交互式对话",
        description="加载 ego-output 下指定 Edition 的 Ego_Package 并启动 REPL。",
    )
    run_parser.add_argument(
        "--edition",
        metavar="NAME",
        default=DEFAULT_EDITION,
        help="要加载的 Edition 名称（MVP 仅支持 normal）",
    )

    return parser


def _dispatch(args: argparse.Namespace) -> int:
    """根据已解析的 ``args.command`` 分发到对应子命令实现并返回其退出码。

    各子命令实现模块在对应分支内**惰性导入**：``train`` / ``run`` 等模块可能尚未
    实现（分别属于任务 11.1 / 10.3），惰性导入可保证本模块在它们缺失时仍可被导入
    与解析参数，仅在用户实际调用相应子命令时才需要其存在。

    :param args: :func:`build_parser` 解析得到的命名空间，``args.command`` 必为
        ``"init"`` / ``"train"`` / ``"run"`` 之一（``None`` 的情形已在 :func:`main`
        中处理）。
    :returns: 子命令返回的进程退出码。
    """
    if args.command == "init":
        from ego_pod.commands.init import run_init

        return run_init(args)
    if args.command == "train":
        from ego_pod.commands.train import run_train

        return run_train(args)
    if args.command == "run":
        from ego_pod.commands.run import run_repl

        return run_repl(args)

    # 理论不可达：argparse 已限定 command 取值；保留以显式表达不变量。
    raise AssertionError(f"未知子命令: {args.command!r}")


def main(argv: Sequence[str] | None = None) -> int:
    """``ego`` 命令主入口。

    流程：

    1. 构建解析器并解析 ``argv``（``None`` 时取 ``sys.argv[1:]``）。未知子命令、
       缺失/非法参数由 argparse 直接以退出码 2 结束（R1-AC4 / R1-AC7）；
       ``--version`` / ``--help`` 由 argparse 以退出码 0 结束（R1-AC6 / R1-AC1）。
    2. 无子命令时打印帮助并以退出码 0 结束（R1-AC2）。
    3. 通过 :func:`~ego_pod.privacy.offline.init_offline_mode` 装配离线模式
       （读取顶层 ``--offline`` 标志）。
    4. 在 ``try/except`` 中分发到对应子命令实现：受控 :class:`EgoError` 按其
       ``exit_code`` 退出并打印消息到标准错误；其它未预期 ``Exception`` 兜底映射到
       :data:`EXIT_RUNTIME_FAILURE` 且不透出 traceback。argparse 的
       ``SystemExit`` 不受此兜底影响。

    :param argv: 命令行参数序列（不含程序名）；``None`` 时使用 ``sys.argv[1:]``。
    :returns: 进程退出码。
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # R1-AC2：未提供子命令 → 输出与 --help 等价的帮助并以退出码 0 结束。
    if args.command is None:
        parser.print_help()
        return EXIT_OK

    # R9-AC6：装配离线模式开关（设置 HF 离线环境变量并打印提示）。
    from ego_pod.privacy.offline import init_offline_mode

    init_offline_mode(cli_offline=args.offline)

    # ------------------------------------------------------------------
    # 顶层异常→退出码映射（任务 4.2）。
    #
    # 仅包裹 _dispatch 调用：此时 argparse 已完成解析，``--help`` /
    # ``--version`` / 未知子命令 / 缺失或非法参数等情形均已由 argparse 通过
    # ``SystemExit`` 直接结束进程（退出码 0 或 2）。``SystemExit`` 派生自
    # ``BaseException`` 而非 ``Exception``，因此下面的 ``except Exception`` 兜底
    # 不会吞掉它，argparse 的退出码 2 行为（R1-AC4 / R1-AC7）得以保留。
    #
    # * 受控错误（``EgoError`` 族）：打印异常消息到标准错误并返回其携带的
    #   ``exit_code``（ConfigError→2，其余运行期错误→1），对应
    #   R1-AC4/AC7、R11-AC2/AC3、R12-AC5 等条款（设计 §Error Handling）。
    # * 其它未预期异常：兜底映射到 EXIT_RUNTIME_FAILURE（1），仅向用户打印
    #   一行简要错误，绝不透出 traceback（R12-AC5：不让 traceback 透出给用户）。
    # ------------------------------------------------------------------
    try:
        return _dispatch(args)
    except EgoError as e:
        print(e, file=sys.stderr)
        return e.exit_code
    except Exception as e:  # noqa: BLE001 - 顶层兜底：杜绝向用户透出 traceback
        print(f"ego: 运行时发生未预期错误: {e}", file=sys.stderr)
        return EXIT_RUNTIME_FAILURE
