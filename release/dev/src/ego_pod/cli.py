"""``ego`` 命令行入口：argparse 解析、子命令分发与离线模式装配。

提供七个顶层子命令：``init`` / ``train`` / ``run`` / ``serve`` / ``slice`` / ``school`` / ``vault``。
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from ego_pod import __version__
from ego_pod.exit_codes import EXIT_OK, EXIT_RUNTIME_FAILURE, EgoError

__all__ = ["build_parser", "main"]

PROG = "ego"
DEFAULT_EDITION = "normal"
DEFAULT_CONFIG_FILENAME = "ego-config.yaml"


def build_parser() -> argparse.ArgumentParser:
    """构建并返回 ``ego`` 的顶层 argparse 解析器（含 init / train / run / vault）。"""

    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Ego Pod —— 本地、隐私优先的个性化 AI 模型修炼场与认知服务。"
            "提供 init / train / run / serve / slice / school / vault 命令。"
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
        help="打印 Ego Pod 版本号并退出",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="启用离线模式，拒绝任何外发网络请求",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        title="子命令",
        metavar="{init,train,run,serve,slice,school,vault}",
        help="要执行的操作",
    )

    # --- ego init ---------------------------------------------------------
    init_parser = subparsers.add_parser(
        "init",
        help="在当前目录生成默认或多规格 ego-config.yaml",
        description="在当前工作目录初始化 Ego_Config（ego-config.yaml）。",
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="若 ego-config.yaml 已存在则覆盖它",
    )
    init_parser.add_argument(
        "--all-editions",
        action="store_true",
        help="生成包含 lite / normal / plus / super 四档标准规格的配置模板",
    )
    init_parser.add_argument(
        "--interactive",
        action="store_true",
        help="启动交互式初始化向导定制数据源目录、规格与基座模型",
    )

    # --- ego train --------------------------------------------------------
    train_parser = subparsers.add_parser(
        "train",
        help="运行 采集→蒸馏→多规格训练→签名入库 端到端管线",
        description=(
            "依次执行 Collector → Compiler（深层专精度蒸馏）→ Trainer → Vault，"
            "产出单规格或多规格签名的 Ego_Package。"
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
    train_parser.add_argument(
        "--update",
        action="store_true",
        help="增量更新模式：基于内容哈希复用未变更的已采集源文本缓存",
    )
    train_parser.add_argument(
        "--parallel-editions",
        action="store_true",
        help="在安全隔离目录前提下并行执行多 Edition 训练",
    )

    # --- ego run ----------------------------------------------------------
    run_parser = subparsers.add_parser(
        "run",
        help="按需加载指定 Edition 与网格深度并进入交互式对话",
        description="加载 ego-output 下指定 Edition 的 Ego_Package 并启动 REPL。",
    )
    run_parser.add_argument(
        "--edition",
        metavar="NAME",
        default=DEFAULT_EDITION,
        help="要加载的 Edition 名称（默认 normal；支持 lite / normal / plus / super 或自定义规格）",
    )
    run_parser.add_argument(
        "--zone",
        metavar="ZONE:DEPTH",
        action="append",
        default=None,
        help="按需加载指定功能分区及深度前缀（可重复指定，如 --zone coding:2 --zone knowledge:1）",
    )
    run_parser.add_argument(
        "--kernel-only",
        action="store_true",
        help="仅加载与分区无关的共享 Kernel（Level-0），与 --zone 互斥",
    )

    # --- ego vault --------------------------------------------------------
    vault_parser = subparsers.add_parser(
        "vault",
        help="管理 Ego_Package 版本历史快照、验签回滚与缓存清理",
        description="查询、验签回滚或彻底清理 ego-output 版本仓库。",
    )
    vault_sub = vault_parser.add_subparsers(
        dest="vault_action",
        required=True,
        metavar="{list,rollback,purge}",
        help="Vault 管理动作",
    )
    vault_sub.add_parser(
        "list",
        help="列出 ego-output/.history 下的历史快照清单",
    )
    rb_parser = vault_sub.add_parser(
        "rollback",
        help="在验签与完整性核验通过后原子恢复指定历史快照",
    )
    rb_parser.add_argument(
        "snapshot_id",
        help="目标历史快照标识（如 0.2.0-0001）",
    )
    rb_parser.add_argument(
        "--public-key-hex",
        default=None,
        help="可选：要求快照签名匹配指定的 64 位十六进制 Ed25519 受信任公钥",
    )
    purge_parser = vault_sub.add_parser(
        "purge",
        help="彻底删除本地模型产物（含版本历史）与 .ego 中间缓存",
    )
    purge_parser.add_argument(
        "--all",
        action="store_true",
        required=True,
        help="确认删除全部 ego-output/ 与 .ego/ 本地数据",
    )

    # --- ego serve --------------------------------------------------------
    serve_parser = subparsers.add_parser(
        "serve",
        help="启动快慢双系统三合一本地认知服务（HTTP JSON API）",
        description=(
            "验签加载 .egopod（或工作区 ego-output/）并启动三合一本地认知服务："
            "/health、/v1/intuition、/v1/dynamic-context、/v1/guard-rewrite。"
        ),
    )
    serve_parser.add_argument(
        "package_or_workspace",
        nargs="?",
        default=".",
        help="目标 .egopod 模型包目录或包含 ego-output/ 的工作区目录（默认当前目录）",
    )
    serve_parser.add_argument(
        "--edition",
        metavar="NAME",
        default=None,
        help="要加载的 Edition 名称（默认优先 normal，单规格切片包自动匹配）",
    )
    serve_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="HTTP 监听地址（默认 127.0.0.1）",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=7980,
        help="HTTP 监听端口（默认 7980；传 0 自动分配临时空闲端口）",
    )
    serve_parser.add_argument(
        "--zone",
        metavar="ZONE:DEPTH",
        action="append",
        default=None,
        help="按需激活指定功能分区及深度（可重复指定，如 --zone coding:2）",
    )
    serve_parser.add_argument(
        "--kernel-only",
        action="store_true",
        help="仅加载共享 Kernel（Level-0），与 --zone 互斥",
    )
    serve_parser.add_argument(
        "--allow-unsigned-dev",
        action="store_true",
        help="显式允许在本地开发调试期挂载未签名的开发包（篡改包仍一律拒绝）",
    )
    serve_parser.add_argument(
        "--trusted-public-key-hex",
        default=None,
        help="可选：要求签名公钥严格匹配指定的 64 位十六进制 Ed25519 公钥",
    )

    # --- ego slice --------------------------------------------------------
    slice_parser = subparsers.add_parser(
        "slice",
        help="按场景/领域与体积上限导出独立可验签的微型 .egopod 子模型切片",
        description=(
            "提取 Kernel + 指定领域 Cell，按体积上限裁剪并嵌入 ego-submodel-slice/1.0 "
            "描述符后完成 Ed25519 独立签名。"
        ),
    )
    slice_parser.add_argument(
        "source_package",
        help="源 .egopod 模型包目录（或含 ego-output/ 的工作区目录）",
    )
    slice_parser.add_argument(
        "--domain",
        action="append",
        required=True,
        help=(
            "目标切片领域（支持 coding|social|knowledge|creative|sense|drone-photo "
            "或自定义分区名；可重复指定或逗号分隔）"
        ),
    )
    slice_parser.add_argument(
        "--out",
        required=True,
        help="导出独立微型 .egopod 切片的目标目录路径",
    )
    slice_parser.add_argument(
        "--max-size-mb",
        type=float,
        default=None,
        help="切片包最大允许体积上限（MB）；超限时自动削减深层 Cell",
    )
    slice_parser.add_argument(
        "--signing-key",
        required=True,
        help="用于签发独立切片的密钥（64 位 hex 种子、subkey:<hex> 或密钥文件路径）",
    )
    slice_parser.add_argument(
        "--signer-fay-id",
        default=None,
        help="可选：显式指定签名者规范 FayID（ifay_...）；省略时继承源包或从公钥确定性派生",
    )
    slice_parser.add_argument(
        "--edition",
        default=None,
        help="可选：指定从源包哪个 Edition 提取切片",
    )
    slice_parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="可选：限制每个保留分区的最大网格深度",
    )
    slice_parser.add_argument(
        "--target-device-class",
        default=None,
        help="可选：目标受限设备类型标识（如 drone-camera / wearable-sensor）",
    )

    # --- ego school -------------------------------------------------------
    school_parser = subparsers.add_parser(
        "school",
        help="Ego School：冷启动对话/账本脱敏导入与伴生增量蒸馏进化",
        description="导入人类原型自有表达与纠偏瞬间（import），或运行伴生增量修炼闭环（evolve）。",
    )
    school_sub = school_parser.add_subparsers(
        dest="school_action",
        required=True,
        metavar="{import,evolve}",
        help="Ego School 动作",
    )
    imp_parser = school_sub.add_parser(
        "import",
        help="冷启动导入历史笔记、对话或主体纠偏账本（仅提取人类原型发言与纠偏对并自动脱敏）",
    )
    imp_parser.add_argument(
        "--format",
        choices=("markdown", "jsonl", "openai", "ifay-ledger"),
        required=True,
        help="导入数据源格式",
    )
    imp_parser.add_argument(
        "--input",
        required=True,
        help="输入文件或目录路径",
    )
    imp_parser.add_argument(
        "--workspace",
        default=".",
        help="目标 Ego 工作区目录（默认当前目录）",
    )

    evo_parser = school_sub.add_parser(
        "evolve",
        help="执行伴生增量蒸馏：吸收新回流纠偏账本 -> 增量训练与归档 -> 一致性探针晋升",
    )
    evo_parser.add_argument(
        "workspace",
        nargs="?",
        default=".",
        help="目标 Ego 工作区目录（默认当前目录）",
    )
    evo_parser.add_argument(
        "--ledger",
        default=None,
        help="可选：iFay 主体回流事件账本路径（如 primary_body_events.jsonl）",
    )
    evo_parser.add_argument(
        "--config",
        default=None,
        help="可选：自定义 ego-config.yaml 路径",
    )
    evo_parser.add_argument(
        "--min-consistency",
        type=float,
        default=0.70,
        help="晋升所需最低一致性/偏好探针门限（默认 0.70；不达标自动回滚）",
    )

    return parser


def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "init":
        from ego_pod.commands.init import run_init

        return run_init(args)
    if args.command == "train":
        from ego_pod.commands.train import run_train

        return run_train(args)
    if args.command == "run":
        from ego_pod.commands.run import run_repl

        return run_repl(args)
    if args.command == "vault":
        from ego_pod.commands.vault_cmd import run_vault_cmd

        return run_vault_cmd(args)
    if args.command == "serve":
        from ego_pod.commands.serve import run_serve_cmd

        return run_serve_cmd(args)
    if args.command == "slice":
        from ego_pod.commands.slice_cmd import run_slice_cmd

        return run_slice_cmd(args)
    if args.command == "school":
        from ego_pod.commands.school import run_school_cmd

        return run_school_cmd(args)

    raise AssertionError(f"未知子命令: {args.command!r}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return EXIT_OK

    from ego_pod.privacy.offline import init_offline_mode

    init_offline_mode(cli_offline=args.offline)

    try:
        return _dispatch(args)
    except EgoError as e:
        print(e, file=sys.stderr)
        return e.exit_code
    except Exception as e:  # noqa: BLE001
        print(f"ego: 运行时发生未预期错误: {e}", file=sys.stderr)
        return EXIT_RUNTIME_FAILURE
