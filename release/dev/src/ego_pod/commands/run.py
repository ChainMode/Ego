"""``ego run`` 实现：加载 Ego_Package 并进入交互式对话（Requirement 12）。

本模块是 ``ego run`` 子命令的业务入口，由 :mod:`ego_pod.cli` 在分发阶段惰性导入
（``from ego_pod.commands.run import run_repl`` → ``return run_repl(args)``）。它把
设计文档 §7「Runtime 模块」与「``ego run`` 数据流」落地为以下步骤：

1. **解析 Edition**：从 argparse 命名空间读取 ``--edition``，缺省为 ``normal``
   （R12-AC2）。
2. **R12-AC4**：Edition 取值非 ``normal`` → 抛
   :class:`~ego_pod.exit_codes.RuntimeError_`（携带"MVP 仅支持 ``normal`` Edition"
   文案，退出码 1）。
3. **R12-AC3**：``ego-output/<edition>/`` 子目录不存在 → 抛
   :class:`~ego_pod.exit_codes.RuntimeError_`（携带"未找到 Ego_Package，请先执行
   ``ego train``"文案，退出码 1）。
4. **加载与完整性校验（R12-AC1 / R12-AC5 / R12-AC9）**：委托
   :func:`ego_pod.runtime.inference.load_inference_session` 读取并校验
   ``manifest.json`` / ``<edition>/config.json`` /
   ``<edition>/adapters/ego-lora.safetensors``；任一缺失或损坏时该函数抛
   :class:`~ego_pod.exit_codes.RuntimeError_`（退出码 1）。Stub 训练产出会在此
   打印 Stub 模式提示并返回固定回复会话（R12-AC9）。
5. **进入 REPL（R12-AC6 / R12-AC7 / R12-AC8）**：把加载得到的推理会话注入运行时
   REPL 循环并返回其退出码。

**命名冲突的解耦**：运行时的交互循环函数同样命名为 ``run_repl``
（:func:`ego_pod.runtime.repl.run_repl`），而 CLI 期望 ``commands.run`` 暴露一个
名为 ``run_repl`` 的命令入口。为避免同名遮蔽，本模块把运行时 REPL 以别名
``run_repl_loop`` 导入，并将命令入口本身命名为 :func:`run_repl`（供 ``cli._dispatch``
调用）。

**异常→退出码映射**：与 :mod:`ego_pod.commands.init` 保持一致——错误一律以
:class:`~ego_pod.exit_codes.EgoError`（这里为其子类
:class:`~ego_pod.exit_codes.RuntimeError_`）形式上抛，由 CLI 顶层
（:func:`ego_pod.cli.main` 的 ``try/except EgoError``）统一
``print(e, file=sys.stderr); return e.exit_code`` 映射为退出码 1，因此本模块不自行
向标准错误打印错误文案，错误文案即异常消息。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, TextIO

from ego_pod.exit_codes import RuntimeError_
from ego_pod.runtime.inference import (
    DEFAULT_EDITION,
    OUTPUT_DIRNAME,
    load_inference_session,
)
from ego_pod.runtime.repl import run_repl as run_repl_loop
from ego_pod.runtime.selection import LoadSelection

__all__ = [
    "EDITION_NOT_SUPPORTED_MESSAGE",
    "PACKAGE_NOT_FOUND_MESSAGE",
    "run_repl",
]

#: R12-AC4：``--edition`` 取值非 ``normal`` 时的错误文案（须含"仅支持 normal"提示）。
EDITION_NOT_SUPPORTED_MESSAGE = "MVP 仅支持 `normal` Edition"

#: R12-AC3：指定 Edition 的 Ego_Package 子目录不存在时的错误文案
#: （须含"请先执行 ego train"提示）。
PACKAGE_NOT_FOUND_MESSAGE = "未找到 Ego_Package，请先执行 `ego train`"


def _relative_label(path: Path, base: Path) -> str:
    """返回 ``path`` 相对于 ``base`` 的相对路径字符串，用于错误信息。

    无法相对化（不在 ``base`` 子树内或解析失败）时回退为绝对路径字符串。

    :param path: 目标路径。
    :param base: 基准目录（通常为进程工作目录）。
    :returns: 适合放入错误消息的相对（或绝对）路径标签。
    """
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except (ValueError, OSError):
        return str(path)


def run_repl(
    args: argparse.Namespace,
    *,
    cwd: Path | None = None,
    backend: Any | None = None,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    selection: LoadSelection | None = None,
) -> int:
    """``ego run`` 主入口：加载 Ego_Package 并进入交互式 REPL（Requirement 12）。

    流程见模块文档：解析 Edition（默认 ``normal``，R12-AC2）→ 校验 Edition 取值
    （R12-AC4）→ 校验 Edition 子目录存在（R12-AC3）→ 加载并校验产物文件
    （R12-AC1/AC5/AC9，委托 :func:`load_inference_session`）→ 注入推理会话进入运行时
    REPL（R12-AC6/AC7/AC8）。

    错误一律以 :class:`~ego_pod.exit_codes.RuntimeError_` 上抛，由 CLI 顶层统一打印到
    标准错误并映射为退出码 1，因此本函数不自行写标准错误。

    :param args: argparse 解析得到的命名空间，使用其 ``edition`` 属性（缺省 ``normal``）。
    :param cwd: 进程工作目录，默认 :meth:`pathlib.Path.cwd`；用于定位 ``ego-output/``。
        提供此参数主要便于测试注入临时目录。
    :param backend: 显式指定的真实推理后端（便于测试）；默认按探测优先级选择。Stub
        路径不使用此参数。
    :param stdin: REPL 输入流，默认 :data:`sys.stdin`；测试可注入 :class:`io.StringIO`。
    :param stdout: 输出流（Stub 模式提示、REPL 提示符与回复），默认 :data:`sys.stdout`。
    :param stderr: REPL 错误流（推理异常的简要信息），默认 :data:`sys.stderr`。
    :param selection: 一次加载选择（:class:`~ego_pod.runtime.selection.LoadSelection`），
        透传给 :func:`load_inference_session`；默认 ``None`` ⇒ Full_In_Load（加载
        Kernel + 每分区 Level ``1..max``，需求 5.1）。本任务不新增 CLI flag，仅打通
        API 通路，后续可由 ``--zone <name>:<depth>`` 之类解析为 ``LoadSelection``。
    :returns: REPL 正常结束时为 :data:`~ego_pod.exit_codes.EXIT_OK`（0，R12-AC8）。
    :raises RuntimeError_: Edition 非 ``normal``（R12-AC3 之外的 R12-AC4）、指定
        Edition 子目录不存在（R12-AC3），或产物文件缺失/损坏（R12-AC5，由
        :func:`load_inference_session` 上抛）。
    """
    # R12-AC2：解析 Edition，缺省为 normal。
    edition = getattr(args, "edition", DEFAULT_EDITION)
    if edition is None:
        edition = DEFAULT_EDITION

    # R12-AC4：MVP 仅支持 normal Edition；其余取值直接以退出码 1 结束。
    if edition != DEFAULT_EDITION:
        raise RuntimeError_(f"{EDITION_NOT_SUPPORTED_MESSAGE}（收到 --edition {edition!r}）")

    base_cwd = Path.cwd() if cwd is None else Path(cwd)
    edition_dir = base_cwd / OUTPUT_DIRNAME / edition

    # R12-AC3：指定 Edition 的 Ego_Package 子目录不存在 → "请先执行 ego train"。
    # 与 R12-AC5（子目录存在但文件缺失/损坏）区分：后者由 load_inference_session 处理。
    if not edition_dir.is_dir():
        label = _relative_label(edition_dir, base_cwd)
        raise RuntimeError_(f"{PACKAGE_NOT_FOUND_MESSAGE}（缺少目录：{label}）")

    # R12-AC1 / R12-AC5 / R12-AC9：加载并校验 manifest.json / config.json / 适配器；
    # Stub 训练产出在此打印 Stub 模式提示并返回固定回复会话。任一文件缺失/损坏时
    # load_inference_session 抛 RuntimeError_（退出码 1）。
    session = load_inference_session(
        edition,
        cwd=base_cwd,
        backend=backend,
        stdout=stdout,
        selection=selection,
    )

    # R12-AC6/AC7/AC8：注入推理会话进入运行时 REPL；会话退出时由 REPL 调用 close()
    # 释放资源。运行时 REPL 以别名 run_repl_loop 导入以避免与本命令入口同名遮蔽。
    return run_repl_loop(session, stdin=stdin, stdout=stdout, stderr=stderr)
