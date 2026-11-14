"""``ego init`` 实现：在当前工作目录生成默认 ``ego-config.yaml``。

本模块实现 Requirement 2（``ego init`` 初始化工作目录）的全部验收标准：

* **R2-AC1**：在不含 ``ego-config.yaml`` 的目录中以 UTF-8 编码创建该文件。
* **R2-AC2–AC5**：文件内容（顶层四项、五个 Data_Category 的 ``local`` 条目、
  ``normal`` edition、QLoRA 五参数）由 :mod:`ego_pod.config.defaults` 渲染，本模块
  通过 :func:`~ego_pod.config.defaults.render_default_yaml` 复用其产物。
* **R2-AC6**：目录已含 ``ego-config.yaml`` 且未提供 ``--force`` 时拒绝写入，抛出
  携带"文件已存在"提示的 :class:`~ego_pod.exit_codes.EgoError`（退出码 1）。
* **R2-AC7**：提供 ``--force`` 时用默认配置覆盖原文件。
* **R2-AC8**：成功创建后在标准输出打印生成文件的绝对路径并以退出码 0 结束。
* **R2-AC9**：写入过程中发生 I/O / 权限错误时不在目标目录留下任何部分写入的
  ``ego-config.yaml``，并抛出携带错误原因的 :class:`~ego_pod.exit_codes.EgoError`
  （退出码 1）。

设计要点（见 design.md §1 CLI/init）：

* 入口 :func:`run_init` 接收 argparse 命名空间（仅用到 ``args.force``）并返回
  ``int`` 退出码；成功返回 :data:`~ego_pod.exit_codes.EXIT_OK`。失败一律抛出
  :class:`~ego_pod.exit_codes.EgoError`，由 CLI 顶层（任务 4.2）统一
  ``print(e, file=sys.stderr); return e.exit_code`` 映射为退出码 1 并打印到标准
  错误——因此本模块不自行向 stderr 打印错误，错误文案即异常消息。
* **原子写入**：为满足 R2-AC9"不留部分写入文件"，先把渲染内容写入目标目录下的
  临时文件，``fsync`` 后通过 :func:`os.replace` 原子替换为 ``ego-config.yaml``。
  任何阶段失败都会清理临时文件，且绝不产生半成品的 ``ego-config.yaml``；当目标
  已存在（``--force`` 覆盖）时，原文件在替换成功前保持不变。
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

from ego_pod.config.defaults import render_default_yaml
from ego_pod.exit_codes import EXIT_OK, EgoError

__all__ = ["run_init", "CONFIG_FILENAME"]

#: ``ego init`` 生成的配置文件名，固定为工作目录下的 ``ego-config.yaml``（R2-AC1）。
CONFIG_FILENAME = "ego-config.yaml"

#: R2-AC6：目标文件已存在且未提供 ``--force`` 时的错误文案（须含"已存在"提示）。
FILE_EXISTS_MESSAGE = "ego-config.yaml 已存在，拒绝写入"

#: R2-AC9：写入过程中发生 I/O / 权限错误时的错误文案前缀（须含错误原因）。
WRITE_FAILED_MESSAGE = "写入 ego-config.yaml 失败"


def _atomic_write_text(target: Path, text: str) -> None:
    """以原子方式将 ``text`` 以 UTF-8 写入 ``target``，保证不留部分写入文件。

    实现策略（满足 R2-AC9）：

    1. 在 ``target`` 所在目录创建一个临时文件（同目录可保证 :func:`os.replace`
       为同一文件系统内的原子操作）。
    2. 写入文本、``flush`` 并 ``fsync`` 落盘。
    3. 通过 :func:`os.replace` 原子地将临时文件替换为 ``target``——存在则覆盖
       （R2-AC7），不存在则创建（R2-AC1）。

    任意步骤抛出 :class:`OSError`（含 :class:`PermissionError`）时，清理临时文件
    后向上重新抛出原异常；此时目标目录不会留下半成品的 ``ego-config.yaml``，且
    若 ``target`` 原先存在，其内容在替换成功前保持不变。

    :param target: 目标 ``ego-config.yaml`` 路径。
    :param text: 待写入的完整 YAML 文本。
    :raises OSError: 创建临时文件、写入或原子替换过程中发生的任何 I/O / 权限错误。
    """
    parent = target.parent
    # 在目标同目录创建临时文件；dir=parent 确保后续 os.replace 为同文件系统原子操作。
    fd, tmp_name = tempfile.mkstemp(
        prefix=".ego-config-", suffix=".yaml.tmp", dir=parent
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
    except OSError:
        # 清理半成品临时文件，确保目录中不残留任何部分写入内容（R2-AC9）。
        # 临时文件可能尚未创建成功或已被 os.replace 消费，故 unlink 失败需吞掉。
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise


def run_init(args: argparse.Namespace) -> int:
    """``ego init`` 主入口：在当前工作目录生成默认 ``ego-config.yaml``。

    流程：

    1. 解析目标路径为当前工作目录下的 :data:`CONFIG_FILENAME`。
    2. 若目标已存在且未提供 ``--force`` → 抛 :class:`~ego_pod.exit_codes.EgoError`
       （含"已存在"提示，退出码 1，R2-AC6）。
    3. 渲染默认 YAML 并原子写入；写入失败（I/O / 权限错误）→ 清理半成品后抛
       :class:`~ego_pod.exit_codes.EgoError`（含错误原因，退出码 1，R2-AC9）。
    4. 成功后在标准输出打印生成文件的绝对路径并返回
       :data:`~ego_pod.exit_codes.EXIT_OK`（R2-AC8）。

    错误一律以异常形式上抛，由 CLI 顶层（任务 4.2）统一打印到标准错误并映射退出码，
    因此本函数不自行写 stderr。

    :param args: argparse 解析得到的命名空间，仅使用其 ``force`` 布尔标志。
    :returns: 成功时为 :data:`~ego_pod.exit_codes.EXIT_OK`（0）。
    :raises EgoError: 目标文件已存在且未指定 ``--force``，或写入时发生 I/O /
        权限错误（退出码均为 1）。
    """
    force = bool(getattr(args, "force", False))
    target = Path.cwd() / CONFIG_FILENAME
    abs_path = target.resolve()

    # R2-AC6：已存在且未 --force → 拒绝写入。使用 lexists 以便正确识别已存在的
    # 符号链接（即便其指向不存在的目标也视为"已存在"，拒绝覆盖）。
    if os.path.lexists(target) and not force:
        raise EgoError(f"{FILE_EXISTS_MESSAGE}: {abs_path}（如需覆盖请使用 --force）")

    text = render_default_yaml()
    try:
        _atomic_write_text(target, text)
    except OSError as exc:
        # R2-AC9：写入失败，半成品已在 _atomic_write_text 中清理；上抛错误原因。
        raise EgoError(f"{WRITE_FAILED_MESSAGE}: {abs_path}: {exc}") from exc

    # R2-AC8：成功创建，打印绝对路径到标准输出并以退出码 0 结束。
    print(str(abs_path))
    return EXIT_OK
