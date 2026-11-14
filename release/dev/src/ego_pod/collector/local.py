"""本地目录数据源适配器。

本模块实现 :class:`~ego_pod.collector.base.SourceAdapter` 协议的唯一 MVP
具体实现 :class:`LocalDirectoryAdapter`：从一个本地目录递归采集所有
"常规文件"并产出 :class:`~ego_pod.collector.base.CollectedFile` 元数据。

"常规文件"的判定遵循设计文档 §Components and Interfaces / 3. Collector：

* 使用 ``p.is_file() and not p.is_symlink()`` 初判，并以 ``stat.S_ISREG``
  二次确认，从而排除符号链接、设备文件、套接字、FIFO 管道；
* 隐藏文件（以 ``.`` 开头）与 0 字节空文件均保留（需求 R4-AC2）。

错误与边界行为：

* 路径不存在 → 返回空迭代器（不报错），由 Collector 编排层按需求
  R4-AC3 记录跳过警告；
* 根路径存在但无读权限 → 抛 :class:`PermissionError`（需求 R4-AC4）；
* 递归过程中子目录/子文件层级的读权限错误 → 同样抛
  :class:`PermissionError`（需求 R4-AC8）。

两类权限错误均向上抛出，交由 Collector 编排层映射为退出码 1。

隐私约束（需求 R9-AC1）：本适配器仅执行本地文件系统读取，不发起任何
网络请求。
"""

from __future__ import annotations

import os
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

from ego_pod.collector.base import Category, CollectedFile, SourceAdapter

if TYPE_CHECKING:
    # 仅供类型检查使用，避免运行时硬依赖（与 base.py 保持一致风格）。
    from ego_pod.config.schema import SourceEntry

__all__ = ["LocalDirectoryAdapter"]


def _utc_now_iso() -> str:
    """返回当前 UTC 时间的 ISO 8601 秒精度时间戳，例如 ``2025-11-25T08:00:00Z``。

    采用 ``Z`` 后缀表示 UTC（与 manifest/CollectedFile 的约定一致），
    ``microsecond`` 清零以保证秒精度。测试可对本函数进行 monkeypatch
    以注入固定时钟。
    """

    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .strftime("%Y-%m-%dT%H:%M:%SZ")
    )


class LocalDirectoryAdapter(SourceAdapter):
    """从本地目录递归采集常规文件的 :class:`SourceAdapter` 实现。"""

    def collect(
        self, entry: "SourceEntry", category: Category
    ) -> Iterator[CollectedFile]:
        """采集 ``entry.path`` 目录下属于 ``category`` 的所有常规文件。

        Args:
            entry: 来自 Ego_Config 的本地目录数据源条目。
            category: 该数据源对应的 Data_Category 小写标签。

        Yields:
            目录下每个常规文件对应的 :class:`CollectedFile`。

        Raises:
            PermissionError: 根路径或递归过程中遇到的子目录/子文件无读权限。
        """

        # 归一化为绝对路径，确保产出的 absolute_path 满足需求 R4-AC5。
        root = Path(os.path.abspath(entry.path))

        # 路径不存在 → 空迭代器（需求 R4-AC3，跳过警告由编排层负责）。
        if not root.exists():
            return

        yield from self._walk(root, category)

    def _walk(self, directory: Path, category: Category) -> Iterator[CollectedFile]:
        """递归遍历 ``directory``，惰性产出其下的常规文件元数据。

        权限错误（``os.scandir`` 打开目录失败、或对单个条目取 ``stat``
        失败）不被吞掉，直接向上抛出以满足需求 R4-AC4 / R4-AC8。
        """

        # ``os.scandir`` 在目录不可读时直接抛 PermissionError（根/子目录均适用）。
        # 先一次性物化条目并关闭句柄，避免深层递归期间长期占用文件描述符。
        with os.scandir(directory) as scanner:
            entries = list(scanner)

        # 按名称排序使遍历顺序确定，便于复现与测试（需求未规定类别内顺序）。
        for dir_entry in sorted(entries, key=lambda e: e.name):
            # 排除所有符号链接（文件或目录），且不跟随，避免引入符号链接目标
            # 与潜在的循环（需求 R4-AC2 排除符号链接）。
            if dir_entry.is_symlink():
                continue

            # 真实子目录 → 递归（子目录权限错误将在下层 scandir 处抛出，R4-AC8）。
            if dir_entry.is_dir(follow_symlinks=False):
                yield from self._walk(Path(dir_entry.path), category)
                continue

            # 候选常规文件：is_file() 初判（已排除符号链接）。
            # 设备文件 / 套接字 / FIFO 在此处 is_file() 为 False，被自然排除。
            if not dir_entry.is_file(follow_symlinks=False):
                continue

            # 对单个条目取 stat；权限错误向上抛出（需求 R4-AC8 子文件层级）。
            st = dir_entry.stat(follow_symlinks=False)

            # 以 S_ISREG 二次确认确为常规文件（设计文档要求）。
            if not stat.S_ISREG(st.st_mode):
                continue

            yield CollectedFile(
                absolute_path=os.path.abspath(dir_entry.path),
                size_bytes=st.st_size,
                category=category,
                read_at=_utc_now_iso(),
            )
