"""Collector 编排层。

本模块把 :class:`~ego_pod.collector.base.SourceAdapter`（MVP 唯一实现为
:class:`~ego_pod.collector.local.LocalDirectoryAdapter`）按固定的
Data_Category 顺序串接起来，完成一次完整的数据采集：

#. 按 ``sense → action → thought → skill → credentials`` 的固定顺序遍历五个
   Data_Category（需求 R4-AC1）；
#. 仅从 ``type: local`` 的数据源条目读取；遇到非 ``local`` 条目则跳过该条目
   并在标准错误输出 "MVP 仅支持 ``local`` 数据源" 的提示（需求 R9-AC1/AC2）；
#. 路径不存在时——且**仅**此情形——在标准输出打印包含 Data_Category 名称与
   路径的跳过警告（需求 R4-AC3）；
#. 统计每个 Data_Category 的文件数与总字节数；
#. 根/子路径权限错误向上传播为 :class:`~ego_pod.exit_codes.CollectorError`，
   映射到退出码 1（需求 R4-AC4 / R4-AC8）；
#. 全部处理完成后在标准输出打印固定五行采集摘要，缺失/为空的类别填 0
   （需求 R4-AC6）；
#. 若五个 Data_Category 合计 0 个常规文件，则以
   :class:`~ego_pod.exit_codes.CollectorError`（退出码 1）结束（需求 R4-AC7）。

设计依据：design.md §Components and Interfaces / 3. Collector 模块。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ego_pod.collector.base import Category, CollectedFile, SourceAdapter
from ego_pod.collector.local import LocalDirectoryAdapter
from ego_pod.exit_codes import CollectorError
from ego_pod.logging_io import print_summary

if TYPE_CHECKING:
    # 仅供类型检查使用，避免运行时硬依赖（与 base.py / local.py 保持一致风格）。
    from ego_pod.config.schema import EgoConfig

__all__ = ["CATEGORY_ORDER", "CollectionResult", "run"]

#: 五个 Data_Category 的固定处理顺序（需求 R4-AC1）。采集摘要亦按此顺序打印。
CATEGORY_ORDER: tuple[Category, ...] = (
    "sense",
    "action",
    "thought",
    "skill",
    "credentials",
)


@dataclass(frozen=True)
class CollectionResult:
    """一次采集的完整结果，供下游 Compiler 消费。

    Attributes:
        files: 按 Data_Category 固定顺序、类别内确定顺序汇集的全部
            :class:`~ego_pod.collector.base.CollectedFile`。
        summary: Data_Category → ``(文件数, 总字节数)`` 的映射，五个类别均存在，
            缺失/为空的类别为 ``(0, 0)``。
    """

    files: list[CollectedFile]
    summary: dict[Category, tuple[int, int]]


def run(
    config: "EgoConfig",
    *,
    adapter: SourceAdapter | None = None,
) -> CollectionResult:
    """执行一次完整的本地目录数据采集。

    Args:
        config: 已加载并校验通过的 Ego_Config。
        adapter: 数据源适配器，默认 :class:`LocalDirectoryAdapter`。
            允许注入以便测试（保持对 :class:`SourceAdapter` 协议的解耦）。

    Returns:
        :class:`CollectionResult`，包含全部 :class:`CollectedFile` 与按类别的
        计数/字节摘要。

    Raises:
        CollectorError: 采集过程中遇到权限错误（R4-AC4 / R4-AC8），或五个
            Data_Category 合计无任何可用常规文件（R4-AC7）；均映射到退出码 1。
    """

    adapter = adapter if adapter is not None else LocalDirectoryAdapter()

    all_files: list[CollectedFile] = []
    summary: dict[Category, tuple[int, int]] = {}

    # 按固定顺序逐个处理五个 Data_Category（R4-AC1）。
    for category in CATEGORY_ORDER:
        summary[category] = _collect_category(config, adapter, category, all_files)

    # R4-AC6：完成所有类别后打印固定五行采集摘要（缺失/为空填 0）。
    # 在 R4-AC7 的整体性判定之前打印，确保即便全空也能观测到五行摘要。
    for category in CATEGORY_ORDER:
        count, total_bytes = summary[category]
        print_summary(f"{category}: {count} files, {total_bytes} bytes")

    # R4-AC7：五个 Data_Category 合计 0 个常规文件 → 无可用训练数据，退出码 1。
    if not all_files:
        raise CollectorError(
            "[COLLECTOR] 五个 Data_Category 均无可用训练数据："
            "所有本地目录不存在或不包含任何常规文件"
        )

    return CollectionResult(files=all_files, summary=summary)


def _collect_category(
    config: "EgoConfig",
    adapter: SourceAdapter,
    category: Category,
    sink: list[CollectedFile],
) -> tuple[int, int]:
    """处理单个 Data_Category，把采集到的文件追加进 ``sink`` 并返回 ``(计数, 字节)``。

    跳过情形（均返回 ``(0, 0)``）：

    * 该 Data_Category 在 ``config.sources`` 中无条目（防御性处理；正常配置
      经 R2-AC3 保证五类齐全）；
    * 条目 ``type`` 非 ``local`` —— 跳过并在 stderr 报错（R9-AC2）；
    * 条目路径不存在 —— 跳过并在 stdout 记录跳过警告（R4-AC3，仅此情形）。

    Raises:
        CollectorError: 根/子路径权限错误（R4-AC4 / R4-AC8）。
    """

    entry = config.sources.get(category)

    # 防御性：配置缺失该类别条目时按空类别处理，不触发任何警告。
    if entry is None:
        return (0, 0)

    # R9-AC1/AC2：仅从 type=local 条目读取；非 local 条目跳过并在 stderr 报错。
    if entry.type != "local":
        print(
            f"[COLLECTOR] 跳过数据源条目 {category!r}（type={entry.type!r}）："
            "MVP 仅支持 `local` 数据源",
            file=sys.stderr,
            flush=True,
        )
        return (0, 0)

    root = Path(os.path.abspath(entry.path))

    # R4-AC3：路径不存在——且仅此情形——在 stdout 打印跳过警告后继续其他类别。
    # 目录存在但为空/仅含子目录/仅含被排除类型，不在此触发（由 R4-AC7 整体判定）。
    if not root.exists():
        print(
            f"[COLLECTOR] 跳过 Data_Category {category}：路径不存在 {root}",
            flush=True,
        )
        return (0, 0)

    count = 0
    total_bytes = 0
    try:
        # 适配器为惰性生成器：权限错误在迭代过程中抛出（根/子路径均适用）。
        for collected in adapter.collect(entry, category):
            sink.append(collected)
            count += 1
            total_bytes += collected.size_bytes
    except PermissionError as exc:
        # R4-AC4（根路径）/ R4-AC8（子路径）：携带类别名、绝对路径与原因，退出码 1。
        failing_path = exc.filename if exc.filename else str(root)
        reason = exc.strerror if exc.strerror else str(exc)
        raise CollectorError(
            f"[COLLECTOR] 采集 Data_Category {category} 时发生权限错误："
            f"{os.path.abspath(failing_path)}（{reason}）"
        ) from exc

    return (count, total_bytes)
