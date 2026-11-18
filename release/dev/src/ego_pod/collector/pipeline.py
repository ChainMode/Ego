"""Collector 编排层：统一调度本地目录、URL、S3/OSS 云存储与可插拔数据源（ADR 0006 §5）。"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ego_pod.collector.base import Category, CollectedFile, SourceAdapter
from ego_pod.collector.cloud import CloudSourceAdapter
from ego_pod.collector.local import LocalDirectoryAdapter
from ego_pod.collector.plugins import get_collector_plugin
from ego_pod.collector.url import UrlSourceAdapter
from ego_pod.exit_codes import CollectorError
from ego_pod.logging_io import print_summary
from ego_pod.privacy.offline import OfflineMode

if TYPE_CHECKING:
    from ego_pod.config.schema import EgoConfig

__all__ = ["CATEGORY_ORDER", "CollectionResult", "run"]

CATEGORY_ORDER: tuple[Category, ...] = (
    "sense",
    "action",
    "thought",
    "skill",
    "credentials",
)


@dataclass(frozen=True)
class CollectionResult:
    """一次采集的完整结果，供下游 Compiler 消费。"""

    files: list[CollectedFile]
    summary: dict[Category, tuple[int, int]]


def run(
    config: "EgoConfig",
    *,
    adapter: SourceAdapter | None = None,
    cwd: Path | None = None,
    offline: OfflineMode | None = None,
) -> CollectionResult:
    """执行一次跨五类 Data_Category 的数据采集。"""

    default_local = adapter if adapter is not None else LocalDirectoryAdapter()
    base_cwd = Path.cwd() if cwd is None else Path(cwd)

    all_files: list[CollectedFile] = []
    summary: dict[Category, tuple[int, int]] = {}

    for category in CATEGORY_ORDER:
        summary[category] = _collect_category(
            config,
            default_local=default_local,
            category=category,
            sink=all_files,
            cwd=base_cwd,
            offline=offline,
        )

    for category in CATEGORY_ORDER:
        count, total_bytes = summary[category]
        print_summary(f"{category}: {count} files, {total_bytes} bytes")

    if not all_files:
        raise CollectorError(
            "[COLLECTOR] 五个 Data_Category 均无可用训练数据："
            "所有数据源不存在或不包含任何可用文件"
        )

    return CollectionResult(files=all_files, summary=summary)


def _resolve_adapter_for_entry(
    entry_type: str,
    *,
    default_local: SourceAdapter,
    cwd: Path,
    offline: OfflineMode | None,
) -> SourceAdapter | None:
    if entry_type == "local":
        return default_local
    if entry_type == "url":
        return UrlSourceAdapter(cwd=cwd, offline=offline)
    if entry_type in ("s3", "oss"):
        return CloudSourceAdapter(cwd=cwd, offline=offline)
    return get_collector_plugin(entry_type)


def _collect_category(
    config: "EgoConfig",
    *,
    default_local: SourceAdapter,
    category: Category,
    sink: list[CollectedFile],
    cwd: Path,
    offline: OfflineMode | None,
) -> tuple[int, int]:
    entry = config.sources.get(category)
    if entry is None:
        return (0, 0)

    selected_adapter = _resolve_adapter_for_entry(
        entry.type, default_local=default_local, cwd=cwd, offline=offline
    )
    if selected_adapter is None:
        print(
            f"[COLLECTOR] 跳过数据源条目 {category!r}（type={entry.type!r}）："
            "未注册的数据源类型（MVP 仅支持 `local` 数据源基线；v0.2 支持 local/url/s3/oss 或插件）",
            file=sys.stderr,
            flush=True,
        )
        return (0, 0)

    if entry.type == "local":
        raw_path = entry.path or ""
        root = Path(raw_path) if os.path.isabs(raw_path) else (cwd / raw_path).resolve()
        if not root.exists():
            print(
                f"[COLLECTOR] 跳过 Data_Category {category}：路径不存在 {root}",
                flush=True,
            )
            return (0, 0)
        if os.path.abspath(raw_path) != str(root):
            from dataclasses import replace

            entry = replace(entry, path=str(root))

    count = 0
    total_bytes = 0
    try:
        for collected in selected_adapter.collect(entry, category):
            sink.append(collected)
            count += 1
            total_bytes += collected.size_bytes
    except PermissionError as exc:
        failing_path = exc.filename if exc.filename else str(entry.path or cwd)
        reason = exc.strerror if exc.strerror else str(exc)
        raise CollectorError(
            f"[COLLECTOR] 采集 Data_Category {category} 时发生权限错误："
            f"{os.path.abspath(failing_path)}（{reason}）"
        ) from exc

    return (count, total_bytes)
