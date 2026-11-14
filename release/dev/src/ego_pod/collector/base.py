"""Collector 模块的抽象定义。

本模块定义数据源适配器的统一接口与采集结果的数据结构，使 Collector
编排层（:mod:`ego_pod.collector.pipeline`）无需关心具体数据源类型，
亦为 Phase 2+ 的 URL / 云存储数据源预留扩展点。

设计依据：design.md §Components and Interfaces / 3. Collector 模块。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    # 仅供类型检查使用，避免在 Config 模块尚未就绪时产生运行时硬依赖。
    from ego_pod.config.schema import SourceEntry

__all__ = ["Category", "CollectedFile", "SourceAdapter"]

# 五个固定 Data_Category 的小写标签取值集合。
Category = Literal["sense", "action", "thought", "skill", "credentials"]


@dataclass(frozen=True)
class CollectedFile:
    """单个被采集"常规文件"的元数据。

    对应需求 R4-AC5：记录绝对路径、字节大小、Data_Category 标签与
    ISO 8601 UTC 秒精度时间戳；0 字节空文件同样记录全部字段。

    Attributes:
        absolute_path: 被采集文件的绝对路径。
        size_bytes: 文件字节大小（非负整数，0 字节文件取 0）。
        category: 该文件所属的 Data_Category 小写标签。
        read_at: 采集时刻的 ISO 8601 UTC 秒精度时间戳，
            例如 ``2025-11-25T08:00:00Z``。
    """

    absolute_path: str
    size_bytes: int
    category: Category
    read_at: str


@runtime_checkable
class SourceAdapter(Protocol):
    """数据源适配器协议。

    每种数据源类型（MVP 仅 ``local``）提供一个实现：从给定的
    :class:`~ego_pod.config.schema.SourceEntry` 与 Data_Category 标签
    出发，惰性产出该来源下的 :class:`CollectedFile`。

    隐私约束（需求 R9-AC1）：适配器仅从 ``type: local`` 的数据源条目
    读取数据，不发起任何网络请求。
    """

    def collect(
        self, entry: "SourceEntry", category: Category
    ) -> Iterator[CollectedFile]:
        """采集 ``entry`` 指向的数据源下属于 ``category`` 的常规文件。

        Args:
            entry: 来自 Ego_Config 的单个数据源条目。
            category: 该数据源对应的 Data_Category 小写标签。

        Returns:
            惰性产出 :class:`CollectedFile` 的迭代器。
        """
        ...
