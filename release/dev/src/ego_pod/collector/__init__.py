"""Collector 模块 — 从本地目录数据源采集原始文件。

本模块负责按固定的 Data_Category 顺序（sense → action → thought →
skill → credentials）遍历 Ego_Config 中声明的本地目录，收集"常规文件"
清单及其采集元数据，供下游 Compiler 加工。

公开抽象位于 :mod:`ego_pod.collector.base`：

- :class:`~ego_pod.collector.base.SourceAdapter` —— 数据源适配器协议
- :class:`~ego_pod.collector.base.CollectedFile` —— 单个采集文件的元数据
"""

from ego_pod.collector.base import CollectedFile, SourceAdapter

__all__ = ["CollectedFile", "SourceAdapter"]
