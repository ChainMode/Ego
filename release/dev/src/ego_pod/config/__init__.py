"""Ego_Config 配置中心。

定义 Ego_Config 的内存结构（``schema``），并在后续任务中补充 YAML 加载
（``loader``）、默认配置渲染（``defaults``）与范围校验（``validators``）。
"""

from ego_pod.config.defaults import render_default_yaml, write_default
from ego_pod.config.loader import load
from ego_pod.config.schema import (
    EditionDef,
    EgoConfig,
    SourceEntry,
    TrainingParams,
)

__all__ = [
    "EditionDef",
    "EgoConfig",
    "SourceEntry",
    "TrainingParams",
    "render_default_yaml",
    "write_default",
    "load",
]
