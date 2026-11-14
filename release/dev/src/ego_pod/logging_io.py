"""阶段日志与摘要打印工具。

本模块集中提供 Ego_Pod 流水线在标准输出上的两类单行打印能力：

* :func:`stage_event` —— 打印 ``ego train`` 各阶段进入/离开的时间戳记录，
  格式为 ``[<ISO 8601 含时区偏移、秒精度>] <STAGE> START|END``
  （对应 Requirements 11.5）。
* :func:`print_summary` —— 打印可被外部捕获的单行摘要，供 Collector 采集摘要、
  Compiler 脱敏/过滤摘要、Trainer 后端选择/进度记录等场景复用
  （对应 Requirements 4.6、5.5、7.3、8.2、8.8）。

时间戳统一通过 :func:`now_timestamp` 生成；测试可对该函数进行 monkeypatch
以注入固定时钟。
"""

from __future__ import annotations

import sys
from datetime import datetime
from enum import Enum
from typing import TextIO

__all__ = ["Stage", "Event", "now_timestamp", "stage_event", "print_summary"]


class Stage(str, Enum):
    """``ego train`` 流水线的四个阶段名称（固定大写）。"""

    COLLECTOR = "COLLECTOR"
    COMPILER = "COMPILER"
    TRAINER = "TRAINER"
    VAULT = "VAULT"


class Event(str, Enum):
    """阶段事件类型。"""

    START = "START"
    END = "END"


def now_timestamp() -> str:
    """返回当前本地时间的 ISO 8601 字符串（含时区偏移、精确到秒）。

    例如 ``2025-11-25T08:00:00+08:00``。``microsecond`` 被清零以保证秒精度，
    ``astimezone()`` 为时间附加本地时区从而带上偏移量。

    测试可对本函数进行 monkeypatch 以注入固定时钟。
    """

    return datetime.now().astimezone().replace(microsecond=0).isoformat()


def _coerce_stage(stage: Stage | str) -> str:
    """将 ``stage`` 归一化为受支持的大写阶段名称。"""

    value = stage.value if isinstance(stage, Stage) else str(stage).strip().upper()
    try:
        return Stage(value).value
    except ValueError as exc:  # 未知阶段名
        valid = ", ".join(member.value for member in Stage)
        raise ValueError(
            f"未知阶段名称 {stage!r}，必须是 {valid} 之一"
        ) from exc


def _coerce_event(event: Event | str) -> str:
    """将 ``event`` 归一化为 ``START`` 或 ``END``。"""

    value = event.value if isinstance(event, Event) else str(event).strip().upper()
    try:
        return Event(value).value
    except ValueError as exc:  # 未知事件类型
        valid = ", ".join(member.value for member in Event)
        raise ValueError(
            f"未知事件类型 {event!r}，必须是 {valid} 之一"
        ) from exc


def stage_event(
    stage: Stage | str,
    event: Event | str,
    *,
    stream: TextIO | None = None,
) -> str:
    """打印并返回一条阶段事件记录。

    输出形如 ``[2025-11-25T08:00:00+08:00] COLLECTOR START`` 的单行记录到
    标准输出（或指定的 ``stream``）。

    :param stage: 阶段名称，``Stage`` 枚举或大小写不敏感的字符串
        （COLLECTOR / COMPILER / TRAINER / VAULT 之一）。
    :param event: 事件类型，``Event`` 枚举或字符串（START / END）。
    :param stream: 目标输出流，默认 ``sys.stdout``。
    :returns: 实际打印的整行文本（不含末尾换行）。
    :raises ValueError: 当 ``stage`` 或 ``event`` 不在受支持取值内。
    """

    stage_name = _coerce_stage(stage)
    event_name = _coerce_event(event)
    line = f"[{now_timestamp()}] {stage_name} {event_name}"
    target = sys.stdout if stream is None else stream
    print(line, file=target, flush=True)
    return line


def print_summary(line: str, *, stream: TextIO | None = None) -> str:
    """在标准输出以单行形式打印一条摘要并返回它。

    用于过滤/采集/脱敏等阶段输出可被外部捕获的单行摘要。为保证"单行"
    约束，内部的换行符（``\\n`` / ``\\r``）会被替换为单个空格。

    :param line: 待打印的摘要文本。
    :param stream: 目标输出流，默认 ``sys.stdout``。
    :returns: 实际打印的单行文本（不含末尾换行）。
    """

    single_line = line.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    target = sys.stdout if stream is None else stream
    print(single_line, file=target, flush=True)
    return single_line
