"""过短过滤 + SHA-256 去重（design.md §4.4）。

本模块实现语料构建管线中"过滤与去重"子阶段，对应需求 R7 的前三条：

- **R7-AC1（过短/非法过滤）**：当一条候选样本的 ``instruction`` 或
  ``output`` 字段缺失、为 ``null``、或非字符串类型，或两字段**去除前后空白
  后拼接**的字符长度小于 :data:`MIN_CONTENT_LEN`（8）时，丢弃该样本。
- **R7-AC2（去重）**：基于 ``instruction`` 与 ``output`` 字段拼接后的
  ``sha256((instruction + output).encode("utf-8"))`` 哈希值去重，对每组重复
  样本仅保留**输入序列中最先出现**的一条。
- **R7-AC3（过滤摘要）**：在标准输出输出单行可被外部捕获的过滤摘要，至少
  包含 ``dropped_short`` / ``merged_dup`` / ``total_before`` / ``total_after``
  四项计数。

与 :mod:`ego_pod.compiler.redact`、:mod:`ego_pod.compiler.classify` 一致，本
模块的核心过滤/去重函数为**纯函数**：不直接写标准输出，而是把计数收集进
:class:`FilterStats`。摘要的标准输出动作通过 :func:`print_filter_summary`
（复用 :func:`ego_pod.logging_io.print_summary`）提供，由 Compiler
``pipeline.py`` 在适当时机调用，便于测试与外部捕获。

正确性不变量（design.md "Correctness Properties"）：

* **Property 5（单调不增）**：过滤与去重后的样本数 ``total_after`` 恒小于
  等于输入样本数 ``total_before``。两阶段都只删除、不新增样本，且恒有
  ``total_before == total_after + dropped_short + merged_dup``。
* **Property 6（去重正确性）**：:func:`dedup` 输出中任意两条样本的去重键
  互不相等；因相等的 ``(instruction, output)`` 二元组必产生相等的拼接哈希，
  故输出中不存在两条 ``(instruction, output)`` 完全相同的样本。
* **Property 7（确定性）**：对每组重复样本，:func:`dedup` 保留的是**输入
  序列中首次出现**的那条（后续出现者一律合并丢弃）。

字段提取对 :class:`ego_pod.compiler.classify.SampleCandidate` 实例与
``Mapping``（如解析自 JSON 的 ``dict``）两种形态都适用：前者读取属性、后者
读取键，缺失/``None``/非字符串值在 R7-AC1 下被一致地判为"非法"而丢弃。
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Final, Optional, TypeVar

from ego_pod.logging_io import print_summary

__all__ = [
    "MIN_CONTENT_LEN",
    "FilterStats",
    "extract_text_field",
    "is_valid_sample",
    "dedup_key",
    "filter_short",
    "dedup",
    "filter_and_dedup",
    "print_filter_summary",
]

#: 两字段去除前后空白后拼接的最小可接受字符长度（需求 R7-AC1）。
#: 拼接长度严格小于此值的样本视为过短并被丢弃。
MIN_CONTENT_LEN: Final[int] = 8

#: 过滤/去重保持样本对象类型不变，故以类型变量表达"输入即输出元素类型"。
SampleT = TypeVar("SampleT")


# ---------------------------------------------------------------------------
# 字段提取与有效性判定
# ---------------------------------------------------------------------------


def extract_text_field(sample: object, name: str) -> Optional[object]:
    """从样本中读取名为 ``name`` 的字段值。

    同时支持两种样本形态：

    * :class:`~ego_pod.compiler.classify.SampleCandidate` 等对象 —— 通过
      :func:`getattr` 读取属性，缺失时返回 ``None``。
    * ``Mapping``（如解析自 JSONL 的 ``dict``）—— 通过 :meth:`dict.get`
      读取键，缺失时返回 ``None``。

    返回值未做任何类型转换，由调用方按 R7-AC1 判定其是否为合法字符串。
    """
    if isinstance(sample, Mapping):
        return sample.get(name)
    return getattr(sample, name, None)


def is_valid_sample(sample: object) -> bool:
    """判定 ``sample`` 是否通过 R7-AC1 的过短/非法过滤。

    丢弃（返回 ``False``）的条件——满足任一即丢弃：

    * ``instruction`` 或 ``output`` 字段缺失、为 ``None``、或非 :class:`str`
      类型；
    * 两字段各自去除前后空白后拼接的字符长度小于 :data:`MIN_CONTENT_LEN`。

    仅当两字段均为字符串且拼接长度 ``>= MIN_CONTENT_LEN`` 时返回 ``True``。
    """
    instruction = extract_text_field(sample, "instruction")
    output = extract_text_field(sample, "output")
    if not isinstance(instruction, str) or not isinstance(output, str):
        return False
    return len(instruction.strip() + output.strip()) >= MIN_CONTENT_LEN


def dedup_key(sample: object) -> str:
    """计算 ``sample`` 的去重键（需求 R7-AC2）。

    键为 ``sha256((instruction + output).encode("utf-8"))`` 的十六进制摘要。
    拼接使用**未经 strip** 的原始字段值（与 R7-AC1 的过短判定所用的 strip
    后拼接相互独立）；非字符串/缺失值在到达本函数前应已被 :func:`filter_short`
    过滤，此处对残余的非字符串值按空串处理以保证函数总可计算。
    """
    instruction = extract_text_field(sample, "instruction")
    output = extract_text_field(sample, "output")
    instr_str = instruction if isinstance(instruction, str) else ""
    output_str = output if isinstance(output, str) else ""
    digest = hashlib.sha256((instr_str + output_str).encode("utf-8"))
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# 过滤与去重
# ---------------------------------------------------------------------------


def filter_short(samples: Iterable[SampleT]) -> tuple[list[SampleT], int]:
    """按 R7-AC1 丢弃过短/字段非法的样本。

    保留顺序与输入一致。返回 ``(保留样本列表, 被丢弃数量)``，被丢弃数量即
    过滤摘要中的 ``dropped_short``。
    """
    kept: list[SampleT] = []
    dropped_short = 0
    for sample in samples:
        if is_valid_sample(sample):
            kept.append(sample)
        else:
            dropped_short += 1
    return kept, dropped_short


def dedup(samples: Iterable[SampleT]) -> tuple[list[SampleT], int]:
    """按 R7-AC2 基于 SHA-256 拼接哈希去重，保留每组首次出现的样本。

    遍历输入序列，对每条样本计算 :func:`dedup_key`：键首次出现则保留，再次
    出现则计入合并数量并丢弃（Property 7 确定性）。保留样本之间去重键互不相
    等（Property 6 去重正确性）。返回 ``(保留样本列表, 合并数量)``，合并数量
    即过滤摘要中的 ``merged_dup``。
    """
    seen: set[str] = set()
    kept: list[SampleT] = []
    merged_dup = 0
    for sample in samples:
        key = dedup_key(sample)
        if key in seen:
            merged_dup += 1
            continue
        seen.add(key)
        kept.append(sample)
    return kept, merged_dup


# ---------------------------------------------------------------------------
# 过滤摘要
# ---------------------------------------------------------------------------


@dataclass
class FilterStats:
    """过滤阶段的四项计数（需求 R7-AC3）。

    满足不变量 ``total_before == total_after + dropped_short + merged_dup``
    （:func:`filter_and_dedup` 构造时即如此）。
    """

    #: 因过短/字段非法（R7-AC1）被丢弃的样本数。
    dropped_short: int = 0
    #: 因重复（R7-AC2）被合并丢弃的样本数。
    merged_dup: int = 0
    #: 进入过滤阶段前的样本总数。
    total_before: int = 0
    #: 过滤与去重后的样本总数。
    total_after: int = 0

    def summary_line(self) -> str:
        """构造单行过滤摘要文本（不含末尾换行）。

        形如 ``filter summary: dropped_short=2 merged_dup=1 total_before=10
        total_after=7``，四项计数以稳定的 ``key=value`` 形式给出，便于外部
        捕获与解析。
        """
        return (
            "filter summary: "
            f"dropped_short={self.dropped_short} "
            f"merged_dup={self.merged_dup} "
            f"total_before={self.total_before} "
            f"total_after={self.total_after}"
        )


def filter_and_dedup(
    samples: Iterable[SampleT],
) -> tuple[list[SampleT], FilterStats]:
    """对候选样本依次执行过短过滤与去重，返回 ``(保留样本, 过滤统计)``。

    顺序为 :func:`filter_short` → :func:`dedup`：先按 R7-AC1 丢弃过短/非法
    样本，再对其余样本按 R7-AC2 去重。保留样本顺序与输入一致。

    返回的 :class:`FilterStats` 满足 Property 5（``total_after <=
    total_before``）与不变量 ``total_before == total_after + dropped_short +
    merged_dup``。
    """
    materialized = list(samples)
    total_before = len(materialized)
    after_short, dropped_short = filter_short(materialized)
    deduped, merged_dup = dedup(after_short)
    stats = FilterStats(
        dropped_short=dropped_short,
        merged_dup=merged_dup,
        total_before=total_before,
        total_after=len(deduped),
    )
    return deduped, stats


def print_filter_summary(stats: FilterStats, *, stream=None) -> str:
    """将过滤摘要以单行形式打印到标准输出并返回该行（需求 R7-AC3）。

    复用 :func:`ego_pod.logging_io.print_summary` 保证"单行"约束与可被外部
    捕获。``stream`` 默认为 ``sys.stdout``，测试可注入其它流。
    """
    return print_summary(stats.summary_line(), stream=stream)
