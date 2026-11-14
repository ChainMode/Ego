"""凭证脱敏（Credential_Token 模式）。

本模块实现设计文档 §4.1 描述的脱敏阶段，对应需求 R5：

- **R5-AC1**：将所有匹配 R5-AC2 敏感模式的 Credential_Token 替换为
  字符串 :data:`PLACEHOLDER`（``[REDACTED]``）。
- **R5-AC2**：至少识别三类 Credential_Token——
  (a) ``sk-`` 前缀后紧随至少 20 个 ``[A-Za-z0-9_-]`` 字符；
  (b) 长度 32–4096、由 ``[A-Za-z0-9+/=]`` 组成的连续 base64 串；
  (c) 以 ``-----BEGIN [A-Z ]+-----`` 起始、``-----END [A-Z ]+-----``
  结束、块内长度不超过 65536 字符的 PEM 块。
- **R5-AC5 / R5-AC6**：按命中模式（``sk-`` 前缀 / base64 / PEM 块）及来源
  Data_Category 分类计数，供脱敏摘要使用；脱敏对**全部** Data_Category 执行
  （不限于 Credentials）。

**替换顺序为 PEM → sk- → base64**（设计文档 §4.1）：先替换 PEM 块可避免
base64 模式吃掉 PEM 块内部的 base64 正文或 ``sk-`` 子串。替换写入的
``[REDACTED]`` 占位串本身不会再匹配任何一类模式（``[`` / ``]`` 不属于
base64 字符集，占位串内不含 ``sk-`` 前缀，也不含 PEM 边界标记），因此
脱敏满足幂等性（Property 1）与消除性（Property 2）两条不变量。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Final

#: 命中的 Credential_Token 一律替换为此占位串（需求 R5-AC1）。
PLACEHOLDER: Final[str] = "[REDACTED]"

# ---------------------------------------------------------------------------
# 模式名称常量（计数器与摘要的稳定键）
# ---------------------------------------------------------------------------

#: PEM 块模式名。
PEM: Final[str] = "pem"
#: ``sk-`` 前缀 API 密钥模式名。
SK: Final[str] = "sk"
#: base64 连续串模式名。
BASE64: Final[str] = "base64"

#: 三类模式名称按应用顺序排列，供摘要按固定顺序枚举。
PATTERN_NAMES: Final[tuple[str, str, str]] = (PEM, SK, BASE64)

# ---------------------------------------------------------------------------
# 预编译正则（设计文档 §4.1）
# ---------------------------------------------------------------------------

#: PEM 块：``-----BEGIN <TYPE>-----`` 起始、``-----END <TYPE>-----`` 结束，
#: 块内（含换行等任意字符）长度不超过 65536；非贪婪匹配以避免跨块吞并。
PATTERN_PEM: Final[re.Pattern[str]] = re.compile(
    r"-----BEGIN [A-Z ]+-----[\s\S]{0,65536}?-----END [A-Z ]+-----"
)

#: ``sk-`` 前缀后紧随至少 20 个 ``[A-Za-z0-9_-]`` 字符。
PATTERN_SK: Final[re.Pattern[str]] = re.compile(r"sk-[A-Za-z0-9_-]{20,}")

#: 由 ``[A-Za-z0-9+/=]`` 组成、长度 32–4096 的连续 base64 串。
PATTERN_BASE64: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9+/=]{32,4096}")

#: 模式名 → 预编译正则，按 PEM → sk- → base64 的应用顺序排列。
#: 该顺序对脱敏正确性至关重要：PEM 先行避免其内部 base64/``sk-`` 被错误地
#: 单独计数或部分替换。
PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    (PEM, PATTERN_PEM),
    (SK, PATTERN_SK),
    (BASE64, PATTERN_BASE64),
)


# ---------------------------------------------------------------------------
# 计数器
# ---------------------------------------------------------------------------


@dataclass
class RedactCounters:
    """脱敏替换计数，按来源 Data_Category 与命中模式两维分类（需求 R5-AC5/AC6）。

    内部以嵌套字典 ``counts[category][pattern] = n`` 存储；仅记录非零计数以
    保持稀疏。提供按模式、按类别、以及全局总数的便捷聚合方法，并支持跨多次
    :func:`redact_text` 调用合并（Compiler pipeline 按 category 逐批脱敏后汇总）。
    """

    #: ``category -> {pattern_name: count}``，仅含非零项。
    counts: dict[str, dict[str, int]] = field(default_factory=dict)

    def add(self, category: str, pattern: str, n: int) -> None:
        """为 ``(category, pattern)`` 累加 ``n`` 次替换；``n <= 0`` 时不记录。"""
        if n <= 0:
            return
        bucket = self.counts.setdefault(category, {})
        bucket[pattern] = bucket.get(pattern, 0) + n

    @property
    def total(self) -> int:
        """所有类别、所有模式的替换总次数（R5-AC5 的总计数项）。"""
        return sum(n for bucket in self.counts.values() for n in bucket.values())

    def pattern_total(self, pattern: str) -> int:
        """指定模式在所有类别上的替换次数之和（R5-AC5 的按模式子项）。"""
        return sum(bucket.get(pattern, 0) for bucket in self.counts.values())

    def category_total(self, category: str) -> int:
        """指定类别在所有模式上的替换次数之和。"""
        return sum(self.counts.get(category, {}).values())

    def merge(self, other: "RedactCounters") -> "RedactCounters":
        """就地并入另一计数器并返回自身，便于 pipeline 链式汇总。"""
        for category, bucket in other.counts.items():
            for pattern, n in bucket.items():
                self.add(category, pattern, n)
        return self

    def __add__(self, other: "RedactCounters") -> "RedactCounters":
        """返回合并后的新计数器，不修改任一操作数。"""
        combined = RedactCounters()
        combined.merge(self)
        combined.merge(other)
        return combined


# ---------------------------------------------------------------------------
# 脱敏主函数
# ---------------------------------------------------------------------------


def redact_text(text: str, category: str) -> tuple[str, RedactCounters]:
    """对 ``text`` 执行凭证脱敏，返回 ``(脱敏后文本, 计数器)``。

    按 :data:`PATTERNS` 中固定的 **PEM → sk- → base64** 顺序逐模式替换为
    :data:`PLACEHOLDER`，并将每类命中次数记入计数器（归属到来源
    ``category``）。对任意 ``category`` 一律执行，满足 R5-AC6 对 Credentials
    以外类别同样脱敏的要求。

    :param text: 待脱敏的原始文本。
    :param category: 该文本来源的 Data_Category 名称（用于计数归类）。
    :returns: ``(脱敏后文本, RedactCounters)`` 二元组。
    """
    counters = RedactCounters()
    result = text
    for name, pattern in PATTERNS:
        result, n = pattern.subn(PLACEHOLDER, result)
        counters.add(category, name, n)
    return result, counters


__all__ = [
    "PLACEHOLDER",
    "PEM",
    "SK",
    "BASE64",
    "PATTERN_NAMES",
    "PATTERN_PEM",
    "PATTERN_SK",
    "PATTERN_BASE64",
    "PATTERNS",
    "RedactCounters",
    "redact_text",
]
