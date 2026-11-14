"""Compiler 文本规范化 — 控制字符移除与空白压缩。

本模块实现语料构建管线中"标准化"子阶段的两步纯函数变换，对应需求：

* **R5-AC3**：移除控制字符（Unicode 类别 ``Cc``），但保留 ``\\n`` 与 ``\\t``，
  随后再进入下一阶段。
* **R5-AC4**：在保留 ``\\n`` 与 ``\\t`` 不变的前提下，将连续 3 个及以上的
  *其他* 空白字符（Unicode 属性 ``White_Space=yes`` 中除 ``\\n`` 和 ``\\t``
  之外的字符）压缩为单个空格（U+0020）。

两步按设计文档 §4.2 的固定顺序执行：先移除 ``Cc``，再压缩空白。由于
``Cc`` 类别同时覆盖了 VT(U+000B)、FF(U+000C)、CR(U+000D)、NEL(U+0085)
等本身属于 ``White_Space`` 的控制字符，这些字符会在第一步被整体删除，
而不会进入第二步的压缩逻辑——这与 R5-AC3 "移除控制字符后再进入下一阶段"
的语义一致。

本模块保证两条关键不变量（见 design.md Correctness Properties）：

* **Property 3（幂等性）**：``normalize_text(normalize_text(t)) ==
  normalize_text(t)``。
* **Property 4（保留性）**：输入中存在的 ``\\n`` / ``\\t`` 字符不会在输出中
  丢失。
"""

from __future__ import annotations

import re
import unicodedata

__all__ = [
    "remove_control_chars",
    "collapse_whitespace",
    "normalize_text",
]

# 被保留的两个控制字符：换行与制表。
_KEEP = ("\n", "\t")

# Unicode 属性 White_Space=yes 中、除 \n(U+000A) 与 \t(U+0009) 之外的字符集合。
#
# 其中 VT(U+000B)、FF(U+000C)、CR(U+000D)、NEL(U+0085) 属于 Cc 类别，会在
# remove_control_chars 阶段被先行删除，因此实际进入压缩步骤的只有 Zs/Zl/Zp
# 类空白；仍将其纳入字符类，可使 collapse_whitespace 在被单独调用时也忠实
# 反映"所有非 \n/\t 空白"这一定义。
_OTHER_WHITESPACE = (
    "\u000b\u000c\u000d"  # VT, FF, CR (Cc)
    "\u0020"              # SPACE
    "\u0085"              # NEL (Cc)
    "\u00a0"              # NO-BREAK SPACE
    "\u1680"              # OGHAM SPACE MARK
    "\u2000\u2001\u2002\u2003\u2004\u2005\u2006"
    "\u2007\u2008\u2009\u200a"  # EN QUAD .. HAIR SPACE
    "\u2028"              # LINE SEPARATOR
    "\u2029"              # PARAGRAPH SEPARATOR
    "\u202f"              # NARROW NO-BREAK SPACE
    "\u205f"              # MEDIUM MATHEMATICAL SPACE
    "\u3000"              # IDEOGRAPHIC SPACE
)

# 匹配 3 个及以上连续的"其他空白"字符；replace 目标为单个 U+0020。
# 使用 re.escape 逐字符转义以构造安全的字符类。
RE_SPACES = re.compile("[" + "".join(re.escape(c) for c in _OTHER_WHITESPACE) + "]{3,}")


def remove_control_chars(text: str) -> str:
    """移除 Unicode ``Cc`` 类别控制字符，但保留 ``\\n`` 与 ``\\t``。

    对应需求 R5-AC3。``Cc`` 类别覆盖 U+0000–U+001F 与 U+007F–U+009F，
    其中 ``\\t`` (U+0009) 与 ``\\n`` (U+000A) 被显式保留。

    Args:
        text: 待清洗的原始文本。

    Returns:
        移除控制字符后的文本（``\\n`` / ``\\t`` 原样保留）。
    """

    return "".join(
        c for c in text if c in _KEEP or unicodedata.category(c) != "Cc"
    )


def collapse_whitespace(text: str) -> str:
    """将连续 ≥3 个非 ``\\n``/``\\t`` 空白字符压缩为单个空格 U+0020。

    对应需求 R5-AC4。``\\n`` 与 ``\\t`` 不参与压缩、原样保留；连续长度为
    1 或 2 的空白串不被压缩。

    Args:
        text: 待压缩空白的文本。

    Returns:
        压缩后的文本。
    """

    return RE_SPACES.sub(" ", text)


def normalize_text(text: str) -> str:
    """对文本执行规范化：先移除控制字符，再压缩连续空白。

    步骤（design.md §4.2 固定顺序）：

    1. 移除 Unicode ``Cc`` 控制字符，但保留 ``\\n`` 与 ``\\t``（R5-AC3）。
    2. 将连续 ≥3 个非 ``\\n``/``\\t`` 空白字符压缩为单个空格 U+0020（R5-AC4）。

    该函数满足幂等性（Property 3）与 ``\\n``/``\\t`` 保留性（Property 4）。

    Args:
        text: 待规范化的文本。

    Returns:
        规范化后的文本。
    """

    cleaned = remove_control_chars(text)
    return collapse_whitespace(cleaned)
