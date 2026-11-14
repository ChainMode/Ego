"""`ego_pod.compiler.normalize` 文本规范化的属性测试（hypothesis）。

本文件实现 design.md「Correctness Properties」中与 normalize 相关的两条
具名属性，对应 tasks.md 任务 6.5 与 6.6：

* **Property 3: normalize 幂等性** ——
  ``normalize_text(normalize_text(t)) == normalize_text(t)``。
* **Property 4: normalize 保留性** ——
  输入文本中存在的 ``\\n``、``\\t`` 字符必出现在输出中（位置可变，但字符不丢失）。

生成策略刻意混入控制字符（Unicode ``Cc``）、各类空白字符、连续空白游程
（≥3）以及 ``\\n`` / ``\\t``，以充分覆盖 R5-AC3（控制字符移除）与
R5-AC4（连续空白压缩）所定义的输入空间。
"""

from __future__ import annotations

import hypothesis.strategies as st
from hypothesis import given, settings

from ego_pod.compiler.normalize import normalize_text

# ---------------------------------------------------------------------------
# 字符集合 — 覆盖控制字符、空白字符与常规字符
# ---------------------------------------------------------------------------

# Unicode Cc 类别控制字符（U+0000–U+001F 与 U+007F–U+009F），其中包含
# \t (U+0009) 与 \n (U+000A)，也包含 VT/FF/CR/NEL 等本身属于 White_Space 的
# 控制字符 —— 这些应在 remove_control_chars 阶段被删除。
_CONTROL_CHARS = "".join(
    chr(c) for c in list(range(0x00, 0x20)) + list(range(0x7F, 0xA0))
)

# 非 \n/\t 的各类空白字符（含 Zs/Zl/Zp 与若干 Cc 空白），用于触发 R5-AC4 压缩。
_WHITESPACE_CHARS = (
    " "        # SPACE (U+0020)
    "\u000b"   # VT (Cc)
    "\u000c"   # FF (Cc)
    "\u000d"   # CR (Cc)
    "\u0085"   # NEL (Cc)
    "\u00a0"   # NO-BREAK SPACE
    "\u1680"   # OGHAM SPACE MARK
    "\u2003"   # EM SPACE
    "\u2009"   # THIN SPACE
    "\u2028"   # LINE SEPARATOR
    "\u2029"   # PARAGRAPH SEPARATOR
    "\u202f"   # NARROW NO-BREAK SPACE
    "\u205f"   # MEDIUM MATHEMATICAL SPACE
    "\u3000"   # IDEOGRAPHIC SPACE
)

# 常规可见字符（ASCII 字母数字、标点、CJK、非 ASCII 字母）。
_REGULAR_CHARS = "abZ09!?中文λé"

# 被显式保留的两个字符。
_KEEP_CHARS = "\n\t"


# ---------------------------------------------------------------------------
# 生成策略
# ---------------------------------------------------------------------------

# 策略 A：从精心挑选的字符集合中逐字符采样（含 \n / \t / Cc / 空白 / 常规）。
_interesting_alphabet = st.sampled_from(
    _CONTROL_CHARS + _WHITESPACE_CHARS + _REGULAR_CHARS + _KEEP_CHARS
)
_curated_text = st.text(alphabet=_interesting_alphabet, max_size=200)

# 策略 B：从完整 Unicode 空间采样，确保覆盖任意字符。
_broad_text = st.text(max_size=200)


@st.composite
def _chunked_text(draw: st.DrawFn) -> str:
    """构造由「常规片段 / 空白游程 / 控制字符游程 / \\n / \\t」拼接的文本。

    通过显式生成长度 ≥3 的空白游程，确保 R5-AC4 的压缩分支被频繁触发；
    并保证 \\n / \\t 以独立 token 形式出现，覆盖 Property 4 的保留性场景。
    """

    chunks = draw(
        st.lists(
            st.one_of(
                st.text(alphabet=_REGULAR_CHARS, max_size=6),
                # 长度 ≥3 的非 \n/\t 空白游程，触发压缩。
                st.text(alphabet=_WHITESPACE_CHARS, min_size=3, max_size=8),
                st.text(alphabet=_CONTROL_CHARS, min_size=1, max_size=4),
                st.just("\n"),
                st.just("\t"),
            ),
            max_size=20,
        )
    )
    return "".join(chunks)


# 汇总三种策略，覆盖面更广。
text_strategy = st.one_of(_curated_text, _broad_text, _chunked_text())


# ---------------------------------------------------------------------------
# Property 3: normalize 幂等性
# ---------------------------------------------------------------------------


@given(text=text_strategy)
@settings(max_examples=300)
def test_normalize_idempotent(text: str) -> None:
    """连续两次规范化与一次规范化结果相同。

    **Property 3: normalize 幂等性**

    **Validates: Requirements 5.3, 5.4**
    """

    once = normalize_text(text)
    twice = normalize_text(once)
    assert once == twice


# ---------------------------------------------------------------------------
# Property 4: normalize 保留性
# ---------------------------------------------------------------------------


@given(text=text_strategy)
@settings(max_examples=300)
def test_normalize_preserves_newline_and_tab(text: str) -> None:
    """输入中存在的 ``\\n`` / ``\\t`` 字符必出现在输出中（字符不丢失）。

    **Property 4: normalize 保留性**

    **Validates: Requirements 5.3, 5.4**
    """

    out = normalize_text(text)
    if "\n" in text:
        assert "\n" in out
    if "\t" in text:
        assert "\t" in out
