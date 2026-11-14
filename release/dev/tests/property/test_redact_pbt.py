"""`ego_pod.compiler.redact` 凭证脱敏的属性测试（hypothesis）。

对应设计文档 "Correctness Properties" 章节的 Property 1 与 Property 2，
验证需求 R5（语料构建—清洗与脱敏）的核心不变量：

- **Property 1: redact 幂等性** —— 对任意文本 t，连续两次脱敏与一次脱敏
  结果相同：``redact(redact(t)) == redact(t)``。
- **Property 2: redact 消除性** —— 脱敏后输出中任何 Credential_Token 模式
  （``sk-`` 前缀 / base64 / PEM 块）均不再匹配。

策略层面构造"任意文本中嵌入凭证 token"的输入：把 PEM 块、``sk-`` 密钥、
base64 串与任意文本片段交错拼接，覆盖 token 单独出现、相邻拼接、与噪声
文本混合等情形，并对全部五个 Data_Category 取值施加脱敏。
"""

from __future__ import annotations

import string

from hypothesis import given, settings
from hypothesis import strategies as st

from ego_pod.compiler.redact import (
    PATTERN_BASE64,
    PATTERN_PEM,
    PATTERN_SK,
    redact_text,
)

# ---------------------------------------------------------------------------
# 公共策略：Data_Category 与三类 Credential_Token 生成器
# ---------------------------------------------------------------------------

#: 五个合法 Data_Category（R5-AC6 要求全类别脱敏，不限于 credentials）。
CATEGORIES = st.sampled_from(
    ["sense", "action", "thought", "skill", "credentials"]
)

#: sk- 模式尾部允许的字符集 [A-Za-z0-9_-]。
_SK_TAIL_ALPHABET = string.ascii_letters + string.digits + "_-"

#: base64 模式允许的字符集 [A-Za-z0-9+/=]。
_BASE64_ALPHABET = string.ascii_letters + string.digits + "+/="

#: PEM 标签允许的字符集 [A-Z ]（至少含一个大写字母）。
_PEM_LABEL_ALPHABET = string.ascii_uppercase + " "

#: ``sk-`` 前缀后紧随 20–60 个 [A-Za-z0-9_-] 字符（满足 R5-AC2(a) 的 ≥20）。
sk_tokens = st.text(
    alphabet=_SK_TAIL_ALPHABET, min_size=20, max_size=60
).map(lambda tail: "sk-" + tail)

#: 长度 32–200 的连续 base64 串（落在 R5-AC2(b) 的 32–4096 区间内）。
base64_tokens = st.text(
    alphabet=_BASE64_ALPHABET, min_size=32, max_size=200
)


@st.composite
def pem_blocks(draw: st.DrawFn) -> str:
    """生成块内长度远低于 65536 上限的合法 PEM 块（R5-AC2(c)）。"""
    label = draw(
        st.text(alphabet=_PEM_LABEL_ALPHABET, min_size=1, max_size=20)
        # 至少包含一个大写字母，保证 [A-Z ]+ 标签非纯空白且非空
        .filter(lambda s: any(c.isalpha() for c in s))
    )
    body = draw(
        st.text(alphabet=_BASE64_ALPHABET + "\n", min_size=0, max_size=300)
    )
    return f"-----BEGIN {label}-----\n{body}\n-----END {label}-----"


#: 任意噪声文本片段（可能本身偶发含凭证样式字符，覆盖混合场景）。
arbitrary_text = st.text(max_size=60)

#: 单个片段：噪声文本或三类 token 之一。
_segment = st.one_of(arbitrary_text, sk_tokens, base64_tokens, pem_blocks())

#: 交错拼接 0–8 个片段，构造"任意文本中嵌入凭证 token"的复合输入。
embedded_text = st.lists(_segment, min_size=0, max_size=8).map("".join)


# ---------------------------------------------------------------------------
# Property 1: redact 幂等性
# ---------------------------------------------------------------------------


@settings(max_examples=300, deadline=None)
@given(text=embedded_text, category=CATEGORIES)
def test_redact_idempotent(text: str, category: str) -> None:
    """Property 1: redact 幂等性。

    对任意输入文本，连续两次调用脱敏与一次调用结果相同：
    ``redact_text(redact_text(t)[0], cat)[0] == redact_text(t, cat)[0]``。

    **Validates: Requirements 5.1, 5.2, 5.6**
    """
    once, _ = redact_text(text, category)
    twice, _ = redact_text(once, category)
    assert twice == once


# ---------------------------------------------------------------------------
# Property 2: redact 消除性
# ---------------------------------------------------------------------------


@settings(max_examples=300, deadline=None)
@given(text=embedded_text, category=CATEGORIES)
def test_redact_eliminates_all_patterns(text: str, category: str) -> None:
    """Property 2: redact 消除性。

    脱敏完成后，输出文本中三类 Credential_Token 模式（``sk-`` 前缀、base64、
    PEM 块）均不再可匹配。

    **Validates: Requirements 5.1, 5.2, 5.6**
    """
    out, _ = redact_text(text, category)
    assert PATTERN_PEM.search(out) is None, f"PEM 模式残留: {out!r}"
    assert PATTERN_SK.search(out) is None, f"sk- 模式残留: {out!r}"
    assert PATTERN_BASE64.search(out) is None, f"base64 模式残留: {out!r}"
