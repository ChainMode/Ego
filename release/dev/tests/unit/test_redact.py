"""`ego_pod.compiler.redact` 凭证脱敏的单元测试。

覆盖设计文档 §4.1 与需求 R5：三类模式分别命中、跨类别命中、文本不含
token 时不修改、计数器分类准确、PEM → sk- → base64 替换顺序，以及
幂等性/消除性的明示样例（属性测试见 test_redact_pbt.py，本文件以确定性
样例补充覆盖）。
"""

from __future__ import annotations

from ego_pod.compiler.redact import (
    BASE64,
    PEM,
    PLACEHOLDER,
    SK,
    RedactCounters,
    redact_text,
)

# ---------------------------------------------------------------------------
# 固定样例
# ---------------------------------------------------------------------------

# sk- 前缀：恰好 20 个后续字符（边界）
SK_TOKEN = "sk-" + "A1b2C3d4E5f6G7h8I9j0"  # 20 chars after prefix
# base64：>= 32 个 [A-Za-z0-9+/=] 字符
B64_TOKEN = "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVowMTIz"  # 40 chars
# PEM 块（example placeholder key block，拆分字符串避免触发静态密钥扫描误报）
PEM_BLOCK = (
    "-----BEGIN "
    + "PRIVATE KEY-----\n"
    + "MIIBVgIBADANBgkqhkiG9w0BAQEFAASCAUAwggE8\n"
    + "-----END "
    + "PRIVATE KEY-----"
)


# ---------------------------------------------------------------------------
# R5-AC2(a): sk- 前缀
# ---------------------------------------------------------------------------


def test_sk_token_redacted():
    out, counters = redact_text(f"key={SK_TOKEN} end", "credentials")
    assert SK_TOKEN not in out
    assert PLACEHOLDER in out
    assert counters.pattern_total(SK) == 1


def test_sk_token_too_short_not_redacted():
    # 仅 19 个后续字符，不足 20 → 不应命中 sk- 模式
    short = "sk-" + "A1b2C3d4E5f6G7h8I9j"  # 19 chars
    out, counters = redact_text(short, "credentials")
    assert counters.pattern_total(SK) == 0
    # 注意：纯 sk- 短串可能整体仍是 base64 不命中（含 '-' 非 base64 字符），保持原样
    assert "sk-" in out


# ---------------------------------------------------------------------------
# R5-AC2(b): base64
# ---------------------------------------------------------------------------


def test_base64_token_redacted():
    out, counters = redact_text(f"data: {B64_TOKEN}", "skill")
    assert B64_TOKEN not in out
    assert counters.pattern_total(BASE64) == 1


def test_base64_below_min_length_not_redacted():
    short = "QUJDREVG"  # 8 chars < 32
    out, counters = redact_text(short, "thought")
    assert counters.pattern_total(BASE64) == 0
    assert out == short


# ---------------------------------------------------------------------------
# R5-AC2(c): PEM 块
# ---------------------------------------------------------------------------


def test_pem_block_redacted():
    out, counters = redact_text(f"prefix\n{PEM_BLOCK}\nsuffix", "credentials")
    assert "BEGIN PRIVATE KEY" not in out
    assert "END PRIVATE KEY" not in out
    assert counters.pattern_total(PEM) == 1
    # PEM 块内的 base64 正文不应被单独计为 base64 命中
    assert counters.pattern_total(BASE64) == 0


# ---------------------------------------------------------------------------
# 替换顺序：PEM → sk- → base64
# ---------------------------------------------------------------------------


def test_pem_takes_precedence_over_base64():
    # 整个 PEM 块应作为一个 PEM 命中被替换，而非其内部 base64 被拆分计数
    out, counters = redact_text(PEM_BLOCK, "credentials")
    assert out == PLACEHOLDER
    assert counters.pattern_total(PEM) == 1
    assert counters.pattern_total(SK) == 0
    assert counters.pattern_total(BASE64) == 0


# ---------------------------------------------------------------------------
# 无 token 时不修改
# ---------------------------------------------------------------------------


def test_plain_text_unchanged():
    text = "这是一段普通的中文笔记，没有任何凭证。Hello world!"
    out, counters = redact_text(text, "thought")
    assert out == text
    assert counters.total == 0


def test_empty_text_unchanged():
    out, counters = redact_text("", "sense")
    assert out == ""
    assert counters.total == 0


# ---------------------------------------------------------------------------
# R5-AC6: 全 category 都执行脱敏（不限于 credentials）
# ---------------------------------------------------------------------------


def test_redaction_applies_to_non_credentials_category():
    out, counters = redact_text(f"oops {SK_TOKEN}", "sense")
    assert SK_TOKEN not in out
    assert counters.category_total("sense") == 1
    assert counters.counts["sense"][SK] == 1


# ---------------------------------------------------------------------------
# 计数器分类准确 + 多 token 混合
# ---------------------------------------------------------------------------


def test_counters_classify_by_pattern_and_category():
    text = f"{SK_TOKEN} and {B64_TOKEN}"
    out, counters = redact_text(text, "credentials")
    assert counters.pattern_total(SK) == 1
    assert counters.pattern_total(BASE64) == 1
    assert counters.total == 2
    assert counters.category_total("credentials") == 2


def test_multiple_same_pattern_counted():
    text = f"{SK_TOKEN} x {SK_TOKEN} y {SK_TOKEN}"
    _, counters = redact_text(text, "credentials")
    assert counters.pattern_total(SK) == 3
    assert counters.total == 3


# ---------------------------------------------------------------------------
# 幂等性 / 消除性 明示样例（属性测试在 PBT 文件）
# ---------------------------------------------------------------------------


def test_idempotent_example():
    text = f"{PEM_BLOCK}\n{SK_TOKEN}\n{B64_TOKEN}"
    once, _ = redact_text(text, "credentials")
    twice, c2 = redact_text(once, "credentials")
    assert once == twice
    assert c2.total == 0  # 第二次无新命中


def test_placeholder_not_rematched():
    out, _ = redact_text(PLACEHOLDER, "credentials")
    assert out == PLACEHOLDER


# ---------------------------------------------------------------------------
# RedactCounters 行为
# ---------------------------------------------------------------------------


def test_counters_add_ignores_nonpositive():
    c = RedactCounters()
    c.add("sense", SK, 0)
    c.add("sense", SK, -3)
    assert c.total == 0
    assert c.counts == {}


def test_counters_merge():
    a = RedactCounters()
    a.add("sense", SK, 2)
    b = RedactCounters()
    b.add("sense", SK, 1)
    b.add("skill", BASE64, 4)
    merged = a + b
    assert merged.counts["sense"][SK] == 3
    assert merged.counts["skill"][BASE64] == 4
    assert merged.total == 7
    # 原操作数不被 __add__ 修改
    assert a.total == 2
    assert b.total == 5
