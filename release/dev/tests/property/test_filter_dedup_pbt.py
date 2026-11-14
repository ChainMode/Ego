"""`ego_pod.compiler.filter_dedup` 过滤与去重的属性测试（hypothesis）。

本文件实现 design.md「Correctness Properties」中的 **Property 5 / 6 / 7**，
对应 tasks.md 任务 6.10 / 6.11 / 6.12，验证需求 R7（语料构建—过滤与增强）
前两条的核心不变量：

* **Property 5: filter_dedup 单调不增** —— 过滤与去重后的样本数
  ``total_after`` 恒小于等于输入样本数 ``total_before``；两阶段都只删除、
  不新增样本，故恒有
  ``total_before == total_after + dropped_short + merged_dup``。
  **Validates: Requirements 7.1, 7.2**
* **Property 6: filter_dedup 去重正确性** —— 去重后输出中任意两条样本的
  去重键互不相等；因相等的 ``(instruction, output)`` 二元组必产生相等的
  拼接哈希，故输出中不存在两条 ``(instruction, output)`` 完全相同的样本。
  **Validates: Requirements 7.2**
* **Property 7: filter_dedup 确定性** —— 对每组重复样本，保留的是**输入
  序列中首次出现**的那条；在相同输入上重复运行结果完全一致。
  **Validates: Requirements 7.2**

生成策略构造"含重复与过短样本"的混合候选列表：

* 元素同时覆盖 :class:`~ego_pod.compiler.classify.SampleCandidate` 对象与
  ``dict``（解析自 JSONL 的形态），R7-AC1 对两者一致地判定字段缺失/
  ``None``/非字符串为"非法"。
* instruction/output 取自一个小型文本池外加随机文本，使**重复**（同
  ``(instruction, output)``）与**过短**（strip 后拼接长度 < 8）样本自然
  高频出现，从而充分触发过滤与去重两条路径。
"""

from __future__ import annotations

import hypothesis.strategies as st
from hypothesis import given, settings

from ego_pod.compiler.classify import SampleCandidate
from ego_pod.compiler.filter_dedup import (
    dedup_key,
    extract_text_field,
    filter_and_dedup,
    is_valid_sample,
)

# ---------------------------------------------------------------------------
# 公共策略：易撞重复 / 易触发过短的字段文本 + 混合样本生成器
# ---------------------------------------------------------------------------

#: 五个合法 Data_Category（去重只依赖 instruction/output，category 仅作填充）。
_CATEGORIES = ["sense", "action", "thought", "skill", "credentials"]

#: 字段文本池：含长字符串（拼接后 >= 8，可被保留）、短字符串（< 8，被丢弃）、
#: 空串与纯空白（strip 后为空，必被丢弃），小池规模使重复样本高频出现。
_TEXT_POOL = [
    "instruction-A",
    "instruction-B",
    "output-A",
    "output-B",
    "你好世界",
    "abcdefgh",
    "x",
    "ab",
    "   ",
    "",
]

#: 合法字符串字段：池采样为主、辅以随机短文本，兼顾撞重与多样性。
_text_values = st.one_of(
    st.sampled_from(_TEXT_POOL),
    st.text(max_size=12),
)


@st.composite
def sample_candidates(draw: st.DrawFn) -> SampleCandidate:
    """生成字段均为字符串的 :class:`SampleCandidate`（可能过短，可能重复）。"""
    return SampleCandidate(
        instruction=draw(_text_values),
        input_field=draw(st.text(max_size=5)),
        output=draw(_text_values),
        category=draw(st.sampled_from(_CATEGORIES)),  # type: ignore[arg-type]
    )


@st.composite
def sample_dicts(draw: st.DrawFn) -> dict:
    """生成 JSONL 形态的 ``dict`` 样本，覆盖字段缺失/``None``/非字符串。

    R7-AC1 要求 ``instruction``/``output`` 缺失、为 ``None`` 或非字符串类型
    的样本一律被丢弃；本生成器以一定概率省略字段或填入 ``None``/整数，
    用以充分触发过短/非法过滤路径。
    """
    sample: dict = {"category": draw(st.sampled_from(_CATEGORIES))}
    # 各字段以布尔开关决定是否存在；存在时可为合法字符串、None 或整数。
    field_value = st.one_of(_text_values, st.none(), st.integers())
    if draw(st.booleans()):
        sample["instruction"] = draw(field_value)
    if draw(st.booleans()):
        sample["output"] = draw(field_value)
    return sample


#: 单个样本：SampleCandidate 对象或 dict，二者混合出现。
_any_sample = st.one_of(sample_candidates(), sample_dicts())

#: 候选样本列表（含空列表），规模适中以覆盖大量重复/过短组合。
sample_lists = st.lists(_any_sample, max_size=40)


# ---------------------------------------------------------------------------
# Property 5: filter_dedup 单调不增
# ---------------------------------------------------------------------------


@given(samples=sample_lists)
@settings(max_examples=300, deadline=None)
def test_filter_dedup_monotonicity(samples: list) -> None:
    """过滤与去重后样本数不增，且四项计数满足守恒等式。

    **Property 5: filter_dedup 单调不增**

    断言：

    * ``total_before == len(samples)``、``total_after == len(kept)``；
    * ``total_after <= total_before``（单调不增）；
    * ``total_before == total_after + dropped_short + merged_dup``（守恒）；
    * 各计数非负。

    **Validates: Requirements 7.1, 7.2**
    """
    kept, stats = filter_and_dedup(samples)

    assert stats.total_before == len(samples)
    assert stats.total_after == len(kept)
    assert stats.total_after <= stats.total_before
    assert stats.dropped_short >= 0
    assert stats.merged_dup >= 0
    assert (
        stats.total_before
        == stats.total_after + stats.dropped_short + stats.merged_dup
    )


# ---------------------------------------------------------------------------
# Property 6: filter_dedup 去重正确性
# ---------------------------------------------------------------------------


@given(samples=sample_lists)
@settings(max_examples=300, deadline=None)
def test_filter_dedup_uniqueness(samples: list) -> None:
    """去重后保留样本的去重键与 ``(instruction, output)`` 二元组全局唯一。

    **Property 6: filter_dedup 去重正确性**

    保留样本均已通过 R7-AC1 过滤，故其 ``instruction``/``output`` 必为字符串。
    相等的 ``(instruction, output)`` 必产生相等的 SHA-256 拼接哈希，因此
    "去重键两两不等"等价于"不存在两条 ``(instruction, output)`` 相同的样本"。

    **Validates: Requirements 7.2**
    """
    kept, _stats = filter_and_dedup(samples)

    keys = [dedup_key(s) for s in kept]
    assert len(keys) == len(set(keys)), "存在去重键重复的保留样本"

    pairs = [
        (extract_text_field(s, "instruction"), extract_text_field(s, "output"))
        for s in kept
    ]
    assert len(pairs) == len(set(pairs)), "存在 (instruction, output) 相同的保留样本"


# ---------------------------------------------------------------------------
# Property 7: filter_dedup 确定性
# ---------------------------------------------------------------------------


@given(samples=sample_lists)
@settings(max_examples=300, deadline=None)
def test_filter_dedup_determinism_first_occurrence(samples: list) -> None:
    """每组重复保留输入序列首次出现的那条，且重复运行结果一致。

    **Property 7: filter_dedup 确定性**

    通过独立复现实现语义（先按 R7-AC1 过滤、再对每个去重键保留首次出现）
    构造期望结果，并以**对象身份**（``is``）逐一核对，确保保留的正是输入中
    首次出现的对象而非任何后续值相等的副本。最后在相同输入上重复运行，断言
    保留对象序列与统计计数完全一致。

    **Validates: Requirements 7.2**
    """
    kept, stats = filter_and_dedup(samples)

    # 独立复现 filter_short → dedup 的"首次出现"语义。
    after_short = [s for s in samples if is_valid_sample(s)]
    seen: set[str] = set()
    expected_first: list = []
    for sample in after_short:
        key = dedup_key(sample)
        if key not in seen:
            seen.add(key)
            expected_first.append(sample)

    assert len(kept) == len(expected_first)
    for actual, expected in zip(kept, expected_first):
        assert actual is expected, "保留的不是输入序列中首次出现的样本对象"

    # 重复运行：相同输入产出相同对象序列与相同计数（确定性）。
    kept_again, stats_again = filter_and_dedup(samples)
    assert [id(x) for x in kept_again] == [id(x) for x in kept]
    assert stats_again == stats
