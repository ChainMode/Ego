"""`ego_pod.compiler.classify` 模板分类的属性测试（hypothesis）。

本文件实现 design.md「Correctness Properties」中的 **Property 8**，对应
tasks.md 任务 6.8：

* **Property 8: classify 类别透明传递** ——
  若输入样本的 Data_Category 属于五元集
  ``{sense, action, thought, skill, credentials}``，则被保留的输出样本
  （:class:`SampleCandidate`）的 ``category`` 字段严格等于该值的小写形式。

**Validates: Requirements 6.1, 6.2**

生成策略覆盖 Property 8 所约束的整个输入空间：

* **五个类别全覆盖** —— 从五元合法集采样，再对每个字符随机大小写翻转，
  构造任意大小写变体（如 ``Sense`` / ``SENSE`` / ``sEnSe`` / ``credentials``），
  以验证"小写化后透明传递"（R6-AC2）。
* **对话结构 / 非对话结构两条路径** —— 通过 ``is_conversation`` 开关分别
  生成"已含 instruction/output 对"的对话样本与"仅含 text 原文、需套模板"的
  非对话样本（R6-AC3 两个分支）。
* 字段长度从空串到中等长度不等，以同时覆盖"被保留"与"因空字段被丢弃"
  （R6-AC6）两种结局；对被丢弃样本，Property 8 的断言不被触发（真空成立）。
"""

from __future__ import annotations

import hypothesis.strategies as st
from hypothesis import given, settings

from ego_pod.compiler.classify import (
    VALID_CATEGORIES,
    SourceSample,
    classify_sample,
    classify_samples,
)

# ---------------------------------------------------------------------------
# 公共策略：五元合法类别 + 任意大小写变体 + 来源样本生成器
# ---------------------------------------------------------------------------

#: 五个合法 Data_Category 的规范（小写）形式。
_BASE_CATEGORIES = sorted(VALID_CATEGORIES)


@st.composite
def varied_case_categories(draw: st.DrawFn) -> str:
    """从五元合法集采样一个类别，并对每个字符随机翻转大小写。

    保证 ``result.lower()`` 仍属于五元合法集（前置条件），同时覆盖任意
    大小写组合，用于验证 R6-AC2 的"小写化后透明传递"。
    """
    base = draw(st.sampled_from(_BASE_CATEGORIES))
    flags = draw(
        st.lists(st.booleans(), min_size=len(base), max_size=len(base))
    )
    return "".join(c.upper() if flag else c.lower() for c, flag in zip(base, flags))


#: 任意字段文本：含空串，覆盖"保留"与"因空字段丢弃"两种结局。
_field_text = st.text(max_size=200)


@st.composite
def source_samples(draw: st.DrawFn) -> SourceSample:
    """生成来源样本：类别取自五元集（任意大小写），随机走对话/非对话路径。"""
    category = draw(varied_case_categories())
    input_field = draw(st.text(max_size=50))
    if draw(st.booleans()):
        # 已具备对话结构：instruction 与 output 均非 None，直接映射、不套模板。
        return SourceSample(
            category=category,
            instruction=draw(_field_text),
            output=draw(_field_text),
            input_field=input_field,
        )
    # 非对话结构：仅含 text 原文，分类阶段按类别套用模板。
    return SourceSample(
        category=category,
        text=draw(_field_text),
        input_field=input_field,
    )


# ---------------------------------------------------------------------------
# Property 8: classify 类别透明传递（单样本）
# ---------------------------------------------------------------------------


@given(sample=source_samples())
@settings(max_examples=400, deadline=None)
def test_classify_category_transparency(sample: SourceSample) -> None:
    """单条样本：被保留的候选样本 ``category`` 等于来源类别的小写形式。

    **Property 8: classify 类别透明传递**

    **Validates: Requirements 6.1, 6.2**
    """
    result = classify_sample(sample)
    if result.candidate is not None:
        assert result.candidate.category == sample.category.lower()
        # 透明传递的目标值必落在五元合法集内（R6-AC1）。
        assert result.candidate.category in VALID_CATEGORIES


# ---------------------------------------------------------------------------
# Property 8: classify 类别透明传递（批量 API classify_samples）
# ---------------------------------------------------------------------------


@given(samples=st.lists(source_samples(), max_size=30))
@settings(max_examples=200, deadline=None)
def test_classify_samples_category_transparency_batch(
    samples: list[SourceSample],
) -> None:
    """批量分类：每条被保留的候选样本 ``category`` 等于其来源类别的小写形式。

    ``classify_samples`` 按输入顺序保留未被丢弃的样本，因此可通过对每条来源
    样本单独执行 ``classify_sample`` 复原"来源 → 保留候选"的映射，逐一核对
    透明传递成立。

    **Property 8: classify 类别透明传递**

    **Validates: Requirements 6.1, 6.2**
    """
    kept, _stats = classify_samples(samples)

    # 复原保留样本与其来源类别的有序对应关系。
    expected_lower_categories = [
        sample.category.lower()
        for sample in samples
        if classify_sample(sample).candidate is not None
    ]

    assert len(kept) == len(expected_lower_categories)
    for candidate, expected_lower in zip(kept, expected_lower_categories):
        assert candidate.category == expected_lower
        assert candidate.category in VALID_CATEGORIES
