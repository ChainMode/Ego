"""`ego_pod.compiler.augment` 模板增强的属性测试（hypothesis）。

本文件实现 design.md「Correctness Properties」中的 **Property 9** 与
**Property 10**，对应 tasks.md 任务 6.14 与 6.15：

* **Property 9: augment 不必要不增强**（任务 6.14）——
  若过滤后样本数 ``n >= 16``，:func:`augment_samples` 不产生任何新样本，
  输出严格等于输入（无增强样本、计数器 ``augmented_count == 0``）。

  **Validates: Requirements 7.4**

* **Property 10: augment 来源可识别**（任务 6.15）——
  增强阶段产生的每条样本，其 ``instruction`` 字段以固定前缀 ``[AUG]``
  开头；任何原始样本的 ``instruction`` 都不以 ``[AUG]`` 开头。

  **Validates: Requirements 7.5**

生成策略：

* 用 :func:`original_candidates` 生成"过滤后原始样本"
  （:class:`SampleCandidate`，``augmented=False``），其 ``instruction`` 受约束
  **不**以 ``[AUG]`` 前缀开头——这正是分类阶段模板的实际行为，也是 Property 10
  后半句"原始样本不以 ``[AUG]`` 开头"的前提。
* **Property 9** 生成大小 ``>= 16`` 的列表，覆盖"无需增强"分支。
* **Property 10** 生成大小 ``1..20`` 的列表，同时覆盖 ``0 < n < 16``（实际增强）
  与 ``n >= 16``（no-op）两类情形。其中 ``n in {1, 2}`` 时 ``6n < 16``，对每条
  样本套用全部 5 条模板后仍无法补足至 16，:func:`augment_samples` 按 R7-AC7
  抛出 :class:`CompilerError`；此为预期路径，仅在成功增强时断言前缀属性。
"""

from __future__ import annotations

import hypothesis.strategies as st
from hypothesis import given, settings

from ego_pod.compiler.augment import (
    AUGMENT_PREFIX,
    TARGET_SAMPLE_COUNT,
    augment_samples,
    is_augmented_instruction,
)
from ego_pod.compiler.classify import SampleCandidate
from ego_pod.exit_codes import CompilerError

# ---------------------------------------------------------------------------
# 公共策略：过滤后原始样本（augmented=False，instruction 不以 [AUG] 开头）
# ---------------------------------------------------------------------------

#: 五个合法 Data_Category 的规范（小写）形式。
_CATEGORIES = ["sense", "action", "thought", "skill", "credentials"]


@st.composite
def original_candidates(draw: st.DrawFn) -> SampleCandidate:
    """生成一条"过滤后原始样本"。

    模拟分类阶段（:mod:`ego_pod.compiler.classify`）的产出：``augmented=False``，
    且 ``instruction`` **不**以 ``[AUG]`` 前缀开头（任何分类模板都不以此起始）。
    其余字段任意，以覆盖 Property 9 / Property 10 的整个输入空间。
    """
    instruction = draw(
        st.text(min_size=1, max_size=80).filter(
            lambda s: not s.startswith(AUGMENT_PREFIX)
        )
    )
    output = draw(st.text(min_size=1, max_size=80))
    input_field = draw(st.text(max_size=40))
    category = draw(st.sampled_from(_CATEGORIES))
    return SampleCandidate(
        instruction=instruction,
        input_field=input_field,
        output=output,
        category=category,
        augmented=False,
    )


# ---------------------------------------------------------------------------
# Property 9: augment 不必要不增强（任务 6.14）
# ---------------------------------------------------------------------------


@given(samples=st.lists(original_candidates(), min_size=TARGET_SAMPLE_COUNT, max_size=40))
@settings(max_examples=300, deadline=None)
def test_augment_unnecessary_is_noop(samples: list[SampleCandidate]) -> None:
    """``n >= 16`` 时增强为 no-op：输出等于输入，且无任何增强样本。

    **Property 9: augment 不必要不增强**

    **Validates: Requirements 7.4**
    """
    result, stats = augment_samples(samples)

    # 输出严格等于输入（顺序与内容均不变）。
    assert result == list(samples)
    # 未产生任何增强样本。
    assert stats.augmented_count == 0
    assert all(not candidate.augmented for candidate in result)
    assert all(
        not is_augmented_instruction(candidate.instruction) for candidate in result
    )
    # 计数器自洽。
    assert stats.original_count == len(samples)
    assert stats.final_count == len(samples)


# ---------------------------------------------------------------------------
# Property 10: augment 来源可识别（任务 6.15）
# ---------------------------------------------------------------------------


@given(samples=st.lists(original_candidates(), min_size=1, max_size=20))
@settings(max_examples=400, deadline=None)
def test_augment_identifiable(samples: list[SampleCandidate]) -> None:
    """增强样本均以 ``[AUG]`` 开头；原始样本均不以 ``[AUG]`` 开头。

    对 ``n in {1, 2}`` 的列表，``6n < 16`` 无法补足至 16，:func:`augment_samples`
    按 R7-AC7 抛 :class:`CompilerError`——此为预期路径，仅在成功增强时断言前缀
    属性。

    **Property 10: augment 来源可识别**

    **Validates: Requirements 7.5**
    """
    n = len(samples)
    try:
        result, _stats = augment_samples(samples)
    except CompilerError:
        # n 太小（如 1、2）时，对每条样本套用全部模板后仍 < 16（R7-AC7），
        # augment_samples 抛 CompilerError。此为预期路径，不断言前缀属性。
        return

    # 前 n 条为原始样本，原样保留，且都不以 [AUG] 开头（Property 10 后半句）。
    assert result[:n] == list(samples)
    for original in result[:n]:
        assert not original.augmented
        assert not original.instruction.startswith(AUGMENT_PREFIX)
        assert not is_augmented_instruction(original.instruction)

    # 其余为增强样本，每条都以 [AUG] 开头（Property 10 前半句）。
    for augmented in result[n:]:
        assert augmented.augmented
        assert augmented.instruction.startswith(AUGMENT_PREFIX)
        assert is_augmented_instruction(augmented.instruction)

    # 两种识别方式（前缀 / augmented 标记）在全体样本上严格一致。
    for candidate in result:
        assert candidate.instruction.startswith(AUGMENT_PREFIX) == candidate.augmented
