"""Compiler 边界与丢弃路径单元测试（tasks.md 任务 6.18）。

本文件以确定性样例补充覆盖 Compiler 三个子模块的边界与错误分支：

* :mod:`ego_pod.compiler.classify` —— 三条丢弃路径：
  - **R6-AC6** instruction/output 任一字段为空 → ``EMPTY_FIELD``；
  - **R6-AC7** 非对话结构样本的 Data_Category 在模板表中缺失 → ``MISSING_TEMPLATE``；
  - **R6-AC8** category 取值非法（不属五元集）→ ``ILLEGAL_CATEGORY``。
* :mod:`ego_pod.compiler.filter_dedup` —— ``filter_short`` 长度边界
  （拼接长度 7 丢弃 / 8 保留 / 9 保留，**R7-AC1**）与字段缺失/null/非字符串丢弃。
* :mod:`ego_pod.compiler.augment` —— ``augment_samples`` 在
  ``n = 0 / 1 / 15 / 16 / 超过 16`` 五个分支的行为（**R7-AC4**），以及
  模板用尽仍无法补足至 16 时抛 :class:`CompilerError`（**R7-AC7**）。

_Requirements: 6.6, 6.7, 6.8, 7.1, 7.4, 7.6, 7.7_

属性测试（Property 8/9/10/11）见 ``tests/property/`` 下对应文件，本文件聚焦
具体边界数值与三条丢弃路径的明示样例。
"""

from __future__ import annotations

import pytest

from ego_pod.compiler.augment import (
    AUGMENT_PREFIX,
    AUGMENT_TEMPLATES,
    TARGET_SAMPLE_COUNT,
    augment_samples,
)
from ego_pod.compiler.classify import (
    TEMPLATES,
    DiscardReason,
    SampleCandidate,
    SourceSample,
    classify_sample,
    classify_samples,
)
from ego_pod.compiler.filter_dedup import (
    MIN_CONTENT_LEN,
    filter_short,
    is_valid_sample,
)
from ego_pod.exit_codes import EXIT_RUNTIME_FAILURE, CompilerError


# ---------------------------------------------------------------------------
# 辅助构造
# ---------------------------------------------------------------------------


def _candidate(instruction: str, output: str, category: str = "thought") -> SampleCandidate:
    """构造一条原始（非增强）候选样本。"""
    return SampleCandidate(
        instruction=instruction,
        input_field="",
        output=output,
        category=category,  # type: ignore[arg-type]
    )


# ===========================================================================
# classify 丢弃路径一：R6-AC6 任一字段为空 → EMPTY_FIELD
# ===========================================================================


def test_classify_discard_empty_instruction_conversation():
    """对话结构样本 instruction 为空 → 按 R6-AC6 丢弃。"""
    sample = SourceSample(category="thought", instruction="", output="非空输出")
    result = classify_sample(sample)
    assert result.discarded is True
    assert result.candidate is None
    assert result.reason is DiscardReason.EMPTY_FIELD


def test_classify_discard_empty_output_conversation():
    """对话结构样本 output 为空 → 按 R6-AC6 丢弃。"""
    sample = SourceSample(category="thought", instruction="非空指令", output="")
    result = classify_sample(sample)
    assert result.discarded is True
    assert result.reason is DiscardReason.EMPTY_FIELD


def test_classify_discard_both_fields_empty_conversation():
    """对话结构样本两字段同时为空 → 按 R6-AC6 丢弃。"""
    sample = SourceSample(category="thought", instruction="", output="")
    result = classify_sample(sample)
    assert result.discarded is True
    assert result.reason is DiscardReason.EMPTY_FIELD


def test_classify_discard_empty_text_non_conversation():
    """非对话结构样本原文为空 → 模板渲染后 output 为空 → 按 R6-AC6 丢弃。"""
    sample = SourceSample(category="sense", text="")
    result = classify_sample(sample)
    assert result.discarded is True
    assert result.reason is DiscardReason.EMPTY_FIELD


def test_classify_keeps_non_empty_conversation():
    """两字段均非空的对话结构样本应被保留（对照组）。"""
    sample = SourceSample(category="skill", instruction="问", output="答")
    result = classify_sample(sample)
    assert result.discarded is False
    assert result.candidate is not None
    assert result.candidate.category == "skill"


# ===========================================================================
# classify 丢弃路径二：R6-AC7 非对话结构样本模板缺失 → MISSING_TEMPLATE
# ===========================================================================


def test_classify_discard_missing_template(monkeypatch):
    """合法类别但模板表中缺失对应条目的非对话结构样本 → 按 R6-AC7 丢弃。

    默认 ``TEMPLATES`` 五类齐全，故移除 ``thought`` 条目以触发防御性的
    "模板缺失"分支（该路径仅由"对应 Data_Category 在模板表中无条目"触发）。
    """
    monkeypatch.delitem(TEMPLATES, "thought")
    sample = SourceSample(category="thought", text="一些待整理的笔记内容")
    result = classify_sample(sample)
    assert result.discarded is True
    assert result.candidate is None
    assert result.reason is DiscardReason.MISSING_TEMPLATE
    # detail 记录小写化后的类别名。
    assert result.detail == "thought"


def test_classify_missing_template_not_triggered_for_conversation(monkeypatch):
    """模板缺失时，已具备对话结构的样本不查模板、不因 R6-AC7 被丢弃。"""
    monkeypatch.delitem(TEMPLATES, "thought")
    sample = SourceSample(category="thought", instruction="问题", output="回答")
    result = classify_sample(sample)
    assert result.discarded is False
    assert result.candidate is not None
    assert result.candidate.category == "thought"


# ===========================================================================
# classify 丢弃路径三：R6-AC8 非法 category → ILLEGAL_CATEGORY
# ===========================================================================


def test_classify_discard_illegal_category_non_conversation():
    """category 不属五元集的非对话结构样本 → 按 R6-AC8 丢弃。"""
    sample = SourceSample(category="emotion", text="一些内容")
    result = classify_sample(sample)
    assert result.discarded is True
    assert result.candidate is None
    assert result.reason is DiscardReason.ILLEGAL_CATEGORY
    # detail 记录原始（未小写化）非法取值。
    assert result.detail == "emotion"


def test_classify_discard_illegal_category_conversation():
    """非法 category 优先于模板/字段判定触发（即便具备对话结构）。"""
    sample = SourceSample(category="random", instruction="问", output="答")
    result = classify_sample(sample)
    assert result.discarded is True
    assert result.reason is DiscardReason.ILLEGAL_CATEGORY
    assert result.detail == "random"


def test_classify_stats_counts_three_discard_paths(monkeypatch):
    """批量分类时三条丢弃路径分别计数，保留样本不计入。"""
    monkeypatch.delitem(TEMPLATES, "action")
    samples = [
        SourceSample(category="thought", instruction="", output="x"),     # 空字段
        SourceSample(category="action", text="动作记录"),                  # 模板缺失
        SourceSample(category="weird", text="内容"),                       # 非法 category
        SourceSample(category="skill", instruction="问", output="答"),     # 保留
    ]
    kept, stats = classify_samples(samples)
    assert len(kept) == 1
    assert stats.empty_field_discards == 1
    assert stats.missing_template_discards == 1
    assert stats.illegal_category_discards == 1
    assert stats.total_discards == 3
    assert stats.missing_template_categories == ["action"]
    assert stats.illegal_category_values == ["weird"]


# ===========================================================================
# filter_short 长度边界：R7-AC1（拼接 strip 后长度 < 8 丢弃）
# ===========================================================================


def test_min_content_len_is_eight():
    """前置确认：R7-AC1 的最小拼接长度阈值为 8。"""
    assert MIN_CONTENT_LEN == 8


def test_filter_short_length_seven_dropped():
    """拼接长度 7（< 8）→ 丢弃。"""
    sample = _candidate("abc", "defg")  # 3 + 4 = 7
    assert is_valid_sample(sample) is False
    kept, dropped = filter_short([sample])
    assert kept == []
    assert dropped == 1


def test_filter_short_length_eight_kept():
    """拼接长度 8（== 8 边界）→ 保留。"""
    sample = _candidate("abcd", "efgh")  # 4 + 4 = 8
    assert is_valid_sample(sample) is True
    kept, dropped = filter_short([sample])
    assert kept == [sample]
    assert dropped == 0


def test_filter_short_length_nine_kept():
    """拼接长度 9（> 8）→ 保留。"""
    sample = _candidate("abcde", "fghi")  # 5 + 4 = 9
    assert is_valid_sample(sample) is True
    kept, dropped = filter_short([sample])
    assert kept == [sample]
    assert dropped == 0


def test_filter_short_uses_stripped_length():
    """拼接长度判定基于去除前后空白后的字段（R7-AC1）。"""
    # 含大量空白但 strip 后仅 "ab"+"cd" = 4 < 8 → 丢弃。
    sample = _candidate("   ab   ", "   cd   ")
    assert is_valid_sample(sample) is False


def test_filter_short_boundary_mix():
    """混合输入：仅保留拼接长度 >= 8 的样本，丢弃计数准确。"""
    samples = [
        _candidate("abc", "defg"),    # 7 → 丢弃
        _candidate("abcd", "efgh"),   # 8 → 保留
        _candidate("abcde", "fghi"),  # 9 → 保留
    ]
    kept, dropped = filter_short(samples)
    assert len(kept) == 2
    assert dropped == 1


def test_filter_short_missing_field_dropped():
    """字段缺失（dict 形态缺 instruction）→ 非法丢弃（R7-AC1）。"""
    sample = {"output": "some long output text"}
    assert is_valid_sample(sample) is False
    kept, dropped = filter_short([sample])
    assert kept == []
    assert dropped == 1


def test_filter_short_null_field_dropped():
    """字段为 null → 非法丢弃（R7-AC1）。"""
    sample = {"instruction": None, "output": "some long output text"}
    assert is_valid_sample(sample) is False
    _, dropped = filter_short([sample])
    assert dropped == 1


def test_filter_short_non_string_field_dropped():
    """字段非字符串类型 → 非法丢弃（R7-AC1）。"""
    sample = {"instruction": 12345, "output": "some long output text"}
    assert is_valid_sample(sample) is False
    _, dropped = filter_short([sample])
    assert dropped == 1


# ===========================================================================
# augment 分支：R7-AC4（n = 0 / 1 / 15 / 16 / 超过 16）与 R7-AC7（模板用尽）
# ===========================================================================


def _valid_samples(n: int) -> list[SampleCandidate]:
    """构造 n 条互不相同、长度合规的候选样本。"""
    return [_candidate(f"指令编号{i}", f"输出内容编号{i}") for i in range(n)]


def test_target_sample_count_is_sixteen():
    """前置确认：增强目标最小样本数为 16。"""
    assert TARGET_SAMPLE_COUNT == 16


def test_augment_n_zero_no_augment_no_error():
    """n = 0：不产生增强样本、不抛 R7-AC7（空语料由 pipeline 按 R7-AC6 处理）。"""
    result, stats = augment_samples([])
    assert result == []
    assert stats.original_count == 0
    assert stats.augmented_count == 0
    assert stats.final_count == 0


def test_augment_n_one_template_exhaustion_raises():
    """n = 1：1 样本 × 5 模板 = 6 < 16，模板用尽仍不足 → R7-AC7 抛 CompilerError。"""
    with pytest.raises(CompilerError) as exc_info:
        augment_samples(_valid_samples(1))
    # 错误信息提示无法满足最小样本数 16。
    assert "16" in str(exc_info.value)
    # CompilerError 映射到运行期失败退出码 1（R7-AC7 非零退出码）。
    assert exc_info.value.exit_code == EXIT_RUNTIME_FAILURE


def test_augment_n_fifteen_augments_to_sixteen():
    """n = 15（0 < n < 16 边界）：增强补足至恰好 16，达到即停止。"""
    originals = _valid_samples(15)
    result, stats = augment_samples(originals)
    assert len(result) == TARGET_SAMPLE_COUNT
    assert stats.original_count == 15
    assert stats.augmented_count == 1
    assert stats.final_count == 16
    # 前 15 条为原始样本，保持顺序与内容。
    assert result[:15] == originals
    # 第 16 条为增强样本，instruction 带 [AUG] 前缀且 output 与某原始样本一致。
    augmented = result[15]
    assert augmented.augmented is True
    assert augmented.instruction.startswith(AUGMENT_PREFIX)
    assert augmented.output == originals[0].output


def test_augment_n_sixteen_no_augment():
    """n = 16（边界）：已达目标，不产生任何增强样本（Property 9）。"""
    originals = _valid_samples(16)
    result, stats = augment_samples(originals)
    assert result == originals
    assert stats.original_count == 16
    assert stats.augmented_count == 0
    assert stats.final_count == 16


def test_augment_over_sixteen_no_augment():
    """n > 16：超过目标，输出严格等于输入（Property 9）。"""
    originals = _valid_samples(20)
    result, stats = augment_samples(originals)
    assert result == originals
    assert stats.augmented_count == 0
    assert stats.final_count == 20


def test_augment_stops_immediately_at_target():
    """达到 16 立即停止：n = 3 时仅生成 13 条增强样本（3 + 13 = 16）。"""
    originals = _valid_samples(3)  # 最多 3 + 3*5 = 18 ≥ 16
    result, stats = augment_samples(originals)
    assert len(result) == TARGET_SAMPLE_COUNT
    assert stats.augmented_count == 13
    # 停止后不再继续生成，最终恰为 16。
    assert stats.final_count == 16


def test_augment_template_exhaustion_n_two_raises():
    """n = 2：2 样本 × 5 模板 = 12 < 16 → 模板用尽，按 R7-AC7 抛 CompilerError。"""
    with pytest.raises(CompilerError):
        augment_samples(_valid_samples(2))


def test_augment_all_augmented_have_prefix_and_preserve_output():
    """增强样本均带 [AUG] 前缀、output 与对应原始样本一致（R7-AC5）。"""
    originals = _valid_samples(15)
    result, _ = augment_samples(originals)
    for sample in result:
        if sample.augmented:
            assert sample.instruction.startswith(AUGMENT_PREFIX)
        else:
            assert not sample.instruction.startswith(AUGMENT_PREFIX)
    # 模板数量固定（与 R7-AC7 判定相关）。
    assert len(AUGMENT_TEMPLATES) == 5
