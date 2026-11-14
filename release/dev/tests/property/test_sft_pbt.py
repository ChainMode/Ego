"""`ego_pod.trainer.sft` 的 SFT_Dataset 构建属性测试（需求 R3）。

本文件实现 design.md「Correctness Properties」中的 **Property 1–4**，对应
tasks.md 任务 2.2 / 2.3 / 2.4 / 2.5：

* **Property 1: 语料解析逐行守恒** —— ``build_sft_dataset`` 产出的样本数等于
  合法语料的行数（逐行流式解析，一行一样本）。**Validates: Requirements 3.1**
* **Property 2: 输出-only 标签掩码与序列对齐** —— ``input_ids`` / ``labels`` /
  ``attention_mask`` 三者等长；prompt 段 ``labels`` 全为 ``IGNORE_INDEX``，
  output 段 ``labels`` 等于对应 ``input_ids``。
  **Validates: Requirements 3.2, 3.3**
* **Property 3: 截断长度上界** —— 对任意 ``max_seq_len >= 1``，三个数组长度均
  不超过 ``max_seq_len``。**Validates: Requirements 3.6**
* **Property 4: 非法语料行触发带行号的 TrainerError** —— 位置 k 处的非法 JSON
  或缺字段行使 ``build_sft_dataset`` 抛出 :class:`TrainerError`，消息含 1-based
  行号 k。**Validates: Requirements 3.4**

依赖门控：上述属性均为 **纯逻辑**，使用 ``tests/_doubles.py`` 中确定性、无 ML
依赖的 :class:`FakeTokenizer`。默认 ``mode="char"`` 保证 ``encode`` 的前缀一致性
（``encode(a + b) == encode(a) + encode(b)``），从而 ``full_text`` 以 ``prompt_text``
为前缀时，分词后 prompt 段与 output 段严格对齐，支撑 Property 2 的掩码切分。
"""

from __future__ import annotations

import json
from pathlib import Path

import hypothesis.strategies as st
import pytest
from hypothesis import HealthCheck, given, settings

from _doubles import FakeTokenizer

from ego_pod.exit_codes import TrainerError
from ego_pod.trainer.sft import (
    IGNORE_INDEX,
    build_sft_dataset,
    build_sft_example,
    render_chat,
)

# ---------------------------------------------------------------------------
# 文本与语料行生成策略
# ---------------------------------------------------------------------------

#: 五个合法 Data_Category（仅作语料字段填充，build_sft_dataset 不校验该字段）。
_CATEGORIES = ["sense", "action", "thought", "skill", "credentials"]

#: 必需字段文本（instruction / output）：非空字符串。
_required_text = st.text(min_size=1, max_size=50)

#: 可选 input 字段文本：可为空。
_optional_text = st.text(max_size=40)


@st.composite
def valid_rows(draw: st.DrawFn) -> dict:
    """生成一条合法 Conversation_Corpus 行对象（含非空 instruction 与 output）。"""

    row: dict = {
        "instruction": draw(_required_text),
        "output": draw(_required_text),
        "category": draw(st.sampled_from(_CATEGORIES)),
    }
    if draw(st.booleans()):
        row["input"] = draw(_optional_text)
    return row


@st.composite
def triples(draw: st.DrawFn) -> tuple[str, str, str]:
    """生成 ``(instruction, input, output)`` 三元组（instruction / output 非空）。"""

    return (
        draw(_required_text),
        draw(_optional_text),
        draw(_required_text),
    )


#: 明确非法的 JSON 行字面量（解析必失败）。
_malformed_json_line = st.sampled_from(
    [
        "{",
        "}",
        "{not valid",
        '{"instruction": "x"',
        "[1, 2, 3",
        "@@@ not json @@@",
        "'single quotes'",
        "{'instruction': 'x', 'output': 'y'}",
    ]
)

#: 合法 JSON 但缺字段 / 字段非法 / 非对象，触发结构化校验失败的行。
_invalid_field_line = st.sampled_from(
    [
        json.dumps({"output": "有效输出", "category": "thought"}),  # 缺 instruction
        json.dumps({"instruction": "有效指令", "category": "skill"}),  # 缺 output
        json.dumps({"instruction": "", "output": "x"}),  # instruction 为空
        json.dumps({"instruction": "x", "output": ""}),  # output 为空
        json.dumps({"instruction": "x", "output": 123}),  # output 非字符串
        json.dumps([1, 2, 3]),  # 非对象
        json.dumps("只是一个字符串"),  # 非对象
        json.dumps(42),  # 非对象
    ]
)

#: 任意一种非法行。
_bad_line = st.one_of(_malformed_json_line, _invalid_field_line)


@st.composite
def corpus_with_bad_line(draw: st.DrawFn) -> tuple[list[str], int]:
    """生成「若干合法行 + 位置 k 处一行非法行 + 若干合法行」的语料。

    :returns: ``(lines, k)``，``lines`` 为各物理行的字符串列表，``k`` 为非法行的
        1-based 行号（其前的所有行均合法，故 build 的首个错误必报告第 k 行）。
    """

    before = draw(st.lists(valid_rows(), min_size=0, max_size=4))
    after = draw(st.lists(valid_rows(), min_size=0, max_size=4))
    bad = draw(_bad_line)
    k = len(before) + 1
    lines = (
        [json.dumps(r, ensure_ascii=False) for r in before]
        + [bad]
        + [json.dumps(r, ensure_ascii=False) for r in after]
    )
    return lines, k


def _write_jsonl_lines(path: Path, lines: list[str]) -> Path:
    """把若干物理行写为 UTF-8 JSONL 文件（行间 LF，末尾换行；无空行）。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines) + ("\n" if lines else "")
    path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Property 1: 语料解析逐行守恒
# ---------------------------------------------------------------------------


@given(rows=st.lists(valid_rows(), min_size=1, max_size=8))
@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_corpus_line_conservation(rows: list[dict], tmp_path: Path) -> None:
    """合法语料的样本数严格等于其行数（一行一样本，逐行流式解析）。

    **Feature: ego-pod-real-trainer, Property 1**

    **Validates: Requirements 3.1**
    """
    lines = [json.dumps(r, ensure_ascii=False) for r in rows]
    corpus_path = _write_jsonl_lines(tmp_path / "corpus.jsonl", lines)

    tokenizer = FakeTokenizer(mode="char")
    examples = build_sft_dataset(corpus_path, tokenizer, max_seq_len=2048)

    assert len(examples) == len(rows)


# ---------------------------------------------------------------------------
# Property 2: 输出-only 标签掩码与序列对齐
# ---------------------------------------------------------------------------


@given(triple=triples())
@settings(max_examples=200, deadline=None)
def test_output_only_label_mask_alignment(triple: tuple[str, str, str]) -> None:
    """三数组等长；prompt 段全为 IGNORE_INDEX，output 段 labels 等于 input_ids。

    取足够大的 ``max_seq_len`` 以避免截断，从而完整检验掩码切分；prompt 边界
    经公开 API（``render_chat`` + ``encode``）独立计算，验证实现确实在
    ``prompt_len`` 处切分（仅对 output token 计损失）。

    **Feature: ego-pod-real-trainer, Property 2**

    **Validates: Requirements 3.2, 3.3**
    """
    instruction, input_field, output = triple
    tokenizer = FakeTokenizer(mode="char")  # 前缀一致，默认带 chat template
    example = build_sft_example(
        instruction, input_field, output, tokenizer, max_seq_len=100_000
    )

    # 三数组等长。
    assert len(example.input_ids) == len(example.labels) == len(example.attention_mask)

    # 独立计算 prompt 边界（与实现同路径：encode(prompt_text) 的长度）。
    prompt_text, _full_text = render_chat(
        instruction, input_field, output, tokenizer
    )
    prompt_len = len(tokenizer.encode(prompt_text))
    n = len(example.input_ids)
    boundary = min(prompt_len, n)

    # prompt 段（[0, boundary)）全为 IGNORE_INDEX（不计损失）。
    assert all(label == IGNORE_INDEX for label in example.labels[:boundary])
    # output 段（[boundary, N)）的 labels 等于对应位置的 input_ids（仅 output 计损失）。
    assert example.labels[boundary:] == example.input_ids[boundary:]
    # 未截断时 attention_mask 全为 1。
    assert all(mask == 1 for mask in example.attention_mask)


# ---------------------------------------------------------------------------
# Property 3: 截断长度上界
# ---------------------------------------------------------------------------


@given(triple=triples(), max_seq_len=st.integers(min_value=1, max_value=64))
@settings(max_examples=200, deadline=None)
def test_truncation_upper_bound(
    triple: tuple[str, str, str], max_seq_len: int
) -> None:
    """对任意 ``max_seq_len >= 1``，三个数组长度均不超过 ``max_seq_len``。

    ``max_seq_len`` 取小区间以充分触发截断分支。

    **Feature: ego-pod-real-trainer, Property 3**

    **Validates: Requirements 3.6**
    """
    instruction, input_field, output = triple
    tokenizer = FakeTokenizer(mode="char")
    example = build_sft_example(
        instruction, input_field, output, tokenizer, max_seq_len=max_seq_len
    )

    assert len(example.input_ids) <= max_seq_len
    assert len(example.labels) <= max_seq_len
    assert len(example.attention_mask) <= max_seq_len


# ---------------------------------------------------------------------------
# Property 4: 非法语料行触发带行号的 TrainerError
# ---------------------------------------------------------------------------


@given(corpus=corpus_with_bad_line())
@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_invalid_line_raises_trainer_error_with_line_number(
    corpus: tuple[list[str], int], tmp_path: Path
) -> None:
    """位置 k 处的非法行使 build 抛出 TrainerError，消息含 1-based 行号 k。

    **Feature: ego-pod-real-trainer, Property 4**

    **Validates: Requirements 3.4**
    """
    lines, k = corpus
    corpus_path = _write_jsonl_lines(tmp_path / "corpus.jsonl", lines)

    tokenizer = FakeTokenizer(mode="char")
    with pytest.raises(TrainerError) as exc_info:
        build_sft_dataset(corpus_path, tokenizer, max_seq_len=2048)

    message = str(exc_info.value)
    # 消息必须携带该非法行的 1-based 行号。
    assert f"第 {k} 行" in message
    # 消息表明解析失败（携带具体原因）。
    assert "解析失败" in message
