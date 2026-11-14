"""`ego_pod.trainer.sft` 的边界单元测试（tasks.md 任务 2.6）。

聚焦属性测试之外的固定样例与边界（design.md §Testing Strategy「单元测试聚焦
固定样例与边界」）：

* **空语料 → TrainerError**（需求 3.5）：0 行可用样本时抛出携带"语料为空"原因
  的 :class:`TrainerError`。
* **chat template 缺失时的回退模板**（需求 3.2）：tokenizer 无可用 chat template
  时 ``render_chat`` 回退到固定模板，且 ``full_text == prompt_text + output``
  保证 prompt 段为 full 序列的字符串前缀（供输出-only 掩码对齐）。
* **prompt_len 计算边界**（需求 3.2、3.3）：截断到 ``max_seq_len < prompt_len``
  时全部标签为 :data:`IGNORE_INDEX`；前缀不一致（whitespace 切分）时
  ``min(prompt_len, N)`` 钳制不越界。

依赖门控：仅使用 ``tests/_doubles.py`` 的确定性无依赖 :class:`FakeTokenizer`，
无需任何 ML 依赖。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from _doubles import FakeTokenizer

from ego_pod.exit_codes import EgoError, TrainerError
from ego_pod.trainer.sft import (
    IGNORE_INDEX,
    build_sft_dataset,
    build_sft_example,
    render_chat,
)


# ---------------------------------------------------------------------------
# 空语料 → TrainerError（需求 3.5）
# ---------------------------------------------------------------------------


def test_empty_corpus_raises_trainer_error(tmp_path: Path) -> None:
    """0 字节语料（无任何行）→ TrainerError，消息含"为空"原因（R3-AC5）。"""
    corpus_path = tmp_path / "empty.jsonl"
    corpus_path.write_text("", encoding="utf-8")

    tokenizer = FakeTokenizer(mode="char")
    with pytest.raises(TrainerError) as exc_info:
        build_sft_dataset(corpus_path, tokenizer, max_seq_len=2048)

    assert "为空" in str(exc_info.value)


def test_trainer_error_is_ego_error_exit_code_1(tmp_path: Path) -> None:
    """空语料抛出的 TrainerError 为 EgoError 子类且退出码为 1（不引入新退出码）。"""
    corpus_path = tmp_path / "empty.jsonl"
    corpus_path.write_text("", encoding="utf-8")

    tokenizer = FakeTokenizer(mode="char")
    with pytest.raises(TrainerError) as exc_info:
        build_sft_dataset(corpus_path, tokenizer, max_seq_len=2048)

    assert isinstance(exc_info.value, EgoError)
    assert exc_info.value.exit_code == 1


# ---------------------------------------------------------------------------
# chat template 缺失时的回退模板（需求 3.2）
# ---------------------------------------------------------------------------


def test_render_chat_fallback_template_without_input() -> None:
    """无 chat template 时回退固定模板；无 input 时省略 Input 段。"""
    tokenizer = FakeTokenizer(mode="char", chat_template=False)

    prompt_text, full_text = render_chat(
        "请总结", "", "这是一段总结", tokenizer
    )

    assert prompt_text == "### Instruction:\n请总结\n\n### Response:\n"
    assert full_text == prompt_text + "这是一段总结"
    # prompt_text 为 full_text 的字符串前缀（掩码对齐前提）。
    assert full_text.startswith(prompt_text)


def test_render_chat_fallback_template_with_input() -> None:
    """无 chat template 且含非空 input 时，固定模板包含 Input 段。"""
    tokenizer = FakeTokenizer(mode="char", chat_template=False)

    prompt_text, full_text = render_chat(
        "翻译以下文本", "Hello", "你好", tokenizer
    )

    assert prompt_text == (
        "### Instruction:\n翻译以下文本\n\n### Input:\nHello\n\n### Response:\n"
    )
    assert full_text == prompt_text + "你好"
    assert full_text.startswith(prompt_text)


def test_fallback_path_label_mask_alignment() -> None:
    """回退模板路径下，prompt 段全为 IGNORE_INDEX、output 段保留 token（R3-AC3）。"""
    tokenizer = FakeTokenizer(mode="char", chat_template=False)

    example = build_sft_example(
        "请总结", "", "这是一段总结", tokenizer, max_seq_len=100_000
    )

    prompt_text, _full = render_chat("请总结", "", "这是一段总结", tokenizer)
    prompt_len = len(tokenizer.encode(prompt_text))

    assert len(example.input_ids) == len(example.labels) == len(example.attention_mask)
    # prompt 段掩码。
    assert all(label == IGNORE_INDEX for label in example.labels[:prompt_len])
    # output 段保留对应 token id。
    assert example.labels[prompt_len:] == example.input_ids[prompt_len:]
    # output 段非空（output 非空且未截断）。
    assert len(example.labels) > prompt_len


def test_chat_template_path_used_when_available() -> None:
    """tokenizer 具备 chat template 时优先使用对话格式渲染（含 role 标记）。"""
    tokenizer = FakeTokenizer(mode="char", chat_template=True)

    prompt_text, full_text = render_chat(
        "指令", "", "回复", tokenizer
    )

    # FakeTokenizer 的对话格式含 <|user|> / <|assistant|> 标记。
    assert "<|user|>" in prompt_text
    assert prompt_text.endswith("<|assistant|>")
    # prompt_text 仍为 full_text 的字符串前缀（掩码对齐前提）。
    assert full_text.startswith(prompt_text)
    assert "回复" in full_text


# ---------------------------------------------------------------------------
# prompt_len 计算边界（需求 3.2、3.3）
# ---------------------------------------------------------------------------


def test_truncation_below_prompt_len_masks_all_labels() -> None:
    """``max_seq_len < prompt_len`` 时截断后全部标签为 IGNORE_INDEX（无 output 计损失）。"""
    tokenizer = FakeTokenizer(mode="char", chat_template=False)

    # 固定模板的 prompt 前缀较长；取很小的 max_seq_len 使截断落在 prompt 段内。
    prompt_text, _full = render_chat(
        "一个较长的指令文本", "", "输出", tokenizer
    )
    prompt_len = len(tokenizer.encode(prompt_text))
    assert prompt_len > 4  # 前置确认 prompt 段足够长

    example = build_sft_example(
        "一个较长的指令文本", "", "输出", tokenizer, max_seq_len=4
    )

    assert len(example.input_ids) == 4
    assert len(example.labels) == 4
    assert len(example.attention_mask) == 4
    # 截断点落在 prompt 段内 → 全部标签被掩码。
    assert all(label == IGNORE_INDEX for label in example.labels)


def test_prompt_len_clamp_does_not_crash_on_non_prefix_tokenizer() -> None:
    """前缀不一致（whitespace 切分）时 ``min(prompt_len, N)`` 钳制不越界。"""
    tokenizer = FakeTokenizer(mode="whitespace", chat_template=False)

    example = build_sft_example(
        "总结 这段 文本", "额外 输入", "简短 回复", tokenizer, max_seq_len=100_000
    )

    n = len(example.input_ids)
    # 三数组等长，且每个 label 要么是 IGNORE_INDEX 要么等于对应 input_id。
    assert len(example.labels) == n == len(example.attention_mask)
    for token, label in zip(example.input_ids, example.labels):
        assert label == IGNORE_INDEX or label == token


def test_single_valid_line_yields_one_example(tmp_path: Path) -> None:
    """单条合法行产出恰好一条样本（与 Property 1 的最小情形一致）。"""
    corpus_path = tmp_path / "one.jsonl"
    corpus_path.write_text(
        '{"instruction": "指令", "output": "输出", "category": "thought"}\n',
        encoding="utf-8",
    )

    tokenizer = FakeTokenizer(mode="char")
    examples = build_sft_dataset(corpus_path, tokenizer, max_seq_len=2048)

    assert len(examples) == 1
    assert len(examples[0].input_ids) == len(examples[0].labels)
