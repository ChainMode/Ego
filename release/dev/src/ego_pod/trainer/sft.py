"""SFT_Dataset 构建：Conversation_Corpus → 输出-only 监督样本（需求 3）。

本模块把 Compiler 产出的 Conversation_Corpus（JSONL）转换为带"输出-only 标签
掩码"的监督微调（SFT）样本，供 ``trainer.loop`` 组 batch 训练。设计目标（见
design.md §Components and Interfaces 第 2 节）：

* **纯 Python、导入期零重型依赖**——本模块只使用 tokenizer 的鸭子类型接口
  （``apply_chat_template`` / ``__call__`` / ``encode``），**不**直接导入
  ``torch`` / ``transformers``，因此既能被 PEFT / Unsloth 真实后端复用，也能用
  轻量假 tokenizer 在无 ML 依赖的 CI 中做属性测试（design.md §Testing Strategy）。
* **输出-only 监督**——仅对渲染出的 ``output`` 段 token 计算训练损失，prompt
  段（instruction/input）置 :data:`IGNORE_INDEX`（需求 3.3）。

核心数据流（``build_sft_dataset``）::

    UTF-8 流式逐行读取 JSONL
      → 每行 json.loads（失败 → TrainerError 含 1-based 行号，需求 3.4）
      → 校验 instruction / output 字段（缺失 → TrainerError 含行号，需求 3.4）
      → build_sft_example：render_chat → 分词 → 输出-only 掩码 → 截断
      → 可用样本数为 0 → TrainerError（语料为空，需求 3.5）

不变量（design.md §Data Models "SFT 样本与语料契约"）::

    len(input_ids) == len(labels) == len(attention_mask) ≤ max_seq_len
    labels[0:prompt_len] 全为 IGNORE_INDEX          # prompt 段不计损失
    labels[prompt_len:N] == input_ids[prompt_len:N] # output 段计损失

Tokenizer 鸭子类型契约（与 design.md §2 一致）：

* ``apply_chat_template(messages, *, tokenize=False, add_generation_prompt=...)``
  —— 可选；存在且有可用 chat template 时优先用它渲染对话格式。
* ``encode(text) -> Sequence[int]`` 或 ``__call__(text) -> {"input_ids": [...]}``
  —— 二选一即可，用于把文本分词为 token id 序列。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ego_pod.exit_codes import TrainerError

__all__ = [
    "IGNORE_INDEX",
    "SftExample",
    "CorpusLineError",
    "render_chat",
    "build_sft_example",
    "build_sft_dataset",
]


#: 监督训练中"不计损失"的标签哨兵值（HuggingFace 约定，需求 3.3）。
#:
#: ``labels`` 中取该值的位置在交叉熵损失计算时被忽略，从而实现"仅对 output
#: 段 token 计损失、对 prompt 段 token 不计损失"。
IGNORE_INDEX = -100


@dataclass(frozen=True)
class SftExample:
    """一条 SFT 样本：模型输入 token 序列与对齐的监督标签。

    三个字段等长（``len(input_ids) == len(labels) == len(attention_mask)``）。

    :ivar input_ids: 完整输入序列（prompt 段 + output 段）的 token id 列表，
        长度不超过 ``max_seq_len``。
    :ivar labels: 与 ``input_ids`` 等长的监督标签；prompt 段（``[0, prompt_len)``）
        为 :data:`IGNORE_INDEX`，output 段（``[prompt_len, N)``）为对应位置的
        ``input_ids``（仅 output 段计损失，需求 3.3）。
    :ivar attention_mask: 与 ``input_ids`` 等长的注意力掩码，未截断的真实 token
        位置均为 ``1``。
    """

    input_ids: list[int]
    labels: list[int]
    attention_mask: list[int]


@dataclass(frozen=True)
class CorpusLineError(Exception):
    """语料某行解析失败的结构化错误，携带 1-based 行号与原因。

    本异常是 :func:`build_sft_dataset` 内部用于在逐行解析时定位失败的中间载体，
    由 ``build`` 捕获后转换为面向用户的 :class:`~ego_pod.exit_codes.TrainerError`
    （携带行号与原因，需求 3.4）。

    :ivar line_number: 失败行的 1-based 行号（文件首行为 1）。
    :ivar reason: 该行失败的具体原因（如"不是合法 JSON"、"缺少 output 字段"）。
    """

    line_number: int
    reason: str

    def __str__(self) -> str:  # pragma: no cover - 简单字符串拼接
        return f"第 {self.line_number} 行：{self.reason}"


# ---------------------------------------------------------------------------
# Chat_Template 渲染（需求 3.2）
# ---------------------------------------------------------------------------


def _compose_user_content(instruction: str, input_field: str) -> str:
    """把 ``instruction`` 与可选 ``input`` 合并为单轮 user 文本。

    ``input`` 非空时以空行分隔附加在 ``instruction`` 之后；为空时仅用
    ``instruction``。该合并语义在 chat template 路径与固定模板回退路径间保持
    一致，确保两种渲染方式对同一三元组语义等价。
    """

    if input_field:
        return f"{instruction}\n\n{input_field}"
    return instruction


def _supports_chat_template(tokenizer: Any) -> bool:
    """判定 tokenizer 是否具备可用的 chat template（需求 3.2 的"优先"分支）。

    需要 ``apply_chat_template`` 为可调用方法；若 tokenizer 显式暴露
    ``chat_template`` 属性，则要求其为真值（真实 HuggingFace tokenizer 在未设置
    模板时 ``chat_template is None``，应回退固定模板）。若仅提供
    ``apply_chat_template`` 而无 ``chat_template`` 属性（如轻量假 tokenizer），
    则尝试使用它。
    """

    apply = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply):
        return False
    if hasattr(tokenizer, "chat_template"):
        return bool(tokenizer.chat_template)
    return True


def _render_fixed_template(
    instruction: str, input_field: str, output: str
) -> tuple[str, str]:
    """无 chat template 时的固定模板回退（需求 3.2）。

    构造 ``prompt_text`` 为含 instruction(+input) 的提示前缀，``full_text`` 在其
    后直接拼接 ``output``。由于 ``full_text == prompt_text + output``，``prompt_text``
    恒为 ``full_text`` 的字符串前缀，保证分词后 prompt 段能与 output 段对齐
    （供输出-only 掩码使用）。模板语义与 Compiler 的 instruction/input/output
    三元组一致。
    """

    if input_field:
        prompt_text = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_field}\n\n"
            f"### Response:\n"
        )
    else:
        prompt_text = f"### Instruction:\n{instruction}\n\n### Response:\n"
    full_text = prompt_text + output
    return prompt_text, full_text


def render_chat(
    instruction: str, input_field: str, output: str, tokenizer: Any
) -> tuple[str, str]:
    """把 ``(instruction, input, output)`` 三元组渲染为 ``(prompt_text, full_text)``。

    优先使用 ``tokenizer.apply_chat_template`` 渲染 Base_Model 对话格式：把
    instruction(+input) 作为 user 轮、output 作为 assistant 轮；``prompt_text``
    为仅含 user 轮且 ``add_generation_prompt=True`` 的前缀，``full_text`` 为含
    assistant 轮的完整序列。当 tokenizer 无可用 chat template（或渲染过程抛错）
    时回退到固定模板（需求 3.2）。

    :param instruction: 指令字段。
    :param input_field: 输入字段（可为空字符串）。
    :param output: 监督目标（assistant 回复）。
    :param tokenizer: 满足鸭子类型契约的分词器。
    :returns: ``(prompt_text, full_text)`` 两个字符串。
    """

    if _supports_chat_template(tokenizer):
        try:
            user_content = _compose_user_content(instruction, input_field)
            prompt_messages = [{"role": "user", "content": user_content}]
            full_messages = prompt_messages + [
                {"role": "assistant", "content": output}
            ]
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            full_text = tokenizer.apply_chat_template(
                full_messages, tokenize=False, add_generation_prompt=False
            )
            return str(prompt_text), str(full_text)
        except Exception:
            # chat template 渲染失败（如模板缺失/语法异常）→ 回退固定模板。
            pass
    return _render_fixed_template(instruction, input_field, output)


# ---------------------------------------------------------------------------
# 分词（鸭子类型：encode / __call__）
# ---------------------------------------------------------------------------


def _coerce_token_ids(raw: Any) -> list[int]:
    """把 tokenizer 返回的多种形态统一规整为 ``list[int]``。

    支持：``encode`` 直接返回的 id 序列、``__call__`` 返回的映射（取
    ``input_ids``）、以及可能存在的 batch 嵌套（取第 0 维）。
    """

    if isinstance(raw, Mapping):
        raw = raw["input_ids"]
    # 处理可能的 batch 维度（如 [[...]]）：取首个序列。
    if (
        isinstance(raw, Sequence)
        and not isinstance(raw, (str, bytes))
        and len(raw) > 0
        and isinstance(raw[0], Sequence)
        and not isinstance(raw[0], (str, bytes))
    ):
        raw = raw[0]
    return [int(token) for token in raw]


def _encode(tokenizer: Any, text: str) -> list[int]:
    """用 tokenizer 把 ``text`` 分词为 token id 列表（需求 3.2）。

    优先使用 ``encode``（通常直接返回 ``list[int]``）；其不可用时回退到
    ``__call__`` 并从其结果中提取 ``input_ids``。两次调用（prompt 与 full）始终
    走同一路径，保证 prompt 段长度与 full 序列对齐一致。
    """

    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        return _coerce_token_ids(encode(text))
    return _coerce_token_ids(tokenizer(text))


# ---------------------------------------------------------------------------
# 单样本构建：渲染 → 分词 → 输出-only 掩码 → 截断（需求 3.2、3.3、3.6）
# ---------------------------------------------------------------------------


def build_sft_example(
    instruction: str,
    input_field: str,
    output: str,
    tokenizer: Any,
    max_seq_len: int,
) -> SftExample:
    """把一个三元组构建为带输出-only 掩码并截断的 :class:`SftExample`。

    步骤（design.md §2）：

    1. ``(prompt_text, full_text) = render_chat(...)``；
    2. ``full_ids = encode(full_text)``，``prompt_len = len(encode(prompt_text))``；
    3. ``labels = full_ids`` 的副本，前 ``prompt_len`` 个置 :data:`IGNORE_INDEX`
       （仅 output 段计损失，需求 3.3）；
    4. 若 ``len(full_ids) > max_seq_len``，则 ``input_ids`` / ``labels`` /
       ``attention_mask`` 一并截断至 ``max_seq_len``（需求 3.6）。

    ``prompt_len`` 在分词不满足前缀一致时可能不小于序列总长，此处以
    ``min(prompt_len, N)`` 钳制，确保不越界（此时全部 token 被掩码，无 output
    段计损失）。

    :returns: 满足 ``len(input_ids) == len(labels) == len(attention_mask)
        ≤ max_seq_len`` 的样本。
    """

    prompt_text, full_text = render_chat(
        instruction, input_field, output, tokenizer
    )
    input_ids = _encode(tokenizer, full_text)
    prompt_len = len(_encode(tokenizer, prompt_text))

    labels = list(input_ids)
    mask_upto = min(prompt_len, len(labels))
    for index in range(mask_upto):
        labels[index] = IGNORE_INDEX
    attention_mask = [1] * len(input_ids)

    if len(input_ids) > max_seq_len:
        input_ids = input_ids[:max_seq_len]
        labels = labels[:max_seq_len]
        attention_mask = attention_mask[:max_seq_len]

    return SftExample(
        input_ids=input_ids, labels=labels, attention_mask=attention_mask
    )


# ---------------------------------------------------------------------------
# 语料逐行解析与校验（需求 3.1、3.4、3.5）
# ---------------------------------------------------------------------------


def _require_text_field(obj: Mapping[str, Any], key: str) -> str:
    """从已解析的 JSON 对象中取必需文本字段，缺失/非字符串/为空则报错。

    :raises CorpusLineError: ``key`` 不存在、值非字符串、或为空字符串
        （视为"缺少字段"，需求 3.4）。``line_number`` 占位为 0，由调用方
        :func:`build_sft_dataset` 补齐真实行号。
    """

    if key not in obj:
        raise CorpusLineError(0, f"缺少 {key} 字段")
    value = obj[key]
    if not isinstance(value, str) or value == "":
        raise CorpusLineError(0, f"{key} 字段为空或不是字符串")
    return value


def _parse_corpus_line(line: str) -> tuple[str, str, str]:
    """把一行 JSONL 解析为 ``(instruction, input, output)`` 三元组。

    校验：必须是合法 JSON、且为 JSON 对象、且含非空 ``instruction`` 与
    ``output``（需求 3.4）；``input`` 可选，缺失或非字符串时按空串处理。

    :raises CorpusLineError: 该行非法 JSON、非对象、或缺少必需字段
        （``line_number`` 占位为 0，由调用方补齐真实行号）。
    """

    try:
        obj = json.loads(line)
    except json.JSONDecodeError as exc:
        raise CorpusLineError(0, f"不是合法 JSON（{exc.msg}）") from exc

    if not isinstance(obj, Mapping):
        raise CorpusLineError(0, "不是 JSON 对象")

    instruction = _require_text_field(obj, "instruction")
    output = _require_text_field(obj, "output")
    input_value = obj.get("input", "")
    input_field = input_value if isinstance(input_value, str) else ""
    return instruction, input_field, output


def build_sft_dataset(
    corpus_path: Path, tokenizer: Any, max_seq_len: int
) -> list[SftExample]:
    """流式逐行读取 UTF-8 JSONL 语料并构建 SFT 样本列表（需求 3.1、3.4、3.5）。

    以 UTF-8 流式逐行读取 ``corpus_path``，逐行 ``json.loads`` 并校验字段，再经
    :func:`build_sft_example` 渲染/分词/掩码/截断为一条 :class:`SftExample`。
    一行对应一条样本（逐行守恒，需求 3.1）。

    :param corpus_path: Conversation_Corpus（JSONL）中间文件路径。
    :param tokenizer: 满足鸭子类型契约的分词器。
    :param max_seq_len: Base_Model 最大序列长度，超长样本据此截断（需求 3.6）。
    :returns: 与语料行一一对应的 :class:`SftExample` 列表。
    :raises TrainerError: 某行非法 JSON 或缺少 ``instruction``/``output`` 字段
        （消息含 1-based 行号与原因，需求 3.4）；或可用样本数为 0（语料为空，
        需求 3.5）。
    """

    examples: list[SftExample] = []
    with corpus_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\n").rstrip("\r")
            try:
                instruction, input_field, output = _parse_corpus_line(line)
            except CorpusLineError as exc:
                raise TrainerError(
                    f"[TRAINER] Conversation_Corpus 第 {line_number} 行解析失败："
                    f"{exc.reason}"
                ) from exc
            examples.append(
                build_sft_example(
                    instruction, input_field, output, tokenizer, max_seq_len
                )
            )

    if not examples:
        raise TrainerError(
            "[TRAINER] Conversation_Corpus 为空：无可用于训练的 SFT 样本，"
            "无法继续训练"
        )

    return examples
