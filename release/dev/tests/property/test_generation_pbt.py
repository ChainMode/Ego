"""`ego_pod.trainer.generation` 确定性解码的属性测试（需求 R6.3）。

本文件实现 design.md「Correctness Properties」中的 **Property 17**，对应
tasks.md 任务 4.2：

* **Property 17: 确定性解码可复现** —— 对任意非空用户输入，在同一会话与确定性
  （贪心、``do_sample=False``、固定种子）解码设置下连续两次调用
  :func:`~ego_pod.trainer.generation.generate_reply`，返回完全相同的回复字符串。
  **Validates: Requirements 6.3**

依赖门控：本属性面向 **纯解码控制流**，使用 ``tests/_doubles.py`` 中确定性、无
ML 依赖的 :class:`FakeModel` 与 :class:`FakeTokenizer`，因此可在最小环境（仅
``pytest`` + ``hypothesis``）下运行，无需安装 ``torch`` / ``transformers`` / ``peft``。

设计对齐说明：真实 HuggingFace 路径中，``generate_reply`` 以
``tokenizer.apply_chat_template(..., return_tensors="pt")`` 取得 prompt 的 token
张量再交由 ``model.generate``（见 ``trainer/generation.py`` 模块文档）。
:class:`FakeTokenizer.apply_chat_template` 默认返回 **字符串**，故本文件提供一个
轻量子类 :class:`_TokenizingTokenizer`：当调用方传入 ``return_tensors`` 时按真实
tokenizer 的语义返回 token id 列表（``FakeModel.generate`` 据此追加确定性新
token）。该子类仅做"渲染→分词"的形状适配，不引入任何随机性，保持替身的确定性。
"""

from __future__ import annotations

from typing import Any, Sequence

import hypothesis.strategies as st
from hypothesis import given, settings

from _doubles import FakeModel, FakeTokenizer

from ego_pod.trainer.generation import DecodeSettings, generate_reply

# ---------------------------------------------------------------------------
# 解码路径专用的分词 tokenizer
# ---------------------------------------------------------------------------


class _TokenizingTokenizer(FakeTokenizer):
    """``apply_chat_template`` 在被请求张量时返回 token id 的 :class:`FakeTokenizer`。

    ``generate_reply`` 以 ``apply_chat_template(..., return_tensors="pt")`` 渲染
    prompt 并把结果交给 ``model.generate``。真实 HF tokenizer 在 ``return_tensors``
    非 ``None`` 时返回已分词的张量；本子类据此在收到 ``return_tensors`` 时返回
    :meth:`encode` 的 token id 列表（``FakeModel.generate`` 接受整数序列），其余
    行为与父类一致，保持确定性、无随机性。
    """

    def apply_chat_template(
        self,
        messages: Sequence[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        return_tensors: Any = None,
        **kwargs: Any,
    ) -> Any:
        if return_tensors is not None:
            tokenize = True
        return super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# 生成策略
# ---------------------------------------------------------------------------

#: 非空用户输入：至少一个字符的文本（覆盖多语言字符、空白、标点等）。
_nonempty_user_input = st.text(min_size=1, max_size=120)

#: 解码设置取值（do_sample 固定为 False，确保贪心确定性路径）。
_max_new_tokens = st.integers(min_value=1, max_value=64)
_seeds = st.integers(min_value=0, max_value=2**16)


# ===========================================================================
# Property 17: 确定性解码可复现（任务 4.2）
# ===========================================================================


@given(
    user_input=_nonempty_user_input,
    max_new_tokens=_max_new_tokens,
    seed=_seeds,
)
@settings(max_examples=200, deadline=None)
def test_deterministic_decode_is_reproducible(
    user_input: str, max_new_tokens: int, seed: int
) -> None:
    """同一会话、同一非空输入、同一确定性设置下两次 ``generate_reply`` 完全相同。

    对任意非空用户输入与任意（贪心、``do_sample=False``、固定种子）
    :class:`DecodeSettings`：在同一 ``model`` / ``tokenizer`` 实例（同一会话）上
    连续两次调用 :func:`generate_reply`，返回的回复字符串逐字相等（R6-AC3）。

    **Feature: ego-pod-real-trainer, Property 17**

    **Validates: Requirements 6.3**
    """
    model = FakeModel()
    tokenizer = _TokenizingTokenizer(mode="char")
    settings_obj = DecodeSettings(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        seed=seed,
    )

    first = generate_reply(model, tokenizer, user_input, settings_obj)
    second = generate_reply(model, tokenizer, user_input, settings_obj)

    assert first == second


@given(
    user_input=_nonempty_user_input,
    max_new_tokens=_max_new_tokens,
    seed=_seeds,
)
@settings(max_examples=120, deadline=None)
def test_default_decode_settings_are_reproducible(
    user_input: str, max_new_tokens: int, seed: int
) -> None:
    """默认（贪心）:class:`DecodeSettings` 下重复调用亦逐字一致。

    钉死"确定性解码"对默认设置（``do_sample=False`` / ``num_beams=1``）同样成立，
    且回复对非空输入非空（``FakeModel.generate`` 至少生成
    ``max_new_tokens >= 1`` 个新 token）。

    **Feature: ego-pod-real-trainer, Property 17**

    **Validates: Requirements 6.3**
    """
    model = FakeModel()
    tokenizer = _TokenizingTokenizer(mode="char")
    settings_obj = DecodeSettings(max_new_tokens=max_new_tokens, seed=seed)

    first = generate_reply(model, tokenizer, user_input, settings_obj)
    second = generate_reply(model, tokenizer, user_input, settings_obj)

    assert first == second
    assert len(first) > 0
