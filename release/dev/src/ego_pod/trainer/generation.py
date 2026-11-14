"""确定性（贪心）解码辅助（需求 6.3）。

本模块提供训练后自检与 Runtime 真实推理共享的解码配置 :class:`DecodeSettings`
与生成辅助 :func:`generate_reply`，对应设计文档 §4「``trainer/generation.py``」。

设计要点：

- **确定性解码**：以贪心方式（``do_sample=False`` / ``num_beams=1``）生成，并在
  生成前重设随机数种子，使得在相同会话、相同非空用户输入与相同
  :class:`DecodeSettings` 下连续调用 :func:`generate_reply` 返回逐字一致的回复
  （需求 6.3，对应 Correctness Property 17）。
- **导入安全**：本模块顶层不导入 ``torch`` / ``transformers`` / ``peft`` 等可选
  重型依赖；确定性种子设定通过惰性导入 :func:`ego_pod.trainer.loop.set_deterministic_seed`
  完成（缺失时回退到本模块内的最小实现），因此在仅安装默认依赖的环境中
  ``import ego_pod.trainer.generation`` 永不抛错（需求 8.4）。
- **鸭子类型接口**：仅依赖 ``tokenizer.apply_chat_template`` / ``tokenizer.decode``
  与 ``model.generate`` 三个通用接口，且对它们的返回值（torch 张量或纯 Python
  列表）做形状归一化处理，便于用确定性的假 model + 假 tokenizer 在无 ML 依赖
  的 CI 中做属性测试（Property 17）。

真实 HuggingFace 路径与轻量假实现的对应关系：

- 真实：``tokenizer.apply_chat_template(..., return_tensors="pt")`` 返回形状
  ``[1, prompt_len]`` 的张量，``model.generate(...)`` 返回形状
  ``[1, prompt_len + new_len]`` 的张量；``tokenizer.decode`` 把新生成 token 段
  解码为字符串。
- 假实现：``apply_chat_template`` 可返回 ``list[int]`` 或 ``list[list[int]]``，
  ``generate`` 在其后追加确定性的新 token，``decode`` 把 token 列表拼接为字符串。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

__all__ = [
    "DecodeSettings",
    "generate_reply",
]


@dataclass(frozen=True)
class DecodeSettings:
    """确定性（贪心）解码设置（需求 6.3）。

    :ivar max_new_tokens: 单次生成的新 token 数上限。
    :ivar do_sample: 是否采样。固定为 ``False`` 走贪心解码，保证同输入同输出。
    :ivar num_beams: beam 数。固定为 ``1``（贪心，无 beam search）。
    :ivar seed: 生成前重设的随机数种子，进一步保证逐次可复现。
    """

    max_new_tokens: int = 256
    do_sample: bool = False
    num_beams: int = 1
    seed: int = 0


# ---------------------------------------------------------------------------
# 确定性种子设定（惰性解析，导入安全）
# ---------------------------------------------------------------------------


def _resolve_seed_setter() -> Callable[[int], None]:
    """解析确定性种子设定函数。

    优先复用 :func:`ego_pod.trainer.loop.set_deterministic_seed`（与训练循环
    共用同一套种子语义）；该模块缺失或导入失败时回退到本模块的最小实现
    :func:`_fallback_set_deterministic_seed`。

    采用惰性解析（在 :func:`generate_reply` 调用时而非模块导入时解析），以确保
    本模块在 ``loop`` 尚不可用或缺失 ML 依赖时仍可被安全导入（需求 8.4）。
    """

    try:
        from ego_pod.trainer.loop import set_deterministic_seed

        return set_deterministic_seed
    except Exception:  # pragma: no cover - loop 不可用时的防御性回退
        return _fallback_set_deterministic_seed


def _fallback_set_deterministic_seed(seed: int) -> None:
    """最小确定性种子实现：为 ``random`` / ``numpy`` / ``torch`` 设定种子。

    仅在无法从 :mod:`ego_pod.trainer.loop` 取得 ``set_deterministic_seed`` 时使用。
    ``numpy`` 与 ``torch`` 缺失时静默跳过对应部分，保证不依赖任何可选依赖。
    """

    import random

    random.seed(seed)

    try:  # pragma: no cover - 取决于运行环境是否安装 numpy
        import numpy as _np

        _np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass

    try:  # pragma: no cover - 取决于运行环境是否安装 torch
        import torch as _torch

        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)
    except Exception:  # pragma: no cover
        pass


# ---------------------------------------------------------------------------
# 形状归一化辅助（张量 / 列表通用）
# ---------------------------------------------------------------------------


def _sequence_length(input_ids: Any) -> int:
    """计算渲染出的 prompt token 序列长度（支持张量与列表）。

    - torch 张量：以 ``shape[-1]`` 取最后一维长度（``[1, N]`` 或 ``[N]`` 均得 ``N``）。
    - ``list[list[int]]``（批为 1）：取第一行的长度。
    - ``list[int]``：取列表长度。
    """

    shape = getattr(input_ids, "shape", None)
    if shape is not None:
        return int(shape[-1])
    if len(input_ids) > 0 and isinstance(input_ids[0], (list, tuple)):
        return len(input_ids[0])
    return len(input_ids)


def _slice_new_tokens(output: Any, prompt_len: int) -> Any:
    """从 ``model.generate`` 的输出中切出 prompt 之后新生成的 token 段。

    生成结果包含「prompt token + 新生成 token」，本函数丢弃前 ``prompt_len`` 个
    prompt token，仅保留新生成部分（支持张量与列表，且兼容批为 1 的二维形状）。
    """

    shape = getattr(output, "shape", None)
    if shape is not None:
        row = output[0] if len(shape) > 1 else output
        return row[prompt_len:]
    if len(output) > 0 and isinstance(output[0], (list, tuple)):
        return output[0][prompt_len:]
    return output[prompt_len:]


# ---------------------------------------------------------------------------
# 生成入口
# ---------------------------------------------------------------------------


def generate_reply(
    model: Any,
    tokenizer: Any,
    user_input: str,
    settings: DecodeSettings,
) -> str:
    """渲染 user 轮 → 贪心 ``model.generate`` → 解码新生成 token 为字符串（需求 6.3）。

    流程：

    1. 把 ``user_input`` 渲染为单条 user 轮消息，经
       ``tokenizer.apply_chat_template(..., add_generation_prompt=True)`` 得到
       prompt 的 token 序列。
    2. ``set_deterministic_seed(settings.seed)`` 重设随机数种子。
    3. 以贪心设置（``do_sample=False`` / ``num_beams=1`` /
       ``max_new_tokens=settings.max_new_tokens``）调用 ``model.generate``。
    4. 切出 prompt 之后新生成的 token 段，经 ``tokenizer.decode`` 解码为字符串
       （跳过特殊 token）并返回。

    在相同会话、相同非空用户输入与相同 :class:`DecodeSettings` 下，贪心解码 +
    固定种子保证逐次返回完全一致的回复（需求 6.3，Property 17）。

    :param model: 暴露 ``generate`` 方法的（PEFT 叠加后的）语言模型；可选地暴露
        ``device`` 属性以将输入张量迁移到对应设备。
    :param tokenizer: 暴露 ``apply_chat_template`` 与 ``decode`` 的分词器。
    :param user_input: 一次非空用户输入。
    :param settings: 确定性解码设置。
    :returns: 模型对该输入新生成的回复字符串。
    """

    # 1. 渲染 user 轮为 prompt token 序列（add_generation_prompt=True 追加生成前缀）。
    messages = [{"role": "user", "content": user_input}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    # 将输入张量迁移到模型设备（真实张量路径）；假实现或无 device 时跳过。
    device = getattr(model, "device", None)
    if device is not None and hasattr(input_ids, "to"):
        input_ids = input_ids.to(device)

    prompt_len = _sequence_length(input_ids)

    # 2. 生成前重设种子，保证逐次可复现（需求 6.3）。
    set_deterministic_seed = _resolve_seed_setter()
    set_deterministic_seed(settings.seed)

    # 3. 贪心解码：不采样、单 beam，最多生成 max_new_tokens 个新 token。
    output = model.generate(
        input_ids,
        max_new_tokens=settings.max_new_tokens,
        do_sample=settings.do_sample,
        num_beams=settings.num_beams,
    )

    # 4. 仅解码 prompt 之后新生成的 token 段为字符串。
    new_tokens = _slice_new_tokens(output, prompt_len)
    return tokenizer.decode(new_tokens, skip_special_tokens=True)
