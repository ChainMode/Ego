"""共享测试替身（test doubles），供无 ML 依赖的属性/单元测试复用。

本模块是 **非测试** 的纯 Python 辅助模块（文件名以 ``_`` 前缀，且不含
``test_`` 前缀，故 pytest 不会将其作为测试用例收集）。它为 ego-pod-real-trainer
的属性测试提供两件确定性、无第三方依赖的替身：

* :class:`FakeTokenizer` —— 确定性假 tokenizer，暴露 ``apply_chat_template`` /
  ``__call__`` / ``encode`` / ``decode``，按 **字符** 或 **空白** 切分把文本映射
  为稳定的整数 token id。用于在不安装 ``transformers`` 的前提下验证
  ``trainer.sft``（Property 1–4，语料解析 / 输出-only 掩码 / 截断）。
* :class:`FakeModel` —— 假 torch-like 模型，暴露 ``named_parameters()`` 与可控
  loss 的 ``forward`` / ``__call__``（外加确定性 ``generate``）。用于在不安装
  ``torch`` / ``peft`` 的前提下验证 ``trainer.loop``（Property 5 进度回调、
  Property 8 NaN/Inf 与 OOM 守卫）与 ``trainer.generation``（Property 17 确定性
  解码）。

设计约束（对应 R8-AC4 / R8-AC5）：

- **绝不在模块顶层 import torch / transformers / peft 等重型 ML 依赖**。所有替身
  以纯 Python 实现，使依赖门控为 ``real_training`` / ``cuda`` 的测试之外的属性
  测试可在最小环境（仅 ``pytest`` + ``hypothesis``）下运行。
- **确定性**：相同输入恒产生相同输出，且 token id 的映射由内容（而非遭遇顺序）
  决定，跨实例、跨进程稳定（基于 :mod:`hashlib`，不受 ``PYTHONHASHSEED`` 影响）。

确定性 token id 的关键不变量（供 ``trainer.sft`` 的输出-only 标签掩码依赖）：
**前缀一致性** —— 对于字符级切分（默认），``encode(a + b) == encode(a) + encode(b)``
恒成立。因此 ``full_text`` 以 ``prompt_text`` 为前缀时，
``encode(full_text)[: len(encode(prompt_text))] == encode(prompt_text)``，使
"prompt 段置 IGNORE_INDEX、output 段保留 token"的掩码切分严格对齐。空白级切分
不保证该不变量（边界 token 可能跨接），仅在不依赖前缀一致性的场景使用。
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Callable, Iterable, Iterator, Sequence

__all__ = [
    "FakeEncoding",
    "FakeTokenizer",
    "FakeParameter",
    "FakeLoss",
    "FakeOutput",
    "FakeModel",
    "FakePeftModel",
    "FakePeftBackend",
]

# ---------------------------------------------------------------------------
# 稳定 token id 工具
# ---------------------------------------------------------------------------

#: 保留特殊 token 的 id（与内容 token 区间不重叠）。
_PAD_ID = 0
_BOS_ID = 1
_EOS_ID = 2
_UNK_ID = 3

#: 内容 token id 的起始偏移与取值区间宽度。内容 id ∈ [_CONTENT_OFFSET,
#: _CONTENT_OFFSET + _CONTENT_MODULO)，与上面的保留 id 不重叠。
_CONTENT_OFFSET = 16
_CONTENT_MODULO = 50_000


def stable_token_id(token: str) -> int:
    """把一个 token 字符串映射为稳定的正整数 id（基于 SHA-256，内容决定）。

    使用 :mod:`hashlib`（而非内建 ``hash``）以确保结果不受 ``PYTHONHASHSEED``
    影响，从而跨实例与跨进程稳定可复现。
    """

    digest = hashlib.sha256(token.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % _CONTENT_MODULO
    return _CONTENT_OFFSET + bucket


# ---------------------------------------------------------------------------
# 假 tokenizer 的编码返回对象
# ---------------------------------------------------------------------------


class FakeEncoding:
    """``FakeTokenizer.__call__`` 的返回对象，兼容多种常见访问方式。

    同时支持：

    * 属性访问：``enc.input_ids`` / ``enc.attention_mask``；
    * 键访问（dict-like）：``enc["input_ids"]`` / ``enc["attention_mask"]``；
    * ``len(enc)`` 返回 **token 序列长度**（便于 ``prompt_len = len(tokenizer(text))``）;
    * ``iter(enc)`` 迭代 ``input_ids``。

    注意 ``__len__`` 返回的是序列长度而非键个数，刻意贴合 ``trainer.sft`` 设计
    中 ``prompt_len = len(tokenizer(prompt_text))`` 的用法。
    """

    def __init__(
        self, input_ids: Sequence[int], attention_mask: Sequence[int] | None = None
    ) -> None:
        self.input_ids: list[int] = list(input_ids)
        self.attention_mask: list[int] = (
            list(attention_mask)
            if attention_mask is not None
            else [1] * len(self.input_ids)
        )

    def __getitem__(self, key: str) -> list[int]:
        if key == "input_ids":
            return self.input_ids
        if key == "attention_mask":
            return self.attention_mask
        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        return key in ("input_ids", "attention_mask")

    def keys(self) -> list[str]:
        return ["input_ids", "attention_mask"]

    def __len__(self) -> int:
        return len(self.input_ids)

    def __iter__(self) -> Iterator[int]:
        return iter(self.input_ids)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FakeEncoding):
            return (
                self.input_ids == other.input_ids
                and self.attention_mask == other.attention_mask
            )
        return NotImplemented

    def __repr__(self) -> str:  # pragma: no cover - 仅调试用
        return (
            f"FakeEncoding(input_ids={self.input_ids!r}, "
            f"attention_mask={self.attention_mask!r})"
        )


# ---------------------------------------------------------------------------
# 假 tokenizer
# ---------------------------------------------------------------------------


class FakeTokenizer:
    """确定性、无依赖的假 tokenizer。

    :param mode: 切分模式。``"char"``（默认）按字符切分，保证 ``encode`` 的
        **前缀一致性**（``encode(a + b) == encode(a) + encode(b)``），供
        ``trainer.sft`` 的输出-only 标签掩码依赖；``"whitespace"`` 按
        ``\\s+|\\S+`` 切分（空白与非空白各成一段），不保证前缀一致性。
    :param chat_template: 是否提供 ``apply_chat_template``（默认 ``True``）。设为
        ``False`` 时该方法抛 :class:`NotImplementedError`，以便测试 ``render_chat``
        在 tokenizer 无 chat template 时的回退分支。
    :param model_max_length: 暴露给调用方的最大序列长度（仅作信息，``__call__``
        的截断由 ``max_length`` 显式控制）。
    """

    def __init__(
        self,
        *,
        mode: str = "char",
        chat_template: bool = True,
        model_max_length: int = 2048,
    ) -> None:
        if mode not in ("char", "whitespace"):
            raise ValueError(f"未知切分模式：{mode!r}（应为 'char' 或 'whitespace'）")
        self.mode = mode
        self._has_chat_template = chat_template
        self.model_max_length = model_max_length

        # 常见属性，贴合真实 tokenizer 的访问习惯。
        self.pad_token_id = _PAD_ID
        self.bos_token_id = _BOS_ID
        self.eos_token_id = _EOS_ID
        self.unk_token_id = _UNK_ID
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"

    # -- 切分 ---------------------------------------------------------------

    def _split(self, text: str) -> list[str]:
        if self.mode == "char":
            return list(text)
        # whitespace 模式：保留空白段为独立 token，覆盖整串。
        return re.findall(r"\s+|\S+", text)

    # -- 公共 API -----------------------------------------------------------

    def encode(
        self, text: str, *, add_special_tokens: bool = False, **_: Any
    ) -> list[int]:
        """把文本切分并映射为稳定的整数 token id 列表。

        默认 ``add_special_tokens=False``（不添加 BOS/EOS），以保持字符级切分的
        前缀一致性不被特殊 token 破坏。
        """

        ids = [stable_token_id(tok) for tok in self._split(text)]
        if add_special_tokens:
            ids = [self.bos_token_id, *ids, self.eos_token_id]
        return ids

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
        **_: Any,
    ) -> FakeEncoding:
        """对文本编码并返回 :class:`FakeEncoding`（含 ``input_ids`` 与掩码）。

        当 ``truncation=True`` 且给定 ``max_length`` 时，把序列截断至该长度。
        """

        ids = self.encode(text, add_special_tokens=add_special_tokens)
        if truncation and max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]
        return FakeEncoding(ids)

    def apply_chat_template(
        self,
        messages: Sequence[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        **_: Any,
    ) -> Any:
        """把消息列表渲染为确定性的对话格式文本（或 token id）。

        每条消息渲染为 ``<|{role}|>{content}<|end|>\\n``；``add_generation_prompt``
        为真时追加 ``<|assistant|>`` 生成前缀。``tokenize=False``（默认）返回字符串；
        ``tokenize=True`` 返回 :meth:`encode` 的结果。

        当本实例以 ``chat_template=False`` 构造时，抛 :class:`NotImplementedError`，
        模拟"tokenizer 无 chat template"以触发上层回退逻辑。
        """

        if not self._has_chat_template:
            raise NotImplementedError("该 FakeTokenizer 未配置 chat template")

        parts: list[str] = []
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            parts.append(f"<|{role}|>{content}<|end|>\n")
        if add_generation_prompt:
            parts.append("<|assistant|>")
        rendered = "".join(parts)
        if tokenize:
            return self.encode(rendered)
        return rendered

    def decode(
        self, token_ids: Iterable[int], *, skip_special_tokens: bool = True, **_: Any
    ) -> str:
        """把 token id 序列确定性地还原为字符串。

        由于内容 token id 的映射是单向哈希，无法逐字还原原文；本方法仅保证
        **确定性**（相同 id 序列恒得相同字符串），用于解码可复现性测试。
        ``skip_special_tokens=True`` 时跳过 PAD/BOS/EOS/UNK 等保留 id。
        """

        specials = {_PAD_ID, _BOS_ID, _EOS_ID, _UNK_ID}
        pieces = [
            f"<t{int(tid)}>"
            for tid in token_ids
            if not (skip_special_tokens and int(tid) in specials)
        ]
        return "".join(pieces)


# ---------------------------------------------------------------------------
# 假模型相关的轻量类型
# ---------------------------------------------------------------------------


class FakeParameter:
    """假参数：模拟 torch ``Parameter`` 的最小表面。

    :param name: 参数全名（如 ``base_model.layer.weight`` 或 ``...lora_A.weight``）。
    :param data: 参数数值（任意可变的数值列表，仅作可观察状态用）。
    :param requires_grad: 是否参与梯度更新。被冻结的 Base_Model 骨干参数为
        ``False``，LoRA 适配器参数为 ``True``。
    """

    def __init__(
        self, name: str, data: Sequence[float], *, requires_grad: bool
    ) -> None:
        self.name = name
        self.data: list[float] = list(data)
        self.requires_grad = requires_grad

    def snapshot(self) -> list[float]:
        """返回当前数值的浅拷贝，便于"训练前后是否改变"的逐位比较。"""

        return list(self.data)

    def __repr__(self) -> str:  # pragma: no cover - 仅调试用
        return (
            f"FakeParameter(name={self.name!r}, requires_grad={self.requires_grad}, "
            f"data={self.data!r})"
        )


class FakeLoss:
    """可控的假 loss，模拟 torch 标量张量的最小表面。

    支持 ``.item()`` / ``float(loss)`` 取值、``.backward()`` 空操作，以及基本算术
    （``/`` ``*`` ``+``，返回新的 :class:`FakeLoss`），便于训练循环做梯度累积式
    的归一化或聚合而无需真实 torch。
    """

    def __init__(self, value: float) -> None:
        self.value = float(value)

    def item(self) -> float:
        return self.value

    def backward(self, *args: Any, **kwargs: Any) -> None:
        """空操作：假模型不计算真实梯度。"""

    def detach(self) -> "FakeLoss":
        return self

    def __float__(self) -> float:
        return self.value

    def __truediv__(self, other: float) -> "FakeLoss":
        return FakeLoss(self.value / float(other))

    def __mul__(self, other: float) -> "FakeLoss":
        return FakeLoss(self.value * float(other))

    __rmul__ = __mul__

    def __add__(self, other: Any) -> "FakeLoss":
        return FakeLoss(self.value + float(other))

    __radd__ = __add__

    def __repr__(self) -> str:  # pragma: no cover - 仅调试用
        return f"FakeLoss({self.value!r})"


class FakeOutput:
    """假前向输出，模拟 transformers ``ModelOutput`` 的最小表面。

    同时支持属性访问（``out.loss``）与键访问（``out["loss"]``）。
    """

    def __init__(self, loss: FakeLoss) -> None:
        self.loss = loss

    def __getitem__(self, key: str) -> FakeLoss:
        if key == "loss":
            return self.loss
        raise KeyError(key)

    def __repr__(self) -> str:  # pragma: no cover - 仅调试用
        return f"FakeOutput(loss={self.loss!r})"


class _OutOfMemory(RuntimeError):
    """模拟 CUDA OOM 的运行期错误（消息含 'out of memory'）。

    训练循环按"消息含 out of memory 的 RuntimeError"识别 OOM 并包装为
    ``TrainerError``，故此异常继承 :class:`RuntimeError` 即可被识别。
    """


# ---------------------------------------------------------------------------
# 假模型
# ---------------------------------------------------------------------------


class FakeModel:
    """无依赖的假 torch-like 模型，暴露 ``named_parameters()`` 与可控 loss。

    用于在不安装 ``torch`` / ``peft`` 的前提下驱动训练循环的控制流：

    * **可控 loss**：通过 ``loss_value``（常量）或 ``loss_schedule``（按前向调用
      次序逐个取值，耗尽后回落到 ``loss_value``）控制每次 ``forward`` 返回的 loss，
      可用于注入 NaN/Inf 序列以验证数值守卫（Property 8）。
    * **可注入 OOM**：``oom_at_call=k`` 时第 ``k`` 次（1-based）``forward`` 抛出
      消息含 ``out of memory`` 的 :class:`RuntimeError`，验证 OOM → TrainerError。
    * **参数分组**：``base_param_names`` 为冻结骨干（``requires_grad=False``），
      ``lora_param_names`` 为可训练适配器（``requires_grad=True``），供"只更新
      适配器、不改写骨干"类断言与优化器只取可训练参数的逻辑使用。
    * **确定性 generate**：:meth:`generate` 对相同输入与设置恒返回相同 token 序列，
      支撑确定性解码可复现性测试（Property 17）。

    :param loss_value: 每次前向返回的常量 loss（默认 ``1.0``）。
    :param loss_schedule: 可选的 loss 取值序列，按前向调用次序消费；可含
        ``float('nan')`` / ``float('inf')`` 以注入数值发散。
    :param oom_at_call: 可选，于第 N 次前向抛 OOM（1-based）。
    :param base_param_names: 冻结骨干参数名（默认两个占位骨干权重）。
    :param lora_param_names: 可训练适配器参数名（默认一对 lora_A / lora_B）。
    :param vocab_size: ``generate`` 采样的 token id 上界（用于取模）。
    :param name: 模型可读名称（仅作信息）。
    """

    def __init__(
        self,
        *,
        loss_value: float = 1.0,
        loss_schedule: Sequence[float] | None = None,
        oom_at_call: int | None = None,
        base_param_names: Sequence[str] | None = None,
        lora_param_names: Sequence[str] | None = None,
        vocab_size: int = _CONTENT_MODULO,
        name: str = "FakeModel",
    ) -> None:
        self.name = name
        self.training = False
        self.vocab_size = vocab_size

        base_names = (
            list(base_param_names)
            if base_param_names is not None
            else ["base_model.model.layer0.weight", "base_model.model.layer1.weight"]
        )
        lora_names = (
            list(lora_param_names)
            if lora_param_names is not None
            else [
                "base_model.model.layer0.lora_A.weight",
                "base_model.model.layer0.lora_B.weight",
            ]
        )
        # 名称 → FakeParameter，保持声明顺序（named_parameters 的稳定遍历序）。
        self._params: dict[str, FakeParameter] = {}
        for n in base_names:
            self._params[n] = FakeParameter(n, [0.0, 0.0], requires_grad=False)
        for n in lora_names:
            self._params[n] = FakeParameter(n, [0.0, 0.0], requires_grad=True)

        self._loss_value = float(loss_value)
        self._loss_schedule = (
            [float(v) for v in loss_schedule] if loss_schedule is not None else None
        )
        self._oom_at_call = oom_at_call
        self._call_count = 0

    # -- 参数 ---------------------------------------------------------------

    def named_parameters(self) -> Iterator[tuple[str, FakeParameter]]:
        """以稳定顺序产出 ``(name, FakeParameter)``（先骨干后适配器）。"""

        for name, param in self._params.items():
            yield name, param

    def parameters(self) -> list[FakeParameter]:
        """产出全部 :class:`FakeParameter`（与 torch ``parameters()`` 对应）。"""

        return list(self._params.values())

    def trainable_parameters(self) -> list[FakeParameter]:
        """仅产出 ``requires_grad=True`` 的参数（便于断言优化器选参）。"""

        return [p for p in self._params.values() if p.requires_grad]

    # -- 训练态切换（torch.nn.Module 风格，均为空操作返回自身） ----------

    def train(self, mode: bool = True) -> "FakeModel":
        self.training = mode
        return self

    def eval(self) -> "FakeModel":
        self.training = False
        return self

    def to(self, *args: Any, **kwargs: Any) -> "FakeModel":
        return self

    def zero_grad(self, *args: Any, **kwargs: Any) -> None:
        """空操作：假模型不维护真实梯度。"""

    # -- 前向 ---------------------------------------------------------------

    def forward(self, *args: Any, **kwargs: Any) -> FakeOutput:
        """返回携带可控 loss 的 :class:`FakeOutput`。

        若配置了 ``oom_at_call`` 且当前为第 N 次前向，则抛出模拟 OOM 的
        ``RuntimeError``；否则按 ``loss_schedule``（若有，未耗尽）或 ``loss_value``
        取 loss。
        """

        self._call_count += 1
        if self._oom_at_call is not None and self._call_count == self._oom_at_call:
            raise _OutOfMemory("CUDA out of memory: 模拟训练步显存不足")

        if self._loss_schedule and self._call_count <= len(self._loss_schedule):
            loss_value = self._loss_schedule[self._call_count - 1]
        else:
            loss_value = self._loss_value
        return FakeOutput(FakeLoss(loss_value))

    __call__ = forward

    @property
    def call_count(self) -> int:
        """已发生的前向调用次数（便于断言步数）。"""

        return self._call_count

    # -- 解码 ---------------------------------------------------------------

    def generate(
        self,
        input_ids: Sequence[int] | None = None,
        *,
        max_new_tokens: int = 16,
        do_sample: bool = False,
        seed: int = 0,
        **_: Any,
    ) -> list[int]:
        """确定性地生成 token 序列（返回 prompt + 新生成）。

        新 token 完全由 ``input_ids`` 内容、位置与 ``seed`` 决定（不使用任何全局
        随机源），因此相同输入与设置恒得相同输出，支撑确定性解码可复现性测试。
        ``do_sample`` 仅作签名兼容；本实现始终走确定性路径。
        """

        prompt = [int(t) for t in (input_ids or [])]
        base = (sum(prompt) + len(prompt) * 31 + seed) % self.vocab_size
        new_tokens = [
            _CONTENT_OFFSET + (base + i * 97 + 7) % self.vocab_size
            for i in range(max(0, int(max_new_tokens)))
        ]
        return prompt + new_tokens

    def __repr__(self) -> str:  # pragma: no cover - 仅调试用
        return (
            f"FakeModel(name={self.name!r}, params={list(self._params)!r}, "
            f"call_count={self._call_count})"
        )


# ---------------------------------------------------------------------------
# 便捷构造器
# ---------------------------------------------------------------------------


def make_finite_loss_model(loss_value: float = 1.0, **kwargs: Any) -> FakeModel:
    """构造一个每步返回固定有限 loss 的 :class:`FakeModel`（常用于进度回调测试）。"""

    return FakeModel(loss_value=loss_value, **kwargs)


def make_diverging_loss_model(
    finite_steps: int = 1, *, diverge: float = math.inf, **kwargs: Any
) -> FakeModel:
    """构造一个前 ``finite_steps`` 步有限、随后发散（Inf/NaN）的 :class:`FakeModel`。"""

    schedule = [1.0] * max(0, finite_steps) + [diverge]
    return FakeModel(loss_schedule=schedule, **kwargs)


# ---------------------------------------------------------------------------
# 假 PEFT 多适配器模型替身（供 Coexist_Activate 可逆性属性测试，Property 4）
# ---------------------------------------------------------------------------


class FakePeftModel:
    """无依赖的假 PEFT 多适配器模型，模拟 ``runtime.inference`` 所用的 PEFT 表面。

    复刻 ``ego_pod.runtime.inference._activate_cells`` / ``activate`` /
    ``deactivate`` 真正调用到的 PEFT 多适配器 API 子集，使
    :class:`~ego_pod.runtime.inference.BackendInferenceSession` 能在**不安装
    ``peft`` / ``torch``** 的前提下被驱动，从而行使真实的
    ``load_adapter`` / ``set_adapter`` / 活动适配器集合维护逻辑（method A，
    Coexist_Activate，可逆性，对应 design.md Property 4 / 需求 5.4、5.5、7.6）：

    * 暴露 ``peft_config`` 属性（一个 ``adapter_name -> 占位对象`` 的字典），使
      ``_activate_cells`` 走 ``hasattr(base, "peft_config")`` 为真的
      ``load_adapter`` 分支（即把本替身当作"已是 PEFT 模型的 base"）——这正是
      设计建议的"由注入后端返回已暴露 ``peft_config`` 的对象"路径，无需真实
      ``PeftModel.from_pretrained``。
    * ``load_adapter(path, adapter_name=...)``：按调用顺序记录 ``(path, name)``，
      把命名适配器注册进 ``peft_config`` 与 ``loaded_adapters``（去重、保序）。
    * ``set_adapter(names)``：把活动适配器集合重置为 ``names``（接受 ``str`` 或
      ``list[str]``，归一为列表），并把每次调用追加到 ``set_adapter_calls`` 历史，
      使测试能断言"初始激活顺序为 Kernel → 各分区 Level 低→高"与"激活/停用后
      活动集合精确回到先前状态"。
    * ``active_adapters`` 属性：返回当前活动适配器名列表（镜像最近一次
      ``set_adapter`` 的入参），与真实 PEFT 的 ``active_adapters`` 表面一致。

    本替身为**纯 Python、确定性、无任何 ML 依赖**，仅作可观察的调用记录与状态
    机，不承载任何真实权重或张量。

    :param peft_config: 初始 ``peft_config`` 映射；缺省为空字典（表示 base 已被
        视作 PEFT 模型但尚未挂载任何命名适配器）。
    :param name: 可读名称（仅作信息/调试用）。
    """

    def __init__(
        self,
        *,
        peft_config: dict[str, Any] | None = None,
        name: str = "FakePeftModel",
    ) -> None:
        self.name = name
        #: adapter_name -> 占位对象，暴露给 ``hasattr(base, "peft_config")`` 判定。
        self.peft_config: dict[str, Any] = (
            dict(peft_config) if peft_config is not None else {}
        )
        #: 按首次加载顺序去重保存的命名适配器名（load 顺序）。
        self.loaded_adapters: list[str] = []
        #: adapter_name -> 加载路径（最近一次）。
        self.adapter_paths: dict[str, str] = {}
        #: 全量 ``load_adapter`` 调用历史：``(path, adapter_name)``（含重复）。
        self.load_adapter_calls: list[tuple[str, str]] = []
        #: 全量 ``set_adapter`` 调用历史：每项为该次传入的活动适配器名列表。
        self.set_adapter_calls: list[list[str]] = []
        #: 当前活动适配器名列表（镜像最近一次 ``set_adapter`` 的入参）。
        self._active: list[str] = []

    def load_adapter(
        self, path: str, *, adapter_name: str | None = None, **_: Any
    ) -> None:
        """记录一次命名适配器加载（不读任何文件）。

        把 ``(path, adapter_name)`` 追加到 :attr:`load_adapter_calls`，并把该命名
        适配器注册进 :attr:`peft_config` 与 :attr:`loaded_adapters`（保序去重），
        模拟 PEFT ``load_adapter`` 把适配器挂为命名适配器但不立即激活的语义。
        """

        name = adapter_name if adapter_name is not None else f"adapter_{len(self.loaded_adapters)}"
        self.load_adapter_calls.append((str(path), name))
        if name not in self.loaded_adapters:
            self.loaded_adapters.append(name)
            self.peft_config[name] = object()
        self.adapter_paths[name] = str(path)

    def set_adapter(self, adapter_names: Any) -> None:
        """把活动适配器集合重置为 ``adapter_names``（接受 ``str`` 或 ``list``）。

        归一为列表后追加到 :attr:`set_adapter_calls` 历史并更新 :attr:`_active`，
        使测试能逐次断言激活顺序与活动集合（PEFT 对多个激活适配器做加性组合，
        本替身只记录"哪些被激活、以何顺序"）。
        """

        if isinstance(adapter_names, str):
            names = [adapter_names]
        else:
            names = list(adapter_names)
        self.set_adapter_calls.append(list(names))
        self._active = list(names)

    @property
    def active_adapters(self) -> list[str]:
        """当前活动适配器名列表（镜像最近一次 ``set_adapter`` 的入参）。"""

        return list(self._active)

    @classmethod
    def from_pretrained(
        cls, base: Any, path: str, *, adapter_name: str | None = None, **_: Any
    ) -> "FakePeftModel":
        """模拟 ``PeftModel.from_pretrained``：构造替身并挂载首个命名适配器。

        当注入的 base **未**暴露 ``peft_config`` 时，``_activate_cells`` 会走此
        路径；本替身据此构造一个新的 :class:`FakePeftModel` 并 ``load_adapter``
        首个适配器。测试默认让 base 直接为 :class:`FakePeftModel`（已暴露
        ``peft_config``），故通常不经此路径，但保留以覆盖另一分支。
        """

        model = cls()
        model.load_adapter(path, adapter_name=adapter_name)
        return model

    def __repr__(self) -> str:  # pragma: no cover - 仅调试用
        return (
            f"FakePeftModel(name={self.name!r}, loaded={self.loaded_adapters!r}, "
            f"active={self._active!r})"
        )


class FakePeftBackend:
    """注入到 ``BackendInferenceSession(backend=...)`` 的假后端（返回假 PEFT base）。

    其 ``load_base_model`` 返回 ``(FakePeftModel, FakeTokenizer)``——返回的 base
    已暴露 ``peft_config``，使
    :func:`~ego_pod.runtime.inference._activate_cells` 走
    ``load_adapter`` + ``set_adapter`` 的多适配器并存激活分支，从而在无真实
    ``peft`` / ``torch`` 的前提下端到端驱动会话构造与激活/停用逻辑（Property 4）。

    :param model: 作为 base 返回的假 PEFT 模型；缺省新建一个 :class:`FakePeftModel`。
    :param tokenizer: 作为 tokenizer 返回的假分词器；缺省新建一个 :class:`FakeTokenizer`。
    """

    def __init__(
        self,
        *,
        model: "FakePeftModel | None" = None,
        tokenizer: "FakeTokenizer | None" = None,
    ) -> None:
        self.model: FakePeftModel = model if model is not None else FakePeftModel()
        self.tokenizer: FakeTokenizer = (
            tokenizer if tokenizer is not None else FakeTokenizer()
        )
        #: 记录 ``load_base_model`` 的调用入参以便断言。
        self.calls: list[tuple[str, str]] = []

    def load_base_model(self, model_id: str, quantization: str) -> tuple[Any, Any]:
        """记录入参并返回 ``(假 PEFT base, 假 tokenizer)``（不加载任何真实权重）。"""

        self.calls.append((model_id, quantization))
        return self.model, self.tokenizer

    def __repr__(self) -> str:  # pragma: no cover - 仅调试用
        return f"FakePeftBackend(calls={self.calls!r})"
