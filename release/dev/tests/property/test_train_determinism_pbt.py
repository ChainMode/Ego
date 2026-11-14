"""`ego_pod.trainer.loop.train_qlora` 训练确定性的属性测试（hypothesis）。

本文件实现 design.md「Correctness Properties」中的 **Property 7: 训练确定性
可复现**，对应 tasks.md 任务 3.5：

* **Property 7: 训练确定性可复现**（任务 3.5）——
  对任意固定 ``(seed, params, corpus)``，在同一微型模型上重复两次训练，得到的
  LoRA 适配器张量在数值容差内相等（语义等价的适配器），且逐 epoch 的训练损失
  序列亦一致。**Validates: Requirements 4.5**

依赖门控（design.md「Testing Strategy」/ R8-AC5、R8-AC6）
--------------------------------------------------------
与仅验证控制流的 ``test_loop_pbt.py``（用无 ML 依赖的假模型）不同，确定性
属性必须验证**真实**的前向/反向/优化器在固定种子下逐次复现，因此本测试需要
真实的 ``torch`` + ``peft``：

* 模块级 ``pytest.importorskip("peft")`` / ``pytest.importorskip("torch")``——
  在缺失任一依赖的环境（如本仓库默认 CI，**未安装 peft**）下整个模块**干净
  跳过**（SKIPPED）而非失败；
* 每个测试再以 ``@pytest.mark.real_training`` 标记，便于按依赖门控筛选
  （``-m "not real_training"``）。

本测试**全程在 CPU 上**进行（不要求 CUDA、不加 ``cuda`` 标记）：随包文档
（README）已说明 CUDA 上某些算子内核非确定、无法保证逐位可复现（R4-AC6），
而 CPU 上 ``train_qlora`` 的 AdamW 更新在固定种子下应为（近）精确复现。

复现性的构造（关键）
--------------------
``train_qlora`` 在循环开始时调用 :func:`set_deterministic_seed`，但**模型初始
权重**（embedding / lm_head / LoRA 的 ``lora_A`` Kaiming 初始化）是在
``train_qlora`` **之前**构造的。因此本测试在**每一次**运行前都先
``set_deterministic_seed(seed)`` 再构造模型，使两次运行的初始权重逐位一致；
随后以**相同**的 ``seed`` 调用 ``train_qlora`` 在**相同**的内存数据集与超参上
训练。两次运行的逐 epoch 损失序列与最终 LoRA 适配器张量应在紧致容差内相等。

为保证训练步本身无额外随机源，LoRA 的 ``lora_dropout`` 显式置 ``0.0``。

**Feature: ego-pod-real-trainer**
"""

from __future__ import annotations

import pytest

# 真实训练依赖门控：缺失 peft/torch 时整个模块干净跳过（R8-AC5、R8-AC6）。
pytest.importorskip("peft")
pytest.importorskip("torch")

import torch  # noqa: E402 - 必须在 importorskip 之后导入
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402
from peft import (  # noqa: E402
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)

from ego_pod.config.schema import TrainingParams  # noqa: E402
from ego_pod.trainer.loop import set_deterministic_seed, train_qlora  # noqa: E402
from ego_pod.trainer.sft import IGNORE_INDEX, SftExample  # noqa: E402

# ---------------------------------------------------------------------------
# 微型真实模型（torch.nn.Module + peft LoRA）
# ---------------------------------------------------------------------------

#: 微型词表与隐藏维度（保持模型极小，使每个属性样本可两次训练且快速）。
_VOCAB_SIZE = 16
_HIDDEN = 8


class _TinyCausalLM(nn.Module):
    """一个极小的、HF 风格的因果语言模型替身。

    暴露 ``forward(input_ids, attention_mask=None, labels=None)`` 并返回带
    ``.loss`` 属性的输出（与 ``trainer.loop`` 的鸭子类型契约一致）。损失为对
    ``labels`` 的交叉熵，``ignore_index=IGNORE_INDEX`` 使被掩码的 prompt 段
    （``-100``）不计入损失（与 ``sft`` 的输出-only 掩码对齐）。

    ``lm_head`` 是 LoRA 的注入目标；``embed`` 与 ``lm_head`` 的 base 权重在
    ``get_peft_model`` 后被冻结，仅 LoRA 适配器参数可训练。
    """

    def __init__(self, vocab_size: int, hidden: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.lm_head = nn.Linear(hidden, vocab_size)

    def forward(self, input_ids, attention_mask=None, labels=None):  # noqa: ANN001
        hidden = self.embed(input_ids)
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
        return _Output(loss=loss, logits=logits)


class _Output:
    """最小输出对象，提供 ``.loss``（贴合 HuggingFace ``ModelOutput`` 表面）。"""

    def __init__(self, *, loss, logits) -> None:  # noqa: ANN001
        self.loss = loss
        self.logits = logits


def _build_lora_model(rank: int, alpha: int) -> nn.Module:
    """构造一个注入了 LoRA 适配器的 :class:`_TinyCausalLM`。

    调用方负责在调用本函数前 :func:`set_deterministic_seed`，从而使
    ``_TinyCausalLM`` 的随机初始化与 ``get_peft_model`` 的 LoRA 初始化逐位可复现。
    ``lora_dropout=0.0`` 确保训练步本身不引入额外随机源（确定性的前提）。
    """

    base = _TinyCausalLM(_VOCAB_SIZE, _HIDDEN)
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=["lm_head"],
    )
    return get_peft_model(base, config)


def _clone_adapter_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """取出仅含 LoRA 适配器张量的 state dict（克隆，避免后续训练的别名干扰）。"""

    return {
        key: tensor.detach().clone()
        for key, tensor in get_peft_model_state_dict(model).items()
    }


# ---------------------------------------------------------------------------
# 进度回调记录器（采集逐 epoch 损失序列）
# ---------------------------------------------------------------------------


class _LossRecorder:
    """记录每个 epoch 的平均训练损失（``train_qlora`` 以关键字参数回调）。"""

    def __init__(self) -> None:
        self.losses: list[float] = []

    def __call__(self, *, epoch: int, step: int, loss: float) -> None:
        self.losses.append(float(loss))


# ---------------------------------------------------------------------------
# 生成策略：固定 (seed, params, 内存数据集)
# ---------------------------------------------------------------------------


def _make_params(
    *, epochs: int, batch_size: int, lora_rank: int, lora_alpha: int
) -> TrainingParams:
    """构造 ``method=qlora`` / ``quantization=4bit`` 的 :class:`TrainingParams`。"""

    return TrainingParams(
        method="qlora",
        quantization="4bit",
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        learning_rate=1e-2,
        epochs=epochs,
        batch_size=batch_size,
    )


@st.composite
def _sft_example(draw: st.DrawFn) -> SftExample:
    """生成一条微型 :class:`SftExample`：token id ∈ [0, vocab)，含输出-only 掩码。

    序列长度 ``L ∈ [2, 6]``；prompt 段长度 ``prompt_len ∈ [0, L-1]`` 保证至少一个
    output token 未被掩码（否则该样本对损失无贡献，且整批全掩码会使交叉熵为 NaN）。
    """

    length = draw(st.integers(min_value=2, max_value=6))
    input_ids = draw(
        st.lists(
            st.integers(min_value=0, max_value=_VOCAB_SIZE - 1),
            min_size=length,
            max_size=length,
        )
    )
    prompt_len = draw(st.integers(min_value=0, max_value=length - 1))
    labels = [
        IGNORE_INDEX if i < prompt_len else input_ids[i] for i in range(length)
    ]
    attention_mask = [1] * length
    return SftExample(
        input_ids=input_ids, labels=labels, attention_mask=attention_mask
    )


@st.composite
def _scenarios(draw: st.DrawFn):
    """生成 ``(seed, params, dataset)``：固定种子、小规模超参与微型内存数据集。"""

    dataset = draw(st.lists(_sft_example(), min_size=1, max_size=6))
    params = _make_params(
        epochs=draw(st.integers(min_value=1, max_value=3)),
        batch_size=draw(st.integers(min_value=1, max_value=3)),
        lora_rank=draw(st.integers(min_value=1, max_value=4)),
        lora_alpha=draw(st.integers(min_value=1, max_value=16)),
    )
    seed = draw(st.integers(min_value=0, max_value=2**16))
    return seed, params, dataset


# ===========================================================================
# Property 7: 训练确定性可复现（任务 3.5）
# ===========================================================================


def _train_once(seed: int, params: TrainingParams, dataset: list[SftExample]):
    """在固定种子下构造微型 LoRA 模型并训练一次，返回 ``(损失序列, 适配器张量)``。

    先 :func:`set_deterministic_seed`(seed) 再构造模型——保证两次调用的初始权重
    （含 LoRA 初始化）逐位一致；随后以同一 ``seed`` 调用 ``train_qlora``（其内部
    亦会 ``set_deterministic_seed``）在 CPU 上训练。
    """

    set_deterministic_seed(seed)
    model = _build_lora_model(params.lora_rank, params.lora_alpha)
    recorder = _LossRecorder()
    train_qlora(
        model,
        dataset,
        params,
        seed=seed,
        progress_callback=recorder,
        device=torch.device("cpu"),
    )
    return recorder.losses, _clone_adapter_state(model)


@pytest.mark.real_training
@given(scenario=_scenarios())
@settings(
    max_examples=15,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_training_is_deterministic_under_fixed_seed(scenario) -> None:
    """固定 ``(seed, params, corpus)`` 下重复两次训练 → 损失序列与 LoRA 适配器一致。

    对任意固定的种子、超参与微型内存数据集，在同一构造的微型真实模型上重复两次
    QLoRA 训练（CPU），断言：

    * 两次的**逐 epoch 损失序列**长度相同且逐值在紧致容差内相等；
    * 两次产出的 **LoRA 适配器**张量键集合相同，且每个键对应张量
      ``torch.allclose`` 相等（CPU 上应为精确/近精确）——即语义等价的适配器
      （R4-AC5）。

    **Feature: ego-pod-real-trainer, Property 7**

    **Validates: Requirements 4.5**
    """
    seed, params, dataset = scenario

    losses_a, adapter_a = _train_once(seed, params, dataset)
    losses_b, adapter_b = _train_once(seed, params, dataset)

    # 逐 epoch 损失序列一致（每 epoch 恰一次回调，长度应为 params.epochs）。
    assert len(losses_a) == len(losses_b) == params.epochs
    for la, lb in zip(losses_a, losses_b):
        assert la == pytest.approx(lb, rel=1e-6, abs=1e-6)

    # LoRA 适配器张量键集合一致。
    assert set(adapter_a.keys()) == set(adapter_b.keys())
    assert adapter_a, "适配器 state dict 不应为空（应至少含一对 lora_A/lora_B）"

    # 每个适配器张量在紧致容差内逐位相等（语义等价的适配器，R4-AC5）。
    for key, tensor_a in adapter_a.items():
        tensor_b = adapter_b[key]
        assert tensor_a.shape == tensor_b.shape
        assert torch.allclose(tensor_a, tensor_b, atol=1e-6, rtol=1e-5), (
            f"键 {key!r} 的 LoRA 适配器张量两次训练不一致（确定性被破坏）"
        )
