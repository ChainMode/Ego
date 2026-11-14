"""真实训练「只更新适配器、不改写基座骨干」的属性测试（hypothesis）。

本文件实现 design.md「Correctness Properties」中的 **Property 6**，对应
tasks.md 任务 3.4：

* **Property 6: 训练只更新适配器、不改写基座骨干** ——
  对任意种子与小规模 ``params``，在一个**微型真实模型**上执行
  :func:`~ego_pod.trainer.loop.train_qlora` 后，被冻结的 Base_Model 骨干张量与
  训练前**逐位相等**（``torch.equal``），而至少一个 LoRA 适配器张量发生变化。
  这证明优化器只更新 ``requires_grad=True`` 的 LoRA 适配器参数（PEFT 已冻结
  base 骨干），不改写基座权重（R4-AC3）。
  **Validates: Requirements 4.3**

依赖门控（R8-AC5、R8-AC6）
--------------------------
不同于其余可用「无 ML 依赖假对象」覆盖的属性测试，Property 6 必须验证 **真实**
PEFT 适配器与真实 torch 反向传播下「base 冻结、LoRA 更新」的行为，因此需要
``torch`` 与 ``peft`` 两个真实依赖：

* 模块顶层以 :func:`pytest.importorskip` 门控 ``peft`` / ``torch``，二者缺失时
  **整个模块被干净跳过**（SKIPPED），而非失败/报错；
* 测试函数另以 ``@pytest.mark.real_training`` 标记（在 ``pyproject.toml`` 注册），
  便于在无真实训练依赖/硬件的环境（如默认 CI）按标记过滤跳过。

模型与数据集刻意保持**极小**（小词表、小隐藏维、≤3 条样本、≤5 token、≤3
epoch），使得在依赖确实存在时也能快速运行。

**Feature: ego-pod-real-trainer**
"""

from __future__ import annotations

import pytest

# -- 真实依赖门控（缺失则整个模块 SKIPPED，不失败）-------------------------
torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from peft import LoraConfig, get_peft_model

from ego_pod.config.schema import TrainingParams
from ego_pod.trainer.loop import train_qlora
from ego_pod.trainer.sft import IGNORE_INDEX, SftExample

# ---------------------------------------------------------------------------
# 极小的真实 causal-LM-like 模型
# ---------------------------------------------------------------------------

#: 极小词表与隐藏维，保证真实训练在每个 hypothesis 样例下都很快。
_VOCAB_SIZE = 16
_HIDDEN_SIZE = 8

#: 被 LoRA 包裹的线性子模块名（``get_peft_model`` 据此注入 lora_A / lora_B）。
_TARGET_MODULES = ["proj", "lm_head"]


class _CausalLMOutput:
    """模拟 transformers ``CausalLMOutput`` 的最小表面：暴露 ``.loss`` 属性。"""

    def __init__(self, loss, logits) -> None:
        self.loss = loss
        self.logits = logits


class TinyCausalLM(torch.nn.Module):
    """极小的 causal-LM-like 真实 ``nn.Module``。

    结构：``Embedding → Linear(proj) → ReLU → Linear(lm_head)``。``proj`` 与
    ``lm_head`` 两个线性层被 LoRA 包裹（见 :data:`_TARGET_MODULES`）。``forward``
    接受 ``input_ids`` / ``attention_mask`` / ``labels``（与 ``loop._collate`` 产出的
    batch 一致），并以 ``ignore_index=-100`` 计算输出-only 交叉熵损失，返回带
    ``.loss``（标量张量，支持 ``.backward()`` / ``.item()``）的输出对象。
    """

    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = torch.nn.Embedding(vocab_size, hidden_size)
        self.proj = torch.nn.Linear(hidden_size, hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask=None, labels=None):
        hidden = torch.relu(self.proj(self.embed(input_ids)))
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=IGNORE_INDEX,
            )
        return _CausalLMOutput(loss=loss, logits=logits)


def _build_peft_model(lora_rank: int, lora_alpha: int):
    """构造 :class:`TinyCausalLM` 并用 PEFT ``LoraConfig`` 包裹为 PEFT 模型。

    ``get_peft_model`` 会冻结全部原始（base 骨干）参数（``requires_grad=False``），
    仅 LoRA 适配器参数（``lora_A`` / ``lora_B``）保持 ``requires_grad=True``，正是
    本属性所验证的「base 冻结、LoRA 可训练」前置条件（R4-AC3）。
    """

    base_model = TinyCausalLM(_VOCAB_SIZE, _HIDDEN_SIZE)
    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=_TARGET_MODULES,
        lora_dropout=0.0,
        bias="none",
    )
    return get_peft_model(base_model, config)


# ---------------------------------------------------------------------------
# hypothesis 生成器：极小数据集 + 小规模超参 + 种子
# ---------------------------------------------------------------------------

_TOKEN_IDS = st.integers(min_value=0, max_value=_VOCAB_SIZE - 1)


@st.composite
def _train_scenarios(draw: st.DrawFn):
    """生成 ``(dataset, params, seed)``：≤3 条极小 :class:`SftExample` + 小超参。

    每条样本保证 **至少一个 output 段标签**（``prompt_len <= seq_len - 1``），使
    输出-only 交叉熵损失有限且对 LoRA 参数产生非零梯度，从而训练确实更新适配器。
    """

    dataset_size = draw(st.integers(min_value=1, max_value=3))
    dataset: list[SftExample] = []
    for _ in range(dataset_size):
        seq_len = draw(st.integers(min_value=2, max_value=5))
        input_ids = draw(
            st.lists(_TOKEN_IDS, min_size=seq_len, max_size=seq_len)
        )
        # 至少保留一个 output token（prompt 段不计损失）。
        prompt_len = draw(st.integers(min_value=0, max_value=seq_len - 1))
        labels = [IGNORE_INDEX] * prompt_len + list(input_ids[prompt_len:])
        dataset.append(
            SftExample(
                input_ids=list(input_ids),
                labels=labels,
                attention_mask=[1] * seq_len,
            )
        )

    params = TrainingParams(
        method="qlora",
        quantization="4bit",
        lora_rank=draw(st.integers(min_value=1, max_value=4)),
        lora_alpha=draw(st.integers(min_value=1, max_value=8)),
        # 取适中学习率，确保单步 AdamW 更新对 LoRA 参数产生可观测变化。
        learning_rate=draw(
            st.floats(
                min_value=1e-2,
                max_value=5e-1,
                allow_nan=False,
                allow_infinity=False,
            )
        ),
        epochs=draw(st.integers(min_value=1, max_value=3)),
        batch_size=draw(st.integers(min_value=1, max_value=3)),
    )
    seed = draw(st.integers(min_value=0, max_value=2**16))
    return dataset, params, seed


# ===========================================================================
# Property 6: 训练只更新适配器、不改写基座骨干（任务 3.4）
# ===========================================================================


@pytest.mark.real_training
@given(scenario=_train_scenarios())
@settings(
    max_examples=12,
    deadline=None,  # 真实训练较慢，禁用单样例截止时间。
    suppress_health_check=[HealthCheck.too_slow],
)
def test_train_updates_adapter_only(scenario) -> None:
    """真实 PEFT 训练后 base 骨干逐位不变、且至少一个 LoRA 适配器张量改变。

    对任意种子与小规模 ``params``：

    1. 构造极小真实模型 + PEFT LoRA 适配器，并在训练前**逐参数快照**全部张量；
    2. 在极小 :class:`SftExample` 数据集上执行 :func:`train_qlora` 若干步；
    3. 断言：(a) 每个 base 骨干参数（名称不含 ``.lora_``）与快照
       ``torch.equal``（完全未被改写）；(b) 至少一个 LoRA 适配器参数（名称含
       ``.lora_``）发生变化。

    这证明优化器只更新 ``requires_grad=True`` 的 LoRA 参数（R4-AC3）。

    **Feature: ego-pod-real-trainer, Property 6**

    **Validates: Requirements 4.3**
    """
    dataset, params, seed = scenario
    peft_model = _build_peft_model(params.lora_rank, params.lora_alpha)

    # 训练前逐参数快照（脱离计算图的深拷贝）。
    before = {
        name: param.detach().clone()
        for name, param in peft_model.named_parameters()
    }

    # 进度回调仅作存在性占位（本属性不校验回调，由 Property 5 覆盖）。
    def _noop_progress(*, epoch: int, step: int, loss: float) -> None:
        return None

    train_qlora(
        peft_model,
        dataset,
        params,
        seed=seed,
        progress_callback=_noop_progress,
        device="cpu",
    )

    base_changed: list[str] = []
    lora_changed: list[str] = []
    for name, param in peft_model.named_parameters():
        unchanged = torch.equal(param.detach(), before[name])
        if ".lora_" in name:
            if not unchanged:
                lora_changed.append(name)
        elif not unchanged:
            base_changed.append(name)

    # (a) 基座骨干张量逐位未被改写。
    assert not base_changed, f"base 骨干参数被意外改写：{base_changed}"
    # (b) 至少一个 LoRA 适配器张量发生变化（训练确实在更新适配器）。
    assert lora_changed, "训练后没有任何 LoRA 适配器参数发生变化"
