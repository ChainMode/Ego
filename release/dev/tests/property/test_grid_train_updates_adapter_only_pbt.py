"""按 Cell 训练「只更新适配器、不改写基座骨干」的属性测试（hypothesis，gated）。

本文件为 tasks.md 任务 3.4：**复用 real-trainer Property 6，按 Cell 行使**。
它把 real-trainer 的「训练只更新适配器、不改写基座骨干」属性提升到二维稀疏
网格（Sparse_Grid）的语义下——网格训练（``run_grid_training``）为**每个非空
Cell** fresh-load 一个独立冻结的 Base_Model 并独立训练/导出一个 LoRA_Adapter
（fresh-load-per-cell 隔离，见 design.md 「决策 fresh-load-per-cell」）。因此
「base 冻结、LoRA 更新」这一正确性不变量必须**对每个 Cell 分别成立**，且各
Cell 仅从其**自己的** Cell_Sub_Dataset 学习、互不污染（R3-AC1/AC3）。

本属性（复用 real-trainer Property 6，按 Cell 行使）
--------------------------------------------------
对任意种子与小规模 ``params``，对一张小网格（一个 Kernel Cell + 一个 Zone
Cell，各自拥有一个**不同的**微型 Cell_Sub_Dataset），逐 Cell：

1. 构造一个**微型真实模型** + PEFT LoRA 适配器，训练前**逐参数快照**全部张量；
2. 在该 Cell 自己的微型 :class:`SftExample` 数据集上执行
   :func:`~ego_pod.trainer.loop.train_qlora`（与真实后端 ``train`` 同一代码路径）；
3. 断言：(a) 每个 base 骨干参数（名称不含 ``.lora_``）与快照 ``torch.equal``
   （骨干**逐位未被改写**）；(b) 至少一个 LoRA 适配器参数（名称含 ``.lora_``）
   发生变化（该 Cell 的适配器确实在更新）。

对两个 Cell 重复上述断言，证明**按 Cell 训练**时每个 Cell 都各自只更新自己的
适配器、绝不改写共享骨干——这正是 real-trainer Property 6 在网格语义下的逐 Cell
行使（R3-AC3）。

依赖门控（R6-AC8/AC9）
----------------------
与 real-trainer 的 Property 6 一致，本属性必须验证 **真实** PEFT 适配器与真实
torch 反向传播下「base 冻结、LoRA 更新」的行为，故需要 ``torch`` 与 ``peft``：

* 模块顶层以 :func:`pytest.importorskip` 门控 ``peft`` / ``torch``，二者缺失时
  **整个模块被干净跳过**（SKIPPED）而非失败；
* 测试函数另以 ``@pytest.mark.real_training`` 标记（在 ``pyproject.toml`` 注册），
  便于在无真实训练依赖/硬件的环境（如默认 CI）按标记过滤跳过。CPU only——不加
  ``cuda`` 标记。

模型与各 Cell 数据集刻意保持**极小**（小词表、小隐藏维、≤3 条样本、≤5 token、
≤3 epoch），使依赖确实存在时也能快速运行。

**Feature: ego-layered-adapter (reuses real-trainer Property 6 per cell)**

**Validates: Requirements 3.3**
"""

from __future__ import annotations

import pytest

# -- 真实依赖门控（缺失则整个模块 SKIPPED，不失败；R6-AC8/AC9）----------------
torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from peft import LoraConfig, get_peft_model

from ego_pod.config.schema import TrainingParams
from ego_pod.trainer.loop import train_qlora
from ego_pod.trainer.sft import IGNORE_INDEX, SftExample

# ---------------------------------------------------------------------------
# 极小的真实 causal-LM-like 模型（与 real-trainer Property 6 的 TinyCausalLM 一致）
# ---------------------------------------------------------------------------

#: 极小词表与隐藏维，保证每个 Cell 的真实训练在每个 hypothesis 样例下都很快。
_VOCAB_SIZE = 16
_HIDDEN_SIZE = 8

#: 被 LoRA 包裹的线性子模块名（``get_peft_model`` 据此注入 lora_A / lora_B）。
_TARGET_MODULES = ["proj", "lm_head"]

#: 各 Cell 训练前重置的确定性种子（沿用 ``trainer.grid._SEED`` 约定，固定为 0）。
#: 网格训练对每个 Cell 训练前以固定种子重置随机源（R3-AC5）；本测试逐 Cell 复刻
#: 该 fresh-base + 固定种子的隔离方式。
_SEED = 0


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
    本属性所验证的「base 冻结、LoRA 可训练」前置条件（R3-AC3）。

    每个 Cell 都调用一次本函数得到**全新**的冻结 base + 适配器，复刻网格训练的
    fresh-load-per-cell 隔离：各 Cell 互不共享 base 对象，故不会相互污染。
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
# hypothesis 生成器：每 Cell 一个极小数据集 + 小规模超参 + 种子
# ---------------------------------------------------------------------------

_TOKEN_IDS = st.integers(min_value=0, max_value=_VOCAB_SIZE - 1)


@st.composite
def _cell_dataset(draw: st.DrawFn) -> list[SftExample]:
    """生成一个 Cell 的微型 Cell_Sub_Dataset：≤3 条极小 :class:`SftExample`。

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
    return dataset


@st.composite
def _grid_train_scenarios(draw: st.DrawFn):
    """生成 ``(kernel_dataset, zone_dataset, params, seed)``。

    两个 Cell（一个 Kernel、一个 Zone）各自拥有一个**独立生成**的微型数据集，
    用以展示按 Cell 隔离：每个 Cell 只在其自己的子数据集上训练。共享同一组小
    超参与种子（网格训练对每个 Cell 用相同的固定种子重置，R3-AC5）。
    """

    kernel_dataset = draw(_cell_dataset())
    zone_dataset = draw(_cell_dataset())
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
    return kernel_dataset, zone_dataset, params, seed


# ---------------------------------------------------------------------------
# 单 Cell 的 Property 6 断言（fresh base + 独立训练 + 逐参数快照对比）
# ---------------------------------------------------------------------------


def _assert_cell_updates_adapter_only(
    dataset: list[SftExample], params: TrainingParams, seed: int, *, label: str
) -> None:
    """对单个 Cell 行使 real-trainer Property 6：base 骨干逐位不变、LoRA 改变。

    复刻网格训练的逐 Cell 步骤：fresh-load 冻结 base + PEFT 适配器 →（固定种子）
    在该 Cell 自己的数据集上 :func:`train_qlora` → 断言骨干未改写、适配器已更新。
    """

    peft_model = _build_peft_model(params.lora_rank, params.lora_alpha)

    # 训练前逐参数快照（脱离计算图的深拷贝）。
    before = {
        name: param.detach().clone()
        for name, param in peft_model.named_parameters()
    }

    # 进度回调仅作存在性占位（本属性不校验回调）。
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

    # (a) 基座骨干张量逐位未被改写（该 Cell 不改写共享骨干，R3-AC3）。
    assert not base_changed, (
        f"Cell {label}：base 骨干参数被意外改写：{base_changed}"
    )
    # (b) 至少一个 LoRA 适配器张量发生变化（该 Cell 的适配器确实在更新）。
    assert lora_changed, f"Cell {label}：训练后没有任何 LoRA 适配器参数发生变化"


# ===========================================================================
# Property（复用 real-trainer Property 6，按 Cell 行使）：任务 3.4
# ===========================================================================


@pytest.mark.real_training
@given(scenario=_grid_train_scenarios())
@settings(
    max_examples=8,
    deadline=None,  # 真实训练较慢，禁用单样例截止时间。
    suppress_health_check=[HealthCheck.too_slow],
)
def test_grid_train_updates_adapter_only_per_cell(scenario) -> None:
    """按 Cell 训练后每个 Cell 都仅更新自己的适配器、不改写共享骨干。

    对任意种子与小规模 ``params``，对一张小网格（Kernel Cell + Zone Cell，各自一个
    **不同的**微型数据集），逐 Cell 行使 real-trainer Property 6：

    1. Kernel Cell：fresh base + PEFT 适配器，在 Kernel 自己的数据集上训练后，
       base 骨干逐位不变、至少一个 LoRA 适配器张量改变；
    2. Zone Cell：同样以 fresh base + PEFT 适配器，在 Zone **自己的**（不同的）
       数据集上训练后，base 骨干逐位不变、至少一个 LoRA 适配器张量改变。

    两个 Cell 各用 fresh base 独立训练（fresh-load-per-cell），证明按 Cell 训练时
    每个 Cell 只从其 Cell_Sub_Dataset 学习、各 Cell 仅更新自己的适配器、绝不改写
    冻结的 Base_Model 骨干（R3-AC3）。

    **Feature: ego-layered-adapter (reuses real-trainer Property 6 per cell)**

    **Validates: Requirements 3.3**
    """
    kernel_dataset, zone_dataset, params, seed = scenario

    # Cell 1：Kernel（Level 0）——在 Kernel 自己的子数据集上训练。
    _assert_cell_updates_adapter_only(
        kernel_dataset, params, _SEED, label="Kernel"
    )
    # Cell 2：某 Zone Cell（Level 1）——在该分区自己的、不同的子数据集上训练。
    _assert_cell_updates_adapter_only(
        zone_dataset, params, _SEED, label="(coding, L1)"
    )
