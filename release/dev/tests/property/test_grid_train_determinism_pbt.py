"""按 Cell 训练「同 Cell 训练确定性可复现」的属性测试（hypothesis，gated）。

本文件为 tasks.md 任务 3.5：**复用 real-trainer Property 7，按 Cell 行使**。
它把 real-trainer 的「训练确定性可复现」属性提升到二维稀疏网格（Sparse_Grid）
语义下——网格训练（``trainer.grid.run_grid_training``）为**每个非空 Cell** 在训练
前以固定种子 ``set_deterministic_seed(_SEED)`` 重置随机源、fresh-load 一个独立冻结
的 Base_Model、独立训练并导出一个 LoRA_Adapter（见 ``grid.py`` 的逐 Cell 步骤与
design.md「决策 fresh-load-per-cell」）。因此「相同 Base_Model、相同 Cell_Sub_Dataset、
相同训练参数与相同种子 ⇒ 重复训练某 Cell 得到语义等价的 LoRA_Adapter」这一确定性
不变量必须**对每个 Cell 分别成立**（R3-AC5，与 ego-pod-real-trainer Requirement 4
一致）。

本属性（复用 real-trainer Property 7，按 Cell 行使）
--------------------------------------------------
对任意固定 ``(seed, params, cell_sub_dataset)``，把**同一个 Cell**——以一个微型
Cell_Sub_Dataset 表示——训练**两次**，每次都复刻 ``grid.py`` 的逐 Cell 流程
（``set_deterministic_seed(seed)`` → 构造微型 LoRA 模型 → ``train_qlora``），断言两次
得到**等价的适配器**：

* 两次的**逐 epoch 损失序列**长度相同且逐值在紧致容差内相等；
* 两次产出的 **LoRA 适配器**张量键集合相同，且每个键对应张量 ``torch.allclose``
  相等（CPU 上应为精确/近精确）——即语义等价的适配器（R3-AC5）。

这正是 real-trainer Property 7 在网格语义下、对**一个 Cell 的子数据集**逐 Cell
行使（复用 ``test_train_determinism_pbt.py`` 的微型模型 + 训练两次 的完全相同的
构造，仅把数据集参数化为「Cell 子数据集」）。

此外另以一个 gated 测试做**按 Cell 区分性**的可选断言：两个**不同的** Cell（不同
子数据集）在**相同种子**下训练，产出**不同**的适配器——印证各 Cell 仅从其自己的
子数据集学习、互不污染，而确定性并非把所有 Cell 退化为同一适配器。

依赖门控（R6-AC8/AC9）
----------------------
与 real-trainer 的 Property 7 一致，本属性必须验证 **真实** 的前向/反向/优化器在
固定种子下逐次复现，故需要真实的 ``torch`` + ``peft``：

* 模块顶层以 :func:`pytest.importorskip` 门控 ``peft`` / ``torch``，二者缺失时
  **整个模块被干净跳过**（SKIPPED）而非失败；
* 测试函数另以 ``@pytest.mark.real_training`` 标记（已在 ``pyproject.toml`` 注册），
  便于在无真实训练依赖/硬件的环境（如默认 CI）按标记过滤跳过。

本测试**全程在 CPU 上**进行（不要求 CUDA、不加 ``cuda`` 标记）：随包文档已说明
CUDA 上某些算子内核非确定、无法保证逐位可复现，而 CPU 上 ``train_qlora`` 的 AdamW
更新在固定种子下应为（近）精确复现。

模型与各 Cell 数据集刻意保持**极小**（小词表、小隐藏维、≤6 条样本、≤6 token、
≤3 epoch），hypothesis ``max_examples`` 取小、``deadline=None`` 且抑制 ``too_slow``，
使依赖确实存在时也能快速运行。

**Feature: ego-layered-adapter (reuses real-trainer Property 7 per cell)**

**Validates: Requirements 3.5**
"""

from __future__ import annotations

import pytest

# -- 真实依赖门控（缺失则整个模块 SKIPPED，不失败；R6-AC8/AC9）----------------
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
# 微型真实模型（torch.nn.Module + peft LoRA）——与 real-trainer Property 7 一致
# ---------------------------------------------------------------------------

#: 微型词表与隐藏维度（保持模型极小，使每个 Cell 子数据集可两次训练且快速）。
_VOCAB_SIZE = 16
_HIDDEN = 8

#: 各 Cell 训练前重置的确定性种子（沿用 ``trainer.grid._SEED`` 约定，固定为 0）。
#: 网格训练对每个 Cell 训练前以固定种子 ``set_deterministic_seed(_SEED)`` 重置随机源
#: （R3-AC5）；本测试的「按 Cell 区分性」断言以该固定种子复刻网格行为。
_SEED = 0


class _TinyCausalLM(nn.Module):
    """一个极小的、HF 风格的因果语言模型替身（与 Property 7 的 TinyCausalLM 一致）。

    暴露 ``forward(input_ids, attention_mask=None, labels=None)`` 并返回带 ``.loss``
    属性的输出（与 ``trainer.loop`` 的鸭子类型契约一致）。损失为对 ``labels`` 的交叉
    熵，``ignore_index=IGNORE_INDEX`` 使被掩码的 prompt 段（``-100``）不计入损失。
    ``lm_head`` 是 LoRA 的注入目标；其 base 权重在 ``get_peft_model`` 后被冻结，仅
    LoRA 适配器参数可训练。
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
    """构造一个注入了 LoRA 适配器的 :class:`_TinyCausalLM`（复刻 grid.py 的逐 Cell 建模）。

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
# 生成策略：固定 (seed, params, Cell 子数据集)
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
def _cell_sub_dataset(draw: st.DrawFn) -> list[SftExample]:
    """生成一个 Cell 的微型 Cell_Sub_Dataset：≥1 条微型 :class:`SftExample`。

    每条样本：序列长度 ``L ∈ [2, 6]``；prompt 段长度 ``prompt_len ∈ [0, L-1]`` 保证
    至少一个 output token 未被掩码（否则该样本对损失无贡献，且整批全掩码会使交叉熵
    为 NaN）。这正是某个 Cell 被路由后落到该 Cell 的子数据集形态。
    """

    size = draw(st.integers(min_value=1, max_value=6))
    dataset: list[SftExample] = []
    for _ in range(size):
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
            IGNORE_INDEX if i < prompt_len else input_ids[i]
            for i in range(length)
        ]
        attention_mask = [1] * length
        dataset.append(
            SftExample(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
            )
        )
    return dataset


@st.composite
def _scenarios(draw: st.DrawFn):
    """生成 ``(seed, params, cell_sub_dataset)``：固定种子、小超参与一个 Cell 子数据集。"""

    dataset = draw(_cell_sub_dataset())
    params = _make_params(
        epochs=draw(st.integers(min_value=1, max_value=3)),
        batch_size=draw(st.integers(min_value=1, max_value=3)),
        lora_rank=draw(st.integers(min_value=1, max_value=4)),
        lora_alpha=draw(st.integers(min_value=1, max_value=16)),
    )
    seed = draw(st.integers(min_value=0, max_value=2**16))
    return seed, params, dataset


# ---------------------------------------------------------------------------
# 单 Cell 的训练（复刻 grid.py 的逐 Cell：set_deterministic_seed → build → train）
# ---------------------------------------------------------------------------


def _train_cell_once(
    seed: int, params: TrainingParams, cell_sub_dataset: list[SftExample]
):
    """在固定种子下为**一个 Cell** 构造微型 LoRA 模型并训练一次。

    复刻 ``trainer.grid.run_grid_training`` 的逐 Cell 步骤：先
    :func:`set_deterministic_seed`(seed) 再构造模型——保证两次调用的初始权重（含
    LoRA 初始化）逐位一致；随后以同一 ``seed`` 调用 ``train_qlora``（其内部亦会
    ``set_deterministic_seed``）在该 Cell 自己的子数据集上、在 CPU 上训练。

    :returns: ``(逐 epoch 损失序列, 仅含适配器张量的 state dict)``。
    """

    set_deterministic_seed(seed)
    model = _build_lora_model(params.lora_rank, params.lora_alpha)
    recorder = _LossRecorder()
    train_qlora(
        model,
        cell_sub_dataset,
        params,
        seed=seed,
        progress_callback=recorder,
        device=torch.device("cpu"),
    )
    return recorder.losses, _clone_adapter_state(model)


def _assert_adapters_equivalent(
    adapter_a: dict[str, torch.Tensor], adapter_b: dict[str, torch.Tensor]
) -> None:
    """断言两份 LoRA 适配器 state dict 语义等价（键集合相同且逐张量 allclose）。"""

    assert set(adapter_a.keys()) == set(adapter_b.keys())
    assert adapter_a, "适配器 state dict 不应为空（应至少含一对 lora_A/lora_B）"
    for key, tensor_a in adapter_a.items():
        tensor_b = adapter_b[key]
        assert tensor_a.shape == tensor_b.shape
        assert torch.allclose(tensor_a, tensor_b, atol=1e-6, rtol=1e-5), (
            f"键 {key!r} 的 LoRA 适配器张量两次训练不一致（同 Cell 确定性被破坏）"
        )


# ===========================================================================
# Property（复用 real-trainer Property 7，按 Cell 行使）：任务 3.5
# ===========================================================================


@pytest.mark.real_training
@given(scenario=_scenarios())
@settings(
    max_examples=12,
    deadline=None,  # 真实训练较慢，禁用单样例截止时间。
    suppress_health_check=[HealthCheck.too_slow],
)
def test_grid_train_is_deterministic_per_cell(scenario) -> None:
    """同一 Cell 在固定 ``(seed, params, sub_dataset)`` 下重复两次训练 → 适配器等价。

    对任意固定的种子、超参与微型 Cell 子数据集，复刻 ``grid.py`` 的逐 Cell 流程
    （``set_deterministic_seed`` → 构造微型 LoRA 模型 → ``train_qlora``）把**同一个
    Cell** 训练两次（CPU），断言：

    * 两次的**逐 epoch 损失序列**长度相同（应等于 ``params.epochs``）且逐值在紧致
      容差内相等；
    * 两次产出的 **LoRA 适配器**张量键集合相同，且每个键对应张量 ``torch.allclose``
      相等——即语义等价的适配器（R3-AC5）。

    这是 real-trainer Property 7 在二维网格语义下、对一个 Cell 的子数据集逐 Cell
    行使：网格训练对每个 Cell 用固定种子重置随机源，故同一 Cell 在相同 Base_Model /
    子数据集 / 参数 / 种子下重复训练必得语义等价的适配器。

    **Feature: ego-layered-adapter (reuses real-trainer Property 7 per cell)**

    **Validates: Requirements 3.5**
    """
    seed, params, cell_sub_dataset = scenario

    losses_a, adapter_a = _train_cell_once(seed, params, cell_sub_dataset)
    losses_b, adapter_b = _train_cell_once(seed, params, cell_sub_dataset)

    # 逐 epoch 损失序列一致（每 epoch 恰一次回调，长度应为 params.epochs）。
    assert len(losses_a) == len(losses_b) == params.epochs
    for la, lb in zip(losses_a, losses_b):
        assert la == pytest.approx(lb, rel=1e-6, abs=1e-6)

    # 两次产出的 LoRA 适配器语义等价（同 Cell 确定性可复现，R3-AC5）。
    _assert_adapters_equivalent(adapter_a, adapter_b)


# ===========================================================================
# 可选的「按 Cell 区分性」断言：不同 Cell（不同子数据集）→ 不同适配器
# ===========================================================================


@pytest.mark.real_training
def test_grid_train_distinct_cells_yield_distinct_adapters() -> None:
    """两个**不同的** Cell（不同子数据集）在**相同种子**下训练 → 产出**不同**适配器。

    本测试与上面的同 Cell 确定性互补：固定 ``_SEED``（复刻 ``grid.py`` 对每个 Cell
    用同一固定种子重置随机源）下，两个拥有**明显不同** token 分布的 Cell 子数据集
    从**逐位相同的初始权重**出发，应学到**不同**的适配器——印证确定性并未把所有
    Cell 退化为同一适配器，各 Cell 仅从其自己的子数据集学习（R3-AC1/AC3/AC5）。

    使用两份手工构造、token 分布明显不同的子数据集（而非 hypothesis 随机数据），
    以避免随机生成偶然得到等价数据集导致的脆弱失败。

    **Feature: ego-layered-adapter (reuses real-trainer Property 7 per cell)**

    **Validates: Requirements 3.5**
    """
    params = _make_params(epochs=3, batch_size=2, lora_rank=4, lora_alpha=16)

    # Kernel Cell 子数据集：token 集中在词表低端。
    kernel_cell = [
        SftExample(input_ids=[1, 2, 3], labels=[IGNORE_INDEX, 2, 3], attention_mask=[1, 1, 1]),
        SftExample(input_ids=[2, 3, 4], labels=[IGNORE_INDEX, 3, 4], attention_mask=[1, 1, 1]),
    ]
    # Zone Cell 子数据集：token 集中在词表高端（与 Kernel 明显不同）。
    zone_cell = [
        SftExample(input_ids=[12, 13, 14], labels=[IGNORE_INDEX, 13, 14], attention_mask=[1, 1, 1]),
        SftExample(input_ids=[13, 14, 15], labels=[IGNORE_INDEX, 14, 15], attention_mask=[1, 1, 1]),
    ]

    # 相同固定种子下分别训练两个 Cell（各自 fresh-load、逐位相同的初始权重）。
    _, kernel_adapter = _train_cell_once(_SEED, params, kernel_cell)
    _, zone_adapter = _train_cell_once(_SEED, params, zone_cell)

    # 键集合相同（同构模型/同超参），但至少一个张量不同（学到不同适配器）。
    assert set(kernel_adapter.keys()) == set(zone_adapter.keys())
    any_different = any(
        not torch.allclose(kernel_adapter[key], zone_adapter[key], atol=1e-6, rtol=1e-5)
        for key in kernel_adapter
    )
    assert any_different, (
        "两个不同子数据集的 Cell 在相同种子下产出了相同适配器；"
        "按 Cell 区分性被破坏（各 Cell 应只从其自己的子数据集学习）"
    )
