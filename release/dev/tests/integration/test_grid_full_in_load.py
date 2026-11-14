"""网格包 Full_In_Load 多 Cell 并存激活集成测试（ego-layered-adapter 任务 8.4，gated）。

本文件实现 tasks.md 任务 8.4「Write integration test for 网格包 Full_In_Load 多
Cell 并存激活（gated）」：以微型设置走通 **PEFT-CPU 真实路径** 的「网格包磁盘落盘
→ ``load_inference_session`` 默认 Full_In_Load 加载 → 多 Cell ``load_adapter`` +
``set_adapter`` 并存激活（Coexist_Activate，method A）→ 确定性生成 → 资源释放」全
链路，验证：

* **默认 Full_In_Load**（R5-AC1）—— 对一个含 Kernel + 一个分区 Cell（``coding``/L1）
  的 Grid_Package，``load_inference_session(selection=None)`` 默认执行 Full_In_Load，
  加载 Kernel 与每个分区到其最大 Level。
* **多 Cell 并存激活**（R5-AC4）—— 经
  :func:`~ego_pod.runtime.inference._activate_cells` 把多个 Cell 适配器以 PEFT 多
  适配器 ``load_adapter`` + ``set_adapter`` **加性并存激活**到同一冻结 Base_Model
  之上，激活顺序为 Kernel → 各分区 Level 低→高；会话的活动适配器集合恰为
  ``[kernel, coding_L1]``。
* **确定性两次一致**（R5-AC8）—— 对同一非空输入、相同会话、相同已加载 Cell 集合
  与相同确定性（贪心、``do_sample=False``）解码设置连续两次调用
  :meth:`BackendInferenceSession.generate`，返回**逐字一致**的回复。
* **会话结束释放全部 Cell 与 Base_Model**（R5-AC9）——
  :meth:`BackendInferenceSession.close` 后会话不再持有任何 Cell 适配器与
  Base_Model 引用（``session.model is None`` 等，活动适配器集合清空）。

依赖门控（design.md「Testing Strategy」/ R6-AC9）
-------------------------------------------------
真实多适配器并存激活需要真实的 ``peft`` / ``transformers`` / ``torch`` 三个依赖。
本模块在**模块顶层**以 :func:`pytest.importorskip` 门控三者（与既有
``tests/integration/test_real_inference_roundtrip.py`` 一致），任一缺失时**整个模块
被干净跳过**（SKIPPED）而非失败/报错。测试函数另以 ``@pytest.mark.real_training``
标记，便于按依赖门控筛选（``-m "not real_training"``）。

本测试**全程在 CPU 上**进行（不要求 CUDA、不加 ``cuda`` 标记）。

无网络（关键）
--------------
为避免真实 ``load_base_model`` 触发 Hugging Face 权重下载，本测试**不**注入真实
:class:`~ego_pod.trainer.peft_backend.PeftCpuBackend`（其 ``load_base_model`` 会尝试
联网拉取 ``qwen3.5-0.8B``）。改为复刻
``test_real_inference_roundtrip.py`` 的离线方案：

1. 用 ``transformers`` 从一个**微型** :class:`~transformers.LlamaConfig` 离线构造
   真实 ``LlamaForCausalLM``（无网络），经 ``peft`` 包裹 LoRA、在极小 SFT 数据集上
   训练若干步，再用真实 :class:`PeftCpuBackend` ``export_lora_safetensors`` 导出
   **仅含适配器**的 ``.safetensors``——对 **每个 Cell**（Kernel 与 ``coding``/L1）
   各导出一个，复用真实导出格式；
2. 经 :func:`~ego_pod.vault.package.write_grid_package` 把这些 Cell 适配器事务性
   落盘为一个真实 **Grid_Package**（``manifest.json`` 含 ``grid`` 描述、磁盘
   ``normal/adapters/kernel.safetensors`` 与 ``normal/adapters/coding/L1.safetensors``、
   ``normal/config.json``），其 ``training_backend == "peft-cpu"``（**非 stub**，从而
   走真实后端推理路径而非 Stub 路径）；
3. 经 :func:`~ego_pod.runtime.inference.load_inference_session` **注入**一个自定义
   后端 :class:`_RoundtripCpuBackend`（``identifier == "peft-cpu"``），其
   ``load_base_model`` 以**相同种子**离线重建同构的 Base_Model（使并存激活作用于
   逐位一致的基座），从而完整复用真实加载/选择/多适配器并存激活代码路径
   （``parse_grid_descriptor`` / ``select_cells`` / ``_activate_cells`` /
   ``generate_reply`` / ``close``）而无需任何网络。

**Feature: ego-layered-adapter**
"""

from __future__ import annotations

import json

import pytest

# -- 真实依赖门控（缺失任一则整个模块 SKIPPED，不失败；R6-AC9）----------------
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

from peft import LoraConfig, get_peft_model  # noqa: E402 - 必在 importorskip 之后
from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from ego_pod.config.schema import TrainingParams  # noqa: E402
from ego_pod.runtime.inference import (  # noqa: E402
    KERNEL_ADAPTER_NAME,
    load_inference_session,
)
from ego_pod.trainer.cells import TrainedCell  # noqa: E402
from ego_pod.trainer.generation import DecodeSettings  # noqa: E402
from ego_pod.trainer.loop import set_deterministic_seed, train_qlora  # noqa: E402
from ego_pod.trainer.peft_backend import PeftCpuBackend  # noqa: E402
from ego_pod.trainer.sft import IGNORE_INDEX, SftExample  # noqa: E402
from ego_pod.vault.package import write_grid_package  # noqa: E402

# ---------------------------------------------------------------------------
# 微型设置常量（保持模型/语料极小，使端到端在依赖存在时也很快）
# ---------------------------------------------------------------------------

#: 微型词表大小（token id 必须 < 该值）。
_VOCAB = 64
#: LoRA 秩与缩放（训练与推理两侧须一致，使适配器张量形状对齐）。
_RANK = 4
_ALPHA = 8
#: 确定性种子：训练与推理两侧均以此重建逐位一致的 Base_Model。
_SEED = 0
#: 推理时的新生成 token 数（远小于 DecodeSettings 默认 256，使生成快速）。
_MAX_NEW_TOKENS = 8

#: 本测试网格包的分区 Cell（``coding`` 分区 Level 1），与 Kernel 并存激活。
_ZONE = "coding"
_LEVEL = 1
#: 该分区 Cell 对应的稳定 PEFT 适配器名（与 ``inference._cell_name`` 约定一致）。
_ZONE_ADAPTER_NAME = f"{_ZONE}_L{_LEVEL}"


def _build_tiny_llama() -> LlamaForCausalLM:
    """离线构造一个**微型**真实 ``LlamaForCausalLM``（无网络）。

    选用 Llama 架构是因为它在 ``peft`` 的
    ``TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING`` 中具默认 LoRA 注入
    目标（``q_proj`` / ``v_proj``），因此训练侧与推理侧均以 ``task_type="CAUSAL_LM"``
    且不显式指定 ``target_modules`` 时，会推断出**相同**的适配器键集合，保证多个
    Cell 适配器的并存激活严格对齐到同一基座的同一组目标模块。

    把 ``generation_config.eos_token_id`` 置空，避免微型随机模型过早生成 EOS 而
    截断生成（确保生成确定性地填满 ``max_new_tokens``）。
    """

    config = LlamaConfig(
        vocab_size=_VOCAB,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
        pad_token_id=0,
    )
    model = LlamaForCausalLM(config)
    # 关闭提前停止，确保贪心生成填满 max_new_tokens（避免空回复，且仍确定性）。
    model.generation_config.eos_token_id = None
    model.generation_config.pad_token_id = 0
    return model


class _CharTokenizer:
    """无依赖的确定性字符级 tokenizer，兼容 ``generate_reply`` 的鸭子类型契约。

    暴露 ``apply_chat_template``（``return_tensors="pt"`` 时返回形状 ``[1, N]`` 的
    ``torch`` 张量，与真实 HF tokenizer 行为一致）与 ``decode``，token id 落在
    ``[1, _VOCAB)`` 区间内（< 模型词表），可被真实 ``LlamaForCausalLM.generate``
    安全消费。
    """

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size

    def apply_chat_template(
        self,
        messages,
        *,
        add_generation_prompt: bool = False,
        return_tensors=None,
        **_,
    ):
        text = "".join(message.get("content", "") for message in messages)
        ids = [(ord(ch) % (self.vocab_size - 1)) + 1 for ch in text] or [1]
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def decode(self, token_ids, *, skip_special_tokens: bool = True, **_) -> str:
        try:
            ids = token_ids.tolist()
        except AttributeError:
            ids = list(token_ids)
        return "".join(f"<{int(i)}>" for i in ids)


class _RoundtripCpuBackend:
    """注入用的 PEFT-CPU 后端替身：``load_base_model`` 离线重建 Base_Model。

    满足 :class:`~ego_pod.trainer.base.TrainingBackend` 的结构（``name`` /
    ``hardware`` / ``identifier`` / ``is_available`` / 三方法），``identifier`` 取
    ``"peft-cpu"`` 与真实 PEFT-CPU 后端一致（R6-AC1）。``load_base_model`` 以**相同
    种子**离线重建与训练时逐位一致的微型 Base_Model 并返回 ``(base_model, tokenizer)``，
    供并存激活路径（:func:`~ego_pod.runtime.inference._activate_cells`）把多个 Cell
    适配器加性并存激活到同一基座之上。
    """

    name = "HuggingFace PEFT"
    hardware = "CPU"
    identifier = "peft-cpu"

    def __init__(self, tokenizer: _CharTokenizer) -> None:
        self._tokenizer = tokenizer

    @classmethod
    def is_available(cls) -> bool:
        return True

    def load_base_model(self, model_id: str, quantization: str):
        # 以相同种子重建 → base 权重与训练侧逐位一致，使并存激活作用于同一基座。
        set_deterministic_seed(_SEED)
        base = _build_tiny_llama()
        return base, self._tokenizer

    def train(self, *args, **kwargs) -> None:  # pragma: no cover - 本测试不调用
        raise NotImplementedError

    def export_lora_safetensors(self, *args, **kwargs) -> None:  # pragma: no cover
        raise NotImplementedError


def _build_micro_dataset() -> list[SftExample]:
    """构造极小的输出-only 掩码 :class:`SftExample` 数据集（token id < _VOCAB）。

    每条样本均保留至少一个未掩码的 output token，使输出-only 交叉熵损失有限、
    对 LoRA 参数产生非零梯度，训练确实更新适配器。
    """

    raw = [
        [5, 6, 7, 8, 9],
        [10, 11, 12, 13],
        [14, 15, 16, 17, 18, 19],
    ]
    dataset: list[SftExample] = []
    for input_ids in raw:
        prompt_len = len(input_ids) // 2  # 前半段为 prompt（掩码），后半段为 output
        labels = [IGNORE_INDEX] * prompt_len + list(input_ids[prompt_len:])
        dataset.append(
            SftExample(
                input_ids=list(input_ids),
                labels=labels,
                attention_mask=[1] * len(input_ids),
            )
        )
    return dataset


def _train_and_export_cell_adapter(adapter_path, tokenizer: _CharTokenizer) -> None:
    """micro train→export：为一个 Cell 训练微型 LoRA 适配器并导出仅含适配器的文件。

    复刻 ``trainer.grid.run_grid_training`` 的逐 Cell 步骤（fresh-load + 固定种子 +
    train + export）：离线构造微型 Base_Model → ``peft`` 包裹 LoRA →
    :func:`train_qlora` 训练若干步 → 真实 :class:`PeftCpuBackend`
    ``export_lora_safetensors`` 导出仅含适配器张量的单个 ``.safetensors``（R3）。

    每个 Cell 都以相同种子从同一微型基座出发、在同一微型数据集上训练，得到形状
    一致、可并存激活到同一基座的 LoRA 适配器（本测试关注并存激活与确定性，不要求
    各 Cell 适配器彼此不同）。
    """

    # 以相同种子构造 base（与推理侧 _RoundtripCpuBackend.load_base_model 对齐）。
    set_deterministic_seed(_SEED)
    base = _build_tiny_llama()
    # 不显式指定 target_modules → 走 Llama 默认（q_proj/v_proj），与推理侧一致。
    peft_model = get_peft_model(
        base,
        LoraConfig(r=_RANK, lora_alpha=_ALPHA, task_type="CAUSAL_LM"),
    )

    params = TrainingParams(
        method="qlora",
        quantization="4bit",
        lora_rank=_RANK,
        lora_alpha=_ALPHA,
        learning_rate=1e-3,
        epochs=1,
        batch_size=2,
    )

    def _noop_progress(*, epoch: int, step: int, loss: float) -> None:
        return None

    train_qlora(
        peft_model,
        _build_micro_dataset(),
        params,
        seed=_SEED,
        progress_callback=_noop_progress,
        device="cpu",
    )

    # 真实后端导出：仅含 LoRA 适配器张量的单个 .safetensors（R3-AC4）。
    PeftCpuBackend().export_lora_safetensors((peft_model, tokenizer), adapter_path)


def _build_grid_package(tmp_path, tokenizer: _CharTokenizer):
    """构造一个真实 Grid_Package（Kernel + coding/L1）并落盘到 ``tmp_path/ego-output``。

    步骤：把两个 Cell 适配器各导出到临时文件，再经 :func:`write_grid_package` 事务性
    落盘为 Grid_Package（``manifest.json`` 含 ``grid``、``training_backend ==
    "peft-cpu"``、磁盘含 ``kernel.safetensors`` 与 ``coding/L1.safetensors``）。

    :returns: Ego_Package 根目录（``tmp_path/ego-output``）。
    """

    # 1. 为每个 Cell 导出一个仅含适配器的 .safetensors 到临时落点（供 Vault 移动）。
    cells_out = tmp_path / ".ego" / "cells_out"
    kernel_tmp = cells_out / "kernel.safetensors"
    zone_tmp = cells_out / _ZONE / f"L{_LEVEL}.safetensors"
    kernel_tmp.parent.mkdir(parents=True, exist_ok=True)
    zone_tmp.parent.mkdir(parents=True, exist_ok=True)

    _train_and_export_cell_adapter(kernel_tmp, tokenizer)
    _train_and_export_cell_adapter(zone_tmp, tokenizer)
    assert kernel_tmp.is_file() and zone_tmp.is_file(), "两个 Cell 适配器应已导出"

    # 2. 事务性落盘为 Grid_Package（training_backend=peft-cpu ⇒ 走真实后端路径）。
    output_root = tmp_path / "ego-output"
    write_grid_package(
        output_root,
        cells=[
            TrainedCell(zone=None, level=0, output_path=kernel_tmp),
            TrainedCell(zone=_ZONE, level=_LEVEL, output_path=zone_tmp),
        ],
        lora_rank=_RANK,
        lora_alpha=_ALPHA,
        training_backend="peft-cpu",
    )
    return output_root


# ===========================================================================
# 集成测试：网格包 Full_In_Load 多 Cell 并存激活 + 确定性 + 资源释放（任务 8.4）
# ===========================================================================


@pytest.mark.real_training
def test_grid_full_in_load_coexist_activate(tmp_path) -> None:
    """网格包默认 Full_In_Load：多 Cell 并存激活、两次回复一致、close 释放全部资源。

    步骤：

    1. 构造一个真实 Grid_Package（Kernel + ``coding``/L1，``training_backend ==
       "peft-cpu"``）落盘到 ``tmp_path/ego-output``；
    2. 注入 ``identifier=="peft-cpu"`` 的离线后端经
       :func:`load_inference_session` 以**默认** ``selection=None`` 加载——默认执行
       Full_In_Load（R5-AC1），经 ``_activate_cells`` 把 Kernel 与 ``coding``/L1 两个
       Cell 适配器以 ``load_adapter`` + ``set_adapter`` **并存激活**到同一冻结基座
       （R5-AC4）；断言会话活动适配器集合恰为 ``[kernel, coding_L1]``（Kernel 在前，
       即 Kernel → 各分区 Level 低→高的顺序）；
    3. 对同一非空输入连续两次 ``generate``，断言两次回复**逐字一致**（确定性贪心
       解码，R5-AC8）；
    4. ``close()`` 后断言会话不再持有 Base_Model / 任何 Cell 适配器引用（R5-AC9）。

    **Feature: ego-layered-adapter**

    **Validates: Requirements 5.1, 5.4, 5.8, 5.9**
    """

    tokenizer = _CharTokenizer(_VOCAB)

    # 1. 构造真实 Grid_Package 并落盘。
    output_root = _build_grid_package(tmp_path, tokenizer)

    # 健全性检查：磁盘布局与 manifest.grid 结构正确（多 Cell 网格包，非 Flat）。
    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["training_backend"] == "peft-cpu", "应走真实后端路径（非 stub）"
    assert "grid" in manifest, "网格包 manifest 应含 grid 键"
    assert (output_root / "normal" / "adapters" / "kernel.safetensors").is_file()
    assert (
        output_root / "normal" / "adapters" / _ZONE / f"L{_LEVEL}.safetensors"
    ).is_file()

    # 2. 默认 Full_In_Load 加载：注入离线后端，多 Cell 并存激活（R5-AC1、R5-AC4）。
    #    selection 缺省（None）⇒ Full_In_Load（Kernel + 每分区 Level 1..max）。
    session = load_inference_session(
        edition="normal",
        cwd=tmp_path,
        backend=_RoundtripCpuBackend(tokenizer),
    )

    # 真实加载路径应已选用注入后端并完成 base + 多 Cell 并存激活（R5-AC4）。
    assert session.model is not None
    assert session.tokenizer is tokenizer

    # 多 Cell 并存激活：两个 Cell 适配器都被挂载，且活动集合为 Kernel 在前的有序集合
    # （Kernel → 各分区 Level 低→高，R5-AC4）。
    assert set(session.model.peft_config.keys()) == {
        KERNEL_ADAPTER_NAME,
        _ZONE_ADAPTER_NAME,
    }, "Kernel 与 coding/L1 两个 Cell 适配器都应被 load_adapter 挂载"
    assert session.active_adapters == [KERNEL_ADAPTER_NAME, _ZONE_ADAPTER_NAME], (
        "Full_In_Load 应把 Kernel 与各分区 Cell 并存激活，顺序为 Kernel 在前"
    )
    # 真实 PEFT 模型的活动适配器集合与会话维护的活动集合一致（加性并存激活）。
    assert list(session.model.active_adapters) == [
        KERNEL_ADAPTER_NAME,
        _ZONE_ADAPTER_NAME,
    ]

    # 用极小的 max_new_tokens 加速生成（仍为确定性贪心解码，R5-AC8）。
    session.decode = DecodeSettings(max_new_tokens=_MAX_NEW_TOKENS)

    # 3. 同一输入两次推理 → 逐字一致（确定性贪心解码，R5-AC8）。
    user_input = "你好，请介绍一下你自己"
    reply_first = session.generate(user_input)
    reply_second = session.generate(user_input)

    assert isinstance(reply_first, str)
    assert reply_first == reply_second, (
        "相同会话/输入/已加载 Cell 集合/确定性解码设置下两次 generate 应逐字一致（R5-AC8）"
    )

    # 4. close() 释放全部 Cell 适配器与 Base_Model 资源（R5-AC9）。
    session.close()
    assert session.model is None
    assert session.tokenizer is None
    assert session.active_adapters == []
