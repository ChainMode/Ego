"""真实推理「导出—加载往返」集成测试（tasks.md 任务 13.2）。

本文件实现 tasks.md 任务 13.2「Write integration test for 真实推理导出—加载
往返」：以微型设置走通 **PEFT-CPU 真实路径** 的 *export → load → generate*
全链路，验证

* **导出—加载往返**（R5-AC4、R9-AC3）—— 由 :class:`PeftCpuBackend` 经
  ``export_lora_safetensors`` 导出的 ``.safetensors`` 适配器，能被同后端的真实
  推理路径（:func:`ego_pod.runtime.inference._apply_lora_adapter` →
  :func:`peft.set_peft_model_state_dict`）成功加载并应用到对应 Base_Model；
* **确定性真实推理**（R6-AC1、R6-AC2、R6-AC3）—— 对同一非空输入在相同会话与
  确定性（贪心、``do_sample=False``）解码设置下连续两次调用
  :meth:`BackendInferenceSession.generate`，返回**逐字一致**的回复；
* **资源释放**（R6-AC5）—— :meth:`BackendInferenceSession.close` 后会话不再持有
  Base_Model / 适配器引用（``session.model is None`` 等）。

依赖门控（design.md「Testing Strategy」/ R8-AC5、R8-AC6）
--------------------------------------------------------
真实推理需要真实的 ``peft`` / ``transformers`` / ``torch`` 三个依赖。本模块在
**模块顶层**以 :func:`pytest.importorskip` 门控三者，任一缺失时**整个模块被干净
跳过**（SKIPPED）而非失败/报错——本仓库默认 CI **未安装 peft/transformers**，
故本模块在此环境下应被整体跳过。测试函数另以 ``@pytest.mark.real_training`` 标记，
便于按依赖门控筛选（``-m "not real_training"``）。

本测试**全程在 CPU 上**进行（不要求 CUDA、不加 ``cuda`` 标记）。

无网络（关键）
--------------
为避免真实 ``load_base_model`` 触发 Hugging Face 权重下载，本测试**不**注入真实
:class:`~ego_pod.trainer.peft_backend.PeftCpuBackend`（其 ``load_base_model`` 会尝试
``AutoModelForCausalLM.from_pretrained("qwen3.5-0.8B")`` 而需要网络）。改为：

1. 用 ``transformers`` 从一个**微型** :class:`~transformers.LlamaConfig` 离线构造
   一个真实的 ``LlamaForCausalLM``（无网络），经 ``peft`` 包裹 LoRA 适配器、在
   极小的 :class:`~ego_pod.trainer.sft.SftExample` 数据集上经
   :func:`~ego_pod.trainer.loop.train_qlora` 训练若干步，再用真实的
   :class:`PeftCpuBackend` ``export_lora_safetensors`` 导出**仅含适配器**的
   ``.safetensors``（复用任务 13.1 的 micro train→export 流程）；
2. 向 :class:`BackendInferenceSession` **注入**一个自定义后端
   :class:`_RoundtripCpuBackend`（``identifier == "peft-cpu"``），其
   ``load_base_model`` 以**相同种子**离线重建同构的 Base_Model（使往返作用于
   逐位一致的基座），从而完整复用真实推理代码路径
   （``_apply_lora_adapter`` / ``generate_reply`` / ``close``）而无需任何网络。

**Feature: ego-pod-real-trainer**
"""

from __future__ import annotations

import pytest

# -- 真实依赖门控（缺失任一则整个模块 SKIPPED，不失败；R8-AC5、R8-AC6）--------
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

from peft import LoraConfig, get_peft_model  # noqa: E402 - 必在 importorskip 之后
from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from ego_pod.config.schema import TrainingParams  # noqa: E402
from ego_pod.runtime.inference import BackendInferenceSession  # noqa: E402
from ego_pod.trainer.generation import DecodeSettings  # noqa: E402
from ego_pod.trainer.loop import set_deterministic_seed, train_qlora  # noqa: E402
from ego_pod.trainer.peft_backend import PeftCpuBackend  # noqa: E402
from ego_pod.trainer.sft import IGNORE_INDEX, SftExample  # noqa: E402

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


def _build_tiny_llama() -> LlamaForCausalLM:
    """离线构造一个**微型**真实 ``LlamaForCausalLM``（无网络）。

    选用 Llama 架构是因为它在 ``peft`` 的
    ``TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING`` 中具默认 LoRA 注入
    目标（``q_proj`` / ``v_proj``），因此训练侧与推理侧均以 ``task_type="CAUSAL_LM"``
    且不显式指定 ``target_modules`` 时，会推断出**相同**的适配器键集合，保证
    ``set_peft_model_state_dict`` 的往返加载严格对齐。

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
    ``"peft-cpu"`` 与真实 PEFT-CPU 后端一致（R6-AC1、R7-AC5）。``load_base_model``
    以**相同种子**离线重建与训练时逐位一致的微型 Base_Model 并返回
    ``(base_model, tokenizer)``，供推理路径的 :func:`_apply_lora_adapter` 包裹
    LoRA 并经 :func:`peft.set_peft_model_state_dict` 灌入导出的适配器权重。
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
        # 以相同种子重建 → base 权重与训练侧逐位一致，使往返作用于同一基座。
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


def _train_and_export_adapter(adapter_path) -> _CharTokenizer:
    """micro train→export：训练微型 LoRA 适配器并导出仅含适配器的 ``.safetensors``。

    复用任务 13.1 的 PEFT-CPU 端到端流程：离线构造微型 Base_Model → ``peft`` 包裹
    LoRA → :func:`train_qlora` 训练若干步 → 真实 :class:`PeftCpuBackend`
    ``export_lora_safetensors`` 导出（R5）。返回供推理侧复用的 tokenizer。
    """

    tokenizer = _CharTokenizer(_VOCAB)

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

    # 真实后端导出：仅含 LoRA 适配器张量的单个 .safetensors（R5-AC1/AC2/AC3）。
    PeftCpuBackend().export_lora_safetensors((peft_model, tokenizer), adapter_path)
    return tokenizer


# ===========================================================================
# 集成测试：真实推理导出—加载往返 + 确定性解码 + 资源释放（任务 13.2）
# ===========================================================================


@pytest.mark.real_training
def test_real_inference_export_load_roundtrip(tmp_path) -> None:
    """export→load→generate 全链路：往返加载成功、两次推理一致、close 释放资源。

    步骤：

    1. 经 PEFT-CPU 真实路径训练并导出微型 LoRA 适配器到 ``.safetensors``；
    2. 注入 ``identifier=="peft-cpu"`` 的后端构造 :class:`BackendInferenceSession`
       ——构造过程经 :func:`_apply_lora_adapter` 用 :func:`peft.set_peft_model_state_dict`
       加载该适配器，**构造不抛异常即证明导出—加载往返成功**（R5-AC4、R9-AC3）；
    3. 对同一非空输入连续两次 ``generate``，断言两次回复**逐字一致**（确定性贪心
       解码，R6-AC1/AC2/AC3）；
    4. ``close()`` 后断言会话不再持有 Base_Model / 适配器引用（R6-AC5）。

    **Feature: ego-pod-real-trainer**

    **Validates: Requirements 5.4, 6.1, 6.2, 6.3, 9.3**
    """

    # 1. 训练并导出适配器（micro train→export，R5）。
    adapter_path = tmp_path / "ego-lora.safetensors"
    tokenizer = _train_and_export_adapter(adapter_path)
    assert adapter_path.is_file(), "导出的 LoRA 适配器 .safetensors 应已写出"

    # 2. 构造真实推理会话：注入 peft-cpu 后端，加载 base + 应用导出的适配器。
    #    构造成功即证明导出—加载往返成功（R5-AC4、R9-AC3）。
    session = BackendInferenceSession(
        base_model="qwen3.5-0.8B",
        quantization="4bit",
        lora_rank=_RANK,
        lora_alpha=_ALPHA,
        adapter_path=adapter_path,
        identifier="peft-cpu",
        backend=_RoundtripCpuBackend(tokenizer),
    )

    # 真实加载路径应已选用注入后端并完成 base + adapter 装载（R6-AC1、R6-AC2）。
    assert session.model is not None
    assert session.tokenizer is tokenizer

    # 用极小的 max_new_tokens 加速生成（仍为确定性贪心解码，R6-AC3）。
    session.decode = DecodeSettings(max_new_tokens=_MAX_NEW_TOKENS)

    # 3. 同一输入两次推理 → 逐字一致（确定性贪心解码，R6-AC3）。
    user_input = "你好，请介绍一下你自己"
    reply_first = session.generate(user_input)
    reply_second = session.generate(user_input)

    assert isinstance(reply_first, str)
    assert reply_first == reply_second, (
        "相同会话/输入/确定性解码设置下两次 generate 的回复应逐字一致（R6-AC3）"
    )

    # 4. close() 释放 Base_Model 与 LoRA_Adapter 资源（R6-AC5）。
    session.close()
    assert session.model is None
    assert session.tokenizer is None
    assert session.adapter_tensors is None
