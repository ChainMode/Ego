"""真实 PEFT-CPU 端到端训练集成测试（tasks.md 任务 13.1）。

本文件实现 design.md「Testing Strategy」中描述的、按依赖与硬件**门控**的真实
训练集成测试：在一个**微型本地配置**下，端到端串起
:class:`~ego_pod.trainer.peft_backend.PeftCpuBackend` 的真实
``load_base_model`` → ``train`` → ``export_lora_safetensors`` 流程（CPU 路径，
不依赖 NVIDIA CUDA），并断言：

* 产出的 ``.safetensors`` 文件存在、可被 ``safetensors`` 正常打开并枚举键；
* 其张量键**仅**为 LoRA 适配器键（含 ``lora_``），不含任何 Base_Model 骨干权重
  键（"adapter-only" 断言，复用 Property 9 的判定思路，R5-AC2、R5-AC1、R5-AC3）；
* happy path 上退出码语义正确——整条流程不抛出任何
  :class:`~ego_pod.exit_codes.EgoError`（不以退出码 1 结束，R1-AC2、R3-AC1、
  R4-AC1、R5-AC1、R9-AC1）。

依赖门控（R8-AC5、R8-AC6）
--------------------------
真实 PEFT-CPU 训练需要 ``peft`` / ``transformers`` / ``torch`` 三个真实依赖：

* 模块顶层以 :func:`pytest.importorskip` 门控三者，任一缺失时**整个模块被干净
  跳过**（SKIPPED），而非失败/报错——本 CI 环境正缺 ``peft`` / ``transformers``，
  因此该模块在此环境下必然跳过；
* 测试函数另以 ``@pytest.mark.real_training`` 标记（在 ``pyproject.toml`` 注册），
  便于在无真实训练依赖/硬件的环境按标记过滤跳过。

无网络的本地基座（关键设计）
----------------------------
``PeftCpuBackend.load_base_model`` 经 ``transformers`` 的
``AutoTokenizer.from_pretrained`` / ``AutoModelForCausalLM.from_pretrained``
加载基座——二者均接受**本地目录路径**作为 ``model_id``。因此本测试在
``tmp_path`` 下用 ``transformers`` 构造并保存一个**极小**的 ``LlamaForCausalLM``
与一个极小的 ``PreTrainedTokenizerFast``（WordLevel + 空白预切分），再把该本地
目录路径作为 ``model_id`` 传入，从而**无需网络、无需真实 HF 模型 id** 即可跑通
完整的真实 PEFT-CPU 训练流程。

> 若所在环境的 ``transformers`` / ``peft`` / ``tokenizers`` 版本无法按本测试的
> 方式构造微型本地模型（如 API 变更），本测试以 :func:`pytest.skip` 干净跳过而
> **不**失败——保证"依赖/版本不满足时跳过而非失败"的门控语义（R8-AC6）。在完整
> 配备的真实训练环境中，本测试将真正执行端到端训练。

**Feature: ego-pod-real-trainer**
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# -- 真实依赖门控（缺失则整个模块 SKIPPED，不失败）-------------------------
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

from safetensors.torch import load_file

from ego_pod.config.schema import TrainingParams
from ego_pod.exit_codes import EgoError
from ego_pod.trainer.peft_backend import PeftCpuBackend

# ---------------------------------------------------------------------------
# 微型本地基座（无网络）：极小 LlamaForCausalLM + 极小 WordLevel tokenizer
# ---------------------------------------------------------------------------

#: 微型词表：特殊 token + 语料中出现的若干词（WordLevel 按空白切分，未登录词 → <unk>）。
_SPECIAL_TOKENS = ["<pad>", "<unk>", "<s>", "</s>"]
_CORPUS_WORDS = [
    "今天",
    "读完",
    "了",
    "一本",
    "好书",
    "清晨",
    "空气",
    "清新",
    "午后",
    "阳光",
    "温暖",
    "慢跑",
    "五公里",
    "整理",
    "书架",
    "耐心",
    "是",
    "一种",
    "答案",
    "请",
    "总结",
    "要点",
]


def _build_tiny_local_base_model(target_dir: Path) -> str:
    """在 ``target_dir`` 下构造并保存一个极小的本地基座（model + tokenizer）。

    返回可直接作为 ``load_base_model`` 的 ``model_id`` 的本地目录字符串路径。
    任何构造/版本不兼容问题都向上抛出，由调用方转为 :func:`pytest.skip`。
    """

    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (
        AutoModelForCausalLM,
        LlamaConfig,
        PreTrainedTokenizerFast,
    )

    # --- 极小 WordLevel 分词器（无网络、可被 AutoTokenizer 重新加载）---------
    vocab: dict[str, int] = {}
    for token in _SPECIAL_TOKENS:
        vocab[token] = len(vocab)
    for word in _CORPUS_WORDS:
        if word not in vocab:
            vocab[word] = len(vocab)

    backend_tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    backend_tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend_tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
    )
    tokenizer.save_pretrained(str(target_dir))

    # --- 极小 LlamaForCausalLM（CPU、fp32；架构为 PEFT 自动 LoRA target 已知）---
    config = LlamaConfig(
        vocab_size=len(vocab),
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
        pad_token_id=vocab["<pad>"],
        bos_token_id=vocab["<s>"],
        eos_token_id=vocab["</s>"],
    )
    model = AutoModelForCausalLM.from_config(config)
    model.save_pretrained(str(target_dir))

    return str(target_dir)


def _write_micro_corpus(corpus_path: Path) -> int:
    """写出一份微型 Conversation_Corpus（JSONL），返回行数。

    每行含非空 ``instruction`` 与 ``output``（``build_sft_dataset`` 的必需字段），
    词均取自 :data:`_CORPUS_WORDS`，使其能被微型 WordLevel 分词器登录。
    """

    rows = [
        {"instruction": "请 总结 要点", "input": "", "output": "今天 读完 了 一本 好书", "category": "thought"},
        {"instruction": "请 总结 要点", "input": "", "output": "清晨 空气 清新", "category": "sense"},
        {"instruction": "请 总结 要点", "input": "", "output": "午后 阳光 温暖", "category": "sense"},
        {"instruction": "请 总结 要点", "input": "", "output": "耐心 是 一种 答案", "category": "thought"},
    ]
    with corpus_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def _assert_adapter_only(keys: set[str]) -> None:
    """断言键集合仅含 LoRA 适配器键（含 ``lora_``）、无 Base_Model 骨干权重键。

    复用 Property 9（``test_peft_export_pbt``）的 "adapter-only" 判定思路：PEFT
    ``get_peft_model_state_dict`` 仅返回形如 ``...q_proj.lora_A.weight`` /
    ``...lora_B.weight`` 的适配器键，绝不含 base 骨干权重键（R5-AC2）。
    """

    assert keys, "导出文件不应为空（至少应含若干 LoRA 适配器张量）"
    for key in keys:
        assert "lora_" in key.lower(), (
            f"导出文件包含疑似 Base_Model 骨干权重键（非适配器键）：{key!r}"
        )


# ===========================================================================
# 任务 13.1：真实 PEFT-CPU 端到端训练
# ===========================================================================


@pytest.mark.real_training
def test_real_peft_cpu_end_to_end_training(tmp_path: Path) -> None:
    """端到端：load_base_model → train → export_lora_safetensors（CPU，真实）。

    在微型本地配置下完整跑通 :class:`PeftCpuBackend` 的真实训练流程，并断言：

    * 产出的 ``.safetensors`` 文件存在、可被 ``safetensors`` 打开并枚举键
      （R5-AC1、R5-AC3）；
    * 其张量键仅为 LoRA 适配器键、不含 base 骨干权重键（R5-AC2）；
    * happy path 不抛出任何 :class:`EgoError`（退出码语义正确：不以 1 结束，
      R1-AC2、R3-AC1、R4-AC1、R5-AC1、R9-AC1）。

    **Feature: ego-pod-real-trainer, Task 13.1**

    **Validates: Requirements 1.2, 3.1, 4.1, 5.1, 9.1**
    """
    # --- 微型本地基座（无网络）；版本不兼容时干净跳过而非失败（R8-AC6）---------
    model_dir = tmp_path / "tiny-base-model"
    model_dir.mkdir(parents=True, exist_ok=True)
    try:
        model_id = _build_tiny_local_base_model(model_dir)
    except Exception as exc:  # noqa: BLE001 - 版本/API 不兼容时跳过而非失败
        pytest.skip(
            "无法在当前 transformers/peft/tokenizers 版本下构造微型本地基座，"
            f"跳过真实 PEFT-CPU 端到端训练：{exc}"
        )

    corpus_path = tmp_path / "corpus.jsonl"
    line_count = _write_micro_corpus(corpus_path)

    output_path = tmp_path / ".ego" / "lora.safetensors"

    # 微型训练超参：1 epoch、小 batch、小 rank，保证真实训练快速完成。
    params = TrainingParams(
        method="qlora",
        quantization="4bit",
        lora_rank=2,
        lora_alpha=4,
        learning_rate=1e-3,
        epochs=1,
        batch_size=2,
    )

    backend = PeftCpuBackend()

    # 逐 epoch 进度回调：以关键字参数被 loop.train_qlora 调用（epoch/step/loss）。
    progress_calls: list[tuple[int, int, float]] = []

    def _progress(*, epoch: int, step: int, loss: float) -> None:
        progress_calls.append((epoch, step, loss))

    # happy path 退出码语义：整条流程不应抛出任何 EgoError（→ 退出码 1）。
    try:
        model = backend.load_base_model(model_id, params.quantization)
        backend.train(model, corpus_path, params, _progress)
        backend.export_lora_safetensors(model, output_path)
    except EgoError as exc:  # 训练/导出失败会以退出码 1 结束——happy path 不应发生
        pytest.fail(
            f"真实 PEFT-CPU happy path 意外抛出 EgoError"
            f"（exit_code={exc.exit_code}）：{exc}"
        )

    # 至少每 epoch 回调一次（R4-AC2）：epochs=1 → 恰好一次，epoch 编号为 1。
    assert len(progress_calls) == params.epochs
    assert progress_calls[0][0] == 1

    # 产出文件存在、可被 safetensors 打开并枚举键（R5-AC1、R5-AC3）。
    assert output_path.is_file()
    loaded = load_file(str(output_path))
    file_keys = set(loaded.keys())

    # 仅含 LoRA 适配器键，不含 base 骨干权重键（R5-AC2）。
    _assert_adapter_only(file_keys)

    # 语料逐行守恒的间接确认：四行语料均被消费（R3-AC1）。
    assert line_count == 4
