"""HuggingFace PEFT 训练后端（CUDA 与 CPU 两个变体）。

本模块提供两个满足 :class:`ego_pod.trainer.base.TrainingBackend` Protocol 的
后端实现：

- :class:`PeftCudaBackend`：要求 ``peft`` / ``transformers`` / ``torch`` 可导入
  且检测到 NVIDIA CUDA 设备，对应 ``detect`` 优先级中的第 2 位。
- :class:`PeftCpuBackend`：仅要求 ``peft`` / ``transformers`` / ``torch`` 可导入
  （不要求 CUDA），对应 ``detect`` 优先级中的第 4 位，作为 Stub 之前的最后
  一个"真实"后端回退。

两者共享同一套基于 PEFT 的实现 :class:`_PeftBackendBase`，仅在 ``name`` /
``hardware`` / :meth:`is_available` 三处有所不同。

设计约束（见设计文档「多后端 Trainer 抽象」一节）：

- **顶层导入必须永不抛错**。所有可选依赖（``peft`` / ``transformers`` /
  ``torch`` / ``bitsandbytes``）均通过 ``try/except ImportError`` 包裹为模块级
  哨兵，配合函数内惰性导入，使 ``detect`` 能够防御式地无条件 ``import`` 本
  模块，即便对应依赖缺失（R8-AC4）。
- :meth:`_PeftBackendBase.load_base_model` 已落地真实加载：CUDA 且
  ``bitsandbytes`` 可用时以 4-bit NF4 量化加载（QLoRA），否则以 CPU 受支持
  精度回退加载（R1）。:meth:`train` 已落地真实 QLoRA 训练：构造/复用 LoRA
  适配器 → 流式构建 SFT_Dataset → 调用框架无关的训练循环 → 打印训练完成摘要
  （R3、R4）。:meth:`export_lora_safetensors` 已落地仅含适配器的 SafeTensors
  导出：经 ``peft.get_peft_model_state_dict`` 取出 LoRA 适配器张量（绝不含
  base 骨干）→ 转为连续 CPU 张量 → ``safetensors.torch.save_file`` 写出单个
  合法 ``.safetensors`` 文件（R5）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ego_pod.config.schema import TrainingParams
from ego_pod.exit_codes import EgoError, TrainerError
from ego_pod.logging_io import print_summary
from ego_pod.trainer import loop as _loop
from ego_pod.trainer import sft as _sft
from ego_pod.trainer.base import BackendIdentifier, Hardware, ProgressCallback

__all__ = [
    "PeftCpuBackend",
    "PeftCudaBackend",
]


# --- 可选依赖探测 -----------------------------------------------------------
# 以下导入全部包裹在 try/except 中，确保本模块在缺失任一依赖时仍可被安全
# 导入（顶层 import 永不抛错），从而允许 detect 模块防御式地导入本模块。

try:  # pragma: no cover - 依赖是否存在取决于运行环境
    import torch as _torch
except ImportError:  # pragma: no cover
    _torch = None

try:  # pragma: no cover
    import peft as _peft
except ImportError:  # pragma: no cover
    _peft = None

try:  # pragma: no cover
    import transformers as _transformers
except ImportError:  # pragma: no cover
    _transformers = None

try:  # pragma: no cover - 仅在 NVIDIA CUDA 平台通常可安装
    import bitsandbytes as _bitsandbytes
except ImportError:  # pragma: no cover
    _bitsandbytes = None


def _peft_importable() -> bool:
    """判定 HuggingFace PEFT 训练栈是否可导入。

    需要 ``peft``、``transformers`` 与 ``torch`` 三者同时可用，对应设计文档
    探测表中"import peft, torch 成功"一栏（``transformers`` 为 PEFT 训练所
    必需，一并纳入）。

    .. note::
        ``bitsandbytes`` **不**纳入本判定：它仅在 CUDA 上提供 4-bit 量化，
        在 macOS / 纯 CPU 平台往往无法安装。PEFT 的可用性（含 CPU 变体）只要求
        peft / transformers / torch；``bitsandbytes`` 缺失时
        :meth:`_PeftBackendBase.load_base_model` 走 CPU 精度回退（R1-AC3）。
    """

    return _peft is not None and _transformers is not None and _torch is not None


def _bitsandbytes_available() -> bool:
    """判定 ``bitsandbytes`` 4-bit 量化是否可用（R1-AC1、R1-AC3）。

    仅当 ``bitsandbytes`` 模块可导入时返回 ``True``。该哨兵**不**参与
    :meth:`PeftCudaBackend.is_available` / :meth:`PeftCpuBackend.is_available`
    的判定（PEFT 可用性仍只要求 peft/transformers/torch），仅用于
    :meth:`_PeftBackendBase.load_base_model` 内部决定走 4-bit 量化路径还是
    CPU 精度回退路径。
    """

    return _bitsandbytes is not None


def _torch_cuda_available() -> bool:
    """判定当前环境是否存在可用的 NVIDIA CUDA 设备。

    在 ``torch`` 缺失或 ``torch.cuda`` 探测过程中发生任何异常时一律返回
    ``False``，确保探测过程不会因环境问题而抛错。
    """

    if _torch is None:
        return False
    try:
        return bool(_torch.cuda.is_available())
    except Exception:  # pragma: no cover - 防御性兜底
        return False


def _cpu_bf16_supported() -> bool:
    """尽力判定当前 CPU 是否支持 ``bfloat16`` 加载精度（R1-AC3）。

    ``torch`` 缺失或探测过程中发生任何异常时一律返回 ``False``，从而保守回退到
    ``float32`` 这一最广受支持的精度。
    """

    if _torch is None:
        return False
    try:
        return bool(_torch.cpu._is_avx512_bf16_supported())  # type: ignore[attr-defined]
    except Exception:
        # 该私有探测在不同 torch 版本/平台上可能不存在；保守回退到 fp32。
        return False


def _dtype_name(dtype: Any) -> str:
    """把 torch dtype 渲染为简洁可读名（如 ``torch.float32`` → ``fp32``）。

    仅用于加载摘要的人类可读输出（R1-AC4），无法识别时回退到 ``str(dtype)``。
    """

    text = str(dtype).rsplit(".", 1)[-1]
    return {
        "float32": "fp32",
        "float16": "fp16",
        "bfloat16": "bf16",
    }.get(text, text)


def _unpack_model(model: Any) -> tuple[Any, Any]:
    """把 :meth:`_PeftBackendBase.load_base_model` 的复合返回拆为 ``(model_obj, tokenizer)``。

    :meth:`load_base_model` 返回 ``(peft_model_or_base, tokenizer)`` 二元组；本
    辅助负责稳健解包。对非二元组形态（防御性）原样返回 ``(model, None)``。
    """

    if isinstance(model, tuple) and len(model) == 2:
        return model[0], model[1]
    return model, None


def _is_peft_model(model_obj: Any) -> bool:
    """判定 ``model_obj`` 是否已是一个 PEFT 适配器模型。

    优先用 ``isinstance(model_obj, peft.PeftModel)``；``peft`` 不可导入或类型
    探测异常时回退到鸭子类型检查（是否具备 ``peft_config`` 属性）。
    """

    if _peft is not None:
        peft_model_cls = getattr(_peft, "PeftModel", None)
        if peft_model_cls is not None:
            try:
                if isinstance(model_obj, peft_model_cls):
                    return True
            except Exception:  # pragma: no cover - 防御性兜底
                pass
    return hasattr(model_obj, "peft_config")


def _resolve_device(model_obj: Any) -> Any:
    """从模型对象推断其所在设备（供 :func:`loop.train_qlora` 移动 batch 用）。

    取模型首个参数的 ``device``；无法取得（如假模型或无参数）时返回 ``None``，
    此时训练循环不移动张量。
    """

    parameters = getattr(model_obj, "parameters", None)
    if not callable(parameters):
        return None
    try:
        first = next(iter(parameters()))
    except (StopIteration, TypeError):
        return None
    return getattr(first, "device", None)


# --- 共享实现 ---------------------------------------------------------------


class _PeftBackendBase:
    """基于 HuggingFace PEFT 的共享后端实现。

    :class:`PeftCudaBackend` 与 :class:`PeftCpuBackend` 均继承本类并复用其
    训练方法实现；:class:`ego_pod.trainer.unsloth_backend.UnslothBackend`
    亦通过组合本类的实例委托其训练方法。

    :meth:`load_base_model` 已落地真实加载（R1）：CUDA 且 ``bitsandbytes`` 可用
    时以 4-bit NF4 量化加载基座并 ``prepare_model_for_kbit_training``；CPU 或
    ``bitsandbytes`` 不可用时以受支持精度（fp32 / bf16）加载且不使用 4-bit，并在
    标准输出记录实际加载精度。:meth:`train` 已落地真实 QLoRA 训练（R3、R4）：
    构造/复用 LoRA 适配器、流式构建 SFT_Dataset、经框架无关的训练循环逐 epoch
    回调并仅更新适配器参数、打印训练完成摘要。
    :meth:`export_lora_safetensors` 已落地仅含适配器的 SafeTensors 导出（R5）：
    经 ``peft.get_peft_model_state_dict`` 取出 LoRA 适配器张量（绝不含 base 骨干）
    → 转为连续 CPU 张量 → ``safetensors.torch.save_file`` 写出单个合法
    ``.safetensors`` 文件，供 Vault 持久化。
    """

    #: 子类覆盖：后端的可读名称。
    name: str = "HuggingFace PEFT"
    #: 子类覆盖：后端运行的硬件类型。
    hardware: Hardware = "CPU"
    #: 子类覆盖：稳定区分各后端的 Backend_Identifier（R7-AC1、R7-AC4）。
    identifier: BackendIdentifier = "peft-cpu"

    #: SFT_Dataset 的最大序列长度（供 :meth:`train` 截断使用，见任务 5.3）。
    _MAX_SEQ_LEN = 2048
    #: 训练/加载的确定性种子（见任务 5.3）。
    _SEED = 0

    def load_base_model(self, model_id: str, quantization: str) -> Any:
        """加载 Base_Model 并按 QLoRA 要求量化（R1）。

        加载策略（R1-AC1 / AC2 / AC3）：

        - **CUDA 且 ``bitsandbytes`` 可用**：构造
          ``BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
          bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=bf16/fp16)``
          加载基座，并 ``prepare_model_for_kbit_training`` 以便后续 QLoRA 训练。
        - **CPU 或 ``bitsandbytes`` 不可用**：以 CPU 受支持精度（优先 ``bf16``，
          否则 ``fp32``）加载，不使用 4-bit 量化（R1-AC2、R1-AC3）。

        加载成功后在标准输出打印一行包含 ``model_id`` / 目标设备（CUDA|CPU）/
        实际量化或精度方案的加载摘要（R1-AC4），并返回 ``(peft_model, tokenizer)``
        复合对象，供 :meth:`train` 与 :meth:`export_lora_safetensors` 使用。

        本方法**不**在此配置 ``LoraConfig`` / ``get_peft_model``：LoRA 适配器的
        秩（``lora_rank``）与缩放（``lora_alpha``）来自 :meth:`train` 的 ``params``，
        故 LoRA 的配置推迟到 :meth:`train` 内完成（见设计 §5）。

        所有可选依赖（``torch`` / ``transformers`` / ``peft`` / ``bitsandbytes``）
        在本方法内惰性导入，保证模块顶层 ``import`` 永不因缺依赖而失败（R8-AC4）。

        :param model_id: 基座模型标识，MVP 固定为 ``qwen3.5-0.8B``。
        :param quantization: 量化方案，MVP 固定为 ``4bit``（在 CPU/无 bnb 时被
            精度回退覆盖）。
        :returns: ``(peft_model, tokenizer)`` 复合对象。
        :raises TrainerError: 加载因权重缺失、依赖缺失或显存/内存不足而失败时，
            其消息包含 ``model_id`` 与具体失败原因（R1-AC5）。
        """

        use_cuda = self.hardware == "CUDA" and _torch_cuda_available()
        use_4bit = use_cuda and _bitsandbytes_available()

        try:
            # 惰性导入：保证模块顶层 import 永不因缺依赖而失败（R8-AC4）。
            import torch  # noqa: PLC0415
            from transformers import (  # noqa: PLC0415
                AutoModelForCausalLM,
                AutoTokenizer,
            )

            tokenizer = AutoTokenizer.from_pretrained(model_id)

            if use_4bit:
                from transformers import BitsAndBytesConfig  # noqa: PLC0415
                from peft import prepare_model_for_kbit_training  # noqa: PLC0415

                compute_dtype = (
                    torch.bfloat16
                    if torch.cuda.is_bf16_supported()
                    else torch.float16
                )
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    quantization_config=bnb_config,
                    device_map={"": 0},
                )
                model = prepare_model_for_kbit_training(model)
                quant_desc = f"4bit-nf4 (compute_dtype={_dtype_name(compute_dtype)})"
                device_desc = "CUDA"
            else:
                # CPU / 无 bnb 精度回退：优先 bf16（若支持），否则 fp32（R1-AC2、R1-AC3）。
                load_dtype = torch.bfloat16 if _cpu_bf16_supported() else torch.float32
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=load_dtype,
                )
                if use_cuda:
                    model = model.to("cuda")
                    device_desc = "CUDA"
                else:
                    device_desc = "CPU"
                quant_desc = f"none (precision={_dtype_name(load_dtype)})"
        except EgoError:
            # OfflineMissingWeightsError / 已构造的 TrainerError 等原样透传，
            # 保持各自退出码语义不被二次包装（与 pipeline 的处理一致）。
            raise
        except Exception as exc:
            raise TrainerError(
                f"Base_Model 加载失败（model_id={model_id!r}）：{exc}"
            ) from exc

        # 加载摘要：model_id / 设备 / 实际量化或精度（R1-AC4）。
        print_summary(
            f"Loaded base model: model_id={model_id} "
            f"device={device_desc} quantization={quant_desc}"
        )
        return model, tokenizer

    def train(
        self,
        model: Any,
        corpus_path: Path,
        params: TrainingParams,
        progress_callback: ProgressCallback,
    ) -> None:
        """在给定语料上训练 LoRA 适配器（R3 + R4）。

        步骤（见设计 §5「``_PeftBackendBase.train``」）：

        1. 解包 :meth:`load_base_model` 返回的 ``(model_obj, tokenizer)`` 复合对象；
           若 ``model_obj`` 尚未配置 LoRA 适配器（加载阶段未 ``get_peft_model``），
           则用 ``params.lora_rank`` / ``params.lora_alpha`` 构造 ``LoraConfig`` 并
           ``peft.get_peft_model`` 应用 LoRA（``task_type=CAUSAL_LM``，R4-AC1）；
        2. ``dataset = sft.build_sft_dataset(corpus_path, tokenizer, self._MAX_SEQ_LEN)``
           —— 流式构建带输出-only 掩码的 SFT 样本（R3）；
        3. ``metrics = loop.train_qlora(peft_model, dataset, params, seed=self._SEED,
           progress_callback=progress_callback, device=...)`` —— 逐 epoch 回调、
           仅更新 LoRA 适配器、确定性种子；OOM/NaN 由 ``loop`` 内部已包装为
           :class:`TrainerError`，原样上抛（R4-AC2/AC3/AC5/AC7）；
        4. 训练完成后在标准输出打印一行含已完成 epoch 数、累计训练步数与最终
           训练损失的训练完成摘要（R4-AC4）。

        所有可选依赖（``peft`` / ``torch``）在本方法内惰性导入，保证模块顶层
        ``import`` 永不因缺依赖而失败（R8-AC4）。

        :param model: :meth:`load_base_model` 返回的 ``(model_obj, tokenizer)``
            复合对象。
        :param corpus_path: Conversation_Corpus（JSONL）中间文件路径。
        :param params: 训练参数，提供 ``lora_rank`` / ``lora_alpha`` /
            ``learning_rate`` / ``epochs`` / ``batch_size``。
        :param progress_callback: 进度回调，至少每 epoch 调用一次（R4-AC2）。
        :raises TrainerError: 语料解析失败（R3-AC4/AC5）、训练数值发散或 OOM
            （R4-AC7）等，均由下游 ``sft`` / ``loop`` 抛出并原样上抛。
        """

        model_obj, tokenizer = _unpack_model(model)

        try:
            # 若加载阶段未配置 LoRA，则在此用 params 构造 LoraConfig 并应用（R4-AC1）。
            if not _is_peft_model(model_obj):
                from peft import (  # noqa: PLC0415 - 惰性导入保证导入安全
                    LoraConfig,
                    get_peft_model,
                )

                lora_config = LoraConfig(
                    r=params.lora_rank,
                    lora_alpha=params.lora_alpha,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                model_obj = get_peft_model(model_obj, lora_config)
        except EgoError:
            # 已构造的 TrainerError 等原样透传，保持退出码语义不被二次包装。
            raise

        device = _resolve_device(model_obj)

        # 流式构建 SFT_Dataset（非法行/空语料 → TrainerError，R3）。
        dataset = _sft.build_sft_dataset(
            corpus_path, tokenizer, self._MAX_SEQ_LEN
        )

        # 执行 QLoRA 训练循环；OOM/NaN 已由 loop 包装为 TrainerError，原样上抛。
        metrics = _loop.train_qlora(
            model_obj,
            dataset,
            params,
            seed=self._SEED,
            progress_callback=progress_callback,
            device=device,
        )

        # 训练完成摘要：epochs / steps / final_loss（R4-AC4）。
        print_summary(
            f"Training complete: epochs={metrics.completed_epochs} "
            f"steps={metrics.total_steps} final_loss={metrics.final_loss:.6f}"
        )

    def export_lora_safetensors(self, model: Any, output_path: Path) -> None:
        """以 SafeTensors 格式仅导出 LoRA_Adapter 权重（R5）。

        步骤（见设计 §5「``_PeftBackendBase.export_lora_safetensors``」）：

        1. 解包 :meth:`load_base_model`/:meth:`train` 流转的
           ``(model_obj, tokenizer)`` 复合对象，取出已带 LoRA 适配器的 PEFT
           模型；
        2. ``adapter_state = peft.get_peft_model_state_dict(model_obj)`` ——
           **仅**含 LoRA 适配器张量，绝不含 Base_Model 骨干权重（R5-AC2）；
        3. 把每个张量转为连续（contiguous）CPU 张量，确保
           ``safetensors`` 可安全序列化（避免非连续/非 CPU 张量导致的写入失败）；
        4. ``output_path.parent.mkdir(parents=True, exist_ok=True)`` 后
           ``safetensors.torch.save_file`` 写出单个 ``.safetensors`` 文件——
           该文件可被 ``safetensors`` 正常打开并枚举张量键（R5-AC1、R5-AC3），
           供 Vault 完整性校验通过。

        导出文件的落点与命名由 :func:`ego_pod.trainer.pipeline.run_training` 既有
        约定处理（``{cwd}/.ego/lora.safetensors``），本方法只负责写入 ``output_path``
        指向的单个文件（R5-AC6）。

        所有可选依赖（``torch`` / ``peft`` / ``safetensors``）在本方法内惰性导入，
        保证模块顶层 ``import`` 永不因缺依赖而失败（R8-AC4）。

        :param model: :meth:`load_base_model` 返回、经 :meth:`train` 训练后的
            ``(model_obj, tokenizer)`` 复合对象。
        :param output_path: 目标 ``.safetensors`` 文件路径，供 Vault 持久化。
        :raises TrainerError: 写入 ``.safetensors`` 文件遭遇 I/O 错误（或其他底层
            失败）时，其消息包含目标 ``output_path`` 与失败原因（R5-AC5）。
        """

        model_obj, _tokenizer = _unpack_model(model)

        try:
            # 惰性导入：保证模块顶层 import 永不因缺依赖而失败（R8-AC4）。
            from peft import get_peft_model_state_dict  # noqa: PLC0415
            from safetensors.torch import save_file  # noqa: PLC0415

            # 仅取 LoRA 适配器张量，绝不含 base 骨干权重（R5-AC2）。
            adapter_state = get_peft_model_state_dict(model_obj)

            # 转为连续 CPU 张量，确保 safetensors 可安全序列化。
            tensors = {
                key: tensor.detach().cpu().contiguous()
                for key, tensor in adapter_state.items()
            }

            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_file(tensors, str(output_path))
        except EgoError:
            # 已构造的 TrainerError 等原样透传，保持退出码语义不被二次包装。
            raise
        except Exception as exc:
            raise TrainerError(
                f"LoRA_Adapter 导出失败（output_path={output_path}）：{exc}"
            ) from exc


# --- 对外后端 ---------------------------------------------------------------


class PeftCudaBackend(_PeftBackendBase):
    """运行于 NVIDIA CUDA 的 HuggingFace PEFT 后端（探测优先级 2）。"""

    name = "HuggingFace PEFT"
    hardware: Hardware = "CUDA"
    identifier: BackendIdentifier = "peft-cuda"

    @classmethod
    def is_available(cls) -> bool:
        """``peft`` / ``transformers`` / ``torch`` 可导入且检测到 CUDA 设备。"""

        return _peft_importable() and _torch_cuda_available()


class PeftCpuBackend(_PeftBackendBase):
    """运行于 CPU 的 HuggingFace PEFT 后端（探测优先级 4）。"""

    name = "HuggingFace PEFT"
    hardware: Hardware = "CPU"
    identifier: BackendIdentifier = "peft-cpu"

    @classmethod
    def is_available(cls) -> bool:
        """``peft`` / ``transformers`` / ``torch`` 可导入即可（不要求 CUDA）。"""

        return _peft_importable()
