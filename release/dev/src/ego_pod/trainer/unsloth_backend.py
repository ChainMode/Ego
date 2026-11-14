"""Unsloth 训练后端（NVIDIA CUDA）。

本模块提供满足 :class:`ego_pod.trainer.base.TrainingBackend` Protocol 的
:class:`UnslothBackend`，对应 ``detect`` 优先级中的第 1 位：当 ``unsloth``
可导入且检测到 NVIDIA CUDA 设备时可用。

设计约束（见设计文档「关键架构决策 · 决策 2」）：

- **顶层导入必须永不抛错**。可选依赖（``unsloth``）通过
  ``try/except ImportError`` 包裹为模块级哨兵，使 ``detect`` 能防御式地无
  条件 ``import`` 本模块，即便对应依赖缺失（R8-AC4）。
- :class:`UnslothBackend` 由 MVP 的"组合委托"改为
  ``UnslothBackend(_PeftBackendBase)`` **继承**：既然 PEFT 共享实现已落地真实
  的加载/训练/导出，Unsloth 只需覆盖 :meth:`is_available` 与
  :meth:`load_base_model`（用 unsloth 的 ``FastLanguageModel.from_pretrained``
  加载 4-bit 量化基座并返回 ``(model, tokenizer)``），:meth:`train` /
  :meth:`export_lora_safetensors` 直接复用基类实现——两条路径最终都得到一个
  带 PEFT 适配器的 torch 模型，故训练/导出逻辑通用（R9-AC1、R9-AC3）。仍满足
  ``TrainingBackend`` Protocol（R10-AC1）。
"""

from __future__ import annotations

from typing import Any

from ego_pod.exit_codes import EgoError, TrainerError
from ego_pod.logging_io import print_summary
from ego_pod.trainer.base import BackendIdentifier, Hardware
from ego_pod.trainer.peft_backend import _PeftBackendBase, _torch_cuda_available

__all__ = [
    "UnslothBackend",
]


# --- 可选依赖探测 -----------------------------------------------------------
# 包裹在 try/except 中，确保本模块在缺失 unsloth 时仍可被安全导入
# （顶层 import 永不抛错），从而允许 detect 模块防御式地导入本模块。

try:  # pragma: no cover - 依赖是否存在取决于运行环境
    import unsloth as _unsloth
except ImportError:  # pragma: no cover
    _unsloth = None


def _unsloth_importable() -> bool:
    """判定 Unsloth 是否可导入。"""

    return _unsloth is not None


class UnslothBackend(_PeftBackendBase):
    """运行于 NVIDIA CUDA 的 Unsloth 后端（探测优先级 1）。

    继承 :class:`ego_pod.trainer.peft_backend._PeftBackendBase` 并复用其真实
    QLoRA 训练（:meth:`train`）与仅含适配器的 SafeTensors 导出
    （:meth:`export_lora_safetensors`），仅覆盖 :meth:`is_available` 与
    :meth:`load_base_model`（用 unsloth 的 ``FastLanguageModel`` 加载基座）。
    """

    name = "Unsloth"
    hardware: Hardware = "CUDA"
    identifier: BackendIdentifier = "unsloth"

    @classmethod
    def is_available(cls) -> bool:
        """``unsloth`` 可导入且检测到 CUDA 设备。"""

        return _unsloth_importable() and _torch_cuda_available()

    def load_base_model(self, model_id: str, quantization: str) -> Any:
        """用 Unsloth 加载 4-bit 量化的 Base_Model（R1、R9-AC1）。

        通过 ``unsloth.FastLanguageModel.from_pretrained(load_in_4bit=True)``
        加载 Qwen 3.5 0.8B 基座并量化，返回 ``(model, tokenizer)`` 复合对象，供
        继承自基类的 :meth:`train`（在其内用 ``params.lora_rank`` /
        ``params.lora_alpha`` 构造 ``LoraConfig`` 并应用 LoRA 适配器）与
        :meth:`export_lora_safetensors` 使用。

        加载成功后在标准输出打印一行包含 ``model_id`` / 目标设备（CUDA）/ 实际
        量化方案的加载摘要，与基类格式保持一致（R1-AC4）。

        ``unsloth`` 在本方法内惰性导入，保证模块顶层 ``import`` 永不因缺依赖
        而失败（R8-AC4）。

        :param model_id: 基座模型标识，MVP 固定为 ``qwen3.5-0.8B``。
        :param quantization: 量化方案，MVP 固定为 ``4bit``。
        :returns: ``(model, tokenizer)`` 复合对象。
        :raises TrainerError: 加载因权重缺失、依赖缺失或显存不足而失败时，其
            消息包含 ``model_id`` 与具体失败原因（R1-AC5）。
        """

        try:
            # 惰性导入：保证模块顶层 import 永不因缺依赖而失败（R8-AC4）。
            from unsloth import FastLanguageModel  # noqa: PLC0415

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_id,
                max_seq_length=self._MAX_SEQ_LEN,
                load_in_4bit=True,
            )
        except EgoError:
            # OfflineMissingWeightsError / 已构造的 TrainerError 等原样透传，
            # 保持各自退出码语义不被二次包装（与基类处理一致）。
            raise
        except Exception as exc:
            raise TrainerError(
                f"Base_Model 加载失败（model_id={model_id!r}）：{exc}"
            ) from exc

        # 加载摘要：model_id / 设备 / 实际量化方案（R1-AC4），与基类格式一致。
        print_summary(
            f"Loaded base model: model_id={model_id} "
            f"device=CUDA quantization=4bit (unsloth)"
        )
        return model, tokenizer
