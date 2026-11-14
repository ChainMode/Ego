"""Stub_Backend：无机器学习依赖的模拟训练后端。

``StubBackend`` 满足 ``TrainingBackend`` Protocol（见 ``base.py``），用于在
缺失真实训练框架（Unsloth / PEFT / MLX）的环境（尤其是 CI）下让整条
``ego train`` 流水线端到端可跑通：

- 不下载、不加载任何真实 Base_Model 权重（需求 9.4/9.5 隐私约束自然满足）。
- 不对模型权重做任何更新（需求 8.5）。
- 导出一个可被 SafeTensors 库正常读回、并能被 Vault 与 ``ego run`` 识别为
  stub 产物的占位 LoRA_Adapter 文件（需求 8.6、12.9）。

作为后端探测链（见 ``detect.py``）的最终回退，:meth:`StubBackend.is_available`
始终返回 ``True``（需求 8.1）。``training_backend == "stub"`` 这一 stub 标记由
Vault 写入 ``manifest.json``，``ego run`` 据此进入 Stub REPL（需求 12.9）；本
模块只负责产出占位权重，标记的写入与识别分别由 Vault 与 Runtime 完成。
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.numpy import save_file

from ego_pod.config.schema import TrainingParams
from ego_pod.trainer.base import BackendIdentifier, Hardware, ProgressCallback

__all__ = ["StubBackend"]


class StubBackend:
    """模拟训练后端，结构上满足 ``TrainingBackend`` Protocol。

    无需继承 ``TrainingBackend``（Protocol 采用鸭子类型），只要具备相同的
    属性与方法即视为满足接口。
    """

    #: 后端可读名称，用于流水线打印所选后端并作为 ``manifest.json`` 中的
    #: ``training_backend`` stub 标记来源（需求 8.2、12.9）。
    name: str = "stub"
    #: 后端硬件类型；stub 后端不绑定任何真实硬件。
    hardware: Hardware = "STUB"
    #: 稳定的 Backend_Identifier，写入 ``manifest.json`` 的 ``training_backend``
    #: 字段并供 Runtime 据此选择推理路径（需求 7.2、8.7）。
    identifier: BackendIdentifier = "stub"

    @classmethod
    def is_available(cls) -> bool:
        """Stub_Backend 始终可用，充当后端探测链的最终回退（需求 8.1）。"""
        return True

    def load_base_model(self, model_id: str, quantization: str) -> Any:
        """返回占位模型对象，不下载或加载任何真实权重（需求 8.5）。

        :param model_id: 基座模型标识，MVP 固定为 ``qwen3.5-0.8B``；stub 仅作
            记录用途，不据此发起任何下载。
        :param quantization: 量化方案，MVP 固定为 ``4bit``；stub 不实际量化。
        :returns: 一个 ``dict[str, torch.Tensor]`` 占位对象（torch 为必装依赖）。
        """
        return {
            "model_id": model_id,
            "quantization": quantization,
            "placeholder": torch.zeros((1, 1), dtype=torch.float32),
        }

    def train(
        self,
        model: Any,
        corpus_path: Path,
        params: TrainingParams,
        progress_callback: ProgressCallback,
    ) -> None:
        """模拟训练：跳过权重更新，仅触发一次进度回调（需求 8.5、8.8）。

        直接以 ``epoch == params.epochs`` 调用一次 ``progress_callback``，满足
        "至少每完成一个 epoch 打印一次进度记录"的约束（需求 8.8），而无需真正
        迭代任何 epoch。

        :param model: :meth:`load_base_model` 返回的占位对象，stub 不读取其内容。
        :param corpus_path: Conversation_Corpus（JSONL）路径，stub 不读取其内容。
        :param params: 训练参数，仅取 ``epochs`` 用于回调的 epoch/step 取值。
        :param progress_callback: 进度回调，至少每完成一个 epoch 调用一次。
        """
        progress_callback(epoch=params.epochs, step=params.epochs, loss=0.0)

    def export_lora_safetensors(self, model: Any, output_path: Path) -> None:
        """写入占位 LoRA_Adapter 权重文件（需求 8.6、12.9）。

        以 ``safetensors.numpy.save_file`` 写入单个 1×1 占位张量，使 Vault 能将
        其当作合法 LoRA_Adapter 持久化、``ego run`` 能正常读回。

        :param model: 已"训练"的占位对象，stub 不读取其内容。
        :param output_path: 目标 ``.safetensors`` 文件路径，供 Vault 持久化。
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tensors = {"placeholder": np.zeros((1, 1), dtype=np.float32)}
        save_file(tensors, str(output_path))
