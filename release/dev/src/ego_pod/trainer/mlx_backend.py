"""Apple MLX 训练后端（Apple Silicon）。

本模块提供满足 :class:`ego_pod.trainer.base.TrainingBackend` Protocol 的
:class:`MlxBackend`，对应 ``detect`` 优先级中的第 3 位：当 ``mlx_lm`` 可导入、
``platform.machine() == "arm64"`` 且运行于 macOS 时可用。

设计约束（见设计文档「多后端 Trainer 抽象」一节）：

- **顶层导入必须永不抛错**。可选依赖（``mlx_lm``）通过 ``try/except
  ImportError`` 包裹为模块级哨兵，使 ``detect`` 能防御式地无条件 ``import``
  本模块，即便对应依赖缺失。
- **本特性（Real Trainer）不落地 MLX 真实训练**。MLX 的张量栈（``mlx``）与
  ``torch`` / ``peft`` 不兼容，需独立的 SFT/LoRA 实现，超出本次特性范围。因此
  MLX 后端的 ``is_available`` 仍保持 MVP 判定（``mlx_lm`` 可导入 + Apple
  Silicon），但三个训练方法（``load_base_model`` / ``train`` /
  ``export_lora_safetensors``）均**显式抛出携带明确「未实现」原因的**
  :class:`~ego_pod.exit_codes.TrainerError`（退出码 1，需求 9.4），杜绝
  「探测可用 → 调用静默产出无效/空适配器」的隐患。

  实现影响（需求 8.1 + 9.4 的共同结果）：在已安装 ``mlx`` extra 的 Apple
  Silicon 机器上，按探测优先级 MLX（第 3）先于 PEFT-CPU（第 4）被选中，从而
  以退出码 1 失败。欲在 Apple Silicon 上用 PEFT-CPU 训练，请安装 ``peft``
  extra 但**不**安装 ``mlx`` extra（直至 MLX 真实训练落地）。
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Any

from ego_pod.config.schema import TrainingParams
from ego_pod.exit_codes import TrainerError
from ego_pod.trainer.base import BackendIdentifier, Hardware, ProgressCallback

__all__ = [
    "MlxBackend",
]


# --- 可选依赖探测 -----------------------------------------------------------
# 包裹在 try/except 中，确保本模块在缺失 mlx_lm 时仍可被安全导入
# （顶层 import 永不抛错），从而允许 detect 模块防御式地导入本模块。

try:  # pragma: no cover - 依赖是否存在取决于运行环境
    import mlx_lm as _mlx_lm
except ImportError:  # pragma: no cover
    _mlx_lm = None


def _mlx_importable() -> bool:
    """判定 ``mlx_lm`` 是否可导入。"""

    return _mlx_lm is not None


def _is_apple_silicon() -> bool:
    """判定当前是否运行于 Apple Silicon（macOS + arm64）。

    在探测过程中发生任何异常时一律返回 ``False``，确保探测过程不会抛错。
    """

    try:
        return sys.platform == "darwin" and platform.machine() == "arm64"
    except Exception:  # pragma: no cover - 防御性兜底
        return False


class MlxBackend:
    """运行于 Apple Silicon 的 MLX 后端（探测优先级 3）。

    本特性范围内 MLX 真实训练尚未落地，三个训练方法均显式抛出携带明确
    「未实现」原因的 :class:`~ego_pod.exit_codes.TrainerError`（退出码 1，
    需求 9.4），而非静默产出无效适配器。``is_available`` 保持 MVP 判定不变
    （需求 8.1）。
    """

    name = "Apple MLX"
    hardware: Hardware = "Apple Silicon"
    #: 稳定区分本后端的 Backend_Identifier（需求 7.1、7.4、9.5）。
    identifier: BackendIdentifier = "mlx"

    #: 显式「未实现」原因，三个训练方法共用，确保失败响亮且给出可行替代方案。
    _NOT_IMPLEMENTED_MSG = (
        "Apple MLX 后端的真实 QLoRA 训练在本特性范围内暂未落地"
        "（mlx 张量栈与 torch/peft 不兼容，需独立实现）；"
        "请在 Apple Silicon 上改用 PEFT-CPU（安装 peft extra、不安装 mlx extra）。"
    )

    @classmethod
    def is_available(cls) -> bool:
        """``mlx_lm`` 可导入且运行于 Apple Silicon（macOS + arm64）。"""

        return _mlx_importable() and _is_apple_silicon()

    def load_base_model(self, model_id: str, quantization: str) -> Any:
        """加载 Base_Model 并应用量化（本特性范围内未实现）。

        显式抛出 :class:`TrainerError`（退出码 1），不静默产出无效适配器
        （需求 9.4、9.5）。
        """

        raise TrainerError(self._NOT_IMPLEMENTED_MSG)

    def train(
        self,
        model: Any,
        corpus_path: Path,
        params: TrainingParams,
        progress_callback: ProgressCallback,
    ) -> None:
        """在给定语料上训练 LoRA 适配器（本特性范围内未实现）。

        防御性显式抛出 :class:`TrainerError`（退出码 1），确保任何调用路径
        都不静默产出无效适配器（需求 9.4）。
        """

        raise TrainerError(self._NOT_IMPLEMENTED_MSG)

    def export_lora_safetensors(self, model: Any, output_path: Path) -> None:
        """以 SafeTensors 格式导出 LoRA_Adapter 权重（本特性范围内未实现）。

        防御性显式抛出 :class:`TrainerError`（退出码 1），确保任何调用路径
        都不静默产出无效适配器（需求 9.4）。
        """

        raise TrainerError(self._NOT_IMPLEMENTED_MSG)
