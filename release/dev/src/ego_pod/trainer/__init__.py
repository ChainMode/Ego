"""Trainer 训练引擎模块。

负责探测可用的 Training_Backend、加载 Base_Model、以 QLoRA 训练 LoRA
适配器并以 SafeTensors 格式导出权重。本模块的对外抽象由 ``base`` 定义
（``TrainingBackend`` Protocol 与 ``ProgressCallback`` 类型别名），后续任务
将补充后端探测（``detect``）、Stub_Backend（``stub``）、可选后端实现
（``unsloth_backend`` / ``peft_backend`` / ``mlx_backend``）与编排（``pipeline``）。
"""

from ego_pod.trainer.base import ProgressCallback, TrainingBackend

__all__ = [
    "ProgressCallback",
    "TrainingBackend",
]
