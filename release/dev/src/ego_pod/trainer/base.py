"""Trainer 模块的后端抽象。

定义所有 Training_Backend 必须满足的结构化接口 ``TrainingBackend``。采用
``typing.Protocol`` 而非抽象基类（ABC）的原因（见设计文档「多后端 Trainer
抽象」一节）：

- 各后端对外只需暴露相同的一组属性与方法，使用鸭子类型即可；
- 部分后端（Unsloth、MLX）在缺失对应依赖时无法 ``import``，若以 ABC
  作为公共基类会引入对全部后端依赖的硬绑定，而 Protocol 允许任意类在
  无需继承的前提下满足接口。

后端按 ``detect`` 模块定义的优先级顺序探测：Unsloth（CUDA）→ HuggingFace
PEFT（CUDA）→ Apple MLX（Apple Silicon）→ HuggingFace PEFT（CPU）→
Stub_Backend，且 Stub_Backend 始终视为可用以作为最终回退。
"""

from pathlib import Path
from typing import Any, Callable, Literal, Protocol, runtime_checkable

from ego_pod.config.schema import TrainingParams

__all__ = [
    "Hardware",
    "ProgressCallback",
    "TrainingBackend",
    "BackendIdentifier",
    "resolve_identifier",
]


#: 后端所运行的硬件类型。``STUB`` 表示无 ML 依赖的模拟后端。
Hardware = Literal["CUDA", "Apple Silicon", "CPU", "STUB"]

#: Backend_Identifier 的固定取值集合（需求 7.1、7.4）。
#:
#: 写入 ``manifest.json`` 的 ``training_backend`` 字段，是能稳定区分各后端的
#: 字符串标识，使 Runtime 能据此选择匹配的推理路径。
BackendIdentifier = Literal["unsloth", "peft-cuda", "mlx", "peft-cpu", "stub"]

#: 训练进度回调签名。
#:
#: 训练过程中（至少每完成一个 epoch）由后端调用一次，参数依次为：
#:
#: - ``epoch``：当前已完成的 epoch 编号（正整数）。
#: - ``step``：累计训练步数（非负整数）。
#: - ``loss``：当前 epoch 的训练损失值。
#:
#: 由调用方（``trainer.pipeline``）提供具体实现，通常委托 ``logging_io``
#: 打印固定格式的进度记录。
ProgressCallback = Callable[[int, int, float], None]


@runtime_checkable
class TrainingBackend(Protocol):
    """实际执行 LoRA 训练计算的底层框架后端的统一接口。

    实现类无需显式继承本 Protocol，只要在结构上具备以下属性与方法即视为
    满足接口。
    """

    #: 后端的可读名称，用于在标准输出记录所选后端（需求 8.2）。
    name: str
    #: 后端运行的硬件类型（需求 8.2）。
    hardware: Hardware
    #: 能稳定区分各后端的 Backend_Identifier（需求 7.1、7.4）。
    #:
    #: 写入 ``manifest.json`` 的 ``training_backend`` 字段，使 Runtime 能据此
    #: 选择匹配的推理路径。各后端类以类属性声明该取值。
    identifier: BackendIdentifier

    @classmethod
    def is_available(cls) -> bool:
        """判定该后端在当前环境是否可用。

        通常检查所需依赖能否成功导入以及目标硬件（CUDA / Apple Silicon）
        是否存在。Stub_Backend 始终返回 ``True`` 以充当最终回退（需求 8.1）。
        """
        ...

    def load_base_model(self, model_id: str, quantization: str) -> Any:
        """加载 Base_Model 并应用量化。

        :param model_id: 基座模型标识，MVP 固定为 ``qwen3.5-0.8B``。
        :param quantization: 量化方案，MVP 固定为 ``4bit``（需求 8.3）。
        :returns: 后端内部表示的模型对象，作为 :meth:`train` 与
            :meth:`export_lora_safetensors` 的输入。
        """
        ...

    def train(
        self,
        model: Any,
        corpus_path: Path,
        params: TrainingParams,
        progress_callback: ProgressCallback,
    ) -> None:
        """在给定语料上训练 LoRA 适配器。

        :param model: :meth:`load_base_model` 返回的模型对象。
        :param corpus_path: Conversation_Corpus（JSONL）中间文件路径。
        :param params: 来自 Ego_Config 的训练参数，提供
            ``lora_rank`` / ``lora_alpha`` / ``learning_rate`` / ``epochs`` /
            ``batch_size`` 五个值（需求 8.4）。
        :param progress_callback: 进度回调，至少每完成一个 epoch 调用一次
            （需求 8.8）。
        """
        ...

    def export_lora_safetensors(self, model: Any, output_path: Path) -> None:
        """以 SafeTensors 格式将训练得到的 LoRA_Adapter 权重导出到指定路径。

        :param model: 已完成训练的模型对象。
        :param output_path: 目标 ``.safetensors`` 文件路径，供 Vault 持久化
            （需求 8.6）。
        """
        ...


#: ``name`` → Backend_Identifier 的回退映射（用于 ``identifier`` 缺失时推断）。
#:
#: HuggingFace PEFT 的两个变体（CUDA/CPU）共享同一 ``name``，无法靠 ``name``
#: 区分，故由 :func:`resolve_identifier` 按 ``hardware`` 单独处理；其余后端靠
#: ``name`` 唯一映射。
_NAME_TO_IDENTIFIER: dict[str, BackendIdentifier] = {
    "stub": "stub",
    "Unsloth": "unsloth",
    "Apple MLX": "mlx",
}

#: Backend_Identifier 的合法取值集合（用于校验 ``identifier`` 成员）。
_VALID_IDENTIFIERS: frozenset[str] = frozenset(
    {"unsloth", "peft-cuda", "mlx", "peft-cpu", "stub"}
)


def resolve_identifier(backend: Any) -> BackendIdentifier:
    """解析一个 Training_Backend 的稳定 Backend_Identifier（需求 7.1、7.2、7.4）。

    优先读取 ``backend.identifier``；当该成员缺失或取值非法时，按 ``name`` /
    ``hardware`` 回退推断，以稳健满足「Protocol 形状不变」的兼容性要求
    （需求 10.1）——即便历史/外部后端实现未声明 ``identifier`` 也能得到正确标识。

    回退规则：

    - HuggingFace PEFT 的两个变体共享 ``name == "HuggingFace PEFT"``，靠
      ``hardware`` 区分：``CUDA`` → ``peft-cuda``，否则 → ``peft-cpu``。
    - 其余后端靠 ``name`` 唯一映射（见 :data:`_NAME_TO_IDENTIFIER`）。
    - 无法识别时兜底为 ``stub``。

    :param backend: 任意满足（或近似满足）``TrainingBackend`` 的后端对象。
    :returns: 取自固定集合 ``{unsloth, peft-cuda, mlx, peft-cpu, stub}`` 的标识。
    """
    ident = getattr(backend, "identifier", None)
    if ident in _VALID_IDENTIFIERS:
        return ident  # type: ignore[return-value]
    # 回退：PEFT 两变体靠 hardware 区分 cuda/cpu；其余靠 name 映射。
    if getattr(backend, "name", "") == "HuggingFace PEFT":
        return "peft-cuda" if getattr(backend, "hardware", "") == "CUDA" else "peft-cpu"
    return _NAME_TO_IDENTIFIER.get(getattr(backend, "name", ""), "stub")
