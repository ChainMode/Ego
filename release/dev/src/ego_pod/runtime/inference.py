"""``ego run`` 的模型加载与推理（需求 12.1、12.5、12.9）。

本模块实现设计文档 §7「Runtime 模块」的「加载逻辑（inference.py）」：

1. 从 ``ego-output/<edition>/config.json`` 读取 ``base_model`` 与 LoRA 参数
   （``lora_rank`` / ``lora_alpha`` / ``quantization``）（需求 12.1）。
2. 从 ``ego-output/manifest.json`` 读取 ``training_backend`` 标记。
3. 若 ``training_backend == "stub"`` → 进入 **Stub 推理路径**：在 ``ego run``
   启动时于标准输出打印 Stub 模式提示，并对每次非空输入返回同一条预定义占位
   字符串（需求 12.9）。
4. 否则 → 据 ``manifest.json`` 的 ``training_backend`` 标识（``Backend_Identifier``）
   经 :func:`_resolve_inference_backend` 选择与训练时匹配的真实后端，加载
   Base_Model 并应用 ``adapters/ego-lora.safetensors`` 中的 LoRA_Adapter 以进行
   真实推理（需求 6.1、7.5）。

加载与完整性校验（需求 12.5）：``manifest.json`` / ``config.json`` /
``adapters/ego-lora.safetensors`` 中任一文件缺失或损坏时，抛出
:class:`~ego_pod.exit_codes.RuntimeError_`（退出码 1），其消息携带该文件相对于
进程工作目录的相对路径，供 CLI 顶层打印到标准错误。

对外接口：:func:`load_inference_session` 完成加载与校验并返回一个
:class:`InferenceSession`；后者暴露 ``generate(user_input) -> str``，供 REPL
（任务 10.2）对每次用户输入生成一次回复。空白/空行的过滤由 REPL 负责
（需求 12.7），本模块的 ``generate`` 面向「非空输入」。

**真实推理路径（需求 6）**：:class:`BackendInferenceSession` 在构造时按
``Backend_Identifier`` 解析对应后端，调用 ``backend.load_base_model`` 加载
Base_Model（返回 ``(model, tokenizer)``），再用 :func:`peft.set_peft_model_state_dict`
把适配器权重应用到 base，并以确定性（贪心）解码生成回复（需求 6.2、6.3）。
加载阶段若所需 Training_Dependencies 缺失或硬件不可用，则以
:class:`RuntimeError_` 结束（需求 6.4）；会话关闭时释放 base + adapter 资源
（需求 6.5）。Stub 推理路径行为保持不变（需求 6.6）。
"""

from __future__ import annotations

import gc
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TextIO, runtime_checkable

from ego_pod.exit_codes import EgoError, RuntimeError_
from ego_pod.runtime.selection import KERNEL, CellKey, LoadSelection, select_cells
from ego_pod.trainer.detect import detect_backend
from ego_pod.trainer.generation import DecodeSettings, generate_reply
from ego_pod.vault.grid_descriptor import parse_grid_descriptor

__all__ = [
    "OUTPUT_DIRNAME",
    "MANIFEST_FILENAME",
    "CONFIG_FILENAME",
    "ADAPTER_RELPATH",
    "DEFAULT_EDITION",
    "DEFAULT_BASE_MODEL",
    "DEFAULT_QUANTIZATION",
    "STUB_TRAINING_BACKEND",
    "STUB_MODE_STARTUP_NOTICE",
    "STUB_PLACEHOLDER_REPLY",
    "InferenceSession",
    "StubInferenceSession",
    "BackendInferenceSession",
    "load_inference_session",
    "_resolve_inference_backend",
]


# ---------------------------------------------------------------------------
# Ego_Package 结构常量（与 vault.package 保持一致）
# ---------------------------------------------------------------------------

#: Ego_Package 根目录名，位于 ``ego run`` 执行时所在的当前工作目录下（需求 12.1）。
OUTPUT_DIRNAME = "ego-output"
#: Ego_Package 根部 manifest 文件名。
MANIFEST_FILENAME = "manifest.json"
#: Edition 子目录下的配置文件名（需求 12.1）。
CONFIG_FILENAME = "config.json"
#: LoRA_Adapter 权重文件相对于 Edition 子目录的相对路径（需求 12.1）。
ADAPTER_RELPATH = os.path.join("adapters", "ego-lora.safetensors")
#: MVP 唯一 Edition 名称。
DEFAULT_EDITION = "normal"
#: 当 ``config.json`` 未声明 ``base_model`` 时的回退基座标识。
DEFAULT_BASE_MODEL = "qwen3.5-0.8B"
#: 当 ``config.json`` 未声明 ``quantization`` 时的回退量化方案。
DEFAULT_QUANTIZATION = "4bit"

#: ``manifest.json`` 中标识训练阶段使用 Stub_Backend 的取值
#: （与 ``trainer.stub.StubBackend.name`` 一致，需求 12.9）。
STUB_TRAINING_BACKEND = "stub"

#: Stub 模式启动提示，在 ``ego run`` 启动时打印到标准输出（需求 12.9）。
STUB_MODE_STARTUP_NOTICE = (
    "[STUB MODE] 当前 Ego_Package 由 Stub_Backend 训练产出；"
    "ego run 进入 Stub 模式，对每次输入返回固定占位回复。"
)

#: Stub 模式下对每次非空输入返回的固定占位字符串（需求 12.9）。
STUB_PLACEHOLDER_REPLY = (
    "[STUB MODE] This is a placeholder reply from a stub-trained Ego."
)


# ---------------------------------------------------------------------------
# 会话抽象
# ---------------------------------------------------------------------------


@runtime_checkable
class InferenceSession(Protocol):
    """一次 ``ego run`` 会话的推理接口，供 REPL（任务 10.2）逐轮调用。

    REPL 对每次**非空**用户输入调用一次 :meth:`generate`（空白/空行由 REPL 过滤，
    见需求 12.7），并在退出时调用 :meth:`close` 释放资源（需求 12.8）。
    """

    def generate(self, user_input: str) -> str:
        """对一次非空用户输入生成一条回复（需求 12.6）。"""
        ...

    def close(self) -> None:
        """释放会话持有的资源（需求 12.8）。"""
        ...


@dataclass
class StubInferenceSession:
    """Stub 推理会话：对每次非空输入返回同一条固定占位字符串（需求 12.9）。

    不加载任何真实模型权重，与 Stub_Backend 的"无 ML 依赖"语义一致，保证
    ``ego run`` 在 stub 训练产出上端到端可跑通。
    """

    #: 对每次非空输入返回的固定回复，默认 :data:`STUB_PLACEHOLDER_REPLY`。
    reply: str = STUB_PLACEHOLDER_REPLY

    def generate(self, user_input: str) -> str:
        """返回固定占位回复（需求 12.9）。

        :param user_input: 用户输入文本；Stub 模式下不据其内容变化回复。
        :returns: 预定义的占位字符串 :attr:`reply`。
        """
        return self.reply

    def close(self) -> None:
        """Stub 会话不持有任何需释放的资源；空操作。"""
        return None


def _resolve_inference_backend(identifier: str) -> Any:
    """把 ``Backend_Identifier`` 解析为对应的真实推理后端实例（需求 6.1、7.5）。

    映射四个真实后端标识到各自的后端工厂；标识不在固定集合内时抛
    :class:`RuntimeError_`（退出码 1），以避免对未知标识静默走错路径。

    所有后端类在函数内惰性导入，使本模块在缺失任一 Training_Dependencies 的
    环境中仍可被安全导入（需求 8.4）：后端模块自身的顶层 ``import`` 已对可选
    依赖做哨兵化处理，真正的依赖缺失会在 ``load_base_model`` 阶段暴露并由
    :class:`BackendInferenceSession` 包装为 :class:`RuntimeError_`（需求 6.4）。

    :param identifier: ``manifest.json`` 的 ``training_backend`` 取值，应属于
        ``{peft-cuda, peft-cpu, unsloth, mlx}``。
    :returns: 与该标识匹配的后端实例。
    :raises RuntimeError_: 标识不在上述固定集合内。
    """

    from ego_pod.trainer.mlx_backend import MlxBackend
    from ego_pod.trainer.peft_backend import PeftCpuBackend, PeftCudaBackend
    from ego_pod.trainer.unsloth_backend import UnslothBackend

    factories: dict[str, Any] = {
        "peft-cuda": PeftCudaBackend,
        "peft-cpu": PeftCpuBackend,
        "unsloth": UnslothBackend,
        "mlx": MlxBackend,
    }
    factory = factories.get(identifier)
    if factory is None:
        raise RuntimeError_(
            f"未知的 training_backend 标识：{identifier!r}；"
            "无法选择匹配的真实推理后端。"
        )
    return factory()


def _release_torch_memory() -> None:
    """尽力释放进程持有的（GPU）显存与未引用对象（需求 6.5）。

    始终调用 :func:`gc.collect` 回收已置空的模型/适配器对象；``torch`` 可导入且
    检测到 CUDA 时再调用 ``torch.cuda.empty_cache()`` 释放缓存显存。全部步骤均
    包裹在防御性 ``try/except`` 中——资源释放是「尽力而为」，任何底层异常都不应
    使会话关闭流程失败，亦不要求 ``torch`` 存在（需求 8.4）。
    """

    try:
        gc.collect()
    except Exception:  # pragma: no cover - 防御性：gc 不应抛错
        pass

    try:
        import torch  # noqa: PLC0415 - 惰性导入以保证导入安全

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover - torch 缺失或 CUDA 探测异常时忽略
        pass


def _apply_lora_adapter(
    model: Any,
    adapter_path: Path,
    lora_rank: int,
    lora_alpha: int,
) -> Any:
    """把 ``adapter_path`` 中的 LoRA 适配器应用到 base 模型（需求 6.1、6.2）。

    步骤（见设计 §9「``_apply_lora_adapter`` intent」）：

    1. ``backend.load_base_model`` 返回的是**未挂适配器的 base 模型**，故先用
       ``LoraConfig(r=lora_rank, lora_alpha=lora_alpha)`` 经 ``peft.get_peft_model``
       构造与训练时同构的 PEFT 适配器骨架；
    2. 经 ``safetensors.torch.load_file`` 读取适配器张量；
    3. 经 ``peft.set_peft_model_state_dict`` 把权重灌入 PEFT 模型，使 base 叠加
       训练得到的 LoRA_Adapter。

    全部 ML 依赖（``peft`` / ``safetensors`` / ``torch``）在函数内惰性导入，保证
    本模块顶层 ``import`` 永不因缺依赖而失败（需求 8.4）。依赖缺失（``ImportError``）
    或权重不兼容（``RuntimeError`` / ``OSError`` 等）由调用方
    :class:`BackendInferenceSession` 统一包装为 :class:`RuntimeError_`（需求 6.4）。

    :param model: ``backend.load_base_model`` 返回的 base 模型对象。
    :param adapter_path: 已校验合法的 LoRA 适配器 SafeTensors 路径。
    :param lora_rank: LoRA 秩，来自 ``config.json``，须与训练时一致。
    :param lora_alpha: LoRA alpha，来自 ``config.json``，须与训练时一致。
    :returns: 已应用 LoRA 适配器的 PEFT 模型对象。
    """

    from peft import (  # noqa: PLC0415 - 惰性导入保证导入安全
        LoraConfig,
        get_peft_model,
        set_peft_model_state_dict,
    )
    from safetensors.torch import load_file  # noqa: PLC0415

    # 若 base 已是 PEFT 模型（部分后端在加载阶段已挂适配器）则复用，否则构造同构骨架。
    if hasattr(model, "peft_config"):
        peft_model = model
    else:
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            bias="none",
            task_type="CAUSAL_LM",
        )
        peft_model = get_peft_model(model, lora_config)

    adapter_state = load_file(str(adapter_path))
    set_peft_model_state_dict(peft_model, adapter_state)
    return peft_model


# ---------------------------------------------------------------------------
# 多适配器并存激活（Coexist_Activate，method A）
# ---------------------------------------------------------------------------

#: 网格包中 Kernel 适配器的稳定 PEFT 适配器名（Stable PEFT adapter name for Kernel）。
KERNEL_ADAPTER_NAME = "kernel"


def _cell_name(cell: CellKey) -> str:
    """把一个 :class:`CellKey` 映射为稳定的 PEFT 适配器名（R5-AC4）。

    Map a :class:`~ego_pod.runtime.selection.CellKey` to a stable PEFT adapter
    name used as the ``adapter_name`` for ``load_adapter`` / ``set_adapter``：

    * Kernel（``zone=None, level=0``）⇒ :data:`KERNEL_ADAPTER_NAME`（``"kernel"``）。
    * 分区 Cell（``zone, level>=1``）⇒ ``"<zone>_L<level>"``（如 ``"coding_L1"``）。

    ``Zone_Name`` 已被约束为 ``[a-z0-9-]+``（R1-AC3），故该命名稳定且无歧义；
    Kernel 的名字与任何分区名（不含下划线 ``_L`` 形态）不冲突。

    :param cell: 网格中的一个 Cell 键。
    :returns: 作为 PEFT ``adapter_name`` 的稳定字符串。
    """

    if cell.zone is None and cell.level == 0:
        return KERNEL_ADAPTER_NAME
    return f"{cell.zone}_L{cell.level}"


def _set_active_adapters(peft_model: Any, adapter_names: list[str]) -> None:
    """把活动适配器集合重置为 ``adapter_names``（兼容真实 PEFT 与测试替身）。

    Set the active adapter set to ``adapter_names`` in a way that works for both
    a real :class:`peft.PeftModel` and the in-test fake double.

    PEFT 的多适配器激活语义（R5-AC4）：要**同时激活多个**命名适配器，须把名字
    列表交给底层 ``LoraModel`` 的 ``set_adapter``。真实 ``PeftModel.set_adapter``
    只接受**单个**适配器名（``str``），传入列表会以
    ``TypeError: unhashable type: 'list'`` 失败；而其 ``base_model``（``LoraModel``）
    的 ``set_adapter`` 才接受名字列表并对多个激活适配器做加性组合。因此本函数优先
    调用 ``peft_model.base_model.set_adapter(adapter_names)``。

    测试用的假 PEFT 模型替身（:class:`tests._doubles.FakePeftModel`）不暴露
    ``base_model``，但其自身的 ``set_adapter`` 直接接受名字列表——故在无
    ``base_model`` 时回退到 ``peft_model.set_adapter(adapter_names)``，保持
    Coexist_Activate 可逆性属性测试（Property 4）行为不变。

    :param peft_model: 已挂载全部命名适配器的 PEFT 模型（或其测试替身）。
    :param adapter_names: 要同时激活的有序适配器名列表（Kernel 在前）。
    """

    base_model = getattr(peft_model, "base_model", None)
    if base_model is not None and hasattr(base_model, "set_adapter"):
        # 真实 PEFT：经底层 LoraModel 同时激活多个命名适配器（加性组合，R5-AC4）。
        base_model.set_adapter(adapter_names)
    else:
        # 测试替身：其 set_adapter 直接接受名字列表。
        peft_model.set_adapter(adapter_names)


def _activate_cells(
    base: Any,
    cells: list[tuple[CellKey, Path]],
    lora_rank: int,
    lora_alpha: int,
) -> Any:
    """以 Coexist_Activate 把多个 Cell 适配器加性并存激活到冻结的 base（R5-AC4）。

    Additively load and activate multiple Cell adapters on the frozen base via
    PEFT's multi-adapter API（method A，Coexist_Activate）：

    1. **首个**（按 ``cells`` 顺序，即 Kernel）适配器：
       - base 尚不是 PEFT 模型（真实路径）⇒ 用 ``LoraConfig`` 经
         ``peft.get_peft_model(base, config, adapter_name=<cell_name>)`` 构造与
         训练时同构的 PEFT 骨架，再经 ``safetensors`` 读取该 Cell 的**仅含适配器**
         权重并经 ``peft.set_peft_model_state_dict`` 灌入该命名适配器（与单适配器
         路径 :func:`_apply_lora_adapter` 同源；Vault 落盘的是**裸 ``.safetensors``
         文件**而非 PEFT 适配器目录，故不能用需要 ``adapter_config.json`` 的
         ``PeftModel.from_pretrained`` / ``load_adapter`` 路径）；
       - base 已暴露 ``peft_config``（测试替身或预包裹）⇒ 经其
         ``load_adapter(path, adapter_name=...)`` 挂为命名适配器。
    2. **其余各 Cell**（顺序为 Kernel → 各分区 Level 低→高，由调用方排好序）：
       - 真实路径 ⇒ ``peft_model.add_adapter(<cell_name>, config)`` 追加同构命名
         适配器，再经 ``set_peft_model_state_dict`` 灌入其权重；
       - 测试替身 ⇒ ``peft_model.load_adapter(path, adapter_name=<cell_name>)``。
    3. 最后经 :func:`_set_active_adapters` **同时激活**全部活动适配器（PEFT 对多个
       激活适配器做加性组合，R5-AC4）。

    调用方对**纯 Flat 单 Kernel**的字节级兼容由 :class:`BackendInferenceSession`
    通过复用 :func:`_apply_lora_adapter` 单独处理（``_activate_cells`` 仅用于多 Cell
    并存激活路径）。

    全部 ML 依赖（``peft`` / ``safetensors``）在函数内惰性导入，保证本模块导入安全
    （需求 8.4）。

    :param base: ``backend.load_base_model`` 返回的 base 模型对象（未挂适配器）。
    :param cells: 有序的 ``(CellKey, 适配器绝对路径)`` 列表，Kernel 在首位。
    :param lora_rank: LoRA 秩（来自 ``config.json``，与训练时一致）。
    :param lora_alpha: LoRA alpha（来自 ``config.json``，与训练时一致）。
    :returns: 已加载并激活全部选定适配器的 PEFT 模型对象。
    """

    first_cell, first_path = cells[0]
    first_name = _cell_name(first_cell)

    if hasattr(base, "peft_config"):
        # base 已暴露 peft_config（测试替身 / 预包裹）：经 load_adapter 逐个挂载。
        peft_model = base
        peft_model.load_adapter(str(first_path), adapter_name=first_name)
        for cell, adapter_path in cells[1:]:
            peft_model.load_adapter(str(adapter_path), adapter_name=_cell_name(cell))
    else:
        # 真实路径：base 是裸的 transformers 模型，Vault 落盘的是裸 .safetensors
        # 适配器文件（无 adapter_config.json）。用 get_peft_model 构造同构骨架并经
        # set_peft_model_state_dict 逐 Cell 灌入权重（与 _apply_lora_adapter 同源）。
        from peft import (  # noqa: PLC0415 - 惰性导入保证导入安全
            LoraConfig,
            get_peft_model,
            set_peft_model_state_dict,
        )
        from safetensors.torch import load_file  # noqa: PLC0415

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            bias="none",
            task_type="CAUSAL_LM",
        )
        # 1. 首个（Kernel）适配器：构造同构 PEFT 骨架并灌入其适配器权重。
        peft_model = get_peft_model(base, lora_config, adapter_name=first_name)
        set_peft_model_state_dict(
            peft_model, load_file(str(first_path)), adapter_name=first_name
        )
        # 2. 其余各 Cell：追加同构命名适配器并灌入其权重（Kernel → 分区 Level 升序）。
        for cell, adapter_path in cells[1:]:
            name = _cell_name(cell)
            peft_model.add_adapter(name, lora_config)
            set_peft_model_state_dict(
                peft_model, load_file(str(adapter_path)), adapter_name=name
            )

    # 3. 同时激活全部活动适配器（加性并存激活，R5-AC4）。
    _set_active_adapters(peft_model, [_cell_name(cell) for cell, _ in cells])
    return peft_model


class BackendInferenceSession:
    """真实后端推理会话：据 ``Backend_Identifier`` 加载 base + 并存激活多 Cell 适配器。

    构造时按需求 6.1 解析与训练时匹配的后端（经
    :func:`_resolve_inference_backend`），调用 ``backend.load_base_model`` 加载
    Base_Model（返回 ``(model, tokenizer)``），再按**有序 Cell 列表**（Kernel →
    各分区 Level 低→高，由 :func:`~ego_pod.runtime.selection.select_cells` 排好序）
    以 Coexist_Activate 把各 Cell 适配器加性并存激活到冻结的 base（需求 5.4），并以
    确定性（贪心）:class:`~ego_pod.trainer.generation.DecodeSettings` 解码（需求
    6.2、6.3）。

    **向后兼容（单 Kernel Cell）**：当 ``cells`` 恰为单个 Kernel Cell（纯
    Flat_Package 或 kernel-only 退化包）时，复用 :func:`_apply_lora_adapter` 走
    与 MVP / real-trainer **字节级一致**的单适配器路径（需求 6.1、6.2）；多 Cell
    时走 :func:`_activate_cells` 的 PEFT 多适配器 ``load_adapter`` + ``set_adapter``
    路径（需求 5.4）。:meth:`activate` / :meth:`deactivate` 维护
    :attr:`active_adapters` 并调用 ``set_adapter`` 实现可逆性（需求 5.5）。

    若加载阶段所需 Training_Dependencies 缺失或硬件不可用（``ImportError`` /
    ``NotImplementedError`` / ``RuntimeError`` / ``OSError``），构造抛出
    :class:`RuntimeError_`（含原因，需求 6.4）；已构造的 :class:`EgoError`
    （如后端抛出的 :class:`~ego_pod.exit_codes.TrainerError`）优先原样透传，
    不被二次包装。:meth:`close` 释放全部 Cell 适配器 + Base_Model 引用并尽力
    回收显存（需求 5.9、6.5）。
    """

    def __init__(
        self,
        *,
        base_model: str,
        quantization: str,
        lora_rank: int,
        lora_alpha: int,
        cells: list[tuple[CellKey, Path]] | None = None,
        adapter_path: Path | None = None,
        identifier: str | None = None,
        backend: Any | None = None,
    ) -> None:
        """加载 Base_Model 并以 Coexist_Activate 并存激活各 Cell 适配器（需求 5.4、6.1、6.4）。

        :param base_model: 基座模型标识，来自 ``config.json``。
        :param quantization: 量化方案，来自 ``config.json``（MVP 固定 ``4bit``）。
        :param lora_rank: LoRA 秩，来自 ``config.json``。
        :param lora_alpha: LoRA alpha，来自 ``config.json``。
        :param cells: 有序的 ``(CellKey, 适配器绝对路径)`` 列表，Kernel 在首位
            （由 :func:`~ego_pod.runtime.selection.select_cells` 排序）。各路径须已
            通过 :func:`_validate_adapter_file` 校验。
        :param adapter_path: **向后兼容**入参——单个 Kernel 适配器路径；当 ``cells``
            缺省时据此构造 ``[(KERNEL, adapter_path)]``，复现 MVP 单适配器行为。
            ``cells`` 与 ``adapter_path`` 须至少提供其一。
        :param identifier: ``manifest.json`` 的 ``training_backend`` 标识，据此经
            :func:`_resolve_inference_backend` 选择匹配的真实后端（需求 6.1、7.5）。
        :param backend: 显式指定的后端（便于测试）；提供时优先于 ``identifier``。
            ``identifier`` 与 ``backend`` 均缺省时回退到 :func:`detect_backend`。
        :raises RuntimeError_: 加载 Base_Model 或应用/激活 LoRA 适配器时依赖缺失或
            硬件不可用（需求 6.4）。
        """
        self.base_model = base_model
        self.quantization = quantization
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.identifier = identifier
        self.tokenizer = None
        self.adapter_tensors = None

        # 归一为有序 Cell 列表：cells 优先；否则用 adapter_path 构造单 Kernel Cell
        # 以兼容既有调用方（MVP 单适配器路径）。
        if cells is not None:
            self.cells: list[tuple[CellKey, Path]] = [
                (cell, Path(path)) for cell, path in cells
            ]
        elif adapter_path is not None:
            self.cells = [(KERNEL, Path(adapter_path))]
        else:
            raise RuntimeError_(
                "BackendInferenceSession 需要 cells 或 adapter_path 之一"
            )
        # 向后兼容属性：保留首个（Kernel）适配器路径供既有调用方引用。
        self.adapter_path = self.cells[0][1]

        try:
            # 据 Backend_Identifier 选择与训练时匹配的后端（需求 6.1、7.5）。
            if backend is not None:
                self.backend = backend
            elif identifier is not None:
                self.backend = _resolve_inference_backend(identifier)
            else:
                self.backend = detect_backend()

            # 加载 base（依赖/硬件缺失 → 由下方 except 包装为 RuntimeError_）。
            loaded = self.backend.load_base_model(base_model, quantization)
            if isinstance(loaded, tuple) and len(loaded) == 2:
                base, self.tokenizer = loaded
            else:
                base, self.tokenizer = loaded, None

            # 单 Kernel Cell ⇒ 复用 _apply_lora_adapter，保持与 Flat_Package 字节级
            # 一致的单适配器行为（需求 6.1、6.2）；多 Cell ⇒ Coexist_Activate（需求 5.4）。
            if len(self.cells) == 1 and self.cells[0][0].is_kernel:
                self.model = _apply_lora_adapter(
                    base, self.cells[0][1], lora_rank, lora_alpha
                )
            else:
                self.model = _activate_cells(
                    base, self.cells, lora_rank, lora_alpha
                )

            # 活动适配器集合（按 Cell 顺序，Kernel 在前）——支持可逆激活（需求 5.5）。
            self.active_adapters: list[str] = [
                _cell_name(cell) for cell, _ in self.cells
            ]

            # 确定性（贪心）解码设置（需求 6.3）。
            self.decode = DecodeSettings()
        except EgoError:
            # OfflineMissingWeightsError / 后端 TrainerError 等原样透传，
            # 保持各自退出码语义不被二次包装（与 trainer 各模块的处理一致）。
            raise
        except (
            ImportError,
            NotImplementedError,
            RuntimeError,
            OSError,
            AttributeError,
            TypeError,
            ValueError,
        ) as exc:
            raise RuntimeError_(
                "加载真实推理后端失败（缺失依赖、硬件不可用或模型不兼容）："
                f"{exc}"
            ) from exc

    def deactivate(self, cell: CellKey) -> None:
        """停用某 Cell 适配器，并以剩余活动集合重置 ``set_adapter``（可逆性，需求 5.5）。

        从 :attr:`active_adapters` 移除该 Cell 的适配器名后，按 :attr:`cells` 的
        规范顺序重建活动列表并调用 ``model.set_adapter(active)``。该 Cell 不在活动
        集合中时为幂等空操作。

        :param cell: 要停用的 Cell 键。
        """
        name = _cell_name(cell)
        active = set(self.active_adapters)
        if name not in active:
            return
        active.discard(name)
        self.active_adapters = [
            _cell_name(c) for c, _ in self.cells if _cell_name(c) in active
        ]
        _set_active_adapters(self.model, list(self.active_adapters))

    def activate(self, cell: CellKey) -> None:
        """（重新）激活某 Cell 适配器，并以更新后的活动集合重置 ``set_adapter``（需求 5.5）。

        把该 Cell 的适配器名加入活动集合后，按 :attr:`cells` 的规范顺序（Kernel →
        各分区 Level 低→高）重建活动列表并调用 ``model.set_adapter(active)``，使
        「激活某 Cell 后再停用」能让活动集合精确回到先前状态（可逆性）。

        :param cell: 要激活的 Cell 键。
        """
        name = _cell_name(cell)
        active = set(self.active_adapters)
        active.add(name)
        self.active_adapters = [
            _cell_name(c) for c, _ in self.cells if _cell_name(c) in active
        ]
        _set_active_adapters(self.model, list(self.active_adapters))

    def generate(self, user_input: str) -> str:
        """对一次非空用户输入生成回复（需求 6.2、6.3）。

        委托 :func:`ego_pod.trainer.generation.generate_reply` 以确定性（贪心）
        解码生成回复：相同会话、相同输入与相同 :class:`DecodeSettings` 下逐次
        返回一致的回复（需求 6.3）。

        :param user_input: 非空用户输入文本。
        :returns: 模型生成的回复字符串。
        """
        return generate_reply(self.model, self.tokenizer, user_input, self.decode)

    def close(self) -> None:
        """释放会话持有的全部 Cell 适配器与 Base_Model 资源（需求 5.9、6.5）。

        置空 ``model`` / ``tokenizer`` / ``adapter_tensors`` 引用、清空
        :attr:`active_adapters` 后调用 :func:`_release_torch_memory`
        （``gc.collect()`` + 可用时 ``torch.cuda.empty_cache()``）尽力回收内存与
        显存，从而释放本次加载的全部 Cell（Kernel 与各分区 Cell）适配器（需求 5.9）。
        """
        self.model = None
        self.tokenizer = None
        self.adapter_tensors = None
        self.active_adapters = []
        _release_torch_memory()


# ---------------------------------------------------------------------------
# 文件读取与完整性校验（需求 12.5）
# ---------------------------------------------------------------------------


def _relative_label(path: Path, base: Path) -> str:
    """返回 ``path`` 相对于 ``base`` 的相对路径字符串，用于错误信息（需求 12.5）。

    无法相对化（不在 ``base`` 子树内）时回退为绝对路径字符串。
    """
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except (ValueError, OSError):
        return str(path)


def _read_json_artifact(path: Path, label: str) -> dict:
    """读取并解析一个 Ego_Package JSON 产物文件（需求 12.5）。

    :param path: 目标文件绝对路径。
    :param label: 用于错误信息的相对路径标签。
    :returns: 解析得到的 JSON 对象（``dict``）。
    :raises RuntimeError_: 文件缺失、不可读、或非法 JSON（含非 ``dict`` 顶层）。
    """
    if not path.is_file():
        raise RuntimeError_(f"未找到 Ego_Package 文件：{label}")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError_(
            f"Ego_Package 文件读取失败：{label}；原因：{exc}"
        ) from exc
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RuntimeError_(
            f"Ego_Package 文件损坏（非法 JSON）：{label}；原因：{exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError_(
            f"Ego_Package 文件损坏（顶层不是 JSON 对象）：{label}"
        )
    return payload


def _validate_adapter_file(path: Path, label: str) -> None:
    """校验 LoRA 适配器 SafeTensors 文件存在且未损坏（需求 12.5）。

    通过 ``safetensors.safe_open`` 读取张量键以验证文件头合法；任何读取/解析
    错误均视为"损坏"。

    :param path: 适配器文件绝对路径。
    :param label: 用于错误信息的相对路径标签。
    :raises RuntimeError_: 文件缺失或损坏。
    """
    if not path.is_file():
        raise RuntimeError_(f"未找到 Ego_Package 文件：{label}")
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - safetensors 为必装依赖
        raise RuntimeError_(
            f"无法校验适配器文件 {label}：safetensors 不可用（{exc}）"
        ) from exc
    try:
        with safe_open(str(path), framework="numpy") as handle:
            # 读取键即触发文件头解析；损坏文件在此抛错。
            list(handle.keys())
    except Exception as exc:
        raise RuntimeError_(
            f"Ego_Package 文件损坏：{label}；原因：{exc}"
        ) from exc


def _load_adapter_tensors(path: Path) -> dict:
    """读取 LoRA 适配器的全部张量（以 numpy 数组形式）。

    供真实推理路径加载适配器权重。调用前应已通过 :func:`_validate_adapter_file`
    校验文件完整性。

    :param path: 适配器文件绝对路径。
    :returns: ``{张量名: numpy.ndarray}`` 映射。
    :raises RuntimeError_: 读取失败。
    """
    try:
        from safetensors.numpy import load_file
    except ImportError as exc:  # pragma: no cover - safetensors 为必装依赖
        raise RuntimeError_(
            f"无法读取适配器权重：safetensors 不可用（{exc}）"
        ) from exc
    try:
        return load_file(str(path))
    except Exception as exc:
        raise RuntimeError_(
            f"读取 LoRA 适配器权重失败：{path}；原因：{exc}"
        ) from exc


def _coerce_positive_int(value: Any, field: str, label: str) -> int:
    """把 ``config.json`` 中的 LoRA 整数参数转换为正整数（需求 12.1）。

    :param value: 原始取值。
    :param field: 字段名（用于错误信息）。
    :param label: ``config.json`` 的相对路径标签。
    :returns: 校验通过的正整数。
    :raises RuntimeError_: 取值缺失或非正整数（视为 ``config.json`` 损坏）。
    """
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError_(
            f"Ego_Package 文件损坏（{field} 非正整数）：{label}"
        )
    return value


# ---------------------------------------------------------------------------
# 加载入口
# ---------------------------------------------------------------------------


def _cell_label(cell: CellKey) -> str:
    """把一个 :class:`CellKey` 渲染为错误信息中的可读 Cell 标识（需求 5.7）。

    * Kernel（``zone=None, level=0``）⇒ ``"Kernel"``。
    * 分区 Cell ⇒ ``"(<zone>, L<level>)"``（如 ``"(coding, L1)"``）。

    :param cell: 网格中的一个 Cell 键。
    :returns: 适合放入 :class:`RuntimeError_` 消息的 Cell 标识字符串。
    """

    if cell.zone is None and cell.level == 0:
        return "Kernel"
    return f"({cell.zone}, L{cell.level})"


def load_inference_session(
    edition: str = DEFAULT_EDITION,
    *,
    cwd: Path | None = None,
    output_root: Path | None = None,
    backend: Any | None = None,
    stdout: TextIO | None = None,
    selection: LoadSelection | None = None,
) -> InferenceSession:
    """加载 Ego_Package 并返回一个推理会话（需求 5.1、5.6、5.7、5.10、6.1、12.5、12.9）。

    步骤（见模块文档与设计 §"ego run 的网格化加载流"）：读取并校验
    ``manifest.json`` / ``<edition>/config.json`` → 经
    :func:`~ego_pod.vault.grid_descriptor.parse_grid_descriptor` 解析 ``grid``
    描述（无 ``grid`` 键 ⇒ 单 Kernel Cell 的 Flat 网格，需求 6.1/6.2）→ 经
    :func:`~ego_pod.runtime.selection.select_cells` 据 ``selection`` 选出有序 Cell
    列表（默认 ``None`` ⇒ Full_In_Load，需求 5.1；非法深度/未知分区 ⇒
    :class:`RuntimeError_`，需求 5.6）→ 逐 Cell 校验其适配器文件存在且未损坏
    （缺失/损坏 ⇒ :class:`RuntimeError_` 含 Cell 标识与相对路径，需求 5.7）→ 据
    ``manifest.training_backend`` 选择 Stub 推理路径（打印启动提示，无论一个还是
    多个 Cell 行为不变，需求 5.10）或真实后端多适配器 Coexist_Activate 路径
    （需求 5.4）。

    本函数假定调用方（``commands/run.py``）已先行处理 Ego_Package 子目录不存在
    与 Edition 非 ``normal`` 的情形；本函数聚焦于文件级加载与完整性校验、网格
    选择与 Stub/真实路径分流。

    :param edition: Edition 名称，默认 :data:`DEFAULT_EDITION`（``normal``）。
    :param cwd: 进程工作目录，默认 :meth:`pathlib.Path.cwd`；用于定位
        ``ego-output/`` 并计算错误信息中的相对路径。
    :param output_root: 显式指定的 Ego_Package 根目录，默认 ``{cwd}/ego-output``。
    :param backend: 显式指定的真实后端（便于测试）；提供时优先于 manifest 的
        ``training_backend`` 标识（``Backend_Identifier``）。默认 ``None`` 时由
        :class:`BackendInferenceSession` 据 ``identifier`` 经
        :func:`_resolve_inference_backend` 选择与训练时匹配的后端（需求 6.1、7.5）。
        Stub 路径不使用此参数。
    :param stdout: Stub 启动提示的目标输出流，默认 ``sys.stdout``。
    :param selection: 一次加载选择（:class:`~ego_pod.runtime.selection.LoadSelection`）；
        默认 ``None`` ⇒ Full_In_Load（Kernel + 每分区 Level ``1..max``，需求 5.1）。
    :returns: 一个 :class:`InferenceSession`（Stub 或真实后端）。
    :raises RuntimeError_: ``manifest.json`` / ``config.json`` 缺失或损坏（需求
        12.5）；加载选择非法（需求 5.6）；某选定 Cell 的适配器文件缺失或损坏
        （需求 5.7）；或真实推理路径在 MVP 中未实现。
    """
    base_cwd = Path.cwd() if cwd is None else Path(cwd)
    root = (base_cwd / OUTPUT_DIRNAME) if output_root is None else Path(output_root)

    edition_dir = root / edition
    manifest_path = root / MANIFEST_FILENAME
    config_path = edition_dir / CONFIG_FILENAME

    manifest_label = _relative_label(manifest_path, base_cwd)
    config_label = _relative_label(config_path, base_cwd)

    # 1. 读取 manifest.json（需求 12.5）。
    manifest = _read_json_artifact(manifest_path, manifest_label)

    # 2. 读取 <edition>/config.json（需求 12.1、12.5）。
    config = _read_json_artifact(config_path, config_label)

    # 3. 解析 grid 描述（无 grid 键 ⇒ 单 Kernel Cell 的 Flat 网格，需求 6.1/6.2）。
    grid = parse_grid_descriptor(manifest.get("grid"), edition)

    # 4. 据 selection 选出有序 Cell 列表（默认 Full_In_Load，需求 5.1）；非法深度/
    #    未知分区 ⇒ select_cells 抛 RuntimeError_（需求 5.6），原样上抛。
    selected = select_cells(grid, selection)

    # 5. 逐 Cell 解析适配器绝对路径并校验存在且未损坏（需求 5.7）。
    #    grid 由 Kernel + 各分区 Cell 列表构成；按 relpath 定位文件。
    relpath_by_cell: dict[CellKey, str] = {grid.kernel.cell: grid.kernel.relpath}
    for zone_cells in grid.zones.values():
        for grid_cell in zone_cells:
            relpath_by_cell[grid_cell.cell] = grid_cell.relpath

    cells: list[tuple[CellKey, Path]] = []
    for cell in selected:
        relpath = relpath_by_cell[cell]
        adapter_path = edition_dir / relpath
        # 错误信息含 Cell 标识与文件相对路径（需求 5.7）。
        cell_error_label = f"{_cell_label(cell)} → {relpath}"
        _validate_adapter_file(adapter_path, cell_error_label)
        cells.append((cell, adapter_path))

    # 6. 据 manifest.training_backend 分流（需求 5.10、12.9）。
    training_backend = manifest.get("training_backend")
    if training_backend == STUB_TRAINING_BACKEND:
        # Stub 推理路径：启动时打印 Stub 模式提示；无论一个还是多个 Cell 行为不变
        # （需求 5.10）。
        target = sys.stdout if stdout is None else stdout
        print(STUB_MODE_STARTUP_NOTICE, file=target, flush=True)
        return StubInferenceSession()

    # 7. 真实后端推理路径：把 manifest.training_backend 作为 Backend_Identifier
    #    传入，经 _resolve_inference_backend 选择与训练时匹配的真实后端，加载
    #    base + 按有序 Cell 列表 Coexist_Activate 并存激活各 Cell 适配器（需求 5.4、6.1）。
    base_model = config.get("base_model", DEFAULT_BASE_MODEL)
    quantization = config.get("quantization", DEFAULT_QUANTIZATION)
    lora_rank = _coerce_positive_int(config.get("lora_rank"), "lora_rank", config_label)
    lora_alpha = _coerce_positive_int(
        config.get("lora_alpha"), "lora_alpha", config_label
    )

    try:
        return BackendInferenceSession(
            base_model=base_model,
            quantization=quantization,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            cells=cells,
            identifier=training_backend,
            backend=backend,
        )
    except RuntimeError_:
        raise
    except NotImplementedError as exc:
        raise RuntimeError_(
            "真实后端推理在 Ego Pod MVP 中尚未实现"
            f"（training_backend={training_backend!r}）：{exc}"
        ) from exc
