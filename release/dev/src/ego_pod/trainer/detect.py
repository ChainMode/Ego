"""Training_Backend 探测与优先级选择（需求 8.1、8.2）。

本模块实现 :func:`detect_backend`：按设计文档「多后端 Trainer 抽象」一节
规定的固定优先级顺序逐个探测候选后端，返回第一个 ``is_available()`` 为
``True`` 的后端实例。Stub_Backend 始终视为可用，作为最终回退，从而保证在
无任何机器学习依赖的环境（如 CI）中管线仍可端到端跑通。

探测优先级（需求 8.1）：

==========  ==================  ====================================================
优先级       后端                可用性判定
==========  ==================  ====================================================
1           ``UnslothBackend``   ``import unsloth`` 成功 + CUDA 可用
2           ``PeftCudaBackend``  ``import peft, torch`` 成功 + CUDA 可用
3           ``MlxBackend``       ``import mlx_lm`` 成功 + Apple Silicon（arm64 macOS）
4           ``PeftCpuBackend``   ``import peft, torch`` 成功（不要求 CUDA）
5           ``StubBackend``      永远可用（最终回退）
==========  ==================  ====================================================

可用性的具体判定由各后端自身的 ``is_available()`` 类方法负责（见
``trainer.base.TrainingBackend`` Protocol），各后端可复用本模块提供的
:func:`cuda_is_available` 与 :func:`apple_silicon_is_available` 硬件检测工具。

为支持各后端模块在缺失对应依赖时无法 ``import`` 的情形，本模块对每个候选
后端的模块导入均以 ``try/except ImportError`` 包裹（见 :func:`_load_backend_class`）：
任一候选模块不可导入或尚未存在，仅意味着该候选被跳过，探测继续向下进行，
最终安全回退到 Stub_Backend。因此本模块自身的导入不依赖任何可选后端模块或
机器学习依赖的存在。
"""

from __future__ import annotations

import importlib
import platform
import sys
from typing import TYPE_CHECKING

from ego_pod.exit_codes import TrainerError

if TYPE_CHECKING:  # 仅用于类型注解，避免运行时硬依赖后端模块
    from ego_pod.trainer.base import TrainingBackend

__all__ = [
    "cuda_is_available",
    "apple_silicon_is_available",
    "detect_backend",
]


# ---------------------------------------------------------------------------
# 硬件检测工具（CUDA / Apple Silicon）
# ---------------------------------------------------------------------------


def cuda_is_available() -> bool:
    """检测当前环境是否存在可用的 NVIDIA CUDA 设备。

    依赖 ``torch.cuda.is_available()``。``torch`` 虽是运行时必装依赖，本函数
    仍以 ``try/except`` 包裹其导入与探测，使得在 torch 缺失或 CUDA 驱动异常
    （如部分 CPU-only 构建在查询时抛错）时安全降级为 ``False`` 而非崩溃。

    :returns: 存在可用 CUDA 设备时返回 ``True``，否则返回 ``False``。
    """
    try:
        import torch
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        # 某些环境下 CUDA 驱动/运行时异常会在查询时抛出非 ImportError 异常，
        # 一律视为 CUDA 不可用。
        return False


def apple_silicon_is_available() -> bool:
    """检测当前是否运行于 Apple Silicon（arm64 架构的 macOS）。

    用于 MLX 后端的硬件前置条件判定：MLX 仅在 Apple Silicon 上可用。

    :returns: 当前为 arm64 macOS 时返回 ``True``，否则返回 ``False``。
    """
    return sys.platform == "darwin" and platform.machine() == "arm64"


# ---------------------------------------------------------------------------
# 候选后端规格与探测逻辑
# ---------------------------------------------------------------------------

#: 候选后端规格 ``(模块路径, 类名)``，按需求 8.1 的优先级从高到低排列。
#: 末位的 Stub_Backend 作为始终可用的最终回退（需求 8.1）。
#: ``PeftCudaBackend`` 与 ``PeftCpuBackend`` 同处于 ``peft_backend`` 模块，
#: 二者通过各自的 ``is_available()``（是否要求 CUDA）区分优先级 2 与 4。
_BACKEND_SPECS: tuple[tuple[str, str], ...] = (
    ("ego_pod.trainer.unsloth_backend", "UnslothBackend"),
    ("ego_pod.trainer.peft_backend", "PeftCudaBackend"),
    ("ego_pod.trainer.mlx_backend", "MlxBackend"),
    ("ego_pod.trainer.peft_backend", "PeftCpuBackend"),
    ("ego_pod.trainer.stub", "StubBackend"),
)


def _load_backend_class(module_path: str, class_name: str) -> type | None:
    """defensively 加载候选后端类。

    将模块导入以 ``try/except ImportError`` 包裹：当后端模块尚未存在、或其
    在导入期因缺失可选依赖（如 ``import unsloth`` 失败）而抛出 ``ImportError``
    （含子类 ``ModuleNotFoundError``）时，返回 ``None`` 表示该候选不可用，
    由调用方跳过并继续探测下一候选。

    :param module_path: 后端所在模块的完整导入路径。
    :param class_name: 后端类在该模块中的属性名。
    :returns: 成功加载到的后端类；模块不可导入或类不存在时返回 ``None``。
    """
    try:
        module = importlib.import_module(module_path)
    except ImportError:
        return None
    return getattr(module, class_name, None)


def detect_backend() -> "TrainingBackend":
    """按优先级探测并返回第一个可用的 Training_Backend 实例（需求 8.1）。

    遍历 :data:`_BACKEND_SPECS` 中的候选后端，对每个候选：

    1. 以 :func:`_load_backend_class` defensively 加载后端类，加载失败则跳过；
    2. 调用其 ``is_available()`` 类方法判定可用性（探测期间的任何异常均视为
       不可用并跳过）；
    3. 返回首个判定为可用的后端的实例。

    Stub_Backend 位于候选末位且 ``is_available()`` 永远返回 ``True``，因此在
    所有真实后端均不可用时充当最终回退。

    :returns: 选中的 Training_Backend 实例，其 ``name`` 与 ``hardware`` 属性供
        ``trainer.pipeline`` 打印「Selected backend: <name> (<hardware>)」记录。
    :raises TrainerError: 当连 Stub_Backend 都无法加载（安装损坏）时抛出。
        正常安装下不会发生，因为 Stub_Backend 不依赖任何可选组件。
    """
    for module_path, class_name in _BACKEND_SPECS:
        backend_cls = _load_backend_class(module_path, class_name)
        if backend_cls is None:
            continue
        try:
            available = bool(backend_cls.is_available())
        except Exception:
            # 后端可用性探测（含硬件查询）抛出的任何异常均视为该后端不可用。
            available = False
        if available:
            return backend_cls()

    raise TrainerError(
        "未找到任何可用的 Training_Backend；Stub_Backend 应始终可用作为最终回退，"
        "当前环境可能存在安装损坏（trainer.stub 模块缺失或不可导入）。"
    )
