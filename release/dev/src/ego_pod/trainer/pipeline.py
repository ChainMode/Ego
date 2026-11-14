"""Trainer 编排（需求 8.2、8.3、8.4、8.6、8.7、8.8、8.9、9.4、9.5、9.7）。

本模块实现 :func:`run_training`：把 ``detect`` 探测出的 Training_Backend、
``base`` 定义的后端接口、``stub`` 兜底实现与 ``privacy.offline`` 离线守卫装配为
一条完整的 LoRA 训练流水线，对应设计文档 §5「Trainer 模块」的 Pipeline 步骤：

1. **探测后端**：调用 :func:`ego_pod.trainer.detect.detect_backend` 选择首个可用
   后端，并在标准输出打印一行 ``Selected backend: <name> (<hardware>)``（R8-AC2）。
2. **校验五个 LoRA 参数**：``lora_rank`` / ``lora_alpha`` / ``learning_rate`` /
   ``epochs`` / ``batch_size`` 缺失或非法时抛 :class:`~ego_pod.exit_codes.TrainerError`
   （退出码 1，R8-AC9）。该校验是 Trainer 层的防御性复核——Config 校验阶段
   （``config.validators``）已对同一组参数做过预检，但 R8-AC9 把该约束归属于
   Trainer，故此处再次把关。
3. **加载基座**（R8-AC3）：
   - **离线模式**（``--offline`` / ``EGO_OFFLINE=1`` 启用）下，在调用后端加载之前
     先经 :meth:`~ego_pod.privacy.offline.OfflineMode.require_base_model_available`
     检查本地 Base_Model 权重是否可用；缺失时由 ``privacy.offline`` 抛
     :class:`~ego_pod.privacy.offline.OfflineMissingWeightsError`，本模块不拦截，
     交由 CLI 顶层按 R9-AC7 处理（stderr 固定文案 + 非零退出码）。
   - **在线模式**（默认）下后端加载失败 → 按 R8-AC7 包装为 ``TrainerError``
     （携带失败原因，CLI 顶层打印到 stderr 并以非零退出码结束）；此场景**不**触发
     R9-AC7。
4. **训练循环**（R8-AC8）：把 :func:`make_progress_callback` 构造的进度回调传给
   ``backend.train``；后端至少每完成一个 epoch 调用一次回调，回调经
   ``logging_io.print_summary`` 在标准输出打印固定格式的进度记录。
5. **导出权重**（R8-AC6）：以 SafeTensors 格式导出 LoRA_Adapter 到
   ``{cwd}/.ego/lora.safetensors`` 临时目录，供 Vault 持久化。

返回的 :class:`TrainingResult` 携带导出路径与所选后端名称/硬件，供 ``ego train``
编排层（任务 11.1）把权重交给 Vault，并由 Vault 把后端名称写入 ``manifest.json``
的 ``training_backend`` 标记（stub 后端据此让 ``ego run`` 进入 Stub REPL，需求 12.9）。

隐私约束（R9-AC4/AC5）：本模块只在本地把语料路径与张量交给后端计算，不调用任何
HTTP 客户端发送原始训练数据；基座权重的下载（若发生）由后端经 huggingface_hub
完成，请求体不附带训练数据。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TextIO

from ego_pod.config.schema import TrainingParams
from ego_pod.exit_codes import EgoError, TrainerError
from ego_pod.logging_io import print_summary
from ego_pod.privacy.offline import OfflineMode, get_active_offline_mode
from ego_pod.trainer.base import (
    BackendIdentifier,
    Hardware,
    ProgressCallback,
    TrainingBackend,
    resolve_identifier,
)
from ego_pod.trainer.detect import detect_backend

__all__ = [
    "TrainingResult",
    "make_progress_callback",
    "run_training",
]

#: MVP 唯一支持的基座模型标识（与 ``config.validators`` 中的 R3-AC7 约束一致）。
DEFAULT_MODEL_ID = "qwen3.5-0.8B"

#: LoRA 适配器权重的临时导出文件名，落于 ``{cwd}/.ego/`` 下供 Vault 取用。
LORA_OUTPUT_FILENAME = "lora.safetensors"


@dataclass(frozen=True)
class TrainingResult:
    """:func:`run_training` 的结果。

    :ivar output_path: 导出的 LoRA_Adapter SafeTensors 文件绝对路径
        （``{cwd}/.ego/lora.safetensors``），供 Vault 持久化（R8-AC6）。
    :ivar backend_name: 所选 Training_Backend 的可读名称，供 Vault 写入
        ``manifest.json`` 的 ``training_backend`` 标记（需求 12.9）。
    :ivar backend_hardware: 所选后端的硬件类型（CUDA / Apple Silicon / CPU /
        STUB）。
    :ivar backend_identifier: 所选后端的稳定 Backend_Identifier（取值集合
        ``{unsloth, peft-cuda, mlx, peft-cpu, stub}``），供 ``ego train`` 编排层
        把其作为 ``training_backend`` 写入 ``manifest.json``，使 Runtime 据此选择
        匹配的推理路径（R7-AC1、R7-AC2、R7-AC6）。新增的带默认值字段，保持
        ``TrainingResult`` 对外构造的向后兼容（R10-AC2）。
    """

    output_path: Path
    backend_name: str
    backend_hardware: Hardware
    backend_identifier: BackendIdentifier = "stub"


# ---------------------------------------------------------------------------
# LoRA 参数校验（R8-AC9）
# ---------------------------------------------------------------------------

#: 五个 LoRA 训练参数的检测顺序与各自约束类型，与 ``config.validators`` 保持一致。
#: ``"pos_int"`` 表示正整数；``"pos_num"`` 表示大于 0 的数值。
_TRAINING_PARAM_SPECS: tuple[tuple[str, str], ...] = (
    ("lora_rank", "pos_int"),
    ("lora_alpha", "pos_int"),
    ("learning_rate", "pos_num"),
    ("epochs", "pos_int"),
    ("batch_size", "pos_int"),
)

_CONSTRAINT_TEXT = {
    "pos_int": "必须为正整数",
    "pos_num": "必须为大于 0 的数值",
}


def _is_pos_int(value: Any) -> bool:
    """是否为正整数（排除 ``bool``，因其在 Python 中是 ``int`` 的子类）。"""

    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_pos_num(value: Any) -> bool:
    """是否为大于 0 的数值（``int`` 或 ``float``，排除 ``bool``）。"""

    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value > 0
    )


def _validate_training_params(params: TrainingParams) -> None:
    """R8-AC9：校验五个 LoRA 训练参数的完备性与合法性。

    按 lora_rank → lora_alpha → learning_rate → epochs → batch_size 顺序检测，
    报告首个缺失或非法的参数名并抛 :class:`TrainerError`（退出码 1）。
    lora_rank/lora_alpha/epochs/batch_size 必须为正整数，learning_rate 必须为
    大于 0 的数值。以 ``getattr(..., None)`` 取值，使"字段缺失"与"取值非法"统一
    走同一报错分支。
    """

    for name, kind in _TRAINING_PARAM_SPECS:
        value = getattr(params, name, None)
        valid = _is_pos_int(value) if kind == "pos_int" else _is_pos_num(value)
        if not valid:
            raise TrainerError(
                f"训练参数 '{name}' 缺失或非法（{_CONSTRAINT_TEXT[kind]}）"
            )


# ---------------------------------------------------------------------------
# 进度回调（R8-AC8）
# ---------------------------------------------------------------------------


def make_progress_callback(
    *, stream: TextIO | None = None
) -> ProgressCallback:
    """构造训练进度回调（R8-AC8）。

    返回的回调被后端在训练期间（至少每完成一个 epoch）调用一次，经
    :func:`ego_pod.logging_io.print_summary` 在标准输出打印一行包含当前 epoch
    编号、累计训练步数与当前 epoch 训练损失值的进度记录。

    回调以关键字参数 ``epoch`` / ``step`` / ``loss`` 接收数据，与
    ``trainer.stub.StubBackend.train`` 的调用约定一致。

    :param stream: 进度记录的目标输出流，默认 ``sys.stdout``。
    """

    def _callback(epoch: int, step: int, loss: float) -> None:
        print_summary(
            f"[progress] epoch={epoch} step={step} loss={float(loss):.4f}",
            stream=stream,
        )

    return _callback


# ---------------------------------------------------------------------------
# 离线模式下的基座权重可用性
# ---------------------------------------------------------------------------


def _base_model_available(
    backend: TrainingBackend,
    model_id: str,
    *,
    check: Callable[[str], bool] | None,
) -> bool:
    """判定离线模式下 Base_Model 权重是否可用。

    - Stub_Backend（``hardware == "STUB"``）不下载、不加载任何真实权重（设计 §5），
      其端到端可跑通正是为无 ML 依赖/无权重缓存的环境（如 CI）设计的，因此对 stub
      而言权重恒"可用"，离线模式下不应因缺失真实权重而短路。
    - 其余真实后端：优先使用注入的 ``check`` 回调（便于测试与未来定制）；未注入时
      回退到 :func:`_hf_cache_contains` 扫描本地 HuggingFace 缓存目录。
    """

    if backend.hardware == "STUB":
        return True
    if check is not None:
        return bool(check(model_id))
    return _hf_cache_contains(model_id)


def _hf_cache_contains(model_id: str) -> bool:
    """尽力检查本地 HuggingFace 缓存目录中是否已存在目标基座权重。

    通过 ``huggingface_hub.scan_cache_dir`` 扫描默认缓存目录（通常为
    ``~/.cache/huggingface/hub/``）。``huggingface_hub`` 缺失或扫描过程中发生
    任何异常时一律保守返回 ``False``（视为不可用），以便离线模式下据此触发
    R9-AC7。本函数不发起任何网络请求。
    """

    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        return False
    try:
        cache_info = scan_cache_dir()
    except Exception:
        # 缓存目录不存在/无权限/格式异常等：保守视为不可用。
        return False
    needle = model_id.lower()
    for repo in cache_info.repos:
        if needle in repo.repo_id.lower():
            return True
    return False


# ---------------------------------------------------------------------------
# 编排入口
# ---------------------------------------------------------------------------


def run_training(
    corpus_path: Path,
    params: TrainingParams,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    cwd: Path | None = None,
    output_path: Path | None = None,
    backend: TrainingBackend | None = None,
    offline: OfflineMode | None = None,
    base_model_available_check: Callable[[str], bool] | None = None,
    stdout: TextIO | None = None,
) -> TrainingResult:
    """执行 LoRA 训练流水线并返回导出结果。

    步骤（见模块文档）：探测后端 → 打印所选后端 → 校验五个 LoRA 参数 → 加载基座
    （离线缺权重走 R9-AC7、在线加载失败走 R8-AC7）→ 训练循环（逐 epoch 进度记录）
    → 导出 ``{cwd}/.ego/lora.safetensors``。

    :param corpus_path: Conversation_Corpus（JSONL）中间文件路径，由 Compiler 产出。
    :param params: 来自 Ego_Config 的训练参数，提供 ``quantization`` 与五个 LoRA 参数。
    :param model_id: 基座模型标识，MVP 固定为 ``qwen3.5-0.8B``（R8-AC3）。
    :param cwd: 进程工作目录，默认 :meth:`pathlib.Path.cwd`；导出路径以 ``cwd/.ego``
        为根（隐私约束 R9-AC4：产物仅落于工作目录内）。
    :param output_path: 显式指定的导出路径；默认 ``{cwd}/.ego/lora.safetensors``。
    :param backend: 显式指定的 Training_Backend；默认由 :func:`detect_backend` 探测。
    :param offline: 离线模式状态；默认回退到 :func:`get_active_offline_mode` 注册的
        进程内活动模式（``None`` 等价于在线模式）。
    :param base_model_available_check: 离线模式下判定基座权重是否可用的回调
        （签名 ``check(model_id) -> bool``）；默认扫描本地 HuggingFace 缓存。
        对 Stub_Backend 不生效（stub 恒视为可用）。
    :param stdout: 后端选择记录与进度记录的目标输出流，默认 ``sys.stdout``。
    :returns: :class:`TrainingResult`，含导出路径与所选后端名称/硬件。
    :raises TrainerError: 五个 LoRA 参数缺失或非法（R8-AC9）；或在线模式下基座加载
        失败（R8-AC7）。
    :raises OfflineMissingWeightsError: 离线模式下基座权重缓存缺失（R9-AC7，由
        ``privacy.offline`` 抛出，本模块不拦截）。
    """

    base_cwd = Path.cwd() if cwd is None else Path(cwd)
    target_path = (
        base_cwd / ".ego" / LORA_OUTPUT_FILENAME
        if output_path is None
        else Path(output_path)
    )

    # 1. 探测后端并在标准输出打印所选后端（R8-AC1 / R8-AC2）。
    selected = backend if backend is not None else detect_backend()
    print_summary(
        f"Selected backend: {selected.name} ({selected.hardware})",
        stream=stdout,
    )

    # 2. 校验五个 LoRA 参数（缺失/非法 → 退出码 1，R8-AC9）。
    _validate_training_params(params)

    # 3. 离线模式下加载基座前先检查权重可用性；缺失由 privacy.offline 抛
    #    OfflineMissingWeightsError，交由 CLI 顶层按 R9-AC7 处理（本模块不拦截）。
    active_offline = offline if offline is not None else get_active_offline_mode()
    if active_offline is not None and active_offline.enabled:
        available = _base_model_available(
            selected, model_id, check=base_model_available_check
        )
        active_offline.require_base_model_available(available)

    # 4. 加载基座（R8-AC3）。在线模式下加载失败按 R8-AC7 包装为 TrainerError；
    #    EgoError 子类（含 OfflineMissingWeightsError / TrainerError）原样上抛，
    #    以保持各自的退出码语义不被覆盖。
    try:
        model = selected.load_base_model(model_id, params.quantization)
    except EgoError:
        raise
    except Exception as exc:
        raise TrainerError(f"Base_Model 加载失败：{exc}") from exc

    # 5. 训练循环（R8-AC8）：进度回调由后端至少每 epoch 调用一次。
    progress_callback = make_progress_callback(stream=stdout)
    selected.train(model, corpus_path, params, progress_callback)

    # 6. 导出 LoRA_Adapter 到 {cwd}/.ego/lora.safetensors（R8-AC6）。
    target_path.parent.mkdir(parents=True, exist_ok=True)
    selected.export_lora_safetensors(model, target_path)

    return TrainingResult(
        output_path=target_path,
        backend_name=selected.name,
        backend_hardware=selected.hardware,
        backend_identifier=resolve_identifier(selected),
    )
