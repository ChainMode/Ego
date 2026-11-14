"""框架无关的 QLoRA 训练循环（需求 4.1、4.2、4.3、4.5、4.7）。

本模块实现一个**手写**的、对"torch 模型 + PEFT 适配器"通用的 QLoRA 训练
循环（见设计文档「关键架构决策 3」：手写循环而非 HuggingFace ``Trainer``），
由 :class:`ego_pod.trainer.peft_backend._PeftBackendBase` 与
:class:`ego_pod.trainer.unsloth_backend.UnslothBackend` 共用。

提供三个对外能力：

- :func:`set_deterministic_seed`：为 ``random`` / ``numpy`` / ``torch``（含
  ``torch.cuda``）设定确定性种子并尽力开启确定性算子（R4-AC5）。
- :func:`assert_finite_loss`：损失为 ``NaN`` / ``Inf`` 时抛
  :class:`~ego_pod.exit_codes.TrainerError`（R4-AC7）。
- :func:`train_qlora`：逐 epoch 执行训练循环，每 epoch 恰好回调一次
  ``progress_callback``，仅更新 ``requires_grad=True`` 的参数（PEFT 已冻结
  base 骨干，故只更新 LoRA，R4-AC3），并把 OOM 与数值发散统一包装为
  ``TrainerError``（R4-AC7）。

**导入安全（R8-AC4）**：``torch`` / ``numpy`` 等重型可选依赖一律在函数内部
**惰性导入**（``try/except ImportError`` 返回 ``None`` 哨兵），模块顶层只依赖
标准库与 import-safe 的 ``ego_pod`` 子模块。因此本模块在**未安装任何 ML
依赖**的环境下也能被安全导入，便于属性测试在无 ML 依赖的 CI 中运行。

鸭子类型契约（duck-typed interface）
-----------------------------------
:func:`train_qlora` 同时服务于**真实路径**（torch ``nn.Module`` + PEFT 适配器）
与**测试路径**（轻量假对象），二者只需满足以下结构契约：

``model``（模型对象）

- ``model.named_parameters()`` → 产出 ``(name, param)`` 对；每个 ``param`` 具备
  布尔属性 ``requires_grad``。真实 torch ``Parameter`` 与假参数对象均满足。
  优化器仅收集 ``requires_grad=True`` 且为真实 ``torch.Tensor`` 的参数；因此
  对**假模型**而言优化器为 ``None``（无真实反向传播），循环仍会执行前向、
  有限性检查、OOM 处理与逐 epoch 回调。
- ``model(**batch)``（``batch`` 为 ``dict``，真实路径）或 ``model(batch)``
  （``batch`` 为 ``list``，自定义/假路径）→ 返回带 ``.loss`` 属性的输出对象
  （HuggingFace 约定），或直接返回可经 ``float(...)`` 转为标量的损失值。
  真实 HF 模型返回的 ``.loss`` 为标量张量（具 ``.backward`` / ``.item``）；
  假模型可返回一个**可控**的损失（普通 ``float`` 或预置序列中的下一个值），
  用于在不依赖真实训练的前提下测试循环控制流（逐 epoch 回调、NaN/Inf 检测、
  OOM→TrainerError 包装）。
- 仅当 ``model`` 为真实 ``torch.nn.Module`` 时调用 ``model.train()`` 切换训练
  模式；假模型不会被调用该方法。

``dataset``（数据集）

- 支持 ``len(dataset)`` 与下标访问 ``dataset[i]``（如 ``list``）。
- 元素为 :class:`ego_pod.trainer.sft.SftExample`（具 ``input_ids`` /
  ``attention_mask`` / ``labels``）时，本模块按"pad 到 batch 内最长、pad 位置
  ``attention_mask=0`` 且 ``labels=IGNORE_INDEX``"组成 batch；真实路径下
  ``torch`` 可用则组成张量 ``dict``，否则组成等价的 Python ``list`` ``dict``。
- 元素不具 ``input_ids`` 时（自定义/假数据集），整批以 ``list`` 原样传给
  ``model``，由模型自行解释（便于测试只关心损失序列而非张量内容）。

``progress_callback``

- 签名见 :data:`ego_pod.trainer.base.ProgressCallback`，以关键字参数
  ``epoch`` / ``step`` / ``loss`` 调用，与 ``trainer.pipeline.make_progress_callback``
  / ``trainer.stub.StubBackend.train`` 的约定一致。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

from ego_pod.config.schema import TrainingParams
from ego_pod.exit_codes import TrainerError
from ego_pod.trainer.base import ProgressCallback

__all__ = [
    "IGNORE_INDEX",
    "TrainingMetrics",
    "set_deterministic_seed",
    "assert_finite_loss",
    "train_qlora",
]


#: 监督训练中"不计损失"的标签哨兵值（HuggingFace 约定）。与
#: :data:`ego_pod.trainer.sft.IGNORE_INDEX` 取值一致；在此本地定义以保持
#: 本模块对 ``sft`` 的导入解耦与导入安全。collate 时用于填充 pad 位置的
#: ``labels``。
IGNORE_INDEX = -100


@dataclass(frozen=True)
class TrainingMetrics:
    """:func:`train_qlora` 的训练结果度量。

    :ivar completed_epochs: 已完成的 epoch 数（正常完成时等于
        ``params.epochs``）。
    :ivar total_steps: 全程累计训练步数（每个 batch 计一步）。
    :ivar final_loss: 最后一个 epoch 的平均训练损失值。
    """

    completed_epochs: int
    total_steps: int
    final_loss: float


# ---------------------------------------------------------------------------
# 惰性依赖导入（导入安全：顶层永不依赖 torch / numpy）
# ---------------------------------------------------------------------------


def _try_import_torch():
    """惰性导入 ``torch``；不可用时返回 ``None``。"""

    try:
        import torch  # noqa: PLC0415 - 故意惰性导入以保证导入安全
    except ImportError:
        return None
    return torch


def _try_import_numpy():
    """惰性导入 ``numpy``；不可用时返回 ``None``。"""

    try:
        import numpy  # noqa: PLC0415 - 故意惰性导入以保证导入安全
    except ImportError:
        return None
    return numpy


# ---------------------------------------------------------------------------
# 确定性种子（R4-AC5）
# ---------------------------------------------------------------------------


def set_deterministic_seed(seed: int) -> None:
    """为各随机源设定确定性种子并尽力开启确定性算子（R4-AC5）。

    始终为标准库 ``random`` 设定种子；``numpy`` / ``torch`` 可导入时一并设定
    （含 ``torch.cuda`` 全部设备）。此外尽力开启 torch 的确定性算子
    （``torch.use_deterministic_algorithms(True, warn_only=True)``）并固定 cudnn，
    使得在相同 Base_Model、相同语料、相同 ``params`` 与相同种子下重复训练产出
    语义等价的 LoRA_Adapter。

    注意：CUDA 上某些算子内核仍可能非确定（见随包文档对 R4-AC6 的说明），
    本函数尽力而为，对任何底层异常均吞掉以保证不因环境问题中断训练。

    :param seed: 确定性种子。
    """

    random.seed(seed)

    numpy = _try_import_numpy()
    if numpy is not None:
        numpy.random.seed(seed)

    torch = _try_import_torch()
    if torch is None:
        return

    torch.manual_seed(seed)
    try:
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # pragma: no cover - 防御性：CUDA 探测异常不应中断
        pass
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:  # pragma: no cover - 旧版本 torch 或不支持时忽略
        pass
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:  # pragma: no cover - 无 cudnn 后端时忽略
        pass


# ---------------------------------------------------------------------------
# 损失有限性守卫（R4-AC7）
# ---------------------------------------------------------------------------


def assert_finite_loss(loss_value: float, *, epoch: int, step: int) -> None:
    """损失为非有限值（``NaN`` / ``Inf``）时抛 :class:`TrainerError`（R4-AC7）。

    仅依赖标准库 ``math`` 对一个 Python ``float`` 作判定，**不需要 torch**：
    调用方应先把损失张量转为标量 ``float`` 再传入。

    :param loss_value: 当前训练步的标量损失值。
    :param epoch: 当前 epoch 编号（1-based），用于错误消息定位。
    :param step: 当前累计训练步数，用于错误消息定位。
    :raises TrainerError: 当 ``loss_value`` 为 ``NaN`` 或 ``±Inf`` 时。
    """

    value = float(loss_value)
    if math.isnan(value) or math.isinf(value):
        raise TrainerError(
            f"训练在 epoch={epoch} step={step} 处损失为非有限值（NaN/Inf）："
            f"loss={value}；训练数值发散，已中止。"
        )


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------


def _is_oom_error(exc: BaseException) -> bool:
    """判定一个异常是否表示显存/内存不足（OOM）。

    覆盖 ``torch.cuda.OutOfMemoryError``（若 torch 可用）以及消息中包含
    ``"out of memory"`` 的 :class:`RuntimeError`（CPU / 部分驱动场景），见设计
    文档 §Error Handling「OOM 识别」。
    """

    torch = _try_import_torch()
    if torch is not None:
        oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
        if oom_type is not None and isinstance(exc, oom_type):
            return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _make_batches(dataset: Any, batch_size: int, seed: int) -> list[list[Any]]:
    """把 ``dataset`` 确定性地切分为若干 batch（``batch_size`` 大小）。

    用 ``random.Random(seed)`` 控制确定性 shuffle（不依赖 torch，保证无 ML 依赖
    环境下亦可运行且可复现），再按 ``batch_size`` 顺序切块。相同 ``seed`` 必得
    相同切分（R4-AC5）。
    """

    n = len(dataset)
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    return [
        [dataset[j] for j in indices[i : i + batch_size]]
        for i in range(0, n, batch_size)
    ]


def _collate(examples: list[Any]) -> Any:
    """把一个 batch 的样本组成模型可消费的输入。

    - 元素为 :class:`~ego_pod.trainer.sft.SftExample`（具 ``input_ids`` /
      ``attention_mask`` / ``labels``）时：pad 到 batch 内最长长度，pad 位置
      ``attention_mask=0`` 且 ``labels=IGNORE_INDEX``；``torch`` 可用则返回张量
      ``dict``，否则返回等价的 Python ``list`` 组成的 ``dict``。
    - 否则（自定义/假样本）：原样返回 ``list``，交由模型自行解释。
    """

    if not examples:
        return examples
    first = examples[0]
    if not (hasattr(first, "input_ids") and hasattr(first, "labels")):
        return list(examples)

    max_len = max(len(ex.input_ids) for ex in examples)
    input_ids: list[list[int]] = []
    attention_mask: list[list[int]] = []
    labels: list[list[int]] = []
    for ex in examples:
        pad = max_len - len(ex.input_ids)
        input_ids.append(list(ex.input_ids) + [0] * pad)
        attention_mask.append(list(ex.attention_mask) + [0] * pad)
        labels.append(list(ex.labels) + [IGNORE_INDEX] * pad)

    torch = _try_import_torch()
    if torch is not None:
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def _move_to_device(batch: Any, device: Any) -> Any:
    """把 batch 中支持 ``.to`` 的张量移动到 ``device``；其余原样返回。"""

    if device is None or not isinstance(batch, dict):
        return batch
    moved = {}
    for key, value in batch.items():
        to_method = getattr(value, "to", None)
        moved[key] = to_method(device) if callable(to_method) else value
    return moved


def _build_optimizer(model: Any, learning_rate: float) -> Any:
    """构造仅覆盖可训练 LoRA 参数的 AdamW 优化器（R4-AC3）。

    仅收集 ``requires_grad=True`` 且为真实 ``torch.Tensor`` 的参数——PEFT 已把
    Base_Model 骨干权重置为 ``requires_grad=False``，因此优化器只更新 LoRA
    适配器参数（R4-AC3）。当 ``torch`` 不可用或不存在真实可训练张量参数（如
    假模型）时返回 ``None``，循环将跳过反向传播与参数更新。
    """

    torch = _try_import_torch()
    if torch is None:
        return None
    trainable = [
        param
        for _, param in model.named_parameters()
        if getattr(param, "requires_grad", False) and isinstance(param, torch.Tensor)
    ]
    if not trainable:
        return None
    return torch.optim.AdamW(trainable, lr=learning_rate)


def _forward_loss(model: Any, batch: Any) -> Any:
    """前向计算并取出损失对象（保留原始对象以便对真实张量 ``backward``）。"""

    outputs = model(**batch) if isinstance(batch, dict) else model(batch)
    return getattr(outputs, "loss", outputs)


def _loss_to_float(loss: Any) -> float:
    """把损失对象转为标量 ``float``（torch 标量张量优先用 ``.item()``）。"""

    item = getattr(loss, "item", None)
    if callable(item):
        return float(item())
    return float(loss)


# ---------------------------------------------------------------------------
# 训练循环（R4-AC1/AC2/AC3/AC7）
# ---------------------------------------------------------------------------


def train_qlora(
    model: Any,
    dataset: Any,
    params: TrainingParams,
    *,
    seed: int,
    progress_callback: ProgressCallback,
    device: Any = None,
) -> TrainingMetrics:
    """执行 QLoRA 训练循环并返回训练度量（R4-AC1/AC2/AC3/AC7）。

    步骤（见设计文档 §Components 3「trainer/loop.py」）：

    1. :func:`set_deterministic_seed`（R4-AC5）。
    2. 切换模型到训练模式（仅真实 ``nn.Module``）并构造仅覆盖可训练 LoRA
       参数的优化器（R4-AC3）。
    3. ``for epoch in range(params.epochs)``：以 ``batch_size=params.batch_size``
       确定性地切分 batch，逐 batch 前向取损失、:func:`assert_finite_loss`
       数值守卫、对真实张量损失反向并 ``optimizer.step()``；前向/反向遭遇 OOM
       （``torch.cuda.OutOfMemoryError`` 或含 ``"out of memory"`` 的
       :class:`RuntimeError`）统一包装为 :class:`TrainerError`（R4-AC7）。
    4. 每个 epoch 结束**恰好一次**调用 ``progress_callback(epoch=epoch+1,
       step=total_steps, loss=mean_epoch_loss)``，``epoch`` 取值 ``1..epochs``，
       ``step`` 为累计步数（单调不减），``loss`` 为该 epoch 真实平均损失
       （R4-AC1、R4-AC2）。
    5. 返回 :class:`TrainingMetrics`。

    :param model: 满足模块文档「鸭子类型契约」的模型对象（真实 PEFT 模型或
        假模型）。
    :param dataset: 满足契约的数据集（如 ``list[SftExample]``）。
    :param params: 训练参数；使用 ``epochs`` / ``batch_size`` / ``learning_rate``。
    :param seed: 确定性种子（R4-AC5）。
    :param progress_callback: 进度回调，每 epoch 调用一次（R4-AC2）。
    :param device: 目标设备（如 ``"cpu"`` / ``"cuda"`` / ``torch.device``）；为
        ``None`` 时不移动张量（便于无 torch 的假路径测试）。
    :returns: :class:`TrainingMetrics`，含完成 epoch 数、累计步数与最终损失。
    :raises TrainerError: 损失为 ``NaN`` / ``Inf`` 或发生 OOM（R4-AC7）。
    """

    set_deterministic_seed(seed)

    torch = _try_import_torch()
    if torch is not None and isinstance(model, getattr(torch.nn, "Module", ())):
        model.train()

    optimizer = _build_optimizer(model, params.learning_rate)

    total_steps = 0
    final_loss = 0.0

    for epoch in range(params.epochs):
        epoch_number = epoch + 1
        # 每个 epoch 以派生种子做确定性 reshuffle（相同 seed 全程可复现）。
        batches = _make_batches(dataset, params.batch_size, seed + epoch)

        epoch_loss_sum = 0.0
        steps_in_epoch = 0
        for raw_batch in batches:
            total_steps += 1
            try:
                if optimizer is not None:
                    optimizer.zero_grad()
                batch = _move_to_device(_collate(raw_batch), device)
                loss = _forward_loss(model, batch)
                loss_value = _loss_to_float(loss)
                assert_finite_loss(loss_value, epoch=epoch_number, step=total_steps)
                if optimizer is not None and hasattr(loss, "backward"):
                    loss.backward()
                    optimizer.step()
            except TrainerError:
                # 已是受控错误（含数值发散）：原样上抛，不二次包装。
                raise
            except Exception as exc:  # noqa: BLE001 - 需归一化底层异常
                if _is_oom_error(exc):
                    raise TrainerError(
                        f"训练在 epoch={epoch_number} step={total_steps} 处显存/"
                        f"内存不足（OOM）：{exc}；训练已中止。"
                    ) from exc
                raise

            epoch_loss_sum += loss_value
            steps_in_epoch += 1

        mean_epoch_loss = (
            epoch_loss_sum / steps_in_epoch if steps_in_epoch else 0.0
        )
        final_loss = mean_epoch_loss
        progress_callback(
            epoch=epoch_number, step=total_steps, loss=mean_epoch_loss
        )

    return TrainingMetrics(
        completed_epochs=params.epochs,
        total_steps=total_steps,
        final_loss=final_loss,
    )
