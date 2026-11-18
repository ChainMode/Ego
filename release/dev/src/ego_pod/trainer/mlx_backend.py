"""Apple MLX 训练后端（Apple Silicon 原生 LoRA SFT 训练 + PEFT 兼容导出，ADR 0004）。

本模块提供满足 :class:`ego_pod.trainer.base.TrainingBackend` Protocol 的
:class:`MlxBackend`，对应 ``detect`` 优先级中的第 3 位：当 ``mlx_lm`` 可导入、
``platform.machine() == "arm64"`` 且运行于 macOS 时自动启用。

核心契约（ADR 0004）：

1. **导入安全（R8-AC4）**：``mlx.core``、``mlx.nn``、``mlx.optimizers``、``mlx_lm``、
   ``numpy`` 与 ``safetensors`` 一律在方法内部惰性导入；模块顶层仅保留 ``try/except
   ImportError`` 哨兵，确保未安装 ``mlx-lm`` 的宿主导入本模块永不报错。
2. **SFT 输出-only 掩码与确定性训练**：复用 :func:`ego_pod.trainer.sft.build_sft_dataset`
   构建带 ``IGNORE_INDEX = -100`` 掩码的监督样本，设定确定性种子（``mx.random.seed``），
   冻结基座权重、仅训练并更新线性层 LoRA 参数（``lora_a`` / ``lora_b``），并将 OOM 与
   ``NaN``/``Inf`` 损失统一映射为 :class:`~ego_pod.exit_codes.TrainerError`。
3. **PEFT 兼容转置导出**：将 MLX 约定的 ``lora_a``（``[in_features, r]``）与 ``lora_b``
   （``[r, out_features]``）分别转置为 ``[r, in_features]`` 与 ``[out_features, r]``，
   以 ``base_model.model.<layer>.lora_A.weight`` / ``lora_B.weight`` 键名写出仅含适配器
   权重的标准 ``.safetensors`` 文件。
"""

from __future__ import annotations

import math
import os
import platform
import sys
from pathlib import Path
from typing import Any, Iterator

from ego_pod.config.base_models import resolve_hf_repo_id
from ego_pod.config.schema import TrainingParams
from ego_pod.exit_codes import EgoError, TrainerError
from ego_pod.logging_io import print_summary
from ego_pod.trainer import loop as _loop
from ego_pod.trainer import sft as _sft
from ego_pod.trainer.base import BackendIdentifier, Hardware, ProgressCallback

__all__ = [
    "MlxBackend",
]

# --- 可选依赖探测哨兵（顶层 import 永不抛错）-----------------------------------

try:  # pragma: no cover - 依赖是否存在取决于运行环境
    import mlx_lm as _mlx_lm
except ImportError:  # pragma: no cover
    _mlx_lm = None


def _mlx_importable() -> bool:
    """判定 ``mlx_lm`` 是否可导入。"""

    return _mlx_lm is not None


def _is_apple_silicon() -> bool:
    """判定当前是否运行于 Apple Silicon（macOS + arm64）。"""

    try:
        return sys.platform == "darwin" and platform.machine() == "arm64"
    except Exception:  # pragma: no cover - 防御性兜底
        return False


def _load_mlx_stack() -> tuple[Any, Any, Any, Any]:
    """惰性加载 ``(mlx.core, mlx.nn, mlx.optimizers, mlx_lm)`` 四元组。"""

    if _mlx_lm is None and sys.modules.get("mlx_lm") is None:
        raise ImportError("未安装 mlx-lm（请通过 pip install 'ego-pod[mlx]' 安装）")
    import mlx.core as mx  # noqa: PLC0415
    import mlx.nn as nn  # noqa: PLC0415
    import mlx.optimizers as optim  # noqa: PLC0415
    import mlx_lm  # noqa: PLC0415

    if mlx_lm is None:
        raise ImportError("mlx_lm 模块不可用")
    return mx, nn, optim, mlx_lm


def _unpack_model(model: Any) -> tuple[Any, Any]:
    """把 :meth:`MlxBackend.load_base_model` 返回的 ``(model_obj, tokenizer)`` 解包。"""

    if isinstance(model, tuple) and len(model) == 2:
        return model[0], model[1]
    return model, None


class _MlxLoraLinearWrapper:
    """轻量 MLX LoRA 线性包装层：冻结原始投影，叠加低秩增量 ``scale * ((x @ lora_a) @ lora_b)``。"""

    def __init__(
        self,
        base_layer: Any,
        *,
        lora_a: Any,
        lora_b: Any,
        scale: float,
    ) -> None:
        self.base_layer = base_layer
        self.lora_a = lora_a
        self.lora_b = lora_b
        self.scale = float(scale)

    @property
    def linear(self) -> Any:
        """与 ``mlx_lm.tuner.lora.LoRALinear.linear`` 对齐的只读原始层引用。"""

        return self.base_layer

    @property
    def weight(self) -> Any:
        """暴露被冻结原始线性层的只读权重，供冻结不变性检查与诊断使用。"""

        return getattr(self.base_layer, "weight", None)

    def __call__(self, x: Any) -> Any:
        base_out = self.base_layer(x)
        lora_out = (x @ self.lora_a) @ self.lora_b
        return base_out + (lora_out * self.scale)


def _is_2d_tensor_like(value: Any) -> bool:
    shape = getattr(value, "shape", None)
    return isinstance(shape, tuple) and len(shape) == 2


def _iter_lora_modules(
    root_obj: Any,
    *,
    _prefix: str = "",
    _visited: set[int] | None = None,
) -> Iterator[tuple[str, Any]]:
    """按确定性顺序递归枚举模型树中全部暴露 ``lora_a`` 与 ``lora_b`` 的模块。"""

    if root_obj is None or isinstance(root_obj, (str, bytes, int, float, bool)):
        return
    visited = set() if _visited is None else _visited
    obj_id = id(root_obj)
    if obj_id in visited:
        return
    visited.add(obj_id)

    lora_a = getattr(root_obj, "lora_a", None)
    lora_b = getattr(root_obj, "lora_b", None)
    if _is_2d_tensor_like(lora_a) and _is_2d_tensor_like(lora_b):
        path = _prefix or "lora_layer"
        yield path, root_obj
        return

    if isinstance(root_obj, (list, tuple)):
        for idx, item in enumerate(root_obj):
            sub_prefix = f"{_prefix}.{idx}" if _prefix else str(idx)
            yield from _iter_lora_modules(item, _prefix=sub_prefix, _visited=visited)
        return

    if isinstance(root_obj, dict):
        for key in sorted(root_obj.keys(), key=str):
            sub_prefix = f"{_prefix}.{key}" if _prefix else str(key)
            yield from _iter_lora_modules(
                root_obj[key], _prefix=sub_prefix, _visited=visited
            )
        return

    obj_dict = getattr(root_obj, "__dict__", None)
    if isinstance(obj_dict, dict):
        for attr_name, attr_val in obj_dict.items():
            if attr_name.startswith("_"):
                continue
            sub_prefix = f"{_prefix}.{attr_name}" if _prefix else attr_name
            yield from _iter_lora_modules(
                attr_val, _prefix=sub_prefix, _visited=visited
            )


def _inject_fallback_lora_layers(
    model_obj: Any,
    *,
    mx: Any,
    lora_rank: int,
    lora_alpha: int,
) -> int:
    """在未暴露 ``mlx_lm.tuner`` 注入器的模型树上，将线性投影层原位包装为 LoRA 层。"""

    scale = float(lora_alpha) / float(lora_rank)
    injected = 0
    excluded_names = {"lm_head", "embed", "embed_tokens", "wte"}

    def _walk_and_wrap(container: Any, visited: set[int]) -> None:
        nonlocal injected
        if container is None or id(container) in visited:
            return
        visited.add(id(container))

        if isinstance(container, list):
            for item in container:
                _walk_and_wrap(item, visited)
            return

        attrs = getattr(container, "__dict__", None)
        if not isinstance(attrs, dict):
            return

        for name, child in list(attrs.items()):
            if name.startswith("_") or name in excluded_names:
                continue
            if isinstance(child, _MlxLoraLinearWrapper):
                injected += 1
                continue
            weight = getattr(child, "weight", None)
            if callable(child) and _is_2d_tensor_like(weight):
                out_features, in_features = weight.shape
                bound = 1.0 / math.sqrt(max(int(in_features), 1))
                lora_a = mx.random.uniform(
                    -bound, bound, shape=(int(in_features), int(lora_rank))
                )
                lora_b = mx.zeros(
                    (int(lora_rank), int(out_features)), dtype=mx.float32
                )
                setattr(
                    container,
                    name,
                    _MlxLoraLinearWrapper(
                        child, lora_a=lora_a, lora_b=lora_b, scale=scale
                    ),
                )
                injected += 1
            else:
                _walk_and_wrap(child, visited)

    _walk_and_wrap(model_obj, set())
    return injected


def _ensure_lora_attached(
    model_obj: Any,
    *,
    mx: Any,
    mlx_lm: Any,
    params: TrainingParams,
) -> None:
    """冻结基座骨干并确保模型已挂载至少一个可训练的 MLX LoRA 层。"""

    freeze_fn = getattr(model_obj, "freeze", None)
    if callable(freeze_fn):
        freeze_fn()

    if any(True for _ in _iter_lora_modules(model_obj)):
        return

    # 优先尝试真实 mlx_lm.tuner.utils.linear_to_lora_layers
    try:
        from mlx_lm.tuner.utils import linear_to_lora_layers  # noqa: PLC0415

        layers_seq = getattr(model_obj, "layers", None)
        if layers_seq is None and hasattr(model_obj, "model"):
            layers_seq = getattr(model_obj.model, "layers", None)
        num_layers = len(layers_seq) if isinstance(layers_seq, (list, tuple)) else -1
        lora_cfg = {
            "rank": int(params.lora_rank),
            "alpha": float(params.lora_alpha),
            "scale": float(params.lora_alpha) / float(params.lora_rank),
            "dropout": 0.0,
        }
        linear_to_lora_layers(model_obj, num_layers, lora_cfg)
    except Exception:
        pass

    if any(True for _ in _iter_lora_modules(model_obj)):
        return

    injected = _inject_fallback_lora_layers(
        model_obj,
        mx=mx,
        lora_rank=params.lora_rank,
        lora_alpha=params.lora_alpha,
    )
    if injected <= 0:
        raise TrainerError(
            "MLX 模型中未找到任何可挂载 LoRA 适配器的线性投影层"
        )


def _is_mlx_oom(exc: BaseException) -> bool:
    """识别 Apple Silicon / Metal 统一内存不足（OOM）异常。"""

    if isinstance(exc, MemoryError):
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or "insufficient memory" in msg


class MlxBackend:
    """运行于 Apple Silicon 的原生 MLX LoRA 训练后端（探测优先级 3）。"""

    name = "Apple MLX"
    hardware: Hardware = "Apple Silicon"
    identifier: BackendIdentifier = "mlx"
    #: 真实 SFT/LoRA 训练与 PEFT 兼容导出已落地（ADR 0004）。
    REAL_TRAINING_IMPLEMENTED: bool = True

    _MAX_SEQ_LEN = 2048
    _SEED = 0

    @classmethod
    def is_available(cls) -> bool:
        """当真实训练已实现、``mlx_lm`` 可导入且运行于 Apple Silicon 时返回 True。"""

        return (
            cls.REAL_TRAINING_IMPLEMENTED
            and _mlx_importable()
            and _is_apple_silicon()
        )

    _SUPPORTED_QUANTIZATIONS = frozenset(
        {"4bit", "8bit", "bf16", "fp16", "fp32", "none", ""}
    )

    def load_base_model(self, model_id: str, quantization: str) -> Any:
        """经 ``mlx_lm.load`` 加载 Apple Silicon 基座模型与分词器并输出加载摘要。"""

        if quantization not in self._SUPPORTED_QUANTIZATIONS:
            raise TrainerError(
                f"Base_Model 加载失败（model_id={model_id!r}）："
                f"不支持的量化精度 {quantization!r}"
            )
        try:
            resolved_id = resolve_hf_repo_id(model_id)
            _mx, _nn, _optim, mlx_lm = _load_mlx_stack()
            loaded = mlx_lm.load(resolved_id)
            if isinstance(loaded, tuple) and len(loaded) >= 2:
                model_obj, tokenizer = loaded[0], loaded[1]
            else:
                raise RuntimeError(
                    "mlx_lm.load 未返回预期的 (model, tokenizer) 元组"
                )
        except EgoError:
            raise
        except Exception as exc:
            raise TrainerError(
                f"Base_Model 加载失败（model_id={model_id!r}）：{exc}"
            ) from exc

        quant_desc = quantization if quantization else "none"
        repo_part = f"hf_repo={resolved_id} " if resolved_id != model_id else ""
        print_summary(
            f"Loaded base model: model_id={model_id} {repo_part}"
            f"device=Apple Silicon quantization={quant_desc}"
        )
        return model_obj, tokenizer

    def train(
        self,
        model: Any,
        corpus_path: Path,
        params: TrainingParams,
        progress_callback: ProgressCallback,
    ) -> None:
        """在给定语料上执行确定性、输出-only 掩码的 MLX LoRA SFT 训练循环。"""

        model_obj, tokenizer = _unpack_model(model)
        if model_obj is None or tokenizer is None:
            raise TrainerError(
                "MLX 训练失败：传入的 model 不是合法的 (model, tokenizer) 复合对象"
            )

        corpus = Path(corpus_path)
        if not corpus.is_file():
            raise TrainerError(
                f"MLX 训练失败：Conversation_Corpus 文件不存在：{corpus}"
            )

        try:
            mx, nn, optim, mlx_lm = _load_mlx_stack()
        except Exception as exc:
            raise TrainerError(f"MLX 训练失败：缺少运行依赖（{exc}）") from exc

        # 1. 确定性种子设定
        _loop.set_deterministic_seed(self._SEED)
        mx.random.seed(self._SEED)

        # 2. 流式构建带 IGNORE_INDEX=-100 输出-only 掩码的 SFT 数据集
        dataset = _sft.build_sft_dataset(corpus, tokenizer, self._MAX_SEQ_LEN)

        # 3. 冻结基座并挂载 LoRA 线性层
        try:
            _ensure_lora_attached(
                model_obj, mx=mx, mlx_lm=mlx_lm, params=params
            )
            optimizer = optim.AdamW(learning_rate=params.learning_rate)
        except EgoError:
            raise
        except Exception as exc:
            raise TrainerError(f"MLX LoRA 初始化失败：{exc}") from exc

        ignore_idx = _sft.IGNORE_INDEX

        def _masked_causal_lm_loss(
            curr_model: Any, input_ids_arr: Any, labels_arr: Any
        ) -> Any:
            logits = curr_model(input_ids_arr)
            seq_len = logits.shape[1]
            if seq_len >= 2:
                shift_logits = logits[:, :-1, :]
                shift_labels = labels_arr[:, 1:]
            else:
                shift_logits = logits
                shift_labels = labels_arr

            active_mask = shift_labels != ignore_idx
            safe_targets = mx.where(
                active_mask, shift_labels, mx.zeros_like(shift_labels)
            )
            token_ce = nn.losses.cross_entropy(
                shift_logits, safe_targets, reduction="none"
            )
            masked_ce = token_ce * active_mask
            denom = mx.maximum(mx.sum(active_mask), mx.array(1.0, dtype=mx.float32))
            return mx.sum(masked_ce) / denom

        loss_and_grad_fn = nn.value_and_grad(model_obj, _masked_causal_lm_loss)

        total_steps = 0
        final_loss = 0.0

        for epoch in range(params.epochs):
            epoch_number = epoch + 1
            batches = _loop._make_batches(
                dataset, params.batch_size, self._SEED + epoch
            )
            epoch_loss_sum = 0.0
            steps_in_epoch = 0

            for raw_batch in batches:
                total_steps += 1
                try:
                    max_len = max(len(ex.input_ids) for ex in raw_batch)
                    padded_ids = [
                        list(ex.input_ids) + [0] * (max_len - len(ex.input_ids))
                        for ex in raw_batch
                    ]
                    padded_labels = [
                        list(ex.labels)
                        + [ignore_idx] * (max_len - len(ex.labels))
                        for ex in raw_batch
                    ]
                    input_ids_mx = mx.array(padded_ids, dtype=mx.int32)
                    labels_mx = mx.array(padded_labels, dtype=mx.int32)

                    loss, grads = loss_and_grad_fn(
                        model_obj, input_ids_mx, labels_mx
                    )
                    mx.eval(loss)
                    loss_value = float(loss.item())
                    _loop.assert_finite_loss(
                        loss_value, epoch=epoch_number, step=total_steps
                    )

                    optimizer.update(model_obj, grads)
                    params_fn = getattr(model_obj, "parameters", None)
                    if callable(params_fn):
                        mx.eval(params_fn(), getattr(optimizer, "state", None))
                    else:
                        mx.eval(getattr(optimizer, "state", None))
                except TrainerError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    if _is_mlx_oom(exc):
                        raise TrainerError(
                            f"训练在 epoch={epoch_number} step={total_steps} 处内存不足"
                            f"（OOM）：{exc}；训练已中止。"
                        ) from exc
                    raise TrainerError(
                        f"MLX 训练在 epoch={epoch_number} step={total_steps} 失败：{exc}"
                    ) from exc

                epoch_loss_sum += loss_value
                steps_in_epoch += 1

            mean_epoch_loss = (
                epoch_loss_sum / steps_in_epoch if steps_in_epoch else 0.0
            )
            final_loss = mean_epoch_loss
            progress_callback(
                epoch=epoch_number, step=total_steps, loss=mean_epoch_loss
            )

        print_summary(
            f"Training complete: epochs={params.epochs} "
            f"steps={total_steps} final_loss={final_loss:.6f}"
        )

    def export_lora_safetensors(self, model: Any, output_path: Path) -> None:
        """将 MLX LoRA 权重转置并按 PEFT SafeTensors 键名规范仅导出适配器权重。"""

        target = Path(output_path)
        model_obj, _tokenizer = _unpack_model(model)
        lora_pairs = list(_iter_lora_modules(model_obj))
        if not lora_pairs:
            raise TrainerError(
                f"LoRA_Adapter 导出失败（output_path={target}）："
                "模型不含任何可导出的 MLX LoRA 参数（lora_a / lora_b）"
            )

        tmp_target = target.with_name(f"{target.name}.tmp.{os.getpid()}")
        try:
            import numpy as np  # noqa: PLC0415
            from safetensors.numpy import save_file  # noqa: PLC0415

            tensors: dict[str, Any] = {}
            for raw_layer_path, lora_mod in lora_pairs:
                clean_layer = raw_layer_path
                if clean_layer.startswith("base_model.model."):
                    clean_layer = clean_layer[len("base_model.model.") :]

                arr_a = np.asarray(lora_mod.lora_a, dtype=np.float32)
                arr_b = np.asarray(lora_mod.lora_b, dtype=np.float32)
                if arr_a.ndim != 2 or arr_b.ndim != 2 or arr_a.shape[1] != arr_b.shape[0]:
                    raise ValueError(
                        f"层 {clean_layer} 的 MLX LoRA 形状不兼容："
                        f"lora_a={arr_a.shape}, lora_b={arr_b.shape}"
                    )

                # MLX: lora_a [in, r], lora_b [r, out] -> PEFT: lora_A [r, in], lora_B [out, r]
                key_a = f"base_model.model.{clean_layer}.lora_A.weight"
                key_b = f"base_model.model.{clean_layer}.lora_B.weight"
                tensors[key_a] = np.ascontiguousarray(arr_a.T)
                tensors[key_b] = np.ascontiguousarray(arr_b.T)

            target.parent.mkdir(parents=True, exist_ok=True)
            save_file(tensors, str(tmp_target))
            os.replace(tmp_target, target)
        except EgoError:
            if tmp_target.exists():
                tmp_target.unlink(missing_ok=True)
            raise
        except Exception as exc:
            if tmp_target.exists():
                tmp_target.unlink(missing_ok=True)
            raise TrainerError(
                f"LoRA_Adapter 导出失败（output_path={target}）：{exc}"
            ) from exc
