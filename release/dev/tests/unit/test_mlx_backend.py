"""`ego_pod.trainer.mlx_backend.MlxBackend` 真实训练与 PEFT 兼容导出单元测试（ADR 0004）。"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from safetensors import safe_open
from safetensors.numpy import load_file as load_numpy_safetensors

from ego_pod.config.schema import TrainingParams
from ego_pod.exit_codes import EXIT_RUNTIME_FAILURE, TrainerError
from ego_pod.privacy.offline import OfflineMissingWeightsError
from ego_pod.trainer import mlx_backend
from ego_pod.trainer.mlx_backend import MlxBackend


# ---------------------------------------------------------------------------
# Numpy 驱动的轻量确定性 MLX 替身栈（无需本机安装真实 mlx / mlx-lm 即可完整验证算子与转置契约）
# ---------------------------------------------------------------------------


class _FakeMxArray:
    """包装 numpy.ndarray 以模拟 mlx.core.array 的形状、转置、算术与标量接口。"""

    def __init__(self, data: Any, dtype: Any = None) -> None:
        arr = np.asarray(data, dtype=dtype if dtype is not None else None)
        self._arr = arr

    @property
    def shape(self) -> tuple[int, ...]:
        return self._arr.shape

    @property
    def ndim(self) -> int:
        return self._arr.ndim

    @property
    def dtype(self) -> Any:
        return self._arr.dtype

    @property
    def T(self) -> _FakeMxArray:
        return _FakeMxArray(self._arr.T)

    def item(self) -> float | int:
        return self._arr.item()

    def tolist(self) -> Any:
        return self._arr.tolist()

    def __array__(self, dtype: Any = None) -> np.ndarray:
        return np.asarray(self._arr, dtype=dtype) if dtype is not None else self._arr

    def __getitem__(self, idx: Any) -> _FakeMxArray:
        return _FakeMxArray(self._arr[idx])

    def __matmul__(self, other: Any) -> _FakeMxArray:
        return _FakeMxArray(self._arr @ np.asarray(other))

    def __add__(self, other: Any) -> _FakeMxArray:
        return _FakeMxArray(self._arr + np.asarray(other))

    def __sub__(self, other: Any) -> _FakeMxArray:
        return _FakeMxArray(self._arr - np.asarray(other))

    def __mul__(self, other: Any) -> _FakeMxArray:
        return _FakeMxArray(self._arr * np.asarray(other))

    def __rmul__(self, other: Any) -> _FakeMxArray:
        return _FakeMxArray(np.asarray(other) * self._arr)

    def __truediv__(self, other: Any) -> _FakeMxArray:
        return _FakeMxArray(self._arr / np.asarray(other))

    def __ne__(self, other: Any) -> _FakeMxArray:  # type: ignore[override]
        return _FakeMxArray(self._arr != other)


class _FakeLinear:
    """模拟线性投影层（in_features -> out_features）。"""

    def __init__(self, in_features: int, out_features: int, rng: np.random.Generator) -> None:
        w = rng.normal(0.0, 0.1, size=(out_features, in_features)).astype(np.float32)
        self.weight = _FakeMxArray(w)
        self._frozen = False

    def __call__(self, x: _FakeMxArray) -> _FakeMxArray:
        return _FakeMxArray(np.asarray(x) @ np.asarray(self.weight).T)


class _FakeTransformerBlock:
    def __init__(self, hidden: int, rng: np.random.Generator) -> None:
        self.q_proj = _FakeLinear(hidden, hidden, rng)
        self.v_proj = _FakeLinear(hidden, hidden, rng)


class _FakeCausalLM:
    """具备 embedding + layers[0].{q_proj, v_proj} + lm_head 的最小因果语言模型。"""

    def __init__(
        self,
        vocab_size: int = 32,
        hidden: int = 8,
        seed: int = 7,
        *,
        force_loss: float | None = None,
        raise_on_forward: Exception | None = None,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.vocab_size = vocab_size
        self.hidden = hidden
        self.embed = _FakeMxArray(
            rng.normal(0.0, 0.1, size=(vocab_size, hidden)).astype(np.float32)
        )
        self.layers = [_FakeTransformerBlock(hidden, rng)]
        self.lm_head = _FakeLinear(hidden, vocab_size, rng)
        self.frozen = False
        self.force_loss = force_loss
        self.raise_on_forward = raise_on_forward
        self.last_seen_input_ids: np.ndarray | None = None

    def freeze(self) -> None:
        self.frozen = True

    def __call__(self, input_ids: _FakeMxArray) -> _FakeMxArray:
        if self.raise_on_forward is not None:
            raise self.raise_on_forward
        ids = np.asarray(input_ids, dtype=np.int64)
        self.last_seen_input_ids = ids
        x = _FakeMxArray(np.asarray(self.embed)[ids % self.vocab_size])
        for blk in self.layers:
            x = x + blk.q_proj(x) + blk.v_proj(x)
        logits = self.lm_head(x)
        if self.force_loss is not None:
            return _FakeMxArray(np.full(logits.shape, self.force_loss, dtype=np.float32))
        return logits


class _FakeTokenizer:
    """确定性字节级分词器，满足 sft.build_sft_dataset 鸭子类型契约。"""

    chat_template = None

    def encode(self, text: str) -> list[int]:
        return [(b % 28) + 2 for b in text.encode("utf-8")]


def _install_fake_mlx(
    monkeypatch: pytest.MonkeyPatch,
    *,
    loaded_model: Any | None = None,
    load_error: Exception | None = None,
    recorded_masks: list[np.ndarray] | None = None,
) -> types.SimpleNamespace:
    """通过 monkeypatch 在 sys.modules 注入轻量确定性 mlx / mlx_lm 模块族。"""

    rng_holder = {"rng": np.random.default_rng(0), "seeds": []}

    mx = types.ModuleType("mlx.core")
    mx.int32 = np.int32
    mx.float32 = np.float32
    mx.array = lambda data, dtype=None: _FakeMxArray(data, dtype=dtype)
    mx.zeros = lambda shape, dtype=np.float32: _FakeMxArray(np.zeros(shape, dtype=dtype))
    mx.zeros_like = lambda arr: _FakeMxArray(np.zeros_like(np.asarray(arr)))
    mx.where = lambda cond, a, b: _FakeMxArray(
        np.where(np.asarray(cond), np.asarray(a), np.asarray(b))
    )
    mx.sum = lambda arr, axis=None: _FakeMxArray(np.sum(np.asarray(arr), axis=axis))
    mx.maximum = lambda a, b: _FakeMxArray(np.maximum(np.asarray(a), np.asarray(b)))
    mx.eval = lambda *args, **kwargs: None

    random_mod = types.SimpleNamespace()

    def _seed_fn(s: int) -> None:
        rng_holder["seeds"].append(s)
        rng_holder["rng"] = np.random.default_rng(s)

    def _uniform(low: float = 0.0, high: float = 1.0, shape: tuple[int, ...] = ()) -> _FakeMxArray:
        vals = rng_holder["rng"].uniform(low, high, size=shape).astype(np.float32)
        return _FakeMxArray(vals)

    random_mod.seed = _seed_fn
    random_mod.uniform = _uniform
    mx.random = random_mod

    nn = types.ModuleType("mlx.nn")
    losses_mod = types.SimpleNamespace()

    def _cross_entropy(
        logits: _FakeMxArray, targets: _FakeMxArray, reduction: str = "none"
    ) -> _FakeMxArray:
        arr = np.asarray(logits, dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            nan_ce = np.full(np.asarray(targets).shape, np.nan, dtype=np.float32)
            return _FakeMxArray(nan_ce)
        shifted = arr - np.max(arr, axis=-1, keepdims=True)
        log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
        tgt = np.asarray(targets, dtype=np.int64)
        picked = -np.take_along_axis(log_probs, tgt[..., None], axis=-1).squeeze(-1)
        out = picked.astype(np.float32)
        if reduction == "mean":
            return _FakeMxArray(np.mean(out))
        return _FakeMxArray(out)

    losses_mod.cross_entropy = _cross_entropy
    nn.losses = losses_mod

    def _value_and_grad(model_obj: Any, loss_fn: Any):
        def _wrapped(m: Any, input_ids_mx: _FakeMxArray, labels_mx: _FakeMxArray):
            if recorded_masks is not None:
                lbl = np.asarray(labels_mx)
                recorded_masks.append(lbl[:, 1:] != -100)
            base_loss = loss_fn(m, input_ids_mx, labels_mx)
            loss_scalar = float(base_loss.item())
            if not np.isfinite(loss_scalar):
                return base_loss, {}

            # 对全部挂载的 LoRA 参数采用确定性有限差分求真实梯度，验证真实参数下降
            eps = 1e-3
            grads: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for path_key, lora_mod in mlx_backend._iter_lora_modules(m):
                a_orig = np.array(lora_mod.lora_a, copy=True)
                b_orig = np.array(lora_mod.lora_b, copy=True)

                grad_a = np.zeros_like(a_orig)
                it_a = np.ndindex(a_orig.shape)
                for idx in it_a:
                    perturbed = a_orig.copy()
                    perturbed[idx] += eps
                    lora_mod.lora_a = _FakeMxArray(perturbed)
                    l_plus = float(loss_fn(m, input_ids_mx, labels_mx).item())
                    grad_a[idx] = (l_plus - loss_scalar) / eps
                lora_mod.lora_a = _FakeMxArray(a_orig)

                grad_b = np.zeros_like(b_orig)
                for idx in np.ndindex(b_orig.shape):
                    perturbed = b_orig.copy()
                    perturbed[idx] += eps
                    lora_mod.lora_b = _FakeMxArray(perturbed)
                    l_plus = float(loss_fn(m, input_ids_mx, labels_mx).item())
                    grad_b[idx] = (l_plus - loss_scalar) / eps
                lora_mod.lora_b = _FakeMxArray(b_orig)

                grads[path_key] = (grad_a, grad_b)
            return base_loss, grads

        return _wrapped

    nn.value_and_grad = _value_and_grad

    optim = types.ModuleType("mlx.optimizers")

    class _FakeAdamW:
        def __init__(self, learning_rate: float = 1e-4) -> None:
            self.learning_rate = learning_rate
            self.state: dict[str, int] = {"step": 0}

        def update(self, model_obj: Any, grads: Any) -> None:
            self.state["step"] += 1
            if isinstance(grads, dict):
                modules_by_key = dict(mlx_backend._iter_lora_modules(model_obj))
                for key, pair in grads.items():
                    if key in modules_by_key and isinstance(pair, tuple) and len(pair) == 2:
                        grad_a, grad_b = pair
                        mod = modules_by_key[key]
                        mod.lora_a = _FakeMxArray(
                            np.asarray(mod.lora_a) - self.learning_rate * grad_a
                        )
                        mod.lora_b = _FakeMxArray(
                            np.asarray(mod.lora_b) - self.learning_rate * grad_b
                        )

    optim.AdamW = _FakeAdamW

    mlx_pkg = types.ModuleType("mlx")
    mlx_pkg.core = mx
    mlx_pkg.nn = nn
    mlx_pkg.optimizers = optim

    mlx_lm = types.ModuleType("mlx_lm")

    def _fake_load(model_id: str) -> tuple[Any, Any]:
        if load_error is not None:
            raise load_error
        m = loaded_model if loaded_model is not None else _FakeCausalLM()
        return m, _FakeTokenizer()

    mlx_lm.load = _fake_load

    monkeypatch.setitem(sys.modules, "mlx", mlx_pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    monkeypatch.setitem(sys.modules, "mlx.nn", nn)
    monkeypatch.setitem(sys.modules, "mlx.optimizers", optim)
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setattr(mlx_backend, "_mlx_lm", mlx_lm)
    return types.SimpleNamespace(mx=mx, nn=nn, optim=optim, mlx_lm=mlx_lm, rng_holder=rng_holder)


def _write_corpus(path: Path, n: int = 4) -> Path:
    rows = [
        {
            "instruction": f"简短提问 {i}",
            "input": "",
            "output": f"回答 {i}",
        }
        for i in range(n)
    ]
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    return path


def _default_params(*, lora_rank: int = 4, lora_alpha: int = 8, epochs: int = 2) -> TrainingParams:
    return TrainingParams(
        method="qlora",
        quantization="4bit",
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        learning_rate=0.05,
        epochs=epochs,
        batch_size=2,
    )


# ---------------------------------------------------------------------------
# 1. REAL_TRAINING_IMPLEMENTED 开关与 load_base_model 测试
# ---------------------------------------------------------------------------


def test_mlx_real_training_implemented_flag_is_true() -> None:
    assert MlxBackend.REAL_TRAINING_IMPLEMENTED is True
    assert MlxBackend.identifier == "mlx"
    assert MlxBackend.hardware == "Apple Silicon"


def test_load_base_model_prints_summary_and_returns_model_and_tokenizer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _install_fake_mlx(monkeypatch)
    backend = MlxBackend()

    loaded = backend.load_base_model("qwen3.5-0.8B", "4bit")
    out = capsys.readouterr().out

    assert isinstance(loaded, tuple) and len(loaded) == 2
    assert "Loaded base model: model_id=qwen3.5-0.8B" in out
    assert "device=Apple Silicon" in out
    assert "quantization=4bit" in out


def test_load_base_model_maps_errors_and_preserves_ego_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = MlxBackend()

    # 通用异常包装为含 model_id 的 TrainerError
    _install_fake_mlx(monkeypatch, load_error=RuntimeError("weights corrupted"))
    with pytest.raises(TrainerError) as exc_info:
        backend.load_base_model("qwen3.5-0.8B", "4bit")
    assert "qwen3.5-0.8B" in str(exc_info.value)
    assert exc_info.value.exit_code == EXIT_RUNTIME_FAILURE

    # EgoError（如 OfflineMissingWeightsError）原样透传
    _install_fake_mlx(
        monkeypatch, load_error=OfflineMissingWeightsError("offline missing")
    )
    with pytest.raises(OfflineMissingWeightsError):
        backend.load_base_model("qwen3.5-0.8B", "4bit")


# ---------------------------------------------------------------------------
# 2. SFT 输出-only 掩码、冻结骨干、确定性训练与进度回调
# ---------------------------------------------------------------------------


def test_train_applies_output_only_mask_updates_only_lora_and_callbacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    recorded_masks: list[np.ndarray] = []
    fake_env = _install_fake_mlx(monkeypatch, recorded_masks=recorded_masks)
    backend = MlxBackend()

    model, tokenizer = backend.load_base_model("qwen3.5-0.8B", "4bit")
    base_q_before = np.array(model.layers[0].q_proj.weight, copy=True)
    base_head_before = np.array(model.lm_head.weight, copy=True)

    corpus = _write_corpus(tmp_path / "corpus.jsonl", n=4)
    params = _default_params(lora_rank=3, lora_alpha=6, epochs=2)
    progress_events: list[tuple[int, int, float]] = []

    backend.train(
        (model, tokenizer),
        corpus,
        params,
        lambda *, epoch, step, loss: progress_events.append((epoch, step, loss)),
    )
    out = capsys.readouterr().out

    # 1) 基座已冻结且权重逐元素未变
    assert model.frozen is True
    np.testing.assert_array_equal(np.asarray(model.layers[0].q_proj.weight), base_q_before)
    np.testing.assert_array_equal(np.asarray(model.lm_head.weight), base_head_before)

    # 2) mx.random.seed 被确定性调用
    assert len(fake_env.rng_holder["seeds"]) >= 1

    # 3) 输出-only 掩码：每条序列前部 prompt 段为 False（被掩码），尾部 output 段含 True
    assert len(recorded_masks) > 0
    first_mask = recorded_masks[0]
    assert bool(first_mask[0, 0]) is False  # prompt 首 token 不计损失
    assert bool(np.any(first_mask[0])) is True  # output 段 token 计损失

    # 4) LoRA 参数确实从全零的 lora_b 更新为非零
    lora_mods = dict(mlx_backend._iter_lora_modules(model))
    assert len(lora_mods) >= 2
    assert any(not np.allclose(np.asarray(m.lora_b), 0.0) for m in lora_mods.values())

    # 5) 每个 epoch 恰好回调一次，且打印 Training complete 摘要
    assert [e[0] for e in progress_events] == [1, 2]
    assert progress_events[1][1] >= progress_events[0][1] > 0
    assert "Training complete: epochs=2" in out


# ---------------------------------------------------------------------------
# 3. 确定性可复现性与 PEFT SafeTensors 键名 + 形状转置契约（[in, r] -> [r, in]）
# ---------------------------------------------------------------------------


def test_deterministic_training_and_peft_compatible_transposed_safetensors_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_mlx(monkeypatch)
    corpus = _write_corpus(tmp_path / "corpus.jsonl", n=4)
    params = _default_params(lora_rank=3, lora_alpha=6, epochs=2)

    # 连续两次以相同种子独立加载、训练并导出
    out_paths = [tmp_path / "run1.safetensors", tmp_path / "run2.safetensors"]
    for out_path in out_paths:
        backend = MlxBackend()
        loaded = backend.load_base_model("qwen3.5-0.8B", "4bit")
        backend.train(loaded, corpus, params, lambda **_: None)
        backend.export_lora_safetensors(loaded, out_path)

    tensors_1 = load_numpy_safetensors(str(out_paths[0]))
    tensors_2 = load_numpy_safetensors(str(out_paths[1]))

    assert set(tensors_1.keys()) == set(tensors_2.keys())
    assert len(tensors_1) >= 4  # 至少 q_proj/v_proj 各一对 lora_A / lora_B

    for key, arr1 in tensors_1.items():
        # 严禁混入任何非 LoRA 骨干权重
        assert key.startswith("base_model.model.")
        assert key.endswith(".lora_A.weight") or key.endswith(".lora_B.weight")
        # 两次运行逐元素完全一致（确定性）
        np.testing.assert_allclose(arr1, tensors_2[key], rtol=0.0, atol=0.0)

        # 验证 [in=8, r=3] -> [r=3, in=8] 与 [r=3, out=8] -> [out=8, r=3] 转置形状契约！
        if key.endswith(".lora_A.weight"):
            assert arr1.shape == (3, 8)
        else:
            assert arr1.shape == (8, 3)

    # 验证标准 safetensors.safe_open 可直接读取
    with safe_open(str(out_paths[0]), framework="numpy") as handle:
        assert set(handle.keys()) == set(tensors_1.keys())


# ---------------------------------------------------------------------------
# 4. OOM、NaN/Inf、损坏语料与非法导出错误映射
# ---------------------------------------------------------------------------


def test_train_maps_oom_and_nan_loss_to_trainer_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = _write_corpus(tmp_path / "corpus.jsonl", n=2)
    params = _default_params(epochs=1)

    # 1) OOM -> TrainerError
    oom_model = _FakeCausalLM(raise_on_forward=RuntimeError("Metal out of memory"))
    _install_fake_mlx(monkeypatch, loaded_model=oom_model)
    backend = MlxBackend()
    loaded = backend.load_base_model("qwen3.5-0.8B", "4bit")
    with pytest.raises(TrainerError, match="OOM"):
        backend.train(loaded, corpus, params, lambda **_: None)

    # 2) NaN 损失 -> TrainerError
    nan_model = _FakeCausalLM(force_loss=float("nan"))
    _install_fake_mlx(monkeypatch, loaded_model=nan_model)
    loaded_nan = backend.load_base_model("qwen3.5-0.8B", "4bit")
    with pytest.raises(TrainerError, match="NaN/Inf"):
        backend.train(loaded_nan, corpus, params, lambda **_: None)


def test_train_corrupt_jsonl_reports_line_number(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_mlx(monkeypatch)
    bad_corpus = tmp_path / "bad.jsonl"
    bad_corpus.write_text(
        '{"instruction":"ok","output":"ok"}\nNOT_JSON\n', encoding="utf-8"
    )
    backend = MlxBackend()
    loaded = backend.load_base_model("qwen3.5-0.8B", "4bit")
    with pytest.raises(TrainerError, match="第 2 行"):
        backend.train(loaded, bad_corpus, _default_params(epochs=1), lambda **_: None)


def test_export_without_lora_weights_raises_and_writes_no_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_mlx(monkeypatch)
    backend = MlxBackend()
    target = tmp_path / "should_not_exist.safetensors"
    with pytest.raises(TrainerError):
        backend.export_lora_safetensors(_FakeCausalLM(), target)
    assert not target.exists()


def test_train_empty_corpus_raises_trainer_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_mlx(monkeypatch)
    empty_corpus = tmp_path / "empty.jsonl"
    empty_corpus.write_text("", encoding="utf-8")
    backend = MlxBackend()
    loaded = backend.load_base_model("qwen3.5-0.8B", "4bit")
    with pytest.raises(TrainerError):
        backend.train(loaded, empty_corpus, _default_params(epochs=1), lambda **_: None)


def test_train_maps_python_memory_error_and_inf_loss_to_trainer_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = _write_corpus(tmp_path / "corpus.jsonl", n=2)
    params = _default_params(epochs=1)
    backend = MlxBackend()

    # MemoryError -> OOM TrainerError
    mem_model = _FakeCausalLM(raise_on_forward=MemoryError("unified memory exhausted"))
    _install_fake_mlx(monkeypatch, loaded_model=mem_model)
    loaded_mem = backend.load_base_model("qwen3.5-0.8B", "4bit")
    with pytest.raises(TrainerError, match="OOM"):
        backend.train(loaded_mem, corpus, params, lambda **_: None)

    # Inf 损失 -> NaN/Inf TrainerError
    inf_model = _FakeCausalLM(force_loss=float("inf"))
    _install_fake_mlx(monkeypatch, loaded_model=inf_model)
    loaded_inf = backend.load_base_model("qwen3.5-0.8B", "4bit")
    with pytest.raises(TrainerError, match="NaN/Inf"):
        backend.train(loaded_inf, corpus, params, lambda **_: None)


def test_export_strips_existing_base_model_prefix_and_rejects_mismatched_rank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_mlx(monkeypatch)
    backend = MlxBackend()

    class _PrefixedHolder:
        def __init__(self, lora_a: Any, lora_b: Any) -> None:
            self.lora_a = lora_a
            self.lora_b = lora_b

    # 1) 已有 base_model.model. 前缀时幂等去重，不生成双重前缀
    good_tree = {
        "base_model.model.layers.0.self_attn.q_proj": _PrefixedHolder(
            np.ones((6, 2), dtype=np.float32),
            np.zeros((2, 4), dtype=np.float32),
        )
    }
    out_good = tmp_path / "prefixed.safetensors"
    backend.export_lora_safetensors(good_tree, out_good)
    loaded = load_numpy_safetensors(str(out_good))
    assert set(loaded.keys()) == {
        "base_model.model.layers.0.self_attn.q_proj.lora_A.weight",
        "base_model.model.layers.0.self_attn.q_proj.lora_B.weight",
    }
    assert loaded["base_model.model.layers.0.self_attn.q_proj.lora_A.weight"].shape == (2, 6)
    assert loaded["base_model.model.layers.0.self_attn.q_proj.lora_B.weight"].shape == (4, 2)

    # 2) lora_a 列数 != lora_b 行数 -> 抛 TrainerError 且不残留目标或临时文件
    bad_tree = {
        "layers.0.q_proj": _PrefixedHolder(
            np.ones((6, 3), dtype=np.float32),
            np.zeros((2, 4), dtype=np.float32),
        )
    }
    out_bad = tmp_path / "bad_rank.safetensors"
    with pytest.raises(TrainerError, match="形状不兼容"):
        backend.export_lora_safetensors(bad_tree, out_bad)
    assert not out_bad.exists()
    assert list(tmp_path.glob("*.tmp.*")) == []


def test_load_base_model_rejects_invalid_quantization_before_mlx_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_mlx(monkeypatch)
    backend = MlxBackend()
    with pytest.raises(TrainerError, match="不支持的量化精度"):
        backend.load_base_model("qwen3.5-0.8B", "int2")  # type: ignore[arg-type]
