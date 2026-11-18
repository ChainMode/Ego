"""多规格（Multi-Edition）配置、基座白名单、OOM 自动降批重试与多规格 Vault 单测（ADR 0006）。"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from safetensors.torch import save_file
import torch

from ego_pod.config.defaults import render_all_editions_yaml, render_default_yaml
from ego_pod.config.loader import load
from ego_pod.config.schema import (
    EDITION_PRESETS,
    SUPPORTED_BASE_MODELS,
    TrainingParams,
    resolve_edition_spec,
)
from ego_pod.config.validators import validate
from ego_pod.exit_codes import (
    ConfigError,
    OutOfMemoryTrainerError,
    TrainerError,
    VaultError,
)
from ego_pod.trainer.cells import CellSpec, TrainedCell
from ego_pod.trainer.grid import run_grid_training
from ego_pod.trainer.loop import is_oom_error
from ego_pod.trainer.pipeline import run_training_with_telemetry
from ego_pod.vault.history import HISTORY_DIRNAME, list_vault_snapshots
from ego_pod.vault.package import EditionBuildInput, write_multi_edition_package
from ego_pod.vault.signing import verify_package_signature


def _make_safetensor(path: Path, num_floats: int = 4) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"lora_A": torch.ones((num_floats,), dtype=torch.float32)}, str(path))
    return path


def _make_non_empty_jsonl(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"instruction":"q","input":"","output":"valid output sample","category":"skill"}\n',
        encoding="utf-8",
    )
    return path


class _OomSimulatingBackend:
    """在前 fail_above_batch 之上模拟显存不足，减半后恢复成功的测试后端。"""

    name = "oom-sim"
    hardware = "CPU"
    identifier = "stub"

    def __init__(self, *, min_working_batch: int, always_oom: bool = False) -> None:
        self.min_working_batch = min_working_batch
        self.always_oom = always_oom
        self.attempted_batches: list[int] = []

    def is_available(self) -> bool:
        return True

    def load_base_model(self, model_id: str, quantization: str = "4bit") -> object:
        return {"model_id": model_id, "quantization": quantization}

    def train(self, model, corpus_path: Path, params: TrainingParams, progress_callback) -> None:
        self.attempted_batches.append(params.batch_size)
        if self.always_oom or params.batch_size > self.min_working_batch:
            raise OutOfMemoryTrainerError(
                f"Simulated CUDA out of memory at batch_size={params.batch_size}"
            )
        progress_callback(epoch=1, step=1, loss=0.4321)

    def export_lora_safetensors(self, model, output_path: Path) -> None:
        _make_safetensor(output_path)


def test_supported_base_models_allowlist_and_error_listing(tmp_path: Path) -> None:
    """十款受控基座全部通过校验；未知基座抛 ConfigError 并完整列出全部受支持基座。"""
    cfg_file = tmp_path / "ego-config.yaml"
    cfg_file.write_text(render_default_yaml(), encoding="utf-8")
    _, abs_path = load(cfg_file)

    for model_id in SUPPORTED_BASE_MODELS:
        raw = {
            "sources": {
                cat: {"type": "local", "path": f"./ego-data/{cat}"}
                for cat in ("sense", "action", "thought", "skill", "credentials")
            },
            "training": {
                "method": "qlora",
                "quantization": "4bit",
                "lora_rank": 16,
                "lora_alpha": 32,
                "learning_rate": 0.0002,
                "epochs": 2,
                "batch_size": 4,
            },
            "editions": {"normal": {"base": model_id}},
        }
        validate(raw, abs_path, stream=io.StringIO())

    bad_raw = {
        "sources": {
            cat: {"type": "local", "path": f"./ego-data/{cat}"}
            for cat in ("sense", "action", "thought", "skill", "credentials")
        },
        "training": {
            "method": "qlora",
            "quantization": "4bit",
            "lora_rank": 16,
            "lora_alpha": 32,
            "learning_rate": 0.0002,
            "epochs": 2,
            "batch_size": 4,
        },
        "editions": {"normal": {"base": "llama3-70B"}},
    }
    with pytest.raises(ConfigError) as exc:
        validate(bad_raw, abs_path, stream=io.StringIO())
    err_msg = str(exc.value)
    for supported in SUPPORTED_BASE_MODELS:
        assert supported in err_msg


def test_all_editions_and_custom_edition_resolution(tmp_path: Path) -> None:
    """支持四档内置预设 + 自定义规格加载、校验与参数归一。"""
    import yaml

    raw = yaml.safe_load(render_all_editions_yaml())
    raw["editions"]["research-lab"] = {
        "base": "gemma4-9B",
        "description": "自定义研究规格",
        "lora_rank": 24,
        "lora_alpha": 48,
        "max_level": 2,
        "max_size_mb": 512,
        "zones": ["coding"],
    }
    cfg_path = tmp_path / "ego-config.yaml"
    cfg_path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")

    validate(raw, cfg_path, stream=io.StringIO())
    cfg, _ = load(cfg_path)

    assert set(cfg.editions.keys()) == {
        "lite",
        "normal",
        "plus",
        "super",
        "research-lab",
    }
    lite_spec = resolve_edition_spec("lite", cfg.editions["lite"], cfg.training)
    assert lite_spec.max_level == 1
    assert lite_spec.training.lora_rank == EDITION_PRESETS["lite"].lora_rank
    assert lite_spec.max_size_mb == 64.0

    custom_spec = resolve_edition_spec(
        "research-lab", cfg.editions["research-lab"], cfg.training
    )
    assert custom_spec.base == "gemma4-9B"
    assert custom_spec.training.lora_rank == 24
    assert custom_spec.max_level == 2
    assert custom_spec.max_size_mb == 512.0
    assert custom_spec.zones == ("coding",)


@pytest.mark.parametrize(
    "forbidden_key",
    ["access_key", "secret_key", "access_key_id", "secret_access_key", "token", "api_key"],
)
def test_plaintext_credentials_in_sources_rejected(tmp_path: Path, forbidden_key: str) -> None:
    """ego-config.yaml 的 sources 条目内嵌任何明文密钥字段时立即以 ConfigError 拒绝。"""
    raw = {
        "sources": {
            "sense": {
                "type": "s3",
                "bucket": "my-ego-bucket",
                "prefix": "sense/",
                forbidden_key: "PLAINTEXT_SECRET_VALUE",
            },
            "action": {"type": "local", "path": "./ego-data/action"},
            "thought": {"type": "local", "path": "./ego-data/thought"},
            "skill": {"type": "local", "path": "./ego-data/skill"},
            "credentials": {"type": "local", "path": "./ego-data/credentials"},
        },
        "training": {
            "method": "qlora",
            "quantization": "4bit",
            "lora_rank": 16,
            "lora_alpha": 32,
            "learning_rate": 0.0002,
            "epochs": 1,
            "batch_size": 2,
        },
        "editions": {"normal": {"base": "qwen3.5-0.8B"}},
    }
    with pytest.raises(ConfigError) as exc:
        validate(raw, tmp_path / "ego-config.yaml", stream=io.StringIO())
    assert forbidden_key in str(exc.value) or "EGO_CLOUD_" in str(exc.value)


def test_oom_batch_size_halving_succeeds_within_three_retries(tmp_path: Path) -> None:
    """遭遇 OOM 时自动将 batch_size 减半（8 -> 4 -> 2）并在恢复后记录 effective_batch_size 与 oom_retries。"""
    cell_ds = _make_non_empty_jsonl(tmp_path / ".ego" / "cells" / "kernel.jsonl")
    params = TrainingParams(
        method="qlora",
        quantization="4bit",
        lora_rank=8,
        lora_alpha=16,
        learning_rate=2e-4,
        epochs=1,
        batch_size=8,
    )
    backend = _OomSimulatingBackend(min_working_batch=2)
    out = io.StringIO()

    grid_res = run_grid_training(
        [CellSpec(zone=None, level=0, sub_dataset_path=cell_ds)],
        params,
        cwd=tmp_path,
        backend=backend,  # type: ignore[arg-type]
        stdout=out,
    )

    assert backend.attempted_batches == [8, 4, 2]
    assert grid_res.effective_batch_size == 2
    assert grid_res.oom_retries == 2
    assert "batch_size" in out.getvalue()

    # 单适配器遥测包装同样支持自动减半重试
    backend_single = _OomSimulatingBackend(min_working_batch=4)
    single_res, eff_bs, retries = run_training_with_telemetry(
        cell_ds,
        params,
        cwd=tmp_path,
        backend=backend_single,  # type: ignore[arg-type]
        stdout=io.StringIO(),
    )
    assert single_res.output_path.is_file()
    assert eff_bs == 4
    assert retries == 1


def test_oom_exhausts_retries_and_raises_trainer_error(tmp_path: Path) -> None:
    """连续减半超过最多 3 次重试仍 OOM 时抛出受控 TrainerError。"""
    cell_ds = _make_non_empty_jsonl(tmp_path / ".ego" / "cells" / "kernel.jsonl")
    params = TrainingParams(
        method="qlora",
        quantization="4bit",
        lora_rank=8,
        lora_alpha=16,
        learning_rate=2e-4,
        epochs=1,
        batch_size=16,
    )
    backend = _OomSimulatingBackend(min_working_batch=0, always_oom=True)

    with pytest.raises(TrainerError) as exc:
        run_grid_training(
            [CellSpec(zone=None, level=0, sub_dataset_path=cell_ds)],
            params,
            cwd=tmp_path,
            backend=backend,  # type: ignore[arg-type]
            stdout=io.StringIO(),
        )
    assert len(backend.attempted_batches) == 4  # 初始 16 + 3 次重试 (8, 4, 2)
    assert "OOM" in str(exc.value) or "out of memory" in str(exc.value).lower()


def test_multi_edition_vault_package_signing_history_and_size_ceiling(
    tmp_path: Path,
) -> None:
    """多规格同包落盘、Tier 1+2 全包签名验签、覆盖归档到 .history/ 以及体积上限拦截。"""
    root = tmp_path / "ego-output"
    seed_hex = "11" * 32
    fay_id = "ifay_ed234567abcdefghij"

    lite_k = _make_safetensor(tmp_path / "tmp_train" / "lite_k.safetensors")
    lite_c1 = _make_safetensor(tmp_path / "tmp_train" / "lite_c1.safetensors")
    normal_k = _make_safetensor(tmp_path / "tmp_train" / "normal_k.safetensors")
    normal_c1 = _make_safetensor(tmp_path / "tmp_train" / "normal_c1.safetensors")
    normal_c2 = _make_safetensor(tmp_path / "tmp_train" / "normal_c2.safetensors")

    artifacts_v1 = write_multi_edition_package(
        root,
        edition_builds=[
            EditionBuildInput(
                edition="lite",
                base_model="qwen3.5-0.8B",
                lora_rank=8,
                lora_alpha=16,
                max_size_mb=64.0,
                cells=[
                    TrainedCell(zone=None, level=0, output_path=lite_k),
                    TrainedCell(zone="coding", level=1, output_path=lite_c1),
                ],
            ),
            EditionBuildInput(
                edition="normal",
                base_model="gemma4-2B",
                lora_rank=16,
                lora_alpha=32,
                max_size_mb=256.0,
                cells=[
                    TrainedCell(zone=None, level=0, output_path=normal_k),
                    TrainedCell(zone="coding", level=1, output_path=normal_c1),
                    TrainedCell(zone="coding", level=2, output_path=normal_c2),
                ],
            ),
        ],
        training_backend="stub",
        version="0.2.0",
        include_integrity=True,
        signing_seed_hex=seed_hex,
        signer_fay_id=fay_id,
    )

    manifest = json.loads(artifacts_v1.manifest_path.read_text(encoding="utf-8"))
    assert manifest["editions"] == ["lite", "normal"]
    assert set(manifest["grid"].keys()) == {"lite", "normal"}
    assert "lite/adapters/kernel.safetensors" in manifest["integrity"]["cell_digests"]
    assert "normal/adapters/coding/L2.safetensors" in manifest["integrity"]["cell_digests"]

    verify_res = verify_package_signature(root)
    assert verify_res.status == "verified"
    assert len(verify_res.verified_cells) == 5

    # 第二次写入覆盖：旧包自动归档到 .history/，且新包验签不受 .history/ 内权重干扰
    lite_k2 = _make_safetensor(tmp_path / "tmp_train2" / "lite_k.safetensors")
    lite_c1_2 = _make_safetensor(tmp_path / "tmp_train2" / "lite_c1.safetensors")
    write_multi_edition_package(
        root,
        edition_builds=[
            EditionBuildInput(
                edition="lite",
                base_model="qwen3.5-0.8B",
                lora_rank=8,
                lora_alpha=16,
                max_size_mb=64.0,
                cells=[
                    TrainedCell(zone=None, level=0, output_path=lite_k2),
                    TrainedCell(zone="coding", level=1, output_path=lite_c1_2),
                ],
            ),
        ],
        training_backend="stub",
        version="0.2.0",
        include_integrity=True,
        signing_seed_hex=seed_hex,
        signer_fay_id=fay_id,
        max_history=5,
    )
    snapshots = list_vault_snapshots(root)
    assert len(snapshots) == 1
    assert (root / HISTORY_DIRNAME / snapshots[0].snapshot_id / "manifest.json").is_file()
    assert verify_package_signature(root).status == "verified"

    # 体积上限检查：设置极小 max_size_mb=0.00001 MB 触发 VaultError，且保留既有可用包
    oversize_k = _make_safetensor(tmp_path / "tmp_oversize" / "k.safetensors", num_floats=256)
    oversize_c1 = _make_safetensor(tmp_path / "tmp_oversize" / "c1.safetensors", num_floats=256)
    with pytest.raises(VaultError) as exc:
        write_multi_edition_package(
            root,
            edition_builds=[
                EditionBuildInput(
                    edition="lite",
                    base_model="qwen3.5-0.8B",
                    lora_rank=8,
                    lora_alpha=16,
                    max_size_mb=0.00001,
                    cells=[
                        TrainedCell(zone=None, level=0, output_path=oversize_k),
                        TrainedCell(zone="coding", level=1, output_path=oversize_c1),
                    ],
                )
            ],
            training_backend="stub",
        )
    assert "max_size_mb" in str(exc.value) or "体积超限" in str(exc.value)
    assert verify_package_signature(root).status == "verified"


@pytest.mark.parametrize(
    "exc_obj,expected",
    [
        (MemoryError("host ram exhausted"), True),
        (OutOfMemoryTrainerError("explicit oom"), True),
        (RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"), True),
        (RuntimeError("Metal backend out of memory"), True),
        (ValueError("unrelated value error"), False),
        (RuntimeError("shape mismatch [16, 8]"), False),
    ],
)
def test_is_oom_error_classification_variants(
    exc_obj: BaseException, expected: bool
) -> None:
    assert is_oom_error(exc_obj) is expected


@pytest.mark.parametrize(
    "mutator",
    [
        lambda d: d.__setitem__("zones", {"zones": ["coding"], "routing": {}, "max_level": 0}),
        lambda d: d.__setitem__("output", {"dir": "./ego-output", "max_history": -1}),
        lambda d: d["editions"]["normal"].update({"max_size_mb": -5}),
        lambda d: d["editions"]["normal"].update({"max_level": 0}),
        lambda d: d["editions"]["normal"].update({"zones": ["INVALID_ZONE"]}),
    ],
)
def test_validators_reject_invalid_v02_overrides(tmp_path: Path, mutator) -> None:
    import yaml

    raw = yaml.safe_load(render_default_yaml())
    mutator(raw)
    with pytest.raises(ConfigError):
        validate(raw, tmp_path / "ego-config.yaml", stream=io.StringIO())
