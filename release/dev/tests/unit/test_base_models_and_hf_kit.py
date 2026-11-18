"""Unit tests for base-model resolution, PEFT target inference, transformers 5.x BatchEncoding, PyPI metadata, and HF Space kit."""

from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import sys
import tomllib
import types

import pytest

from ego_pod.config.base_models import (
    BASE_MODEL_REGISTRY,
    COMPACT_FAST_BASE,
    RECOMMENDED_DEFAULT_BASE,
    BaseModelResolutionError,
    is_hf_repo_id_syntax,
    is_valid_base_model_spec,
    real_base_model_choices,
    resolve_hf_repo_id,
)
from ego_pod.config.defaults import render_default_yaml
from ego_pod.config.schema import SUPPORTED_BASE_MODELS
from ego_pod.config.validators import validate
from ego_pod.exit_codes import ConfigError, TrainerError
from ego_pod.trainer.generation import DecodeSettings, generate_reply
from ego_pod.trainer.mlx_backend import MlxBackend
from ego_pod.trainer.peft_backend import PeftCpuBackend, resolve_lora_target_modules
from ego_pod.trainer.unsloth_backend import UnslothBackend


def test_registry_covers_all_supported_base_models_and_compact_default() -> None:
    assert RECOMMENDED_DEFAULT_BASE == "qwen3.5-0.8B"
    assert COMPACT_FAST_BASE == "qwen2.5-0.5B"
    assert set(SUPPORTED_BASE_MODELS) == set(BASE_MODEL_REGISTRY.keys())

    assert resolve_hf_repo_id("qwen2.5-0.5B") == "Qwen/Qwen2.5-0.5B-Instruct"
    assert resolve_hf_repo_id("qwen3.5-0.8B") == "Qwen/Qwen3.5-0.8B"
    assert resolve_hf_repo_id("gemma4-2B") == "google/gemma-4-E2B-it"
    assert resolve_hf_repo_id("phi4-mini") == "microsoft/Phi-4-mini-instruct"
    assert resolve_hf_repo_id("phi4-14B") == "microsoft/phi-4"

    choices = real_base_model_choices()
    assert "qwen2.5-0.5B" in choices
    assert "qwen3.5-0.8B" in choices
    assert "gemma4-9B" not in choices
    assert "gemma4-27B" not in choices


@pytest.mark.parametrize("unmapped_name", ["gemma4-9B", "gemma4-27B"])
def test_resolve_hf_repo_id_raises_actionable_error_for_unmapped_friendly_names(
    unmapped_name: str,
) -> None:
    with pytest.raises(BaseModelResolutionError) as exc_info:
        resolve_hf_repo_id(unmapped_name)
    msg = str(exc_info.value)
    assert unmapped_name in msg
    assert "Qwen/Qwen2.5-0.5B-Instruct" in msg
    assert "Qwen/Qwen3.5-0.8B" in msg


def test_resolve_hf_repo_id_passes_through_repo_ids_and_local_dirs(tmp_path: Path) -> None:
    assert resolve_hf_repo_id("Qwen/Qwen2.5-0.5B-Instruct") == "Qwen/Qwen2.5-0.5B-Instruct"
    local_dir = tmp_path / "custom-checkpoint"
    local_dir.mkdir()
    assert resolve_hf_repo_id(str(local_dir)) == str(local_dir)
    assert is_hf_repo_id_syntax("Qwen/Qwen2.5-0.5B-Instruct") is True
    assert is_hf_repo_id_syntax("llama3-70B") is False
    assert is_valid_base_model_spec(str(local_dir)) is True


def test_validator_accepts_compact_base_direct_hf_repo_and_local_dir(tmp_path: Path) -> None:
    import yaml

    local_dir = tmp_path / "local-ckpt"
    local_dir.mkdir()

    for valid_base in ("qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct", str(local_dir)):
        raw = yaml.safe_load(render_default_yaml())
        raw["editions"]["normal"]["base"] = valid_base
        validate(raw, tmp_path / "ego-config.yaml", stream=io.StringIO())

    bad = yaml.safe_load(render_default_yaml())
    bad["editions"]["normal"]["base"] = "not-a-repo"
    with pytest.raises(ConfigError):
        validate(bad, tmp_path / "ego-config.yaml", stream=io.StringIO())


def test_peft_mlx_and_unsloth_backends_resolve_friendly_names_to_hf_repos(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen_peft_repos: list[str] = []
    fake_tf = types.ModuleType("transformers")

    class _FakeLoader:
        @staticmethod
        def from_pretrained(repo_id: str, **_kwargs: object) -> object:
            seen_peft_repos.append(repo_id)
            return types.SimpleNamespace(repo_id=repo_id)

    fake_tf.AutoTokenizer = _FakeLoader  # type: ignore[attr-defined]
    fake_tf.AutoModelForCausalLM = _FakeLoader  # type: ignore[attr-defined]
    fake_tf.BitsAndBytesConfig = object  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)

    peft_backend = PeftCpuBackend()
    peft_backend.load_base_model("qwen2.5-0.5B", "4bit")
    assert seen_peft_repos == [
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-0.5B-Instruct",
    ]
    out_peft = capsys.readouterr().out
    assert "model_id=qwen2.5-0.5B" in out_peft
    assert "hf_repo=Qwen/Qwen2.5-0.5B-Instruct" in out_peft

    with pytest.raises(TrainerError) as peft_err:
        peft_backend.load_base_model("gemma4-9B", "4bit")
    assert isinstance(peft_err.value.__cause__, BaseModelResolutionError)

    # MlxBackend resolution
    from ego_pod.trainer import mlx_backend as mlx_mod

    seen_mlx_repos: list[str] = []
    fake_mlx_pkg = types.ModuleType("mlx")
    fake_mlx_pkg.core = types.ModuleType("mlx.core")  # type: ignore[attr-defined]
    fake_mlx_pkg.nn = types.ModuleType("mlx.nn")  # type: ignore[attr-defined]
    fake_mlx_pkg.optimizers = types.ModuleType("mlx.optimizers")  # type: ignore[attr-defined]
    fake_mlx_lm = types.ModuleType("mlx_lm")
    fake_mlx_lm.load = lambda repo_id: (seen_mlx_repos.append(repo_id) or object(), object())  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx_pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mlx_pkg.core)
    monkeypatch.setitem(sys.modules, "mlx.nn", fake_mlx_pkg.nn)
    monkeypatch.setitem(sys.modules, "mlx.optimizers", fake_mlx_pkg.optimizers)
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx_lm)
    monkeypatch.setattr(mlx_mod, "_mlx_lm", fake_mlx_lm)

    MlxBackend().load_base_model("gemma4-2B", "4bit")
    assert seen_mlx_repos == ["google/gemma-4-E2B-it"]

    with pytest.raises(TrainerError) as mlx_err:
        MlxBackend().load_base_model("gemma4-27B", "4bit")
    assert isinstance(mlx_err.value.__cause__, BaseModelResolutionError)

    # UnslothBackend resolution
    seen_unsloth_repos: list[str] = []
    fake_unsloth = types.ModuleType("unsloth")

    class _FakeFastLM:
        @staticmethod
        def from_pretrained(*, model_name: str, **_kwargs: object):
            seen_unsloth_repos.append(model_name)
            return object(), object()

    fake_unsloth.FastLanguageModel = _FakeFastLM  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "unsloth", fake_unsloth)
    UnslothBackend().load_base_model("phi4-mini", "4bit")
    assert seen_unsloth_repos == ["microsoft/Phi-4-mini-instruct"]


def test_resolve_lora_target_modules_handles_unknown_architectures_and_preserves_known() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")

    class _UnknownArchModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = types.SimpleNamespace(model_type="qwen3_5_text")
            self.q_proj = torch.nn.Linear(4, 4)
            self.v_proj = torch.nn.Linear(4, 4)
            self.lm_head = torch.nn.Linear(4, 8)

    class _KnownArchModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = types.SimpleNamespace(model_type="qwen2")
            self.q_proj = torch.nn.Linear(4, 4)
            self.v_proj = torch.nn.Linear(4, 4)

    assert resolve_lora_target_modules(_UnknownArchModel()) == ["q_proj", "v_proj"]
    assert resolve_lora_target_modules(_KnownArchModel()) is None
    assert resolve_lora_target_modules(object()) is None


def test_generate_reply_unwraps_transformers_v5_batch_encoding() -> None:
    class _V5Tokenizer:
        def apply_chat_template(self, messages, *, add_generation_prompt: bool, return_tensors: str):
            assert add_generation_prompt is True
            assert return_tensors == "pt"
            return {"input_ids": [[10, 11, 12]], "attention_mask": [[1, 1, 1]]}

        def decode(self, token_ids, *, skip_special_tokens: bool = True) -> str:
            assert skip_special_tokens is True
            return ",".join(str(t) for t in token_ids)

    class _DummyModel:
        def generate(self, input_ids, *, max_new_tokens: int, do_sample: bool, num_beams: int):
            assert input_ids == [[10, 11, 12]]
            return [[10, 11, 12, 99, 100]]

    reply = generate_reply(_DummyModel(), _V5Tokenizer(), "hello", DecodeSettings(max_new_tokens=4))
    assert reply == "99,100"


def test_pyproject_pypi_metadata_completeness_and_hf_space_verification() -> None:
    release_dev = Path(__file__).resolve().parents[2]
    pyproject = tomllib.loads((release_dev / "pyproject.toml").read_text(encoding="utf-8"))
    proj = pyproject["project"]

    assert proj["name"] == "ego-pod"
    assert proj["readme"] == "README.md"
    assert proj["license"] == {"text": "MIT"}
    assert proj["authors"] and proj["authors"][0]["email"] == "jah.g@ifay.ai"
    assert "classifiers" in proj and len(proj["classifiers"]) >= 8
    assert proj["urls"]["Repository"] == "https://github.com/ChainMode/Ego"
    assert proj["urls"]["Homepage"] == "https://ifay.ai"

    hf_app_path = release_dev.parent / "hf" / "app.py"
    assert hf_app_path.is_file()
    spec = importlib.util.spec_from_file_location("ego_hf_space_app", hf_app_path)
    assert spec is not None and spec.loader is not None
    hf_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hf_mod)

    signed_manifest = hf_mod.build_demo_signed_manifest()
    raw_manifest = json.dumps(signed_manifest)
    audit_md = hf_mod.inspect_manifest_json(raw_manifest)
    assert "passes FayGer" in audit_md

    verify_ok = hf_mod.verify_manifest_signature_json(raw_manifest)
    assert "Cryptographic Signature Verified" in verify_ok

    tampered = hf_mod.simulate_tamper(raw_manifest)
    verify_bad = hf_mod.verify_manifest_signature_json(tampered)
    assert "Tamper Detected" in verify_bad
