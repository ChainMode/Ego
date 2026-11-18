"""Post-training cell_consistency_score probe and canonical signer FayID enforcement."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ego_pod.exit_codes import VaultError
from ego_pod.fidelity.consistency_probe import (
    collect_holdout_prompts,
    measure_package_cell_consistency,
)
from ego_pod.fidelity.metrics import split_holdout_deterministic
from ego_pod.vault.package import refresh_manifest_fidelity
from ego_pod.vault.signing import (
    compute_package_integrity,
    sign_manifest,
    verify_package_signature,
)

CANONICAL_FAY_ID = "ifay_consistencyprobeaa"
SEED_HEX = "44" * 32


def _write_corpus(path: Path, n: int = 12) -> list[dict]:
    records = [
        {
            "instruction": f"prompt number {i}",
            "input": "" if i % 3 else f"context {i}",
            "output": f"answer {i}",
            "category": "thought",
        }
        for i in range(n)
    ]
    records.append({"instruction": None, "input": "   ", "output": "x", "category": "sense"})
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return records


def _write_package(root: Path, *, backend: str, zoned: bool, signed: bool) -> None:
    (root / "lite" / "adapters").mkdir(parents=True)
    (root / "lite" / "adapters" / "kernel.safetensors").write_bytes(b"kernel-bytes")
    grid_zones = {"coding": [{"level": 1, "relpath": "adapters/coding/L1.safetensors"}]} if zoned else {}
    if zoned:
        (root / "lite" / "adapters" / "coding").mkdir()
        (root / "lite" / "adapters" / "coding" / "L1.safetensors").write_bytes(b"l1-bytes")
    manifest = {
        "version": "0.2.0",
        "created_at": "2026-11-18T00:00:00Z",
        "base_model": "qwen3.5-0.8B",
        "editions": ["lite"],
        "training_backend": backend,
        "grid": {"lite": {"kernel": "adapters/kernel.safetensors", "zones": grid_zones}},
        "fidelity_report": {
            "schema_version": "1.0",
            "status": "measured",
            "holdout_sample_count": 3,
            "holdout_loss": 1.0,
            "holdout_perplexity": 2.718282,
            "cell_consistency_score": None,
            "probe_win_rate": None,
            "drift_from_baseline": None,
        },
    }
    manifest["integrity"] = compute_package_integrity(root, base_model_digest="unverified_offline")
    if signed:
        manifest = sign_manifest(manifest, signer_fay_id=CANONICAL_FAY_ID, master_seed=SEED_HEX)
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


class _FakeSession:
    def __init__(self, kernel_only: bool) -> None:
        self.kernel_only = kernel_only
        self.closed = False

    def generate(self, prompt: str) -> str:
        return f"reply to {prompt}" if self.kernel_only else f"reply to {prompt} with depth"

    def close(self) -> None:
        self.closed = True


def _fake_loader(sessions: list[_FakeSession]):
    def loader(edition, *, cwd, output_root, selection):
        s = _FakeSession(kernel_only=selection is not None and not selection.depths)
        sessions.append(s)
        return s

    return loader


def test_holdout_prompts_come_only_from_holdout_split_and_skip_blank(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    records = _write_corpus(corpus)
    _train, holdout = split_holdout_deterministic(records)
    holdout_instructions = {r["instruction"] for r in holdout if r.get("instruction")}
    prompts = collect_holdout_prompts(corpus)
    assert prompts, "expected at least one usable holdout prompt"
    for p in prompts:
        assert p.split("\n")[0] in holdout_instructions
    assert all(p.strip() for p in prompts)


def test_stub_backend_is_never_scored(tmp_path: Path) -> None:
    root = tmp_path / "ego-output"
    _write_package(root, backend="stub", zoned=True, signed=False)
    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(corpus)
    score, reason = measure_package_cell_consistency(
        root, corpus, cwd=tmp_path, session_loader=_fake_loader([])
    )
    assert score is None and "stub" in reason


def test_kernel_only_grid_is_skipped(tmp_path: Path) -> None:
    root = tmp_path / "ego-output"
    _write_package(root, backend="peft-cpu", zoned=False, signed=False)
    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(corpus)
    score, reason = measure_package_cell_consistency(
        root, corpus, cwd=tmp_path, session_loader=_fake_loader([])
    )
    assert score is None and "zone" in reason


def test_real_backend_scores_kernel_vs_full_and_closes_sessions(tmp_path: Path) -> None:
    root = tmp_path / "ego-output"
    _write_package(root, backend="peft-cpu", zoned=True, signed=False)
    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(corpus)
    sessions: list[_FakeSession] = []
    score, _reason = measure_package_cell_consistency(
        root, corpus, cwd=tmp_path, session_loader=_fake_loader(sessions)
    )
    assert score is not None and 0.0 < score < 1.0
    assert len(sessions) == 2 and all(s.closed for s in sessions)
    assert {s.kernel_only for s in sessions} == {True, False}


def test_refresh_manifest_fidelity_resigns_and_still_verifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EGO_SIGNING_SEED_HEX", SEED_HEX)
    monkeypatch.setenv("EGO_SIGNER_FAY_ID", CANONICAL_FAY_ID)
    root = tmp_path / "ego-output"
    _write_package(root, backend="peft-cpu", zoned=True, signed=True)
    before = json.loads((root / "manifest.json").read_text())
    report = dict(before["fidelity_report"], cell_consistency_score=0.83)
    refresh_manifest_fidelity(root, report)
    after = json.loads((root / "manifest.json").read_text())
    assert after["fidelity_report"]["cell_consistency_score"] == 0.83
    assert after["integrity"] == before["integrity"]
    assert after["signature"]["sig_hex"] != before["signature"]["sig_hex"]
    assert verify_package_signature(root).status == "verified"


def test_refresh_on_signed_package_without_signing_config_fails_loudly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("EGO_SIGNING_SEED_HEX", raising=False)
    monkeypatch.delenv("EGO_SIGNER_FAY_ID", raising=False)
    root = tmp_path / "ego-output"
    _write_package(root, backend="peft-cpu", zoned=True, signed=True)
    report = dict(json.loads((root / "manifest.json").read_text())["fidelity_report"])
    with pytest.raises(VaultError):
        refresh_manifest_fidelity(root, report)


@pytest.mark.parametrize(
    "bad_id",
    [
        "ifay_alice_01",  # underscore + digits outside base32 alphabet
        "ifay_ed25519_feedfacecafebeef",
        "ifay_ABCDEFGHIJKLMNOPQR",  # uppercase
        "ifay_short",  # body < 16
        "cofay_aaaaaaaaaaaaaaaaaaaa",  # wrong prefix
    ],
)
def test_sign_manifest_rejects_non_canonical_fay_ids(tmp_path: Path, bad_id: str) -> None:
    root = tmp_path / "pkg"
    root.mkdir()
    (root / "kernel.safetensors").write_bytes(b"k")
    manifest = {
        "integrity": compute_package_integrity(root, base_model_digest="unverified_offline")
    }
    with pytest.raises(VaultError):
        sign_manifest(manifest, signer_fay_id=bad_id, master_seed=SEED_HEX)
