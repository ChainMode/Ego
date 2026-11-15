"""`ego_pod.vault.signing` 模型包两级签名、按需子集验签与防篡改单元测试。"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from safetensors.numpy import save_file

from ego_pod.exit_codes import RuntimeError_, VaultError
from ego_pod.runtime.inference import load_inference_session
from ego_pod.runtime.selection import LoadSelection
from ego_pod.trainer.cells import TrainedCell
from ego_pod.vault.pipeline import run_vault
from ego_pod.vault.signing import (
    HKDF_INFO,
    HKDF_SALT,
    SIGNING_ALG,
    SIGNING_DOMAIN,
    SIGNING_SCHEMA_VERSION,
    SignatureVerificationResult,
    canonicalize_jcs,
    compute_package_integrity,
    derive_ego_signing_subkey,
    sign_manifest,
    verify_package_signature,
)


def _seed_hex_a() -> str:
    return bytes(range(32)).hex()


def _seed_hex_b() -> str:
    return bytes(range(32, 64)).hex()


def _make_safetensors(path: Path, value: float = 1.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.array([[value, value + 1.0]], dtype=np.float32)
    save_file({"lora_A.weight": arr}, str(path))
    return path


# ---------------------------------------------------------------------------
# 1. HKDF-SHA256 子密钥派生（derive_ego_signing_subkey）
# ---------------------------------------------------------------------------


def test_derive_ego_signing_subkey_deterministic_and_distinct() -> None:
    assert HKDF_SALT == b"ifay.fayid.hkdf.v1"
    assert HKDF_INFO == b"ifay.ego.package-signing.v1"

    sub_a1 = derive_ego_signing_subkey(_seed_hex_a())
    sub_a2 = derive_ego_signing_subkey(bytes.fromhex(_seed_hex_a()))
    sub_b = derive_ego_signing_subkey(_seed_hex_b())

    assert isinstance(sub_a1, bytes) and len(sub_a1) == 32
    assert sub_a1 == sub_a2
    assert sub_a1 != bytes.fromhex(_seed_hex_a())
    assert sub_a1 != sub_b


def test_derive_ego_signing_subkey_rejects_wrong_seed_length() -> None:
    with pytest.raises(VaultError):
        derive_ego_signing_subkey(b"too-short")
    with pytest.raises(VaultError):
        derive_ego_signing_subkey("abcd")


# ---------------------------------------------------------------------------
# 2. RFC 8785 JCS 规范化（canonicalize_jcs）
# ---------------------------------------------------------------------------


def test_canonicalize_jcs_sorts_keys_compact_utf8_and_numbers() -> None:
    payload = {
        "z": [3, 1, {"b": True, "a": None}],
        "a": "中文与符号",
        "num_zero": -0.0,
        "num_float": 1.2345,
        "num_int": 4,
    }
    jcs = canonicalize_jcs(payload)
    assert isinstance(jcs, bytes)
    decoded = jcs.decode("utf-8")
    assert decoded == (
        '{"a":"中文与符号","num_float":1.2345,"num_int":4,"num_zero":0,'
        '"z":[3,1,{"a":null,"b":true}]}'
    )


def test_canonicalize_jcs_rejects_nan_and_inf() -> None:
    with pytest.raises(VaultError):
        canonicalize_jcs({"bad": float("nan")})
    with pytest.raises(VaultError):
        canonicalize_jcs({"bad": float("inf")})


# ---------------------------------------------------------------------------
# 3. Tier 1 包完整性摘要（compute_package_integrity）
# ---------------------------------------------------------------------------


def test_compute_package_integrity_flat_and_grid_posix_paths(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    cell_k = _make_safetensors(pkg / "normal" / "adapters" / "kernel.safetensors", 1.0)
    cell_c = _make_safetensors(
        pkg / "normal" / "adapters" / "coding" / "L1.safetensors", 2.0
    )

    integrity = compute_package_integrity(pkg)
    assert integrity["schema_version"] == "1.0"
    assert integrity["base_model_digest"] == "unverified_offline"
    assert list(integrity["cell_digests"].keys()) == [
        "normal/adapters/coding/L1.safetensors",
        "normal/adapters/kernel.safetensors",
    ]
    expected_k = "sha256:" + hashlib.sha256(cell_k.read_bytes()).hexdigest()
    expected_c = "sha256:" + hashlib.sha256(cell_c.read_bytes()).hexdigest()
    assert integrity["cell_digests"]["normal/adapters/kernel.safetensors"] == expected_k
    assert (
        integrity["cell_digests"]["normal/adapters/coding/L1.safetensors"] == expected_c
    )


# ---------------------------------------------------------------------------
# 4. Tier 2 签名公式、无签名兼容、篡改拒绝与 Sparse Grid 按需单 Cell 验签
# ---------------------------------------------------------------------------


def test_sign_manifest_exact_formula_and_verify_roundtrip(tmp_path: Path) -> None:
    lora_src = _make_safetensors(tmp_path / "src_lora.safetensors", 3.0)
    pkg_root = tmp_path / "ego-output"

    run_vault(
        lora_src,
        lora_rank=16,
        lora_alpha=32,
        training_backend="stub",
        output_root=pkg_root,
        version="0.1.0",
        created_at="2026-11-15T10:00:00Z",
        signing_seed_hex=_seed_hex_a(),
        signer_fay_id="ifay_alice_01",
        stdout=io.StringIO(),
    )

    manifest = json.loads((pkg_root / "manifest.json").read_text(encoding="utf-8"))
    assert "integrity" in manifest
    assert "signature" in manifest

    sig = manifest["signature"]
    assert sig["schema_version"] == SIGNING_SCHEMA_VERSION == "1.0"
    assert sig["alg"] == SIGNING_ALG == "Ed25519"
    assert sig["domain"] == SIGNING_DOMAIN == "ifay.ego.package-signing.v1"
    assert sig["signer_fay_id"] == "ifay_alice_01"
    assert sig["key_id"] == "ifay_alice_01#ego-sign-1"
    assert len(sig["public_key_hex"]) == 64
    assert len(sig["sig_hex"]) == 128

    # 独立复算签名公式，逐字节比对 payload_jcs_sha256 与 Ed25519 签名
    payload_without_sig = {k: v for k, v in manifest.items() if k != "signature"}
    jcs_bytes = canonicalize_jcs(payload_without_sig)
    assert sig["payload_jcs_sha256"] == hashlib.sha256(jcs_bytes).hexdigest()

    subkey = derive_ego_signing_subkey(_seed_hex_a())
    priv = Ed25519PrivateKey.from_private_bytes(subkey)
    expected_sig = priv.sign(b"ifay.ego.package-signing.v1:" + jcs_bytes).hex()
    assert sig["sig_hex"] == expected_sig

    res = verify_package_signature(
        pkg_root, trusted_public_key_hex=sig["public_key_hex"]
    )
    assert isinstance(res, SignatureVerificationResult)
    assert res.status == "verified"
    assert res.signer_fay_id == "ifay_alice_01"
    assert res.verified_cells == ("normal/adapters/ego-lora.safetensors",)


def test_sign_manifest_accepts_direct_32_byte_subkey(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    _make_safetensors(pkg / "normal" / "adapters" / "ego-lora.safetensors", 1.0)
    integrity = compute_package_integrity(pkg)
    manifest = {
        "version": "0.1.0",
        "created_at": "2026-11-15T10:00:00Z",
        "base_model": "qwen3.5-0.8B",
        "editions": ["normal"],
        "training_backend": "stub",
        "integrity": integrity,
    }
    direct_subkey = derive_ego_signing_subkey(_seed_hex_b())
    signed = sign_manifest(
        manifest, signer_fay_id="ifay_direct_sub", subkey=direct_subkey
    )
    (pkg / "manifest.json").write_text(
        json.dumps(signed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    res = verify_package_signature(pkg)
    assert res.status == "verified"
    assert res.signer_fay_id == "ifay_direct_sub"


def test_unsigned_package_returns_unsigned_unless_trusted_pubkey_required(
    tmp_path: Path,
) -> None:
    lora_src = _make_safetensors(tmp_path / "src_lora.safetensors", 1.0)
    pkg_root = tmp_path / "ego-output"
    run_vault(
        lora_src,
        lora_rank=16,
        lora_alpha=32,
        training_backend="stub",
        output_root=pkg_root,
        stdout=io.StringIO(),
    )

    res = verify_package_signature(pkg_root)
    assert res.status == "unsigned"

    # 若显式要求受信任公钥，未签名包必须被拒绝
    subkey = derive_ego_signing_subkey(_seed_hex_a())
    pub_hex = Ed25519PrivateKey.from_private_bytes(subkey).public_key().public_bytes_raw().hex()
    with pytest.raises(VaultError):
        verify_package_signature(pkg_root, trusted_public_key_hex=pub_hex)


def test_tampered_manifest_or_safetensors_or_wrong_pubkey_rejected(
    tmp_path: Path,
) -> None:
    lora_src = _make_safetensors(tmp_path / "src_lora.safetensors", 5.0)
    pkg_root = tmp_path / "ego-output"
    run_vault(
        lora_src,
        lora_rank=16,
        lora_alpha=32,
        training_backend="stub",
        output_root=pkg_root,
        signing_seed_hex=_seed_hex_a(),
        signer_fay_id="ifay_bob_02",
        stdout=io.StringIO(),
    )

    # 1) 错误的受信任公钥 -> VaultError
    other_sub = derive_ego_signing_subkey(_seed_hex_b())
    other_pub_hex = (
        Ed25519PrivateKey.from_private_bytes(other_sub)
        .public_key()
        .public_bytes_raw()
        .hex()
    )
    with pytest.raises(VaultError):
        verify_package_signature(pkg_root, trusted_public_key_hex=other_pub_hex)

    # 2) 篡改适配器 .safetensors 为另一份合法 SafeTensors 文件 -> VaultError & ego run RuntimeError_
    adapter_path = pkg_root / "normal" / "adapters" / "ego-lora.safetensors"
    original_bytes = adapter_path.read_bytes()
    _make_safetensors(adapter_path, 999.0)

    with pytest.raises(VaultError):
        verify_package_signature(pkg_root)
    with pytest.raises(RuntimeError_):
        load_inference_session("normal", output_root=pkg_root, stdout=io.StringIO())

    # 还原适配器后恢复可加载
    adapter_path.write_bytes(original_bytes)
    session = load_inference_session(
        "normal", output_root=pkg_root, stdout=io.StringIO()
    )
    session.close()

    # 3) 篡改 manifest.json 字段 -> VaultError & ego run RuntimeError_
    manifest_path = pkg_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["base_model"] = "tampered-model"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(VaultError):
        verify_package_signature(pkg_root)
    with pytest.raises(RuntimeError_):
        load_inference_session("normal", output_root=pkg_root, stdout=io.StringIO())


def test_sparse_grid_on_demand_single_cell_subset_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sparse Grid 按需加载：只校验选中的 Kernel + coding/L1，无需未选中的 knowledge/L1 在盘。"""
    monkeypatch.setenv("EGO_SIGNING_SEED_HEX", _seed_hex_a())
    monkeypatch.setenv("EGO_SIGNER_FAY_ID", "ifay_grid_signer")

    src_k = _make_safetensors(tmp_path / "k.safetensors", 1.0)
    src_c = _make_safetensors(tmp_path / "c.safetensors", 2.0)
    src_kn = _make_safetensors(tmp_path / "kn.safetensors", 3.0)

    cells = [
        TrainedCell(zone=None, level=0, output_path=src_k),
        TrainedCell(zone="coding", level=1, output_path=src_c),
        TrainedCell(zone="knowledge", level=1, output_path=src_kn),
    ]
    pkg_root = tmp_path / "ego-output"
    run_vault(
        cells=cells,
        lora_rank=16,
        lora_alpha=32,
        training_backend="stub",
        output_root=pkg_root,
        stdout=io.StringIO(),
    )

    # 模拟按需拉取场景：未选中的 knowledge/L1 尚未下载到本地磁盘
    unselected_path = pkg_root / "normal" / "adapters" / "knowledge" / "L1.safetensors"
    unselected_path.unlink()

    # 全量校验应失败（因为 knowledge/L1 缺失）
    with pytest.raises(VaultError):
        verify_package_signature(pkg_root)

    # 按需子集校验（仅 Kernel + coding/L1）应通过！
    subset = [
        "normal/adapters/kernel.safetensors",
        "normal/adapters/coding/L1.safetensors",
    ]
    res = verify_package_signature(pkg_root, selected_cell_rel_paths=subset)
    assert res.status == "verified"
    assert set(res.verified_cells) == set(subset)

    # Nested_In_Load 仅请求 coding 到深度 1：load_inference_session 自动执行子集验签并成功！
    session = load_inference_session(
        "normal",
        output_root=pkg_root,
        selection=LoadSelection(depths={"coding": 1}),
        stdout=io.StringIO(),
    )
    session.close()

    # 一旦被选中的 coding/L1 遭到篡改，子集验签与 load_inference_session 立即拦截！
    _make_safetensors(
        pkg_root / "normal" / "adapters" / "coding" / "L1.safetensors", 42.0
    )
    with pytest.raises(VaultError):
        verify_package_signature(pkg_root, selected_cell_rel_paths=subset)
    with pytest.raises(RuntimeError_):
        load_inference_session(
            "normal",
            output_root=pkg_root,
            selection=LoadSelection(depths={"coding": 1}),
            stdout=io.StringIO(),
        )


def test_partial_signing_env_raises_vault_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """仅设置 EGO_SIGNING_SEED_HEX 而缺少 EGO_SIGNER_FAY_ID 时必须显式报错。"""
    monkeypatch.setenv("EGO_SIGNING_SEED_HEX", _seed_hex_a())
    monkeypatch.delenv("EGO_SIGNER_FAY_ID", raising=False)
    lora_src = _make_safetensors(tmp_path / "src.safetensors", 1.0)
    with pytest.raises(VaultError):
        run_vault(
            lora_src,
            lora_rank=16,
            lora_alpha=32,
            training_backend="stub",
            output_root=tmp_path / "ego-output",
            stdout=io.StringIO(),
        )


def test_canonicalize_jcs_es6_float_boundary_cases() -> None:
    """RFC 8785 JCS 浮点数边界（-0.0、整数化浮点、1e-6 固定点、1e-7/1e21 指数格式、拒绝 NaN/Inf）。"""
    jcs = canonicalize_jcs(
        {
            "neg_zero": -0.0,
            "int_float": 42.0,
            "fixed_small": 1e-6,
            "exp_small": 1e-7,
            "exp_large": 1e21,
        }
    ).decode("utf-8")
    assert (
        jcs
        == '{"exp_large":1e+21,"exp_small":1e-7,"fixed_small":0.000001,"int_float":42,"neg_zero":0}'
    )
    with pytest.raises(VaultError):
        canonicalize_jcs({"bad": float("nan")})
    with pytest.raises(VaultError):
        canonicalize_jcs({"bad": float("inf")})


def test_derive_subkey_rejects_bad_length_and_non_hex() -> None:
    with pytest.raises(VaultError):
        derive_ego_signing_subkey("abcd")
    with pytest.raises(VaultError):
        derive_ego_signing_subkey("zz" * 32)
    with pytest.raises(VaultError):
        derive_ego_signing_subkey(b"short")


def test_end_to_end_ego_train_writes_fidelity_integrity_and_verifiable_signature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """端到端 ego train：自动写入 fidelity_report + integrity + signature，且不泄露原始语料并可通过 ego run 验签。"""
    import argparse

    from ego_pod.commands.train import run_train
    from ego_pod.config.defaults import DEFAULT_SOURCE_PATHS, write_default
    from ego_pod.trainer.stub import StubBackend

    ws = tmp_path / "ws"
    ws.mkdir(parents=True)
    for rel in DEFAULT_SOURCE_PATHS.values():
        (ws / rel).mkdir(parents=True, exist_ok=True)

    canary_utterance = "PRIVATE_CANARY_UTTERANCE_EXAMPLE_99421"
    # 8 条 sense (Kernel) + 8 条 thought (knowledge/L1) = 16 条样本，端到端覆盖 Sparse Grid
    for i in range(8):
        (ws / "ego-data" / "sense" / f"sense_{i:02d}.md").write_text(
            f"Observation {i}: concise direct tone preferred ({canary_utterance}_s{i}).\n",
            encoding="utf-8",
        )
        (ws / "ego-data" / "thought" / f"note_{i:02d}.md").write_text(
            f"Architecture principle {i}: keep module {i} pure and deterministic ({canary_utterance}_t{i}).\n",
            encoding="utf-8",
        )
    cfg_path = ws / "ego-config.yaml"
    write_default(cfg_path)

    monkeypatch.chdir(ws)
    monkeypatch.setenv("EGO_SIGNING_SEED_HEX", _seed_hex_a())
    monkeypatch.setenv("EGO_SIGNER_FAY_ID", "ifay_e2e_prime_01")
    monkeypatch.setattr("ego_pod.trainer.detect.detect_backend", lambda: StubBackend())
    monkeypatch.setattr("ego_pod.trainer.grid.detect_backend", lambda: StubBackend())
    monkeypatch.setattr("ego_pod.trainer.pipeline.detect_backend", lambda: StubBackend())

    rc = run_train(argparse.Namespace(config=str(cfg_path)))
    assert rc == 0

    pkg_root = ws / "ego-output"
    manifest_raw = (pkg_root / "manifest.json").read_text(encoding="utf-8")
    assert canary_utterance not in manifest_raw  # 隐私红线：绝不泄露原始文本

    manifest = json.loads(manifest_raw)
    assert "fidelity_report" in manifest
    assert manifest["fidelity_report"]["schema_version"] == "1.0"
    assert manifest["fidelity_report"]["status"] in ("measured", "insufficient_data")
    assert "integrity" in manifest
    assert len(manifest["integrity"]["cell_digests"]) >= 1
    assert "signature" in manifest
    assert manifest["signature"]["signer_fay_id"] == "ifay_e2e_prime_01"

    ver = verify_package_signature(pkg_root)
    assert ver.status == "verified"

    session = load_inference_session("normal", output_root=pkg_root, stdout=io.StringIO())
    session.close()
