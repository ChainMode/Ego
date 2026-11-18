"""`ego vault list / rollback / purge` CLI 与完整性防护单测（ADR 0006 §6）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from safetensors.torch import save_file
import torch

from ego_pod.cli import main
from ego_pod.exit_codes import EXIT_OK, EXIT_RUNTIME_FAILURE, VaultError
from ego_pod.trainer.cells import TrainedCell
from ego_pod.vault.history import list_vault_snapshots, rollback_vault_snapshot
from ego_pod.vault.package import EditionBuildInput, write_multi_edition_package
from ego_pod.vault.signing import derive_ego_signing_subkey, verify_package_signature


def _make_safetensor(path: Path, val: float = 1.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"lora_A": torch.tensor([val], dtype=torch.float32)}, str(path))
    return path


def _pubkey_hex_from_seed(seed_hex: str) -> str:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    sub = derive_ego_signing_subkey(seed_hex)
    return Ed25519PrivateKey.from_private_bytes(sub).public_key().public_bytes_raw().hex()


def test_ego_vault_list_rollback_and_tamper_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """验证 ego vault list、带验签回滚、篡改快照拦截以及缺失快照提示。"""
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "ego-output"
    seed_hex = "22" * 32
    fay_id = "ifay_edabcdefghijklmnopqr"
    pubkey_hex = _pubkey_hex_from_seed(seed_hex)

    # 空目录调用 ego vault list 正常返回 0
    assert main(["vault", "list"]) == EXIT_OK
    out_empty = capsys.readouterr().out
    assert "no snapshots" in out_empty.lower() or "无历史快照" in out_empty

    # 构建 v0.1.9 (lite + normal) -> 再构建 v0.2.0 (仅 normal) 以触发归档 0.1.9-0001
    write_multi_edition_package(
        root,
        edition_builds=[
            EditionBuildInput(
                edition="lite",
                base_model="qwen3.5-0.8B",
                lora_rank=8,
                lora_alpha=16,
                cells=[
                    TrainedCell(None, 0, _make_safetensor(tmp_path / "s1" / "lk.safetensors", 1.0)),
                    TrainedCell("coding", 1, _make_safetensor(tmp_path / "s1" / "lc1.safetensors", 1.0)),
                ],
            ),
            EditionBuildInput(
                edition="normal",
                base_model="qwen3.5-2B",
                lora_rank=16,
                lora_alpha=32,
                cells=[
                    TrainedCell(None, 0, _make_safetensor(tmp_path / "s1" / "nk.safetensors", 1.0)),
                    TrainedCell("coding", 1, _make_safetensor(tmp_path / "s1" / "nc1.safetensors", 1.0)),
                ],
            ),
        ],
        training_backend="stub",
        version="0.1.9",
        include_integrity=True,
        signing_seed_hex=seed_hex,
        signer_fay_id=fay_id,
    )

    write_multi_edition_package(
        root,
        edition_builds=[
            EditionBuildInput(
                edition="normal",
                base_model="gemma4-2B",
                lora_rank=16,
                lora_alpha=32,
                cells=[
                    TrainedCell(None, 0, _make_safetensor(tmp_path / "s2" / "nk.safetensors", 2.0)),
                    TrainedCell("coding", 1, _make_safetensor(tmp_path / "s2" / "nc1.safetensors", 2.0)),
                ],
            ),
        ],
        training_backend="stub",
        version="0.2.0",
        include_integrity=True,
        signing_seed_hex=seed_hex,
        signer_fay_id=fay_id,
    )
    capsys.readouterr()

    # 1. ego vault list 列出归档快照 0.1.9-0001
    assert main(["vault", "list"]) == EXIT_OK
    list_out = capsys.readouterr().out
    assert "0.1.9-0001" in list_out
    assert "lite" in list_out

    # 2. 不存在的快照 ID 报错退出码 1 并列出可用快照 0.1.9-0001
    assert main(["vault", "rollback", "non-existent-snap"]) == EXIT_RUNTIME_FAILURE
    err_missing = capsys.readouterr().err
    assert "0.1.9-0001" in err_missing

    # 3. 故意篡改快照内的适配器字节：rollback 必须在恢复前拒绝且不破坏当前 0.2.0 活动包
    snap_adapter = root / ".history" / "0.1.9-0001" / "lite" / "adapters" / "kernel.safetensors"
    orig_bytes = snap_adapter.read_bytes()
    snap_adapter.write_bytes(b"corrupted-bytes")
    try:
        rc_tamper = main(
            ["vault", "rollback", "0.1.9-0001", "--public-key-hex", pubkey_hex]
        )
        assert rc_tamper == EXIT_RUNTIME_FAILURE
        active_manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        assert active_manifest["version"] == "0.2.0"
        assert active_manifest["editions"] == ["normal"]
    finally:
        snap_adapter.write_bytes(orig_bytes)

    # 4. 恢复正常字节后执行带受信任公钥验签的回滚 -> 成功回到 0.1.9 (lite + normal)
    rc_ok = main(["vault", "rollback", "0.1.9-0001", "--public-key-hex", pubkey_hex])
    assert rc_ok == EXIT_OK
    restored_manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert restored_manifest["version"] == "0.1.9"
    assert restored_manifest["editions"] == ["lite", "normal"]
    assert verify_package_signature(root, trusted_public_key_hex=pubkey_hex).status == "verified"


def test_ego_vault_purge_all_deletes_output_and_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ego vault purge --all 彻底删除 ego-output/（含 .history/）与 .ego/ 缓存。"""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ego-output" / ".history" / "0.1.0-0001").mkdir(parents=True)
    (tmp_path / "ego-output" / "manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / ".ego" / "cells").mkdir(parents=True)
    (tmp_path / ".ego" / "corpus.jsonl").write_text("{}", encoding="utf-8")

    rc = main(["vault", "purge", "--all"])
    assert rc == EXIT_OK
    assert not (tmp_path / "ego-output").exists()
    assert not (tmp_path / ".ego").exists()


def test_vault_history_fifo_pruning_and_zero_retention(tmp_path: Path) -> None:
    """连续写入超出 max_history=3 时按 FIFO 仅保留最近 3 份快照；max_history=0 时清空历史。"""
    root = tmp_path / "ego-output"
    for idx in range(1, 6):
        write_multi_edition_package(
            root,
            edition_builds=[
                EditionBuildInput(
                    edition="lite",
                    base_model="qwen3.5-0.8B",
                    lora_rank=8,
                    lora_alpha=16,
                    cells=[
                        TrainedCell(None, 0, _make_safetensor(tmp_path / f"w{idx}" / "k.safetensors", float(idx))),
                        TrainedCell("coding", 1, _make_safetensor(tmp_path / f"w{idx}" / "c1.safetensors", float(idx))),
                    ],
                )
            ],
            training_backend="stub",
            version=f"0.2.{idx}",
            include_integrity=True,
            max_history=3,
        )

    # 共写 5 次（产生 4 次覆盖归档：0.2.1..0.2.4），max_history=3 应淘汰 0.2.1-0001，保留 0002..0004
    snaps = list_vault_snapshots(root)
    assert [s.snapshot_id for s in snaps] == [
        "0.2.2-0002",
        "0.2.3-0003",
        "0.2.4-0004",
    ]

    # max_history=0 关闭保留并清理旧快照目录
    write_multi_edition_package(
        root,
        edition_builds=[
            EditionBuildInput(
                edition="lite",
                base_model="qwen3.5-0.8B",
                lora_rank=8,
                lora_alpha=16,
                cells=[
                    TrainedCell(None, 0, _make_safetensor(tmp_path / "w6" / "k.safetensors", 6.0)),
                    TrainedCell("coding", 1, _make_safetensor(tmp_path / "w6" / "c1.safetensors", 6.0)),
                ],
            )
        ],
        training_backend="stub",
        version="0.2.6",
        include_integrity=True,
        max_history=0,
    )
    assert list_vault_snapshots(root) == []


@pytest.mark.parametrize("bad_id", ["", "../escape", "sub/dir", "sub\\dir", ".."])
def test_rollback_rejects_path_traversal_snapshot_ids(tmp_path: Path, bad_id: str) -> None:
    with pytest.raises(VaultError):
        rollback_vault_snapshot(tmp_path / "ego-output", bad_id)


def test_rollback_rejects_wrong_trusted_public_key(tmp_path: Path) -> None:
    """回滚传入不匹配的受信任公钥时拒绝恢复且保持活动包不变。"""
    root = tmp_path / "ego-output"
    seed_hex = "44" * 32
    fay_id = "ifay_ed4444444444444444"

    for idx, ver in enumerate(("0.1.0", "0.2.0"), start=1):
        write_multi_edition_package(
            root,
            edition_builds=[
                EditionBuildInput(
                    edition="lite",
                    base_model="qwen3.5-0.8B",
                    lora_rank=8,
                    lora_alpha=16,
                    cells=[
                        TrainedCell(None, 0, _make_safetensor(tmp_path / f"k{idx}" / "k.safetensors")),
                        TrainedCell("coding", 1, _make_safetensor(tmp_path / f"k{idx}" / "c.safetensors")),
                    ],
                )
            ],
            training_backend="stub",
            version=ver,
            include_integrity=True,
            signing_seed_hex=seed_hex,
            signer_fay_id=fay_id,
        )

    wrong_pubkey = "ff" * 32
    with pytest.raises(VaultError):
        rollback_vault_snapshot(root, "0.1.0-0001", trusted_public_key_hex=wrong_pubkey)

    active = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert active["version"] == "0.2.0"
