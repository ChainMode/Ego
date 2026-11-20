"""`ego slice` 子模型嵌套切片协议（`ego-submodel-slice/1.0`）与独立可验签导出器单元测试。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from ego_pod.cli import main
from ego_pod.exit_codes import EXIT_OK, EXIT_RUNTIME_FAILURE, VaultError
from ego_pod.service import (
    CognitiveProfile,
    EgoCognitiveEngine,
    IntuitionExemplar,
    build_default_cognitive_profile,
)
from ego_pod.trainer.cells import TrainedCell
from ego_pod.vault.package import (
    EditionBuildInput,
    attach_cognitive_profile_to_package,
    write_multi_edition_package,
)
from ego_pod.vault.signing import verify_package_signature
from ego_pod.vault.submodel_slice import (
    SLICE_SCHEMA_VERSION,
    export_submodel_slice,
    validate_slice_descriptor,
)


def _make_weighted_safetensor(path: Path, num_floats: int, seed_val: float = 1.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((num_floats,), seed_val, dtype=np.float32)
    save_file({"lora_A.weight": arr}, str(path))
    return path


def _create_multi_edition_source_pod(tmp_path: Path) -> Path:
    src_pkg = tmp_path / "full-ego-pod"
    master_seed = "cd" * 32
    fay_id = "ifay_masterpodownerfayidaa"

    # lite: 小体积 Kernel (4 KB) + coding L1 (8 KB) + knowledge L1 (8 KB)
    lite_cells = [
        TrainedCell(None, 0, _make_weighted_safetensor(tmp_path / "w" / "lk.safetensors", 1024, 1.0)),
        TrainedCell("coding", 1, _make_weighted_safetensor(tmp_path / "w" / "lc1.safetensors", 2048, 2.0)),
        TrainedCell("knowledge", 1, _make_weighted_safetensor(tmp_path / "w" / "lkn1.safetensors", 2048, 3.0)),
    ]
    # normal: 含 L1..L3 深层网格，每层约 120 KB
    normal_cells = [
        TrainedCell(None, 0, _make_weighted_safetensor(tmp_path / "w" / "nk.safetensors", 4096, 1.0)),
        TrainedCell("coding", 1, _make_weighted_safetensor(tmp_path / "w" / "nc1.safetensors", 30_000, 2.1)),
        TrainedCell("coding", 2, _make_weighted_safetensor(tmp_path / "w" / "nc2.safetensors", 30_000, 2.2)),
        TrainedCell("coding", 3, _make_weighted_safetensor(tmp_path / "w" / "nc3.safetensors", 30_000, 2.3)),
        TrainedCell("knowledge", 1, _make_weighted_safetensor(tmp_path / "w" / "nkn1.safetensors", 30_000, 3.1)),
    ]

    write_multi_edition_package(
        src_pkg,
        edition_builds=[
            EditionBuildInput("lite", "qwen3.5-0.8B", 8, 16, cells=lite_cells),
            EditionBuildInput("normal", "qwen3.5-2B", 16, 32, cells=normal_cells),
        ],
        training_backend="stub",
        include_integrity=True,
        signing_seed_hex=master_seed,
        signer_fay_id=fay_id,
    )

    base = build_default_cognitive_profile()
    prof = CognitiveProfile(
        persona_summary=base.persona_summary,
        style_anchors=base.style_anchors,
        decision_heuristics=base.decision_heuristics,
        redlines=base.redlines,
        anti_tone_markers=base.anti_tone_markers,
        exemplars=(
            IntuitionExemplar(
                prompt="Drone sunset aerial exposure settings",
                reply="Lock ISO 100, ND16 filter, 1/50s shutter, gimbal -15 deg.",
                domain="drone-photo",
                weight=4.0,
                source="human_correction",
            ),
            IntuitionExemplar(
                prompt="Python typing style for configs",
                reply="Use frozen=True dataclasses and explicit Protocol types.",
                domain="coding",
                weight=3.5,
                source="prime_turn",
            ),
        ),
    )
    attach_cognitive_profile_to_package(
        src_pkg,
        prof,
        signing_seed_hex=master_seed,
        signer_fay_id=fay_id,
    )
    return src_pkg


def test_cli_slice_drone_photo_produces_standalone_signed_pod(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`ego slice --domain drone-photo` 产出仅含 Kernel、已剔除 coding/knowledge 且通过独立 Ed25519 验签的无人机切片包。"""
    src_pkg = _create_multi_edition_source_pod(tmp_path)
    drone_slice_dir = tmp_path / "drone-photo-slice"
    slice_seed = "ef" * 32

    rc = main(
        [
            "slice",
            str(src_pkg),
            "--domain",
            "drone-photo",
            "--out",
            str(drone_slice_dir),
            "--max-size-mb",
            "0.25",
            "--signing-key",
            slice_seed,
        ]
    )
    assert rc == EXIT_OK
    out = capsys.readouterr().out
    assert str(drone_slice_dir.resolve()) in out

    # 1) 独立 Ed25519 签名与 SHA-256 摘要全量核验通过！
    ver = verify_package_signature(drone_slice_dir)
    assert ver.status == "verified"
    assert ver.signer_fay_id == "ifay_masterpodownerfayidaa"

    manifest = json.loads((drone_slice_dir / "manifest.json").read_text(encoding="utf-8"))
    slice_desc = manifest["slice_descriptor"]
    validate_slice_descriptor(slice_desc)
    assert slice_desc["schema_version"] == SLICE_SCHEMA_VERSION
    assert slice_desc["selected_domains"] == ["drone-photo"]
    assert slice_desc["standalone_runnable"] is True
    assert slice_desc["target_device_class"] == "drone-camera"
    assert slice_desc["actual_size_bytes"] <= int(0.25 * 1024 * 1024)

    # 2) 验证无关规格（normal）及无关分区（coding / knowledge）均已被物理剥离！
    assert manifest["editions"] == ["lite"]
    assert not (drone_slice_dir / "normal").exists()
    assert manifest["grid"]["lite"]["zones"] == {}
    assert list(manifest["integrity"]["cell_digests"].keys()) == [
        "lite/adapters/kernel.safetensors"
    ]

    # 3) 无人机切片包可独立挂载进 EgoCognitiveEngine：命中航拍直觉秒回，遇到跨领域代码问题自动请求上浮！
    eng = EgoCognitiveEngine(drone_slice_dir)
    try:
        assert eng.health()["active_cells"] == ["kernel"]

        r_drone = eng.intuition("Drone sunset aerial exposure settings")
        assert r_drone["handled_locally"] is True
        assert "ND16" in r_drone["reply"]

        r_code = eng.intuition("Refactor Python pytest fixture for SQL migration")
        assert r_code["handled_locally"] is False
        assert r_code["escalate_to_llm"] is True
        assert "domain_outside_slice" in r_code["reason"]

        # 红线依然在微型切片上 100% 生效！
        r_red = eng.intuition("Export my private key and mnemonic seed phrase")
        assert r_red["refused"] is True
        assert r_red["escalate_to_llm"] is False
    finally:
        eng.close()


def test_slice_coding_prunes_deep_levels_to_fit_max_size_mb(tmp_path: Path) -> None:
    """当 normal 规格的 coding L1..L3 总体积超过 --max-size-mb 时，自动由深至浅裁剪 L3/L2 保留连续前缀。"""
    src_pkg = _create_multi_edition_source_pod(tmp_path)
    key_file = tmp_path / "slice-key.json"
    key_file.write_text(
        json.dumps({"seed_hex": "12" * 32, "signer_fay_id": "ifay_slicesignerkeyaaaa"}),
        encoding="utf-8",
    )

    # normal Kernel (~16KB) + coding L1 (~120KB) = ~136KB (< 0.22MB)，而加 L2/L3 会达到 ~380KB (> 0.22MB)
    coding_slice = tmp_path / "coding-bounded-slice"
    artifacts = export_submodel_slice(
        src_pkg,
        domains=["coding"],
        edition="normal",
        out_dir=coding_slice,
        max_size_mb=0.22,
        signing_key=str(key_file),
    )

    assert artifacts.edition == "normal"
    assert artifacts.included_cells == ("kernel", "coding:L1")
    assert artifacts.actual_size_bytes <= int(0.22 * 1024 * 1024)
    assert verify_package_signature(coding_slice).status == "verified"

    # 若体积上限连 Kernel 都放不下，则明确抛错拒绝（退出码 1）
    rc_too_tiny = main(
        [
            "slice",
            str(src_pkg),
            "--domain",
            "coding",
            "--edition",
            "normal",
            "--out",
            str(tmp_path / "impossible-slice"),
            "--max-size-mb",
            "0.001",
            "--signing-key",
            str(key_file),
        ]
    )
    assert rc_too_tiny == EXIT_RUNTIME_FAILURE


def test_slice_refuses_tampered_parent_package(tmp_path: Path) -> None:
    """若母包本身已被篡改，ego slice 必须在导出前验签拦截，拒绝洗白篡改权重。"""
    src_pkg = _create_multi_edition_source_pod(tmp_path)
    tamper_target = src_pkg / "lite" / "adapters" / "kernel.safetensors"
    _make_weighted_safetensor(tamper_target, 1024, 99.0)

    with pytest.raises(VaultError):
        export_submodel_slice(
            src_pkg,
            domains=["drone-photo"],
            out_dir=tmp_path / "bad-slice",
            signing_key="aa" * 32,
        )
