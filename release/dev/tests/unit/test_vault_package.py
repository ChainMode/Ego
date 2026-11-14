"""`ego_pod.vault.package` 的字段与类型单元测试（任务 9.5）。

覆盖设计文档 §6「Vault 模块」与需求：

* **R10-AC2**：``manifest.json`` 含 ``version``（SemVer 字符串）、``created_at``
  （UTC ISO 8601 字符串）、``base_model``、``editions``（非空字符串数组且含
  ``normal``）；MVP 补充 ``training_backend`` 字段（需求 12.9）。
* **R10-AC4**：``normal/config.json`` 含 ``base_model``（字符串）、
  ``quantization``（固定 ``4bit``）、``lora_rank``（正整数）、``lora_alpha``
  （正整数）。

本文件以明示样例 + 边界用例验证 :func:`build_config`、:func:`build_manifest`、
:func:`utc_now_iso8601` 三个纯函数，以及 :func:`write_package` 落盘后从磁盘读回
的字段格式与类型；重点关注 ``training_backend`` 的「写入与读回」往返一致。

测试使用 ``tmp_path`` 写入临时文件，并以 :func:`safetensors.numpy.save_file`
构造 1×1 占位张量作为 Trainer 输出夹具，不依赖网络或真实 ML 依赖。
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

from ego_pod import __version__
from ego_pod.exit_codes import VaultError
from ego_pod.vault.package import (
    ADAPTER_FILENAME,
    BASE_MODEL,
    CONFIG_FILENAME,
    EDITION_NAME,
    MANIFEST_FILENAME,
    QUANTIZATION,
    build_config,
    build_manifest,
    utc_now_iso8601,
    write_package,
)

# 宽松 SemVer 2.0.0 核心匹配（MAJOR.MINOR.PATCH，允许预发布/构建元数据后缀）。
_SEMVER_RE = re.compile(
    r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
)
# 严格 ISO 8601 UTC 秒精度格式（以 Z 结尾），与 utc_now_iso8601 一致。
_ISO8601_FMT = "%Y-%m-%dT%H:%M:%SZ"


def _make_lora_source(path: Path) -> Path:
    """写一个合法的 1×1 ``.safetensors`` 文件作为 Trainer 输出夹具。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"placeholder": np.zeros((1, 1), dtype=np.float32)}, str(path))
    return path


# ===========================================================================
# build_config —— config.json 字段与类型（R10-AC4）
# ===========================================================================


def test_build_config_has_required_fields_and_types():
    config = build_config(lora_rank=16, lora_alpha=32)

    assert set(config) == {"base_model", "quantization", "lora_rank", "lora_alpha"}
    assert isinstance(config["base_model"], str)
    assert config["base_model"] == BASE_MODEL
    assert config["quantization"] == QUANTIZATION == "4bit"
    assert isinstance(config["lora_rank"], int)
    assert isinstance(config["lora_alpha"], int)
    assert config["lora_rank"] == 16
    assert config["lora_alpha"] == 32


def test_build_config_preserves_positive_integer_values():
    config = build_config(lora_rank=1, lora_alpha=1)
    assert config["lora_rank"] == 1
    assert config["lora_alpha"] == 1

    big = build_config(lora_rank=4096, lora_alpha=8192)
    assert big["lora_rank"] == 4096
    assert big["lora_alpha"] == 8192


def test_build_config_custom_base_model():
    config = build_config(lora_rank=8, lora_alpha=16, base_model="custom-base")
    assert config["base_model"] == "custom-base"
    # quantization 始终固定为 4bit，不随 base_model 改变。
    assert config["quantization"] == "4bit"


# ===========================================================================
# build_manifest —— manifest.json 字段、类型与 editions 约束（R10-AC2、12.9）
# ===========================================================================


def test_build_manifest_has_required_fields_and_types():
    manifest = build_manifest(training_backend="stub")

    assert set(manifest) == {
        "version",
        "created_at",
        "base_model",
        "editions",
        "training_backend",
    }
    assert isinstance(manifest["version"], str)
    assert isinstance(manifest["created_at"], str)
    assert isinstance(manifest["base_model"], str)
    assert isinstance(manifest["editions"], list)
    assert isinstance(manifest["training_backend"], str)


def test_build_manifest_version_is_semver():
    manifest = build_manifest(training_backend="stub")
    # 默认取包 __version__，应为合法 SemVer。
    assert manifest["version"] == __version__
    assert _SEMVER_RE.match(manifest["version"])


def test_build_manifest_explicit_version_is_used():
    manifest = build_manifest(training_backend="stub", version="2.3.4")
    assert manifest["version"] == "2.3.4"
    assert _SEMVER_RE.match(manifest["version"])


def test_build_manifest_created_at_is_iso8601_utc():
    manifest = build_manifest(training_backend="stub")
    created_at = manifest["created_at"]
    # 以 Z 结尾且可被严格 ISO 8601 UTC 秒精度格式解析。
    assert created_at.endswith("Z")
    datetime.strptime(created_at, _ISO8601_FMT)


def test_build_manifest_explicit_created_at_is_used():
    fixed = "2025-11-25T08:00:00Z"
    manifest = build_manifest(training_backend="stub", created_at=fixed)
    assert manifest["created_at"] == fixed


def test_build_manifest_editions_non_empty_and_contains_normal():
    manifest = build_manifest(training_backend="stub")
    assert manifest["editions"]  # 非空
    assert EDITION_NAME in manifest["editions"]
    assert manifest["editions"] == ["normal"]
    assert all(isinstance(e, str) for e in manifest["editions"])


def test_build_manifest_editions_multi_contains_normal():
    manifest = build_manifest(
        training_backend="stub", editions=("normal", "plus", "lite")
    )
    assert "normal" in manifest["editions"]
    assert manifest["editions"] == ["normal", "plus", "lite"]


@pytest.mark.parametrize(
    "bad_editions",
    [
        (),               # 空数组
        ("plus",),        # 不含 normal
        ("lite", "plus"),  # 不含 normal
    ],
)
def test_build_manifest_rejects_invalid_editions(bad_editions):
    # editions 为空或不含 normal 时抛 VaultError（R10-AC2）。
    with pytest.raises(VaultError):
        build_manifest(training_backend="stub", editions=bad_editions)


@pytest.mark.parametrize(
    "backend", ["stub", "peft-cpu", "peft-cuda", "unsloth", "mlx"]
)
def test_build_manifest_training_backend_preserved(backend):
    manifest = build_manifest(training_backend=backend)
    assert manifest["training_backend"] == backend


def test_build_manifest_base_model_default_and_custom():
    assert build_manifest(training_backend="stub")["base_model"] == BASE_MODEL
    custom = build_manifest(training_backend="stub", base_model="custom-base")
    assert custom["base_model"] == "custom-base"


# ===========================================================================
# utc_now_iso8601 —— 格式约束（R10-AC2）
# ===========================================================================


def test_utc_now_iso8601_format():
    value = utc_now_iso8601()
    assert value.endswith("Z")
    # 严格匹配 YYYY-MM-DDTHH:MM:SSZ。
    parsed = datetime.strptime(value, _ISO8601_FMT)
    assert parsed is not None


# ===========================================================================
# write_package —— 落盘后从磁盘读回的字段格式与类型（R10-AC2、AC4、12.9）
# ===========================================================================


def test_write_package_manifest_fields_round_trip(tmp_path: Path):
    root = tmp_path / "ego-output"
    source = _make_lora_source(tmp_path / ".ego" / "lora.safetensors")

    artifacts = write_package(
        root,
        source,
        lora_rank=16,
        lora_alpha=32,
        training_backend="stub",
        version="1.2.3",
        created_at="2025-11-25T08:00:00Z",
    )

    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))

    # version：SemVer 字符串。
    assert isinstance(manifest["version"], str)
    assert manifest["version"] == "1.2.3"
    assert _SEMVER_RE.match(manifest["version"])

    # created_at：ISO 8601 UTC 字符串。
    assert isinstance(manifest["created_at"], str)
    assert manifest["created_at"] == "2025-11-25T08:00:00Z"
    datetime.strptime(manifest["created_at"], _ISO8601_FMT)

    # base_model：固定字符串。
    assert manifest["base_model"] == BASE_MODEL

    # editions：非空字符串数组且含 normal。
    assert isinstance(manifest["editions"], list)
    assert manifest["editions"]
    assert EDITION_NAME in manifest["editions"]

    # training_backend：写入与读回一致（需求 12.9）。
    assert manifest["training_backend"] == "stub"


def test_write_package_config_fields_round_trip(tmp_path: Path):
    root = tmp_path / "ego-output"
    source = _make_lora_source(tmp_path / ".ego" / "lora.safetensors")

    artifacts = write_package(
        root,
        source,
        lora_rank=64,
        lora_alpha=128,
        training_backend="peft-cpu",
    )

    config = json.loads(artifacts.config_path.read_text(encoding="utf-8"))

    assert config["base_model"] == BASE_MODEL
    assert config["quantization"] == "4bit"
    assert isinstance(config["lora_rank"], int)
    assert isinstance(config["lora_alpha"], int)
    assert config["lora_rank"] == 64
    assert config["lora_alpha"] == 128


def test_write_package_default_version_is_package_version(tmp_path: Path):
    root = tmp_path / "ego-output"
    source = _make_lora_source(tmp_path / ".ego" / "lora.safetensors")

    artifacts = write_package(
        root, source, lora_rank=8, lora_alpha=16, training_backend="stub"
    )

    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    assert manifest["version"] == __version__


def test_write_package_training_backend_round_trip_variants(tmp_path: Path):
    # 多个后端名称分别写入并读回，确认 training_backend 透明保存。
    for idx, backend in enumerate(["stub", "unsloth", "mlx", "peft-cuda"]):
        root = tmp_path / f"ego-output-{idx}"
        source = _make_lora_source(tmp_path / f".ego-{idx}" / "lora.safetensors")
        artifacts = write_package(
            root, source, lora_rank=16, lora_alpha=32, training_backend=backend
        )
        manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
        assert manifest["training_backend"] == backend


def test_write_package_json_files_are_utf8_no_bom(tmp_path: Path):
    root = tmp_path / "ego-output"
    source = _make_lora_source(tmp_path / ".ego" / "lora.safetensors")

    artifacts = write_package(
        root, source, lora_rank=16, lora_alpha=32, training_backend="stub"
    )

    for path in (artifacts.manifest_path, artifacts.config_path):
        raw = path.read_bytes()
        # 合法 JSON、UTF-8 编码、不含 BOM（R10-AC2、AC4）。
        assert not raw.startswith(b"\xef\xbb\xbf")
        assert json.loads(raw.decode("utf-8"))


def test_write_package_adapter_is_valid_safetensors(tmp_path: Path):
    root = tmp_path / "ego-output"
    source = _make_lora_source(tmp_path / ".ego" / "lora.safetensors")

    artifacts = write_package(
        root, source, lora_rank=16, lora_alpha=32, training_backend="stub"
    )

    # adapter 落于 normal/adapters/ego-lora.safetensors 且可被 safetensors 读回。
    assert artifacts.adapter_path == (
        root / EDITION_NAME / "adapters" / ADAPTER_FILENAME
    ).resolve()
    tensors = load_file(str(artifacts.adapter_path))
    assert tensors["placeholder"].shape == (1, 1)


def test_write_package_rejects_missing_lora_source(tmp_path: Path):
    root = tmp_path / "ego-output"
    missing = tmp_path / ".ego" / "absent.safetensors"

    # lora_source 不存在 → VaultError，且不产出任何文件。
    with pytest.raises(VaultError):
        write_package(
            root, missing, lora_rank=16, lora_alpha=32, training_backend="stub"
        )
    assert not (root / MANIFEST_FILENAME).exists()
    assert not (root / EDITION_NAME).exists()


def test_write_package_rejects_invalid_editions_before_write(tmp_path: Path):
    root = tmp_path / "ego-output"
    source = _make_lora_source(tmp_path / ".ego" / "lora.safetensors")

    # editions 不含 normal → 在落盘前即抛 VaultError（R10-AC2）。
    with pytest.raises(VaultError):
        write_package(
            root,
            source,
            lora_rank=16,
            lora_alpha=32,
            training_backend="stub",
            editions=("plus",),
        )
    # 源文件未被移动，且未产出产物。
    assert source.exists()
    assert not (root / MANIFEST_FILENAME).exists()
