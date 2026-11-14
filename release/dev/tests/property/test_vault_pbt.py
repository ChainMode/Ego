"""`ego_pod.vault` 的 Ego_Package 产出属性测试。

本文件实现 design.md「Correctness Properties」中的 **Property 12** 与
**Property 13**，对应 tasks.md 任务 9.3 与 9.4：

* **Property 12: vault 产物完整性**（任务 9.3）——
  任意一次成功的 ``write_package`` / ``run_vault`` 完成后，
  ``ego-output/manifest.json``、``ego-output/normal/config.json``、
  ``ego-output/normal/adapters/ego-lora.safetensors`` 三者同时存在；且
  ``manifest.json`` 含 ``version`` / ``created_at`` / ``base_model`` /
  ``editions``（非空且含 ``normal``）/ ``training_backend`` 字段。

  **Validates: Requirements 10.1, 10.2, 10.3, 10.6**

* **Property 13: vault 失败清理性**（任务 9.4）——
  若写入中途遭遇 I/O 错误，``ego-output/.tmp/`` 暂存目录不残留任何文件，
  上一次成功的 ``normal/`` 与 ``manifest.json`` 被原样保留，并以
  :class:`~ego_pod.exit_codes.VaultError` 抛出。

  **Validates: Requirements 10.7**

测试方法：用 :func:`safetensors.numpy.save_file` 写一个 1×1 占位张量作为
Trainer 输出的 ``.ego/lora.safetensors`` 输入夹具；用 hypothesis 在
``lora_rank`` / ``lora_alpha`` / ``training_backend`` / ``version`` 等输入维度上
随机取值。Property 13 通过 :func:`unittest.mock.patch` 把写入链路中的某一步
（config 写入 / adapter 移动 / 原子替换）替换为抛 ``OSError``，以模拟磁盘 I/O
错误（磁盘已满、权限拒绝、目标只读等）。

每个 hypothesis 示例在自己的 :class:`tempfile.TemporaryDirectory` 内运行，
避免函数级 fixture 在多示例间复用导致的状态污染（与 ``test_corpus_jsonl_pbt``
的写盘往返测试一致）。
"""

from __future__ import annotations

import io
import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest import mock

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, settings
from safetensors.numpy import save_file

from ego_pod.exit_codes import VaultError
from ego_pod.vault.package import (
    ADAPTER_FILENAME,
    BASE_MODEL,
    CONFIG_FILENAME,
    EDITION_NAME,
    MANIFEST_FILENAME,
    TMP_DIRNAME,
    PackageArtifacts,
    write_package,
)
from ego_pod.vault.pipeline import run_vault

# ---------------------------------------------------------------------------
# 输入生成策略
# ---------------------------------------------------------------------------

#: lora_rank：正整数（需求 10.4）。
_lora_rank = st.integers(min_value=1, max_value=4096)
#: lora_alpha：正整数（需求 10.4）。
_lora_alpha = st.integers(min_value=1, max_value=8192)
#: training_backend：Trainer 选中的后端名称（需求 12.9 的 stub 标记来源之一）。
_training_backend = st.sampled_from(
    ["stub", "peft-cpu", "peft-cuda", "unsloth", "mlx"]
)
#: 合法 SemVer 2.0.0 核心版本号 MAJOR.MINOR.PATCH（需求 10.2）。
_semver = st.builds(
    lambda a, b, c: f"{a}.{b}.{c}",
    st.integers(min_value=0, max_value=999),
    st.integers(min_value=0, max_value=999),
    st.integers(min_value=0, max_value=999),
)


# ---------------------------------------------------------------------------
# 夹具辅助
# ---------------------------------------------------------------------------


def _make_lora_source(path: Path) -> Path:
    """在 ``path`` 写一个合法的 1×1 ``.safetensors`` 文件作为 Trainer 输出夹具。

    模拟 Stub_Backend 通过 :func:`safetensors.numpy.save_file` 导出的占位 LoRA
    权重（设计文档「Stub_Backend 行为」），供 ``write_package`` 移动到
    ``normal/adapters/ego-lora.safetensors``。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"placeholder": np.zeros((1, 1), dtype=np.float32)}, str(path))
    return path


def _assert_three_files_exist(artifacts: PackageArtifacts, root: Path) -> None:
    """断言 Ego_Package 三个产物文件均存在且路径符合结构约定（R10-AC3、AC6）。"""

    # 三文件均已落盘（R10-AC6）。
    assert artifacts.manifest_path.is_file()
    assert artifacts.config_path.is_file()
    assert artifacts.adapter_path.is_file()

    # 路径符合 Ego_Package 结构（R10-AC2/AC3）。
    assert artifacts.manifest_path == (root / MANIFEST_FILENAME).resolve()
    assert artifacts.config_path == (root / EDITION_NAME / CONFIG_FILENAME).resolve()
    assert (
        artifacts.adapter_path
        == (root / EDITION_NAME / "adapters" / ADAPTER_FILENAME).resolve()
    )


def _assert_manifest_fields(
    manifest_path: Path, *, expected_version: str, expected_backend: str
) -> None:
    """断言 ``manifest.json`` 含全部必需字段且取值合规（R10-AC2、12.9）。"""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # version：合法 SemVer 字符串，且等于传入值。
    assert isinstance(manifest["version"], str)
    assert manifest["version"] == expected_version

    # created_at：UTC ISO 8601 字符串（可被 strptime 解析、以 Z 结尾）。
    assert isinstance(manifest["created_at"], str)
    assert manifest["created_at"].endswith("Z")
    datetime.strptime(manifest["created_at"], "%Y-%m-%dT%H:%M:%SZ")

    # base_model：固定 qwen3.5-0.8B。
    assert manifest["base_model"] == BASE_MODEL

    # editions：非空字符串数组且含 normal（R10-AC2）。
    assert isinstance(manifest["editions"], list)
    assert manifest["editions"]
    assert EDITION_NAME in manifest["editions"]
    assert all(isinstance(e, str) for e in manifest["editions"])

    # training_backend：原样写入与读回（需求 12.9）。
    assert manifest["training_backend"] == expected_backend


# ===========================================================================
# Property 12: vault 产物完整性（任务 9.3）
# ===========================================================================


@given(
    lora_rank=_lora_rank,
    lora_alpha=_lora_alpha,
    training_backend=_training_backend,
    version=_semver,
)
@settings(max_examples=150, deadline=None)
def test_write_package_product_completeness(
    lora_rank: int, lora_alpha: int, training_backend: str, version: str
) -> None:
    """``write_package`` 成功后三文件齐全且 manifest 字段合规。

    **Property 12: vault 产物完整性**

    **Validates: Requirements 10.1, 10.2, 10.3, 10.6**
    """
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        root = base / "ego-output"
        source = _make_lora_source(base / ".ego" / "lora.safetensors")

        artifacts = write_package(
            root,
            source,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            training_backend=training_backend,
            version=version,
        )

        _assert_three_files_exist(artifacts, root)
        _assert_manifest_fields(
            artifacts.manifest_path,
            expected_version=version,
            expected_backend=training_backend,
        )

        # config.json 字段类型与取值（R10-AC4），与传入参数一致。
        config = json.loads(artifacts.config_path.read_text(encoding="utf-8"))
        assert config["base_model"] == BASE_MODEL
        assert config["quantization"] == "4bit"
        assert config["lora_rank"] == lora_rank
        assert config["lora_alpha"] == lora_alpha


@given(
    lora_rank=_lora_rank,
    lora_alpha=_lora_alpha,
    training_backend=_training_backend,
    version=_semver,
)
@settings(max_examples=120, deadline=None)
def test_run_vault_product_completeness(
    lora_rank: int, lora_alpha: int, training_backend: str, version: str
) -> None:
    """``run_vault`` 成功后三文件齐全、manifest 合规，并打印 package 根路径。

    覆盖 R10-AC1（创建/识别 ego-output/ 后写出 Ego_Package）与 R10-AC6
    （校验三文件存在并在标准输出打印根目录绝对路径）。

    **Property 12: vault 产物完整性**

    **Validates: Requirements 10.1, 10.2, 10.3, 10.6**
    """
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        root = base / "ego-output"
        source = _make_lora_source(base / ".ego" / "lora.safetensors")
        buf = io.StringIO()

        artifacts = run_vault(
            source,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            training_backend=training_backend,
            output_root=root,
            version=version,
            stdout=buf,
        )

        _assert_three_files_exist(artifacts, root)
        _assert_manifest_fields(
            artifacts.manifest_path,
            expected_version=version,
            expected_backend=training_backend,
        )

        # R10-AC6：标准输出包含 Ego_Package 根目录绝对路径（以 / 开头的单行）。
        out = buf.getvalue()
        assert str(artifacts.root) in out
        assert str(artifacts.root).startswith("/")
        # Trainer 输出已被移动到产物路径，源文件不再残留。
        assert not source.exists()


# ===========================================================================
# Property 13: vault 失败清理性（任务 9.4）
# ===========================================================================

#: 写入链路中可被注入 ``OSError`` 的步骤 → ``unittest.mock.patch`` 目标。
_FAIL_TARGETS = {
    "config_write": "ego_pod.vault.package._write_json",
    "adapter_move": "ego_pod.vault.package.shutil.move",
    "atomic_replace": "ego_pod.vault.package._atomic_replace_dir",
}
_fail_point = st.sampled_from(sorted(_FAIL_TARGETS))


@given(
    lora_rank=_lora_rank,
    lora_alpha=_lora_alpha,
    training_backend=_training_backend,
    fail_point=_fail_point,
)
@settings(max_examples=120, deadline=None)
def test_write_package_failure_cleans_tmp_no_baseline(
    lora_rank: int, lora_alpha: int, training_backend: str, fail_point: str
) -> None:
    """无既有产物时写入失败：抛 VaultError 且 ``.tmp/`` 不残留任何文件。

    模拟磁盘 I/O 错误（OSError）注入到写入链路的不同步骤，验证半成品被清理。

    **Property 13: vault 失败清理性**

    **Validates: Requirements 10.7**
    """
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        root = base / "ego-output"
        source = _make_lora_source(base / ".ego" / "lora.safetensors")

        with mock.patch(
            _FAIL_TARGETS[fail_point], side_effect=OSError("simulated disk failure")
        ):
            with pytest.raises(VaultError):
                write_package(
                    root,
                    source,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    training_backend=training_backend,
                    version="9.9.9",
                )

        # R10-AC7：.tmp/ 暂存目录被清理，不残留任何半成品。
        tmp_dir = root / TMP_DIRNAME
        assert not tmp_dir.exists()
        # 既无既有产物，则失败后不应产出 normal/ 与 manifest.json。
        assert not (root / EDITION_NAME).exists()
        assert not (root / MANIFEST_FILENAME).exists()


@given(
    lora_rank=_lora_rank,
    lora_alpha=_lora_alpha,
    training_backend=_training_backend,
    fail_point=_fail_point,
)
@settings(max_examples=120, deadline=None)
def test_write_package_failure_preserves_previous_good(
    lora_rank: int, lora_alpha: int, training_backend: str, fail_point: str
) -> None:
    """已有成功产物时写入失败：旧 ``normal/`` 与 ``manifest.json`` 原样保留。

    先成功写一份基线 Ego_Package，再让第二次写入在某一步抛 ``OSError``；断言
    ``.tmp/`` 被清理，且上一次的 ``normal/config.json`` 与 ``manifest.json``
    字节内容不变（R10-AC5「Trainer 成功提供权重前不删除既有产物」与 R10-AC7
    「清理半成品、保留上一次成功产物」）。

    **Property 13: vault 失败清理性**

    **Validates: Requirements 10.7**
    """
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        root = base / "ego-output"

        # --- 基线：一次成功写入，构成"上一次成功的 normal/"-------------------
        good_source = _make_lora_source(base / ".ego" / "good.safetensors")
        write_package(
            root,
            good_source,
            lora_rank=8,
            lora_alpha=16,
            training_backend="stub",
            version="0.0.1",
        )
        normal_dir = root / EDITION_NAME
        manifest_path = root / MANIFEST_FILENAME
        adapter_path = normal_dir / "adapters" / ADAPTER_FILENAME
        assert normal_dir.is_dir()
        old_config = (normal_dir / CONFIG_FILENAME).read_bytes()
        old_manifest = manifest_path.read_bytes()
        old_adapter = adapter_path.read_bytes()

        # --- 第二次写入：注入 I/O 错误 ---------------------------------------
        bad_source = _make_lora_source(base / ".ego" / "bad.safetensors")
        with mock.patch(
            _FAIL_TARGETS[fail_point], side_effect=OSError("simulated disk failure")
        ):
            with pytest.raises(VaultError):
                write_package(
                    root,
                    bad_source,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    training_backend=training_backend,
                    version="9.9.9",
                )

        # R10-AC7：.tmp/ 被清理，不残留半成品。
        assert not (root / TMP_DIRNAME).exists()
        # 上一次成功的产物原样保留（字节级不变）。
        assert normal_dir.is_dir()
        assert (normal_dir / CONFIG_FILENAME).read_bytes() == old_config
        assert manifest_path.read_bytes() == old_manifest
        assert adapter_path.read_bytes() == old_adapter


@given(
    lora_rank=_lora_rank,
    lora_alpha=_lora_alpha,
    training_backend=_training_backend,
    fail_point=_fail_point,
)
@settings(max_examples=80, deadline=None)
def test_run_vault_failure_cleans_tmp(
    lora_rank: int, lora_alpha: int, training_backend: str, fail_point: str
) -> None:
    """``run_vault`` 在写入失败时上抛 VaultError 且 ``.tmp/`` 不残留。

    验证 R10-AC7 在 Vault 编排层（run_vault）同样成立：异常原样上抛、暂存目录
    被清理。

    **Property 13: vault 失败清理性**

    **Validates: Requirements 10.7**
    """
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        root = base / "ego-output"
        source = _make_lora_source(base / ".ego" / "lora.safetensors")
        buf = io.StringIO()

        with mock.patch(
            _FAIL_TARGETS[fail_point], side_effect=OSError("simulated disk failure")
        ):
            with pytest.raises(VaultError):
                run_vault(
                    source,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    training_backend=training_backend,
                    output_root=root,
                    version="9.9.9",
                    stdout=buf,
                )

        # run_vault 已创建/识别 ego-output/（directory-created 事件），但写入失败
        # 后 .tmp/ 不应残留任何半成品（R10-AC7）。
        assert not (root / TMP_DIRNAME).exists()
        assert not (root / EDITION_NAME).exists()
        assert not (root / MANIFEST_FILENAME).exists()
