"""Backend_Identifier 经 ``manifest.json`` 往返一致的属性测试。

本文件实现 design.md「Correctness Properties」中的 **Property 14**，对应
tasks.md 任务 9.4：

* **Property 14: Backend_Identifier 经 manifest 往返一致** —— 对任意取自固定
  集合 ``{unsloth, peft-cuda, mlx, peft-cpu, stub}`` 的 Backend_Identifier，
  ``build_manifest(training_backend=id)`` 写出后再读回 ``manifest.json`` 的
  ``training_backend`` 字段，得到的取值与写入值相等。

  **Validates: Requirements 7.3**

测试覆盖两个角度：

(a) **纯构造往返** —— ``build_manifest(training_backend=id)`` 返回的字典中
    ``training_backend`` 字段直接等于 ``id``（不经落盘的最小往返单元）。

(b) **完整写包往返** —— 用 :func:`safetensors.numpy.save_file` 写一个 1×1
    占位张量作为 Trainer 输出的 ``.ego/lora.safetensors`` 输入夹具，调用
    :func:`~ego_pod.vault.package.write_package`，再从落盘的
    ``ego-output/manifest.json`` 读回并断言 ``training_backend == id``。这覆盖
    Vault「原样写入入参 ``training_backend``」的端到端契约（R7-AC3）。

hypothesis 在固定 Backend_Identifier 集合上随机取值（``lora_rank`` /
``lora_alpha`` 取正整数，``version`` 取合法 SemVer）。每个 hypothesis 示例在
自己的 :class:`tempfile.TemporaryDirectory` 内运行，避免函数级 fixture 在多示例
间复用导致的状态污染（与 ``test_vault_pbt`` 的写盘往返测试一致）。

**Feature: ego-pod-real-trainer**
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import hypothesis.strategies as st
import numpy as np
from hypothesis import given, settings
from safetensors.numpy import save_file

from ego_pod.trainer.base import BackendIdentifier
from ego_pod.vault.package import (
    MANIFEST_FILENAME,
    build_manifest,
    write_package,
)

# ---------------------------------------------------------------------------
# 输入生成策略
# ---------------------------------------------------------------------------

#: Backend_Identifier 的固定取值集合（R7-AC1、R7-AC4）；Property 14 的论域。
_FIXED_IDENTIFIERS: tuple[BackendIdentifier, ...] = (
    "unsloth",
    "peft-cuda",
    "mlx",
    "peft-cpu",
    "stub",
)

#: 取自固定集合的 Backend_Identifier。
_identifier = st.sampled_from(_FIXED_IDENTIFIERS)
#: lora_rank：正整数（写入 config.json，与本属性无关但需合法）。
_lora_rank = st.integers(min_value=1, max_value=4096)
#: lora_alpha：正整数。
_lora_alpha = st.integers(min_value=1, max_value=8192)
#: 合法 SemVer 2.0.0 核心版本号 MAJOR.MINOR.PATCH。
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

    模拟后端导出的占位 LoRA 权重，供 ``write_package`` 移动到
    ``normal/adapters/ego-lora.safetensors``。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"placeholder": np.zeros((1, 1), dtype=np.float32)}, str(path))
    return path


# ===========================================================================
# Property 14: Backend_Identifier 经 manifest 往返一致（任务 9.4）
# ===========================================================================


@given(identifier=_identifier)
@settings(max_examples=200, deadline=None)
def test_build_manifest_preserves_training_backend(
    identifier: BackendIdentifier,
) -> None:
    """角度 (a)：``build_manifest`` 原样保留传入的 Backend_Identifier。

    构造往返的最小单元：``build_manifest(training_backend=id)`` 返回字典的
    ``training_backend`` 字段直接等于 ``id``。

    **Feature: ego-pod-real-trainer, Property 14**

    **Validates: Requirements 7.3**
    """
    manifest = build_manifest(training_backend=identifier)

    assert manifest["training_backend"] == identifier


@given(
    identifier=_identifier,
    lora_rank=_lora_rank,
    lora_alpha=_lora_alpha,
    version=_semver,
)
@settings(max_examples=200, deadline=None)
def test_write_package_manifest_roundtrip_training_backend(
    identifier: BackendIdentifier,
    lora_rank: int,
    lora_alpha: int,
    version: str,
) -> None:
    """角度 (b)：完整写包后从落盘 manifest.json 读回的标识与写入值相等。

    用 1×1 ``.safetensors`` 夹具走 ``write_package`` 全流程，再从
    ``ego-output/manifest.json`` 读回 ``training_backend`` 字段，断言它等于
    写入的 Backend_Identifier（Vault 原样写入入参的端到端往返，R7-AC3）。

    **Feature: ego-pod-real-trainer, Property 14**

    **Validates: Requirements 7.3**
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
            training_backend=identifier,
            version=version,
        )

        # 从落盘的 manifest.json 读回 training_backend 字段。
        manifest_path = root / MANIFEST_FILENAME
        assert artifacts.manifest_path == manifest_path.resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        # 往返一致：读回值等于写入的 Backend_Identifier（R7-AC3）。
        assert manifest["training_backend"] == identifier
