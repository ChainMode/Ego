"""Flat_Package 退化为单 Kernel Cell 的属性测试。

本文件实现 design.md「Correctness Properties」中的 **Property 7**，对应
tasks.md 任务 5.3：

* **Property 7: Flat_Package 退化为单 Kernel Cell** —— *For any* 不含 ``grid``
  键的 manifest（Flat_Package），``parse_grid_descriptor(manifest.get("grid"))``
  （即 ``parse_grid_descriptor(None)``）返回的内存
  :class:`~ego_pod.runtime.selection.Grid` 恰含单个 Kernel Cell 且无任何分区，
  其 Kernel 相对路径为 ``adapters/ego-lora.safetensors``
  （:data:`~ego_pod.vault.grid_descriptor.FLAT_KERNEL_RELPATH`）；在该网格上
  ``select_cells(grid, None)`` 选出且**仅**选出该 Kernel Cell。

  **Feature: ego-layered-adapter, Property 7**

  **Validates: Requirements 6.1, 6.2**

测试在**无任何机器学习依赖**的环境下运行：``parse_grid_descriptor`` 与
``select_cells`` 均为纯 Python 纯函数（设计「决策 D」/ R6-AC8 / R7-AC2），故
hypothesis 直接以合成的 Flat_Package manifest 字典行使该退化语义，每个属性测试
最少运行 **100 次迭代**。

策略：合成**只含五个既有字段**（``version`` / ``created_at`` / ``base_model`` /
``editions`` / ``training_backend``，取值随机变化）、**不含 ``grid`` 键**的
manifest 字典（即 Flat_Package 的形状，见
:func:`~ego_pod.vault.package.build_manifest` 在 ``grid=None`` 时的输出）。由于
``parse_grid_descriptor(None)`` 是**确定性**的，"for any manifest without grid"
通过断言所生成 manifest 的 ``manifest.get("grid") is None``、且其解析结果恒为该
固定的扁平网格来行使。``training_backend`` 取自固定集合
``{unsloth, peft-cuda, mlx, peft-cpu, stub}``，并以非平凡的 ``editions`` 列表与
各后端取值的显式 ``@example`` 覆盖代表性场景。
"""

from __future__ import annotations

from datetime import datetime, timezone

import hypothesis.strategies as st
from hypothesis import example, given, settings

from ego_pod.runtime.selection import KERNEL, Grid, select_cells
from ego_pod.trainer.base import BackendIdentifier
from ego_pod.vault.grid_descriptor import FLAT_KERNEL_RELPATH, parse_grid_descriptor
from ego_pod.vault.package import EDITION_NAME

# ---------------------------------------------------------------------------
# 输入生成策略
# ---------------------------------------------------------------------------

#: Backend_Identifier 的固定取值集合（R6-AC5），写入 manifest 的 training_backend。
_FIXED_IDENTIFIERS: tuple[BackendIdentifier, ...] = (
    "unsloth",
    "peft-cuda",
    "mlx",
    "peft-cpu",
    "stub",
)

#: 合法 SemVer 2.0.0 核心版本号 MAJOR.MINOR.PATCH（version 字段，取值随机变化）。
_semver = st.builds(
    lambda a, b, c: f"{a}.{b}.{c}",
    st.integers(min_value=0, max_value=999),
    st.integers(min_value=0, max_value=999),
    st.integers(min_value=0, max_value=999),
)

#: created_at：UTC ISO 8601 字符串（取值随机变化）。
_created_at = st.datetimes(
    min_value=datetime(2020, 1, 1),
    max_value=datetime(2030, 12, 31),
).map(lambda dt: dt.replace(tzinfo=timezone.utc).isoformat())

#: base_model：基座模型标识（含默认值与其它合成取值，行使"取值随机变化"）。
_base_model = st.sampled_from(["qwen3.5-0.8B", "qwen3.5-1.5B", "some-base-model"])

#: editions：非空且必含 "normal" 的列表；可追加合成的额外 Edition 名（非平凡）。
_extra_edition = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789-", min_size=1, max_size=8
).filter(lambda s: s != EDITION_NAME)
_editions = st.lists(_extra_edition, min_size=0, max_size=3, unique=True).map(
    lambda extras: [EDITION_NAME, *extras]
)


@st.composite
def flat_manifests(draw) -> dict:
    """合成一个 Flat_Package manifest 字典：**仅含五个既有字段、无 ``grid`` 键**。

    与 :func:`~ego_pod.vault.package.build_manifest` 在 ``grid=None`` 时的输出
    形状一致（R4-AC4 / R6-AC1）：``version`` / ``created_at`` / ``base_model`` /
    ``editions`` / ``training_backend`` 五字段，取值随机变化，**不**含 ``grid``。
    """

    return {
        "version": draw(_semver),
        "created_at": draw(_created_at),
        "base_model": draw(_base_model),
        "editions": draw(_editions),
        "training_backend": draw(st.sampled_from(_FIXED_IDENTIFIERS)),
    }


# 一个非平凡的显式样例：多 Edition 列表 + 代表性后端取值（覆盖任务要求）。
_NON_TRIVIAL_EXAMPLE = {
    "version": "1.2.3",
    "created_at": "2025-11-25T08:00:00+00:00",
    "base_model": "qwen3.5-0.8B",
    "editions": ["normal", "lite", "plus"],
    "training_backend": "peft-cpu",
}


# ===========================================================================
# Property 7: Flat_Package 退化为单 Kernel Cell（任务 5.3）
# ===========================================================================


@given(manifest=flat_manifests())
@settings(max_examples=200, deadline=None)
@example(manifest=dict(_NON_TRIVIAL_EXAMPLE))
@example(manifest={**_NON_TRIVIAL_EXAMPLE, "training_backend": "unsloth"})
@example(manifest={**_NON_TRIVIAL_EXAMPLE, "training_backend": "peft-cuda"})
@example(manifest={**_NON_TRIVIAL_EXAMPLE, "training_backend": "mlx"})
@example(manifest={**_NON_TRIVIAL_EXAMPLE, "training_backend": "stub"})
def test_flat_manifest_parses_to_single_kernel_cell(manifest: dict) -> None:
    """无 ``grid`` 键的 manifest ⇒ ``parse_grid_descriptor`` 还原单 Kernel 扁平网格。

    断言：

    * 所生成 manifest 不含 ``grid`` 键（``manifest.get("grid") is None``），即它
      确为 Flat_Package 的形状（R6-AC1）；
    * ``parse_grid_descriptor(manifest.get("grid"))`` 返回的 :class:`Grid` 恰含
      单个 Kernel Cell：``kernel.cell == KERNEL`` 且 ``kernel.relpath`` 为
      ``adapters/ego-lora.safetensors``（:data:`FLAT_KERNEL_RELPATH`，R6-AC1）；
    * 无任何分区（``zones == {}``）——退化为仅含 Kernel 的单 Cell 网格。

    **Feature: ego-layered-adapter, Property 7**

    **Validates: Requirements 6.1, 6.2**
    """
    # Flat_Package 的判据：manifest 无 grid 键，故 manifest.get("grid") 为 None。
    assert "grid" not in manifest
    assert manifest.get("grid") is None

    grid = parse_grid_descriptor(manifest.get("grid"))

    assert isinstance(grid, Grid)
    # 恰含单个 Kernel Cell（与分区无关的 Level-0 Cell）。
    assert grid.kernel.cell == KERNEL
    assert grid.kernel.cell.is_kernel
    # Kernel 落于扁平包路径 adapters/ego-lora.safetensors（R6-AC1）。
    assert grid.kernel.relpath == FLAT_KERNEL_RELPATH
    assert grid.kernel.relpath == "adapters/ego-lora.safetensors"
    # 无任何分区——退化为单 Cell 网格。
    assert grid.zones == {}


@given(manifest=flat_manifests())
@settings(max_examples=200, deadline=None)
@example(manifest=dict(_NON_TRIVIAL_EXAMPLE))
@example(manifest={**_NON_TRIVIAL_EXAMPLE, "training_backend": "unsloth"})
@example(manifest={**_NON_TRIVIAL_EXAMPLE, "training_backend": "peft-cuda"})
@example(manifest={**_NON_TRIVIAL_EXAMPLE, "training_backend": "mlx"})
@example(manifest={**_NON_TRIVIAL_EXAMPLE, "training_backend": "stub"})
def test_flat_grid_select_cells_returns_only_kernel(manifest: dict) -> None:
    """扁平网格上 ``select_cells(grid, None)`` 选出且仅选出该 Kernel Cell。

    断言在退化扁平网格上的 Full_In_Load：

    * 结果恰为 ``[KERNEL]``——单个 Kernel Cell，无任何分区 Cell（R6-AC1/AC2）；
    * Kernel 排在首位且仅出现一次（kernel-always，R5-AC3 派生）。

    这正是 R6-AC1/AC2 要求的"把无 ``grid`` 的包视为单个 Kernel Cell、其加载行为
    等同于今天的单适配器包"。

    **Feature: ego-layered-adapter, Property 7**

    **Validates: Requirements 6.1, 6.2**
    """
    grid = parse_grid_descriptor(manifest.get("grid"))

    selected = select_cells(grid, None)

    # 选出且仅选出单个 Kernel Cell（无分区 Cell）。
    assert selected == [KERNEL]
    assert len(selected) == 1
    assert selected[0] == KERNEL
    assert selected[0].is_kernel
