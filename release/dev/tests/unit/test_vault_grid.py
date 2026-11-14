"""Vault 二维网格产物的字段、布局与事务性单元测试（任务 4.6）。

Example-based unit tests for the Vault Grid_Package artifact contract.

本文件以**明示样例 + 边界用例**验证 Vault 模块二维稀疏网格（Sparse_Grid）产物
契约的可观测行为，作为属性测试（Property 2 往返一致、Property 6 manifest 五字段
兼容）的互补面。覆盖设计文档 §"Components and Interfaces → 4. Vault" 与需求：

* **R4-AC6（空分区 ⇒ 空 level 数组）**：Grid_Descriptor 把含 0 个 Cell 的分区
  表达为空数组 ``[]``，:func:`parse_grid_descriptor` 据此还原为空 Cell 列表
  （``max_level == 0``），而非省略分区或赋予虚构 Level；MVP 不产生空分区，但
  schema 与解析均支持。
* **R4-AC7（kernel-only ⇒ 纯 Flat 产物逐字节一致）**：仅含单个 Kernel Cell 的
  网格经 :func:`run_vault` 退化为纯 Flat_Package——不写 ``grid`` 键、Kernel 适配器
  落 ``adapters/ego-lora.safetensors``，其 ``manifest.json`` 与既有
  :func:`write_package`（MVP / real-trainer）路径**逐字节一致**。
* **R4-AC8（成功写入后 manifest + config + 全部 Cell 文件同时存在）**：网格写入
  成功后，``manifest.json``、``normal/config.json`` 与 ``grid`` 所列每个 Cell 的
  适配器文件（Kernel 与各分区/层级）同时存在于 Ego_Package。
* **R4-AC9（事务性中途失败不残留、保留上次成功产物）**：写入中途遭遇 I/O 错误时
  不残留 ``.tmp/`` 半成品目录，且完整保留上一次成功的产物。
* **R4-AC10（I/O 错误 ⇒ VaultError 含路径）**：网格写出遭遇磁盘 I/O 错误时抛出
  :class:`VaultError`，消息含失败路径与原因。

另外覆盖 Vault 网格写入的 **kernel-always 硬约束**（设计「决策 B」）：缺失 Kernel
Cell 时在写任何文件之前以 :class:`VaultError` 快速失败。

测试使用 ``tmp_path`` 写入临时文件，并以 :func:`safetensors.numpy.save_file`
构造 1×1 占位张量作为各 Cell 适配器夹具，不依赖网络或真实 ML 依赖。

**Feature: ego-layered-adapter**
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

# 导入顺序：先 ``runtime.selection`` 再 ``vault.grid_descriptor``，以规避
# ``runtime`` ↔ ``vault`` 之间的循环导入（``grid_descriptor`` 顶层依赖
# ``runtime.selection``）。
from ego_pod.exit_codes import VaultError
from ego_pod.runtime.selection import KERNEL, CellKey
from ego_pod.trainer.cells import TrainedCell
import ego_pod.vault.package as pkg
from ego_pod.vault.grid_descriptor import (
    KERNEL_RELPATH,
    build_grid_descriptor,
    parse_grid_descriptor,
)
from ego_pod.vault.package import (
    ADAPTER_FILENAME,
    CONFIG_FILENAME,
    EDITION_NAME,
    MANIFEST_FILENAME,
    TMP_DIRNAME,
    GridPackageArtifacts,
    PackageArtifacts,
    write_grid_package,
    write_package,
)
from ego_pod.vault.pipeline import run_vault

# 固定时钟/版本以消除非确定性，便于逐字节比较与产物断言。
_FIXED_VERSION = "1.2.3"
_FIXED_CREATED_AT = "2025-11-25T08:00:00Z"


# ---------------------------------------------------------------------------
# 夹具辅助
# ---------------------------------------------------------------------------


def _make_adapter(path: Path, value: float = 0.0) -> Path:
    """写一个合法的 1×1 ``.safetensors`` 文件作为某 Cell 的适配器夹具。

    ``value`` 用于在需要时区分不同 Cell 的张量内容；默认 0.0。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"placeholder": np.full((1, 1), value, dtype=np.float32)}, str(path))
    return path


def _kernel_cell(src_dir: Path, value: float = 0.0) -> TrainedCell:
    """构造一个 Kernel 已训练 Cell（``zone=None, level=0``）及其适配器源文件。"""

    path = _make_adapter(src_dir / "kernel.safetensors", value)
    return TrainedCell(zone=None, level=0, output_path=path)


def _zone_cell(src_dir: Path, zone: str, level: int, value: float = 0.0) -> TrainedCell:
    """构造一个分区已训练 Cell（``zone, level>=1``）及其适配器源文件。"""

    path = _make_adapter(src_dir / f"{zone}-L{level}.safetensors", value)
    return TrainedCell(zone=zone, level=level, output_path=path)


# ===========================================================================
# R4-AC6：空分区 ⇒ 空 level 数组（schema 与解析支持）
# ===========================================================================


def test_parse_grid_descriptor_empty_zone_is_empty_level_list():
    """含 0 个 Cell 的分区在描述中为空数组 ⇒ 解析回空 Cell 列表（R4-AC6）。

    手工构造一个含空分区的 ``grid`` 描述，断言 :func:`parse_grid_descriptor`
    把该分区还原为**空 Cell 列表**且 ``max_level == 0``，而非省略分区或赋予虚构
    Level；非空分区则按 Level 升序还原。
    """

    grid = {
        EDITION_NAME: {
            "kernel": {"path": KERNEL_RELPATH},
            "zones": {
                "coding": [
                    {"level": 1, "path": "adapters/coding/L1.safetensors"},
                    {"level": 2, "path": "adapters/coding/L2.safetensors"},
                ],
                "empty_zone": [],
            },
        }
    }

    parsed = parse_grid_descriptor(grid)

    # 空分区被保留且为空 Cell 列表（不省略、不虚构 Level）。
    assert "empty_zone" in parsed.zones
    assert parsed.zones["empty_zone"] == []
    assert parsed.max_level("empty_zone") == 0

    # 非空分区按 Level 升序还原，max_level 正确。
    assert [gc.cell.level for gc in parsed.zones["coding"]] == [1, 2]
    assert parsed.max_level("coding") == 2


def test_build_grid_descriptor_omits_zones_without_cells_and_sorts_levels():
    """``build_grid_descriptor`` 不为无 Cell 的分区虚构条目，且各分区 Level 升序（R4-AC5/AC6）。

    已训练 Cell 列表无法表达"空分区"，故构造产物中只出现含至少一个 Cell 的分区，
    且每个分区的 Level 列表按升序、路径符合 ``adapters/<zone>/L<level>.safetensors``。
    """

    # build 仅读 (zone, level)，不读 output_path，故源文件无需真实存在。
    cells = [
        TrainedCell(zone=None, level=0, output_path=Path("/tmp/k.safetensors")),
        TrainedCell(zone="coding", level=2, output_path=Path("/tmp/c2.safetensors")),
        TrainedCell(zone="coding", level=1, output_path=Path("/tmp/c1.safetensors")),
    ]

    grid = build_grid_descriptor(cells)
    zones = grid[EDITION_NAME]["zones"]

    # Kernel 路径固定；只出现含 Cell 的分区（coding），无虚构的空分区键。
    assert grid[EDITION_NAME]["kernel"] == {"path": KERNEL_RELPATH}
    assert set(zones) == {"coding"}

    # coding 分区按 Level 升序、连续 1..2，路径符合约定。
    assert zones["coding"] == [
        {"level": 1, "path": "adapters/coding/L1.safetensors"},
        {"level": 2, "path": "adapters/coding/L2.safetensors"},
    ]


# ===========================================================================
# R4-AC7：kernel-only ⇒ 纯 Flat 产物逐字节一致
# ===========================================================================


def test_run_vault_kernel_only_is_byte_identical_flat_package(tmp_path: Path):
    """仅含单个 Kernel Cell 的网格经 ``run_vault`` 退化为逐字节一致的 Flat_Package（R4-AC7）。

    分别以既有扁平路径 :func:`write_package` 与网格入口
    ``run_vault(cells=[kernel])`` 在相同参数下写出 Ego_Package，断言二者的
    ``manifest.json`` **逐字节一致**、均为五字段且不含 ``grid`` 键，且 Kernel 适配器
    均落于 ``adapters/ego-lora.safetensors``、不存在 ``kernel.safetensors``。
    """

    common = dict(
        lora_rank=16,
        lora_alpha=32,
        training_backend="stub",
        version=_FIXED_VERSION,
        created_at=_FIXED_CREATED_AT,
    )

    # 参照：既有 MVP / real-trainer 扁平写入路径。
    ref_root = tmp_path / "ref-output"
    ref_source = _make_adapter(tmp_path / "ref-src" / "lora.safetensors")
    ref_artifacts = write_package(ref_root, ref_source, **common)
    ref_manifest_bytes = ref_artifacts.manifest_path.read_bytes()

    # 被测：仅含单个 Kernel Cell 的网格经 run_vault 退化。
    grid_root = tmp_path / "grid-output"
    kernel = _kernel_cell(tmp_path / "grid-src")
    sink = io.StringIO()
    artifacts = run_vault(
        cells=[kernel],
        output_root=grid_root,
        cwd=tmp_path,
        stdout=sink,
        **common,
    )

    # 退化路径返回扁平的 PackageArtifacts（而非 GridPackageArtifacts）。
    assert isinstance(artifacts, PackageArtifacts)

    # manifest.json 逐字节一致（五字段、无 grid 键）。
    grid_manifest_bytes = artifacts.manifest_path.read_bytes()
    assert grid_manifest_bytes == ref_manifest_bytes

    manifest = json.loads(grid_manifest_bytes.decode("utf-8"))
    assert "grid" not in manifest
    assert list(manifest.keys()) == [
        "version",
        "created_at",
        "base_model",
        "editions",
        "training_backend",
    ]

    # 磁盘布局与 Flat_Package 完全一致：ego-lora.safetensors 存在、kernel.safetensors 不存在。
    adapters_dir = grid_root / EDITION_NAME / "adapters"
    assert (adapters_dir / ADAPTER_FILENAME).is_file()
    assert not (adapters_dir / "kernel.safetensors").exists()
    assert artifacts.adapter_path == (adapters_dir / ADAPTER_FILENAME).resolve()


# ===========================================================================
# R4-AC8：成功写入后 manifest + config + 全部 Cell 文件同时存在
# ===========================================================================


def test_run_vault_grid_writes_manifest_config_and_all_cell_files(tmp_path: Path):
    """含分区的网格成功写入后 manifest + config + 全部 Cell 适配器同时存在（R4-AC8）。

    以 Kernel + coding/L1 + knowledge/L1 三个 Cell 走网格路径，断言：返回
    :class:`GridPackageArtifacts`、``manifest.json`` 含正确的 ``grid`` 结构、
    ``normal/config.json`` 与 grid 所列三个 Cell 适配器文件全部落盘存在。
    """

    src = tmp_path / "src"
    cells = [
        _kernel_cell(src, value=0.0),
        _zone_cell(src, "coding", 1, value=1.0),
        _zone_cell(src, "knowledge", 1, value=2.0),
    ]

    root = tmp_path / "ego-output"
    sink = io.StringIO()
    artifacts = run_vault(
        cells=cells,
        output_root=root,
        cwd=tmp_path,
        lora_rank=8,
        lora_alpha=16,
        training_backend="stub",
        version=_FIXED_VERSION,
        created_at=_FIXED_CREATED_AT,
        stdout=sink,
    )

    # 网格路径返回 GridPackageArtifacts。
    assert isinstance(artifacts, GridPackageArtifacts)

    # manifest + config 存在。
    assert artifacts.manifest_path.is_file()
    assert artifacts.config_path.is_file()
    assert artifacts.config_path == (root / EDITION_NAME / CONFIG_FILENAME).resolve()

    # grid 所列全部 Cell 适配器文件存在，且键覆盖三个 Cell。
    assert set(artifacts.cell_paths) == {
        KERNEL,
        CellKey(zone="coding", level=1),
        CellKey(zone="knowledge", level=1),
    }
    for cell_path in artifacts.cell_paths.values():
        assert cell_path.is_file(), f"缺少 Cell 适配器文件：{cell_path}"

    # 磁盘布局符合约定：kernel.safetensors + <zone>/L1.safetensors。
    adapters_dir = root / EDITION_NAME / "adapters"
    assert (adapters_dir / "kernel.safetensors").is_file()
    assert (adapters_dir / "coding" / "L1.safetensors").is_file()
    assert (adapters_dir / "knowledge" / "L1.safetensors").is_file()

    # manifest 含正确的 grid 结构（含分区 ⇒ 写 grid 键）。
    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    assert "grid" in manifest
    normal_grid = manifest["grid"][EDITION_NAME]
    assert normal_grid["kernel"]["path"] == KERNEL_RELPATH
    assert {"level": 1, "path": "adapters/coding/L1.safetensors"} in normal_grid[
        "zones"
    ]["coding"]
    assert {"level": 1, "path": "adapters/knowledge/L1.safetensors"} in normal_grid[
        "zones"
    ]["knowledge"]


def test_write_grid_package_returns_all_cell_paths(tmp_path: Path):
    """直接调用 ``write_grid_package`` 也使全部 Cell 文件与 config/manifest 同时存在（R4-AC8）。"""

    src = tmp_path / "src"
    cells = [
        _kernel_cell(src),
        _zone_cell(src, "coding", 1),
        _zone_cell(src, "coding", 2),
    ]

    root = tmp_path / "ego-output"
    artifacts = write_grid_package(
        root,
        cells=cells,
        lora_rank=16,
        lora_alpha=32,
        training_backend="stub",
        version=_FIXED_VERSION,
        created_at=_FIXED_CREATED_AT,
    )

    assert artifacts.manifest_path.is_file()
    assert artifacts.config_path.is_file()
    assert set(artifacts.cell_paths) == {
        KERNEL,
        CellKey(zone="coding", level=1),
        CellKey(zone="coding", level=2),
    }
    for path in artifacts.cell_paths.values():
        assert path.is_file()


# ===========================================================================
# kernel-always 硬约束（决策 B）：缺失 Kernel ⇒ 写任何文件前快速失败
# ===========================================================================


def test_write_grid_package_fast_fails_when_kernel_missing(tmp_path: Path):
    """缺失 Kernel Cell ⇒ 构建网格前以 VaultError 快速失败，不产任何文件（决策 B）。"""

    src = tmp_path / "src"
    # 仅含一个分区 Cell、无 Kernel。
    cells = [_zone_cell(src, "coding", 1)]

    root = tmp_path / "ego-output"
    with pytest.raises(VaultError, match="Kernel"):
        write_grid_package(
            root,
            cells=cells,
            lora_rank=16,
            lora_alpha=32,
            training_backend="stub",
        )

    # 未产出任何产物。
    assert not (root / MANIFEST_FILENAME).exists()
    assert not (root / EDITION_NAME).exists()


# ===========================================================================
# R4-AC9：事务性中途失败不残留、保留上次成功产物
# ===========================================================================


def test_write_grid_package_mid_failure_preserves_last_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """写入中途 I/O 失败 ⇒ 不残留 .tmp/、保留上一次成功产物（R4-AC9）。

    先成功写出一份 Grid_Package（``version=1.0.0``）作为"上次成功产物"，再以
    monkeypatch 让第二次写入在写 ``manifest.json`` 时（已写 config、已移动各 Cell
    到 ``.tmp/`` 之后、原子替换之前）抛 ``OSError``。断言：第二次写入以
    :class:`VaultError` 失败、``.tmp/`` 半成品目录被清理、且 ``normal/`` 与
    ``manifest.json`` 仍为第一次成功写入的内容（``version=1.0.0``）。
    """

    root = tmp_path / "ego-output"

    # --- 第一次：成功写出 version=1.0.0 的 Grid_Package ----------------------
    cells_v1 = [_kernel_cell(tmp_path / "src1"), _zone_cell(tmp_path / "src1", "coding", 1)]
    write_grid_package(
        root,
        cells=cells_v1,
        lora_rank=16,
        lora_alpha=32,
        training_backend="stub",
        version="1.0.0",
        created_at=_FIXED_CREATED_AT,
    )

    manifest_path = root / MANIFEST_FILENAME
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["version"] == "1.0.0"
    kernel_file = root / EDITION_NAME / "adapters" / "kernel.safetensors"
    coding_file = root / EDITION_NAME / "adapters" / "coding" / "L1.safetensors"
    assert kernel_file.is_file()
    assert coding_file.is_file()

    # --- 第二次：注入 manifest 写入失败（中途 I/O 错误）----------------------
    real_write_json = pkg._write_json

    def _failing_write_json(path: Path, payload: dict) -> None:
        # 仅在写 manifest.json 时失败：此时 config 已写、各 Cell 已移入 .tmp/，
        # 但原子替换尚未发生 ⇒ 旧产物完好。
        if path.name == MANIFEST_FILENAME:
            raise OSError(28, "No space left on device", str(path))
        return real_write_json(path, payload)

    monkeypatch.setattr(pkg, "_write_json", _failing_write_json)

    cells_v2 = [_kernel_cell(tmp_path / "src2"), _zone_cell(tmp_path / "src2", "coding", 1)]
    with pytest.raises(VaultError):
        write_grid_package(
            root,
            cells=cells_v2,
            lora_rank=16,
            lora_alpha=32,
            training_backend="stub",
            version="2.0.0",
            created_at=_FIXED_CREATED_AT,
        )

    # .tmp/ 半成品目录被清理，不残留。
    assert not (root / TMP_DIRNAME).exists()

    # 上一次成功产物完整保留（manifest 仍为 1.0.0，旧 Cell 文件仍在）。
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["version"] == "1.0.0"
    assert kernel_file.is_file()
    assert coding_file.is_file()


# ===========================================================================
# R4-AC10：I/O 错误 ⇒ VaultError 含路径
# ===========================================================================


def test_write_grid_package_io_error_raises_vault_error_with_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """网格写出遭遇 I/O 错误 ⇒ VaultError 含失败路径与原因（R4-AC10）。"""

    root = tmp_path / "ego-output"
    cells = [_kernel_cell(tmp_path / "src"), _zone_cell(tmp_path / "src", "coding", 1)]

    real_write_json = pkg._write_json
    captured: dict[str, Path] = {}

    def _failing_write_json(path: Path, payload: dict) -> None:
        if path.name == MANIFEST_FILENAME:
            captured["path"] = path
            raise OSError(28, "No space left on device", str(path))
        return real_write_json(path, payload)

    monkeypatch.setattr(pkg, "_write_json", _failing_write_json)

    with pytest.raises(VaultError) as excinfo:
        write_grid_package(
            root,
            cells=cells,
            lora_rank=16,
            lora_alpha=32,
            training_backend="stub",
            version=_FIXED_VERSION,
            created_at=_FIXED_CREATED_AT,
        )

    message = str(excinfo.value)
    # 消息含失败文件的（解析后）绝对路径。
    assert str(captured["path"].resolve()) in message
    # 消息含具体原因（OSError 文本）。
    assert "No space left on device" in message


def test_write_grid_package_rejects_missing_cell_source(tmp_path: Path):
    """某 Cell 适配器源文件不存在 ⇒ 在替换旧产物前以 VaultError 含路径失败（R4-AC10）。"""

    src = tmp_path / "src"
    kernel = _kernel_cell(src)
    # 分区 Cell 的源文件不存在（未创建）。
    missing = TrainedCell(
        zone="coding", level=1, output_path=src / "missing-coding-L1.safetensors"
    )

    root = tmp_path / "ego-output"
    with pytest.raises(VaultError) as excinfo:
        write_grid_package(
            root,
            cells=[kernel, missing],
            lora_rank=16,
            lora_alpha=32,
            training_backend="stub",
        )

    # 消息含缺失源文件的解析路径，且未产出任何产物。
    assert str(missing.output_path.resolve()) in str(excinfo.value)
    assert not (root / MANIFEST_FILENAME).exists()
    assert not (root / EDITION_NAME).exists()
