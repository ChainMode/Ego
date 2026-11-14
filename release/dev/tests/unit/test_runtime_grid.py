"""Runtime 网格加载的单元测试（任务 5.6）。

Unit tests for Runtime grid loading (Ego Layered Adapter, task 5.6).

本文件覆盖 ``runtime/inference.py`` 的网格加载路径中、属性测试（Property 3/4/7）
之外的四组固定样例验收标准，且**完全无需安装任何机器学习依赖**（``torch`` /
``transformers`` / ``peft`` / ``unsloth`` / ``mlx`` 均不要求存在）：

* **缺失 / 损坏 Cell ⇒ ``RuntimeError_`` 含标识与相对路径（R5-AC7）**：构造一个
  含 ``grid`` 键的 Grid_Package，但令某个 Cell 的适配器文件缺失或损坏，断言
  :func:`~ego_pod.runtime.inference.load_inference_session` 抛出
  :class:`~ego_pod.exit_codes.RuntimeError_`，其消息同时包含该 Cell 的可读标识
  （``Kernel`` 或 ``(<zone>, L<level>)``）与文件相对路径。该校验发生在 Stub /
  真实后端分流**之前**，故这组测试用 ``training_backend == "stub"`` 即可触发，
  无需任何后端。

* **``close()`` 释放全部 Cell（R5-AC9）**：用注入的假 PEFT 后端
  （:class:`_doubles.FakePeftBackend`）端到端驱动一个**多 Cell**的
  :class:`~ego_pod.runtime.inference.BackendInferenceSession`，断言构造后活动适配器
  集合含全部 Cell（Kernel + 各分区 Cell），调用 ``close()`` 后
  ``model`` / ``tokenizer`` / ``active_adapters`` 均被释放/清空。

* **Stub 路径无论一个还是多个 Cell 行为不变（R5-AC10）**：分别加载一个**单 Cell**
  的纯 Flat_Package（无 ``grid`` 键）与一个**多 Cell**的 Grid_Package（含 ``grid``
  键、多分区多层级），二者 ``training_backend`` 均为 ``stub``。断言两者都返回
  :class:`~ego_pod.runtime.inference.StubInferenceSession`、打印同一条启动提示、对
  任意输入返回同一条固定占位回复、``close()`` 均为空操作——即 Stub 行为与 Cell
  数目无关。

* **未指定 selection ⇒ Full_In_Load（R5-AC1）**：加载一个多分区多层级的
  Grid_Package 且**不**传 ``selection``，断言会话加载的有序 Cell 列表恰为
  Full_In_Load——Kernel 在首位，再每个分区按 Level ``1..max`` 升序（与
  :func:`~ego_pod.runtime.selection.select_cells` 的 ``selection is None`` 语义一致）。

测试在无 ML 依赖的环境下运行：构造真实后端会话的用例由注入的假后端
（:class:`_doubles.FakePeftBackend`，其 ``load_base_model`` 返回已暴露
``peft_config`` 的 :class:`_doubles.FakePeftModel`）驱动，使
:func:`~ego_pod.runtime.inference._activate_cells` 走 ``load_adapter`` +
``set_adapter`` 的多适配器并存激活分支，从而行使真实的活动适配器维护逻辑而无需
任何真实权重。``_activate_cells`` 内以 ``from peft import PeftModel`` 惰性导入
``peft``，故以一个 autouse fixture 把仅暴露 ``PeftModel`` 的假 ``peft`` 模块塞入
``sys.modules['peft']``，使惰性导入语句在 ``peft`` 有/无安装时均可解析。

**Feature: ego-layered-adapter**
"""

from __future__ import annotations

import io
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

# 共享测试替身（tests/_doubles.py，经 conftest.py 把 tests/ 加入 sys.path）。
from _doubles import FakePeftBackend, FakePeftModel

from ego_pod.exit_codes import RuntimeError_
from ego_pod.runtime.inference import (
    ADAPTER_RELPATH,
    CONFIG_FILENAME,
    DEFAULT_EDITION,
    KERNEL_ADAPTER_NAME,
    MANIFEST_FILENAME,
    OUTPUT_DIRNAME,
    STUB_MODE_STARTUP_NOTICE,
    STUB_PLACEHOLDER_REPLY,
    BackendInferenceSession,
    StubInferenceSession,
    _cell_name,
    load_inference_session,
)
from ego_pod.runtime.selection import KERNEL, CellKey
from ego_pod.trainer.cells import TrainedCell
from ego_pod.vault.grid_descriptor import (
    KERNEL_RELPATH,
    build_grid_descriptor,
    zone_cell_relpath,
)


# ---------------------------------------------------------------------------
# peft 惰性导入补丁：使 ``from peft import PeftModel`` 在有/无 peft 时均可解析
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_peft_import(monkeypatch):
    """把假 ``peft`` 模块塞入 ``sys.modules`` 以替换/覆盖真实 ``peft``。

    :func:`~ego_pod.runtime.inference._activate_cells` 惰性 ``from peft import
    PeftModel``；注入的 base（:class:`FakePeftModel`）已暴露 ``peft_config``，故
    ``PeftModel`` 永不被调用——此补丁只保证导入语句可解析，使本套测试在 ``peft``
    未安装（替身替换）与已安装（替身覆盖，确保不误用真实实现）两种环境下均可
    运行。``monkeypatch`` 会在每个用例后自动还原。不构造真实后端会话的用例
    （缺失/损坏校验、Stub 路径）不会触达该导入，此补丁对它们是无害空操作。
    """

    fake_peft = types.ModuleType("peft")
    fake_peft.PeftModel = FakePeftModel
    monkeypatch.setitem(sys.modules, "peft", fake_peft)
    yield


# ---------------------------------------------------------------------------
# 磁盘 Ego_Package 构造辅助
# ---------------------------------------------------------------------------


def _write_valid_adapter(path: Path) -> None:
    """在 ``path`` 写入一个合法的 SafeTensors 适配器文件（含 1×1 占位张量）。

    使 :func:`~ego_pod.runtime.inference._validate_adapter_file` 的完整性校验通过。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"w": np.zeros((1, 1), dtype=np.float32)}, str(path))


def _write_corrupt_adapter(path: Path) -> None:
    """在 ``path`` 写入损坏（非法 SafeTensors）字节，使完整性校验失败。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"this is definitely not a valid safetensors file header")


def _grid_relpaths(zone_levels: dict[str, int]) -> list[str]:
    """据 ``{zone: max_level}`` 列出该网格全部 Cell 的相对路径（Kernel 在前）。"""

    relpaths = [KERNEL_RELPATH]
    for zone in sorted(zone_levels):
        for level in range(1, zone_levels[zone] + 1):
            relpaths.append(zone_cell_relpath(zone, level))
    return relpaths


def _write_grid_package(
    cwd: Path,
    *,
    zone_levels: dict[str, int],
    training_backend: str = "stub",
    edition: str = DEFAULT_EDITION,
    missing: set[str] = frozenset(),
    corrupt: set[str] = frozenset(),
) -> Path:
    """在 ``cwd`` 下构造一个含 ``grid`` 键的 Grid_Package。

    写入根部 ``manifest.json``（五字段 + 经
    :func:`~ego_pod.vault.grid_descriptor.build_grid_descriptor` 构造的 ``grid``
    描述）、``<edition>/config.json`` 与各 Cell 适配器文件
    （Kernel → ``adapters/kernel.safetensors``、分区 Cell →
    ``adapters/<zone>/L<level>.safetensors``）。

    :param zone_levels: ``{Zone_Name: 最大 Level}``，每分区生成 Level ``1..max`` 的
        连续 Cell。
    :param training_backend: 写入 manifest 的后端标记（``"stub"`` 走 Stub 路径，
        其余走真实后端路径）。
    :param missing: 不写出（模拟缺失）的 Cell 相对路径集合。
    :param corrupt: 写入损坏字节的 Cell 相对路径集合。
    :returns: Ego_Package 根目录（``cwd/ego-output``）。
    """

    root = cwd / OUTPUT_DIRNAME
    edition_dir = root / edition
    edition_dir.mkdir(parents=True, exist_ok=True)

    # 构造已训练 Cell 列表（output_path 仅作占位，build_grid_descriptor 不使用它）。
    trained = [TrainedCell(zone=None, level=0, output_path=Path("unused"))]
    for zone in sorted(zone_levels):
        for level in range(1, zone_levels[zone] + 1):
            trained.append(
                TrainedCell(zone=zone, level=level, output_path=Path("unused"))
            )
    grid = build_grid_descriptor(trained, edition=edition)

    manifest = {
        "version": "1.0.0",
        "created_at": "2025-11-25T08:00:00Z",
        "base_model": "qwen3.5-0.8B",
        "editions": [edition],
        "training_backend": training_backend,
        "grid": grid,
    }
    (root / MANIFEST_FILENAME).write_text(json.dumps(manifest), encoding="utf-8")

    config = {
        "base_model": "qwen3.5-0.8B",
        "quantization": "4bit",
        "lora_rank": 16,
        "lora_alpha": 32,
    }
    (edition_dir / CONFIG_FILENAME).write_text(json.dumps(config), encoding="utf-8")

    for relpath in _grid_relpaths(zone_levels):
        if relpath in missing:
            continue
        target = edition_dir / relpath
        if relpath in corrupt:
            _write_corrupt_adapter(target)
        else:
            _write_valid_adapter(target)

    return root


def _write_flat_stub_package(cwd: Path, *, edition: str = DEFAULT_EDITION) -> Path:
    """在 ``cwd`` 下构造一个**单 Cell**的纯 Flat_Package（无 ``grid`` 键，stub 标记）。

    适配器落于 ``<edition>/adapters/ego-lora.safetensors``，与 ego-pod-mvp 产物布局
    一致；``training_backend == "stub"`` 使其走 Stub 推理路径。
    """

    root = cwd / OUTPUT_DIRNAME
    edition_dir = root / edition
    edition_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "version": "1.0.0",
        "created_at": "2025-11-25T08:00:00Z",
        "base_model": "qwen3.5-0.8B",
        "editions": [edition],
        "training_backend": "stub",
    }
    (root / MANIFEST_FILENAME).write_text(json.dumps(manifest), encoding="utf-8")

    config = {
        "base_model": "qwen3.5-0.8B",
        "quantization": "4bit",
        "lora_rank": 16,
        "lora_alpha": 32,
    }
    (edition_dir / CONFIG_FILENAME).write_text(json.dumps(config), encoding="utf-8")

    _write_valid_adapter(edition_dir / ADAPTER_RELPATH)
    return root


def _make_multi_cell_session(
    zone_levels: dict[str, int],
) -> tuple[BackendInferenceSession, FakePeftModel, list[CellKey]]:
    """用注入的假 PEFT 后端构造一个**多 Cell**的 :class:`BackendInferenceSession`。

    cells 顺序与 Full_In_Load 一致（Kernel 在前，再各分区 Level 升序）；返回会话、
    底层假 PEFT 模型与有序 Cell 键列表，便于断言活动适配器与释放语义。路径为合成
    串（假后端只记录、从不读取），无 I/O。
    """

    ordered: list[CellKey] = [KERNEL]
    cells: list[tuple[CellKey, Path]] = [
        (KERNEL, Path("adapters/kernel.safetensors"))
    ]
    for zone in sorted(zone_levels):
        for level in range(1, zone_levels[zone] + 1):
            key = CellKey(zone, level)
            ordered.append(key)
            cells.append((key, Path(f"adapters/{zone}/L{level}.safetensors")))

    fake_model = FakePeftModel()
    backend = FakePeftBackend(model=fake_model)
    session = BackendInferenceSession(
        base_model="qwen3.5-0.8B",
        quantization="4bit",
        lora_rank=16,
        lora_alpha=32,
        cells=cells,
        backend=backend,
    )
    return session, fake_model, ordered


# ===========================================================================
# R5-AC7：缺失 / 损坏 Cell ⇒ RuntimeError_ 含 Cell 标识与文件相对路径
# ===========================================================================


def test_missing_kernel_cell_raises_with_label_and_relpath(tmp_path):
    # 缺失 Kernel 适配器 ⇒ RuntimeError_，消息含 "Kernel" 与 kernel 相对路径。
    _write_grid_package(
        tmp_path,
        zone_levels={"coding": 1},
        missing={KERNEL_RELPATH},
    )

    with pytest.raises(RuntimeError_) as excinfo:
        load_inference_session(cwd=tmp_path, stdout=io.StringIO())

    message = str(excinfo.value)
    assert "Kernel" in message
    assert "adapters/kernel.safetensors" in message


def test_missing_zone_cell_raises_with_label_and_relpath(tmp_path):
    # 缺失某分区 Cell（Kernel 合法）⇒ RuntimeError_，消息含 "(coding, L1)" 与其相对路径。
    _write_grid_package(
        tmp_path,
        zone_levels={"coding": 1},
        missing={zone_cell_relpath("coding", 1)},
    )

    with pytest.raises(RuntimeError_) as excinfo:
        load_inference_session(cwd=tmp_path, stdout=io.StringIO())

    message = str(excinfo.value)
    assert "(coding, L1)" in message
    assert "adapters/coding/L1.safetensors" in message


def test_corrupt_kernel_cell_raises_with_label_and_relpath(tmp_path):
    # 损坏的 Kernel 适配器 ⇒ RuntimeError_，消息含 "Kernel" 与 kernel 相对路径。
    _write_grid_package(
        tmp_path,
        zone_levels={"coding": 1},
        corrupt={KERNEL_RELPATH},
    )

    with pytest.raises(RuntimeError_) as excinfo:
        load_inference_session(cwd=tmp_path, stdout=io.StringIO())

    message = str(excinfo.value)
    assert "Kernel" in message
    assert "adapters/kernel.safetensors" in message


def test_corrupt_zone_cell_raises_with_label_and_relpath(tmp_path):
    # 损坏的某分区 Cell（Kernel 合法）⇒ RuntimeError_，消息含 "(knowledge, L2)" 与其相对路径。
    _write_grid_package(
        tmp_path,
        zone_levels={"knowledge": 2},
        corrupt={zone_cell_relpath("knowledge", 2)},
    )

    with pytest.raises(RuntimeError_) as excinfo:
        load_inference_session(cwd=tmp_path, stdout=io.StringIO())

    message = str(excinfo.value)
    assert "(knowledge, L2)" in message
    assert "adapters/knowledge/L2.safetensors" in message


# ===========================================================================
# R5-AC9：close() 释放全部 Cell（Kernel + 各分区 Cell）
# ===========================================================================


def test_multi_cell_session_activates_all_cells_before_close():
    # 构造后活动适配器集合含全部 Cell（Kernel + 各分区 Cell），Kernel 在首位。
    session, fake_model, ordered = _make_multi_cell_session(
        {"coding": 2, "knowledge": 1}
    )

    expected_names = [_cell_name(cell) for cell in ordered]
    assert session.model is not None
    assert session.tokenizer is not None
    assert session.active_adapters == expected_names
    assert session.active_adapters[0] == KERNEL_ADAPTER_NAME
    # 多 Cell：含 Kernel 在内共 1 + 2 + 1 = 4 个 Cell 被激活。
    assert len(session.active_adapters) == 4
    # 假模型镜像的活动集合与会话一致。
    assert fake_model.active_adapters == expected_names


def test_close_releases_all_cells():
    # R5-AC9：close() 释放全部 Cell 与 Base_Model —— model/tokenizer 置空、活动集合清空。
    session, _fake_model, _ordered = _make_multi_cell_session(
        {"coding": 2, "knowledge": 1}
    )
    assert len(session.active_adapters) == 4

    session.close()

    assert session.model is None
    assert session.tokenizer is None
    assert session.adapter_tensors is None
    # 全部 Cell 适配器被释放：活动集合清空。
    assert session.active_adapters == []


def test_close_is_idempotent():
    # 重复 close() 保持幂等、不抛错（资源释放为尽力而为）。
    session, _fake_model, _ordered = _make_multi_cell_session({"coding": 1})

    session.close()
    session.close()

    assert session.model is None
    assert session.active_adapters == []


# ===========================================================================
# R5-AC10：Stub 路径无论一个还是多个 Cell 行为不变
# ===========================================================================


def test_stub_single_cell_flat_package_behavior(tmp_path):
    # 单 Cell 纯 Flat_Package（stub）⇒ StubInferenceSession + 启动提示 + 固定回复。
    _write_flat_stub_package(tmp_path)
    stdout = io.StringIO()

    session = load_inference_session(cwd=tmp_path, stdout=stdout)

    assert isinstance(session, StubInferenceSession)
    assert STUB_MODE_STARTUP_NOTICE in stdout.getvalue()
    assert session.generate("anything") == STUB_PLACEHOLDER_REPLY
    assert session.close() is None


def test_stub_multi_cell_grid_package_behavior(tmp_path):
    # 多 Cell Grid_Package（stub）⇒ 同样 StubInferenceSession + 启动提示 + 固定回复。
    _write_grid_package(
        tmp_path,
        zone_levels={"coding": 2, "knowledge": 1},
        training_backend="stub",
    )
    stdout = io.StringIO()

    session = load_inference_session(cwd=tmp_path, stdout=stdout)

    assert isinstance(session, StubInferenceSession)
    assert STUB_MODE_STARTUP_NOTICE in stdout.getvalue()
    assert session.generate("anything") == STUB_PLACEHOLDER_REPLY
    assert session.close() is None


def test_stub_behavior_identical_single_vs_multi_cell(tmp_path):
    # R5-AC10：Stub 行为与 Cell 数目无关——单 Cell 与多 Cell 包的可观测行为完全一致。
    flat_cwd = tmp_path / "flat"
    flat_cwd.mkdir()
    _write_flat_stub_package(flat_cwd)

    grid_cwd = tmp_path / "grid"
    grid_cwd.mkdir()
    _write_grid_package(
        grid_cwd,
        zone_levels={"coding": 2, "knowledge": 1},
        training_backend="stub",
    )

    flat_out = io.StringIO()
    grid_out = io.StringIO()
    flat_session = load_inference_session(cwd=flat_cwd, stdout=flat_out)
    grid_session = load_inference_session(cwd=grid_cwd, stdout=grid_out)

    # 同类型会话、同启动提示、同固定回复（与输入内容、Cell 数目均无关）。
    assert type(flat_session) is type(grid_session) is StubInferenceSession
    assert flat_out.getvalue() == grid_out.getvalue()
    for probe in ("hello", "另一个完全不同的输入", ""):
        assert flat_session.generate(probe) == grid_session.generate(probe)
    assert flat_session.generate("x") == STUB_PLACEHOLDER_REPLY


# ===========================================================================
# R5-AC1：未指定 selection ⇒ Full_In_Load（Kernel + 每分区 Level 1..max）
# ===========================================================================


def test_unspecified_selection_loads_full_in_load(tmp_path):
    # R5-AC1：不传 selection ⇒ Full_In_Load——加载 Kernel + 每个分区 Level 1..max。
    _write_grid_package(
        tmp_path,
        zone_levels={"coding": 2, "knowledge": 1},
        training_backend="peft-cpu",
    )
    fake_model = FakePeftModel()
    backend = FakePeftBackend(model=fake_model)

    session = load_inference_session(
        cwd=tmp_path, backend=backend, stdout=io.StringIO()
    )

    # 走真实后端多适配器路径而非 Stub。
    assert isinstance(session, BackendInferenceSession)

    # 加载的有序 Cell 列表恰为 Full_In_Load：Kernel 在首位，再各分区 Level 升序。
    loaded_cells = [cell for cell, _path in session.cells]
    assert loaded_cells == [
        KERNEL,
        CellKey("coding", 1),
        CellKey("coding", 2),
        CellKey("knowledge", 1),
    ]
    # 活动适配器顺序与 Full_In_Load 顺序逐一一致（Kernel → 各分区 Level 低→高）。
    assert session.active_adapters == [
        KERNEL_ADAPTER_NAME,
        "coding_L1",
        "coding_L2",
        "knowledge_L1",
    ]


def test_unspecified_selection_single_zone_full_in_load(tmp_path):
    # 单分区多层级：Full_In_Load 加载 Kernel + 该分区全部 Level（1..max）。
    _write_grid_package(
        tmp_path,
        zone_levels={"coding": 3},
        training_backend="peft-cpu",
    )
    fake_model = FakePeftModel()
    backend = FakePeftBackend(model=fake_model)

    session = load_inference_session(
        cwd=tmp_path, backend=backend, stdout=io.StringIO()
    )

    loaded_cells = [cell for cell, _path in session.cells]
    assert loaded_cells == [
        KERNEL,
        CellKey("coding", 1),
        CellKey("coding", 2),
        CellKey("coding", 3),
    ]
