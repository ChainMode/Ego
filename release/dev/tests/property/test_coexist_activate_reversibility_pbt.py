"""Coexist_Activate 可逆性的属性测试。

Feature: ego-layered-adapter, Property 4

Validates: Requirements 5.4, 5.5, 7.6

本文件实现 design.md「Correctness Properties」中的 **Property 4**，对应
tasks.md 任务 5.5：

* **Property 4: Coexist_Activate 可逆性** —— *For any* 有序 Cell 序列与任意一次
  "激活后停用"操作，使用 PEFT 多适配器激活状态机（以假 PEFT 模型替身记录
  ``load_adapter`` / ``set_adapter`` 调用与活动适配器集合）后，激活某 Cell 再停用
  该 Cell，活动 Cell 集合回到激活该 Cell 之前的状态；初始激活顺序恒为
  Kernel → 各分区 Level 由低到高。

测试在**无任何机器学习依赖**的环境下运行：
:class:`~ego_pod.runtime.inference.BackendInferenceSession` 由注入的假后端
（:class:`_doubles.FakePeftBackend`）驱动——其 ``load_base_model`` 返回一个
已暴露 ``peft_config`` 的假 PEFT 模型（:class:`_doubles.FakePeftModel`），使
:func:`~ego_pod.runtime.inference._activate_cells` 走
``load_adapter`` + ``set_adapter`` 的多适配器并存激活分支，从而行使**真实**的
``active_adapters`` 维护与 ``activate`` / ``deactivate`` 可逆逻辑，而无需任何
真实权重或张量。

``ego_pod.runtime.inference._activate_cells`` 内以 ``from peft import PeftModel``
惰性导入 ``peft``；为使测试在 ``peft`` **未安装**（替身替换）与 ``peft`` **已安装**
（替身覆盖、避免误用真实实现）两种环境下均可运行，本模块以一个 autouse fixture
把一个仅暴露 ``PeftModel`` 的假 ``peft`` 模块塞入 ``sys.modules['peft']``。由于注入
的 base 已暴露 ``peft_config``，``_activate_cells`` 实际走 ``load_adapter`` 分支、
从不调用 ``PeftModel.from_pretrained``，该补丁仅确保惰性导入语句本身可解析。

每个属性最少运行 **100 次迭代**（``max_examples >= 100``）。
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings

# 共享测试替身（tests/_doubles.py，经 conftest.py 把 tests/ 加入 sys.path）。
from _doubles import FakePeftBackend, FakePeftModel

import ego_pod.runtime.inference as inference
from ego_pod.runtime.inference import (
    KERNEL_ADAPTER_NAME,
    BackendInferenceSession,
    _cell_name,
)
from ego_pod.runtime.selection import KERNEL, CellKey


# ---------------------------------------------------------------------------
# peft 惰性导入补丁：使 ``from peft import PeftModel`` 在有/无 peft 时均可解析
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_peft_import(monkeypatch):
    """把假 ``peft`` 模块塞入 ``sys.modules`` 以替换/覆盖真实 ``peft``。

    ``_activate_cells`` 惰性 ``from peft import PeftModel``；注入的 base 已暴露
    ``peft_config``，故 ``PeftModel`` 永不被调用——此补丁只保证导入语句可解析，
    使本属性测试在 ``peft`` 未安装（替身替换）与已安装（替身覆盖，确保不误用
    真实实现）两种环境下均可运行。``monkeypatch`` 会在每个用例后自动还原。
    """

    fake_peft = types.ModuleType("peft")
    fake_peft.PeftModel = FakePeftModel
    monkeypatch.setitem(sys.modules, "peft", fake_peft)
    yield


# ---------------------------------------------------------------------------
# 输入生成策略：有序 Cell 序列（Kernel 在前 + 随机分区/层级，Level 低→高）
# ---------------------------------------------------------------------------

#: Zone_Name：小写字母、数字、连字符（R1-AC3 的 ``^[a-z0-9-]+$``）。
_ZONE_NAME = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789-", min_size=1, max_size=6
)

#: 每个分区的最大 Level（小的随机数，保证连续 ``1..max`` 且分区非空）。
_MAX_LEVEL = st.integers(min_value=1, max_value=4)


def _build_cells(zone_levels: dict[str, int]) -> list[tuple[CellKey, Path]]:
    """据 ``{zone: max_level}`` 构造有序的 ``(CellKey, Path)`` 列表。

    顺序与 :func:`~ego_pod.runtime.selection.select_cells` 的 Full_In_Load 一致：
    Kernel 在首位，再按分区名字典序、各分区 Level 由低到高（R5-AC4）。每个 Cell
    的路径为合成串（假后端只记录、从不读取），无 I/O。
    """

    cells: list[tuple[CellKey, Path]] = [
        (KERNEL, Path("adapters/kernel.safetensors"))
    ]
    for zone in sorted(zone_levels):
        for level in range(1, zone_levels[zone] + 1):
            cells.append(
                (
                    CellKey(zone, level),
                    Path(f"adapters/{zone}/L{level}.safetensors"),
                )
            )
    return cells


@st.composite
def ordered_cells(draw, *, min_zones: int = 1, max_zones: int = 4):
    """合成有序 Cell 列表：Kernel + 至少一个分区（每分区连续 ``1..max`` Level）。

    至少含一个分区 Cell，确保 :class:`BackendInferenceSession` 走多适配器
    Coexist_Activate 路径（``_activate_cells``），而非单 Kernel 的
    ``_apply_lora_adapter`` 兼容路径。
    """

    zone_levels = draw(
        st.dictionaries(_ZONE_NAME, _MAX_LEVEL, min_size=min_zones, max_size=max_zones)
    )
    return _build_cells(zone_levels)


def _make_session(
    cells: list[tuple[CellKey, Path]],
) -> tuple[BackendInferenceSession, FakePeftModel]:
    """用注入的假 PEFT 后端构造一个 :class:`BackendInferenceSession`。

    返回会话及其底层假 PEFT 模型（``session.model``），以便断言其记录的
    ``load_adapter`` / ``set_adapter`` 调用与活动适配器集合。
    """

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
    return session, fake_model


def _expected_names(cells: list[tuple[CellKey, Path]]) -> list[str]:
    """有序 Cell 列表对应的 PEFT 适配器名列表（Kernel 在前，与激活顺序一致）。"""

    return [_cell_name(cell) for cell, _ in cells]


# ===========================================================================
# Property 4a：初始激活顺序为 Kernel → 各分区 Level 低→高（R5-AC4）
# ===========================================================================


@given(cells=ordered_cells())
@settings(max_examples=200, deadline=None)
def test_initial_activation_order_is_kernel_first_then_levels_ascending(
    cells: list[tuple[CellKey, Path]],
) -> None:
    """构造会话后，初始激活的有序适配器集合恒为 Kernel → 各分区 Level 低→高。

    断言：(a) 构造期间恰发生一次 ``set_adapter`` 调用，其入参列表等于按 Cell 顺序
    （Kernel 在前）的适配器名；(b) 首个被激活的适配器为 Kernel
    （:data:`KERNEL_ADAPTER_NAME`）；(c) ``load_adapter`` 的调用顺序与 Cell 顺序
    逐一一致；(d) 会话的 ``active_adapters`` 与假模型的 ``active_adapters`` 镜像
    一致（R5-AC4）。

    Feature: ego-layered-adapter, Property 4

    Validates: Requirements 5.4, 7.6
    """
    session, fake_model = _make_session(cells)
    expected = _expected_names(cells)

    # (a) 构造期间恰一次 set_adapter，入参为 Kernel 在前的有序激活集合。
    assert fake_model.set_adapter_calls == [expected]

    # (b) 首个激活的适配器是 Kernel。
    assert expected[0] == KERNEL_ADAPTER_NAME
    assert fake_model.active_adapters[0] == KERNEL_ADAPTER_NAME

    # (c) load_adapter 调用顺序与 Cell 顺序逐一一致（Kernel → 各分区 Level 低→高）。
    loaded_names = [name for _path, name in fake_model.load_adapter_calls]
    assert loaded_names == expected

    # (d) 会话活动集合与假模型镜像一致。
    assert session.active_adapters == expected
    assert fake_model.active_adapters == expected


# ===========================================================================
# Property 4b：可逆性 —— 激活某 Cell 再停用，活动集合精确回到先前状态（R5-AC5）
# ===========================================================================


@given(cells=ordered_cells(), data=st.data())
@settings(max_examples=200, deadline=None)
def test_activate_then_deactivate_restores_prior_active_set(
    cells: list[tuple[CellKey, Path]],
    data: st.DataObject,
) -> None:
    """对任意 Cell：先确保其未激活，再激活后停用 ⇒ 活动集合精确回到先前状态。

    这是 R5-AC5 的字面表述："激活某 Cell 后再停用该 Cell，活动 Cell 集合回到
    激活该 Cell 之前的状态"。由于活动列表始终按规范 Cell 顺序重建，断言可强化为
    **有序列表逐一相等**（"回到 EXACTLY 先前状态"）。

    Feature: ego-layered-adapter, Property 4

    Validates: Requirements 5.5, 7.6
    """
    session, _fake_model = _make_session(cells)

    target_cell, _path = data.draw(st.sampled_from(cells))

    # 先把目标 Cell 停用，确立一个"目标未激活"的先前状态（其它 Cell 仍激活）。
    session.deactivate(target_cell)
    prior = list(session.active_adapters)
    name = _cell_name(target_cell)
    assert name not in prior

    # 激活该 Cell（进入"已激活"中间态），再停用该 Cell。
    session.activate(target_cell)
    assert name in set(session.active_adapters)
    session.deactivate(target_cell)

    # 活动集合精确回到激活该 Cell 之前的状态（有序列表逐一相等）。
    assert session.active_adapters == prior


@given(cells=ordered_cells(), data=st.data())
@settings(max_examples=200, deadline=None)
def test_deactivate_then_reactivate_restores_prior_active_set(
    cells: list[tuple[CellKey, Path]],
    data: st.DataObject,
) -> None:
    """对任意当前已激活的 Cell：停用后再激活 ⇒ 活动集合精确回到先前状态。

    构造后全部 Cell 均激活；选任一 Cell，停用再重新激活，活动有序列表应逐一
    回到停用前的状态（可逆性的对偶方向，R5-AC5）。

    Feature: ego-layered-adapter, Property 4

    Validates: Requirements 5.5, 7.6
    """
    session, _fake_model = _make_session(cells)

    target_cell, _path = data.draw(st.sampled_from(cells))

    prior = list(session.active_adapters)
    name = _cell_name(target_cell)
    assert name in prior  # 构造后全部 Cell 均激活。

    # 停用该 Cell（进入"已停用"中间态），再重新激活该 Cell。
    session.deactivate(target_cell)
    assert name not in set(session.active_adapters)
    session.activate(target_cell)

    # 活动集合精确回到停用该 Cell 之前的状态（有序列表逐一相等）。
    assert session.active_adapters == prior


@given(cells=ordered_cells(), data=st.data())
@settings(max_examples=200, deadline=None)
def test_round_trip_restores_from_arbitrary_prior_state(
    cells: list[tuple[CellKey, Path]],
    data: st.DataObject,
) -> None:
    """从任意先前活动状态出发，对任意 Cell 的"操作 + 其逆操作"恒回到先前状态。

    先停用一个随机子集，得到一个任意的"先前状态"；再选任一 Cell，依据其当前是否
    激活分别走"停用→再激活"或"激活→停用"，断言活动有序列表精确回到先前状态。
    这把 R5-AC5 的可逆性强化为"从任意状态出发、对任意 Cell 成立"。

    Feature: ego-layered-adapter, Property 4

    Validates: Requirements 5.5, 7.6
    """
    session, fake_model = _make_session(cells)

    all_cells = [cell for cell, _ in cells]
    # 随机停用一个子集（建立任意先前状态）；Kernel 也可被停用以充分行使状态机。
    to_deactivate = data.draw(st.lists(st.sampled_from(all_cells), unique=True))
    for cell in to_deactivate:
        session.deactivate(cell)

    prior = list(session.active_adapters)
    prior_set = set(prior)

    target_cell, _path = data.draw(st.sampled_from(cells))
    name = _cell_name(target_cell)
    calls_before = len(fake_model.set_adapter_calls)

    if name in prior_set:
        # 当前激活 ⇒ 停用后再激活恢复。
        session.deactivate(target_cell)
        assert name not in set(session.active_adapters)
        session.activate(target_cell)
    else:
        # 当前未激活 ⇒ 激活后再停用恢复。
        session.activate(target_cell)
        assert name in set(session.active_adapters)
        session.deactivate(target_cell)

    # 活动集合精确回到先前状态（有序列表逐一相等，且集合相等）。
    assert session.active_adapters == prior
    assert set(session.active_adapters) == prior_set

    # 中间确实发生了状态变更（每个方向各两次 set_adapter 调用，确认走了真实状态机）。
    assert len(fake_model.set_adapter_calls) == calls_before + 2
    # 假模型镜像的活动集合与会话一致，且其最近一次 set_adapter 即为恢复后的状态。
    assert fake_model.active_adapters == prior
    assert fake_model.set_adapter_calls[-1] == prior


# ===========================================================================
# Property 4c：deactivate 幂等 —— 重复停用同一 Cell 不改变活动集合（可逆性辅助）
# ===========================================================================


@given(cells=ordered_cells(), data=st.data())
@settings(max_examples=100, deadline=None)
def test_deactivate_is_idempotent_no_op_when_inactive(
    cells: list[tuple[CellKey, Path]],
    data: st.DataObject,
) -> None:
    """对已停用的 Cell 再次 ``deactivate`` 为幂等空操作，不触发额外 ``set_adapter``。

    支撑可逆性的稳态语义：停用不在活动集合中的 Cell 既不改变活动集合，也不产生
    新的 ``set_adapter`` 调用（``deactivate`` 对不活动 Cell 直接返回）。

    Feature: ego-layered-adapter, Property 4

    Validates: Requirements 5.5, 7.6
    """
    session, fake_model = _make_session(cells)

    target_cell, _path = data.draw(st.sampled_from(cells))
    session.deactivate(target_cell)
    state = list(session.active_adapters)
    calls = len(fake_model.set_adapter_calls)

    # 再次停用：幂等空操作。
    session.deactivate(target_cell)

    assert session.active_adapters == state
    assert len(fake_model.set_adapter_calls) == calls
