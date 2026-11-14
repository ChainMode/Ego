"""manifest 五字段兼容不变量的属性测试。

本文件实现 design.md「Correctness Properties」中的 **Property 6**，对应
tasks.md 任务 4.4：

* **Property 6: manifest 五字段兼容不变量** —— 对任意合法输入（任意
  ``training_backend`` ∈ Backend_Identifier 固定集合
  ``{unsloth, peft-cuda, mlx, peft-cpu, stub}``、任意 ``grid`` 为 ``None`` 或
  合法网格描述），:func:`~ego_pod.vault.package.build_manifest` 的输出恒包含且
  仅以 ``version`` / ``created_at`` / ``base_model`` / ``editions`` /
  ``training_backend`` 五个既有字段为基础（名称、类型、取值约定不变）；当且仅当
  ``grid is not None`` 时追加 ``grid`` 键，``grid=None`` 时输出与不带 ``grid``
  参数的调用逐字段一致。

  **Validates: Requirements 4.4, 6.1**

本测试覆盖四个不变量（与 tasks.md 任务 4.4 的四点一一对应）：

1. **五字段恒在且类型/取值正确** —— 无论 ``grid`` 为 ``None`` 还是合法网格，
   输出恒含且仅以五个既有字段为基础：``version`` / ``created_at`` 为 ``str``、
   ``base_model`` 为 ``str``、``editions`` 为非空 ``list[str]`` 且含 ``"normal"``、
   ``training_backend`` 为 ``str`` 且**取值原样保留**传入的 Backend_Identifier。
2. **grid 键当且仅当 grid is not None 出现** —— ``grid is None`` ⇒ 输出无
   ``grid`` 键（恰五键）；``grid is not None`` ⇒ 输出恰含 ``grid`` 键且其值即传入
   的网格描述。
3. **grid=None 与不带 grid 参数逐字段一致** —— 固定 ``version`` / ``created_at``
   以确定性比较，``build_manifest(grid=None)`` 与 ``build_manifest()``（不传
   ``grid`` kwarg）逐字段完全相等（Flat_Package 形状不变，R4-AC4 / R6-AC1）。
4. **五字段不受 grid 影响** —— 对相同的其它入参，带任意 ``grid``（含 ``None``）
   与不带 ``grid`` 的调用，其**五字段投影**完全一致（``grid`` 的存在/缺失/内容
   都不改动五字段，R4-AC4）。

本测试在**无机器学习依赖**的环境下运行：:func:`build_manifest` 为纯函数（仅构造
字典、不读写文件、无 ML 依赖），故 hypothesis 直接以随机 Backend_Identifier 与
合成网格描述（或 ``None``）行使不变量，每个属性测试最少运行 100 次迭代。

**Feature: ego-layered-adapter, Property 6**

**Validates: Requirements 4.4, 6.1**
"""

from __future__ import annotations

import hypothesis.strategies as st
from hypothesis import given, settings

from ego_pod.vault.package import EDITION_NAME, build_manifest

# ---------------------------------------------------------------------------
# 输入生成策略
# ---------------------------------------------------------------------------

#: Backend_Identifier 的固定取值集合（R6-AC5，取值集合不变）；Property 6 的论域。
_BACKEND_IDENTIFIERS: tuple[str, ...] = (
    "unsloth",
    "peft-cuda",
    "mlx",
    "peft-cpu",
    "stub",
)

#: 取自固定集合的 Backend_Identifier。
_training_backend = st.sampled_from(_BACKEND_IDENTIFIERS)

#: 固定 version / created_at，使 grid=None 与不带 grid 的输出可确定性逐字段比较
#: （否则 created_at 默认取当前时间会引入非确定性）。
_FIXED_VERSION = "0.1.0"
_FIXED_CREATED_AT = "2025-11-25T08:00:00Z"

#: Zone_Name：小写字母、数字、连字符（匹配 ^[a-z0-9-]+$，R1-AC3），短小以加速收缩。
_zone_name = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789-",
    min_size=1,
    max_size=6,
)

#: 每个分区的最大 Level（小而随机；schema 支持任意深度）。
_max_level = st.integers(min_value=1, max_value=4)


def _build_zone_entries(zone: str, max_level: int) -> list[dict]:
    """据 ``(zone, max_level)`` 构造一个分区的 Level 条目数组（连续 1..max）。"""

    return [
        {"level": level, "path": f"adapters/{zone}/L{level}.safetensors"}
        for level in range(1, max_level + 1)
    ]


@st.composite
def _grids(draw: st.DrawFn) -> dict:
    """生成一个 well-formed 的合成 Grid_Descriptor 字典（单 Edition）。

    形如 ``{<edition>: {"kernel": {"path": ...}, "zones": {<zone>: [{level, path}, ...]}}}``，
    与 design.md §"Data Models" 的磁盘 JSON 形状一致；分区集合可为空（仅 Kernel）。
    """

    zone_maxlevels = draw(
        st.dictionaries(keys=_zone_name, values=_max_level, max_size=3)
    )
    zones = {
        zone: _build_zone_entries(zone, max_level)
        for zone, max_level in zone_maxlevels.items()
    }
    return {
        EDITION_NAME: {
            "kernel": {"path": "adapters/kernel.safetensors"},
            "zones": zones,
        }
    }


#: grid 入参：None（Flat_Package）或一个合法的合成网格描述。
_grid_or_none = st.one_of(st.none(), _grids())

# 五个既有字段名（名称、顺序、取值约定不变，R4-AC4）。
_BASE_FIELDS = ("version", "created_at", "base_model", "editions", "training_backend")


# ===========================================================================
# Property 6: manifest 五字段兼容不变量（任务 4.4）
# ===========================================================================


@given(training_backend=_training_backend, grid=_grid_or_none)
@settings(max_examples=200, deadline=None)
def test_manifest_always_has_five_base_fields_with_correct_types(
    training_backend: str,
    grid: dict | None,
) -> None:
    """不变量 1：输出恒含且仅以五个既有字段为基础，类型/取值约定不变。

    无论 ``grid`` 为 ``None`` 还是合法网格，``build_manifest`` 的输出恒含五个既有
    字段，且：``version`` / ``created_at`` / ``base_model`` / ``training_backend``
    为 ``str``、``editions`` 为非空 ``list[str]`` 且含 ``"normal"``；
    ``training_backend`` 取值原样保留传入的 Backend_Identifier（R4-AC4）。

    **Feature: ego-layered-adapter, Property 6**

    **Validates: Requirements 4.4, 6.1**
    """
    manifest = build_manifest(
        training_backend=training_backend,
        version=_FIXED_VERSION,
        created_at=_FIXED_CREATED_AT,
        grid=grid,
    )

    # 五个既有字段恒存在（名称不变，R4-AC4）。
    for field in _BASE_FIELDS:
        assert field in manifest

    # 类型约定不变（R4-AC4）。
    assert isinstance(manifest["version"], str)
    assert isinstance(manifest["created_at"], str)
    assert isinstance(manifest["base_model"], str)
    assert isinstance(manifest["training_backend"], str)
    assert isinstance(manifest["editions"], list)
    assert all(isinstance(e, str) for e in manifest["editions"])

    # 取值约定：editions 非空且含 "normal"；training_backend 取值原样保留。
    assert manifest["editions"]
    assert EDITION_NAME in manifest["editions"]
    assert manifest["training_backend"] == training_backend

    # 固定入参原样回写（取值约定不变）。
    assert manifest["version"] == _FIXED_VERSION
    assert manifest["created_at"] == _FIXED_CREATED_AT


@given(training_backend=_training_backend, grid=_grid_or_none)
@settings(max_examples=200, deadline=None)
def test_grid_key_present_iff_grid_not_none(
    training_backend: str,
    grid: dict | None,
) -> None:
    """不变量 2：输出含 ``grid`` 键当且仅当 ``grid is not None``。

    * ``grid is None`` ⇒ 输出恰为五键、无 ``grid`` 键（Flat_Package 形状，R6-AC1）；
    * ``grid is not None`` ⇒ 输出恰为「五字段 + ``grid``」六键，且 ``grid`` 键的值
      即传入的网格描述（R4-AC1）。

    **Feature: ego-layered-adapter, Property 6**

    **Validates: Requirements 4.4, 6.1**
    """
    manifest = build_manifest(
        training_backend=training_backend,
        version=_FIXED_VERSION,
        created_at=_FIXED_CREATED_AT,
        grid=grid,
    )

    has_grid_key = "grid" in manifest
    # 当且仅当（iff）：grid 键存在 ⇔ grid 入参非 None。
    assert has_grid_key == (grid is not None)

    if grid is None:
        # 恰五键、无 grid（逐字节 Flat_Package）。
        assert set(manifest) == set(_BASE_FIELDS)
    else:
        # 恰「五字段 + grid」六键，且 grid 值即传入描述。
        assert set(manifest) == set(_BASE_FIELDS) | {"grid"}
        assert manifest["grid"] == grid


@given(training_backend=_training_backend)
@settings(max_examples=200, deadline=None)
def test_grid_none_equals_call_without_grid_kwarg(
    training_backend: str,
) -> None:
    """不变量 3：``grid=None`` 与不带 ``grid`` 参数的调用逐字段一致。

    固定 ``version`` / ``created_at`` 以确定性比较：``build_manifest(grid=None)``
    与 ``build_manifest()``（完全不传 ``grid`` kwarg）的输出字典完全相等——键集合、
    各字段取值、字段顺序均一致，确保默认即 Flat_Package、不引入任何新键
    （R4-AC4 / R6-AC1）。

    **Feature: ego-layered-adapter, Property 6**

    **Validates: Requirements 4.4, 6.1**
    """
    with_none = build_manifest(
        training_backend=training_backend,
        version=_FIXED_VERSION,
        created_at=_FIXED_CREATED_AT,
        grid=None,
    )
    without_kwarg = build_manifest(
        training_backend=training_backend,
        version=_FIXED_VERSION,
        created_at=_FIXED_CREATED_AT,
    )

    # 逐字段一致（dict 相等比较键集合与取值）。
    assert with_none == without_kwarg
    # 字段顺序亦一致（逐字节 manifest 的前提）。
    assert list(with_none) == list(without_kwarg)
    # 二者均为纯五键 Flat_Package 形状。
    assert set(with_none) == set(_BASE_FIELDS)


@given(training_backend=_training_backend, grid=_grids())
@settings(max_examples=200, deadline=None)
def test_five_base_fields_unaffected_by_grid(
    training_backend: str,
    grid: dict,
) -> None:
    """不变量 4：五字段投影不受 ``grid`` 的存在/缺失/内容影响。

    对相同的其它入参，比较「带任意非空 ``grid``」与「``grid=None``」两次调用的
    **五字段投影**：二者完全一致——即 ``grid`` 的追加既不改动五字段的取值，也不
    改动其相对顺序（R4-AC4）。

    **Feature: ego-layered-adapter, Property 6**

    **Validates: Requirements 4.4, 6.1**
    """
    with_grid = build_manifest(
        training_backend=training_backend,
        version=_FIXED_VERSION,
        created_at=_FIXED_CREATED_AT,
        grid=grid,
    )
    without_grid = build_manifest(
        training_backend=training_backend,
        version=_FIXED_VERSION,
        created_at=_FIXED_CREATED_AT,
        grid=None,
    )

    # 五字段投影（仅取五个既有字段、保持顺序）。
    projection_with = {k: with_grid[k] for k in _BASE_FIELDS}
    projection_without = {k: without_grid[k] for k in _BASE_FIELDS}

    # 取值一致：grid 不改动任何五字段取值。
    assert projection_with == projection_without
    # 相对顺序一致：grid 仅追加在五字段之后，不打乱既有字段顺序。
    assert [k for k in with_grid if k in _BASE_FIELDS] == list(_BASE_FIELDS)
    assert [k for k in without_grid if k in _BASE_FIELDS] == list(_BASE_FIELDS)
