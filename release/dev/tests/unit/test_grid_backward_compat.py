"""既有契约不变与扁平兼容的单元测试（任务 7.3）。

Backward-compatibility unit tests for the Ego Layered Adapter feature.

本文件验证二维网格（Sparse_Grid）特性在扩展 Compiler / Trainer / Vault /
Runtime / Config 五个模块时，**完整保留** ego-pod-mvp 与 ego-pod-real-trainer
的既有对外契约，并对退化为单 Kernel Cell 的包保持与现有 Flat_Package 的逐字节
兼容。覆盖需求：

* **R6-AC3**：``TrainingBackend`` Protocol 形状不变——五个后端
  （Stub / PeftCuda / PeftCpu / Unsloth / Mlx）仍是
  ``runtime_checkable`` 的 ``TrainingBackend``，且各自暴露
  ``name`` / ``hardware`` / ``identifier`` / ``is_available`` /
  ``load_base_model`` / ``train`` / ``export_lora_safetensors``。
* **R6-AC4**：``run_training`` 对外签名与 ``TrainingResult`` 对外字段不变；
  新增的 ``GridTrainingResult`` 是**独立**的新类型，不替换 ``TrainingResult``。
* **R6-AC5**：``detect_backend`` 探测优先级顺序不变
  （Unsloth → PeftCuda → Mlx → PeftCpu → Stub），且 Backend_Identifier 取值
  集合不变（``{unsloth, peft-cuda, mlx, peft-cpu, stub}``）。
* **R6-AC6**：退出码取值不变（``EXIT_OK == 0`` / ``EXIT_RUNTIME_FAILURE == 1``
  / ``EXIT_USAGE_OR_CONFIG == 2``），且本特性不引入新的退出码取值——全部
  ``EgoError`` 子类的退出码 ⊆ ``{0, 1, 2}``（``ConfigError`` → 2，其余 → 1）。
* **R6-AC1 / R6-AC2**：``build_manifest(grid=None)`` 与不传 ``grid`` 逐字节一致
  且为五字段、无 ``grid`` 键；``parse_grid_descriptor(None)`` 还原单 Kernel Cell
  的 Flat 网格（Kernel 相对路径 ``adapters/ego-lora.safetensors``）；
  ``select_cells(flat_grid, None) == [KERNEL]``。
* **R7-AC1（轻量）**：本特性新增模块均位于 ``release/dev/`` 目录树下。

本文件刻意**不**在模块顶层或运行时导入任何机器学习依赖（``torch`` / ``peft`` /
``transformers`` / ``unsloth`` / ``mlx`` 等）：被断言的契约均来自纯 Python 的数据
结构、``dataclass`` 字段、``inspect.signature`` 与纯函数，后端类的实例化亦不触碰
任何 ML 依赖（``load_base_model`` 等才会惰性导入）。因此本套测试可在无任何
Training_Dependencies 的环境（如 CI）中运行。

**Feature: ego-layered-adapter**
"""

from __future__ import annotations

import dataclasses
import inspect
import json
from pathlib import Path

import pytest

# 注意（导入顺序）：先导入 ``ego_pod.runtime.selection`` 再导入
# ``ego_pod.vault.grid_descriptor``，以规避 ``runtime`` ↔ ``vault`` 之间的
# 循环导入（``grid_descriptor`` 顶层依赖 ``runtime.selection``）。
from ego_pod.exit_codes import (
    EXIT_OK,
    EXIT_RUNTIME_FAILURE,
    EXIT_USAGE_OR_CONFIG,
    CollectorError,
    CompilerError,
    ConfigError,
    EgoError,
    RuntimeError_,
    TrainerError,
    VaultError,
)
from ego_pod.runtime.selection import KERNEL, Grid, select_cells
from ego_pod.trainer import detect
from ego_pod.trainer.base import TrainingBackend
from ego_pod.trainer.cells import GridTrainingResult
from ego_pod.trainer.mlx_backend import MlxBackend
from ego_pod.trainer.peft_backend import PeftCpuBackend, PeftCudaBackend
from ego_pod.trainer.pipeline import TrainingResult, run_training
from ego_pod.trainer.stub import StubBackend
from ego_pod.trainer.unsloth_backend import UnslothBackend
from ego_pod.vault.grid_descriptor import FLAT_KERNEL_RELPATH, parse_grid_descriptor
from ego_pod.vault.package import build_manifest

# ---------------------------------------------------------------------------
# 共享常量与辅助
# ---------------------------------------------------------------------------

#: Backend_Identifier 的固定取值集合（R6-AC5）。
_VALID_IDENTIFIERS = {"unsloth", "peft-cuda", "mlx", "peft-cpu", "stub"}

#: ``TrainingBackend`` Protocol 要求暴露的成员（属性 + 方法）（R6-AC3）。
_PROTOCOL_MEMBERS = (
    "name",
    "hardware",
    "identifier",
    "is_available",
    "load_base_model",
    "train",
    "export_lora_safetensors",
)

#: 按探测优先级从高到低排列的真实后端类（不含始终可用的 Stub 末位回退）。
_REAL_BACKENDS = (UnslothBackend, PeftCudaBackend, MlxBackend, PeftCpuBackend)
#: 含 Stub 末位回退的完整优先级序列（R6-AC5）。
_ALL_BACKENDS = _REAL_BACKENDS + (StubBackend,)

#: ``(后端类, 文档化 identifier)`` 对，覆盖全部五个 Training_Backend。
_BACKEND_CASES = [
    (StubBackend, "stub"),
    (PeftCudaBackend, "peft-cuda"),
    (PeftCpuBackend, "peft-cpu"),
    (UnslothBackend, "unsloth"),
    (MlxBackend, "mlx"),
]

#: 本特性新增模块（应全部位于 ``release/dev/`` 目录树下，R7-AC1）。
_FEATURE_MODULES = (
    "ego_pod.compiler.routing",
    "ego_pod.trainer.cells",
    "ego_pod.trainer.grid",
    "ego_pod.vault.grid_descriptor",
    "ego_pod.runtime.selection",
)


def _set_availability(monkeypatch, available: set[type]) -> None:
    """把每个候选后端类的 ``is_available()`` patch 为固定返回值。

    ``available`` 中的后端类返回 ``True``，其余返回 ``False``（等价于模拟
    "import 成功 + 硬件可用" 与 "import 失败或硬件不可用"）。
    """

    for backend_cls in _ALL_BACKENDS:
        is_avail = backend_cls in available
        monkeypatch.setattr(
            backend_cls,
            "is_available",
            classmethod(lambda cls, _v=is_avail: _v),
        )


# ===========================================================================
# R6-AC3：TrainingBackend Protocol 形状不变
# ===========================================================================


@pytest.mark.parametrize(
    "backend_cls,expected_id",
    _BACKEND_CASES,
    ids=[cls.__name__ for cls, _ in _BACKEND_CASES],
)
def test_backend_is_runtime_checkable_training_backend(
    backend_cls: type, expected_id: str
) -> None:
    """每个后端实例仍结构上满足 ``runtime_checkable`` 的 ``TrainingBackend``（R6-AC3）。

    **Validates: Requirements 6.3**
    """
    instance = backend_cls()
    assert isinstance(instance, TrainingBackend)


@pytest.mark.parametrize(
    "backend_cls,expected_id",
    _BACKEND_CASES,
    ids=[cls.__name__ for cls, _ in _BACKEND_CASES],
)
def test_backend_exposes_protocol_members(
    backend_cls: type, expected_id: str
) -> None:
    """每个后端暴露 Protocol 要求的全部属性与方法，且三个训练方法可调用（R6-AC3）。

    **Validates: Requirements 6.3**
    """
    instance = backend_cls()
    for member in _PROTOCOL_MEMBERS:
        assert hasattr(instance, member), (
            f"{backend_cls.__name__} 缺少 Protocol 成员 {member!r}"
        )
    for method in (
        "is_available",
        "load_base_model",
        "train",
        "export_lora_safetensors",
    ):
        assert callable(getattr(instance, method)), (
            f"{backend_cls.__name__}.{method} 不可调用"
        )


# ===========================================================================
# R6-AC4：run_training / TrainingResult 对外形状不变；GridTrainingResult 独立
# ===========================================================================


def test_training_result_fields_unchanged() -> None:
    """``TrainingResult`` 对外字段不变（四字段，原始三字段在前、顺序不变，R6-AC4）。

    **Validates: Requirements 6.4**
    """
    field_names = [f.name for f in dataclasses.fields(TrainingResult)]

    # 既有四个字段名与相对顺序不变（output_path/backend_name/backend_hardware 在前）。
    assert field_names == [
        "output_path",
        "backend_name",
        "backend_hardware",
        "backend_identifier",
    ]


def test_run_training_signature_unchanged() -> None:
    """``run_training`` 对外签名兼容：仅 ``corpus_path`` / ``params`` 必填（R6-AC4）。

    其余参数均为带默认值的关键字参数，既有 ``run_training(corpus_path, params)``
    调用方式不被破坏；参数名集合保持稳定。

    **Validates: Requirements 6.4**
    """
    sig = inspect.signature(run_training)

    required = [
        name
        for name, p in sig.parameters.items()
        if p.default is inspect.Parameter.empty
        and p.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    assert required == ["corpus_path", "params"]

    # 既有可选关键字参数名稳定存在且均带默认值（向后兼容的接线点）。
    for name in (
        "model_id",
        "cwd",
        "output_path",
        "backend",
        "offline",
        "base_model_available_check",
        "stdout",
    ):
        assert name in sig.parameters, f"缺失预期可选参数：{name}"
        assert sig.parameters[name].default is not inspect.Parameter.empty


def test_grid_training_result_is_distinct_new_type() -> None:
    """``GridTrainingResult`` 是独立的新类型，不替换 ``TrainingResult``（R6-AC4）。

    新网格路径走 ``GridTrainingResult``（携带多 Cell 的 ``cells`` 字段），单适配器
    路径仍走 ``TrainingResult``（携带单 ``output_path``），二者并存。

    **Validates: Requirements 6.4**
    """
    assert GridTrainingResult is not TrainingResult
    assert not issubclass(GridTrainingResult, TrainingResult)

    grid_fields = {f.name for f in dataclasses.fields(GridTrainingResult)}
    flat_fields = {f.name for f in dataclasses.fields(TrainingResult)}

    # GridTrainingResult 以 cells（多 Cell）为核心，TrainingResult 以 output_path
    # （单适配器）为核心——两者字段集不同，是不同的契约。
    assert "cells" in grid_fields
    assert "cells" not in flat_fields
    assert "output_path" in flat_fields
    assert "output_path" not in grid_fields


# ===========================================================================
# R6-AC5：detect_backend 优先级不变 + Backend_Identifier 取值集合不变
# ===========================================================================


def test_detect_backend_spec_order_unchanged() -> None:
    """``detect`` 候选规格的类名顺序固定为既有优先级（R6-AC5）。

    Unsloth → PeftCuda → Mlx → PeftCpu → Stub（Stub 末位为始终可用回退）。

    **Validates: Requirements 6.5**
    """
    class_names = [class_name for _module, class_name in detect._BACKEND_SPECS]
    assert class_names == [
        "UnslothBackend",
        "PeftCudaBackend",
        "MlxBackend",
        "PeftCpuBackend",
        "StubBackend",
    ]


@pytest.mark.parametrize(
    ("first_available_index", "expected_cls"),
    [
        (0, UnslothBackend),
        (1, PeftCudaBackend),
        (2, MlxBackend),
        (3, PeftCpuBackend),
        (4, StubBackend),
    ],
)
def test_detect_backend_priority_unchanged(
    monkeypatch, first_available_index: int, expected_cls: type
) -> None:
    """从某优先级起全部可用、其之前全部不可用 ⇒ 选中该优先级后端（R6-AC5）。

    以 monkeypatch 模拟各后端 ``is_available()``，验证探测仍按
    Unsloth → PeftCuda → Mlx → PeftCpu → Stub 的固定优先级选择第一个可用者。

    **Validates: Requirements 6.5**
    """
    available = set(_ALL_BACKENDS[first_available_index:])
    _set_availability(monkeypatch, available)

    backend = detect.detect_backend()

    assert isinstance(backend, expected_cls)


def test_backend_identifier_set_unchanged() -> None:
    """五个后端的 ``identifier`` 恰好覆盖固定取值集合且两两不同（R6-AC5）。

    **Validates: Requirements 6.5**
    """
    identifiers = {cls().identifier for cls, _ in _BACKEND_CASES}
    assert identifiers == _VALID_IDENTIFIERS


@pytest.mark.parametrize(
    "backend_cls,expected_id",
    _BACKEND_CASES,
    ids=[cls.__name__ for cls, _ in _BACKEND_CASES],
)
def test_backend_identifier_in_fixed_set(
    backend_cls: type, expected_id: str
) -> None:
    """每个后端 ``identifier`` 属于固定集合且等于其文档化取值（R6-AC5）。

    **Validates: Requirements 6.5**
    """
    instance = backend_cls()
    assert instance.identifier in _VALID_IDENTIFIERS
    assert instance.identifier == expected_id


# ===========================================================================
# R6-AC6：退出码取值不变，且不引入新的退出码取值
# ===========================================================================


def test_exit_code_values_unchanged() -> None:
    """退出码常量取值不变：成功 0 / 运行期失败 1 / 用法或配置错误 2（R6-AC6）。

    **Validates: Requirements 6.6**
    """
    assert EXIT_OK == 0
    assert EXIT_RUNTIME_FAILURE == 1
    assert EXIT_USAGE_OR_CONFIG == 2


def test_ego_error_exit_codes_subset_of_fixed_values() -> None:
    """全部 ``EgoError`` 子类的退出码 ⊆ ``{0, 1, 2}``——不引入新取值（R6-AC6）。

    遍历模块内全部 ``EgoError`` 子类，断言其 ``exit_code`` 落在既有取值集合内，
    从而保证本特性未引入任何新的退出码常量。

    **Validates: Requirements 6.6**
    """
    fixed_values = {EXIT_OK, EXIT_RUNTIME_FAILURE, EXIT_USAGE_OR_CONFIG}
    subclasses = {EgoError, *_all_subclasses(EgoError)}

    for cls in subclasses:
        assert cls.exit_code in fixed_values, (
            f"{cls.__name__}.exit_code={cls.exit_code} 不在固定取值集合 {fixed_values} 内"
        )


def test_config_error_maps_to_two_others_to_one() -> None:
    """``ConfigError`` → 2（用法/配置错误），其余受控错误 → 1（运行期失败）（R6-AC6）。

    **Validates: Requirements 6.6**
    """
    assert ConfigError.exit_code == EXIT_USAGE_OR_CONFIG  # 2

    for cls in (
        EgoError,
        CollectorError,
        CompilerError,
        TrainerError,
        VaultError,
        RuntimeError_,
    ):
        assert cls.exit_code == EXIT_RUNTIME_FAILURE, (  # 1
            f"{cls.__name__} 应映射到退出码 1"
        )


def _all_subclasses(cls: type) -> set[type]:
    """递归收集 ``cls`` 的全部子类。"""

    result: set[type] = set()
    for sub in cls.__subclasses__():
        result.add(sub)
        result |= _all_subclasses(sub)
    return result


# ===========================================================================
# R6-AC1 / R6-AC2：扁平兼容（manifest 五字段 + Flat 退化为单 Kernel Cell）
# ===========================================================================


def test_build_manifest_grid_none_byte_identical_to_no_grid() -> None:
    """``build_manifest(grid=None)`` 与不传 ``grid`` 逐字节一致且为五字段（R6-AC1）。

    固定 ``version`` / ``created_at`` 以消除时钟/版本差异，断言两种调用方式产出的
    manifest 序列化后逐字节一致、含且仅含五个旧字段、不含 ``grid`` 键。

    **Validates: Requirements 6.1**
    """
    kwargs = dict(
        training_backend="peft-cpu",
        version="0.1.0",
        created_at="2025-01-01T00:00:00Z",
    )
    without_grid = build_manifest(**kwargs)
    with_grid_none = build_manifest(**kwargs, grid=None)

    # 五字段、无 grid 键。
    assert list(without_grid.keys()) == [
        "version",
        "created_at",
        "base_model",
        "editions",
        "training_backend",
    ]
    assert "grid" not in without_grid
    assert "grid" not in with_grid_none

    # 逐字节一致（序列化后比较，保留键顺序）。
    assert json.dumps(without_grid, ensure_ascii=False) == json.dumps(
        with_grid_none, ensure_ascii=False
    )


def test_parse_grid_descriptor_none_returns_single_kernel_flat_grid() -> None:
    """``parse_grid_descriptor(None)`` 还原单 Kernel Cell 的 Flat 网格（R6-AC1/AC2）。

    无 ``grid`` 键的 Flat_Package 被视为「单个 Kernel Cell」：解析得到的 :class:`Grid`
    恰含一个 Kernel，Kernel 相对路径为 ``adapters/ego-lora.safetensors``，无任何分区。

    **Validates: Requirements 6.1, 6.2**
    """
    grid = parse_grid_descriptor(None)

    assert isinstance(grid, Grid)
    assert grid.kernel.cell == KERNEL
    assert grid.kernel.relpath == FLAT_KERNEL_RELPATH == "adapters/ego-lora.safetensors"
    assert grid.zones == {}


def test_select_cells_on_flat_grid_is_kernel_only() -> None:
    """``select_cells(flat_grid, None) == [KERNEL]``（退化包仅加载单 Kernel，R6-AC1/AC2）。

    **Validates: Requirements 6.1, 6.2**
    """
    grid = parse_grid_descriptor(None)

    assert select_cells(grid, None) == [KERNEL]


# ===========================================================================
# R7-AC1（轻量）：本特性新增模块均位于 release/dev 目录树下
# ===========================================================================


def _contains_release_dev(path: Path) -> bool:
    """路径中是否含连续的 ``release/dev`` 两级目录（跨平台稳健判定）。"""

    parts = path.resolve().parts
    for i in range(len(parts) - 1):
        if parts[i] == "release" and parts[i + 1] == "dev":
            return True
    return False


@pytest.mark.parametrize("module_name", _FEATURE_MODULES)
def test_feature_modules_live_under_release_dev(module_name: str) -> None:
    """每个新增模块的 ``__file__`` 路径均位于 ``release/dev/`` 目录树下（R7-AC1）。

    **Validates: Requirements 7.1**
    """
    import importlib

    module = importlib.import_module(module_name)
    module_path = Path(module.__file__)

    assert _contains_release_dev(module_path), (
        f"{module_name} 不在 release/dev 目录树下：{module_path}"
    )
