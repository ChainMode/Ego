"""导入安全、Protocol 契约与 ``pyproject.toml`` extras 的单元测试（任务 12.4）。

覆盖设计文档「Testing Strategy」与下列需求：

* **R8-AC4**：仅安装默认依赖（不含任何 Training_Dependencies extras）的环境
  导入 ``ego_pod`` 包及其 Trainer / Runtime 子模块，顶层 ``import`` 均不抛错。
* **R10-AC1**：全部后端（Stub / PEFT-CUDA / PEFT-CPU / Unsloth / MLX）仍满足
  ``TrainingBackend`` Protocol 形状不变（``runtime_checkable`` 结构化检查），
  且各暴露稳定的 :data:`Backend_Identifier`。
* **R10-AC3**：本特性全部源代码、测试与配置均位于 ``release/dev/`` 目录下
  （以 ``pyproject.toml`` 的解析路径轻量校验）。
* **R10-AC4**：Unsloth / PEFT / MLX 及其支撑的 Training_Dependencies 均声明为
  ``pyproject.toml`` 的可选 extras，默认 ``dependencies`` 不引入任何训练依赖。

本文件刻意**不**在模块顶层导入任何 ML 依赖（``peft`` / ``transformers`` /
``bitsandbytes`` / ``unsloth`` / ``mlx`` 等）。被测模块的可选依赖一律以
``try/except ImportError`` 哨兵 + 函数内惰性导入包裹，故本套测试可在不安装任何
Training_Dependencies 的环境（如 CI）中运行（R8-AC4/AC5）。
"""

from __future__ import annotations

import importlib
import importlib.util
import re
import tomllib
from pathlib import Path

import pytest

from ego_pod.trainer.base import TrainingBackend
from ego_pod.trainer.mlx_backend import MlxBackend
from ego_pod.trainer.peft_backend import PeftCpuBackend, PeftCudaBackend
from ego_pod.trainer.stub import StubBackend
from ego_pod.trainer.unsloth_backend import UnslothBackend


# ---------------------------------------------------------------------------
# 共享常量与辅助
# ---------------------------------------------------------------------------

#: 必须在无任何 Training_Dependencies 的环境下也能安全导入的模块（R8-AC4）。
_IMPORT_SAFE_MODULES = [
    "ego_pod",
    "ego_pod.trainer.sft",
    "ego_pod.trainer.loop",
    "ego_pod.trainer.generation",
    "ego_pod.trainer.peft_backend",
    "ego_pod.trainer.unsloth_backend",
    "ego_pod.trainer.mlx_backend",
    "ego_pod.trainer.pipeline",
    "ego_pod.runtime.inference",
]

#: Backend_Identifier 的固定取值集合（R7-AC1、R7-AC4）。
_VALID_IDENTIFIERS = {"unsloth", "peft-cuda", "mlx", "peft-cpu", "stub"}

#: ``TrainingBackend`` Protocol 要求暴露的成员（属性 + 方法）（R10-AC1）。
_PROTOCOL_MEMBERS = (
    "name",
    "hardware",
    "identifier",
    "is_available",
    "load_base_model",
    "train",
    "export_lora_safetensors",
)

#: 默认 ``dependencies`` 中**不得**出现的 Training_Dependencies（R8-AC4、R10-AC4）。
_TRAINING_DEPS = (
    "peft",
    "transformers",
    "bitsandbytes",
    "accelerate",
    "datasets",
    "unsloth",
    "mlx-lm",
)


def _backend_cases() -> list[tuple[str, type, str]]:
    """返回 ``(可读名, 后端类, 预期 identifier)`` 三元组列表。"""

    return [
        ("StubBackend", StubBackend, "stub"),
        ("PeftCudaBackend", PeftCudaBackend, "peft-cuda"),
        ("PeftCpuBackend", PeftCpuBackend, "peft-cpu"),
        ("UnslothBackend", UnslothBackend, "unsloth"),
        ("MlxBackend", MlxBackend, "mlx"),
    ]


def _find_release_dev_dir() -> Path:
    """从本测试文件向上回溯，定位 ``release/dev`` 目录（R10-AC3）。"""

    for parent in Path(__file__).resolve().parents:
        if parent.name == "dev" and parent.parent.name == "release":
            return parent
    raise AssertionError("未能从测试文件向上定位到 release/dev 目录")


def _requirement_name(spec: str) -> str:
    """从依赖声明（如 ``peft>=0.10``）提取规范化的包名（小写）。"""

    match = re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]*", spec.strip())
    assert match is not None, f"无法解析依赖声明：{spec!r}"
    return match.group(0).lower()


def _load_pyproject() -> tuple[dict, Path]:
    """解析 ``release/dev/pyproject.toml`` 并返回 ``(数据, 路径)``。"""

    path = _find_release_dev_dir() / "pyproject.toml"
    assert path.is_file(), f"pyproject.toml 不存在：{path}"
    with path.open("rb") as fh:
        return tomllib.load(fh), path


# ---------------------------------------------------------------------------
# R8-AC4：导入安全 —— 无 Training_Dependencies 时顶层导入不抛错
# ---------------------------------------------------------------------------


class _BlockTrainingDepsFinder:
    """``sys.meta_path`` 拦截器：在已安装 ML 依赖的开发机上模拟无训练依赖环境。"""

    BLOCKED_PREFIXES = (
        "peft",
        "transformers",
        "bitsandbytes",
        "accelerate",
        "datasets",
        "unsloth",
        "mlx",
        "mlx_lm",
        "torch",
    )

    def find_spec(self, fullname: str, path=None, target=None):  # noqa: ANN001
        top = fullname.split(".", 1)[0]
        if top in self.BLOCKED_PREFIXES:
            raise ModuleNotFoundError(
                f"Simulated missing optional training dependency: {fullname}"
            )
        return None


@pytest.mark.parametrize("module_name", _IMPORT_SAFE_MODULES)
def test_module_imports_without_training_deps(
    module_name: str, monkeypatch: pytest.MonkeyPatch
):
    """每个模块的导入在屏蔽全部 Training_Dependencies 时均不抛错（R8-AC4）。"""
    import sys

    blocker = _BlockTrainingDepsFinder()
    monkeypatch.setattr(sys, "meta_path", [blocker, *sys.meta_path])
    for loaded_name in list(sys.modules):
        top = loaded_name.split(".", 1)[0]
        if top in blocker.BLOCKED_PREFIXES or top == "ego_pod":
            monkeypatch.delitem(sys.modules, loaded_name, raising=False)

    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - 失败即测试目的
        pytest.fail(f"导入 {module_name} 抛出异常：{exc!r}")
    assert module is not None


def test_training_deps_blocker_enforces_absence(monkeypatch: pytest.MonkeyPatch):
    """确认隔离拦截器下全部 Training_Dependencies 均不可导入（R8-AC4）。

    无论宿主机 Python 环境是否预装了 ``peft`` / ``transformers`` 等包，
    上方导入安全测试均在屏蔽全部训练依赖的前提下重导 ``ego_pod`` 各子模块。
    """
    import sys

    blocker = _BlockTrainingDepsFinder()
    monkeypatch.setattr(sys, "meta_path", [blocker, *sys.meta_path])
    for loaded_name in list(sys.modules):
        if loaded_name.split(".", 1)[0] in blocker.BLOCKED_PREFIXES:
            monkeypatch.delitem(sys.modules, loaded_name, raising=False)

    for dep in ("peft", "transformers", "bitsandbytes", "unsloth", "mlx"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(dep)


# ---------------------------------------------------------------------------
# R10-AC1：全部后端满足 TrainingBackend Protocol（runtime_checkable）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,backend_cls,expected_id",
    _backend_cases(),
    ids=[case[0] for case in _backend_cases()],
)
def test_backend_satisfies_protocol(label: str, backend_cls: type, expected_id: str):
    """实例化每个后端并断言其结构上满足 ``TrainingBackend`` Protocol（R10-AC1）。"""

    instance = backend_cls()
    # runtime_checkable 的结构化类型检查。
    assert isinstance(instance, TrainingBackend)


@pytest.mark.parametrize(
    "label,backend_cls,expected_id",
    _backend_cases(),
    ids=[case[0] for case in _backend_cases()],
)
def test_backend_exposes_protocol_members(
    label: str, backend_cls: type, expected_id: str
):
    """每个后端暴露 Protocol 要求的全部属性与方法（R10-AC1）。"""

    instance = backend_cls()
    for member in _PROTOCOL_MEMBERS:
        assert hasattr(instance, member), f"{label} 缺少 Protocol 成员 {member!r}"
    # 三个训练方法与 is_available 均可调用。
    for method in (
        "is_available",
        "load_base_model",
        "train",
        "export_lora_safetensors",
    ):
        assert callable(getattr(instance, method)), f"{label}.{method} 不可调用"


@pytest.mark.parametrize(
    "label,backend_cls,expected_id",
    _backend_cases(),
    ids=[case[0] for case in _backend_cases()],
)
def test_backend_identifier_in_literal_set(
    label: str, backend_cls: type, expected_id: str
):
    """每个后端的 ``identifier`` 属于固定取值集合且等于文档化取值（R7-AC1/AC4）。"""

    instance = backend_cls()
    assert instance.identifier in _VALID_IDENTIFIERS
    assert instance.identifier == expected_id


def test_all_backend_identifiers_are_distinct():
    """五个后端的 ``identifier`` 两两不同，覆盖整个固定取值集合。"""

    identifiers = {cls().identifier for _, cls, _ in _backend_cases()}
    assert identifiers == _VALID_IDENTIFIERS


# ---------------------------------------------------------------------------
# R10-AC4：pyproject 可选依赖分组完整且默认依赖不含训练依赖
# ---------------------------------------------------------------------------


def test_optional_dependency_groups_present():
    """``[project.optional-dependencies]`` 含全部预期 extras 分组（R10-AC4）。"""

    data, _ = _load_pyproject()
    optional = data["project"]["optional-dependencies"]
    for group in ("peft", "bitsandbytes", "cuda", "unsloth", "mlx", "all", "test"):
        assert group in optional, f"缺少可选依赖分组：{group!r}"


def test_default_dependencies_exclude_training_deps():
    """默认 ``[project].dependencies`` 不含任何 Training_Dependencies（R8-AC4、R10-AC4）。"""

    data, _ = _load_pyproject()
    base_names = {_requirement_name(spec) for spec in data["project"]["dependencies"]}
    for dep in _TRAINING_DEPS:
        assert dep not in base_names, f"默认 dependencies 不应包含训练依赖 {dep!r}"


def test_cuda_group_is_peft_plus_bitsandbytes():
    """``cuda`` 一键组等于 ``peft`` 组叠加 ``bitsandbytes``（R10-AC4）。"""

    data, _ = _load_pyproject()
    optional = data["project"]["optional-dependencies"]
    expected = set(optional["peft"]) | set(optional["bitsandbytes"])
    assert set(optional["cuda"]) == expected


def test_all_group_is_superset_of_individual_groups():
    """``all`` 组是各独立后端分组的超集（R10-AC4）。"""

    data, _ = _load_pyproject()
    optional = data["project"]["optional-dependencies"]
    union = (
        set(optional["peft"])
        | set(optional["bitsandbytes"])
        | set(optional["unsloth"])
        | set(optional["mlx"])
    )
    assert union <= set(optional["all"])


# ---------------------------------------------------------------------------
# R10-AC3：所有改动文件均位于 release/dev 目录下
# ---------------------------------------------------------------------------


def test_pyproject_resolves_under_release_dev():
    """解析出的 ``pyproject.toml`` 路径位于 ``release/dev`` 目录下（R10-AC3）。"""

    _, path = _load_pyproject()
    assert path.parent.name == "dev"
    assert path.parent.parent.name == "release"


def test_this_test_file_resolves_under_release_dev():
    """本测试文件自身亦位于 ``release/dev`` 目录树下（R10-AC3）。"""

    release_dev = _find_release_dev_dir()
    assert str(Path(__file__).resolve()).startswith(str(release_dev))
