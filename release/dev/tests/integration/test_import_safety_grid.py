"""新增模块导入安全的集成测试（任务 8.5，ego-layered-adapter）。

本文件验证 Ego Layered Adapter 在 ego-pod-mvp / ego-pod-real-trainer 之上**新增**的
二维稀疏网格（Sparse_Grid）模块的**导入安全**，覆盖下列需求：

* **R6-AC8**：仅安装默认依赖（不含任何 Training_Dependencies extras）的环境
  导入 ``ego_pod`` 包及其新增子模块，顶层 ``import`` 均不抛错。
* **R7-AC2**：把二维网格训练所需的全部机器学习依赖保持为惰性导入或可选 extras，
  使默认安装与 ``import ego_pod`` 不引入这些依赖。

本特性把 ML 依赖（``torch`` / ``peft`` / ``transformers`` 等）一律收敛到**函数内
惰性导入**，故下列**纯逻辑/数据结构**模块的顶层导入在任何环境下都不应需要任何
Training_Dependencies：

* :mod:`ego_pod.compiler.routing` —— ``CellKey`` / ``ResolvedZoneConfig`` /
  ``route_cells``（纯函数路由）。
* :mod:`ego_pod.runtime.selection` —— ``Grid`` / ``LoadSelection`` /
  ``select_cells``（纯函数加载选择）。
* :mod:`ego_pod.vault.grid_descriptor` —— ``build_grid_descriptor`` /
  ``parse_grid_descriptor``（往返对称）。
* :mod:`ego_pod.trainer.cells` —— ``CellSpec`` / ``TrainedCell`` /
  ``GridTrainingResult``（纯数据结构）。
* :mod:`ego_pod.trainer.grid` —— ``run_grid_training``（编排层，ML 依赖经其调用的
  后端在函数内惰性导入）。

设计取向（与 ``tests/unit/test_import_safety_and_extras.py`` 一致，但**刻意不**复制
其 ``test_training_deps_actually_absent`` 的"训练依赖必须缺失"断言）：本环境可能**已**
安装 ``peft`` / ``torch``，因此本文件**不**断言这些依赖缺失。其要点是"**导入这些
模块本身**不会触发/要求 ``torch`` / ``peft`` 的导入"。为此采用两层稳健、不依赖环境
状态的断言：

1. **导入成功**（主断言）：``importlib.import_module`` 每个模块均返回模块对象且不抛错。
2. **源码无顶层 ML 导入**（AST 静态检查）：解析每个**新增**模块的源文件，断言其
   模块顶层（module body）不含 ``import torch`` / ``import peft`` / ``import
   transformers`` 之类语句——直接验证"ML 依赖惰性导入"的实现约定，且完全不依赖
   运行环境是否装有这些依赖（故非脆弱）。

本文件**不**在模块顶层导入任何 ML 依赖，可在仅装 ``pytest`` / ``hypothesis`` 的最小
环境中运行（R6-AC8 / R7-AC2）。
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# 共享常量
# ---------------------------------------------------------------------------

#: 本特性（ego-layered-adapter）**新增**的、必须在无任何 Training_Dependencies 的
#: 环境下也能安全导入的纯逻辑/数据结构模块（R6-AC8、R7-AC2）。
_NEW_GRID_MODULES = [
    "ego_pod.compiler.routing",
    "ego_pod.runtime.selection",
    "ego_pod.vault.grid_descriptor",
    "ego_pod.trainer.cells",
    "ego_pod.trainer.grid",
]

#: 既有（MVP / real-trainer）模块——再次确认本特性的扩展未破坏其顶层导入安全。
_EXISTING_IMPORT_SAFE_MODULES = [
    "ego_pod",
    "ego_pod.compiler.pipeline",
    "ego_pod.vault.pipeline",
    "ego_pod.vault.package",
    "ego_pod.runtime.inference",
    "ego_pod.commands.train",
    "ego_pod.commands.run",
]

#: 全部待断言"顶层导入不抛错"的模块。
_ALL_IMPORT_SAFE_MODULES = _NEW_GRID_MODULES + _EXISTING_IMPORT_SAFE_MODULES

#: 一律应**惰性导入**、不得出现于模块顶层 import 语句的 Training_Dependencies 顶层包名。
_FORBIDDEN_TOPLEVEL_IMPORTS = frozenset(
    {
        "torch",
        "peft",
        "transformers",
        "bitsandbytes",
        "accelerate",
        "datasets",
        "unsloth",
        "mlx",
        "mlx_lm",
        "safetensors",
    }
)


# ---------------------------------------------------------------------------
# 辅助：定位模块源文件并提取其"顶层 import 的顶级包名"
# ---------------------------------------------------------------------------


def _module_source_path(module_name: str) -> Path:
    """返回模块对应源文件的绝对路径（不导入该模块即可定位）。"""

    spec = importlib.util.find_spec(module_name)
    assert spec is not None, f"无法定位模块 {module_name!r} 的导入规格"
    assert spec.origin is not None, f"模块 {module_name!r} 无源文件 origin"
    return Path(spec.origin)


def _toplevel_imported_root_packages(source_path: Path) -> set[str]:
    """解析源文件 AST，返回其**模块顶层**（module body）导入的顶级包名集合。

    仅检查模块体（``tree.body``）中的 ``import`` / ``from ... import`` 语句，
    因此**函数内/方法内的惰性导入不计入**——这正是要区分的：惰性导入合法，顶层
    导入 ML 依赖才违反"导入安全"约定。``from __future__ import`` 与相对导入
    （``from . import``，``node.level > 0``）一律忽略。
    """

    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    roots: set[str] = set()
    for node in tree.body:  # 仅模块顶层语句；不递归进入函数/类体
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue  # 相对导入（包内），非第三方依赖
            if node.module:
                roots.add(node.module.split(".", 1)[0])
    return roots


# ---------------------------------------------------------------------------
# R6-AC8 / R7-AC2：顶层导入不抛错（主断言）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module_name", _ALL_IMPORT_SAFE_MODULES)
def test_module_imports_without_error(module_name: str):
    """每个模块的 ``importlib.import_module`` 均返回模块对象且不抛错。

    本环境可能已装有 ``peft`` / ``torch``，但**导入这些模块本身**不应触发任何错误，
    无论 Training_Dependencies 是否存在（R6-AC8、R7-AC2）。
    """

    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - 失败即测试目的
        pytest.fail(f"导入 {module_name} 抛出异常：{exc!r}")
    assert module is not None


# ---------------------------------------------------------------------------
# R7-AC2：新增模块源码无顶层 ML 导入（AST 静态检查，不依赖环境状态）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module_name", _NEW_GRID_MODULES)
def test_new_module_has_no_toplevel_ml_imports(module_name: str):
    """新增网格模块的源文件在**模块顶层**不导入任何 Training_Dependencies。

    直接验证"ML 依赖一律函数内惰性导入"的实现约定（R7-AC2）。该断言基于源码 AST，
    完全不依赖运行环境是否装有 ``torch`` / ``peft`` 等，故稳健、非脆弱。
    """

    source_path = _module_source_path(module_name)
    toplevel_roots = _toplevel_imported_root_packages(source_path)
    leaked = sorted(toplevel_roots & _FORBIDDEN_TOPLEVEL_IMPORTS)
    assert not leaked, (
        f"{module_name} 在模块顶层导入了应惰性导入的 Training_Dependencies：{leaked}"
        f"（源文件：{source_path}）"
    )


def test_new_grid_modules_are_importable_as_a_set():
    """一次性导入全部新增网格模块均成功，且彼此可共存（无导入期副作用冲突）。"""

    modules = [importlib.import_module(name) for name in _NEW_GRID_MODULES]
    assert all(m is not None for m in modules)
    assert len(modules) == len(_NEW_GRID_MODULES)
