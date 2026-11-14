"""二维网格新增模块的导入安全单元测试（任务 7.2）。

覆盖设计文档「Testing Strategy」与下列需求：

* **R6-AC8**：仅安装默认依赖（不含任何 Training_Dependencies extras）的环境
  导入 ``ego_pod`` 包及本特性新增/扩展的 Compiler / Runtime / Vault / Trainer
  子模块，顶层 ``import`` 均不抛错。
* **R7-AC2**：二维网格所需的全部机器学习依赖保持为**惰性导入**或可选 extras，
  即被测模块在**模块作用域（top-level）**不得 ``import torch`` / ``peft`` /
  ``transformers`` 等训练依赖（一律收敛到函数内惰性导入）。

实现说明
--------
* 第一组测试以 :func:`importlib.import_module` 参数化逐一导入各模块，断言不抛错。
  此为「能导入」的运行期证据，但其环境敏感性在于：若 ``peft`` 恰好已安装，
  即便某模块在顶层惰性导入失败也可能被掩盖。
* 第二组测试为**环境无关**的核心断言：读取每个被测模块自身的源码文件，用
  :mod:`ast` 仅遍历**模块顶层**（不深入函数/方法体内）的 ``Import`` /
  ``ImportFrom`` 节点，断言其中没有任何训练依赖（torch/peft/transformers/
  bitsandbytes/unsloth/mlx）。这才是 R7-AC2 惰性导入纪律的可判定证据，且不受
  运行环境是否安装了 ``peft`` 影响（故意不复刻 real-trainer 套件中那条
  ``test_training_deps_actually_absent`` 的「依赖确实缺失」断言）。
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# 共享常量
# ---------------------------------------------------------------------------

#: 本特性新增/扩展、必须在无任何 Training_Dependencies 的环境下也能安全导入的模块
#: （R6-AC8）。涵盖网格的纯逻辑新模块与接入网格的既有编排模块。
_IMPORT_SAFE_MODULES = [
    "ego_pod",
    "ego_pod.compiler.routing",
    "ego_pod.runtime.selection",
    "ego_pod.vault.grid_descriptor",
    "ego_pod.trainer.cells",
    "ego_pod.trainer.grid",
    "ego_pod.compiler.pipeline",
    "ego_pod.vault.pipeline",
    "ego_pod.vault.package",
    "ego_pod.runtime.inference",
]

#: 被测模块在**模块顶层**作用域均不得导入的训练依赖（R7-AC2）。
#: 比对时按顶级包名（``a.b.c`` 的 ``a``）归一，故 ``mlx_lm`` / ``mlx.core`` 等
#: 子模块亦被覆盖。
_FORBIDDEN_TOP_LEVEL_IMPORTS = frozenset(
    {
        "torch",
        "peft",
        "transformers",
        "bitsandbytes",
        "unsloth",
        "mlx",
    }
)


def _top_package(dotted_name: str) -> str:
    """取点分模块名的顶级包名（``torch.nn`` → ``torch``）。"""

    return dotted_name.split(".", 1)[0]


def _module_source_path(module_name: str) -> Path:
    """导入模块并解析其源码文件路径（经 ``__file__``）。"""

    module = importlib.import_module(module_name)
    assert getattr(module, "__file__", None), f"{module_name} 无 __file__，无法定位源码"
    return Path(module.__file__).resolve()


def _top_level_imported_packages(source_path: Path) -> set[str]:
    """解析源码，返回**仅模块顶层**导入语句涉及的顶级包名集合。

    通过遍历 :class:`ast.Module` 的直接子节点（``tree.body``）而非递归全树，
    确保只检查模块作用域的 ``import`` / ``from ... import``，排除函数 / 方法 /
    类方法体内的惰性导入。
    """

    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    packages: set[str] = set()
    for node in tree.body:  # 仅模块顶层语句，不深入函数/方法体。
        if isinstance(node, ast.Import):
            for alias in node.names:
                packages.add(_top_package(alias.name))
        elif isinstance(node, ast.ImportFrom):
            # 仅绝对导入（level == 0）才可能引入第三方训练依赖；
            # 相对导入（``from . import x``）的 module 可能为 None，安全跳过。
            if node.level == 0 and node.module:
                packages.add(_top_package(node.module))
    return packages


# ---------------------------------------------------------------------------
# R6-AC8：导入安全 —— 各模块顶层导入不抛错
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module_name", _IMPORT_SAFE_MODULES)
def test_module_imports_without_training_deps(module_name: str):
    """每个网格相关模块的导入均不抛错（R6-AC8）。"""

    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - 失败即测试目的
        pytest.fail(f"导入 {module_name} 抛出异常：{exc!r}")
    assert module is not None


# ---------------------------------------------------------------------------
# R7-AC2：惰性导入纪律 —— 模块顶层不得导入任何训练依赖（环境无关）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module_name", _IMPORT_SAFE_MODULES)
def test_module_has_no_top_level_training_imports(module_name: str):
    """每个网格相关模块在**模块作用域**均不导入任何训练依赖（R7-AC2）。

    这是导入安全的实质性、环境无关的断言：训练依赖必须收敛到函数内惰性导入，
    故其顶级包名不应出现在模块顶层 ``import`` 语句中。
    """

    source_path = _module_source_path(module_name)
    top_level = _top_level_imported_packages(source_path)
    offending = sorted(top_level & _FORBIDDEN_TOP_LEVEL_IMPORTS)
    assert not offending, (
        f"{module_name}（{source_path}）在模块顶层导入了训练依赖 {offending}，"
        f"应改为函数内惰性导入以满足导入安全（R7-AC2）"
    )
