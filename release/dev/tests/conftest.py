"""pytest 共享配置与 fixture。

为支持在未执行 ``pip install`` 的环境（如 CI 首次检出、本地开发）直接运行
``pytest``，本模块在测试收集前把 ``release/dev/src`` 加入 ``sys.path``，使
``import ego_pod`` 无需 ``PYTHONPATH=src`` 亦可解析（采用 src-layout 时
源码不在仓库根，需显式加入）。已通过 ``pip install ./release/dev`` 安装时，
此插入为幂等的无害操作。
"""

from __future__ import annotations

import sys
from pathlib import Path

# tests/ 的同级目录 src/ 为包根（src-layout）。
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir():
    src_str = str(_SRC)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

# 把 tests/ 目录加入 sys.path，使共享测试替身模块 ``tests/_doubles.py`` 可作为
# 顶层模块 ``import _doubles`` 被任意子目录下的测试（property/ unit/ integration/）
# 复用，而不依赖 pytest 的具体 import 模式。此插入为幂等的无害操作。
_TESTS_DIR = Path(__file__).resolve().parent
if _TESTS_DIR.is_dir():
    tests_str = str(_TESTS_DIR)
    if tests_str not in sys.path:
        sys.path.insert(0, tests_str)
