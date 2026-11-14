"""`pip install ./release/dev` 安装集成测试（任务 12.5）。

本文件验证 Requirement 13（代码物理路径与开发分支约束）的验收标准 **R13-AC3**：

    WHEN 用户在仓库根目录执行 ``pip install ./release/dev``，THE Ego_Pod SHALL
    完成安装并使该 pip 子进程以退出码 0 结束……本条所定义的"安装成功"判定标准
    **严格等同于** "pip 子进程的退出码等于 0"，不依赖也不要求验证安装产物的实际
    完整性、功能性或是否真正完成所有安装步骤——只要 pip 子进程退出码为 0，即视为
    本条满足。

判定标准（严格遵循 R13-AC3 与设计文档 Testing Strategy 中的 ``test_pip_install``）：

- **唯一通过条件**：``pip install`` 子进程的 ``returncode == 0``。
- **不做** 任何额外断言：不导入已安装的包、不调用 ``ego`` 可执行入口、不检查
  site-packages 中的产物完整性、不验证 console_script 是否注册。这些"功能性
  完整性"验证由后续真实端到端测试（如 ``test_train_stub`` / ``test_run_repl``）
  承担，与本条严格分离。

工程化取舍（在保持"真实执行 ``pip install ./release/dev``"语义的前提下让测试可在
CI 中稳定、快速运行）：

- 在 ``sys.executable -m venv`` 创建的**全新临时虚拟环境**中执行安装，避免污染
  当前解释器环境，并真实走一遍构建（PEP 517 / setuptools）+ 安装流程。
- 使用 ``--no-deps`` 安装：本包的运行时依赖含 ``torch``（数 GB、下载常需 25 分钟
  以上），若拉取完整依赖将使该测试在 CI 中不可用。R13-AC3 的判定标准只关心
  "本 **包** 是否能完成构建与安装并令 pip 退出码为 0"，即 sdist/wheel 构建、
  元数据生成、console_script 注册等步骤是否成功，而非 ``torch`` 能否下载。因此
  ``--no-deps`` 既忠实于"真实安装本包"语义，又避免无关的巨型下载主导测试时长。
  注意：此处 **未** 使用 ``--no-build-isolation``——保留默认的构建隔离，使 pip
  按 ``[build-system].requires``（``setuptools>=61``）准备构建环境，从而真正
  执行包的构建步骤。
- 给 pip 子进程设置较宽裕的超时，仅在 venv 创建从根本上不可用时 ``pytest.skip``。
"""

from __future__ import annotations

import subprocess
import sys
import venv
from pathlib import Path

import pytest

# ``release/dev`` 目录：本测试文件位于 ``release/dev/tests/integration/``，
# 因此其上溯三级父目录即为待安装的包根（含 pyproject.toml）。
RELEASE_DEV_DIR = Path(__file__).resolve().parents[2]

# pip 子进程超时（秒）。``--no-deps`` 下仅需构建本包并按需为构建隔离准备
# setuptools，通常远快于此；给出宽裕上限以容忍较慢的 CI 与构建隔离网络。
PIP_INSTALL_TIMEOUT_SECONDS = 900


def _venv_python(venv_dir: Path) -> Path:
    """返回临时虚拟环境中的 Python 解释器绝对路径（跨平台）。"""
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


@pytest.mark.slow
def test_pip_install_release_dev_exit_code_zero(tmp_path):
    """``pip install ./release/dev`` 子进程退出码必须为 0（R13-AC3）。

    判定标准严格等同于 pip 子进程 ``returncode == 0``；不验证安装产物完整性。
    """
    # 前置健全性检查：包根存在且含 pyproject.toml（不属于 R13-AC3 判定本身，
    # 仅用于在路径解析出错时给出清晰的失败信息）。
    assert RELEASE_DEV_DIR.is_dir(), f"release/dev 目录不存在: {RELEASE_DEV_DIR}"
    assert (RELEASE_DEV_DIR / "pyproject.toml").is_file(), (
        f"release/dev 下缺少 pyproject.toml: {RELEASE_DEV_DIR}"
    )

    # 1) 创建全新临时虚拟环境（含 pip）。若 venv 创建从根本上不可用则跳过。
    venv_dir = tmp_path / "install_venv"
    try:
        venv.create(venv_dir, with_pip=True, clear=True)
    except Exception as exc:  # pragma: no cover - 环境性跳过
        pytest.skip(f"无法创建虚拟环境，跳过安装测试: {exc}")

    venv_python = _venv_python(venv_dir)
    if not venv_python.exists():  # pragma: no cover - 环境性跳过
        pytest.skip(f"虚拟环境未提供 Python 解释器，跳过安装测试: {venv_python}")

    # 2) 在该虚拟环境中执行 ``pip install ./release/dev``。
    #    使用 ``--no-deps`` 避免拉取巨型 torch 等运行时依赖（见模块 docstring），
    #    仅安装本包自身，真实执行 PEP 517 构建 + 安装流程。
    cmd = [
        str(venv_python),
        "-m",
        "pip",
        "install",
        "--no-deps",
        str(RELEASE_DEV_DIR),
    ]
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=PIP_INSTALL_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:  # pragma: no cover - 环境性失败
        pytest.fail(
            f"pip install 子进程在 {PIP_INSTALL_TIMEOUT_SECONDS}s 内未完成: {exc}"
        )

    # 3) 唯一判定标准：pip 子进程退出码 == 0（R13-AC3）。
    #    失败时附带 pip 输出便于排查，但不对产物完整性做任何额外断言。
    assert completed.returncode == 0, (
        "pip install ./release/dev 退出码非 0，"
        f"实际为 {completed.returncode}\n"
        f"--- 命令 ---\n{' '.join(cmd)}\n"
        f"--- stdout ---\n{completed.stdout}\n"
        f"--- stderr ---\n{completed.stderr}"
    )
