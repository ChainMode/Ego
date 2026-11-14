"""`ego run` 的单元测试（任务 10.4）。

本文件覆盖 Requirement 12（``ego run`` 加载与对话）的验收标准，分两层验证：

**命令入口层**（:func:`ego_pod.commands.run.run_repl`，错误以
:class:`~ego_pod.exit_codes.RuntimeError_` 上抛、退出码 1）：

- **R12-AC4**：``--edition`` 取值非 ``normal`` → ``RuntimeError_``（退出码 1，
  文案含"仅支持 ``normal``"）。
- **R12-AC2**：未提供 ``--edition``（``edition=None``）默认按 ``normal`` 处理。
- **R12-AC3**：``ego-output/<edition>/`` 子目录不存在 → ``RuntimeError_``（退出码
  1，文案含"请先"/``ego train`` 指引）。
- **R12-AC5**：Edition 子目录存在但 ``config.json`` / ``adapters/ego-lora.safetensors``
  缺失或损坏 → ``RuntimeError_``（退出码 1）。
- **R12-AC9 / AC6 / AC8**：在有效 stub Ego_Package 上驱动整段 REPL——打印 Stub
  模式提示、对 ``hello`` 返回固定占位回复、空行不触发推理、``:quit`` 与 EOF 均以
  退出码 0 结束。

**REPL 行为层**（:func:`ego_pod.runtime.repl.run_repl`，注入计数桩生成器）：

- **R12-AC6**：每条非空输入恰调用一次响应生成并打印其回复。
- **R12-AC7**：空行/仅空白行不调用推理。
- **R12-AC8**：``:quit`` 与 EOF 均释放资源并以退出码 0 结束。

测试不依赖网络或真实 ML 依赖：Ego_Package 夹具用 :func:`safetensors.numpy.save_file`
写入 1×1 占位张量，REPL I/O 用 :class:`io.StringIO` 注入。
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from ego_pod.commands.run import (
    EDITION_NOT_SUPPORTED_MESSAGE,
    PACKAGE_NOT_FOUND_MESSAGE,
    run_repl,
)
from ego_pod.exit_codes import EXIT_OK, EXIT_RUNTIME_FAILURE, RuntimeError_
from ego_pod.runtime.inference import (
    ADAPTER_RELPATH,
    CONFIG_FILENAME,
    DEFAULT_EDITION,
    MANIFEST_FILENAME,
    OUTPUT_DIRNAME,
    STUB_MODE_STARTUP_NOTICE,
    STUB_PLACEHOLDER_REPLY,
)
from ego_pod.runtime.repl import PROMPT, QUIT_COMMAND
from ego_pod.runtime.repl import run_repl as run_repl_loop


# ===========================================================================
# 夹具构造工具
# ===========================================================================


def _make_args(edition=DEFAULT_EDITION) -> argparse.Namespace:
    """构造 ``run_repl`` 命令入口所需的 argparse 命名空间（仅 ``edition`` 字段）。"""
    return argparse.Namespace(edition=edition)


def _write_adapter(path: Path) -> None:
    """用 safetensors 写入一个合法的 1×1 占位适配器文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"placeholder": np.zeros((1, 1), dtype=np.float32)}, str(path))


def _make_stub_package(
    cwd: Path,
    *,
    edition: str = DEFAULT_EDITION,
    write_manifest: bool = True,
    write_config: bool = True,
    write_adapter: bool = True,
    corrupt_config: bool = False,
    corrupt_adapter: bool = False,
) -> Path:
    """在 ``cwd`` 下构造一个 stub 训练产出的 Ego_Package（需求 12.9）。

    默认写入完整且合法的三件产物：``manifest.json``（``training_backend="stub"``）、
    ``<edition>/config.json``、``<edition>/adapters/ego-lora.safetensors``。可通过
    参数省略或损坏其中某一文件以构造 R12-AC5 的缺失/损坏场景；此时 ``<edition>/``
    子目录仍被创建，以便先通过 R12-AC3（子目录存在性）检查、再触发 R12-AC5。

    :returns: Ego_Package 根目录（``{cwd}/ego-output``）。
    """
    root = cwd / OUTPUT_DIRNAME
    edition_dir = root / edition
    edition_dir.mkdir(parents=True, exist_ok=True)

    if write_manifest:
        manifest = {
            "version": "1.0.0",
            "created_at": "2025-11-25T08:00:00Z",
            "base_model": "qwen3.5-0.8B",
            "editions": [edition],
            "training_backend": "stub",
        }
        (root / MANIFEST_FILENAME).write_text(
            json.dumps(manifest), encoding="utf-8"
        )

    if write_config:
        config_path = edition_dir / CONFIG_FILENAME
        if corrupt_config:
            config_path.write_text("{ this is not valid json", encoding="utf-8")
        else:
            config = {
                "base_model": "qwen3.5-0.8B",
                "quantization": "4bit",
                "lora_rank": 16,
                "lora_alpha": 32,
            }
            config_path.write_text(json.dumps(config), encoding="utf-8")

    if write_adapter:
        adapter_path = edition_dir / ADAPTER_RELPATH
        if corrupt_adapter:
            adapter_path.parent.mkdir(parents=True, exist_ok=True)
            adapter_path.write_bytes(b"not a real safetensors file")
        else:
            _write_adapter(adapter_path)

    return root


def _drive_run(cwd: Path, stdin_text: str, *, edition=DEFAULT_EDITION):
    """以注入的 I/O 流驱动 ``ego run`` 命令入口，返回 ``(rc, stdout, stderr)``。"""
    stdout = io.StringIO()
    stderr = io.StringIO()
    rc = run_repl(
        _make_args(edition=edition),
        cwd=cwd,
        stdin=io.StringIO(stdin_text),
        stdout=stdout,
        stderr=stderr,
    )
    return rc, stdout.getvalue(), stderr.getvalue()


# ===========================================================================
# R12-AC4：--edition 非 normal → RuntimeError_（退出码 1）
# ===========================================================================


def test_non_normal_edition_raises_runtime_error(tmp_path):
    with pytest.raises(RuntimeError_) as excinfo:
        run_repl(_make_args(edition="plus"), cwd=tmp_path, stdin=io.StringIO(""))

    assert excinfo.value.exit_code == EXIT_RUNTIME_FAILURE
    # 文案应含"仅支持 normal"提示，且回显收到的非法 edition 值。
    assert EDITION_NOT_SUPPORTED_MESSAGE in str(excinfo.value)
    assert "plus" in str(excinfo.value)


def test_non_normal_edition_checked_before_package_lookup(tmp_path):
    # 即使存在有效的 normal 包，--edition lite 仍应先因 AC4 失败（不进入加载）。
    _make_stub_package(tmp_path)
    with pytest.raises(RuntimeError_) as excinfo:
        run_repl(_make_args(edition="lite"), cwd=tmp_path, stdin=io.StringIO(""))
    assert excinfo.value.exit_code == EXIT_RUNTIME_FAILURE
    assert EDITION_NOT_SUPPORTED_MESSAGE in str(excinfo.value)


# ===========================================================================
# R12-AC2：未提供 --edition（edition=None）默认 normal
# ===========================================================================


def test_missing_edition_defaults_to_normal(tmp_path):
    # edition=None 表示用户未提供 --edition；应按 normal 处理。
    # 此时无 Ego_Package，应走 AC3（请先 ego train），而非 AC4（非 normal）。
    with pytest.raises(RuntimeError_) as excinfo:
        run_repl(_make_args(edition=None), cwd=tmp_path, stdin=io.StringIO(""))
    assert excinfo.value.exit_code == EXIT_RUNTIME_FAILURE
    assert PACKAGE_NOT_FOUND_MESSAGE in str(excinfo.value)
    assert EDITION_NOT_SUPPORTED_MESSAGE not in str(excinfo.value)


# ===========================================================================
# R12-AC3：ego-output/<edition>/ 不存在 → RuntimeError_（退出码 1，含 ego train 指引）
# ===========================================================================


def test_missing_package_directory_raises_with_guidance(tmp_path):
    # tmp_path 下没有任何 ego-output 目录。
    with pytest.raises(RuntimeError_) as excinfo:
        run_repl(_make_args(), cwd=tmp_path, stdin=io.StringIO(""))

    message = str(excinfo.value)
    assert excinfo.value.exit_code == EXIT_RUNTIME_FAILURE
    assert PACKAGE_NOT_FOUND_MESSAGE in message
    # 指引用户先执行 ego train。
    assert "请先" in message
    assert "ego train" in message


def test_output_root_exists_but_edition_subdir_missing(tmp_path):
    # ego-output/ 存在，但其下没有 normal/ 子目录 → 仍按 AC3 处理。
    (tmp_path / OUTPUT_DIRNAME).mkdir()
    with pytest.raises(RuntimeError_) as excinfo:
        run_repl(_make_args(), cwd=tmp_path, stdin=io.StringIO(""))
    assert excinfo.value.exit_code == EXIT_RUNTIME_FAILURE
    assert PACKAGE_NOT_FOUND_MESSAGE in str(excinfo.value)


# ===========================================================================
# R12-AC5：子目录存在但 config.json / 适配器缺失或损坏 → RuntimeError_（退出码 1）
# ===========================================================================


def test_missing_config_json_raises(tmp_path):
    # normal/ 存在、manifest 与适配器在位，但缺 config.json。
    _make_stub_package(tmp_path, write_config=False)
    with pytest.raises(RuntimeError_) as excinfo:
        run_repl(_make_args(), cwd=tmp_path, stdin=io.StringIO(""))
    assert excinfo.value.exit_code == EXIT_RUNTIME_FAILURE
    assert CONFIG_FILENAME in str(excinfo.value)


def test_corrupt_config_json_raises(tmp_path):
    # config.json 存在但是非法 JSON（损坏）。
    _make_stub_package(tmp_path, corrupt_config=True)
    with pytest.raises(RuntimeError_) as excinfo:
        run_repl(_make_args(), cwd=tmp_path, stdin=io.StringIO(""))
    assert excinfo.value.exit_code == EXIT_RUNTIME_FAILURE
    assert CONFIG_FILENAME in str(excinfo.value)


def test_missing_adapter_raises(tmp_path):
    # normal/ 存在、manifest 与 config 在位，但缺适配器文件。
    _make_stub_package(tmp_path, write_adapter=False)
    with pytest.raises(RuntimeError_) as excinfo:
        run_repl(_make_args(), cwd=tmp_path, stdin=io.StringIO(""))
    assert excinfo.value.exit_code == EXIT_RUNTIME_FAILURE
    assert "ego-lora.safetensors" in str(excinfo.value)


def test_corrupt_adapter_raises(tmp_path):
    # 适配器文件存在但内容损坏（非法 safetensors）。
    _make_stub_package(tmp_path, corrupt_adapter=True)
    with pytest.raises(RuntimeError_) as excinfo:
        run_repl(_make_args(), cwd=tmp_path, stdin=io.StringIO(""))
    assert excinfo.value.exit_code == EXIT_RUNTIME_FAILURE
    assert "ego-lora.safetensors" in str(excinfo.value)


# ===========================================================================
# R12-AC9 / AC6 / AC8：有效 stub 包上端到端驱动 REPL
# ===========================================================================


def test_stub_mode_fixed_reply_and_quit_exit_zero(tmp_path):
    # hello → 一次固定占位回复；空行 → 不调推理；:quit → 退出码 0。
    _make_stub_package(tmp_path)
    rc, out, err = _drive_run(tmp_path, "hello\n\n:quit\n")

    assert rc == EXIT_OK
    # R12-AC9：启动时打印 Stub 模式提示。
    assert STUB_MODE_STARTUP_NOTICE in out
    # R12-AC9/AC6：对 "hello" 返回固定占位回复。
    assert STUB_PLACEHOLDER_REPLY in out
    # R12-AC7：空行不触发推理 → 占位回复恰好出现一次（仅 "hello" 那一行）。
    assert out.count(STUB_PLACEHOLDER_REPLY) == 1
    # 每轮交互前打印提示符（hello / 空行 / :quit 三轮）。
    assert out.count(PROMPT) == 3


def test_stub_mode_eof_exits_zero(tmp_path):
    # 不输入 :quit，仅以 EOF（StringIO 耗尽）结束 → 退出码 0。
    _make_stub_package(tmp_path)
    rc, out, err = _drive_run(tmp_path, "hello\n")

    assert rc == EXIT_OK
    assert STUB_PLACEHOLDER_REPLY in out
    # 只有 "hello" 一行非空输入触发推理。
    assert out.count(STUB_PLACEHOLDER_REPLY) == 1


def test_stub_mode_blank_only_then_eof_no_inference(tmp_path):
    # 仅空白行后 EOF：从不调用推理，无占位回复，退出码 0。
    _make_stub_package(tmp_path)
    rc, out, err = _drive_run(tmp_path, "   \n\t\n")

    assert rc == EXIT_OK
    assert STUB_PLACEHOLDER_REPLY not in out


def test_stub_mode_multiple_inputs_each_get_reply(tmp_path):
    # 两条非空输入 → 两次固定回复；中间空行不计入。
    _make_stub_package(tmp_path)
    rc, out, err = _drive_run(tmp_path, "hi\n\nthere\n:quit\n")

    assert rc == EXIT_OK
    assert out.count(STUB_PLACEHOLDER_REPLY) == 2


# ===========================================================================
# REPL 行为层：注入计数桩生成器，精确验证 AC6 / AC7 / AC8
# ===========================================================================


class _SpyGenerator:
    """记录调用次数与输入的响应生成桩，用于精确验证 REPL 是否调用推理。"""

    def __init__(self, reply: str = "REPLY") -> None:
        self.calls: list[str] = []
        self.closed = False
        self.reply = reply

    def generate(self, user_input: str) -> str:
        self.calls.append(user_input)
        return self.reply

    def close(self) -> None:
        self.closed = True


def test_repl_blank_lines_do_not_call_inference():
    # R12-AC7：仅空白/空行 → generate 从不被调用。
    spy = _SpyGenerator()
    stdout = io.StringIO()
    rc = run_repl_loop(
        spy,
        stdin=io.StringIO("\n   \n\t\n:quit\n"),
        stdout=stdout,
        stderr=io.StringIO(),
    )
    assert rc == EXIT_OK
    assert spy.calls == []
    assert spy.closed is True


def test_repl_non_empty_input_calls_inference_once_per_line():
    # R12-AC6：每条非空输入恰调用一次推理并打印回复；空行被跳过。
    spy = _SpyGenerator(reply="ECHO")
    stdout = io.StringIO()
    rc = run_repl_loop(
        spy,
        stdin=io.StringIO("alpha\n\nbeta\n:quit\n"),
        stdout=stdout,
        stderr=io.StringIO(),
    )
    assert rc == EXIT_OK
    # 仅 alpha / beta 触发推理（去除首尾空白）。
    assert spy.calls == ["alpha", "beta"]
    assert stdout.getvalue().count("ECHO") == 2


def test_repl_quit_command_exits_zero_and_releases():
    # R12-AC8：:quit → 退出码 0 并调用 close 释放资源。
    spy = _SpyGenerator()
    rc = run_repl_loop(
        spy,
        stdin=io.StringIO(f"{QUIT_COMMAND}\n"),
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )
    assert rc == EXIT_OK
    assert spy.calls == []
    assert spy.closed is True


def test_repl_eof_exits_zero_and_releases():
    # R12-AC8：EOF（输入耗尽，无 :quit）→ 退出码 0 并释放资源。
    spy = _SpyGenerator()
    rc = run_repl_loop(
        spy,
        stdin=io.StringIO("gamma\n"),
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )
    assert rc == EXIT_OK
    assert spy.calls == ["gamma"]
    assert spy.closed is True
