"""``ego run`` REPL 集成测试（stub 训练后，任务 12.3）。

本文件先运行完整的 ``ego train``（Stub_Backend，产出 stub 标记的 Ego_Package），
再驱动 ``ego run`` 的交互式 REPL，验证 Requirement 12 在 stub 模式下的端到端
行为：

* **R12-AC1 / R12-AC2**：未提供 ``--edition`` 时默认加载 ``normal`` Edition 的
  ``ego-output/normal/`` 产物。
* **R12-AC9**：``manifest.training_backend == "stub"`` → 启动时打印 Stub 模式提示，
  对每次非空输入返回同一条固定占位回复。
* **R12-AC6**：每条非空输入恰得到一次回复。
* **R12-AC7**：空行 / 仅空白行不调用推理（不产生额外回复）。
* **R12-AC8**：``:quit`` 与 EOF 均以退出码 0 结束。

``ego run`` 经 :func:`ego_pod.commands.run.run_repl` 驱动，注入 ``cwd=tmp_path``
与 :class:`io.StringIO` 形式的 stdin/stdout/stderr。autouse ``force_stub_backend``
夹具保证训练回退到 Stub_Backend；产物均落于 ``tmp_path``，不依赖网络。
"""

from __future__ import annotations

import argparse
import io

from ego_pod.commands.run import run_repl
from ego_pod.exit_codes import EXIT_OK
from ego_pod.runtime.inference import (
    STUB_MODE_STARTUP_NOTICE,
    STUB_PLACEHOLDER_REPLY,
)
from ego_pod.runtime.repl import PROMPT


def _train_stub_package(tmp_path, make_category_dirs, write_config, run_train):
    """运行完整 ``ego train``（stub）并返回训练后的工作目录（``tmp_path``）。"""
    contents = {
        "sense": [("s0.txt", "清晨的空气清新宜人。"), ("s1.txt", "午后阳光温暖。")],
        "action": [("a0.txt", "上午慢跑五公里。"), ("a1.txt", "下午整理书架。")],
        "thought": [("t0.txt", "关于文明尺度的思考。"), ("t1.txt", "耐心是一种答案。")],
        "skill": [("k0.txt", "手冲咖啡的三个要点。"), ("k1.txt", "十分钟整理思路法。")],
        "credentials": [("c0.txt", "一段普通说明文本。"), ("c1.txt", "另一段备注内容。")],
    }
    sources = make_category_dirs(tmp_path, contents)
    write_config(tmp_path / "ego-config.yaml", sources)
    rc, _out, err = run_train()
    assert rc == EXIT_OK, f"前置 ego train 未成功，stderr=\n{err}"
    # 确认产物齐全，作为 ego run 的前置条件。
    assert (tmp_path / "ego-output" / "normal" / "config.json").is_file()
    return tmp_path


def _drive_run(cwd, stdin_text: str, *, edition="normal"):
    """以注入的 I/O 流驱动 ``ego run`` 命令入口，返回 ``(rc, stdout, stderr)``。"""
    stdout = io.StringIO()
    stderr = io.StringIO()
    rc = run_repl(
        argparse.Namespace(edition=edition),
        cwd=cwd,
        stdin=io.StringIO(stdin_text),
        stdout=stdout,
        stderr=stderr,
    )
    return rc, stdout.getvalue(), stderr.getvalue()


def test_run_after_train_stub_fixed_reply_and_quit(
    tmp_path, make_category_dirs, write_config, run_train
):
    """train(stub) → run：固定回复、空行不调推理、:quit 退出码 0。"""
    cwd = _train_stub_package(tmp_path, make_category_dirs, write_config, run_train)

    # 三行输入：非空 "你好" → 一次回复；空行 → 不调推理；:quit → 退出。
    rc, out, err = _drive_run(cwd, "你好\n\n:quit\n")

    # R12-AC8：:quit → 退出码 0。
    assert rc == EXIT_OK
    # R12-AC9：启动时打印 Stub 模式提示。
    assert STUB_MODE_STARTUP_NOTICE in out
    # R12-AC9 / R12-AC6：非空输入返回固定占位回复。
    assert STUB_PLACEHOLDER_REPLY in out
    # R12-AC7：空行不触发推理 → 占位回复恰好一次（仅 "你好" 那一轮）。
    assert out.count(STUB_PLACEHOLDER_REPLY) == 1
    # 三轮交互各打印一次提示符（你好 / 空行 / :quit）。
    assert out.count(PROMPT) == 3


def test_run_after_train_multiple_inputs_each_reply(
    tmp_path, make_category_dirs, write_config, run_train
):
    """R12-AC6：两条非空输入各得一次固定回复，中间空行被跳过。"""
    cwd = _train_stub_package(tmp_path, make_category_dirs, write_config, run_train)

    rc, out, err = _drive_run(cwd, "第一句\n  \n第二句\n:quit\n")

    assert rc == EXIT_OK
    assert out.count(STUB_PLACEHOLDER_REPLY) == 2


def test_run_after_train_eof_exits_zero(
    tmp_path, make_category_dirs, write_config, run_train
):
    """R12-AC8：以 EOF（输入耗尽，无 :quit）结束 → 退出码 0。"""
    cwd = _train_stub_package(tmp_path, make_category_dirs, write_config, run_train)

    rc, out, err = _drive_run(cwd, "唯一一句\n")

    assert rc == EXIT_OK
    # 仅一条非空输入触发推理。
    assert out.count(STUB_PLACEHOLDER_REPLY) == 1


def test_run_after_train_blank_only_no_inference(
    tmp_path, make_category_dirs, write_config, run_train
):
    """R12-AC7：仅空白行后 EOF → 从不调用推理，无占位回复，退出码 0。"""
    cwd = _train_stub_package(tmp_path, make_category_dirs, write_config, run_train)

    rc, out, err = _drive_run(cwd, "   \n\t\n")

    assert rc == EXIT_OK
    assert STUB_PLACEHOLDER_REPLY not in out


def test_run_after_train_default_edition_normal(
    tmp_path, make_category_dirs, write_config, run_train
):
    """R12-AC1 / R12-AC2：edition=None（未提供 --edition）默认加载 normal。"""
    cwd = _train_stub_package(tmp_path, make_category_dirs, write_config, run_train)

    rc, out, err = _drive_run(cwd, ":quit\n", edition=None)

    assert rc == EXIT_OK
    # 成功进入 stub REPL（打印启动提示）即证明默认按 normal 加载成功。
    assert STUB_MODE_STARTUP_NOTICE in out
