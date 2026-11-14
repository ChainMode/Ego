"""Stub 路径回归不变集成测试（ego-pod-real-trainer 任务 13.3）。

本文件是 **real-trainer 特性引入后的回归门**：在**完全不安装任何
Training_Dependencies**（torch / transformers / peft / bitsandbytes / mlx 等）的
环境中，验证 MVP 既有的 Stub_Backend 端到端路径（``ego train`` → ``ego run``）
保持不变。它必须在本 CI 中**通过**（而非跳过）——这正是 Requirement 8 所追求的
「真实训练能力的引入不破坏既有管线、CI 保持绿色」契约。

覆盖的验收标准：

* **R8-AC3**：全部四个 Real_Backend 均不可用时，Trainer 选择始终可用的
  Stub_Backend 作为最终回退，使 ``ego train`` 端到端跑通至退出码 0。
* **R8-AC7**：Stub_Backend 保持 MVP 既有行为不变——其产出的 ``manifest.json``
  的 ``training_backend`` 标记为 ``stub``（``STUB_TRAINING_BACKEND``）。
* **R6-AC6**：当加载的 Ego_Package 的 ``training_backend`` 取值为 ``stub`` 时，
  Runtime 保持 MVP 既有 Stub 推理路径行为不变——启动时打印 Stub 模式提示
  （``STUB_MODE_STARTUP_NOTICE``），并对每次非空输入返回同一条固定占位回复
  （``STUB_PLACEHOLDER_REPLY``）。

为强制走 Stub 回退路径，本文件复用 ``tests/integration/conftest.py`` 的 autouse
``force_stub_backend`` 夹具：它把四个真实后端的 ``is_available`` 一律改写为返回
``False``，于是 :func:`ego_pod.trainer.detect.detect_backend` 跳过全部真实后端并
回退到 Stub_Backend——即便在恰好装有 torch/peft 的机器上也能确定性地复现「无 ML
依赖」的回归场景。``ego train`` 复用 ``run_train`` 夹具（经 :func:`ego_pod.cli.main`
驱动）；``ego run`` 经 :func:`ego_pod.commands.run.run_repl` 注入 :class:`io.StringIO`
形式的 stdin/stdout/stderr 驱动（与 ``test_run_after_train`` 一致）。产物均落于
``tmp_path``，不依赖网络或真实 ML 依赖。
"""

from __future__ import annotations

import argparse
import io
import json

from ego_pod.commands.run import run_repl
from ego_pod.exit_codes import EXIT_OK
from ego_pod.runtime.inference import (
    STUB_MODE_STARTUP_NOTICE,
    STUB_PLACEHOLDER_REPLY,
    STUB_TRAINING_BACKEND,
)
from ego_pod.runtime.repl import PROMPT


def _tiny_valid_contents() -> dict[str, list[tuple[str, str]]]:
    """构造**仅** ``sense`` + ``credentials`` 两类、各含两个内容互异非空文件的最小合法语料。

    本文件是 **Flat_Package（扁平包）回归门**：它断言 ``ego train`` 产出单适配器
    ``adapters/ego-lora.safetensors``（而非二维网格）。在二维网格特性引入后，默认
    Zone_Config 基线会把 ``skill``/``action`` → ``coding``、``thought`` → ``knowledge``
    路由出分区 Cell，从而产出 Grid_Package（``kernel.safetensors`` + ``<zone>/L<level>``）。
    为继续作为扁平回归门，本语料**刻意只含** ``sense`` + ``credentials`` 两类——二者
    均不在默认路由表中，连同任何未匹配样本一律归入 Kernel（R2-AC3）⇒ 零个分区 Cell
    ⇒ Vault 走退化扁平路径，产出与 MVP / real-trainer 逐字节一致的 Flat_Package
    （决策 C / R4-AC7 / R6-AC1/AC2）。

    每个文件内容互不相同以避免 SHA-256 去重把样本合并，且文本足够长以通过过滤
    阶段的最小长度约束；四条过滤后样本足以让增强阶段从非空 Kernel 集合补足至最小
    样本数 16，使 collect→compile→train→vault 四个阶段在 Stub_Backend 下全部成功。
    """
    samples = {
        "sense": ["清晨的空气清新宜人，窗外鸟鸣阵阵。", "午后阳光斜照进安静的书房。"],
        "credentials": ["这是一段不含敏感信息的普通说明文本。", "另一段平常的备注内容。"],
    }
    return {
        category: [(f"{category}_{idx}.txt", text) for idx, text in enumerate(texts)]
        for category, texts in samples.items()
    }


def _drive_run(cwd, stdin_text: str, *, edition="normal"):
    """以注入的 I/O 流驱动 ``ego run`` 命令入口，返回 ``(rc, stdout, stderr)``。

    与 ``test_run_after_train`` 的驱动方式一致：注入 ``cwd`` 与 :class:`io.StringIO`
    形式的 stdin/stdout/stderr，使 REPL 不触碰真实控制台。
    """
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


def test_stub_path_regression_train_then_run_unchanged(
    tmp_path, make_category_dirs, write_config, run_train
):
    """无 ML 依赖下 ``ego train``(stub) → ``ego run``：完整回归不变。

    验证 R8-AC3（stub 回退 → train 退出码 0）、R8-AC7（manifest.training_backend
    == "stub"）、R6-AC6（stub REPL 启动提示 + 每次非空输入固定占位回复）。
    """
    # --- 构造最小合法配置 + kernel-only 语料（仅 sense+credentials）----------
    sources = make_category_dirs(tmp_path, _tiny_valid_contents())
    write_config(tmp_path / "ego-config.yaml", sources)

    # --- R8-AC3：Stub_Backend 回退使 ego train 端到端跑通至退出码 0 ----------
    rc, train_out, train_err = run_train()
    assert rc == EXIT_OK, f"Stub 回退路径下 ego train 未成功，stderr=\n{train_err}"

    # Trainer 在标准输出记录所选后端为 stub（回退确实生效，而非命中真实后端）。
    assert "Selected backend: stub (STUB)" in train_out

    # --- R8-AC7：manifest.training_backend == "stub" ------------------------
    manifest_path = tmp_path / "ego-output" / "manifest.json"
    assert manifest_path.is_file(), "缺少 ego-output/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["training_backend"] == STUB_TRAINING_BACKEND == "stub"

    # kernel-only 退化 ⇒ 纯 Flat_Package：manifest 不含 grid 键（R6-AC1）。
    assert "grid" not in manifest, (
        "kernel-only 回归产物应为纯 Flat_Package（manifest 无 grid 键）"
    )

    # 产物齐全，作为 ego run 的前置条件（R10-AC6 不变）。扁平单适配器布局（R4-AC7）。
    assert (tmp_path / "ego-output" / "normal" / "config.json").is_file()
    assert (
        tmp_path / "ego-output" / "normal" / "adapters" / "ego-lora.safetensors"
    ).is_file()

    # --- R6-AC6：stub REPL 启动提示 + 每次非空输入固定占位回复 ---------------
    # 三轮输入：两条非空（各得一次固定回复）+ 一个空行（不调推理），随后 EOF 结束。
    rc_run, run_out, run_err = _drive_run(tmp_path, "你好\n\n今天怎么样\n")

    # EOF 结束 → 退出码 0（R12-AC8）。
    assert rc_run == EXIT_OK, f"stub ego run 未以退出码 0 结束，stderr=\n{run_err}"

    # 启动时打印 Stub 模式提示（R6-AC6）。
    assert STUB_MODE_STARTUP_NOTICE in run_out

    # 每条非空输入返回同一条固定占位回复，且恰好两次（空行不触发推理，R6-AC6）。
    assert STUB_PLACEHOLDER_REPLY in run_out
    assert run_out.count(STUB_PLACEHOLDER_REPLY) == 2

    # REPL 在每次 readline 前打印提示符，包括最后一次读到 EOF 那轮：
    # 三行输入（你好 / 空行 / 今天怎么样）+ EOF 读取 = 4 次提示符。
    assert run_out.count(PROMPT) == 4


def test_stub_path_regression_fixed_reply_is_input_independent(
    tmp_path, make_category_dirs, write_config, run_train
):
    """R6-AC6：stub 模式对不同非空输入均返回**同一条**固定占位回复。

    断言 stub 回复与输入内容无关——三条互异的非空输入产生三条完全相同的回复，
    且均等于 ``STUB_PLACEHOLDER_REPLY``。
    """
    sources = make_category_dirs(tmp_path, _tiny_valid_contents())
    write_config(tmp_path / "ego-config.yaml", sources)

    rc, _train_out, train_err = run_train()
    assert rc == EXIT_OK, f"前置 ego train 未成功，stderr=\n{train_err}"

    rc_run, run_out, _run_err = _drive_run(tmp_path, "第一句\n第二句不同\n第三句也不同\n")
    assert rc_run == EXIT_OK

    # REPL 把回复打印在提示符（``ego> `` 以 end="" 输出）之后的同一行，故按行精确
    # 匹配不可行；改为提取每行去除提示符前缀后的回复文本，断言三条完全一致。
    reply_texts = [
        line[len(PROMPT):]
        for line in run_out.splitlines()
        if line.startswith(PROMPT) and STUB_PLACEHOLDER_REPLY in line
    ]
    assert len(reply_texts) == 3, "三条非空输入应各得一次固定回复"
    assert set(reply_texts) == {STUB_PLACEHOLDER_REPLY}, (
        "stub 回复应与输入内容无关（恒为同一条固定占位回复）"
    )
