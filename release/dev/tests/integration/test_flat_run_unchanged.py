"""Flat_Package ``ego run`` 行为不变集成测试（ego-layered-adapter 任务 8.3）。

本文件验证 **向后兼容契约**（Requirement 6）：对一个**不含 ``grid`` 键**的
Flat_Package（ego-pod-mvp / ego-pod-real-trainer 现有管线产出的形状）执行
``ego run`` 时，Runtime 把该 Edition 视为**仅含单个 Kernel Cell**，从
``adapters/ego-lora.safetensors`` 加载并进入 Stub 推理路径，其可观测的 REPL
行为与 MVP **逐字节一致**。

覆盖的验收标准：

* **R6-AC1**：Runtime 加载不含 Grid_Descriptor 的 Flat_Package 时，把该 Edition
  视为仅含单个 Kernel Cell，从 ``adapters/ego-lora.safetensors`` 加载并成功进入
  Stub 路径。
* **R6-AC2**：仅含单个 Kernel Cell 的包产生与加载等价 Flat_Package 相同的可观测
  推理行为——本文件以「同一组输入下两个独立 Flat_Package（一个手工构造、一个由
  Stub 训练退化产出）的 ``ego run`` 输出逐字节相等」直接行使该等价性。

构造方式（任务 8.3 给出的两条路径，本文件二者皆行使）：

1. **手工构造 Flat_Package**：用 :func:`safetensors.numpy.save_file` 写一个合法的
   1×1 占位适配器（仅依赖 numpy + safetensors，无任何 ML 框架），并写出无 ``grid``
   键的 ``manifest.json`` 与 ``normal/config.json``——与现有 MVP 产物形状一致。
2. **Stub 训练退化产出**：仅在 ``sense`` 与 ``credentials`` 两个 Data_Category 提供
   语料；默认 Zone_Config 基线把这两类路由到 Kernel（``skill``/``action`` → coding、
   ``thought`` → knowledge 均无样本），于是网格退化为**仅含 Kernel** 的单 Cell，
   Vault 据「决策 C」走扁平路径，产出**纯 Flat_Package**（``ego-lora.safetensors``、
   无 ``grid`` 键）。

``ego run`` 经 :func:`ego_pod.commands.run.run_repl` 驱动，注入 ``cwd`` 与
:class:`io.StringIO` 形式的 stdin/stdout/stderr（与 ``test_run_after_train`` /
``test_stub_path_regression`` 一致）。autouse ``force_stub_backend`` 夹具保证训练
回退到 Stub_Backend；产物均落于 ``tmp_path``，不依赖网络或真实 ML 依赖。
"""

from __future__ import annotations

import argparse
import io
import json

import numpy as np
from safetensors.numpy import save_file

from ego_pod.commands.run import run_repl
from ego_pod.exit_codes import EXIT_OK
from ego_pod.runtime.inference import (
    ADAPTER_RELPATH,
    CONFIG_FILENAME,
    DEFAULT_EDITION,
    MANIFEST_FILENAME,
    OUTPUT_DIRNAME,
    STUB_MODE_STARTUP_NOTICE,
    STUB_PLACEHOLDER_REPLY,
    STUB_TRAINING_BACKEND,
    StubInferenceSession,
    load_inference_session,
)
from ego_pod.runtime.repl import PROMPT
from ego_pod.runtime.selection import KERNEL, select_cells
from ego_pod.vault.grid_descriptor import FLAT_KERNEL_RELPATH, parse_grid_descriptor


# ---------------------------------------------------------------------------
# 手工构造 Flat_Package（无 grid 键 + 合法占位适配器）
# ---------------------------------------------------------------------------


def _build_flat_package(cwd, *, training_backend: str = STUB_TRAINING_BACKEND):
    """在 ``cwd/ego-output/`` 下手工构造一个**无 ``grid`` 键**的 Flat_Package。

    写出与现有 ego-pod-mvp / ego-pod-real-trainer 产物形状一致的三件套：

    * ``manifest.json``：五字段（``version`` / ``created_at`` / ``base_model`` /
      ``editions`` / ``training_backend``），**不含** ``grid`` 键；
    * ``normal/config.json``：合法 LoRA 参数；
    * ``normal/adapters/ego-lora.safetensors``：用 numpy 后端写入的合法 1×1 占位
      张量（可被 ``safetensors`` 正常打开，从而通过 ``load_inference_session`` 的
      完整性校验）。

    :param cwd: 工作目录；Ego_Package 落于其下的 ``ego-output/``。
    :param training_backend: 写入 manifest 的后端标记，默认 ``"stub"``。
    :returns: Ego_Package 根目录路径（``cwd/ego-output``）。
    """
    root = cwd / OUTPUT_DIRNAME
    edition_dir = root / DEFAULT_EDITION
    adapter_path = edition_dir / ADAPTER_RELPATH
    adapter_path.parent.mkdir(parents=True, exist_ok=True)

    # manifest.json：五字段、无 grid 键（Flat_Package 形状，R6-AC1）。
    manifest = {
        "version": "0.1.0",
        "created_at": "2025-11-25T08:00:00Z",
        "base_model": "qwen3.5-0.8B",
        "editions": [DEFAULT_EDITION],
        "training_backend": training_backend,
    }
    (root / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    config = {
        "base_model": "qwen3.5-0.8B",
        "quantization": "4bit",
        "lora_rank": 16,
        "lora_alpha": 32,
    }
    (edition_dir / CONFIG_FILENAME).write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # 合法占位适配器（仅 numpy + safetensors，无 ML 框架）。
    save_file({"placeholder": np.zeros((1, 1), dtype=np.float32)}, str(adapter_path))
    return root


def _drive_run(cwd, stdin_text: str, *, edition="normal"):
    """以注入的 I/O 流驱动 ``ego run`` 命令入口，返回 ``(rc, stdout, stderr)``。

    与 ``test_run_after_train`` / ``test_stub_path_regression`` 的驱动方式一致：
    注入 ``cwd`` 与 :class:`io.StringIO` 形式的 stdin/stdout/stderr，使 REPL 不
    触碰真实控制台。
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


def _kernel_only_contents() -> dict[str, list[tuple[str, str]]]:
    """仅在 ``sense`` 与 ``credentials`` 提供语料（默认基线下二者均路由到 Kernel）。

    其余三类（``action`` / ``thought`` / ``skill``）不提供目录，使路由结果**仅含
    Kernel** ⇒ Vault 走扁平路径产出**纯 Flat_Package**（无 ``grid`` 键）。
    """
    return {
        "sense": [
            ("s0.txt", "清晨的空气清新宜人，窗外鸟鸣阵阵。"),
            ("s1.txt", "午后阳光斜照进安静的书房。"),
        ],
        "credentials": [
            ("c0.txt", "这是一段不含敏感信息的普通说明文本。"),
            ("c1.txt", "另一段平常的备注内容。"),
        ],
    }


# ---------------------------------------------------------------------------
# R6-AC1：无 grid 的 Flat_Package 加载为单 Kernel Cell 并进入 Stub 路径
# ---------------------------------------------------------------------------


def test_flat_package_parses_as_single_kernel_cell(tmp_path):
    """R6-AC1：无 ``grid`` 键 ⇒ 解析为单 Kernel Cell 的 Flat 网格（``ego-lora.safetensors``）。

    直接行使加载层语义：``parse_grid_descriptor(None)`` 返回单 Kernel Cell、无任何
    分区、Kernel 相对路径为 ``adapters/ego-lora.safetensors``；``select_cells`` 在
    Full_In_Load 下恰好选出 ``[KERNEL]``。
    """
    grid = parse_grid_descriptor(None, DEFAULT_EDITION)

    # 单 Kernel Cell：Kernel 落 adapters/ego-lora.safetensors，无任何分区（R6-AC1）。
    assert grid.kernel.cell == KERNEL
    assert grid.kernel.relpath == FLAT_KERNEL_RELPATH == "adapters/ego-lora.safetensors"
    assert grid.zones == {}

    # Full_In_Load（selection=None）下恰好选出单个 Kernel Cell。
    assert select_cells(grid, None) == [KERNEL]


def test_flat_package_load_returns_stub_session(tmp_path):
    """R6-AC1：手工构造的无 ``grid`` Flat_Package 经 ``load_inference_session`` 进入 Stub 路径。"""
    cwd = tmp_path / "flatpkg"
    cwd.mkdir()
    root = _build_flat_package(cwd)

    # manifest 确实不含 grid 键（Flat_Package 形状）。
    manifest = json.loads((root / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert "grid" not in manifest

    stdout = io.StringIO()
    session = load_inference_session(DEFAULT_EDITION, cwd=cwd, stdout=stdout)

    # training_backend == "stub" ⇒ Stub 推理会话 + 启动提示（R6-AC1 进入 Stub 路径）。
    assert isinstance(session, StubInferenceSession)
    assert STUB_MODE_STARTUP_NOTICE in stdout.getvalue()
    session.close()


def test_flat_package_run_stub_fixed_reply_and_quit(tmp_path):
    """R6-AC1：无 ``grid`` Flat_Package ``ego run``——启动提示 + 固定回复 + 空行不调推理 + :quit 退出 0。"""
    cwd = tmp_path / "flatpkg"
    cwd.mkdir()
    _build_flat_package(cwd)

    # 四行输入：两条非空（各一次固定回复）+ 一个空白行（不调推理）+ :quit 退出。
    rc, out, err = _drive_run(cwd, "第一句\n  \n第二句\n:quit\n")

    # :quit → 退出码 0（与 MVP R12-AC8 一致）。
    assert rc == EXIT_OK, f"stub ego run 未以退出码 0 结束，stderr=\n{err}"
    # 启动时打印 Stub 模式提示（R6-AC1）。
    assert STUB_MODE_STARTUP_NOTICE in out
    # 每条非空输入返回同一条固定占位回复，且恰好两次（空白行不触发推理）。
    assert STUB_PLACEHOLDER_REPLY in out
    assert out.count(STUB_PLACEHOLDER_REPLY) == 2
    # 四行输入各打印一次提示符（第一句 / 空白 / 第二句 / :quit）。
    assert out.count(PROMPT) == 4


def test_flat_package_run_eof_exits_zero(tmp_path):
    """R6-AC1：无 ``grid`` Flat_Package 以 EOF 结束 ``ego run`` → 退出码 0、每条非空输入一次回复。"""
    cwd = tmp_path / "flatpkg"
    cwd.mkdir()
    _build_flat_package(cwd)

    # 两条非空输入 + 一个空行，随后 EOF（输入耗尽，无 :quit）。
    rc, out, err = _drive_run(cwd, "你好\n\n今天怎么样\n")

    assert rc == EXIT_OK, f"stub ego run 未以退出码 0 结束，stderr=\n{err}"
    assert STUB_MODE_STARTUP_NOTICE in out
    # 两条非空输入 → 两次固定回复（空行不触发推理）。
    assert out.count(STUB_PLACEHOLDER_REPLY) == 2
    # 三行输入 + 最后一次读到 EOF = 4 次提示符。
    assert out.count(PROMPT) == 4


# ---------------------------------------------------------------------------
# R6-AC2：单 Kernel Cell 包与等价 Flat_Package 的可观测行为逐字节一致
# ---------------------------------------------------------------------------


def test_kernel_only_stub_train_produces_flat_package(
    tmp_path, make_category_dirs, write_config, run_train
):
    """R6-AC2 前置：仅 ``sense``/``credentials`` 语料 → Stub 训练退化产出**纯 Flat_Package**。

    默认 Zone_Config 基线把 ``sense``/``credentials`` 路由到 Kernel，其余分区无样本
    ⇒ 网格退化为仅 Kernel 的单 Cell ⇒ Vault 走扁平路径：产物为 ``ego-lora.safetensors``、
    ``manifest.json`` **不含** ``grid`` 键（与既有 MVP 产物形状一致）。
    """
    sources = make_category_dirs(tmp_path, _kernel_only_contents())
    write_config(tmp_path / "ego-config.yaml", sources)

    rc, _out, err = run_train()
    assert rc == EXIT_OK, f"kernel-only ego train 未成功，stderr=\n{err}"

    # 退化为纯 Flat_Package：adapters/ego-lora.safetensors 存在、manifest 无 grid 键。
    root = tmp_path / OUTPUT_DIRNAME
    assert (root / DEFAULT_EDITION / "adapters" / "ego-lora.safetensors").is_file()
    manifest = json.loads((root / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert "grid" not in manifest, "kernel-only 退化包不应写出 grid 键（纯 Flat_Package）"
    assert manifest["training_backend"] == STUB_TRAINING_BACKEND


def test_flat_run_byte_for_byte_matches_mvp(
    tmp_path, make_category_dirs, write_config, run_train
):
    """R6-AC2：手工 Flat_Package 与 Stub 训练退化 Flat_Package 的 ``ego run`` 输出逐字节相等。

    构造两个独立的 Flat_Package（一个手工构造、一个由 Stub 训练退化产出），对**同一
    组输入**分别驱动 ``ego run``，断言两份标准输出**逐字节相等**——即仅含单个 Kernel
    Cell 的包产生与加载等价 Flat_Package 相同的可观测推理行为（R6-AC2）。
    """
    # --- (1) 手工构造 Flat_Package（无 grid 键）-------------------------------
    hand_cwd = tmp_path / "hand"
    hand_cwd.mkdir()
    _build_flat_package(hand_cwd)

    # --- (2) Stub 训练退化产出 Flat_Package（kernel-only）---------------------
    sources = make_category_dirs(tmp_path, _kernel_only_contents())
    write_config(tmp_path / "ego-config.yaml", sources)
    rc_train, _train_out, train_err = run_train()
    assert rc_train == EXIT_OK, f"kernel-only ego train 未成功，stderr=\n{train_err}"
    # run_train 已 chdir 到 tmp_path，故训练产物落于 tmp_path/ego-output。
    trained_cwd = tmp_path

    # 二者均为 Flat_Package（无 grid 键）。
    for cwd in (hand_cwd, trained_cwd):
        manifest = json.loads(
            (cwd / OUTPUT_DIRNAME / MANIFEST_FILENAME).read_text(encoding="utf-8")
        )
        assert "grid" not in manifest

    # --- (3) 同一组输入分别驱动 ego run，断言输出逐字节相等 -------------------
    stdin_text = "你好\n\n今天怎么样\n:quit\n"
    rc_hand, out_hand, _ = _drive_run(hand_cwd, stdin_text)
    rc_trained, out_trained, _ = _drive_run(trained_cwd, stdin_text)

    assert rc_hand == EXIT_OK
    assert rc_trained == EXIT_OK
    # R6-AC2：可观测推理行为逐字节一致。
    assert out_hand == out_trained, (
        "手工 Flat_Package 与 Stub 训练退化 Flat_Package 的 ego run 输出应逐字节相等"
    )
    # 二者均行使了 MVP 既有的 Stub REPL 行为：启动提示 + 两次固定占位回复。
    assert STUB_MODE_STARTUP_NOTICE in out_hand
    assert out_hand.count(STUB_PLACEHOLDER_REPLY) == 2
