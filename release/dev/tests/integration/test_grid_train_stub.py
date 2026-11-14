"""Stub 端到端二维网格训练集成测试（ego-layered-adapter 任务 8.1）。

本文件验证 **含分区的默认 Zone_Config 基线**下，一次完整 ``ego train`` 产出的是
一份 **Grid_Package**——而非旧的扁平 Flat_Package。覆盖的验收标准：

* **R2-AC4**：Compiler 路由阶段把五类语料分派到 Kernel 与各 Functional_Zone，
  并在 stdout 打印网格摘要行（``grid summary: Kernel=... (coding, L1)=...
  (knowledge, L1)=...``）。
* **R3-AC1**：Trainer 按 Cell 训练（Kernel + coding/L1 + knowledge/L1）。
* **R4-AC1**：Vault 落盘 Grid_Package——``manifest.json`` 含 ``grid`` 键，磁盘上
  ``normal/adapters/`` 下含 ``kernel.safetensors`` 与各 ``<zone>/L<level>.safetensors``。
* **R5-AC10**：``training_backend == "stub"``，Stub 路径端到端确定性跑通。

如何构造含分区场景：在**默认 Zone_Config 基线**下，``DEFAULT_ROUTING`` 把
``skill``/``action`` → ``coding``、``thought`` → ``knowledge``，而 ``sense`` 与
``credentials`` 不在路由表中 ⇒ 归入 Kernel（R2-AC3，天然保证 Kernel 非空）。因此
提供全部五类语料 ⇒ ``route_cells`` 产出一个 Kernel Cell + ``coding``/``knowledge``
两个分区 Cell ⇒ ``commands/train.py`` 的 ``use_grid`` 判定为真 ⇒ Vault 走
``write_grid_package`` 产出 Grid_Package。

为强制走 Stub 回退路径，本文件复用 ``tests/integration/conftest.py`` 的 autouse
``force_stub_backend`` 夹具（把四个真实后端的 ``is_available`` 改写为返回
``False``），使 ``ego train`` 在任何环境下都确定性地以 Stub_Backend 端到端跑通。
产物均落于 ``tmp_path``，不依赖网络或真实 ML 依赖。
"""

from __future__ import annotations

import json

from ego_pod.exit_codes import EXIT_OK

# 四个流水线阶段，用于断言 START/END 日志。
_STAGES = ("COLLECTOR", "COMPILER", "TRAINER", "VAULT")


def _five_category_contents() -> dict[str, list[tuple[str, str]]]:
    """构造五个 Data_Category 各含两个内容互异、非空文件的夹具内容。

    每个文件内容互不相同，避免 SHA-256 去重把它们合并，从而过滤后样本数足够
    （>= 3），可经增强阶段补足至最小样本数 16（R7-AC4）。所有文本均较长以
    确保过滤阶段的最小拼接长度（8）约束被满足。

    在默认 Zone_Config 基线下，``skill``/``action`` → ``coding``、``thought`` →
    ``knowledge``、``sense``/``credentials`` → Kernel，故五类齐全 ⇒ 一个 Kernel
    Cell + ``coding``/``knowledge`` 两个分区 Cell（含分区的网格场景）。
    """
    contents: dict[str, list[tuple[str, str]]] = {}
    samples = {
        "sense": ["今天清晨的空气格外清新，窗外有鸟鸣。", "午后阳光斜照进书房。"],
        "action": ["上午完成了一次五公里慢跑。", "下午整理了书架上的旧书。"],
        "thought": ["读完《流浪地球》后我在想文明的尺度。", "也许耐心本身就是一种答案。"],
        "skill": ["用三个步骤泡一壶手冲咖啡的要点。", "如何在十分钟内整理思路。"],
        "credentials": ["这是一段不含敏感信息的说明文本。", "另一段普通的备注内容。"],
    }
    for category, texts in samples.items():
        contents[category] = [
            (f"{category}_{idx}.txt", text) for idx, text in enumerate(texts)
        ]
    return contents


def test_grid_train_stub_end_to_end_success(
    tmp_path, make_category_dirs, write_config, run_train
):
    """含分区的默认 Zone_Config → ``ego train``(stub) 产出 Grid_Package。

    验证 R2-AC4 / R3-AC1 / R4-AC1 / R5-AC10：五类齐全 ⇒ Kernel + coding/L1 +
    knowledge/L1 三个 Cell ⇒ ``manifest.json`` 含正确的 ``grid`` 结构、磁盘含
    ``kernel.safetensors`` 与各 ``<zone>/L1.safetensors``、``training_backend ==
    "stub"``、退出码 0、四对阶段日志与网格摘要行齐全。
    """
    # --- 构造五类语料 + 合法配置（默认 Zone_Config 基线）-------------------
    sources = make_category_dirs(tmp_path, _five_category_contents())
    write_config(tmp_path / "ego-config.yaml", sources)

    # --- ego train（Stub 回退）端到端跑通至退出码 0 -------------------------
    rc, out, err = run_train()
    assert rc == EXIT_OK, f"grid ego train 未成功，stderr=\n{err}"

    # --- R11-AC5：四对 START/END 阶段日志 -----------------------------------
    for stage in _STAGES:
        assert f"] {stage} START" in out, f"缺少 {stage} START 阶段日志"
        assert f"] {stage} END" in out, f"缺少 {stage} END 阶段日志"

    # --- R2-AC4/AC6：网格摘要行含 Kernel 与两个分区 Cell --------------------
    summary_lines = [
        line for line in out.splitlines() if line.startswith("grid summary:")
    ]
    assert len(summary_lines) == 1, "网格摘要应恰为单行"
    grid_summary = summary_lines[0]
    assert "Kernel=" in grid_summary, f"网格摘要应含 Kernel 计数，实际：{grid_summary}"
    assert "(coding, L1)=" in grid_summary, (
        f"网格摘要应含 coding L1 计数，实际：{grid_summary}"
    )
    assert "(knowledge, L1)=" in grid_summary, (
        f"网格摘要应含 knowledge L1 计数，实际：{grid_summary}"
    )

    output_root = tmp_path / "ego-output"

    # --- R4-AC1：manifest.json 含 grid 键且结构正确 -------------------------
    manifest_path = output_root / "manifest.json"
    assert manifest_path.is_file(), "缺少 ego-output/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert "grid" in manifest, (
        f"网格产物的 manifest 应含 grid 键，实际 keys={sorted(manifest)}"
    )

    normal_grid = manifest["grid"]["normal"]
    # Kernel 相对路径。
    assert normal_grid["kernel"]["path"] == "adapters/kernel.safetensors", (
        f"Kernel 相对路径应为 adapters/kernel.safetensors，实际：{normal_grid['kernel']}"
    )

    # 两个分区各含一条 Level-1 Cell（路径相对该 Edition 目录）。
    zones = normal_grid["zones"]
    for zone in ("coding", "knowledge"):
        assert zone in zones, f"grid.normal.zones 应含分区 {zone}，实际：{sorted(zones)}"
        entries = zones[zone]
        assert isinstance(entries, list), f"分区 {zone} 应为列表，实际：{entries!r}"
        assert {"level": 1, "path": f"adapters/{zone}/L1.safetensors"} in entries, (
            f"分区 {zone} 应含 L1 条目，实际：{entries}"
        )

    # --- R5-AC10 / R12-AC9：training_backend == "stub" ----------------------
    assert manifest["training_backend"] == "stub"

    # --- R4-AC1/AC3：磁盘上三个 Cell 适配器与 config.json 齐全 --------------
    adapters_dir = output_root / "normal" / "adapters"
    assert (adapters_dir / "kernel.safetensors").is_file(), (
        "缺少 normal/adapters/kernel.safetensors"
    )
    assert (adapters_dir / "coding" / "L1.safetensors").is_file(), (
        "缺少 normal/adapters/coding/L1.safetensors"
    )
    assert (adapters_dir / "knowledge" / "L1.safetensors").is_file(), (
        "缺少 normal/adapters/knowledge/L1.safetensors"
    )
    assert (output_root / "normal" / "config.json").is_file(), (
        "缺少 ego-output/normal/config.json"
    )

    # Trainer 记录所选后端为 stub（回退确实生效）。
    assert "Selected backend: stub (STUB)" in out
