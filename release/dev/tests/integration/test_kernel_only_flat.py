"""kernel-only 退化为 Flat_Package 集成测试（ego-layered-adapter 任务 8.2）。

本文件验证 **二维网格特性的向后兼容退化路径**：当一次 ``ego train`` 的语料
**只命中 Kernel**（无任何 Functional_Zone 被填充）时，产出的 Ego_Package 必须是
一份**纯 Flat_Package**——与 ego-pod-mvp / ego-pod-real-trainer 既有产物逐字节
一致：``ego-output/normal/adapters/ego-lora.safetensors`` 单适配器 +
``manifest.json`` **不含 `grid` 键**，磁盘上既无 ``kernel.safetensors`` 亦无任何
``<zone>/`` 子目录。

如何构造 kernel-only 场景：在**默认 Zone_Config 基线**下，``sense`` 与
``credentials`` 两个 Data_Category 均未出现在路由表中（``DEFAULT_ROUTING`` 只把
``skill``/``action`` → ``coding``、``thought`` → ``knowledge``），故二者连同任何
未匹配样本一律归入 Kernel（R2-AC3）。因此**只**提供 ``sense`` + ``credentials``
两类语料 ⇒ ``route_cells`` 产出零个分区 Cell、仅一个 Kernel Cell ⇒
``commands/train.py`` 的 ``use_grid`` 判定为假 ⇒ Vault 走退化扁平路径
（``write_package``），产物为纯 Flat_Package（决策 C / R4-AC7 / R6-AC1/AC2）。

覆盖的验收标准：

* **R4-AC7**：仅含 Kernel 的网格被允许把 Kernel 适配器落于
  ``adapters/ego-lora.safetensors``，磁盘布局与现有 Flat_Package 完全一致。
* **R6-AC1**：不含 Grid_Descriptor 的 Flat_Package 被视为仅含单个 Kernel Cell。
* **R6-AC2**：退化的 Single_Cell_Package 与等价 Flat_Package 产生相同可观测行为。

为强制走 Stub 回退路径，本文件复用 ``tests/integration/conftest.py`` 的 autouse
``force_stub_backend`` 夹具（把四个真实后端的 ``is_available`` 改写为返回
``False``），使 ``ego train`` 在任何环境下都确定性地以 Stub_Backend 端到端跑通。
产物均落于 ``tmp_path``，不依赖网络或真实 ML 依赖。
"""

from __future__ import annotations

import json

from ego_pod.exit_codes import EXIT_OK
from ego_pod.runtime.inference import STUB_TRAINING_BACKEND


def _kernel_only_contents() -> dict[str, list[tuple[str, str]]]:
    """构造**只含** ``sense`` + ``credentials`` 两类、各三个内容互异非空文件的语料。

    在默认 Zone_Config 基线下，``sense`` 与 ``credentials`` 均不在路由表中 ⇒ 全部
    样本归入 Kernel，零个分区 Cell（kernel-only 退化场景）。每个文件内容互不相同
    以避免 SHA-256 去重把样本合并；六条过滤后样本足以让增强阶段从非空 Kernel 集合
    补足至最小样本数 16（6 + 6×5 = 36 ≥ 16，R7-AC4）。
    """
    samples = {
        "sense": [
            "清晨的空气清新宜人，窗外鸟鸣阵阵令人愉快。",
            "午后阳光斜照进安静的书房，尘埃在光束里浮动。",
            "傍晚的微风带着青草与泥土的气息缓缓拂过。",
        ],
        "credentials": [
            "这是一段不含任何敏感信息的普通说明文本，仅作占位之用。",
            "另一段平常的备注内容，描述了一次寻常的日程安排。",
            "第三段无害的记录，记述了一些日常事务的处理流程。",
        ],
    }
    return {
        category: [(f"{category}_{idx}.txt", text) for idx, text in enumerate(texts)]
        for category, texts in samples.items()
    }


def test_kernel_only_train_produces_flat_package(
    tmp_path, make_category_dirs, write_config, run_train
):
    """kernel-only 语料 → ``ego train``(stub) 产出纯 Flat_Package。

    验证 R4-AC7 / R6-AC1：只命中 Kernel（sense+credentials，零分区 Cell）时，
    产物为纯 Flat_Package——``manifest.json`` 无 ``grid`` 键、Kernel 适配器落于
    ``adapters/ego-lora.safetensors``、磁盘无 ``kernel.safetensors`` 亦无任何
    ``<zone>/`` 子目录，且 ``training_backend == "stub"``、退出码 0。
    """
    # --- 构造仅 sense+credentials 的语料 + 合法配置 -------------------------
    sources = make_category_dirs(tmp_path, _kernel_only_contents())
    write_config(tmp_path / "ego-config.yaml", sources)

    # --- ego train（Stub 回退）端到端跑通至退出码 0 -------------------------
    rc, out, err = run_train()
    assert rc == EXIT_OK, f"kernel-only ego train 未成功，stderr=\n{err}"

    # Trainer 记录所选后端为 stub（回退确实生效）。
    assert "Selected backend: stub (STUB)" in out

    # 网格摘要应只含 Kernel——零个分区 Cell（确认 kernel-only 退化的前提成立）。
    summary_lines = [
        line for line in out.splitlines() if line.startswith("grid summary:")
    ]
    assert len(summary_lines) == 1, "网格摘要应恰为单行"
    grid_summary = summary_lines[0]
    assert "Kernel=" in grid_summary, "网格摘要应含 Kernel 计数"
    assert "(coding" not in grid_summary and "(knowledge" not in grid_summary, (
        f"kernel-only 场景不应出现任何分区 Cell，实际摘要：{grid_summary}"
    )

    output_root = tmp_path / "ego-output"

    # --- R6-AC1：manifest.json 不含 grid 键（纯 Flat_Package）----------------
    manifest_path = output_root / "manifest.json"
    assert manifest_path.is_file(), "缺少 ego-output/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "grid" not in manifest, (
        f"kernel-only 退化产物的 manifest 不应含 grid 键，实际 keys={sorted(manifest)}"
    )

    # --- R8-AC7 / R12-AC9：training_backend == "stub" -----------------------
    assert manifest["training_backend"] == STUB_TRAINING_BACKEND == "stub"

    # --- R4-AC7：Kernel 适配器落于扁平的 ego-lora.safetensors ----------------
    adapters_dir = output_root / "normal" / "adapters"
    assert (adapters_dir / "ego-lora.safetensors").is_file(), (
        "缺少扁平的 normal/adapters/ego-lora.safetensors"
    )
    assert (output_root / "normal" / "config.json").is_file()

    # --- 磁盘布局与 Flat_Package 完全一致：无网格专属产物 -------------------
    # 既无 kernel.safetensors（网格包的 Kernel 文件名），
    assert not (adapters_dir / "kernel.safetensors").exists(), (
        "纯 Flat_Package 不应含 kernel.safetensors"
    )
    # 亦无任何 <zone>/ 子目录（分区 Cell 的落点）。
    subdirs = [child.name for child in adapters_dir.iterdir() if child.is_dir()]
    assert subdirs == [], f"纯 Flat_Package 的 adapters/ 不应含任何分区子目录，实际：{subdirs}"

    # adapters/ 下恰好只有 ego-lora.safetensors 单个文件。
    entries = sorted(child.name for child in adapters_dir.iterdir())
    assert entries == ["ego-lora.safetensors"], (
        f"adapters/ 应只含 ego-lora.safetensors，实际：{entries}"
    )
