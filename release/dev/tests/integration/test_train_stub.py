"""``ego train`` 端到端集成测试（Stub_Backend，任务 12.1）。

本文件验证 Requirement 11（``ego train`` 端到端编排）与 Requirement 10（Vault
产出）在 Stub_Backend 下的成功路径：构造五个 Data_Category 的本地数据源目录与
一份合法 ``ego-config.yaml``，运行完整的 ``ego train`` 流水线，断言：

* **R11-AC3 / R11-AC1**：四阶段（Collector → Compiler → Trainer → Vault）全部
  成功，``ego train`` 以退出码 0 结束。
* **R10-AC6**：产出 ``ego-output/manifest.json``、``ego-output/normal/config.json``
  与（默认五类 ⇒ Grid_Package）``ego-output/normal/adapters/`` 下的
  ``kernel.safetensors`` + ``coding/L1.safetensors`` + ``knowledge/L1.safetensors``，
  且 ``manifest.json`` 含 ``grid`` 键。
* **R11-AC5**：标准输出包含四对 START/END 阶段日志
  （COLLECTOR / COMPILER / TRAINER / VAULT 各一对）。
* **R4-AC6**：标准输出包含固定五行采集摘要（五个 Data_Category 各一行）。
* **R7-AC3**：标准输出包含单行过滤摘要。

集成测试通过 ``conftest`` 的 autouse ``force_stub_backend`` 夹具强制 Trainer 回退
到 Stub_Backend，从而在装有 torch/peft 的机器上仍能确定性地端到端跑通；产物均
落于 ``tmp_path``，不依赖网络或真实 ML 依赖。
"""

from __future__ import annotations

import re
from pathlib import Path

from ego_pod.exit_codes import EXIT_OK

# 采集摘要单行格式，例如 "sense: 2 files, 11 bytes"。
_SUMMARY_RE = re.compile(
    r"^(sense|action|thought|skill|credentials): (\d+) files, (\d+) bytes$"
)

# 四个流水线阶段，用于断言 START/END 日志。
_STAGES = ("COLLECTOR", "COMPILER", "TRAINER", "VAULT")


def _five_category_contents() -> dict[str, list[tuple[str, str]]]:
    """构造五个 Data_Category 各含两个内容互异、非空文件的夹具内容。

    每个文件内容互不相同，避免 SHA-256 去重把它们合并，从而过滤后样本数足够
    （>= 3），可经增强阶段补足至最小样本数 16（R7-AC4）。所有文本均较长以
    确保过滤阶段的最小拼接长度（8）约束被满足。
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


def test_ego_train_stub_end_to_end_success(
    tmp_path, make_category_dirs, write_config, run_train
):
    """完整 ``ego train`` 在 Stub_Backend 下成功并产出齐全 Ego_Package。"""

    # --- 构造五目录 fixture 与合法配置 -------------------------------------
    sources = make_category_dirs(tmp_path, _five_category_contents())
    write_config(tmp_path / "ego-config.yaml", sources)

    # --- 运行完整 ego train --------------------------------------------------
    rc, out, err = run_train()

    # --- R11-AC3：退出码 0 ---------------------------------------------------
    assert rc == EXIT_OK, f"ego train 未成功，stderr=\n{err}"

    # --- R10-AC6 / R4-AC1：产物齐全（默认五类 ⇒ Grid_Package）---------------
    # 默认 Zone_Config 基线把 skill/action→coding、thought→knowledge、
    # sense/credentials→Kernel，故五类齐全 ⇒ 含分区的网格包：Kernel + coding/L1 +
    # knowledge/L1，产物为 kernel.safetensors + 各 <zone>/L1.safetensors，且
    # manifest 含 grid 键（不再是扁平的 ego-lora.safetensors）。
    import json

    output_root = tmp_path / "ego-output"
    manifest_path = output_root / "manifest.json"
    config_json = output_root / "normal" / "config.json"
    adapters_dir = output_root / "normal" / "adapters"
    assert manifest_path.is_file(), "缺少 ego-output/manifest.json"
    assert config_json.is_file(), "缺少 ego-output/normal/config.json"
    assert (adapters_dir / "kernel.safetensors").is_file(), (
        "缺少 normal/adapters/kernel.safetensors"
    )
    assert (adapters_dir / "coding" / "L1.safetensors").is_file(), (
        "缺少 normal/adapters/coding/L1.safetensors"
    )
    assert (adapters_dir / "knowledge" / "L1.safetensors").is_file(), (
        "缺少 normal/adapters/knowledge/L1.safetensors"
    )

    # 含分区 ⇒ manifest 含 grid 键（Grid_Package）。
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "grid" in manifest, (
        f"含分区的网格产物 manifest 应含 grid 键，实际 keys={sorted(manifest)}"
    )

    # --- R11-AC5：四对 START/END 阶段日志 -----------------------------------
    for stage in _STAGES:
        assert f"] {stage} START" in out, f"缺少 {stage} START 阶段日志"
        assert f"] {stage} END" in out, f"缺少 {stage} END 阶段日志"

    # --- R4-AC6：固定五行采集摘要 -------------------------------------------
    summary_lines = [
        line for line in out.splitlines() if _SUMMARY_RE.match(line)
    ]
    assert [_SUMMARY_RE.match(l).group(1) for l in summary_lines] == [
        "sense",
        "action",
        "thought",
        "skill",
        "credentials",
    ]

    # --- R7-AC3：单行过滤摘要 -----------------------------------------------
    filter_lines = [
        line for line in out.splitlines() if line.startswith("filter summary:")
    ]
    assert len(filter_lines) == 1, "过滤摘要应恰为单行"
    # 摘要含四项计数 key（R7-AC3）。
    for key in ("dropped_short", "merged_dup", "total_before", "total_after"):
        assert key in filter_lines[0]


def test_ego_train_stub_artifacts_have_valid_content(
    tmp_path, make_category_dirs, write_config, run_train
):
    """成功训练后 manifest/config 内容符合 R10-AC2/AC4 字段约束。"""

    import json

    sources = make_category_dirs(tmp_path, _five_category_contents())
    write_config(tmp_path / "ego-config.yaml", sources)

    rc, out, err = run_train()
    assert rc == EXIT_OK, err

    output_root = tmp_path / "ego-output"
    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    config = json.loads(
        (output_root / "normal" / "config.json").read_text(encoding="utf-8")
    )

    # R10-AC2：manifest 字段。
    assert manifest["base_model"] == "qwen3.5-0.8B"
    assert "normal" in manifest["editions"]
    assert isinstance(manifest["version"], str) and manifest["version"]
    # R12-AC9 的补充字段：stub 训练标记。
    assert manifest["training_backend"] == "stub"

    # R10-AC4：config 字段。
    assert config["base_model"] == "qwen3.5-0.8B"
    assert config["quantization"] == "4bit"
    assert isinstance(config["lora_rank"], int) and config["lora_rank"] > 0
    assert isinstance(config["lora_alpha"], int) and config["lora_alpha"] > 0

    # Vault 在标准输出打印 Ego_Package 根目录绝对路径（R10-AC6）。
    assert str(output_root.resolve()) in out

    # Trainer 在标准输出记录所选后端为 stub（R8-AC2）。
    assert "Selected backend: stub (STUB)" in out
