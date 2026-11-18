"""Dojo 全流程端到端集成测试：深层 L1..L3 蒸馏、多规格构建、基座变更重训、--update 与 Vault 回滚。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from ego_pod.cli import main
from ego_pod.exit_codes import EXIT_OK
from ego_pod.vault.signing import verify_package_signature


def _seed_rich_workspace_data(root: Path) -> None:
    """写入覆盖五类 Data_Category 且具有明显专精度梯度的真实形态本地语料。"""
    sense_dir = root / "ego-data" / "sense"
    action_dir = root / "ego-data" / "action"
    thought_dir = root / "ego-data" / "thought"
    skill_dir = root / "ego-data" / "skill"
    cred_dir = root / "ego-data" / "credentials"
    for d in (sense_dir, action_dir, thought_dir, skill_dir, cred_dir):
        d.mkdir(parents=True, exist_ok=True)

    for idx in range(4):
        (sense_dir / f"sense_{idx}.txt").write_text(
            f"清晨在书桌前整理第 {idx + 1} 份日常观察记录，保持专注与平静。",
            encoding="utf-8",
        )
    (cred_dir / "cred_0.txt").write_text(
        "配置管理原则：绝不在代码库硬编码 sk-1234567890abcdef12345678 密钥。",
        encoding="utf-8",
    )

    # coding 分区（skill + action）：浅层 L1、中层 L2、深层代码推理 L3
    (action_dir / "act_simple.txt").write_text(
        "运行本地测试命令查看输出日志状态。", encoding="utf-8"
    )
    (skill_dir / "skill_mid.txt").write_text(
        "在 Python 模块中采用不可变 dataclass 封装配置参数，因此可以在编译与训练阶段之间消除隐式状态耦合。",
        encoding="utf-8",
    )
    (skill_dir / "skill_deep.txt").write_text(
        "```python\ndef verify_jcs_ed25519(payload: dict, sig: bytes) -> bool:\n"
        "    # 基于 RFC 8785 JCS 规范化与 HKDF-SHA256 派生子密钥校验签名不变量\n"
        "    return True\n```\n"
        "时间复杂度与确定性边界证明：因为键按 UTF-16 码元排序，因此跨语言实现逐字节等价。",
        encoding="utf-8",
    )

    # knowledge 分区（thought）：不同深度的思考样本
    for idx in range(8):
        if idx < 3:
            body = f"阅读笔记第 {idx} 篇：记录今日关于工具简明性的简短想法。"
        elif idx < 6:
            body = (
                f"架构思考第 {idx} 篇：因为终端设备算力受限，因此需要通过按需加载机制解耦核心人格与专精技能边界。"
            )
        else:
            body = (
                f"深度理论推导第 {idx} 篇：在多维稀疏网格模型中，证明连续分位数压缩算法在拜占庭噪声与小样本退化条件下的单调一致性不变量，权衡秩容量与体积上限。"
            )
        (thought_dir / f"thought_{idx}.txt").write_text(body, encoding="utf-8")


def test_dojo_end_to_end_multi_edition_deep_grid_update_and_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """全链路验证：init -> 多规格深层网格训练（含并行模式与签名）-> 基座变更检测 -> --update -> vault 回滚 -> run。"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("EGO_SIGNING_SEED_HEX", "33" * 32)
    monkeypatch.setenv("EGO_SIGNER_FAY_ID", "ifay_edfeedfacecafebeefaa")

    _seed_rich_workspace_data(tmp_path)

    # 1. 初始化并配置 lite + normal + plus 三档规格
    assert main(["init"]) == EXIT_OK
    cfg_path = tmp_path / "ego-config.yaml"
    raw_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw_cfg["editions"] = {
        "lite": {"base": "qwen3.5-0.8B", "description": "轻量端侧"},
        "normal": {"base": "qwen3.5-0.8B", "description": "日常均衡"},
        "plus": {"base": "qwen3.5-4B", "description": "深度全网格"},
    }
    raw_cfg["training"]["epochs"] = 1
    cfg_path.write_text(yaml.safe_dump(raw_cfg, allow_unicode=True), encoding="utf-8")
    capsys.readouterr()

    # 2. 执行多规格并行修炼（--parallel-editions），验证 25%/50%/85%/100% 进度与 cultivation summary
    assert main(["train", "--parallel-editions"]) == EXIT_OK
    train_out = capsys.readouterr().out
    for pct in ("25%", "50%", "85%", "100%"):
        assert pct in train_out
    assert "cultivation summary:" in train_out

    out_root = tmp_path / "ego-output"
    manifest_v1 = json.loads((out_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_v1["editions"] == ["lite", "normal", "plus"]

    # 验证深层网格分层结果：lite 最高 L1，normal 最高 L2，plus 产出 L1..L3！
    lite_coding = [e["level"] for e in manifest_v1["grid"]["lite"]["zones"]["coding"]]
    normal_coding = [e["level"] for e in manifest_v1["grid"]["normal"]["zones"]["coding"]]
    plus_coding = [e["level"] for e in manifest_v1["grid"]["plus"]["zones"]["coding"]]
    assert lite_coding == [1]
    assert normal_coding == [1, 2]
    assert plus_coding == [1, 2, 3]

    # 验证多规格包全量 Ed25519 签名与摘要通过
    assert verify_package_signature(out_root).status == "verified"

    # 3. 切换 normal 基座为 gemma4-2B 并追加新笔记后运行 `ego train --update`
    raw_cfg["editions"] = {
        "normal": {"base": "gemma4-2B", "description": "切换基座后的日常规格"}
    }
    cfg_path.write_text(yaml.safe_dump(raw_cfg, allow_unicode=True), encoding="utf-8")
    (tmp_path / "ego-data" / "skill" / "skill_new_inc.txt").write_text(
        "增量新增的一条模块化单元测试工程实践记录。", encoding="utf-8"
    )

    assert main(["train", "--update"]) == EXIT_OK
    inc_out = capsys.readouterr().out
    assert "gemma4-2B" in inc_out  # 记录了基座模型切换事件
    assert "incremental" in inc_out.lower() or "unchanged=" in inc_out

    normal_cfg_v2 = json.loads(
        (out_root / "normal" / "config.json").read_text(encoding="utf-8")
    )
    assert normal_cfg_v2["base_model"] == "gemma4-2B"
    assert verify_package_signature(out_root).status == "verified"

    # 4. 验证 .history 已归档第一版三规格包，并可通过 ego vault list 查看
    assert main(["vault", "list"]) == EXIT_OK
    vault_list_out = capsys.readouterr().out
    assert "0.2.0-0001" in vault_list_out

    # 5. 回滚至第一版快照 0.2.0-0001，并用 ego run --edition plus --zone coding:2 加载验证
    assert main(["vault", "rollback", "0.2.0-0001"]) == EXIT_OK
    capsys.readouterr()

    from ego_pod.commands.run import run_repl
    import argparse
    import io

    repl_out = io.StringIO()
    rc_run = run_repl(
        argparse.Namespace(edition="plus", zone=["coding:2"], kernel_only=False),
        cwd=tmp_path,
        stdin=io.StringIO("请分析代码\n:quit\n"),
        stdout=repl_out,
        stderr=io.StringIO(),
    )
    assert rc_run == EXIT_OK
