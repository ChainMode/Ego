"""`Ego School` 冷启动导入（OpenAI / iFay-Ledger / Markdown / JSONL）与伴生增量蒸馏 + Vault 回滚端到端集成测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ego_pod.cli import main
from ego_pod.exit_codes import EXIT_OK, VaultError
from ego_pod.school import evolve_workspace, import_school_data, load_imported_school_records
from ego_pod.service import EgoCognitiveEngine, EgoServiceServer
from ego_pod.vault.history import list_vault_snapshots
from ego_pod.vault.signing import verify_package_signature


def test_ego_school_import_openai_and_ifay_ledger_with_pii_scrubbing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """验证 OpenAI conversations.json 与 ifay-ledger 导入：仅提取人类原型发言与纠偏对，且彻底擦除密钥与个人 PII。"""
    ws = tmp_path / "school-ws"
    fake_sk_token = "sk-" + "example_placeholder_token_987654"
    raw_email = "prime.owner@ifay-private.org"
    raw_phone = "13987654321"
    bot_slop = "UNIQUE_ASSISTANT_SLOP_DO_NOT_LEARN_9911"

    openai_export = [
        {
            "title": "Code Review Workflow",
            "mapping": {
                "root": {"id": "root", "parent": None, "children": ["m1"], "message": None},
                "m1": {
                    "id": "m1",
                    "parent": "root",
                    "children": ["m2"],
                    "message": {
                        "author": {"role": "system"},
                        "content": {"parts": ["You are ChatGPT."]},
                    },
                },
                "m2": {
                    "id": "m2",
                    "parent": "m1",
                    "children": ["m3"],
                    "message": {
                        "author": {"role": "user"},
                        "content": {
                            "parts": [
                                f"Draft a commit policy note for my repo (contact {raw_email}, token {fake_sk_token})."
                            ]
                        },
                    },
                },
                "m3": {
                    "id": "m3",
                    "parent": "m2",
                    "children": ["m4"],
                    "message": {
                        "author": {"role": "assistant"},
                        "content": {"parts": [f"Sure! {bot_slop}"]},
                    },
                },
                "m4": {
                    "id": "m4",
                    "parent": "m3",
                    "children": [],
                    "message": {
                        "author": {"role": "user"},
                        "content": {
                            "parts": [
                                f"不对，太啰嗦了！一律使用 Conventional Commits 并附带 DCO Signed-off-by，严禁提交大体积二进制文件（紧急联系 {raw_phone}）。"
                            ]
                        },
                    },
                },
            },
        }
    ]
    openai_file = tmp_path / "conversations.json"
    openai_file.write_text(json.dumps(openai_export, ensure_ascii=False), encoding="utf-8")

    rc1 = main(
        [
            "school",
            "import",
            "--format",
            "openai",
            "--input",
            str(openai_file),
            "--workspace",
            str(ws),
        ]
    )
    assert rc1 == EXIT_OK
    out1 = capsys.readouterr().out
    assert "correction_pairs=1" in out1

    # 2) 导入 iFay 主体回流账本（含 human_correction + assistant 噪音 + 显式红线）
    ledger_file = tmp_path / "primary_body_events.jsonl"
    ledger_events = [
        {
            "event_id": "ev-001",
            "event_type": "assistant_draft",
            "author": "assistant",
            "content": bot_slop,
        },
        {
            "event_id": "ev-002",
            "event_type": "human_correction",
            "prompt": "How should database migrations be executed?",
            "draft_reply": "Just run DROP TABLE and recreate it in prod.",
            "corrected_reply": "Always use forward-compatible additive migrations with a verified backup snapshot first.",
            "redline": "Never auto-run destructive SQL migrations without explicit backup verification",
            "domain": "coding",
        },
        {
            "event_id": "ev-003",
            "event_type": "human_turn",
            "author": "human_prime",
            "prompt": "Daily reading summary format",
            "human_reply": "优先用三条精炼要点总结核心论点与反例边界。",
            "domain": "knowledge",
        },
    ]
    ledger_file.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in ledger_events) + "\n",
        encoding="utf-8",
    )

    rc2 = main(
        [
            "school",
            "import",
            "--format",
            "ifay-ledger",
            "--input",
            str(ledger_file),
            "--workspace",
            str(ws),
        ]
    )
    assert rc2 == EXIT_OK
    capsys.readouterr()

    # 重复导入同一份账本应幂等跳过（0 新增）
    dup_res = import_school_data(ledger_file, fmt="ifay-ledger", workspace=ws)
    assert dup_res.imported_turns == 0
    assert dup_res.correction_pairs == 0
    assert dup_res.skipped_duplicates >= 2

    # 3) 核查工作区内落盘的所有文件：零明文密钥/邮箱/手机号，且绝不包含未获认可的 assistant 废话作为输出目标！
    records = load_imported_school_records(ws)
    assert len(records) == 4  # 1 user turn + 1 openai correction + 1 ledger correction + 1 ledger human_turn
    for p in ws.rglob("*"):
        if p.is_file():
            txt = p.read_text(encoding="utf-8", errors="replace")
            assert fake_sk_token not in txt
            assert raw_email not in txt
            assert raw_phone not in txt
    for rec in records:
        assert bot_slop not in rec["output"]


def test_ego_school_markdown_and_jsonl_import_and_companion_evolve_with_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """全闭环验证：冷启动导入 -> 首次 evolve 签发 v1 -> 新账本增量 evolve 晋升 v2 -> 探针失败自动回滚 -> CLI 一键回滚 v1。"""
    ws = tmp_path / "companion-ws"
    monkeypatch.setenv("EGO_SIGNING_SEED_HEX", "55" * 32)
    monkeypatch.setenv("EGO_SIGNER_FAY_ID", "ifay_schoolprimeaaaaaaa")

    # 1) 冷启动导入 Markdown 笔记 + JSONL 对话
    notes_md = tmp_path / "prime_notes.md"
    notes_md.write_text(
        "Human: 总结代码评审规范\n"
        "Assistant: 我可以写三千字长文。\n"
        "Human: 不要写长篇空话，直接列出架构边界、测试覆盖率与回滚方案三项结论。\n",
        encoding="utf-8",
    )
    chat_jsonl = tmp_path / "turns.jsonl"
    chat_jsonl.write_text(
        "\n".join(
            [
                json.dumps({"role": "user", "content": "偏好使用不可变数据结构与纯函数消除副作用。"}, ensure_ascii=False),
                json.dumps({"role": "assistant", "content": "Generic bot reply ignored."}, ensure_ascii=False),
                json.dumps({"prompt": "Standup update style", "corrected_reply": "Keep standup updates to 3 bullet points: Done, Next, Blockers."}, ensure_ascii=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert main(["school", "import", "--format", "markdown", "--input", str(notes_md), "--workspace", str(ws)]) == EXIT_OK
    assert main(["school", "import", "--format", "jsonl", "--input", str(chat_jsonl), "--workspace", str(ws)]) == EXIT_OK
    capsys.readouterr()

    # 2) 运行首次伴生进化（v1 诞生并完成 Ed25519 签名）
    assert main(["school", "evolve", str(ws)]) == EXIT_OK
    capsys.readouterr()

    pkg_root = ws / "ego-output"
    assert verify_package_signature(pkg_root).status == "verified"
    eng_v1 = EgoCognitiveEngine(pkg_root)
    assert eng_v1.intuition("Standup update style")["handled_locally"] is True
    eng_v1.close()

    # 3) 模拟 iFay 主体回流新的 human_correction 事件并运行第二轮 `ego school evolve --ledger`
    ledger_v2 = tmp_path / "sync_batch_2.jsonl"
    ledger_v2.write_text(
        json.dumps(
            {
                "event_id": "sync-v2-01",
                "event_type": "human_correction",
                "prompt": "When scheduling deep work blocks, what is my rule?",
                "draft_reply": "Accept any meeting invite between 9am and 11am.",
                "corrected_reply": "Protect 09:00-11:30 every morning for uninterrupted deep work; decline non-urgent invites.",
                "redline": "严禁在上午深度工作时段自动接受普通会议邀请",
                "domain": "social",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    assert (
        main(["school", "evolve", str(ws), "--ledger", str(ledger_v2)]) == EXIT_OK
    )
    evo_out = capsys.readouterr().out
    assert "evolve promoted" in evo_out
    assert verify_package_signature(pkg_root).status == "verified"

    # 核查 Vault 历史已保留上一版快照（0.2.0-0001）
    snaps = list_vault_snapshots(pkg_root)
    assert len(snaps) == 1
    v1_snap_id = snaps[0].snapshot_id

    # 验证 v2 已习得新的深度工作纠偏与红线，并可通过真实 HTTP 服务响应
    eng_v2 = EgoCognitiveEngine(pkg_root)
    with EgoServiceServer(eng_v2, port=0) as srv:
        assert srv.port > 0
        ans_v2 = eng_v2.intuition("When scheduling deep work blocks, what is my rule?")
        assert ans_v2["handled_locally"] is True
        assert "09:00-11:30" in ans_v2["reply"]

        rl_v2 = eng_v2.intuition("请在上午深度工作时段自动接受普通会议邀请")
        assert rl_v2["refused"] is True
        assert rl_v2["escalate_to_llm"] is False

    # 4) 验证晋升前一致性探针守门失败时，自动回滚并拒绝晋升退化包！
    ledger_v3 = tmp_path / "sync_batch_3.jsonl"
    ledger_v3.write_text(
        json.dumps(
            {
                "event_id": "sync-v3-bad",
                "event_type": "human_turn",
                "author": "human_prime",
                "prompt": "Minor note",
                "human_reply": "A small extra observation during afternoon walk.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(VaultError, match="Consistency Gate Rejected"):
        evolve_workspace(
            ws,
            ledger_path=ledger_v3,
            min_consistency_score=0.70,
            consistency_probe_fn=lambda _root: 0.35,  # 模拟一致性探针跌破 0.70 红线
        )

    # 自动回滚后，当前活动包依然通过验签且保持 v2 的完整能力
    assert verify_package_signature(pkg_root).status == "verified"
    eng_after_guard = EgoCognitiveEngine(pkg_root)
    assert "09:00-11:30" in eng_after_guard.intuition("When scheduling deep work blocks, what is my rule?")["reply"]
    eng_after_guard.close()

    # 5) 验证用户手动执行 `ego vault rollback <v1_snap_id>` 可一键回滚至最初 v1 版本！
    monkeypatch.chdir(ws)
    assert main(["vault", "rollback", v1_snap_id]) == EXIT_OK
    assert verify_package_signature(pkg_root).status == "verified"
    eng_rolled_back = EgoCognitiveEngine(pkg_root)
    try:
        # v1 仍记得最初的 Standup 规范，但不含 v2 才引入的 09:00-11:30 规则
        assert "3 bullet points" in eng_rolled_back.intuition("Standup update style")["reply"]
        v2_query_on_v1 = eng_rolled_back.intuition("When scheduling deep work blocks, what is my rule?")
        assert v2_query_on_v1["handled_locally"] is False
    finally:
        eng_rolled_back.close()
