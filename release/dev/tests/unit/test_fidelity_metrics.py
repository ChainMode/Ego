"""`ego_pod.fidelity` 人格保真度与跨 Cell 一致性度量单元测试 + 隐私非泄露属性测试。"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ego_pod.fidelity import (
    FIDELITY_SCHEMA_VERSION,
    HoldoutMetrics,
    build_fidelity_report,
    compute_cell_consistency,
    compute_holdout_metrics,
    evaluate_corpus_fidelity,
    evaluate_style_probes,
    split_holdout_deterministic,
    validate_fidelity_report,
)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    text = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"
    path.write_text(text, encoding="utf-8")
    return path


def _sample_rows(count: int, prefix: str = "样本") -> list[dict]:
    return [
        {
            "instruction": f"请回答问题 {i}",
            "input": "",
            "output": f"{prefix}-回答内容-{i}：保持简洁清晰与一贯的表达风格。",
            "category": "thought",
        }
        for i in range(count)
    ]


# ---------------------------------------------------------------------------
# 1. 确定性留出集切分与 holdout 度量
# ---------------------------------------------------------------------------


def test_split_holdout_is_deterministic_across_calls(tmp_path: Path) -> None:
    rows = _sample_rows(16)
    corpus_path = _write_jsonl(tmp_path / "corpus.jsonl", rows)

    train_a, holdout_a = split_holdout_deterministic(corpus_path)
    train_b, holdout_b = split_holdout_deterministic(corpus_path)

    assert train_a == train_b
    assert holdout_a == holdout_b
    assert len(train_a) + len(holdout_a) == 16
    assert len(holdout_a) >= 2


def test_compute_holdout_metrics_insufficient_data_for_tiny_corpus(
    tmp_path: Path,
) -> None:
    corpus_path = _write_jsonl(tmp_path / "tiny.jsonl", _sample_rows(4))

    metrics = compute_holdout_metrics(corpus_path, min_total_samples=8)

    assert isinstance(metrics, HoldoutMetrics)
    assert metrics.status == "insufficient_data"
    assert metrics.holdout_sample_count == 0
    assert metrics.holdout_loss is None
    assert metrics.holdout_perplexity is None


def test_compute_holdout_metrics_measured_produces_finite_loss_and_perplexity(
    tmp_path: Path,
) -> None:
    corpus_path = _write_jsonl(tmp_path / "enough.jsonl", _sample_rows(16))

    metrics = compute_holdout_metrics(corpus_path)

    assert metrics.status == "measured"
    assert metrics.holdout_sample_count >= 2
    assert metrics.holdout_loss is not None and math.isfinite(metrics.holdout_loss)
    assert metrics.holdout_loss >= 0.0
    assert metrics.holdout_perplexity is not None and math.isfinite(
        metrics.holdout_perplexity
    )
    assert metrics.holdout_perplexity == pytest.approx(
        math.exp(metrics.holdout_loss), rel=1e-5
    )


def test_compute_holdout_metrics_accepts_custom_loss_fn() -> None:
    rows = _sample_rows(12)
    metrics = compute_holdout_metrics(rows, loss_fn=lambda holdout: 1.2345)

    assert metrics.status == "measured"
    assert metrics.holdout_sample_count >= 2
    assert metrics.holdout_loss == pytest.approx(1.2345)
    assert metrics.holdout_perplexity == pytest.approx(math.exp(1.2345), rel=1e-5)


# ---------------------------------------------------------------------------
# 2. 跨 Cell 响应一致性（compute_cell_consistency）
# ---------------------------------------------------------------------------


def test_compute_cell_consistency_identical_responses_returns_one() -> None:
    responses = {
        "kernel": ["我倾向于先写测试再写实现。", "简洁比冗长更重要。"],
        "kernel+coding_L1": ["我倾向于先写测试再写实现。", "简洁比冗长更重要。"],
        "kernel+knowledge_L1": ["我倾向于先写测试再写实现。", "简洁比冗长更重要。"],
    }
    assert compute_cell_consistency(responses) == 1.0


def test_compute_cell_consistency_disjoint_responses_returns_zero() -> None:
    responses = {
        "kernel": ["abcdefg", "hijklmn"],
        "kernel+coding_L1": ["1234567", "890!@#$"],
    }
    assert compute_cell_consistency(responses) == 0.0


def test_compute_cell_consistency_partial_overlap_in_unit_interval() -> None:
    responses = {
        "kernel": ["先写单元测试，再实现核心逻辑。"],
        "kernel+coding_L1": ["先写单元测试，然后补充边界断言。"],
    }
    score = compute_cell_consistency(responses)
    assert score is not None
    assert 0.0 < score < 1.0


def test_compute_cell_consistency_returns_none_for_single_cell_or_empty() -> None:
    assert compute_cell_consistency(None) is None
    assert compute_cell_consistency({}) is None
    assert compute_cell_consistency({"kernel": ["单独一个配置无法做跨组合对比"]}) is None
    assert compute_cell_consistency({"a": [], "b": []}) is None


# ---------------------------------------------------------------------------
# 3. 本地配对风格探针评估（evaluate_style_probes）
# ---------------------------------------------------------------------------


def test_evaluate_style_probes_with_boolean_and_scorer() -> None:
    assert evaluate_style_probes(None) is None
    assert evaluate_style_probes([]) is None

    pre_scored = [
        {"preferred_won": True},
        {"preferred_won": True},
        {"preferred_won": False},
        {"preferred_won": True},
    ]
    assert evaluate_style_probes(pre_scored) == pytest.approx(0.75)

    raw_probes = [
        {"prompt": "p1", "chosen": "短句直给", "rejected": "非常冗长客套的套话套话"},
        {"prompt": "p2", "chosen": "清晰结论", "rejected": "非常冗长客套的套话套话"},
    ]
    # 偏好更短文本的本地评分器
    win_rate = evaluate_style_probes(
        raw_probes, scorer=lambda _prompt, candidate: -float(len(candidate))
    )
    assert win_rate == 1.0


# ---------------------------------------------------------------------------
# 4. build_fidelity_report 结构契约、漂移计算与隐私非泄露属性测试
# ---------------------------------------------------------------------------


def test_build_fidelity_report_exact_wire_schema_and_drift() -> None:
    baseline = {
        "schema_version": "1.0",
        "status": "measured",
        "holdout_sample_count": 4,
        "holdout_loss": 1.5000,
        "holdout_perplexity": round(math.exp(1.5), 6),
        "cell_consistency_score": 0.90,
        "probe_win_rate": None,
        "drift_from_baseline": None,
    }
    holdout = HoldoutMetrics(
        status="measured",
        holdout_sample_count=4,
        holdout_loss=1.2345,
        holdout_perplexity=round(math.exp(1.2345), 6),
    )
    report = build_fidelity_report(
        holdout=holdout,
        cell_consistency_score=0.95,
        probe_win_rate=0.875,
        baseline_report=baseline,
    )

    assert list(report.keys()) == [
        "schema_version",
        "status",
        "holdout_sample_count",
        "holdout_loss",
        "holdout_perplexity",
        "cell_consistency_score",
        "probe_win_rate",
        "drift_from_baseline",
    ]
    assert report["schema_version"] == FIDELITY_SCHEMA_VERSION == "1.0"
    assert report["status"] == "measured"
    assert report["holdout_sample_count"] == 4
    assert report["holdout_loss"] == pytest.approx(1.2345)
    assert report["cell_consistency_score"] == pytest.approx(0.95)
    assert report["probe_win_rate"] == pytest.approx(0.875)
    assert report["drift_from_baseline"] == pytest.approx(1.2345 - 1.5)
    validate_fidelity_report(report)


def test_evaluate_corpus_fidelity_reads_previous_manifest_for_drift(
    tmp_path: Path,
) -> None:
    corpus_a = _write_jsonl(tmp_path / "corpus_a.jsonl", _sample_rows(16, "第一版"))
    prev_manifest = tmp_path / "manifest.json"

    first_report = evaluate_corpus_fidelity(
        corpus_a, previous_manifest_path=prev_manifest
    )
    assert first_report["status"] == "measured"
    assert first_report["drift_from_baseline"] is None

    prev_manifest.write_text(
        json.dumps({"fidelity_report": first_report}, ensure_ascii=False),
        encoding="utf-8",
    )

    corpus_b = _write_jsonl(tmp_path / "corpus_b.jsonl", _sample_rows(16, "第二版微调"))
    second_report = evaluate_corpus_fidelity(
        corpus_b, previous_manifest_path=prev_manifest
    )
    assert second_report["status"] == "measured"
    assert second_report["drift_from_baseline"] is not None
    assert math.isfinite(second_report["drift_from_baseline"])


def test_validate_fidelity_report_rejects_text_leakage_or_extra_keys() -> None:
    leaked = {
        "schema_version": "1.0",
        "status": "measured",
        "holdout_sample_count": 4,
        "holdout_loss": 1.0,
        "holdout_perplexity": 2.718282,
        "cell_consistency_score": None,
        "probe_win_rate": None,
        "drift_from_baseline": None,
        "raw_samples": ["人类原型私密日记原文"],
    }
    with pytest.raises(ValueError, match="fidelity_report"):
        validate_fidelity_report(leaked)


@given(
    secret_tokens=st.lists(
        st.text(
            alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789中文字母私密日记",
            min_size=10,
            max_size=32,
        ).filter(lambda s: s not in {"insufficient_data", "schema_version"}),
        min_size=10,
        max_size=20,
    )
)
@settings(max_examples=50, deadline=None)
def test_privacy_redline_zero_raw_text_leaks_into_fidelity_report(
    secret_tokens: list[str],
) -> None:
    """属性测试：任意人类原型私密语料/探针文本绝不出现在序列化后的 fidelity_report 中。"""
    rows = [
        {
            "instruction": f"PROMPT_SECRET_{tok}",
            "input": f"INPUT_SECRET_{tok}",
            "output": f"OUTPUT_SECRET_{tok}",
        }
        for tok in secret_tokens
    ]
    probes = [
        {
            "prompt": f"PROBE_Q_{secret_tokens[0]}",
            "chosen": f"PROBE_CHOSEN_{secret_tokens[1]}",
            "rejected": f"PROBE_REJECTED_{secret_tokens[2]}",
        }
    ]
    cell_outputs = {
        "kernel": [f"CELL_A_{secret_tokens[3]}"],
        "coding_L1": [f"CELL_B_{secret_tokens[4]}"],
    }

    holdout = compute_holdout_metrics(rows)
    consistency = compute_cell_consistency(cell_outputs)
    win_rate = evaluate_style_probes(probes, scorer=lambda _q, c: float(len(c)))
    report = build_fidelity_report(
        holdout=holdout,
        cell_consistency_score=consistency,
        probe_win_rate=win_rate,
    )
    validate_fidelity_report(report)

    serialized = json.dumps(report, ensure_ascii=False)
    for tok in secret_tokens:
        assert tok not in serialized
    for marker in (
        "PROMPT_SECRET_",
        "INPUT_SECRET_",
        "OUTPUT_SECRET_",
        "PROBE_Q_",
        "PROBE_CHOSEN_",
        "PROBE_REJECTED_",
        "CELL_A_",
        "CELL_B_",
    ):
        assert marker not in serialized
