"""本地隐私优先的人格保真度与跨 Cell 一致性度量实现（ADR 0002）。

本模块实现 ``manifest.json`` 顶层可选 ``fidelity_report`` 的纯本地计算与严格校验，
遵守两条核心治理红线：

1. **隐私红线（ADR 0002 §4）**：人类原型的原始语料、留出集样本、风格探针题目与回复
   文本 **绝不** 进入 :func:`build_fidelity_report` 的返回结构或 ``Ego_Package``；
   :func:`validate_fidelity_report` 对字段集合、值类型与有限性作白名单强校验。
2. **门控分层（ADR 0002 §3）**：保真度报告仅作为客观可追溯度量写入产物，不阻断
   ``ego train`` 退出码 0；是否允许装载进宿主（Host）由 ``FayGer`` 装载策略把关。
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

__all__ = [
    "FIDELITY_SCHEMA_VERSION",
    "FidelityStatus",
    "HoldoutMetrics",
    "split_holdout_deterministic",
    "compute_holdout_metrics",
    "compute_cell_consistency",
    "evaluate_style_probes",
    "build_fidelity_report",
    "validate_fidelity_report",
    "evaluate_corpus_fidelity",
]

#: 与 FayGer 跨仓对齐的 fidelity_report 线协议版本。
FIDELITY_SCHEMA_VERSION = "1.0"

#: 保真度状态枚举：样本充足完成测量 vs 样本不足无法切出稳定留出集。
FidelityStatus = Literal["measured", "insufficient_data"]

#: 留出集困惑度指数上限，防止极端高损失触发 OverflowError。
_MAX_PERPLEXITY_EXPONENT = 50.0

#: 允许出现在 fidelity_report 中的全部键及顺序。
_EXPECTED_KEYS = (
    "schema_version",
    "status",
    "holdout_sample_count",
    "holdout_loss",
    "holdout_perplexity",
    "cell_consistency_score",
    "probe_win_rate",
    "drift_from_baseline",
)


@dataclass(frozen=True)
class HoldoutMetrics:
    """确定性留出集评估结果（纯聚合标量，不含任何样本原文）。"""

    status: FidelityStatus
    holdout_sample_count: int
    holdout_loss: float | None
    holdout_perplexity: float | None


def _load_records(
    corpus: str | Path | Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """从 JSONL 文件路径或内存映射序列读取语料记录列表。"""

    if isinstance(corpus, (str, Path)):
        path = Path(corpus)
        if not path.is_file():
            return []
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, Mapping):
                    records.append(dict(obj))
        return records
    return [dict(item) for item in corpus]


def _sample_digest(index: int, record: Mapping[str, Any]) -> str:
    """为单条样本计算稳定的 SHA-256 排序键，保证留出集切分完全确定。"""

    instruction = str(record.get("instruction", ""))
    input_field = str(record.get("input", ""))
    output = str(record.get("output", ""))
    payload = f"{index}\0{instruction}\0{input_field}\0{output}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def split_holdout_deterministic(
    corpus: str | Path | Iterable[Mapping[str, Any]],
    *,
    holdout_ratio: float = 0.2,
    min_total_samples: int = 8,
    min_holdout_samples: int = 2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """按确定性 SHA-256 哈希序将语料切分为 ``(train_split, holdout_split)``。

    当总样本数小于 ``min_total_samples`` 或切出的留出集不足 ``min_holdout_samples``
    时，返回 ``(records, [])`` 表示样本不足以支撑留出集评估。
    """

    records = _load_records(corpus)
    total = len(records)
    if total < min_total_samples or holdout_ratio <= 0.0:
        return records, []

    holdout_count = max(min_holdout_samples, int(round(total * holdout_ratio)))
    if holdout_count >= total or holdout_count < min_holdout_samples:
        return records, []

    ranked_indices = sorted(
        range(total), key=lambda idx: _sample_digest(idx, records[idx])
    )
    holdout_idx_set = set(ranked_indices[:holdout_count])

    train_split = [records[i] for i in range(total) if i not in holdout_idx_set]
    holdout_split = [records[i] for i in range(total) if i in holdout_idx_set]
    return train_split, holdout_split


def _smoothed_bigram_cross_entropy(
    train_records: Sequence[Mapping[str, Any]],
    holdout_records: Sequence[Mapping[str, Any]],
) -> float:
    """基于训练子集输出文本的拉普拉斯平滑字符二元交叉熵（纯本地无外部依赖基线）。"""

    bos = "\x02"
    eos = "\x03"
    bigram_counts: Counter[tuple[str, str]] = Counter()
    context_counts: Counter[str] = Counter()
    vocab: set[str] = {eos}

    for rec in train_records:
        text = str(rec.get("output", ""))
        seq = bos + text + eos
        for ch in text:
            vocab.add(ch)
        for prev_ch, next_ch in zip(seq, seq[1:]):
            bigram_counts[(prev_ch, next_ch)] += 1
            context_counts[prev_ch] += 1

    # 将留出集字符纳入词表大小，使未知字符在 Laplace 平滑下获得严格正概率。
    for rec in holdout_records:
        for ch in str(rec.get("output", "")):
            vocab.add(ch)

    vocab_size = max(len(vocab), 2)
    total_nll = 0.0
    total_tokens = 0

    for rec in holdout_records:
        seq = bos + str(rec.get("output", "")) + eos
        for prev_ch, next_ch in zip(seq, seq[1:]):
            num = bigram_counts[(prev_ch, next_ch)] + 1.0
            den = context_counts[prev_ch] + float(vocab_size)
            total_nll += -math.log(num / den)
            total_tokens += 1

    if total_tokens == 0:
        return 0.0
    return total_nll / total_tokens


def compute_holdout_metrics(
    corpus: str | Path | Iterable[Mapping[str, Any]],
    *,
    loss_fn: Callable[[Sequence[Mapping[str, Any]]], float] | None = None,
    holdout_ratio: float = 0.2,
    min_total_samples: int = 8,
    min_holdout_samples: int = 2,
) -> HoldoutMetrics:
    """计算确定性留出集的损失与困惑度。

    当样本量不足或评估损失非有限值时，诚实返回 ``status="insufficient_data"``。
    """

    train_split, holdout_split = split_holdout_deterministic(
        corpus,
        holdout_ratio=holdout_ratio,
        min_total_samples=min_total_samples,
        min_holdout_samples=min_holdout_samples,
    )
    if len(holdout_split) < min_holdout_samples:
        return HoldoutMetrics(
            status="insufficient_data",
            holdout_sample_count=0,
            holdout_loss=None,
            holdout_perplexity=None,
        )

    raw_loss = (
        float(loss_fn(holdout_split))
        if loss_fn is not None
        else _smoothed_bigram_cross_entropy(train_split, holdout_split)
    )
    if not math.isfinite(raw_loss) or raw_loss < 0.0:
        return HoldoutMetrics(
            status="insufficient_data",
            holdout_sample_count=len(holdout_split),
            holdout_loss=None,
            holdout_perplexity=None,
        )

    rounded_loss = round(raw_loss, 6)
    perplexity = round(math.exp(min(rounded_loss, _MAX_PERPLEXITY_EXPONENT)), 6)
    return HoldoutMetrics(
        status="measured",
        holdout_sample_count=len(holdout_split),
        holdout_loss=rounded_loss,
        holdout_perplexity=perplexity,
    )


def _char_ngrams(text: str, n: int = 2) -> set[str]:
    """提取归一化文本的字符 n-gram 集合（短文本回退至单字符集合）。"""

    normalized = " ".join(text.strip().split())
    if not normalized:
        return set()
    if len(normalized) < n:
        return {normalized}
    return {normalized[i : i + n] for i in range(len(normalized) - n + 1)}


def _pairwise_text_similarity(text_a: str, text_b: str) -> float:
    """计算两条本地生成回复的归一化 Jaccard 一致性（范围 ``[0.0, 1.0]``）。"""

    if text_a == text_b:
        return 1.0
    grams_a = _char_ngrams(text_a)
    grams_b = _char_ngrams(text_b)
    if not grams_a and not grams_b:
        return 1.0
    if not grams_a or not grams_b:
        return 0.0
    union = grams_a | grams_b
    return len(grams_a & grams_b) / len(union)


def compute_cell_consistency(
    cell_responses: Mapping[str, Sequence[str]] | Sequence[Sequence[str]] | None,
) -> float | None:
    """计算至少两组 Cell 激活配置在相同提示序列上的平均成对回复一致性。

    若未提供多组合回复（``None``、少于 2 个组合或提示数为 0），返回 ``None``。
    """

    if cell_responses is None:
        return None

    groups: list[list[str]]
    if isinstance(cell_responses, Mapping):
        groups = [[str(r) for r in seq] for seq in cell_responses.values()]
    else:
        groups = [[str(r) for r in seq] for seq in cell_responses]

    if len(groups) < 2:
        return None

    prompt_count = min(len(g) for g in groups)
    if prompt_count == 0:
        return None

    prompt_scores: list[float] = []
    num_groups = len(groups)
    for prompt_idx in range(prompt_count):
        pair_scores: list[float] = []
        for i in range(num_groups):
            for j in range(i + 1, num_groups):
                pair_scores.append(
                    _pairwise_text_similarity(
                        groups[i][prompt_idx], groups[j][prompt_idx]
                    )
                )
        prompt_scores.append(sum(pair_scores) / len(pair_scores))

    mean_score = sum(prompt_scores) / len(prompt_scores)
    return round(max(0.0, min(1.0, mean_score)), 6)


def evaluate_style_probes(
    probes: Sequence[Mapping[str, Any]] | None,
    *,
    scorer: Callable[[str, str], float] | None = None,
) -> float | None:
    """在本地评估人类原型配对风格探针胜率，仅返回 ``[0.0, 1.0]`` 标量。"""

    if not probes:
        return None

    wins = 0.0
    evaluated = 0
    for item in probes:
        if "preferred_won" in item and isinstance(item["preferred_won"], bool):
            wins += 1.0 if item["preferred_won"] else 0.0
            evaluated += 1
        elif "chosen_score" in item and "rejected_score" in item:
            c_score = float(item["chosen_score"])
            r_score = float(item["rejected_score"])
            if c_score > r_score:
                wins += 1.0
            elif c_score == r_score:
                wins += 0.5
            evaluated += 1
        elif (
            scorer is not None
            and "prompt" in item
            and "chosen" in item
            and "rejected" in item
        ):
            prompt = str(item["prompt"])
            c_score = float(scorer(prompt, str(item["chosen"])))
            r_score = float(scorer(prompt, str(item["rejected"])))
            if c_score > r_score:
                wins += 1.0
            elif c_score == r_score:
                wins += 0.5
            evaluated += 1

    if evaluated == 0:
        return None
    return round(max(0.0, min(1.0, wins / evaluated)), 6)


def _is_finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def validate_fidelity_report(report: Mapping[str, Any]) -> None:
    """强校验 ``fidelity_report`` 字典结构、类型与隐私红线（禁止任何额外字段或自由文本）。"""

    if not isinstance(report, Mapping):
        raise ValueError("fidelity_report 必须为映射对象")

    actual_keys = tuple(report.keys())
    if set(actual_keys) != set(_EXPECTED_KEYS) or len(actual_keys) != len(
        _EXPECTED_KEYS
    ):
        raise ValueError(
            f"fidelity_report 字段集合非法：期望 {_EXPECTED_KEYS!r}，实际 {actual_keys!r}"
        )

    if report["schema_version"] != FIDELITY_SCHEMA_VERSION:
        raise ValueError(
            f"fidelity_report.schema_version 必须为 {FIDELITY_SCHEMA_VERSION!r}"
        )

    status = report["status"]
    if status not in ("measured", "insufficient_data"):
        raise ValueError(f"fidelity_report.status 非法：{status!r}")

    count = report["holdout_sample_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("fidelity_report.holdout_sample_count 必须为非负整数")

    for nullable_float_key in (
        "holdout_loss",
        "holdout_perplexity",
        "cell_consistency_score",
        "probe_win_rate",
        "drift_from_baseline",
    ):
        val = report[nullable_float_key]
        if val is not None and not _is_finite_number(val):
            raise ValueError(
                f"fidelity_report.{nullable_float_key} 必须为有限数值或 None"
            )

    for unit_key in ("cell_consistency_score", "probe_win_rate"):
        val = report[unit_key]
        if val is not None and not (0.0 <= float(val) <= 1.0):
            raise ValueError(f"fidelity_report.{unit_key} 必须位于 [0.0, 1.0]")


def build_fidelity_report(
    *,
    holdout: HoldoutMetrics,
    cell_consistency_score: float | None = None,
    probe_win_rate: float | None = None,
    baseline_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """构造严格符合跨仓线契约且零原始文本泄露的 ``fidelity_report`` 字典。"""

    drift: float | None = None
    if (
        holdout.status == "measured"
        and holdout.holdout_loss is not None
        and isinstance(baseline_report, Mapping)
        and baseline_report.get("status") == "measured"
        and _is_finite_number(baseline_report.get("holdout_loss"))
    ):
        drift = round(
            float(holdout.holdout_loss) - float(baseline_report["holdout_loss"]),
            6,
        )

    report: dict[str, Any] = {
        "schema_version": FIDELITY_SCHEMA_VERSION,
        "status": holdout.status,
        "holdout_sample_count": int(holdout.holdout_sample_count),
        "holdout_loss": (
            None if holdout.holdout_loss is None else round(float(holdout.holdout_loss), 6)
        ),
        "holdout_perplexity": (
            None
            if holdout.holdout_perplexity is None
            else round(float(holdout.holdout_perplexity), 6)
        ),
        "cell_consistency_score": (
            None
            if cell_consistency_score is None
            else round(float(cell_consistency_score), 6)
        ),
        "probe_win_rate": (
            None if probe_win_rate is None else round(float(probe_win_rate), 6)
        ),
        "drift_from_baseline": drift,
    }
    validate_fidelity_report(report)
    return report


def evaluate_corpus_fidelity(
    corpus_path: str | Path,
    *,
    previous_manifest_path: str | Path | None = None,
    cell_responses: Mapping[str, Sequence[str]] | Sequence[Sequence[str]] | None = None,
    probes: Sequence[Mapping[str, Any]] | None = None,
    loss_fn: Callable[[Sequence[Mapping[str, Any]]], float] | None = None,
) -> dict[str, Any]:
    """在 ``ego train`` 流水线中一站式生成 ``fidelity_report``（含相对上一版漂移计算）。"""

    baseline_report: Mapping[str, Any] | None = None
    if previous_manifest_path is not None:
        prev_path = Path(previous_manifest_path)
        if prev_path.is_file():
            try:
                prev_manifest = json.loads(prev_path.read_text(encoding="utf-8"))
                if isinstance(prev_manifest, Mapping) and isinstance(
                    prev_manifest.get("fidelity_report"), Mapping
                ):
                    baseline_report = prev_manifest["fidelity_report"]
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                baseline_report = None

    holdout = compute_holdout_metrics(corpus_path, loss_fn=loss_fn)
    consistency = compute_cell_consistency(cell_responses)
    win_rate = evaluate_style_probes(probes)
    return build_fidelity_report(
        holdout=holdout,
        cell_consistency_score=consistency,
        probe_win_rate=win_rate,
        baseline_report=baseline_report,
    )
