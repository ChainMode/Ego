"""纯 Python 零 ML 依赖的分区内领域专精度评分与连续层级压缩（ADR 0005）。

本模块实现深层多维稀疏网格（``L1..L3+``）的内容感知分层器（Specificity Distiller），
严格满足三项核心工程与机器学习纪律：

1. **导入安全与离线确定性**：纯 Python 标准库实现，无任何 ``torch`` / ``transformers`` /
   ``sklearn`` 等第三方依赖，跨平台同输入产出逐位一致的评分与层级。
2. **严格防数据泄漏（Strict Featurization Ordering）**：词表文档频率（DF）、IDF 权重与
   各分区分位数切点 **仅** 在训练集样本（``train_samples``）上拟合；留出验证集绝不参与
   统计拟合。
3. **显式缺失/空值检测与连续层级压缩（Contiguous Compaction）**：显式检测 ``None``、
   空串与纯空白文本字段并累计到 :class:`NullFieldStats`，空样本确定性归为 ``0.0`` 分；
   任何分区在分桶后均经 :func:`compact_levels` 保序重映射为严格连续的 ``1..K``（无跳号）。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from ego_pod.compiler.classify import SampleCandidate

__all__ = [
    "NullFieldStats",
    "SpecificityModel",
    "tokenize_for_specificity",
    "compact_levels",
    "fit_specificity_model",
    "score_sample_specificity",
    "assign_zone_levels",
]

_CJK_SPAN_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+")
_WORD_RE = re.compile(r"[a-zA-Z0-9_\u00c0-\u024f\u0400-\u04ff]{2,}")
_FENCED_CODE_RE = re.compile(r"```[\s\S]*?```")
_CODE_SYNTAX_RE = re.compile(
    r"(\bdef\s+\w+|\bclass\s+\w+|\bfn\s+\w+|\bpub\s+fn\b|\breturn\b|->|::|"
    r"@dataclass|\bimport\s+\w+|\basync\s+def\b|\bimpl\b)"
)
_TECH_TOKEN_RE = re.compile(
    r"([a-z]+_[a-z0-9_]+|[A-Z][a-z0-9]+[A-Z][A-Za-z0-9]*|\bO\([^)]+\)|"
    r"\bRFC\s*\d+|\bSHA-?\d+|\bHKDF\b|\bEd25519\b|\b[a-zA-Z_]\w*\()"
)
_REASONING_MARKERS: tuple[str, ...] = (
    "因为",
    "因此",
    "权衡",
    "复杂度",
    "不变量",
    "证明",
    "架构",
    "解耦",
    "协议",
    "算法",
    "边界",
    "原理",
    "一致性",
    "确定性",
    "拜占庭",
    "容错",
    "法定人数",
    "because",
    "therefore",
    "trade-off",
    "tradeoff",
    "invariant",
    "complexity",
    "architecture",
    "decoupl",
    "protocol",
    "algorithm",
    "consensus",
)


@dataclass
class NullFieldStats:
    """样本缺失/空值/纯空白字段审计统计。"""

    null_or_empty_fields: int = 0
    empty_samples: int = 0


@dataclass(frozen=True)
class SpecificityModel:
    """仅从训练子集拟合出的不可变分区专精度统计模型。"""

    max_level: int
    zone_doc_counts: dict[str, int] = field(default_factory=dict)
    zone_df: dict[str, dict[str, int]] = field(default_factory=dict)
    zone_max_idf: dict[str, float] = field(default_factory=dict)
    zone_quantiles: dict[str, tuple[float, ...]] = field(default_factory=dict)


def _inspect_sample(
    sample: SampleCandidate,
    stats: NullFieldStats | None = None,
) -> tuple[str, bool]:
    """检查样本字段空值状况并返回 ``(组合特征文本, 是否为空样本)``。"""

    raw_inst = getattr(sample, "instruction", None)
    raw_inp = getattr(sample, "input_field", None)
    raw_out = getattr(sample, "output", None)

    null_fields = 0
    if raw_inst is None or not str(raw_inst).strip():
        null_fields += 1
    if raw_inp is None:
        null_fields += 1

    out_str = "" if raw_out is None else str(raw_out).strip()
    if raw_out is None or not out_str:
        null_fields += 1
        if stats is not None:
            stats.null_or_empty_fields += null_fields
            stats.empty_samples += 1
        return "", True

    if stats is not None and null_fields > 0:
        stats.null_or_empty_fields += null_fields

    inp_str = "" if raw_inp is None else str(raw_inp).strip()
    combined = f"{inp_str}\n{out_str}".strip() if inp_str else out_str
    return combined, False


def tokenize_for_specificity(text: str) -> list[str]:
    """提取多语种词元（CJK 滑动二元组 + 拉丁/西里尔/标识符词元）。"""

    if not text or not text.strip():
        return []

    tokens: list[str] = []
    for match in _WORD_RE.finditer(text):
        tokens.append(match.group(0).lower())

    for match in _CJK_SPAN_RE.finditer(text):
        span = match.group(0)
        if len(span) == 1:
            tokens.append(span)
        else:
            for idx in range(len(span) - 1):
                tokens.append(span[idx : idx + 2])
    return tokens


def _structural_complexity(text: str) -> float:
    """计算文本的结构/代码/推理复杂度特征，严格落于 ``[0.0, 1.0]``。"""

    stripped = text.strip()
    if not stripped:
        return 0.0

    len_score = min(1.0, math.log1p(len(stripped)) / math.log1p(360.0))

    code_points = 0.0
    if _FENCED_CODE_RE.search(stripped):
        code_points += 0.55
    syntax_hits = len(_CODE_SYNTAX_RE.findall(stripped))
    if syntax_hits:
        code_points += min(0.35, 0.15 * syntax_hits)
    tech_hits = len(_TECH_TOKEN_RE.findall(stripped))
    if tech_hits:
        code_points += min(0.30, 0.08 * tech_hits)
    code_score = min(1.0, code_points)

    lowered = stripped.lower()
    marker_hits = sum(1 for marker in _REASONING_MARKERS if marker in lowered)
    reasoning_score = min(1.0, marker_hits / 3.0)

    blended = 0.35 * len_score + 0.40 * code_score + 0.25 * reasoning_score
    return max(0.0, min(1.0, blended))


def _lexical_idf_score(
    tokens: Sequence[str],
    zone: str,
    zone_df: Mapping[str, Mapping[str, int]],
    zone_doc_counts: Mapping[str, int],
    zone_max_idf: Mapping[str, float],
) -> float:
    """计算样本在目标分区词表下的归一化稀有度得分 ``[0.0, 1.0]``。"""

    if not tokens:
        return 0.0

    unique_tokens = set(tokens)
    doc_count = zone_doc_counts.get(zone, 0)
    df_table = zone_df.get(zone, {})
    max_idf = zone_max_idf.get(zone, math.log(doc_count + 1.0) + 1.0)
    if max_idf <= 0.0:
        max_idf = 1.0

    idf_sum = 0.0
    for tok in unique_tokens:
        df = df_table.get(tok, 0)
        idf = math.log((doc_count + 1.0) / (df + 1.0)) + 1.0
        idf_sum += min(max_idf, idf)

    mean_norm_idf = (idf_sum / len(unique_tokens)) / max_idf
    richness = min(1.0, len(unique_tokens) / 16.0)
    score = mean_norm_idf * (0.35 + 0.65 * richness)
    return max(0.0, min(1.0, score))


def score_sample_specificity(
    sample: SampleCandidate,
    zone: str,
    model: SpecificityModel,
    *,
    stats: NullFieldStats | None = None,
) -> float:
    """对单条样本计算 ``[0.0, 1.0]`` 区间内的确定性专精度得分。"""

    text, is_empty = _inspect_sample(sample, stats=stats)
    if is_empty:
        return 0.0

    tokens = tokenize_for_specificity(text)
    lex_score = _lexical_idf_score(
        tokens,
        zone,
        model.zone_df,
        model.zone_doc_counts,
        model.zone_max_idf,
    )
    struct_score = _structural_complexity(text)
    combined = 0.45 * lex_score + 0.55 * struct_score
    return round(max(0.0, min(1.0, combined)), 8)


def _compute_quantile_cutpoints(
    scores: Sequence[float],
    max_level: int,
) -> tuple[float, ...]:
    """基于分区内训练样本得分序列计算最多 ``max_level - 1`` 个单调递增切点。"""

    if max_level <= 1 or not scores:
        return ()

    sorted_scores = sorted(scores)
    unique_sorted = sorted(set(sorted_scores))
    if len(unique_sorted) <= 1:
        return ()

    if len(unique_sorted) <= max_level:
        return tuple(
            round((unique_sorted[i] + unique_sorted[i + 1]) / 2.0, 10)
            for i in range(len(unique_sorted) - 1)
        )

    cutpoints: list[float] = []
    n = len(sorted_scores)
    for bucket in range(1, max_level):
        idx = max(1, min(n - 1, int(round(bucket * n / max_level))))
        left = sorted_scores[idx - 1]
        right = sorted_scores[idx]
        threshold = round((left + right) / 2.0 if left < right else right, 10)
        if not cutpoints or threshold > cutpoints[-1]:
            cutpoints.append(threshold)
    return tuple(cutpoints)


def fit_specificity_model(
    train_samples: Sequence[SampleCandidate],
    routing: Mapping[str, str],
    *,
    max_level: int = 3,
    stats: NullFieldStats | None = None,
) -> SpecificityModel:
    """严格仅在训练样本序列 ``train_samples`` 上拟合各分区词表 DF、IDF 与分位数切点。"""

    eff_max_level = max(1, int(max_level))
    zone_docs: dict[str, list[tuple[SampleCandidate, list[str], bool]]] = {}

    for sample in train_samples:
        text, is_empty = _inspect_sample(sample, stats=stats)
        zone = routing.get(sample.category)
        if zone is None:
            continue
        tokens = [] if is_empty else tokenize_for_specificity(text)
        zone_docs.setdefault(zone, []).append((sample, tokens, is_empty))

    zone_doc_counts: dict[str, int] = {}
    zone_df: dict[str, dict[str, int]] = {}
    zone_max_idf: dict[str, float] = {}

    for zone, docs in zone_docs.items():
        non_empty_docs = [toks for _s, toks, empty in docs if not empty]
        doc_count = len(non_empty_docs)
        df_counter: Counter[str] = Counter()
        for toks in non_empty_docs:
            for tok in set(toks):
                df_counter[tok] += 1
        zone_doc_counts[zone] = doc_count
        zone_df[zone] = dict(sorted(df_counter.items()))
        zone_max_idf[zone] = math.log(doc_count + 1.0) + 1.0

    preliminary = SpecificityModel(
        max_level=eff_max_level,
        zone_doc_counts=zone_doc_counts,
        zone_df=zone_df,
        zone_max_idf=zone_max_idf,
        zone_quantiles={},
    )

    zone_quantiles: dict[str, tuple[float, ...]] = {}
    for zone, docs in zone_docs.items():
        scores = [
            score_sample_specificity(sample, zone, preliminary)
            for sample, _toks, _empty in docs
        ]
        zone_quantiles[zone] = _compute_quantile_cutpoints(scores, eff_max_level)

    return SpecificityModel(
        max_level=eff_max_level,
        zone_doc_counts=zone_doc_counts,
        zone_df=zone_df,
        zone_max_idf=zone_max_idf,
        zone_quantiles=zone_quantiles,
    )


def compact_levels(raw_levels: Sequence[int]) -> list[int]:
    """将可能存在空桶跳号的原始层级列表保序压缩为从 ``1`` 起严格连续的 ``1..K``。"""

    if not raw_levels:
        return []
    unique_sorted = sorted(set(raw_levels))
    remap = {old_lvl: idx + 1 for idx, old_lvl in enumerate(unique_sorted)}
    return [remap[lvl] for lvl in raw_levels]


def assign_zone_levels(
    samples: Sequence[SampleCandidate],
    zone: str,
    model: SpecificityModel,
    *,
    max_level: int | None = None,
    stats: NullFieldStats | None = None,
) -> list[int]:
    """为同一分区的样本序列分配严格连续且 ``<= max_level`` 的层级 ``1..K``。"""

    if not samples:
        return []

    eff_max = max(1, model.max_level if max_level is None else int(max_level))
    if eff_max == 1:
        if stats is not None:
            for sample in samples:
                _inspect_sample(sample, stats=stats)
        return [1] * len(samples)

    cutpoints = model.zone_quantiles.get(zone, ())
    if eff_max - 1 < len(cutpoints):
        cutpoints = cutpoints[: eff_max - 1]

    raw_levels: list[int] = []
    for sample in samples:
        score = score_sample_specificity(sample, zone, model, stats=stats)
        lvl = 1 + sum(1 for threshold in cutpoints if score > threshold)
        raw_levels.append(min(eff_max, lvl))

    return compact_levels(raw_levels)
