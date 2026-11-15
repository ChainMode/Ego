"""``ego_pod.fidelity`` —— 本地隐私优先的人格保真度与跨 Cell 一致性度量模块。"""

from __future__ import annotations

from ego_pod.fidelity.metrics import (
    FIDELITY_SCHEMA_VERSION,
    FidelityStatus,
    HoldoutMetrics,
    build_fidelity_report,
    compute_cell_consistency,
    compute_holdout_metrics,
    evaluate_corpus_fidelity,
    evaluate_style_probes,
    split_holdout_deterministic,
    validate_fidelity_report,
)

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
