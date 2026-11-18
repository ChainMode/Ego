"""Post-training cross-Cell behavioural consistency probe (``cell_consistency_score``).

训练后跨 Cell 行为一致性探针：在本地留出集提示上，比较同一 Edition 的
**Kernel-only** 加载与 **Full_In_Load**（Kernel + 全部分区深层 Cell）两种激活配置的
回复，经 :func:`ego_pod.fidelity.metrics.compute_cell_consistency` 得到 ``[0.0, 1.0]``
标量。高分表示深层分区 Cell 叠加后没有扭曲核心人格的表达。

Why this exists: FayGer's production ``FidelityGatePolicy`` requires
``cell_consistency_score >= 0.70``. Before this module, ``ego train`` never measured
it, so every package failed the production gate.

Privacy: prompts come from the local holdout split only, inference is local, and
only the aggregate scalar leaves this module (never prompts or responses).

Honesty rules:

* Stub-trained packages return ``None``: a fixed placeholder reply is not model
  behaviour, and scoring it would fake a perfect 1.0.
* Editions whose grid has no zone Cells are skipped (Kernel-only == Full_In_Load,
  nothing to compare).
* Multiple editions: the **minimum** per-edition score is reported, so the
  gate is passed only if every edition is consistent.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ego_pod.fidelity.metrics import compute_cell_consistency, split_holdout_deterministic

__all__ = [
    "MAX_PROBE_PROMPTS",
    "collect_holdout_prompts",
    "measure_package_cell_consistency",
]

#: Upper bound on probe prompts per edition (bounds inference cost).
MAX_PROBE_PROMPTS = 16

SessionLoader = Callable[..., Any]


def _read_corpus_records(corpus_path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    path = Path(corpus_path)
    if not path.is_file():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            records.append(obj)
    return records


def collect_holdout_prompts(
    corpus_path: str | Path, *, limit: int = MAX_PROBE_PROMPTS
) -> list[str]:
    """Return de-duplicated prompts (``instruction`` + ``input``) from the holdout split.

    Uses the same deterministic split as the holdout loss, so probe prompts never
    come from training samples. Records with missing/blank prompt text are skipped.
    """

    records = _read_corpus_records(corpus_path)
    _train, holdout = split_holdout_deterministic(records)
    prompts: list[str] = []
    seen: set[str] = set()
    for rec in holdout:
        instruction = rec.get("instruction")
        extra = rec.get("input")
        parts = [
            str(p).strip()
            for p in (instruction, extra)
            if p is not None and str(p).strip()
        ]
        if not parts:
            continue
        prompt = "\n".join(parts)
        if prompt in seen:
            continue
        seen.add(prompt)
        prompts.append(prompt)
        if len(prompts) >= limit:
            break
    return prompts


def _edition_has_zone_cells(manifest: Mapping[str, Any], edition: str) -> bool:
    grid = manifest.get("grid")
    if not isinstance(grid, Mapping):
        return False
    ed_grid = grid.get(edition, grid)
    zones = ed_grid.get("zones") if isinstance(ed_grid, Mapping) else None
    return isinstance(zones, Mapping) and any(zones.values())


def _responses(session: Any, prompts: Sequence[str]) -> list[str]:
    try:
        return [str(session.generate(p)) for p in prompts]
    finally:
        close = getattr(session, "close", None)
        if callable(close):
            close()


def measure_package_cell_consistency(
    package_root: str | Path,
    corpus_path: str | Path,
    *,
    cwd: Path,
    session_loader: SessionLoader | None = None,
) -> tuple[float | None, str]:
    """Measure ``cell_consistency_score`` for a freshly written package.

    :returns: ``(score_or_None, reason)``; ``reason`` is a short log string
        explaining the outcome (never contains prompt or response text).
    """

    root = Path(package_root)
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None, "manifest unreadable"

    if manifest.get("training_backend") == "stub":
        return None, "stub backend: placeholder replies are not model behaviour"

    editions = [e for e in manifest.get("editions", []) if isinstance(e, str)]
    comparable = [e for e in editions if _edition_has_zone_cells(manifest, e)]
    if not comparable:
        return None, "no edition has zone Cells to compare against the Kernel"

    prompts = collect_holdout_prompts(corpus_path)
    if not prompts:
        return None, "no usable holdout prompts"

    if session_loader is None:
        from ego_pod.runtime.inference import load_inference_session

        session_loader = load_inference_session
    from ego_pod.runtime.selection import LoadSelection

    scores: list[float] = []
    for edition in comparable:
        kernel_only = _responses(
            session_loader(edition, cwd=cwd, output_root=root, selection=LoadSelection()),
            prompts,
        )
        full_load = _responses(
            session_loader(edition, cwd=cwd, output_root=root, selection=None),
            prompts,
        )
        score = compute_cell_consistency([kernel_only, full_load])
        if score is not None:
            scores.append(score)

    if not scores:
        return None, "consistency could not be computed"
    return min(scores), (
        f"min over {len(scores)} edition(s) x {len(prompts)} holdout prompt(s)"
    )
