"""``ego school`` 子命令实现：冷启动对话/账本导入（`import`）与伴生增量蒸馏（`evolve`）。"""

from __future__ import annotations

import argparse
import sys
from typing import TextIO

from ego_pod.exit_codes import EXIT_OK
from ego_pod.school.evolver import DEFAULT_MIN_CONSISTENCY, evolve_workspace
from ego_pod.school.importer import import_school_data

__all__ = ["run_school_cmd"]


def run_school_cmd(
    args: argparse.Namespace,
    *,
    stdout: TextIO | None = None,
) -> int:
    """分发并执行 ``ego school {import,evolve}``。"""

    out = sys.stdout if stdout is None else stdout
    action = getattr(args, "school_action", None)

    if action == "import":
        res = import_school_data(
            getattr(args, "input"),
            fmt=str(getattr(args, "format")),
            workspace=getattr(args, "workspace") or ".",
        )
        print(
            f"[EGO SCHOOL] import completed: format={res.fmt} "
            f"imported_turns={res.imported_turns} correction_pairs={res.correction_pairs} "
            f"redactions={res.total_redactions} redlines={res.redlines_count} "
            f"workspace={res.workspace}",
            file=out,
            flush=True,
        )
        return EXIT_OK

    if action == "evolve":
        evo = evolve_workspace(
            getattr(args, "workspace", None) or ".",
            ledger_path=getattr(args, "ledger", None),
            config_path=getattr(args, "config", None),
            min_consistency_score=float(
                getattr(args, "min_consistency", DEFAULT_MIN_CONSISTENCY)
            ),
        )
        snap_str = evo.archived_snapshot_id or "none"
        probe_str = (
            f"{evo.probe_win_rate:.4f}"
            if evo.probe_win_rate is not None
            else "null"
        )
        print(
            f"[EGO SCHOOL] evolve promoted: package={evo.package_root} "
            f"ingested_turns={evo.ingested_turns} ingested_corrections={evo.ingested_corrections} "
            f"archived_snapshot={snap_str} probe_win_rate={probe_str}",
            file=out,
            flush=True,
        )
        return EXIT_OK

    raise AssertionError(f"未知的 school 子动作: {action!r}")
