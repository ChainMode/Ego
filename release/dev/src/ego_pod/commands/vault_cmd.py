"""``ego vault`` 子命令实现：版本历史列表、验签回滚与彻底清理（ADR 0006 §6）。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TextIO
import sys

from ego_pod.exit_codes import EXIT_OK
from ego_pod.vault.history import (
    list_vault_snapshots,
    purge_vault_and_cache,
    rollback_vault_snapshot,
)
from ego_pod.vault.pipeline import OUTPUT_DIRNAME

__all__ = ["run_vault_cmd"]


def run_vault_cmd(
    args: argparse.Namespace,
    *,
    cwd: Path | None = None,
    stdout: TextIO | None = None,
) -> int:
    """分发并执行 ``ego vault {list,rollback,purge}``。"""

    base_cwd = Path.cwd() if cwd is None else Path(cwd)
    out_stream = sys.stdout if stdout is None else stdout
    output_root = base_cwd / OUTPUT_DIRNAME
    action = getattr(args, "vault_action", None)

    if action == "list":
        snapshots = list_vault_snapshots(output_root)
        if not snapshots:
            print(
                "[VAULT] 当前无历史快照（no snapshots available）",
                file=out_stream,
                flush=True,
            )
            return EXIT_OK
        for snap in snapshots:
            ed_str = ",".join(snap.editions)
            sig_label = "signed" if snap.signed else "unsigned"
            print(
                f"{snap.snapshot_id}  version={snap.version}  "
                f"created_at={snap.created_at}  editions={ed_str}  {sig_label}",
                file=out_stream,
                flush=True,
            )
        return EXIT_OK

    if action == "rollback":
        snapshot_id = str(getattr(args, "snapshot_id", ""))
        pubkey_hex = getattr(args, "public_key_hex", None)
        restored = rollback_vault_snapshot(
            output_root,
            snapshot_id,
            trusted_public_key_hex=pubkey_hex,
        )
        print(
            f"[VAULT] 已成功回滚至快照 {restored.snapshot_id} "
            f"(version={restored.version}, editions={','.join(restored.editions)})",
            file=out_stream,
            flush=True,
        )
        return EXIT_OK

    if action == "purge":
        deleted, freed_bytes = purge_vault_and_cache(
            base_cwd, output_root=output_root
        )
        print(
            f"[VAULT] 已清理 {len(deleted)} 个目录，释放 {freed_bytes} bytes",
            file=out_stream,
            flush=True,
        )
        return EXIT_OK

    raise AssertionError(f"未知的 vault 子动作: {action!r}")
