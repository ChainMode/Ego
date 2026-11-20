"""``ego slice`` 子命令实现：导出按领域与体积裁剪的独立已签名微型 `.egopod` 切片。"""

from __future__ import annotations

import argparse
import sys
from typing import TextIO

from ego_pod.exit_codes import EXIT_OK
from ego_pod.vault.submodel_slice import export_submodel_slice

__all__ = ["run_slice_cmd"]


def run_slice_cmd(
    args: argparse.Namespace,
    *,
    stdout: TextIO | None = None,
) -> int:
    """执行 ``ego slice`` 并输出切片目录绝对路径与签名摘要。"""

    out = sys.stdout if stdout is None else stdout
    domains_arg = getattr(args, "domain", None)
    domains = list(domains_arg) if isinstance(domains_arg, list) else [str(domains_arg)]

    artifacts = export_submodel_slice(
        getattr(args, "source_package"),
        domains=domains,
        out_dir=getattr(args, "out"),
        signing_key=getattr(args, "signing_key"),
        signer_fay_id=getattr(args, "signer_fay_id", None),
        edition=getattr(args, "edition", None),
        max_depth=getattr(args, "max_depth", None),
        max_size_mb=getattr(args, "max_size_mb", None),
        target_device_class=getattr(args, "target_device_class", None),
    )
    print(
        f"[EGO SLICE] exported standalone slice: {artifacts.root} "
        f"edition={artifacts.edition} domains={','.join(artifacts.selected_domains)} "
        f"cells=[{','.join(artifacts.included_cells)}] size_bytes={artifacts.actual_size_bytes} "
        f"signer={artifacts.signer_fay_id}",
        file=out,
        flush=True,
    )
    print(str(artifacts.root), file=out, flush=True)
    return EXIT_OK
