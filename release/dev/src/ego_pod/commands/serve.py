"""``ego serve`` 子命令实现：启动三合一本地认知 HTTP JSON 服务。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import TextIO

from ego_pod.commands.run import parse_cli_load_selection
from ego_pod.exit_codes import EXIT_OK, ConfigError
from ego_pod.service.engine import EgoCognitiveEngine
from ego_pod.service.server import (
    DEFAULT_SERVE_HOST,
    DEFAULT_SERVE_PORT,
    EgoServiceServer,
)

__all__ = ["run_serve_cmd"]


def run_serve_cmd(
    args: argparse.Namespace,
    *,
    stdout: TextIO | None = None,
) -> int:
    """加载并验签 `.egopod`，启动 ``ThreadingHTTPServer`` 提供三合一认知接口。"""

    out = sys.stdout if stdout is None else stdout
    target = getattr(args, "package_or_workspace", None) or "."
    edition = getattr(args, "edition", None)
    host = str(getattr(args, "host", DEFAULT_SERVE_HOST) or DEFAULT_SERVE_HOST)
    port = int(getattr(args, "port", DEFAULT_SERVE_PORT))
    if port < 0 or port > 65535:
        raise ConfigError(f"非法的 --port 端口号：{port}（须位于 0..65535）")

    selection = parse_cli_load_selection(
        getattr(args, "zone", None),
        kernel_only=bool(getattr(args, "kernel_only", False)),
    )
    allow_unsigned_dev = bool(getattr(args, "allow_unsigned_dev", False))
    trusted_pubkey = getattr(args, "trusted_public_key_hex", None)

    engine = EgoCognitiveEngine(
        Path(target),
        edition=edition,
        selection=selection,
        allow_unsigned_dev=allow_unsigned_dev,
        trusted_public_key_hex=trusted_pubkey,
    )
    server = EgoServiceServer(engine, host=host, port=port)
    print(
        f"[EGO SERVE] listening on {server.base_url} "
        f"(edition={engine.edition}, signer={engine.signer_fay_id}, backend={engine.backend_name})",
        file=out,
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return EXIT_OK
