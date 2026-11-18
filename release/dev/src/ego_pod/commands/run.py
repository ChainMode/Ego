"""``ego run`` 实现：支持多规格 Edition 选择与按需网格 ``--zone`` / ``--kernel-only`` 激活。"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TextIO

from ego_pod.exit_codes import RuntimeError_
from ego_pod.runtime.inference import (
    DEFAULT_EDITION,
    MANIFEST_FILENAME,
    OUTPUT_DIRNAME,
    load_inference_session,
)
from ego_pod.runtime.repl import run_repl as run_repl_loop
from ego_pod.runtime.selection import LoadSelection

__all__ = [
    "EDITION_NOT_SUPPORTED_MESSAGE",
    "PACKAGE_NOT_FOUND_MESSAGE",
    "parse_cli_load_selection",
    "run_repl",
]

EDITION_NOT_SUPPORTED_MESSAGE = "当前 Ego_Package 未包含请求的 Edition"
PACKAGE_NOT_FOUND_MESSAGE = "未找到 Ego_Package，请先执行 `ego train`"

_EDITION_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,31}$")
_ZONE_NAME_RE = re.compile(r"^[a-z0-9-]+$")


def _relative_label(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except (ValueError, OSError):
        return str(path)


def parse_cli_load_selection(
    zone_specs: Sequence[str] | None,
    *,
    kernel_only: bool = False,
) -> LoadSelection | None:
    """将命令行 ``--zone <name>:<depth>`` 与 ``--kernel-only`` 解析为 :class:`LoadSelection`。

    * 两者均未指定 ⇒ 返回 ``None``（Full_In_Load 全量加载）；
    * 指定 ``--kernel-only`` ⇒ 返回 ``LoadSelection(depths={})``（仅加载 Kernel）；
    * 同时指定 ``--kernel-only`` 与 ``--zone``，或 ``--zone`` 语法非法 ⇒ 抛
      :class:`RuntimeError_`（退出码 1）。
    """

    has_zones = bool(zone_specs)
    if kernel_only and has_zones:
        raise RuntimeError_("`--kernel-only` 与 `--zone` 互斥，不能同时指定")
    if kernel_only:
        return LoadSelection(depths={})
    if not zone_specs:
        return None

    depths: dict[str, int] = {}
    for raw in zone_specs:
        if not isinstance(raw, str) or ":" not in raw:
            raise RuntimeError_(
                f"非法的 --zone 参数格式 {raw!r}：须为 '<zone>:<depth>'（例如 coding:2）"
            )
        zone_part, depth_part = raw.split(":", 1)
        zone_name = zone_part.strip()
        depth_str = depth_part.strip()
        if not zone_name or not _ZONE_NAME_RE.fullmatch(zone_name):
            raise RuntimeError_(
                f"非法的 --zone 分区名 {zone_name!r}（原参数：{raw!r}）"
            )
        try:
            depth = int(depth_str)
        except ValueError as exc:
            raise RuntimeError_(
                f"非法的 --zone 深度 {depth_str!r}：必须为正整数（原参数：{raw!r}）"
            ) from exc
        if depth < 1:
            raise RuntimeError_(
                f"功能分区「{zone_name}」的加载深度 {depth} 非法：必须 >= 1"
            )
        depths[zone_name] = depth
    return LoadSelection(depths=depths)


def _check_edition_in_manifest(output_root: Path, edition: str) -> None:
    """若 ``manifest.json`` 存在且为合法 JSON 对象，先行核验请求的 Edition 是否已构建。"""

    manifest_path = output_root / MANIFEST_FILENAME
    if not manifest_path.is_file():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        # 交由下游 load_inference_session 按 R12-AC5 报告文件损坏
        return
    if not isinstance(manifest, dict):
        return
    editions = manifest.get("editions")
    if isinstance(editions, list) and edition not in editions:
        raise RuntimeError_(
            f"{EDITION_NOT_SUPPORTED_MESSAGE}：收到 --edition {edition!r}，"
            f"但当前包已构建规格为 {editions!r}"
        )


def run_repl(
    args: argparse.Namespace,
    *,
    cwd: Path | None = None,
    backend: Any | None = None,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    selection: LoadSelection | None = None,
) -> int:
    """``ego run`` 主入口：支持任意已构建 Edition 与按需网格加载。"""

    edition = getattr(args, "edition", DEFAULT_EDITION)
    if edition is None:
        edition = DEFAULT_EDITION

    if not isinstance(edition, str) or not _EDITION_RE.fullmatch(edition):
        raise RuntimeError_(
            f"非法的 --edition 名称 {edition!r}（须由小写字母、数字与连字符组成）"
        )

    effective_selection = (
        selection
        if selection is not None
        else parse_cli_load_selection(
            getattr(args, "zone", None),
            kernel_only=bool(getattr(args, "kernel_only", False)),
        )
    )

    base_cwd = Path.cwd() if cwd is None else Path(cwd)
    output_root = base_cwd / OUTPUT_DIRNAME

    # 若已有包 manifest，优先校验请求规格是否在 manifest.editions 中
    _check_edition_in_manifest(output_root, edition)

    edition_dir = output_root / edition
    if not edition_dir.is_dir():
        label = _relative_label(edition_dir, base_cwd)
        raise RuntimeError_(
            f"{PACKAGE_NOT_FOUND_MESSAGE}（缺少目录：{label}，edition={edition!r}）"
        )

    session = load_inference_session(
        edition,
        cwd=base_cwd,
        backend=backend,
        stdout=stdout,
        selection=effective_selection,
    )
    return run_repl_loop(session, stdin=stdin, stdout=stdout, stderr=stderr)
