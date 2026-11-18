"""``ego init`` 实现：支持零交互默认模板、``--all-editions`` 全规格模板与 ``--interactive`` 引导。"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, TextIO

from ego_pod.config.defaults import (
    DEFAULT_BASE_MODEL,
    DEFAULT_EDITION_NAME,
    DEFAULT_SOURCE_PATHS,
    render_all_editions_yaml,
    render_custom_yaml,
    render_default_yaml,
)
from ego_pod.config.schema import EDITION_PRESETS, SUPPORTED_BASE_MODELS
from ego_pod.exit_codes import EXIT_OK, ConfigError, EgoError

__all__ = ["run_init", "CONFIG_FILENAME"]

CONFIG_FILENAME = "ego-config.yaml"
FILE_EXISTS_MESSAGE = "ego-config.yaml 已存在，拒绝写入"
WRITE_FAILED_MESSAGE = "写入 ego-config.yaml 失败"


def _atomic_write_text(target: Path, text: str) -> None:
    """以原子方式将 ``text`` 以 UTF-8 写入 ``target``，保证不留部分写入文件。"""
    parent = target.parent
    fd, tmp_name = tempfile.mkstemp(
        prefix=".ego-config-", suffix=".yaml.tmp", dir=parent
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
    except OSError:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise


def _prompt_line(prompt: str, in_stream: TextIO, out_stream: TextIO) -> str:
    out_stream.write(prompt)
    out_stream.flush()
    line = in_stream.readline()
    return line.strip() if line else ""


def _run_interactive_wizard(
    target: Path,
    abs_path: Path,
    *,
    force: bool,
    in_stream: TextIO,
    out_stream: TextIO,
) -> str:
    """运行可测试注入 I/O 的交互式初始化向导并返回生成的 YAML 文本。"""

    if os.path.lexists(target) and not force:
        ans = _prompt_line(
            f"检测到 {abs_path} 已存在，是否覆盖？[y/N]: ",
            in_stream,
            out_stream,
        )
        if ans.lower() not in ("y", "yes"):
            raise EgoError(f"{FILE_EXISTS_MESSAGE}: {abs_path}（已取消覆盖）")

    source_paths: dict[str, str] = {}
    for cat in ("sense", "action", "thought", "skill", "credentials"):
        default_dir = DEFAULT_SOURCE_PATHS[cat]
        entered = _prompt_line(
            f"请输入 [{cat}] 本地数据源目录（回车默认 {default_dir}）: ",
            in_stream,
            out_stream,
        )
        if entered:
            source_paths[cat] = entered
        else:
            source_paths[cat] = default_dir
            print(
                f"[init] {cat} 未输入自定义路径，使用默认目录：{default_dir}",
                file=out_stream,
                flush=True,
            )

    raw_editions = _prompt_line(
        "请选择要启用的 Edition 规格（逗号分隔，如 normal 或 lite,normal,plus,super；回车默认 normal）: ",
        in_stream,
        out_stream,
    )
    if not raw_editions:
        selected_names = [DEFAULT_EDITION_NAME]
    elif raw_editions.lower() == "all":
        selected_names = ["lite", "normal", "plus", "super"]
    else:
        selected_names = [
            part.strip() for part in raw_editions.split(",") if part.strip()
        ]
        if not selected_names:
            selected_names = [DEFAULT_EDITION_NAME]

    raw_base = _prompt_line(
        f"可选：指定主基座模型（支持 {', '.join(SUPPORTED_BASE_MODELS)}；回车按各规格预设自动选择）: ",
        in_stream,
        out_stream,
    )
    if raw_base and raw_base not in SUPPORTED_BASE_MODELS:
        raise ConfigError(
            f"交互式初始化收到不受支持的基座模型 {raw_base!r}；"
            f"受支持白名单：{', '.join(SUPPORTED_BASE_MODELS)}"
        )

    editions_cfg: dict[str, dict[str, Any]] = {}
    for idx, name in enumerate(selected_names):
        preset = EDITION_PRESETS.get(name)
        if raw_base and (name == DEFAULT_EDITION_NAME or idx == 0):
            chosen_base = raw_base
        elif preset is not None:
            chosen_base = (
                DEFAULT_BASE_MODEL
                if name == DEFAULT_EDITION_NAME and len(selected_names) == 1
                else preset.default_base
            )
        else:
            chosen_base = raw_base or DEFAULT_BASE_MODEL
        editions_cfg[name] = {
            "base": chosen_base,
            "description": f"{name} edition ({chosen_base})",
        }

    return render_custom_yaml(source_paths=source_paths, editions=editions_cfg)


def run_init(
    args: argparse.Namespace,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
) -> int:
    """``ego init`` 主入口：无交互标志时保持 100% 静默非交互，支持 ``--all-editions`` 与 ``--interactive``。"""

    force = bool(getattr(args, "force", False))
    all_editions = bool(getattr(args, "all_editions", False))
    interactive = bool(getattr(args, "interactive", False))
    out_stream = sys.stdout if stdout is None else stdout

    target = Path.cwd() / CONFIG_FILENAME
    abs_path = target.resolve()

    if interactive:
        in_stream = sys.stdin if stdin is None else stdin
        text = _run_interactive_wizard(
            target,
            abs_path,
            force=force,
            in_stream=in_stream,
            out_stream=out_stream,
        )
    else:
        if os.path.lexists(target) and not force:
            raise EgoError(f"{FILE_EXISTS_MESSAGE}: {abs_path}（如需覆盖请使用 --force）")
        text = render_all_editions_yaml() if all_editions else render_default_yaml()

    try:
        _atomic_write_text(target, text)
    except OSError as exc:
        raise EgoError(f"{WRITE_FAILED_MESSAGE}: {abs_path}: {exc}") from exc

    print(str(abs_path), file=out_stream, flush=True)
    return EXIT_OK
