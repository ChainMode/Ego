"""``ego train`` 端到端修炼场（Dojo）编排：增量采集、深层专精度蒸馏、多规格训练与版本归档。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import argparse
import io
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import yaml

from ego_pod.collector.incremental import compile_sources_incrementally
from ego_pod.collector.pipeline import run as run_collection
from ego_pod.compiler.pipeline import _write_cell_sub_datasets, run_compilation
from ego_pod.compiler.routing import prune_cells_for_edition
from ego_pod.config.loader import load as load_config
from ego_pod.config.schema import (
    EgoConfig,
    ResolvedEditionSpec,
    resolve_edition_spec,
)
from ego_pod.config.validators import validate as validate_config
from ego_pod.exit_codes import EXIT_OK, ConfigError
from ego_pod.fidelity.metrics import evaluate_corpus_fidelity
from ego_pod.logging_io import Event, Stage, print_summary, stage_event
from ego_pod.trainer.cells import CellSpec, TrainedCell
from ego_pod.trainer.grid import run_grid_training
from ego_pod.trainer.pipeline import run_training_with_telemetry
from ego_pod.vault.history import DEFAULT_MAX_HISTORY
from ego_pod.vault.package import (
    CONFIG_FILENAME,
    EditionBuildInput,
    MANIFEST_FILENAME,
)
from ego_pod.vault.pipeline import OUTPUT_DIRNAME, run_multi_edition_vault

__all__ = ["run_train", "DEFAULT_CONFIG_FILENAME"]

DEFAULT_CONFIG_FILENAME = "ego-config.yaml"
_CELLS_DIRNAME = "cells"


def run_train(args: argparse.Namespace) -> int:
    """``ego train`` 主入口：加载校验配置后执行 Collector → Compiler → Trainer → Vault。"""

    config_arg = getattr(args, "config", None)
    config_path = config_arg if config_arg else str(Path.cwd() / DEFAULT_CONFIG_FILENAME)
    update_mode = bool(getattr(args, "update", False))
    parallel_editions = bool(getattr(args, "parallel_editions", False))

    config = _load_and_validate_config(config_path)
    return _run_pipeline(
        config,
        update=update_mode,
        parallel_editions=parallel_editions,
    )


def _load_and_validate_config(config_path: str) -> EgoConfig:
    try:
        config, abs_path = load_config(config_path)
        raw_mapping = _read_raw_mapping(abs_path)
        validate_config(raw_mapping, abs_path)
    except ConfigError as exc:
        raise _ensure_path_in_message(exc, config_path) from exc
    return config


def _read_raw_mapping(abs_path: str) -> Any:
    try:
        raw_bytes = Path(abs_path).read_bytes()
        return yaml.safe_load(raw_bytes)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(
            f"读取配置文件 {abs_path} 时发生错误：{exc}"
        ) from exc


def _ensure_path_in_message(exc: ConfigError, config_path: str) -> ConfigError:
    abs_path = os.path.abspath(config_path)
    message = str(exc)
    if abs_path in message or config_path in message:
        return exc
    return ConfigError(f"{message}（所用配置路径：{abs_path}）")


def _detect_base_model_changes(
    cwd: Path, edition_specs: list[ResolvedEditionSpec]
) -> None:
    """检查已有 ``ego-output/<edition>/config.json`` 是否发生基座模型切换并记录全量重训事件。"""

    output_root = cwd / OUTPUT_DIRNAME
    for spec in edition_specs:
        prev_cfg_path = output_root / spec.name / CONFIG_FILENAME
        if not prev_cfg_path.is_file():
            continue
        try:
            prev_cfg = json.loads(prev_cfg_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(prev_cfg, dict):
            prev_base = prev_cfg.get("base_model")
            if isinstance(prev_base, str) and prev_base != spec.base:
                print_summary(
                    f"[dojo] base model changed for edition '{spec.name}': "
                    f"{prev_base} -> {spec.base}; forcing clean full retrain"
                )


def _train_single_edition_worker(
    spec: ResolvedEditionSpec,
    *,
    cwd: Path,
    corpus_path: Path,
    deep_routed: dict,
    single_flat_mode: bool,
    capture_logs: bool,
) -> tuple[EditionBuildInput, str, str, int]:
    """训练单个 Edition（支持顺序直打日志或并行缓冲日志），返回构建载荷与后端标识。"""

    buf = io.StringIO() if capture_logs else None
    ed_cells_dir = cwd / ".ego" / "cells_editions" / spec.name
    ed_out_dir = cwd / ".ego" / "cells_out" / spec.name
    shutil.rmtree(ed_cells_dir, ignore_errors=True)
    shutil.rmtree(ed_out_dir, ignore_errors=True)

    if single_flat_mode:
        lora_out = ed_out_dir / "lora.safetensors"
        tr_res, eff_bs, oom_retries = run_training_with_telemetry(
            corpus_path,
            spec.training,
            model_id=spec.base,
            cwd=cwd,
            output_path=lora_out,
            stdout=buf,
        )
        build_in = EditionBuildInput(
            edition=spec.name,
            base_model=spec.base,
            lora_rank=spec.training.lora_rank,
            lora_alpha=spec.training.lora_alpha,
            max_size_mb=spec.max_size_mb,
            cells=None,
            lora_source=tr_res.output_path,
            effective_batch_size=eff_bs,
            oom_retries=oom_retries,
        )
        return build_in, tr_res.backend_identifier, (buf.getvalue() if buf else ""), 1

    pruned_routed = prune_cells_for_edition(
        deep_routed,
        max_level=spec.max_level,
        allowed_zones=spec.zones,
    )
    sub_datasets = _write_cell_sub_datasets(pruned_routed, ed_cells_dir)
    cell_specs = [
        CellSpec(
            zone=sub.cell.zone,
            level=sub.cell.level,
            sub_dataset_path=ed_cells_dir / sub.relpath,
        )
        for sub in sub_datasets
    ]
    grid_res = run_grid_training(
        cell_specs,
        spec.training,
        model_id=spec.base,
        cwd=cwd,
        cells_out_dir=ed_out_dir,
        stdout=buf,
    )
    trained_cells: list[TrainedCell] = list(grid_res.cells)
    build_in = EditionBuildInput(
        edition=spec.name,
        base_model=spec.base,
        lora_rank=spec.training.lora_rank,
        lora_alpha=spec.training.lora_alpha,
        max_size_mb=spec.max_size_mb,
        cells=trained_cells,
        lora_source=None,
        effective_batch_size=grid_res.effective_batch_size,
        oom_retries=grid_res.oom_retries,
    )
    return (
        build_in,
        grid_res.backend_identifier,
        (buf.getvalue() if buf else ""),
        len(trained_cells),
    )


def _run_pipeline(
    config: EgoConfig,
    *,
    update: bool = False,
    parallel_editions: bool = False,
) -> int:
    cwd = Path.cwd()
    edition_specs = [
        resolve_edition_spec(name, ed_def, config.training)
        for name, ed_def in config.editions.items()
    ]
    global_max_level = max(spec.max_level for spec in edition_specs)

    # --- 阶段 1：Collector (25%) ----------------------------------------------
    t0 = time.monotonic()
    stage_event(Stage.COLLECTOR, Event.START)
    collection = run_collection(config, cwd=cwd)
    inc_plan = compile_sources_incrementally(
        collection.files, cwd=cwd, update=update
    )
    if update:
        print_summary(
            f"[incremental] unchanged={inc_plan.unchanged_count} "
            f"changed_or_new={inc_plan.changed_or_new_count} "
            f"removed={inc_plan.removed_count}"
        )
    stage_event(Stage.COLLECTOR, Event.END)
    collector_ms = int((time.monotonic() - t0) * 1000)
    print_summary(
        f"[dojo progress] 25% COLLECTOR completed ({len(collection.files)} files, {collector_ms}ms)"
    )

    # --- 阶段 2：Compiler & Deep Grid Distillation (50%) ----------------------
    t1 = time.monotonic()
    stage_event(Stage.COMPILER, Event.START)
    compilation = run_compilation(
        collection.files,
        zone_config=config.zones,
        default_max_level=global_max_level,
        precomputed_source_samples=(
            inc_plan.source_samples,
            inc_plan.redact_counters,
        ),
        cwd=cwd,
    )
    stage_event(Stage.COMPILER, Event.END)
    compiler_ms = int((time.monotonic() - t1) * 1000)
    print_summary(
        f"[dojo progress] 50% COMPILER completed ({len(compilation.samples)} samples, {compiler_ms}ms)"
    )

    deep_routed = {sub.cell: sub.samples for sub in compilation.cells}
    has_zone_cells = any(not sub.cell.is_kernel for sub in compilation.cells)
    single_flat_mode = len(edition_specs) == 1 and not has_zone_cells

    prev_manifest = cwd / OUTPUT_DIRNAME / MANIFEST_FILENAME
    fidelity_report = evaluate_corpus_fidelity(
        compilation.corpus_path,
        previous_manifest_path=prev_manifest,
    )
    _detect_base_model_changes(cwd, edition_specs)

    # --- 阶段 3：Multi-Edition Trainer (85%) ----------------------------------
    t2 = time.monotonic()
    stage_event(Stage.TRAINER, Event.START)
    edition_builds: list[EditionBuildInput] = []
    cell_counts_by_ed: dict[str, int] = {}
    backend_id = "stub"

    if parallel_editions and len(edition_specs) > 1:
        max_workers = min(4, len(edition_specs))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(
                    _train_single_edition_worker,
                    spec,
                    cwd=cwd,
                    corpus_path=compilation.corpus_path,
                    deep_routed=deep_routed,
                    single_flat_mode=False,
                    capture_logs=True,
                )
                for spec in edition_specs
            ]
            for fut in futures:
                b_in, b_id, log_text, n_cells = fut.result()
                if log_text:
                    for line in log_text.splitlines():
                        print_summary(line)
                edition_builds.append(b_in)
                cell_counts_by_ed[b_in.edition] = n_cells
                backend_id = b_id
    else:
        for spec in edition_specs:
            b_in, b_id, _logs, n_cells = _train_single_edition_worker(
                spec,
                cwd=cwd,
                corpus_path=compilation.corpus_path,
                deep_routed=deep_routed,
                single_flat_mode=single_flat_mode,
                capture_logs=False,
            )
            edition_builds.append(b_in)
            cell_counts_by_ed[b_in.edition] = n_cells
            backend_id = b_id

    stage_event(Stage.TRAINER, Event.END)
    trainer_ms = int((time.monotonic() - t2) * 1000)
    print_summary(
        f"[dojo progress] 85% TRAINER completed (editions={','.join(s.name for s in edition_specs)}, {trainer_ms}ms)"
    )

    # --- 阶段 4：Vault & Signing (100%) ---------------------------------------
    t3 = time.monotonic()
    stage_event(Stage.VAULT, Event.START)
    raw_max_hist = (
        config.output.get("max_history", DEFAULT_MAX_HISTORY)
        if isinstance(config.output, dict)
        else DEFAULT_MAX_HISTORY
    )
    max_history = (
        int(raw_max_hist)
        if isinstance(raw_max_hist, int) and not isinstance(raw_max_hist, bool)
        else DEFAULT_MAX_HISTORY
    )
    run_multi_edition_vault(
        edition_builds,
        training_backend=backend_id,
        cwd=cwd,
        fidelity_report=fidelity_report,
        include_integrity=True,
        max_history=max_history,
    )
    stage_event(Stage.VAULT, Event.END)
    vault_ms = int((time.monotonic() - t3) * 1000)
    print_summary(f"[dojo progress] 100% VAULT completed ({vault_ms}ms)")

    cells_desc = " ".join(
        f"{ed}:{cnt}cells" for ed, cnt in cell_counts_by_ed.items()
    )
    fid_loss = fidelity_report.get("holdout_loss")
    fid_str = f"{fid_loss:.4f}" if isinstance(fid_loss, (int, float)) else str(fidelity_report.get("status"))
    print_summary(
        "cultivation summary: "
        f"editions={','.join(s.name for s in edition_specs)} "
        f"samples={len(compilation.samples)} "
        f"grid=[{cells_desc}] "
        f"fidelity={fid_str} "
        f"elapsed_ms={{collector:{collector_ms},compiler:{compiler_ms},trainer:{trainer_ms},vault:{vault_ms}}}"
    )
    return EXIT_OK


def _build_cell_specs(compilation: Any) -> list[CellSpec]:
    """保留供向后兼容引用的辅助函数：把 CompilationResult.cells 转为 CellSpec 列表。"""

    cells = getattr(compilation, "cells", None)
    if not cells:
        return []

    cells_root = Path(compilation.corpus_path).parent / _CELLS_DIRNAME
    return [
        CellSpec(
            zone=sub.cell.zone,
            level=sub.cell.level,
            sub_dataset_path=cells_root / sub.relpath,
        )
        for sub in cells
    ]
