"""Ego_Package 写入与 manifest/config 构造（需求 10.2、10.3、10.4、10.5、12.9）。

本模块提供 Vault 模块把 Trainer 临时产出落盘为 Ego_Package 的核心写入能力，
对应设计文档 §6（Vault 模块）的「Ego_Package 写入流程」与「事务性策略」：

* :func:`build_config` —— 构造 ``normal/config.json`` 的内容字典
  （``base_model`` / ``quantization=4bit`` / ``lora_rank`` / ``lora_alpha``，
  需求 10.4）。
* :func:`build_manifest` —— 构造 ``manifest.json`` 的内容字典
  （``version`` SemVer / ``created_at`` ISO 8601 / ``base_model`` /
  ``editions`` 非空且含 ``normal`` / ``training_backend``，需求 10.2、12.9）。
* :func:`write_package` —— 事务性地写出整份 Ego_Package。

**事务性策略**（设计文档 §6「事务性策略」、需求 10.5）：先把全部产物写入
``ego-output/.tmp/`` 暂存区，全部成功后再以原子 rename 把旧的
``ego-output/normal/`` 与 ``ego-output/manifest.json`` 替换为新内容（先把旧
``normal/`` 改名搁置，再把新 ``normal/`` rename 就位，最后删除被搁置的旧目录）；
中途任一步失败则清理 ``.tmp/`` 半成品并保留上一次成功的旧产物，再以
:class:`~ego_pod.exit_codes.VaultError` 抛出（携带失败路径与原因，供上层
按需求 10.7 打印到标准错误并以退出码 1 结束）。

``training_backend`` 字段虽未在 R10-AC2 中列为强制字段，但 ``ego run`` 在 stub
模式下需据此识别（需求 12.9），因此作为 MVP 的补充字段写入 ``manifest.json``；
其取值即 Trainer 选中的后端名称（Stub_Backend 为 ``"stub"``）。

本模块只负责「写产物」与「构造产物字典」；``ego-output/`` 的创建/识别事件、
三文件存在性校验、Ego_Package 绝对路径打印与退出码映射由 ``vault.pipeline``
（任务 9.2）编排。
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from ego_pod import __version__
from ego_pod.exit_codes import EgoError, VaultError
from ego_pod.trainer.cells import TrainedCell

if TYPE_CHECKING:  # 仅供类型注解；运行期惰性导入以避免与 runtime/grid_descriptor 循环。
    from ego_pod.runtime.selection import CellKey

__all__ = [
    "BASE_MODEL",
    "QUANTIZATION",
    "EDITION_NAME",
    "ADAPTER_FILENAME",
    "MANIFEST_FILENAME",
    "CONFIG_FILENAME",
    "TMP_DIRNAME",
    "PackageArtifacts",
    "GridPackageArtifacts",
    "utc_now_iso8601",
    "build_config",
    "build_manifest",
    "write_package",
    "write_grid_package",
]


# ---------------------------------------------------------------------------
# Ego_Package 结构常量
# ---------------------------------------------------------------------------

#: MVP 固定基座模型标识（需求 10.2、10.4）。
BASE_MODEL = "qwen3.5-0.8B"
#: MVP 固定量化方案（需求 10.4）。
QUANTIZATION = "4bit"
#: MVP 唯一 Edition 名称；Ego_Package 根部恰好包含一个该名称的子目录（需求 10.3）。
EDITION_NAME = "normal"
#: LoRA_Adapter 权重文件名，位于 ``<edition>/adapters/`` 下（需求 10.3）。
ADAPTER_FILENAME = "ego-lora.safetensors"
#: Ego_Package 根部 manifest 文件名（需求 10.2）。
MANIFEST_FILENAME = "manifest.json"
#: Edition 子目录下的配置文件名（需求 10.3、10.4）。
CONFIG_FILENAME = "config.json"
#: 事务性写入使用的暂存子目录名（设计文档 §6「事务性策略」）。
TMP_DIRNAME = ".tmp"


@dataclass(frozen=True)
class PackageArtifacts:
    """一次成功写入后产生的 Ego_Package 关键路径（均为绝对路径）。

    供 ``vault.pipeline`` 校验三文件存在（需求 10.6）并打印 Ego_Package 根目录
    绝对路径。
    """

    #: Ego_Package 根目录（即 ``ego-output/``）的绝对路径。
    root: Path
    #: ``manifest.json`` 的绝对路径。
    manifest_path: Path
    #: ``normal/config.json`` 的绝对路径。
    config_path: Path
    #: ``normal/adapters/ego-lora.safetensors`` 的绝对路径。
    adapter_path: Path


@dataclass(frozen=True)
class GridPackageArtifacts:
    """一次成功 :func:`write_grid_package` 后产生的 Grid_Package 关键路径。

    与扁平路径的 :class:`PackageArtifacts` 并列，供 ``vault.pipeline`` 校验
    「``manifest.json`` + ``normal/config.json`` + grid 所列全部 Cell 适配器文件」
    同时存在（R4-AC8）。所有路径均为绝对路径。

    Grid-aware counterpart of :class:`PackageArtifacts`: instead of a single
    ``adapter_path`` it carries a mapping from each written Cell to its adapter
    file path, so the pipeline can verify every Cell listed in the grid exists.
    """

    #: Ego_Package 根目录（即 ``ego-output/``）的绝对路径。
    root: Path
    #: ``manifest.json`` 的绝对路径。
    manifest_path: Path
    #: ``normal/config.json`` 的绝对路径。
    config_path: Path
    #: 每个已写出 Cell（:class:`~ego_pod.runtime.selection.CellKey`）→ 其适配器
    #: 文件绝对路径的映射（Kernel → ``adapters/kernel.safetensors``，分区 Cell →
    #: ``adapters/<zone>/L<level>.safetensors``）。
    cell_paths: dict[CellKey, Path]


# ---------------------------------------------------------------------------
# 时间戳工具
# ---------------------------------------------------------------------------


def utc_now_iso8601() -> str:
    """返回当前 UTC 时间的 ISO 8601 字符串，精确到秒并以 ``Z`` 结尾。

    例如 ``2025-11-25T08:00:00Z``，满足需求 10.2 对 ``created_at`` 的格式约束。
    测试可对本函数进行 monkeypatch 以注入固定时钟。
    """

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# 产物字典构造（纯函数）
# ---------------------------------------------------------------------------


def build_config(
    *,
    lora_rank: int,
    lora_alpha: int,
    base_model: str = BASE_MODEL,
) -> dict:
    """构造 ``normal/config.json`` 的内容字典（需求 10.4）。

    :param lora_rank: LoRA 秩，正整数。
    :param lora_alpha: LoRA alpha，正整数。
    :param base_model: 基座模型标识，默认 :data:`BASE_MODEL`。
    :returns: 含 ``base_model`` / ``quantization`` / ``lora_rank`` / ``lora_alpha``
        四个字段的字典；``quantization`` 固定为 :data:`QUANTIZATION`。
    """

    return {
        "base_model": base_model,
        "quantization": QUANTIZATION,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
    }


def build_manifest(
    *,
    training_backend: str,
    base_model: str = BASE_MODEL,
    editions: tuple[str, ...] = (EDITION_NAME,),
    version: str | None = None,
    created_at: str | None = None,
    grid: dict | None = None,
) -> dict:
    """构造 ``manifest.json`` 的内容字典（需求 10.2、12.9、R4-AC4）。

    :param training_backend: Trainer 选中的后端名称（Stub_Backend 为 ``"stub"``），
        作为 stub 标记供 ``ego run`` 识别（需求 12.9）。
    :param base_model: 基座模型标识，默认 :data:`BASE_MODEL`。
    :param editions: Edition 名称序列，必须非空且包含 ``normal``（需求 10.2）。
    :param version: 语义化版本号字符串；为 ``None`` 时取包的 ``__version__``。
    :param created_at: UTC ISO 8601 字符串；为 ``None`` 时取
        :func:`utc_now_iso8601` 的当前时间。
    :param grid: 可选的 Grid_Descriptor（``manifest.json`` 顶层 ``grid`` 键的值，
        见 :func:`ego_pod.vault.grid_descriptor.build_grid_descriptor`）。

        * ``grid is None``（默认）⇒ 产出与 ego-pod-mvp / ego-pod-real-trainer
          **逐字节一致**的五字段 manifest（``version`` / ``created_at`` /
          ``base_model`` / ``editions`` / ``training_backend``，顺序/类型/取值不变，
          R4-AC4）；**不**追加任何 ``grid`` 键。这正是 Flat_Package 的形状。
        * ``grid is not None`` ⇒ 在上述五字段之后追加一个 ``grid`` 键
          （Grid_Package 的形状，R4-AC1）；五字段本身完全不受影响。
    :returns: 含五个旧字段（以及当 ``grid`` 非 ``None`` 时追加的 ``grid`` 键）的字典。
    :raises VaultError: 当 ``editions`` 为空或不包含 ``normal`` 时。
    """

    edition_list = list(editions)
    if not edition_list or EDITION_NAME not in edition_list:
        raise VaultError(
            f"manifest.editions 必须为非空数组且包含 '{EDITION_NAME}'，"
            f"实际为 {edition_list!r}"
        )

    # 五个旧字段：名称、顺序、类型、取值与 MVP / real-trainer 逐字节一致（R4-AC4）。
    payload: dict = {
        "version": __version__ if version is None else version,
        "created_at": utc_now_iso8601() if created_at is None else created_at,
        "base_model": base_model,
        "editions": edition_list,
        "training_backend": training_backend,
    }
    # 仅在网格包追加 grid 键；grid=None 时不改动五字段输出（R4-AC1/AC4）。
    if grid is not None:
        payload["grid"] = grid
    return payload


# ---------------------------------------------------------------------------
# 内部写入辅助
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: dict) -> None:
    """以 UTF-8（无 BOM）写入合法 JSON 文件（需求 10.2、10.4）。"""

    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    path.write_text(text, encoding="utf-8")


def _atomic_replace_dir(src: Path, dst: Path) -> None:
    """原子地用 ``src`` 目录替换 ``dst`` 目录（设计文档 §6「事务性策略」）。

    ``os.replace`` 无法直接覆盖非空目录，故采用「先把旧目录改名搁置，再把新
    目录 rename 就位，最后删除被搁置的旧目录」的三步法（先 rename、再 unlink
    旧的）；若新目录就位失败则把旧目录还原，保证旧产物不丢失。
    """

    if dst.exists():
        backup = dst.parent / f"{dst.name}.old.{os.getpid()}"
        shutil.rmtree(backup, ignore_errors=True)
        os.replace(dst, backup)  # 旧 → 搁置（原子 rename）
        try:
            os.replace(src, dst)  # 新 → 就位（原子 rename）
        except OSError:
            os.replace(backup, dst)  # 失败回滚：还原旧产物
            raise
        shutil.rmtree(backup, ignore_errors=True)  # unlink 旧的
    else:
        os.replace(src, dst)


def _remove_orphans(root: Path) -> None:
    """删除 ``ego-output/`` 下既非本次产物亦非暂存目录的孤儿项（需求 10.5）。

    替换完成后，仅保留 ``manifest.json`` 与 ``normal/``；``.tmp/`` 由调用方在
    收尾阶段清理。其余历史遗留项（上一次的孤儿文件/目录）一律删除，从而满足
    R10-AC5「删除 ego-output/ 目录下的所有旧产物（含本次未生成的孤儿文件）」的
    覆盖语义。
    """

    keep = {MANIFEST_FILENAME, EDITION_NAME, TMP_DIRNAME}
    for child in root.iterdir():
        if child.name in keep:
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child, ignore_errors=True)
        else:
            try:
                child.unlink()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# 事务性写入
# ---------------------------------------------------------------------------


def write_package(
    output_root: str | Path,
    lora_source: str | Path,
    *,
    lora_rank: int,
    lora_alpha: int,
    training_backend: str,
    base_model: str = BASE_MODEL,
    editions: tuple[str, ...] = (EDITION_NAME,),
    version: str | None = None,
    created_at: str | None = None,
) -> PackageArtifacts:
    """把 Trainer 临时产出事务性地写为 Ego_Package（需求 10.2–10.5、12.9）。

    写入流程（设计文档 §6）：

    1. 在 ``output_root/.tmp/`` 暂存区构建 ``normal/config.json``、把
       ``lora_source`` 移动到 ``normal/adapters/ego-lora.safetensors``、写
       ``manifest.json``；
    2. 全部成功后，原子地把旧 ``normal/`` 与 ``manifest.json`` 替换为新内容；
    3. 删除历史孤儿项并清理暂存目录。

    中途任一步发生 I/O 错误时，清理 ``.tmp/`` 半成品并保留上一次成功的旧产物，
    再以 :class:`VaultError` 抛出（携带失败路径与原因）。

    :param output_root: Ego_Package 根目录（即 ``ego-output/``）；不存在则创建。
    :param lora_source: Trainer 输出的 ``.ego/lora.safetensors`` 路径，将被移动到
        ``normal/adapters/ego-lora.safetensors``。
    :param lora_rank: 写入 ``config.json`` 的 LoRA 秩（正整数）。
    :param lora_alpha: 写入 ``config.json`` 的 LoRA alpha（正整数）。
    :param training_backend: 写入 ``manifest.json`` 的后端名称（stub 标记）。
    :param base_model: 基座模型标识，默认 :data:`BASE_MODEL`。
    :param editions: Edition 名称序列，默认 ``("normal",)``。
    :param version: 语义化版本号；为 ``None`` 时取包的 ``__version__``。
    :param created_at: UTC ISO 8601 字符串；为 ``None`` 时取当前时间。
    :returns: 写入产物的关键绝对路径（:class:`PackageArtifacts`）。
    :raises VaultError: 当 ``lora_source`` 不存在、或写入过程中遭遇 I/O 错误时。
    """

    root = Path(output_root)
    source = Path(lora_source)

    if not source.is_file():
        raise VaultError(
            f"Trainer 输出的 LoRA 权重不存在或不是常规文件：{source.resolve()}"
        )

    # 构造产物字典（editions 校验在此触发，先于任何落盘动作）。
    config_payload = build_config(
        lora_rank=lora_rank, lora_alpha=lora_alpha, base_model=base_model
    )
    manifest_payload = build_manifest(
        training_backend=training_backend,
        base_model=base_model,
        editions=editions,
        version=version,
        created_at=created_at,
    )

    root.mkdir(parents=True, exist_ok=True)
    tmp_dir = root / TMP_DIRNAME
    # 清理上一次失败可能残留的暂存目录，确保暂存区干净。
    shutil.rmtree(tmp_dir, ignore_errors=True)

    final_normal = root / EDITION_NAME
    final_manifest = root / MANIFEST_FILENAME

    try:
        # --- 1. 在 .tmp/ 暂存区构建全部产物 -----------------------------------
        tmp_normal = tmp_dir / EDITION_NAME
        tmp_adapters = tmp_normal / "adapters"
        tmp_adapters.mkdir(parents=True, exist_ok=True)

        _write_json(tmp_normal / CONFIG_FILENAME, config_payload)
        # 移动 Trainer 输出到暂存区适配器目录（设计文档 §6 步骤 5）。
        shutil.move(str(source), str(tmp_adapters / ADAPTER_FILENAME))

        tmp_manifest = tmp_dir / MANIFEST_FILENAME
        _write_json(tmp_manifest, manifest_payload)

        # --- 2. 全部成功后原子替换旧产物（需求 10.5）-------------------------
        _atomic_replace_dir(tmp_normal, final_normal)
        os.replace(tmp_manifest, final_manifest)

        # --- 3. 删除历史孤儿项，满足覆盖语义（需求 10.5）---------------------
        _remove_orphans(root)
    except OSError as exc:
        # I/O 错误：清理 .tmp/ 半成品，保留上一次成功的旧产物（需求 10.7）。
        shutil.rmtree(tmp_dir, ignore_errors=True)
        failed_path = getattr(exc, "filename", None) or tmp_dir
        raise VaultError(
            f"写入 Ego_Package 失败：{Path(failed_path).resolve()}；原因：{exc}"
        ) from exc
    finally:
        # 收尾：移除可能残留的空暂存目录（成功路径下其内容已被 rename 移出）。
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return PackageArtifacts(
        root=root.resolve(),
        manifest_path=final_manifest.resolve(),
        config_path=(final_normal / CONFIG_FILENAME).resolve(),
        adapter_path=(final_normal / "adapters" / ADAPTER_FILENAME).resolve(),
    )


# ---------------------------------------------------------------------------
# 事务性写入（网格路径）
# ---------------------------------------------------------------------------


def write_grid_package(
    output_root: str | Path,
    *,
    cells: list[TrainedCell],
    lora_rank: int,
    lora_alpha: int,
    training_backend: str,
    base_model: str = BASE_MODEL,
    editions: tuple[str, ...] = (EDITION_NAME,),
    version: str | None = None,
    created_at: str | None = None,
) -> GridPackageArtifacts:
    """把按 Cell 训练的多个适配器事务性地写为 Grid_Package（R4-AC3/AC8/AC9/AC10）。

    Transactionally persist a list of per-Cell trained adapters as a
    Grid_Package, mirroring :func:`write_package` 的事务性写入模式。

    写入流程（设计文档 §"Components and Interfaces → 4. Vault → 4.2"）：

    1. **kernel-always 硬约束（决策 B）**：``cells`` 必须恰好含一个 Kernel Cell
       （``zone=None, level=0``）；缺失 ⇒ 在写任何文件之前以 :class:`VaultError`
       快速失败，避免把"缺失 Kernel"的隐患推迟到加载期。
    2. 在 ``output_root/.tmp/`` 暂存区构建 ``normal/config.json``（复用
       :func:`build_config`）；把每个 :class:`~ego_pod.trainer.cells.TrainedCell`
       的 ``output_path`` 适配器移动到其网格落点——Kernel →
       ``normal/adapters/kernel.safetensors``、分区 Cell →
       ``normal/adapters/<zone>/L<level>.safetensors``（按需创建 ``<zone>/`` 子目录，
       R4-AC3）；经 :func:`~ego_pod.vault.grid_descriptor.build_grid_descriptor`
       构造 ``grid`` 描述并写 ``manifest.json``（含 ``grid``）。
    3. 全部成功后，原子地把旧 ``normal/`` 与 ``manifest.json`` 替换为新内容
       （复用 :func:`_atomic_replace_dir` 与:func:`os.replace`），再删除历史孤儿项
       （复用 :func:`_remove_orphans`，keep 集合不变）。网格的全部 Cell 文件都落在
       ``normal/adapters/`` 子树内，随 ``normal/`` 整体原子替换，事务性语义原样成立
       （R4-AC9）。

    中途任一步发生 I/O 错误时，清理 ``.tmp/`` 半成品并保留上一次成功的旧产物，
    再以 :class:`VaultError` 抛出（携带失败路径与原因，R4-AC10）；已构造的
    :class:`~ego_pod.exit_codes.EgoError` 优先透传不被二次包装。

    :param output_root: Ego_Package 根目录（即 ``ego-output/``）；不存在则创建。
    :param cells: Trainer 产出的已训练 Cell 列表，须含恰好一个 Kernel Cell
        （``zone=None, level=0``）；每个 Cell 的 ``output_path`` 为其导出的 LoRA
        适配器临时文件，将被移动到对应网格落点。
    :param lora_rank: 写入 ``config.json`` 的 LoRA 秩（正整数）。
    :param lora_alpha: 写入 ``config.json`` 的 LoRA alpha（正整数）。
    :param training_backend: 写入 ``manifest.json`` 的后端名称（stub 标记）。
    :param base_model: 基座模型标识，默认 :data:`BASE_MODEL`。
    :param editions: Edition 名称序列，默认 ``("normal",)``。
    :param version: 语义化版本号；为 ``None`` 时取包的 ``__version__``。
    :param created_at: UTC ISO 8601 字符串；为 ``None`` 时取当前时间。
    :returns: 写入产物的关键绝对路径（:class:`GridPackageArtifacts`，含每个 Cell
        适配器文件的绝对路径）。
    :raises VaultError: ``cells`` 缺失 Kernel、某 Cell 适配器源文件不存在、或写入
        过程中遭遇 I/O 错误时。
    """

    # 惰性导入，避免与 grid_descriptor 的模块级循环导入（后者依赖本模块的
    # EDITION_NAME）。grid_descriptor 与 runtime.selection 均为纯 Python、无 ML
    # 依赖，导入安全。
    from ego_pod.runtime.selection import KERNEL, CellKey
    from ego_pod.vault.grid_descriptor import (
        KERNEL_RELPATH,
        build_grid_descriptor,
        zone_cell_relpath,
    )

    root = Path(output_root)

    # --- 1. kernel-always 硬约束（决策 B）：写任何文件之前快速失败 -------------
    kernel_cells = [c for c in cells if c.zone is None and c.level == 0]
    if len(kernel_cells) != 1:
        raise VaultError(
            "Grid_Package 必须含非空 Kernel（与分区无关的 Level-0 Cell），"
            f"实际找到 {len(kernel_cells)} 个 Kernel Cell；"
            "请检查 Zone_Config 是否把全部 Data_Category 都路由出了 Kernel"
        )

    # 解析每个 Cell 的网格相对路径与可寻址键（Kernel → KERNEL_RELPATH，
    # 分区 Cell → adapters/<zone>/L<level>.safetensors）。
    planned: list[tuple[CellKey, Path, str]] = []
    for cell in cells:
        if cell.zone is None and cell.level == 0:
            key = KERNEL
            relpath = KERNEL_RELPATH
        else:
            key = CellKey(zone=cell.zone, level=cell.level)
            relpath = zone_cell_relpath(cell.zone, cell.level)
        planned.append((key, Path(cell.output_path), relpath))

    # 在删除/替换任何旧产物之前，先校验全部 Cell 适配器源文件存在（R4-AC10：
    # 失败路径明确；与 write_package "先有新权重、再替换旧产物"的事务顺序一致）。
    for _key, source, _relpath in planned:
        if not source.is_file():
            raise VaultError(
                f"Trainer 输出的 Cell LoRA 权重不存在或不是常规文件：{source.resolve()}"
            )

    # 构造产物字典（editions 校验在 build_manifest 内触发，先于任何落盘动作）。
    config_payload = build_config(
        lora_rank=lora_rank, lora_alpha=lora_alpha, base_model=base_model
    )
    grid = build_grid_descriptor(cells, edition=EDITION_NAME)
    manifest_payload = build_manifest(
        training_backend=training_backend,
        base_model=base_model,
        editions=editions,
        version=version,
        created_at=created_at,
        grid=grid,
    )

    root.mkdir(parents=True, exist_ok=True)
    tmp_dir = root / TMP_DIRNAME
    # 清理上一次失败可能残留的暂存目录，确保暂存区干净。
    shutil.rmtree(tmp_dir, ignore_errors=True)

    final_normal = root / EDITION_NAME
    final_manifest = root / MANIFEST_FILENAME

    try:
        # --- 2. 在 .tmp/ 暂存区构建全部产物 -----------------------------------
        tmp_normal = tmp_dir / EDITION_NAME
        tmp_normal.mkdir(parents=True, exist_ok=True)

        _write_json(tmp_normal / CONFIG_FILENAME, config_payload)

        # 把每个 Cell 适配器移动到其网格落点（按需创建 <zone>/ 子目录，R4-AC3）。
        for _key, source, relpath in planned:
            dest = tmp_normal / relpath
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(dest))

        tmp_manifest = tmp_dir / MANIFEST_FILENAME
        _write_json(tmp_manifest, manifest_payload)

        # --- 3. 全部成功后原子替换旧产物并清理孤儿（R4-AC8/AC9）---------------
        _atomic_replace_dir(tmp_normal, final_normal)
        os.replace(tmp_manifest, final_manifest)
        _remove_orphans(root)
    except EgoError:
        # 已构造的 EgoError 优先透传，避免二次包装、退出码语义被覆盖。
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    except OSError as exc:
        # I/O 错误：清理 .tmp/ 半成品，保留上一次成功的旧产物（R4-AC10）。
        shutil.rmtree(tmp_dir, ignore_errors=True)
        failed_path = getattr(exc, "filename", None) or tmp_dir
        raise VaultError(
            f"写入 Grid_Package 失败：{Path(failed_path).resolve()}；原因：{exc}"
        ) from exc
    finally:
        # 收尾：移除可能残留的空暂存目录（成功路径下其内容已被 rename 移出）。
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # 解析每个 Cell 适配器文件的最终绝对路径（供 pipeline 校验全部 Cell 存在）。
    cell_paths = {
        key: (final_normal / relpath).resolve() for key, _source, relpath in planned
    }

    return GridPackageArtifacts(
        root=root.resolve(),
        manifest_path=final_manifest.resolve(),
        config_path=(final_normal / CONFIG_FILENAME).resolve(),
        cell_paths=cell_paths,
    )
