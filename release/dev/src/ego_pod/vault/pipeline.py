"""Vault 编排（需求 10.1、10.5、10.6、10.7）。

本模块实现 :func:`run_vault`：把 Trainer 的临时产出装配为一次完整的 Ego_Package
持久化，对应设计文档 §6「Vault 模块」的 Ego_Package 写入流程：

1. **创建/识别 ``ego-output/``**：计算目标根目录 ``{cwd}/ego-output/``，不存在则
   创建；目录「已创建」或「已存在」作为一个独立可观测结果在标准输出记录一行
   （R10-AC1 的第一个独立产出，即 directory-created 事件）。
2. **等待 Trainer 输出**：Trainer 成功导出的 ``{cwd}/.ego/lora.safetensors`` 路径
   由调用方（任务 11.1 的 ``ego train`` 编排）作为入参传入；在 Trainer 成功提供
   该权重之前，本流水线不删除任何既有产物（R10-AC5）。``write_package``（任务 9.1）
   会在删除/替换任何旧产物之前先校验该权重存在，从而保证「先有新权重、再替换旧
   产物」的事务顺序。
3. **写产物**：调用 :func:`ego_pod.vault.package.write_package` 事务性地写出
   ``manifest.json`` 与 ``normal/`` 子目录（先写 ``ego-output/.tmp/`` 暂存区，全部
   成功后原子替换旧产物；R10-AC2/AC3/AC4/AC5、12.9）。
4. **校验三文件存在**（R10-AC6）：核对 ``manifest.json``、``normal/config.json``、
   ``normal/adapters/ego-lora.safetensors`` 三者均已落盘。
5. **打印 package 绝对路径**（R10-AC6）：在标准输出以单行形式打印 Ego_Package 根
   目录的绝对路径（以 ``/`` 开头），作为 package-created 事件——这是 R10-AC1 的第二
   个独立可观测产出，与第 1 步的 directory-created 事件相互区分。
6. 成功返回 :class:`~ego_pod.vault.package.PackageArtifacts`；由 CLI 顶层（任务
   11.1）映射为退出码 0。

**失败处理（R10-AC7）**：写入过程中遭遇磁盘 I/O 错误（磁盘已满、权限拒绝、目标
只读等）时，``write_package`` 已负责清理 ``ego-output/.tmp/`` 半成品并保留上一次
成功的旧 ``normal/``，随后抛出携带「失败文件绝对路径与具体原因」的
:class:`~ego_pod.exit_codes.VaultError`。本流水线**不拦截**该异常，原样上抛，交由
CLI 顶层打印到标准错误并以退出码 1 结束（与 ``collector.run`` /
``trainer.run_training`` 的错误传播约定一致）。三文件存在性校验若意外失败，同样
以 ``VaultError`` 抛出。

设计依据：design.md §Components and Interfaces / 6. Vault 模块。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TextIO

from ego_pod.exit_codes import VaultError
from ego_pod.trainer.cells import TrainedCell
from ego_pod.vault.package import (
    BASE_MODEL,
    GridPackageArtifacts,
    PackageArtifacts,
    write_grid_package,
    write_package,
)

__all__ = ["OUTPUT_DIRNAME", "run_vault"]

#: Ego_Package 根目录名，落于 ``ego train`` 执行时所在的当前工作目录下（R10-AC1）。
OUTPUT_DIRNAME = "ego-output"


def run_vault(
    lora_source: str | Path | None = None,
    *,
    cells: list[TrainedCell] | None = None,
    lora_rank: int,
    lora_alpha: int,
    training_backend: str,
    base_model: str = BASE_MODEL,
    cwd: Path | None = None,
    output_root: Path | None = None,
    version: str | None = None,
    created_at: str | None = None,
    stdout: TextIO | None = None,
) -> PackageArtifacts | GridPackageArtifacts:
    """把 Trainer 的临时产出持久化为 Ego_Package（需求 10.1、10.5、10.6、10.7、R4-AC7、R6-AC1）。

    本函数支持两条互斥的写入路径，由 ``cells`` / ``lora_source`` 决定（设计「决策 C」）：

    * **退化扁平路径（Flat_Package）**：``cells is None``，或 ``cells`` 仅含单个
      Kernel Cell（``zone=None, level=0`` 且无任何分区 Cell）——复用既有
      :func:`write_package`，Kernel 适配器落 ``normal/adapters/ego-lora.safetensors``、
      **不**写 ``grid`` 键，产物与 ego-pod-mvp / ego-pod-real-trainer **逐字节一致**
      （R4-AC7、R6-AC1/AC2）。扁平路径的单个 ``lora_source``：``cells`` 为单 Kernel
      时取该 Kernel 的 ``output_path``；否则取直接传入的 ``lora_source``（今天的
      调用方）。
    * **网格路径（Grid_Package）**：``cells`` 含至少一个分区 Cell——调用
      :func:`write_grid_package` 事务性写出 ``kernel.safetensors`` +
      ``<zone>/L<level>.safetensors`` + ``manifest.grid``。

    两条路径共享相同的可观测事件：先打印 directory-created 事件、写产物后校验产物
    存在、再打印 Ego_Package 根目录绝对路径（package-created 事件）。

    步骤（见模块文档）：创建/识别 ``ego-output/`` 并打印 directory-created 事件 →
    据 ``cells`` 选择 :func:`write_package`（扁平）或 :func:`write_grid_package`
    （网格）事务性写产物 → 校验产物存在 → 打印 Ego_Package 根目录绝对路径
    （package-created 事件）→ 返回产物路径集合。

    :param lora_source: 扁平路径下 Trainer 导出的 ``{cwd}/.ego/lora.safetensors``
        路径（:class:`~ego_pod.trainer.pipeline.TrainingResult` 的 ``output_path``）；
        将被移动到 ``normal/adapters/ego-lora.safetensors``。当走网格路径或以单
        Kernel ``cells`` 走扁平路径时可为 ``None``。
    :param cells: 网格路径下 Trainer 产出的已训练 Cell 列表
        （:class:`~ego_pod.trainer.cells.GridTrainingResult` 的 ``cells``）；``None``
        或仅含单个 Kernel Cell ⇒ 走退化扁平路径。
    :param lora_rank: 写入 ``normal/config.json`` 的 LoRA 秩（正整数，来自 Ego_Config）。
    :param lora_alpha: 写入 ``normal/config.json`` 的 LoRA alpha（正整数，来自 Ego_Config）。
    :param training_backend: 写入 ``manifest.json`` 的后端标识（Stub_Backend 为
        ``"stub"``，来自 ``TrainingResult.backend_identifier`` /
        ``GridTrainingResult.backend_identifier``），供 ``ego run`` 识别 stub
        模式（需求 12.9）。
    :param base_model: 基座模型标识，默认 :data:`~ego_pod.vault.package.BASE_MODEL`。
    :param cwd: 进程工作目录，默认 :meth:`pathlib.Path.cwd`；Ego_Package 根目录为
        ``cwd/ego-output``（R10-AC1）。
    :param output_root: 显式指定的 Ego_Package 根目录；默认 ``{cwd}/ego-output``。
        允许注入以便测试。
    :param version: 写入 ``manifest.json`` 的语义化版本号；为 ``None`` 时取包的
        ``__version__``。
    :param created_at: 写入 ``manifest.json`` 的 UTC ISO 8601 字符串；为 ``None``
        时取当前时间。
    :param stdout: directory-created 与 package-created 事件的目标输出流，默认
        ``sys.stdout``。
    :returns: 写入产物的关键绝对路径（扁平路径返回 :class:`PackageArtifacts`，
        网格路径返回 :class:`GridPackageArtifacts`）。
    :raises VaultError: 扁平路径未提供有效 ``lora_source``；写入过程中遭遇 I/O 错误
        （R10-AC7 / R4-AC10，由 ``write_package`` / ``write_grid_package`` 抛出，
        已清理 ``.tmp/`` 半成品并保留旧产物）；或产物存在性校验意外失败。
    """

    target = sys.stdout if stdout is None else stdout

    base_cwd = Path.cwd() if cwd is None else Path(cwd)
    root = base_cwd / OUTPUT_DIRNAME if output_root is None else Path(output_root)

    # --- 1. 创建/识别 ego-output/，打印 directory-created 事件（R10-AC1）--------
    #     先探测既存性以区分「已创建」与「已存在」；mkdir 不会删除任何既有产物，
    #     旧 normal/ 的删除/替换只发生在 write_package 内部、且在新权重就位之后。
    already_existed = root.exists()
    root.mkdir(parents=True, exist_ok=True)
    root_abs = root.resolve()
    status = "already exists" if already_existed else "created"
    print(f"[VAULT] ego-output directory {status}: {root_abs}", file=target, flush=True)

    # --- 2+3. 等待 Trainer 输出并事务性写产物 ---------------------------------
    #     据 cells 选择扁平或网格写入路径（决策 C）。两条写入函数都先校验源权重
    #     存在、再写 .tmp/ 暂存区并原子替换旧产物；I/O 错误时其已清理 .tmp/
    #     半成品、保留旧 normal/，并抛 VaultError（R10-AC7 / R4-AC10）。本流水线
    #     不拦截该异常，原样上抛交由 CLI 顶层映射为退出码 1。
    if cells is None or _is_single_kernel(cells):
        # 退化扁平路径：与 MVP 产物逐字节一致（R4-AC7、R6-AC1/AC2）。
        flat_source = _flat_lora_source(lora_source, cells)
        artifacts: PackageArtifacts | GridPackageArtifacts = write_package(
            root,
            flat_source,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            training_backend=training_backend,
            base_model=base_model,
            version=version,
            created_at=created_at,
        )
    else:
        # 网格路径：写 kernel.safetensors + <zone>/L<level>.safetensors + manifest.grid。
        artifacts = write_grid_package(
            root,
            cells=cells,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            training_backend=training_backend,
            base_model=base_model,
            version=version,
            created_at=created_at,
        )

    # --- 4. 校验产物均已存在（R10-AC6 / R4-AC8）-------------------------------
    if isinstance(artifacts, GridPackageArtifacts):
        _verify_grid_artifacts(artifacts)
    else:
        _verify_artifacts(artifacts)

    # --- 5. 打印 Ego_Package 根目录绝对路径，作为 package-created 事件（R10-AC6）-
    #     单独成行的绝对路径（以 / 开头），与第 1 步的 directory 事件相互区分，
    #     共同满足 R10-AC1 的「两个独立可观测产出」。
    print(str(artifacts.root), file=target, flush=True)

    # --- 6. 成功返回；退出码 0 由 CLI 顶层映射（R10-AC6）------------------------
    return artifacts


def _is_single_kernel(cells: list[TrainedCell]) -> bool:
    """判定 ``cells`` 是否退化为「仅含单个 Kernel Cell」（决策 C）。

    Whether ``cells`` degenerates to a single Kernel Cell (and thus a Flat_Package).

    退化条件：``cells`` 恰含一个元素，且该元素为 Kernel（``zone is None`` 且
    ``level == 0``，即无任何分区 Cell）。满足时 Vault 走纯扁平路径，产物与既有
    Flat_Package 逐字节一致（R4-AC7、R6-AC1/AC2）。空列表（无任何 Cell）不视为
    单 Kernel——交由网格路径的 kernel-always 硬约束以 :class:`VaultError` 快速失败。

    :param cells: 已训练 Cell 列表。
    :returns: ``cells`` 恰为单个 Kernel Cell 时返回 ``True``，否则 ``False``。
    """

    return (
        len(cells) == 1 and cells[0].zone is None and cells[0].level == 0
    )


def _flat_lora_source(
    lora_source: str | Path | None,
    cells: list[TrainedCell] | None,
) -> str | Path:
    """解析扁平路径下的单个 LoRA 源权重路径（决策 C）。

    Resolve the single LoRA source path for the degenerate Flat_Package path.

    * ``cells`` 为单 Kernel Cell ⇒ 取该 Kernel 的 ``output_path``（网格训练退化为
      单适配器，R3-AC9）。
    * 否则（``cells is None``，今天的单适配器调用方）⇒ 取直接传入的 ``lora_source``。

    :param lora_source: 直接传入的 LoRA 权重路径（``cells is None`` 时使用）。
    :param cells: 已训练 Cell 列表（单 Kernel 时使用其 ``output_path``）。
    :returns: 扁平路径写入所用的单个 LoRA 源权重路径。
    :raises VaultError: 既未提供 ``lora_source`` 又无可用单 Kernel Cell 权重。
    """

    if cells is not None and _is_single_kernel(cells):
        return cells[0].output_path
    if lora_source is None:
        raise VaultError(
            "扁平写入路径需要单个 LoRA 源权重：未提供 lora_source，"
            "且 cells 不含可用的单 Kernel Cell"
        )
    return lora_source


def _verify_artifacts(artifacts: PackageArtifacts) -> None:
    """校验 Ego_Package 三个产物文件均已存在（R10-AC6）。

    正常路径下 :func:`write_package` 的原子写入已保证三文件就位，本校验为防御性
    复核，确保 R10-AC6「校验三文件均已存在」这一可观测约束被显式满足。任一缺失
    则以 :class:`VaultError` 抛出（退出码 1）。

    :param artifacts: :func:`write_package` 返回的产物路径集合。
    :raises VaultError: 三文件中任一不存在。
    """

    required = (
        artifacts.manifest_path,
        artifacts.config_path,
        artifacts.adapter_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise VaultError(
            "Ego_Package 产物校验失败，以下文件缺失：" + "；".join(missing)
        )


def _verify_grid_artifacts(artifacts: GridPackageArtifacts) -> None:
    """校验 Grid_Package 的 manifest、config 与全部 Cell 适配器文件均已存在（R4-AC8）。

    Verify that the Grid_Package's manifest, config and every Cell adapter file
    listed in :class:`~ego_pod.vault.package.GridPackageArtifacts` exist.

    与扁平路径的 :func:`_verify_artifacts` 对应：网格路径下产物含 ``manifest.json``、
    ``normal/config.json`` 与 grid 所列每个 Cell 的适配器文件（Kernel 与各分区/层级）。
    :func:`~ego_pod.vault.package.write_grid_package` 的原子写入已保证这些文件就位，
    本校验为防御性复核，确保 R4-AC8「成功写入后 manifest+config+全部 Cell 文件同时
    存在」这一可观测约束被显式满足。任一缺失则以 :class:`VaultError` 抛出（退出码 1）。

    :param artifacts: :func:`write_grid_package` 返回的产物路径集合。
    :raises VaultError: manifest / config / 任一 Cell 适配器文件不存在。
    """

    required = [artifacts.manifest_path, artifacts.config_path]
    required.extend(artifacts.cell_paths.values())
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise VaultError(
            "Grid_Package 产物校验失败，以下文件缺失：" + "；".join(missing)
        )
