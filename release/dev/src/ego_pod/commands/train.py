"""``ego train`` 实现：采集 → 构建 → 训练 → 产出 端到端编排。

本模块实现 Requirement 11（``ego train`` 端到端编排）的核心验收标准：

* **R11-AC4**：加载 YAML 配置作为 Ego_Config——若用户传入 ``--config <path>``
  则使用该路径，否则使用当前工作目录下的 ``ego-config.yaml``；无论路径来源，
  均须在启动 Collector 之前完成 Ego_Config 的成功加载与校验，且仅在两者均成功
  时方可进入流水线。
* **R11-AC6**：配置加载或校验失败时——不启动 Collector 及之后任何阶段、在标准
  错误输出包含失败原因与**所用配置路径**（实际尝试加载的路径）的信息、并以退出码
  2 结束。本模块通过捕获 :class:`~ego_pod.exit_codes.ConfigError` 并确保其消息含
  配置路径后原样上抛，由 CLI 顶层（任务 4.2）打印到标准错误并以
  ``ConfigError.exit_code``（2）结束。
* **R11-AC1**：校验通过后，按 Collector → Compiler → Trainer → Vault 的固定顺序
  依次调用四个模块，且仅在前一阶段成功结束后才启动下一阶段。
* **R11-AC2**：任意阶段失败（抛出携带退出码的 :class:`~ego_pod.exit_codes.EgoError`
  子类）时立即终止后续未启动的阶段，并以失败阶段的退出码结束——本模块**不**拦截
  各阶段异常，使其原样传播至 CLI 顶层映射为对应退出码（采集/构建/训练/产出失败
  均为 1）。
* **R11-AC3**：四个阶段全部成功（无异常）时返回 :data:`~ego_pod.exit_codes.EXIT_OK`
  （0）。

设计依据（design.md §「`ego train` 数据流」与 §Components and Interfaces）：

* 入口 :func:`run_train` 接收 argparse 命名空间（使用 ``args.config``）并返回
  ``int`` 退出码；成功返回 0，失败一律以异常上抛交由 CLI 顶层统一处理。
* **加载与校验的职责拆分**：:func:`ego_pod.config.loader.load` 负责文件存在性
  / 读权限 / I/O / YAML 解析（R3-AC2/AC3/AC4）并返回**宽松构造**的
  :class:`~ego_pod.config.schema.EgoConfig`；
  :func:`ego_pod.config.validators.validate` 则在**原始映射**上执行结构与取值
  校验（R3-AC5..AC9、R8-AC9）并在全部通过时打印 ``Config validated: <绝对路径>``
  （R3-AC10）。二者按 AC 编号升序串接（loader 的 3/4 先于 validator 的 5–9），
  满足 R3-AC11 的短路顺序。
* **阶段间数据流**（design.md §「`ego train` 数据流」）：
  Collector 产出的 :class:`~ego_pod.collector.pipeline.CollectionResult` 的 ``files``
  → Compiler 的语料构建；Compiler 产出的
  :class:`~ego_pod.compiler.pipeline.CompilationResult` 的 ``corpus_path`` 连同
  ``config.training`` 五参数 → Trainer；Trainer 产出的
  :class:`~ego_pod.trainer.pipeline.TrainingResult` 的 ``output_path`` /
  ``backend_name`` 连同 ``config.training`` 的 ``lora_rank`` / ``lora_alpha`` 与
  ``normal`` Edition 的 ``base`` → Vault 持久化为 Ego_Package。
* **离线模式**由 CLI 顶层（``cli.main`` 调用 ``privacy.init_offline_mode``）在分发
  前装配，Trainer 经 ``get_active_offline_mode`` 自行感知，故本模块无需介入。

阶段进入/离开的 START/END 时间戳日志（R11-AC5）在各阶段调用点前后通过
:func:`ego_pod.logging_io.stage_event` 打印：进入阶段前打印 ``START``，阶段成功
返回后打印 ``END``。由于阶段失败时异常向上传播（不会执行其后的 ``END`` 调用），
失败阶段不打印 ``END``，符合设计预期。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import yaml

from ego_pod.collector.pipeline import run as run_collection
from ego_pod.compiler.pipeline import run_compilation
from ego_pod.config.loader import load as load_config
from ego_pod.config.schema import EgoConfig
from ego_pod.config.validators import validate as validate_config
from ego_pod.exit_codes import EXIT_OK, ConfigError
from ego_pod.logging_io import Event, Stage, stage_event
from ego_pod.trainer.cells import CellSpec
from ego_pod.trainer.grid import run_grid_training
from ego_pod.trainer.pipeline import run_training
from ego_pod.vault.pipeline import run_vault

__all__ = ["run_train", "DEFAULT_CONFIG_FILENAME"]

#: 默认配置文件名。``ego train`` 未提供 ``--config`` 时回退到当前工作目录下的该文件
#: （R11-AC4）。与 ``cli.py`` / ``commands.init`` 中的同名常量保持一致；此处本地定义
#: 以避免对 ``cli`` 的循环导入（``cli`` 惰性导入本模块）。
DEFAULT_CONFIG_FILENAME = "ego-config.yaml"

#: MVP 唯一支持的 Edition 名称（校验通过后必然存在于 ``config.editions``）。
_NORMAL_EDITION = "normal"

#: Compiler 写出各 Cell_Sub_Dataset 的子目录名（相对 ``.ego/``）。各 Cell 的
#: :class:`~ego_pod.compiler.routing.CellSubDataset` 的 ``relpath`` 即相对此目录。
_CELLS_DIRNAME = "cells"


def run_train(args: argparse.Namespace) -> int:
    """``ego train`` 主入口：加载并校验配置，随后顺序执行四阶段流水线。

    流程：

    1. 解析所用配置路径：``args.config`` 非空时取之，否则取
       ``{cwd}/ego-config.yaml``（R11-AC4）。
    2. 加载并校验 Ego_Config（:func:`_load_and_validate_config`）。任一失败抛
       :class:`~ego_pod.exit_codes.ConfigError`（含失败原因与所用配置路径，退出码
       2），且**不**进入流水线（R11-AC6）。
    3. 校验通过后按 Collector → Compiler → Trainer → Vault 顺序执行
       （:func:`_run_pipeline`）；前一阶段成功（无异常）才进入下一阶段（R11-AC1），
       任一阶段抛出的 :class:`~ego_pod.exit_codes.EgoError` 子类原样传播（R11-AC2）。
    4. 四阶段全部成功后返回 :data:`~ego_pod.exit_codes.EXIT_OK`（R11-AC3）。

    :param args: argparse 解析得到的命名空间，使用其 ``config`` 字段
        （``str`` 路径或 ``None``）。
    :returns: 成功时为 :data:`~ego_pod.exit_codes.EXIT_OK`（0）。
    :raises ConfigError: 配置加载或校验失败（退出码 2，R11-AC6）。
    :raises EgoError: 任一流水线阶段失败（退出码由失败阶段决定，R11-AC2）。
    """

    # R11-AC4：确定实际使用的配置路径（--config 优先，否则默认 cwd/ego-config.yaml）。
    config_arg = getattr(args, "config", None)
    config_path = config_arg if config_arg else str(Path.cwd() / DEFAULT_CONFIG_FILENAME)

    # R11-AC4 / R11-AC6：成功加载并校验是进入流水线的必要前置条件；失败即抛
    # ConfigError（退出码 2）且不启动 Collector。
    config = _load_and_validate_config(config_path)

    # R11-AC1/AC2/AC3：校验通过后顺序执行四阶段流水线。
    return _run_pipeline(config)


# ---------------------------------------------------------------------------
# 配置加载与校验（R11-AC4 / R11-AC6）
# ---------------------------------------------------------------------------


def _load_and_validate_config(config_path: str) -> EgoConfig:
    """加载并校验 Ego_Config，返回校验通过的 :class:`EgoConfig`。

    ``loader.load`` 处理文件存在性 / 读权限 / I/O / YAML 解析错误
    （R3-AC2/AC3/AC4）并返回宽松构造的 :class:`EgoConfig` 与绝对路径；随后在
    **原始映射**上调用 ``validators.validate`` 执行结构与取值校验
    （R3-AC5..AC9、R8-AC9），全部通过时由 ``validate`` 打印 ``Config validated:``
    确认信息（R3-AC10）。

    为满足 R11-AC6「错误信息须包含失败原因与所用配置路径」，本函数捕获加载或校验
    抛出的 :class:`ConfigError`，在其消息尚未包含配置路径时补入实际尝试加载的绝对
    路径，再原样上抛——由 CLI 顶层打印到标准错误并以退出码 2 结束。

    :param config_path: 实际使用的配置文件路径（来自 ``--config`` 或默认路径）。
    :returns: 校验通过的 :class:`EgoConfig`。
    :raises ConfigError: 加载或校验失败（退出码 2）。
    """

    try:
        # R3-AC2/AC3/AC4：加载并解析 YAML，得到宽松构造的 EgoConfig 与绝对路径。
        config, abs_path = load_config(config_path)

        # validators.validate 作用于原始映射而非 EgoConfig（dataclass 无法表达
        # 顶层字段缺失，R3-AC5 需要据此报错）。loader.load 已成功解析该文件，故此处
        # 重新读取并解析以得到原始映射；任何残留 I/O 异常被归一化为 ConfigError。
        raw_mapping = _read_raw_mapping(abs_path)

        # R3-AC5..AC9 / R8-AC9：结构与取值校验；全部通过时打印成功确认（R3-AC10）。
        validate_config(raw_mapping, abs_path)
    except ConfigError as exc:
        # R11-AC6：确保错误信息同时包含失败原因与所用配置路径（实际尝试加载的路径）。
        raise _ensure_path_in_message(exc, config_path) from exc

    return config


def _read_raw_mapping(abs_path: str) -> Any:
    """重新读取并解析 ``abs_path`` 处的 YAML，返回原始映射供校验使用。

    ``loader.load`` 在本函数被调用前已成功完成同一文件的存在性检查、读取与 YAML
    解析，因此此处正常不会再触发解析错误；为防御加载与本次读取之间的竞态（文件被
    删除 / 替换等），仍把 :class:`OSError` 与 :class:`yaml.YAMLError` 归一化为
    :class:`ConfigError`，以保证配置类失败统一映射到退出码 2。

    :param abs_path: ``loader.load`` 返回的配置文件绝对路径。
    :returns: PyYAML 解析得到的原始对象（正常为映射）。
    :raises ConfigError: 重新读取或解析失败（退出码 2）。
    """

    try:
        raw_bytes = Path(abs_path).read_bytes()
        return yaml.safe_load(raw_bytes)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(
            f"读取配置文件 {abs_path} 时发生错误：{exc}"
        ) from exc


def _ensure_path_in_message(exc: ConfigError, config_path: str) -> ConfigError:
    """返回一个其消息必然包含所用配置路径的 :class:`ConfigError`（R11-AC6）。

    部分加载/校验错误的原始消息已含路径（如 loader 的 I/O / 解析错误），部分则
    未含（如 R3-AC2 的「未找到 ego-config.yaml」、R3-AC5 的顶层缺失）。本函数据
    实际尝试加载的绝对路径补全消息：原消息已含该路径（绝对或原始形式）时原样返回，
    否则在其后追加「（所用配置路径：<绝对路径>）」。

    :param exc: 加载或校验阶段抛出的 :class:`ConfigError`。
    :param config_path: 实际使用的配置路径（``--config`` 或默认路径）。
    :returns: 消息含配置路径的 :class:`ConfigError`（退出码 2）。
    """

    abs_path = os.path.abspath(config_path)
    message = str(exc)
    if abs_path in message or config_path in message:
        return exc
    return ConfigError(f"{message}（所用配置路径：{abs_path}）")


# ---------------------------------------------------------------------------
# 四阶段流水线编排（R11-AC1 / R11-AC2 / R11-AC3）
# ---------------------------------------------------------------------------


def _run_pipeline(config: EgoConfig) -> int:
    """按 Collector → Compiler → Trainer → Vault 顺序执行流水线并返回退出码。

    各阶段以普通函数调用串接：阶段「成功」即正常返回、「失败」即抛出携带退出码的
    :class:`~ego_pod.exit_codes.EgoError` 子类。因此顺序调用天然实现 R11-AC1
    （前一阶段成功才进入下一阶段）与 R11-AC2（任一阶段失败立即终止、异常上抛由
    CLI 顶层映射为该阶段退出码）——本函数刻意**不**捕获各阶段异常。四阶段全部正常
    返回时以 :data:`~ego_pod.exit_codes.EXIT_OK` 结束（R11-AC3）。

    阶段间数据流见模块文档。各阶段的工作目录、中间产物目录（``.ego/``）与产出目录
    （``ego-output/``）均默认取 :meth:`pathlib.Path.cwd`，保持一致。

    .. note::
        R11-AC5 要求在进入/离开每个阶段时打印 ``[<ISO 8601>] <STAGE> START|END``
        日志。本函数在每个阶段调用前打印 ``START``、成功返回后打印 ``END``，复用
        :func:`ego_pod.logging_io.stage_event`。阶段失败时异常向上传播，其后的
        ``END`` 不会执行，故失败阶段只产生 ``START`` 而无 ``END``（符合设计预期）。

    :param config: 已校验通过的 :class:`EgoConfig`。
    :returns: :data:`~ego_pod.exit_codes.EXIT_OK`（0）。
    :raises EgoError: 任一阶段失败时由该阶段抛出（退出码由其 ``exit_code`` 决定）。
    """

    # --- 阶段 1：Collector（R4）------------------------------------------------
    stage_event(Stage.COLLECTOR, Event.START)
    collection = run_collection(config)
    stage_event(Stage.COLLECTOR, Event.END)

    # --- 阶段 2：Compiler（R5/R6/R7）------------------------------------------
    # 采集文件 → 语料构建，写出 {cwd}/.ego/corpus.jsonl。
    # 透传 Ego_Config 的 zones（Zone_Config）以驱动按 Cell 路由（R2-AC8）；缺省
    # （config.zones is None）⇒ Compiler 内部回退到默认基线（R1-AC8）。Trainer/Vault
    # 的网格分派为后续任务（4.5），本阶段仅产出各 Cell_Sub_Dataset 供其消费。
    stage_event(Stage.COMPILER, Event.START)
    compilation = run_compilation(collection.files, zone_config=config.zones)
    stage_event(Stage.COMPILER, Event.END)

    # --- 阶段 3+4：Trainer（R8/R3）→ Vault（R10/R4）---------------------------
    # 据 Compiler 是否把语料路由出分区 Cell 分流（设计 §3 接入 / §4.3、R3-AC9、
    # R4-AC7）：
    #   * 含分区 Cell ⇒ 网格路径：run_grid_training（按 Cell 多适配器）→
    #     run_vault(cells=...)（写 kernel + <zone>/L<level> + manifest.grid）。
    #   * 否则（仅 Kernel，或 Compiler 未产出 .cells）⇒ 单适配器路径：run_training →
    #     run_vault(lora_source=...)，与 ego-pod-mvp / real-trainer 逐字节一致
    #     （R6-AC1/AC2）。
    # 以 DEFAULT Zone_Config 基线，sense/credentials → Kernel、skill/action → coding、
    # thought → knowledge，典型语料含 coding/knowledge 分区 Cell ⇒ 走网格路径。
    normal_edition = config.editions[_NORMAL_EDITION]
    cell_specs = _build_cell_specs(compilation)
    use_grid = any(not spec.is_kernel for spec in cell_specs)

    if use_grid:
        # 网格路径（R3 / R4）：按 Cell 训练并写出二维网格产物。
        stage_event(Stage.TRAINER, Event.START)
        grid_training = run_grid_training(cell_specs, config.training)
        stage_event(Stage.TRAINER, Event.END)

        stage_event(Stage.VAULT, Event.START)
        run_vault(
            cells=list(grid_training.cells),
            lora_rank=config.training.lora_rank,
            lora_alpha=config.training.lora_alpha,
            training_backend=grid_training.backend_identifier,
            base_model=normal_edition.base,
        )
        stage_event(Stage.VAULT, Event.END)
    else:
        # 单适配器路径（MVP 兼容）：语料路径 + 训练参数 → 导出单个 LoRA 适配器。
        stage_event(Stage.TRAINER, Event.START)
        training = run_training(compilation.corpus_path, config.training)
        stage_event(Stage.TRAINER, Event.END)

        # LoRA 权重路径 + lora_rank/lora_alpha + Backend_Identifier + normal 基座
        # → Ego_Package。传入稳定后端标识（{unsloth, peft-cuda, mlx, peft-cpu,
        # stub}）而非可读后端名，使 manifest.json 的 training_backend 写入稳定标识，
        # 供 Runtime 据此选择匹配推理路径（R7-AC3）；Vault 写入逻辑不变。
        stage_event(Stage.VAULT, Event.START)
        run_vault(
            training.output_path,
            lora_rank=config.training.lora_rank,
            lora_alpha=config.training.lora_alpha,
            training_backend=training.backend_identifier,
            base_model=normal_edition.base,
        )
        stage_event(Stage.VAULT, Event.END)

    # R11-AC3：四阶段均成功 → 退出码 0。
    return EXIT_OK


def _build_cell_specs(compilation: Any) -> list[CellSpec]:
    """据 Compiler 结果构造按 Cell 训练规格列表（网格分流的依据，R3-AC9）。

    Build the list of per-Cell training specs from the Compiler result.

    任务 2.4 在 :class:`~ego_pod.compiler.pipeline.CompilationResult` 上新增 ``cells``
    字段（:class:`~ego_pod.compiler.routing.CellSubDataset` 列表，含每个非空 Cell 的
    ``cell``=:class:`~ego_pod.compiler.routing.CellKey` 与相对 ``.ego/cells/`` 的
    ``relpath``）。本函数把每个 Cell_Sub_Dataset 转换为一个
    :class:`~ego_pod.trainer.cells.CellSpec`，其 ``sub_dataset_path`` 解析为
    ``{cwd}/.ego/cells/<relpath>`` 的绝对路径（``cells`` 根目录由 ``corpus_path`` 的
    父目录 ``.ego/`` 推导，与 Compiler 写出布局一致）。

    经 :func:`getattr` 防御性读取 ``cells``：若 Compiler 结果尚无 ``cells`` 字段或为
    空 ⇒ 返回空列表，使 :func:`_run_pipeline` 回退到单适配器路径（MVP 行为不变）。

    :param compilation: :func:`run_compilation` 的结果。
    :returns: 待训练 :class:`CellSpec` 列表；无 ``cells`` 时为空列表。
    """

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
