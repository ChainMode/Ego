"""Compiler 编排与 JSONL 写入（design.md §4.6）。

本模块实现 :func:`run_compilation`：把 Compiler 的五个纯函数子阶段
（:mod:`~ego_pod.compiler.redact`、:mod:`~ego_pod.compiler.normalize`、
:mod:`~ego_pod.compiler.classify`、:mod:`~ego_pod.compiler.filter_dedup`、
:mod:`~ego_pod.compiler.augment`）按设计文档 §4 的固定顺序串接为一条完整的
语料构建流水线，并把结果写入 Conversation_Corpus 中间文件。

固定加工顺序（design.md §4 顶部流程图）::

    read → redact → normalize → classify → filter_dedup
         → (post_filter_count == 0 时按 R7-AC6 退出码 1)
         → augment → 写 {cwd}/.ego/corpus.jsonl

各阶段职责与对应需求：

* **read**：读取每个 :class:`~ego_pod.collector.base.CollectedFile` 的原文
  （UTF-8 解码，无法解码的字节以替换字符兜底，保证读取永不因编码失败而崩溃）。
* **redact**（R5）：对每个文件原文按其 Data_Category 执行凭证脱敏，汇总
  :class:`~ego_pod.compiler.redact.RedactCounters` 并在标准输出打印脱敏摘要
  （R5-AC5：总数 + 三类模式子项；R5-AC6：非 credentials 类别按模式细分）。
* **normalize**（R5-AC3/AC4）：移除控制字符、压缩连续空白。
* **classify**（R6）：把规范化后的原文按 Data_Category 套模板转换为
  instruction/output 候选样本，丢弃空字段/模板缺失/非法 category 的样本，
  并把丢弃明细输出到标准错误（R6-AC6/AC7/AC8）。
* **filter_dedup**（R7-AC1/AC2/AC3）：过短/非法过滤 + SHA-256 去重，在标准
  输出打印单行过滤摘要。
* **R7-AC6 短路**：当且仅当过滤后样本数 ``post_filter_count == 0`` 时，抛
  :class:`~ego_pod.exit_codes.CompilerError`（退出码 1，"可用语料为空"）；
  该判定**先于** augment 阶段执行（design.md §4.5 校验不变量）。
* **augment**（R7-AC4/AC5/AC7）：过滤后 ``0 < n < 16`` 时模板增强补足至 16；
  无法补足时由 :func:`~ego_pod.compiler.augment.augment_samples` 抛
  ``CompilerError``（R7-AC7）。

JSONL 写入约束（R6-AC4/AC5、design.md §4.6 与 "Conversation_Corpus 文件格式"）：

* 编码 UTF-8、**不含 BOM**；
* 行间分隔符为 LF（``\\n``）；
* 每行一个紧凑 JSON 对象，
  ``json.dumps(sample, ensure_ascii=False, separators=(",", ":"))``；
* 字段顺序固定为 ``instruction`` / ``input`` / ``output`` / ``category``，与
  :class:`~ego_pod.compiler.classify.SampleCandidate` 的 ``input_field`` 属性
  映射到 JSON 的 ``input`` 键。

隐私约束（R9-AC3 / R9-AC8）：所有写文件路径均以 ``{cwd}/.ego`` 为根——
Conversation_Corpus 中间文件只写入进程工作目录的子目录 ``.ego/``，不向任何
其它目标写出数据。

正确性不变量（design.md "Correctness Properties"）：

* **Property 11（corpus_jsonl 往返一致）** —— :func:`serialize_sample` 写出的
  每一行经 :func:`parse_corpus_line` 解析回 :class:`SampleCandidate` 后与原
  样本语义等价（四个字段值逐一相等；``augmented`` 标记可由 ``[AUG]`` 前缀
  无损恢复）。:func:`render_corpus` 与 :func:`parse_corpus` 提供整文件层面的
  对称读写。
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, TextIO, cast

from ego_pod.compiler.augment import AugmentStats, augment_samples
from ego_pod.compiler.classify import (
    Category,
    ClassifyStats,
    SampleCandidate,
    SourceSample,
    classify_samples,
)
from ego_pod.compiler.filter_dedup import (
    FilterStats,
    filter_and_dedup,
    print_filter_summary,
)
from ego_pod.compiler.normalize import normalize_text
from ego_pod.compiler.redact import (
    BASE64,
    PEM,
    SK,
    RedactCounters,
    redact_text,
)
from ego_pod.compiler.routing import (
    CellKey,
    CellSubDataset,
    resolve_zone_config,
    route_cells,
)
from ego_pod.exit_codes import CompilerError
from ego_pod.logging_io import print_summary

if TYPE_CHECKING:
    # 仅供类型检查使用，避免运行时硬依赖（与 collector 子模块保持一致风格）。
    from ego_pod.collector.base import CollectedFile
    from ego_pod.config.schema import ZoneConfig

__all__ = [
    "EGO_DIR_NAME",
    "CORPUS_FILENAME",
    "CELLS_DIR_NAME",
    "CompilationResult",
    "read_source_samples",
    "sample_to_dict",
    "parse_sample_dict",
    "serialize_sample",
    "parse_corpus_line",
    "render_corpus",
    "parse_corpus",
    "write_corpus",
    "run_compilation",
]

#: Compiler 中间产物目录名，位于进程工作目录下（R9-AC8）。
EGO_DIR_NAME = ".ego"

#: Conversation_Corpus 中间文件名（design.md §4.6）。
CORPUS_FILENAME = "corpus.jsonl"

#: 各 Cell_Sub_Dataset 的写出根目录名（相对 ``{cwd}/.ego/``，design.md §2.3）。
#: Kernel → ``cells/kernel.jsonl``；分区 Cell → ``cells/<zone>/L<level>.jsonl``。
CELLS_DIR_NAME = "cells"


@dataclass(frozen=True)
class CompilationResult:
    """:func:`run_compilation` 的结果，供 ``ego train`` 编排层与测试消费。

    :ivar corpus_path: 写出的 Conversation_Corpus 文件绝对/相对路径
        （默认 ``{cwd}/.ego/corpus.jsonl``）。
    :ivar samples: 最终写入语料的样本列表（含原始样本与增强样本）。
    :ivar redact_counters: 脱敏阶段的替换计数（R5-AC5/AC6）。
    :ivar classify_stats: 分类阶段的丢弃统计（R6-AC6/AC7/AC8）。
    :ivar filter_stats: 过滤/去重阶段的四项计数（R7-AC3）。
    :ivar augment_stats: 增强阶段的样本计数（R7-AC4）。
    :ivar cells: 按 Cell 路由产出的各**非空** Cell_Sub_Dataset（design.md §2.3，
        R2-AC4/AC5/AC8）。每个元素为 :class:`~ego_pod.compiler.routing.CellSubDataset`，
        携带其 :class:`~ego_pod.compiler.routing.CellKey`、被路由到该 Cell 的样本与
        相对 ``.ego/cells/`` 的 JSONL 写出路径，供 Trainer 网格路径（任务 3.x）消费。
        默认空列表，使既有（无网格语义的）构造与测试保持向后兼容。
    """

    corpus_path: Path
    samples: list[SampleCandidate]
    redact_counters: RedactCounters
    classify_stats: ClassifyStats
    filter_stats: FilterStats
    augment_stats: AugmentStats
    cells: list[CellSubDataset] = field(default_factory=list)


# ---------------------------------------------------------------------------
# read → redact → normalize
# ---------------------------------------------------------------------------


def _read_file_text(path: Path) -> str:
    """读取 ``path`` 的内容并以 UTF-8 解码为文本。

    采集阶段（Collector）已确认这些文件为可读的常规文件；此处以
    ``errors="replace"`` 兜底任何非 UTF-8 字节，保证读取永不因编码问题抛错。
    MVP 不解析各 Data_Category 的内部数据格式（由姐妹项目定义），故文件原文
    被整体视为一段待加工的非对话结构文本。
    """

    return path.read_bytes().decode("utf-8", errors="replace")


def read_source_samples(
    files: Iterable["CollectedFile"],
) -> tuple[list[SourceSample], RedactCounters]:
    """对每个采集文件执行 read → redact → normalize，产出分类阶段的来源样本。

    顺序遵循 design.md §4：先读取原文，再按文件所属 Data_Category 脱敏
    （汇总到统一的 :class:`RedactCounters`），最后规范化文本。每个文件产出一
    条非对话结构的 :class:`SourceSample`（``text`` 为规范化后的原文），其
    ``category`` 透传文件的 Data_Category 标签。

    :returns: ``(来源样本列表, 合并后的脱敏计数器)``，样本顺序与输入文件顺序一致。
    """

    samples: list[SourceSample] = []
    counters = RedactCounters()
    for collected in files:
        text = _read_file_text(Path(collected.absolute_path))
        redacted, file_counters = redact_text(text, collected.category)
        counters.merge(file_counters)
        normalized = normalize_text(redacted)
        samples.append(SourceSample(category=collected.category, text=normalized))
    return samples, counters


# ---------------------------------------------------------------------------
# JSONL 序列化 / 反序列化（Property 11 往返一致）
# ---------------------------------------------------------------------------


def sample_to_dict(sample: SampleCandidate) -> dict[str, str]:
    """把 :class:`SampleCandidate` 转换为可序列化的有序字典。

    字段顺序固定为 ``instruction`` / ``input`` / ``output`` / ``category``
    （design.md "Conversation_Corpus 文件格式"）；``input_field`` 属性映射到
    JSON 的 ``input`` 键。``augmented`` 标记不写入 JSON——它可由 ``instruction``
    的 ``[AUG]`` 前缀无损恢复（R7-AC5、Property 10）。
    """

    return {
        "instruction": sample.instruction,
        "input": sample.input_field,
        "output": sample.output,
        "category": sample.category,
    }


def parse_sample_dict(obj: Mapping[str, object]) -> SampleCandidate:
    """把一个已解析的 JSON 对象还原为 :class:`SampleCandidate`。

    与 :func:`sample_to_dict` 对称：读取 ``instruction`` / ``input`` /
    ``output`` / ``category`` 四个字段（缺失按空串处理），并据 ``instruction``
    是否以 ``[AUG]`` 前缀开头恢复 ``augmented`` 标记。供 Property 11 的往返
    一致性校验使用。
    """

    from ego_pod.compiler.augment import is_augmented_instruction

    instruction = str(obj.get("instruction", ""))
    return SampleCandidate(
        instruction=instruction,
        input_field=str(obj.get("input", "")),
        output=str(obj.get("output", "")),
        category=cast(Category, obj.get("category", "")),
        augmented=is_augmented_instruction(instruction),
    )


def serialize_sample(sample: SampleCandidate) -> str:
    """把单条样本序列化为一行紧凑 JSON（不含换行符）。

    使用 ``ensure_ascii=False`` 保留非 ASCII 字符原样（UTF-8 友好），
    ``separators=(",", ":")`` 去除多余空白以满足"紧凑单行"约束。
    """

    return json.dumps(
        sample_to_dict(sample), ensure_ascii=False, separators=(",", ":")
    )


def parse_corpus_line(line: str) -> SampleCandidate:
    """把 JSONL 中的一行解析回 :class:`SampleCandidate`（:func:`serialize_sample` 的逆）。"""

    return parse_sample_dict(json.loads(line))


def render_corpus(samples: Iterable[SampleCandidate]) -> str:
    """把样本序列渲染为完整的 JSONL 文本，行间以 LF（``\\n``）分隔。

    每个样本占一行（:func:`serialize_sample`），行之间以单个 ``\\n`` 连接，
    不追加尾随换行（"行间分隔符为 LF"，R6-AC4）。
    """

    return "\n".join(serialize_sample(sample) for sample in samples)


def parse_corpus(text: str) -> list[SampleCandidate]:
    """把 JSONL 文本解析回样本列表（:func:`render_corpus` 的逆）。

    忽略空行（如尾随空白行），其余每一非空行经 :func:`parse_corpus_line`
    还原为 :class:`SampleCandidate`。
    """

    return [
        parse_corpus_line(line)
        for line in text.split("\n")
        if line != ""
    ]


def write_corpus(samples: Iterable[SampleCandidate], corpus_path: Path) -> Path:
    """把样本写入 ``corpus_path``，UTF-8 无 BOM、行间 LF（R6-AC4/AC5）。

    必要时创建父目录 ``.ego/``。``encoding="utf-8"``（而非 ``utf-8-sig``）
    确保不写入 BOM；``newline="\\n"`` 关闭平台换行转换，保证行间分隔符严格为
    LF。返回写入的 ``corpus_path``。
    """

    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    content = render_corpus(samples)
    corpus_path.write_text(content, encoding="utf-8", newline="\n")
    return corpus_path


# ---------------------------------------------------------------------------
# 阶段摘要输出
# ---------------------------------------------------------------------------


def _print_redact_summary(
    counters: RedactCounters, *, stream: TextIO | None = None
) -> None:
    """在标准输出打印脱敏摘要（R5-AC5：总数 + 三类模式子项；R5-AC6：分类细分）。"""

    print_summary(
        "redact summary: "
        f"total={counters.total} "
        f"{SK}={counters.pattern_total(SK)} "
        f"{BASE64}={counters.pattern_total(BASE64)} "
        f"{PEM}={counters.pattern_total(PEM)}",
        stream=stream,
    )
    # R5-AC6：非 credentials 类别命中的 Credential_Token 按来源类别与模式细分。
    for category in sorted(counters.counts):
        if category == "credentials":
            continue
        bucket = counters.counts[category]
        detail = " ".join(
            f"{name}={bucket[name]}" for name in (SK, BASE64, PEM) if bucket.get(name)
        )
        if detail:
            print_summary(f"redact summary[{category}]: {detail}", stream=stream)


def _print_classify_discards(
    stats: ClassifyStats, *, stream: TextIO
) -> None:
    """把分类阶段的丢弃明细输出到标准错误（R6-AC6/AC7/AC8）。"""

    if stats.empty_field_discards:
        print(stats.empty_field_summary_line(), file=stream, flush=True)
    for message in stats.missing_template_messages():
        print(message, file=stream, flush=True)
    for message in stats.illegal_category_messages():
        print(message, file=stream, flush=True)


# ---------------------------------------------------------------------------
# 按 Cell 路由写出（design.md §2.3，R2-AC4/AC5/AC6）
# ---------------------------------------------------------------------------


def _cell_relpath(cell: CellKey) -> str:
    """返回某 Cell 相对 ``.ego/cells/`` 的 JSONL 写出相对路径（design.md §2.3）。

    Kernel → ``kernel.jsonl``；分区 Cell → ``<zone>/L<level>.jsonl``。

    ``Zone_Name`` 已被 ``validators`` 约束为 ``[a-z0-9-]+``（R1-AC3），可直接安全
    用作目录名，无需转义。``L<level>`` 前缀使层级在文件系统中自描述。
    """

    if cell.is_kernel:
        return "kernel.jsonl"
    return f"{cell.zone}/L{cell.level}.jsonl"


def _cell_identifier(cell: CellKey) -> str:
    """返回某 Cell 在网格摘要中的人类可读标识（R2-AC6）。

    Kernel → ``Kernel``；分区 Cell → ``(<zone>, L<level>)``。
    """

    if cell.is_kernel:
        return "Kernel"
    return f"({cell.zone}, L{cell.level})"


def _cell_sort_key(cell: CellKey) -> tuple[int, str, int]:
    """Cell 的稳定排序键：Kernel 优先，再按 Zone_Name 字母序、Level 升序。

    Kernel（``zone is None``）排最前；分区 Cell 按 ``(zone, level)`` 稳定排序，
    使写出与摘要的 Cell 顺序确定可复现。
    """

    if cell.is_kernel:
        return (0, "", 0)
    return (1, cell.zone or "", cell.level)


def _write_cell_sub_datasets(
    routed: Mapping[CellKey, list[SampleCandidate]],
    cells_root: Path,
) -> list[CellSubDataset]:
    """为每个非空 Cell 写出一个 Cell_Sub_Dataset 并返回其 :class:`CellSubDataset` 列表。

    复用既有 :func:`write_corpus`（经 :func:`render_corpus`/:func:`serialize_sample`），
    故各 Cell_Sub_Dataset 的编码与字段约定与 ego-pod-mvp Conversation_Corpus 完全
    一致（UTF-8 无 BOM、行间 LF、紧凑单行 JSON，字段 instruction/input/output/category，
    R2-AC5）。写出布局见 :func:`_cell_relpath`（Kernel → ``cells/kernel.jsonl``，
    分区 → ``cells/<zone>/L<level>.jsonl``，必要时 ``mkdir`` 父目录）。

    :param routed: ``route_cells`` 产出的 ``CellKey → 样本列表``（仅含非空 Cell）。
    :param cells_root: 各 Cell_Sub_Dataset 的写出根目录（``{cwd}/.ego/cells``）。
    :returns: 按稳定顺序（Kernel 优先，再分区字母序/Level 升序）排列的
        :class:`CellSubDataset` 列表，``relpath`` 为相对 ``cells_root`` 的相对路径。
    """

    cells: list[CellSubDataset] = []
    for cell in sorted(routed, key=_cell_sort_key):
        samples = routed[cell]
        relpath = _cell_relpath(cell)
        write_corpus(samples, cells_root / relpath)
        cells.append(CellSubDataset(cell=cell, samples=samples, relpath=relpath))
    return cells


def _grid_summary_line(cells: Iterable[CellSubDataset]) -> str:
    """构造网格摘要单行：每个非空 Cell 的标识与样本计数（R2-AC6）。

    形如 ``grid summary: Kernel=12 (coding, L1)=8 (knowledge, L1)=4``。
    """

    parts = " ".join(
        f"{_cell_identifier(c.cell)}={len(c.samples)}" for c in cells
    )
    return f"grid summary: {parts}" if parts else "grid summary:"


# ---------------------------------------------------------------------------
# 编排入口
# ---------------------------------------------------------------------------


def run_compilation(
    files: Iterable["CollectedFile"],
    *,
    zone_config: "ZoneConfig | None" = None,
    cwd: Path | None = None,
    output_path: Path | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> CompilationResult:
    """执行完整的 Compiler 流水线并写出 Conversation_Corpus，返回编排结果。

    步骤（见模块文档）：read → redact → normalize → classify → filter_dedup
    →（``post_filter_count == 0`` 时按 R7-AC6 抛 ``CompilerError``）→ augment
    →（按 Cell 路由 + 写各 Cell_Sub_Dataset + 网格摘要）→ 写
    ``{cwd}/.ego/corpus.jsonl``。各阶段摘要分别输出到标准输出/标准错误。

    **按 Cell 路由阶段（design.md §2.3，R2-AC4/AC5/AC6/AC8）**：在 ``augment``
    之后、写扁平语料之前，把最终样本经纯函数 :func:`~ego_pod.compiler.routing.route_cells`
    划分到恰好一个 Cell（划分性，R2-AC1），为每个非空 Cell 复用既有序列化器写出
    Cell_Sub_Dataset 到 ``{cwd}/.ego/cells/...``，并在标准输出打印一行网格摘要。
    路由发生在全部既有加工阶段之后，故不改变 ego-pod-mvp 既有语料加工语义（R2-AC8）。
    扁平 ``corpus.jsonl``（全体样本）仍照常写出，保证既有单 Cell / 退化路径与 MVP
    集成测试不受影响（向后兼容）。

    :param files: Collector 产出的 :class:`CollectedFile` 序列。
    :param zone_config: 可选的 Ego_Config ``zones``（Zone_Config）；``None`` ⇒
        经 :func:`~ego_pod.compiler.routing.resolve_zone_config` 回退到默认基线
        （R1-AC8）。
    :param cwd: 进程工作目录，默认 :meth:`pathlib.Path.cwd`；语料与各 Cell_Sub_Dataset
        均以 ``cwd/.ego`` 为根（隐私约束 R9-AC3/AC8）。
    :param output_path: 显式指定的语料文件路径；默认
        ``{cwd}/.ego/corpus.jsonl``。
    :param stdout: 脱敏/过滤/增强/网格摘要的目标输出流，默认 ``sys.stdout``。
    :param stderr: 分类丢弃明细的目标输出流，默认 ``sys.stderr``。
    :returns: :class:`CompilationResult`，含语料路径、最终样本、各阶段统计与各
        非空 Cell 的 :class:`~ego_pod.compiler.routing.CellSubDataset` 列表。
    :raises CompilerError: 过滤后样本数为 0（R7-AC6）；或 ``0 < n < 16`` 且
        应用全部增强模板后仍无法补足至 16（R7-AC7，由 ``augment_samples`` 抛出）。
    """

    base_cwd = Path.cwd() if cwd is None else Path(cwd)
    corpus_path = (
        base_cwd / EGO_DIR_NAME / CORPUS_FILENAME
        if output_path is None
        else Path(output_path)
    )
    err = sys.stderr if stderr is None else stderr

    # 1. read → redact → normalize（R5）。
    source_samples, redact_counters = read_source_samples(files)
    _print_redact_summary(redact_counters, stream=stdout)

    # 2. classify（R6）：套模板转换为 instruction/output 候选样本并丢弃非法样本。
    candidates, classify_stats = classify_samples(source_samples)
    _print_classify_discards(classify_stats, stream=err)

    # 3. filter_dedup（R7-AC1/AC2）+ 过滤摘要（R7-AC3）。
    filtered, filter_stats = filter_and_dedup(candidates)
    print_filter_summary(filter_stats, stream=stdout)

    # 4. R7-AC6：当且仅当过滤后样本数为 0 时，可用语料为空，退出码 1。
    #    该判定先于 augment（design.md §4.5：n == 0 不进入增强分支）。
    if filter_stats.total_after == 0:
        raise CompilerError(
            "[COMPILER] 可用语料为空：过滤后样本数为 0，无法继续训练"
        )

    # 5. augment（R7-AC4/AC5）：不足 16 条时模板增强；无法补足由 augment 抛 R7-AC7。
    final_samples, augment_stats = augment_samples(filtered)
    print_summary(augment_stats.summary_line(), stream=stdout)

    # 6. 按 Cell 路由（R2-AC1/AC8）：纯函数划分发生在全部既有加工阶段之后，不改变
    #    既有语料加工语义。为每个非空 Cell 写出 Cell_Sub_Dataset（复用既有序列化器，
    #    R2-AC4/AC5），并打印一行网格摘要（R2-AC6）。
    resolved = resolve_zone_config(zone_config)
    routed = route_cells(final_samples, resolved)
    cells = _write_cell_sub_datasets(
        routed, base_cwd / EGO_DIR_NAME / CELLS_DIR_NAME
    )
    print_summary(_grid_summary_line(cells), stream=stdout)

    # 7. 写扁平 JSONL 到 {cwd}/.ego/corpus.jsonl（R6-AC4/AC5、R9-AC3/AC8）。保留全体
    #    样本的扁平语料以维持向后兼容（既有单 Cell / 退化路径与 MVP 集成测试）。
    write_corpus(final_samples, corpus_path)

    return CompilationResult(
        corpus_path=corpus_path,
        samples=final_samples,
        redact_counters=redact_counters,
        classify_stats=classify_stats,
        filter_stats=filter_stats,
        augment_stats=augment_stats,
        cells=cells,
    )
