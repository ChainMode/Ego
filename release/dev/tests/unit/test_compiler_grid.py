"""Compiler 网格接入单元测试（tasks.md 任务 2.5）。

本文件为 Compiler 在既有语料加工之后新增的**按 Cell 路由阶段**
（design.md §2.3，:func:`ego_pod.compiler.pipeline.run_compilation`）提供
**无任何机器学习依赖**的示例/边界单元测试，覆盖以下验收标准：

* **R2-AC6（网格摘要行）**：``run_compilation`` 在标准输出打印**恰好一行**
  ``grid summary: ...``，按稳定顺序（Kernel 优先，再分区字母序 / Level 升序）
  列出每个**非空** Cell 的标识（``Kernel`` 或 ``(<zone>, L<level>)``）与样本计数。
* **R2-AC5 / R2-AC8（``CompilationResult.cells`` 内容）**：返回的 ``cells`` 为
  :class:`~ego_pod.compiler.routing.CellSubDataset` 列表，每个携带正确的
  :class:`~ego_pod.compiler.routing.CellKey` 与相对 ``.ego/cells/`` 的写出
  ``relpath``（Kernel → ``kernel.jsonl``，分区 → ``<zone>/L<level>.jsonl``）；
  各 Cell 样本计数之和等于 ``len(result.samples)``（划分性透传到管线层）。
* **R2-AC5（Cell_Sub_Dataset 编码/字段约定）**：各 Cell_Sub_Dataset 文件确实
  落于 ``{cwd}/.ego/cells/...`` 的预期路径，且经 :func:`parse_corpus` 解析回的
  样本与该 Cell 的 ``samples`` 逐字段相等——确认其 UTF-8 / LF / 紧凑单行 JSON
  约定与扁平 ``corpus.jsonl`` 完全一致。
* **向后兼容**：扁平 ``corpus.jsonl`` 仍照常写出且包含全部最终样本。
* **R2-AC2 / R2-AC3（默认基线路由）**：``zone_config=None`` 时按内置默认基线
  路由（skill/action → coding、thought → knowledge、sense/credentials → Kernel）。
* **R2-AC8（路由发生在加工之后）**：全部 Cell 样本（按对象身份）的并集恰好等于
  ``result.samples``（最终样本）——路由只在最终样本上做划分，不新增/不丢弃。
* **R2-AC8 / R2-AC9（路由随配置）**：自定义 :class:`ZoneConfig` 改变路由结果；
  声明但无类别命中的分区不产生 Cell。

构造手法（沿用 ``tests/unit/test_collector_pipeline.py`` 的 ``tmp_path`` 风格）：
在 ``tmp_path`` 下为各 Data_Category 写出真实文本文件，并据此构造
:class:`~ego_pod.collector.base.CollectedFile`，再以 ``tmp_path`` 为 ``cwd`` 调用
``run_compilation``。每个文件经 Compiler 套用对应类别模板转换为恰好一条候选样本；
合计 16 条互不相同的样本（``== TARGET_SAMPLE_COUNT``）⇒ **不触发增强**
（Property 9），使各 Cell 计数完全确定可断言。

_Requirements: 2.5, 2.6, 2.8_

**Feature: ego-layered-adapter**
"""

from __future__ import annotations

import io
from collections import Counter
from pathlib import Path

from ego_pod.collector.base import CollectedFile
from ego_pod.compiler.pipeline import (
    CELLS_DIR_NAME,
    CORPUS_FILENAME,
    EGO_DIR_NAME,
    CompilationResult,
    parse_corpus,
    run_compilation,
)
from ego_pod.compiler.routing import KERNEL, CellKey, CellSubDataset
from ego_pod.config.schema import ZoneConfig

# ---------------------------------------------------------------------------
# 输入构造辅助
# ---------------------------------------------------------------------------

#: 标准输入的每类文件数。默认基线下：
#:   Kernel = sense(4) + credentials(3) = 7
#:   coding = skill(3) + action(2)      = 5
#:   knowledge = thought(4)             = 4
#: 合计 16（== 增强目标），三个 Cell 计数互不相同，便于精确断言。
_COUNTS_PER_CATEGORY = {
    "sense": 4,
    "credentials": 3,
    "skill": 3,
    "action": 2,
    "thought": 4,
}


def _collected_file(
    tmp_path: Path, category: str, idx: int, text: str
) -> CollectedFile:
    """在 ``tmp_path/raw/<category>/`` 下写出一个文本文件并构造其 CollectedFile。

    文件原文 ``text`` 为不含凭证模式、不含控制字符的中文串，确保经
    redact/normalize/classify 后原样落入模板 output 并存活过 filter/dedup。
    """

    raw_dir = tmp_path / "raw" / category
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{category}_{idx}.txt"
    path.write_text(text, encoding="utf-8")
    return CollectedFile(
        absolute_path=str(path),
        size_bytes=path.stat().st_size,
        category=category,  # type: ignore[arg-type]
        read_at="2025-11-25T08:00:00Z",
    )


def _standard_inputs(tmp_path: Path) -> list[CollectedFile]:
    """构造覆盖五个 Data_Category、合计 16 条互不相同样本的 CollectedFile 列表。

    每个文件原文唯一（含类别名与序号），故跨类别/同类别均不会触发 SHA-256 去重；
    每条原文非空且模板 instruction 较长，故必然通过过短过滤。合计 16 条 ⇒ 不增强。
    """

    files: list[CollectedFile] = []
    for category, count in _COUNTS_PER_CATEGORY.items():
        for idx in range(count):
            text = f"{category}类别的唯一语料内容记录第{idx}条样本数据"
            files.append(_collected_file(tmp_path, category, idx, text))
    return files


def _compile(tmp_path: Path, *, zone_config: ZoneConfig | None = None):
    """以 ``tmp_path`` 为 ``cwd`` 运行 ``run_compilation`` 并捕获其 stdout。

    返回 ``(result, stdout_text)``。stdout 经注入的 :class:`io.StringIO` 捕获，
    使网格摘要行（R2-AC6）可被断言。
    """

    out = io.StringIO()
    result = run_compilation(
        _standard_inputs(tmp_path),
        zone_config=zone_config,
        cwd=tmp_path,
        stdout=out,
        stderr=io.StringIO(),
    )
    return result, out.getvalue()


def _grid_summary_lines(stdout: str) -> list[str]:
    """从 stdout 过滤出全部以 ``grid summary:`` 开头的行（按出现顺序）。"""

    return [
        line for line in stdout.splitlines() if line.startswith("grid summary:")
    ]


# ===========================================================================
# R2-AC6：网格摘要行格式（恰好一行，列出每个非空 Cell 的标识与计数）
# ===========================================================================


def test_grid_summary_line_single_and_format(tmp_path: Path) -> None:
    """``run_compilation`` 打印恰好一行 ``grid summary:``，格式与计数正确。

    默认基线下三个非空 Cell：Kernel=7、(coding, L1)=5、(knowledge, L1)=4；
    顺序为 Kernel 优先、再分区字母序（coding < knowledge）。

    _Requirements: 2.6_
    """

    _result, stdout = _compile(tmp_path)

    lines = _grid_summary_lines(stdout)
    # 恰好一行网格摘要（R2-AC6）。
    assert len(lines) == 1
    # 标识 + 计数，顺序为 Kernel → (coding, L1) → (knowledge, L1)。
    assert lines[0] == "grid summary: Kernel=7 (coding, L1)=5 (knowledge, L1)=4"


# ===========================================================================
# R2-AC5 / R2-AC8：CompilationResult.cells 的 CellKey、relpath 与计数守恒
# ===========================================================================


def test_result_cells_keys_relpaths_and_count_sum(tmp_path: Path) -> None:
    """``cells`` 为 CellSubDataset 列表，CellKey/relpath 正确，计数之和 == 样本数。

    _Requirements: 2.5, 2.8_
    """

    result, _stdout = _compile(tmp_path)
    assert isinstance(result, CompilationResult)

    # 全部元素为 CellSubDataset。
    assert all(isinstance(c, CellSubDataset) for c in result.cells)

    by_cell = {c.cell: c for c in result.cells}

    # 默认基线下恰好三个非空 Cell：Kernel、(coding, 1)、(knowledge, 1)。
    assert set(by_cell) == {
        KERNEL,
        CellKey(zone="coding", level=1),
        CellKey(zone="knowledge", level=1),
    }

    # 各 Cell 的相对写出路径（Kernel → kernel.jsonl；分区 → <zone>/L<level>.jsonl）。
    assert by_cell[KERNEL].relpath == "kernel.jsonl"
    assert by_cell[CellKey("coding", 1)].relpath == "coding/L1.jsonl"
    assert by_cell[CellKey("knowledge", 1)].relpath == "knowledge/L1.jsonl"

    # 每个 Cell 的样本计数（默认基线路由的预期分布）。
    assert len(by_cell[KERNEL].samples) == 7  # sense(4) + credentials(3)
    assert len(by_cell[CellKey("coding", 1)].samples) == 5  # skill(3) + action(2)
    assert len(by_cell[CellKey("knowledge", 1)].samples) == 4  # thought(4)

    # 划分性透传：各 Cell 样本计数之和 == 最终样本总数（R2-AC7 在管线层成立）。
    assert sum(len(c.samples) for c in result.cells) == len(result.samples)
    # 16 条原始样本 ⇒ 未触发增强。
    assert len(result.samples) == 16


# ===========================================================================
# R2-AC5：Cell_Sub_Dataset 文件落盘路径与 parse_corpus 往返一致
# ===========================================================================


def test_cell_sub_dataset_files_roundtrip_on_disk(tmp_path: Path) -> None:
    """各 Cell_Sub_Dataset 落于 .ego/cells/ 预期路径，且 parse_corpus 还原同样样本。

    复用 :func:`parse_corpus` 确认 Cell_Sub_Dataset 的 UTF-8 / LF / 紧凑单行 JSON
    与字段约定（instruction/input/output/category）与扁平 corpus 完全一致。

    _Requirements: 2.5_
    """

    result, _stdout = _compile(tmp_path)
    cells_root = tmp_path / EGO_DIR_NAME / CELLS_DIR_NAME

    expected_relpaths = {
        KERNEL: "kernel.jsonl",
        CellKey("coding", 1): "coding/L1.jsonl",
        CellKey("knowledge", 1): "knowledge/L1.jsonl",
    }

    for cell_sub in result.cells:
        file_path = cells_root / cell_sub.relpath
        # 文件确实落于预期相对路径下。
        assert cell_sub.relpath == expected_relpaths[cell_sub.cell]
        assert file_path.is_file()

        # 不写入 UTF-8 BOM（编码约定与扁平 corpus 一致）。
        raw = file_path.read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf")
        # 行间分隔符为 LF，不含 CRLF。
        assert b"\r\n" not in raw

        # parse_corpus 还原出与该 Cell samples 逐字段相等的样本列表。
        parsed = parse_corpus(file_path.read_text(encoding="utf-8"))
        assert parsed == cell_sub.samples


# ===========================================================================
# 向后兼容：扁平 corpus.jsonl 仍写出且包含全部最终样本
# ===========================================================================


def test_flat_corpus_still_written_with_all_samples(tmp_path: Path) -> None:
    """扁平 ``corpus.jsonl`` 仍照常写出，解析回的样本与 result.samples 逐一相等。

    _Requirements: 2.8_
    """

    result, _stdout = _compile(tmp_path)

    # 默认语料路径为 {cwd}/.ego/corpus.jsonl。
    assert result.corpus_path == tmp_path / EGO_DIR_NAME / CORPUS_FILENAME
    assert result.corpus_path.is_file()

    parsed = parse_corpus(result.corpus_path.read_text(encoding="utf-8"))
    # 扁平语料包含全部最终样本（顺序保持）。
    assert parsed == result.samples
    assert len(parsed) == 16


# ===========================================================================
# R2-AC2 / R2-AC3：默认基线路由 —— 断言哪些 Cell 出现
# ===========================================================================


def test_default_routing_expected_cells_appear(tmp_path: Path) -> None:
    """``zone_config=None`` 时按默认基线路由：skill/action→coding、thought→knowledge、
    sense/credentials→Kernel；不出现其它 Cell。

    _Requirements: 2.8_
    """

    result, _stdout = _compile(tmp_path)

    appearing = {c.cell for c in result.cells}
    assert appearing == {
        KERNEL,
        CellKey("coding", 1),
        CellKey("knowledge", 1),
    }
    # MVP 决策 F：被填充分区只产 Level 1，不产生 Level >= 2 的 Cell。
    assert all(c.cell.is_kernel or c.cell.level == 1 for c in result.cells)


# ===========================================================================
# R2-AC8：路由发生在既有加工之后 —— 全部 Cell 样本并集 == result.samples
# ===========================================================================


def test_routing_after_processing_union_equals_samples(tmp_path: Path) -> None:
    """各 Cell 样本（按对象身份）的并集恰好等于最终样本集合：既不丢弃也不复制。

    路由只在 augment 之后的最终样本上做纯函数划分（R2-AC8），故每条最终样本
    恰好出现在一个 Cell 中。

    _Requirements: 2.8_
    """

    result, _stdout = _compile(tmp_path)

    all_cell_samples = [s for cell in result.cells for s in cell.samples]

    # 计数守恒。
    assert len(all_cell_samples) == len(result.samples)
    # 按对象身份的多重集合相等：每条最终样本恰好出现在一个 Cell 中（无丢弃/无复制）。
    assert Counter(id(s) for s in all_cell_samples) == Counter(
        id(s) for s in result.samples
    )


# ===========================================================================
# R2-AC8 / R2-AC9：自定义 ZoneConfig 改变路由；未命中的声明分区不产生 Cell
# ===========================================================================


def test_custom_zone_config_changes_routing(tmp_path: Path) -> None:
    """自定义 ZoneConfig 使路由随配置变化；声明但无类别命中的分区不产生 Cell。

    配置：``zones=(dev, writing)``，``routing={skill:dev, action:dev}``。
    于是 skill/action → dev；thought（不在 routing）、sense、credentials → Kernel；
    ``writing`` 分区声明但无类别命中 ⇒ 不出现对应 Cell。

    _Requirements: 2.8_
    """

    zone_config = ZoneConfig(
        zones=("dev", "writing"),
        routing={"skill": "dev", "action": "dev"},
    )
    result, stdout = _compile(tmp_path, zone_config=zone_config)

    by_cell = {c.cell: c for c in result.cells}

    # 仅出现 Kernel 与 (dev, L1)；writing 声明但无样本 ⇒ 不出现。
    assert set(by_cell) == {KERNEL, CellKey("dev", 1)}

    # Kernel = sense(4) + credentials(3) + thought(4) = 11；dev = skill(3) + action(2) = 5。
    assert len(by_cell[KERNEL].samples) == 11
    assert len(by_cell[CellKey("dev", 1)].samples) == 5
    assert by_cell[CellKey("dev", 1)].relpath == "dev/L1.jsonl"

    # 计数守恒仍成立。
    assert sum(len(c.samples) for c in result.cells) == len(result.samples)

    # 网格摘要随之变化（恰好一行，列出 Kernel 与 (dev, L1)）。
    lines = _grid_summary_lines(stdout)
    assert lines == ["grid summary: Kernel=11 (dev, L1)=5"]
