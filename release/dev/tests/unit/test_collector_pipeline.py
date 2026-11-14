"""`ego_pod.collector.pipeline` 的单元测试。

覆盖设计文档 §Components and Interfaces / 3. Collector（编排层）与需求 R4 / R9：

* R4-AC1：按 sense → action → thought → skill → credentials 固定顺序处理；
* R4-AC3：路径不存在——且**仅**此情形——在标准输出打印跳过警告；目录存在
  但为空 / 仅含子目录 / 仅含被排除类型时**不**打印跳过警告；
* R4-AC4 / R4-AC8：根 / 子路径权限错误向上传播为 :class:`CollectorError`
  （退出码 1）；
* R4-AC6：完成后在标准输出打印固定五行采集摘要，缺失 / 为空的类别填 0；
* R4-AC7：五个 Data_Category 合计 0 个常规文件 → :class:`CollectorError`
  （退出码 1）；
* R9-AC2：非 ``local`` 数据源条目被跳过并在标准错误输出报错。

测试以 ``tmp_path`` 构建临时目录树，不依赖任何外部资源或网络。
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterator

import pytest

from ego_pod.collector.base import Category, CollectedFile
from ego_pod.collector.pipeline import CATEGORY_ORDER, CollectionResult, run
from ego_pod.config.schema import EditionDef, EgoConfig, SourceEntry, TrainingParams
from ego_pod.exit_codes import EXIT_RUNTIME_FAILURE, CollectorError

# 采集摘要单行格式，例如 "sense: 2 files, 11 bytes"
SUMMARY_RE = re.compile(
    r"^(sense|action|thought|skill|credentials): (\d+) files, (\d+) bytes$"
)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _make_config(
    paths: dict[str, Path],
    *,
    types: dict[str, str] | None = None,
) -> EgoConfig:
    """构造一份五类齐全的 Ego_Config，``sources`` 指向给定路径。

    Args:
        paths: Data_Category → 本地目录路径的映射，必须覆盖五个类别。
        types: 可选的 Data_Category → ``type`` 覆盖（默认全部 ``local``），
            用于构造非 ``local`` 条目以测试 R9-AC2。
    """

    types = types or {}
    sources = {
        cat: SourceEntry(type=types.get(cat, "local"), path=str(paths[cat]))
        for cat in CATEGORY_ORDER
    }
    return EgoConfig(
        sources=sources,
        training=TrainingParams(
            method="qlora",
            quantization="4bit",
            lora_rank=16,
            lora_alpha=32,
            learning_rate=2e-4,
            epochs=3,
            batch_size=4,
        ),
        editions={"normal": EditionDef(base="qwen3.5-0.8B")},
        output={},
    )


def _mkdir(base: Path, name: str) -> Path:
    """在 ``base`` 下创建并返回子目录 ``name``。"""

    d = base / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _summary_lines(stdout: str) -> list[str]:
    """从标准输出中过滤出采集摘要的五行（按出现顺序）。"""

    return [line for line in stdout.splitlines() if SUMMARY_RE.match(line)]


# ---------------------------------------------------------------------------
# R4-AC7：五个 Data_Category 整体无可用常规文件 → CollectorError（退出码 1）
# ---------------------------------------------------------------------------


def test_all_categories_nonexistent_raises_collector_error(
    tmp_path: Path, capsys
):
    # 五个类别全部指向不存在路径
    paths = {cat: tmp_path / f"missing_{cat}" for cat in CATEGORY_ORDER}
    config = _make_config(paths)

    with pytest.raises(CollectorError) as exc_info:
        run(config)

    # R4-AC7 映射到退出码 1
    assert exc_info.value.exit_code == EXIT_RUNTIME_FAILURE == 1

    # 即便整体为空，采集摘要五行仍应先行打印（缺失填 0）
    out = capsys.readouterr().out
    assert _summary_lines(out) == [
        "sense: 0 files, 0 bytes",
        "action: 0 files, 0 bytes",
        "thought: 0 files, 0 bytes",
        "skill: 0 files, 0 bytes",
        "credentials: 0 files, 0 bytes",
    ]


def test_all_categories_empty_existing_dirs_raises_without_skip_warning(
    tmp_path: Path, capsys
):
    # 五个类别全部指向存在但为空（或仅含子目录）的目录
    paths = {}
    for cat in CATEGORY_ORDER:
        d = _mkdir(tmp_path, cat)
        paths[cat] = d
    # thought 目录下仅含一个子目录（无常规文件）
    (paths["thought"] / "only_subdir").mkdir()

    config = _make_config(paths)

    with pytest.raises(CollectorError) as exc_info:
        run(config)

    assert exc_info.value.exit_code == 1

    out = capsys.readouterr().out
    # 整体性判定由 R4-AC7 处理：存在但为空的目录 **不** 触发跳过警告
    assert "路径不存在" not in out
    # 仍打印五行摘要且全为 0
    assert _summary_lines(out) == [
        f"{cat}: 0 files, 0 bytes" for cat in CATEGORY_ORDER
    ]


# ---------------------------------------------------------------------------
# R4-AC3：跳过警告仅在路径不存在时触发
# ---------------------------------------------------------------------------


def test_skip_warning_only_for_nonexistent_path(tmp_path: Path, capsys):
    sense_dir = _mkdir(tmp_path, "sense")
    (sense_dir / "note.txt").write_text("hi", encoding="utf-8")  # 避免整体为空

    missing_action = tmp_path / "missing_action"  # 不存在 → 应触发跳过警告
    empty_thought = _mkdir(tmp_path, "thought")  # 存在但为空 → 不应触发跳过警告
    empty_skill = _mkdir(tmp_path, "skill")
    empty_credentials = _mkdir(tmp_path, "credentials")

    paths = {
        "sense": sense_dir,
        "action": missing_action,
        "thought": empty_thought,
        "skill": empty_skill,
        "credentials": empty_credentials,
    }
    config = _make_config(paths)

    result = run(config)
    assert isinstance(result, CollectionResult)

    out = capsys.readouterr().out
    skip_lines = [line for line in out.splitlines() if "路径不存在" in line]

    # 恰好一条跳过警告，且只针对不存在的 action 路径
    assert len(skip_lines) == 1
    assert "action" in skip_lines[0]
    assert str(missing_action) in skip_lines[0]
    # 存在但为空的 thought / skill / credentials 不出现在跳过警告中
    assert "thought" not in skip_lines[0]


def test_no_skip_warning_when_dir_has_only_excluded_entries(
    tmp_path: Path, capsys
):
    # sense 有真实文件以避免整体为空
    sense_dir = _mkdir(tmp_path, "sense")
    (sense_dir / "a.txt").write_text("data", encoding="utf-8")

    # action 目录存在但仅含一个子目录（被排除的非常规项）
    action_dir = _mkdir(tmp_path, "action")
    (action_dir / "subdir").mkdir()

    paths = {"sense": sense_dir, "action": action_dir}
    for cat in ("thought", "skill", "credentials"):
        paths[cat] = _mkdir(tmp_path, cat)

    config = _make_config(paths)
    run(config)

    out = capsys.readouterr().out
    # 目录存在（即便无可用常规文件）→ 不触发任何跳过警告
    assert "路径不存在" not in out


# ---------------------------------------------------------------------------
# R4-AC6：采集摘要五行格式（固定顺序，缺失填 0）
# ---------------------------------------------------------------------------


def test_collection_summary_five_lines_fixed_order(tmp_path: Path, capsys):
    sense_dir = _mkdir(tmp_path, "sense")
    (sense_dir / "f1.txt").write_text("hello", encoding="utf-8")  # 5 bytes
    (sense_dir / "f2.txt").write_text("world!", encoding="utf-8")  # 6 bytes

    skill_dir = _mkdir(tmp_path, "skill")
    (skill_dir / "s.txt").write_text("data", encoding="utf-8")  # 4 bytes

    paths = {
        "sense": sense_dir,
        "action": tmp_path / "missing_action",  # 不存在 → 0,0
        "thought": _mkdir(tmp_path, "thought"),  # 空 → 0,0
        "skill": skill_dir,
        "credentials": _mkdir(tmp_path, "credentials"),  # 空 → 0,0
    }
    config = _make_config(paths)

    result = run(config)

    out = capsys.readouterr().out
    # 严格按 CATEGORY_ORDER 输出五行，缺失 / 为空填 0
    assert _summary_lines(out) == [
        "sense: 2 files, 11 bytes",
        "action: 0 files, 0 bytes",
        "thought: 0 files, 0 bytes",
        "skill: 1 files, 4 bytes",
        "credentials: 0 files, 0 bytes",
    ]

    # 返回结果的 summary 与打印一致
    assert result.summary["sense"] == (2, 11)
    assert result.summary["skill"] == (1, 4)
    assert result.summary["action"] == (0, 0)


def test_summary_always_has_all_five_categories(tmp_path: Path, capsys):
    # 仅 thought 有文件，其余四类不存在
    thought_dir = _mkdir(tmp_path, "thought")
    (thought_dir / "t.txt").write_text("x", encoding="utf-8")

    paths = {cat: tmp_path / f"missing_{cat}" for cat in CATEGORY_ORDER}
    paths["thought"] = thought_dir
    config = _make_config(paths)

    run(config)

    out = capsys.readouterr().out
    lines = _summary_lines(out)
    assert len(lines) == 5
    # 顺序固定且类别齐全
    assert [SUMMARY_RE.match(l).group(1) for l in lines] == list(CATEGORY_ORDER)


# ---------------------------------------------------------------------------
# R4-AC4 / R4-AC8：权限错误 → CollectorError（退出码 1）
# ---------------------------------------------------------------------------


class _PermissionDeniedAdapter:
    """在采集指定类别时抛出 :class:`PermissionError` 的伪适配器。

    用于在不依赖文件系统权限位的前提下，确定性地验证 Collector 编排层
    将权限错误映射为 :class:`CollectorError`（R4-AC4 根路径 / R4-AC8 子路径）。
    """

    def __init__(self, fail_category: str, filename: str, strerror: str):
        self._fail_category = fail_category
        self._filename = filename
        self._strerror = strerror

    def collect(
        self, entry: "SourceEntry", category: Category
    ) -> Iterator[CollectedFile]:
        if category == self._fail_category:
            raise PermissionError(13, self._strerror, self._filename)
        yield from ()


def test_permission_error_mapped_to_collector_error(tmp_path: Path):
    # 所有类别路径均存在（通过 root.exists() 检查后才会调用适配器）
    paths = {cat: _mkdir(tmp_path, cat) for cat in CATEGORY_ORDER}
    config = _make_config(paths)

    denied_path = str(paths["thought"] / "locked")
    adapter = _PermissionDeniedAdapter(
        fail_category="thought",
        filename=denied_path,
        strerror="Permission denied",
    )

    with pytest.raises(CollectorError) as exc_info:
        run(config, adapter=adapter)

    err = exc_info.value
    # R4-AC4 / R4-AC8 映射到退出码 1
    assert err.exit_code == 1
    # 错误信息包含类别名、出错绝对路径与权限原因
    message = str(err)
    assert "thought" in message
    assert os.path.abspath(denied_path) in message
    assert "Permission denied" in message


@pytest.mark.skipif(
    os.name == "nt", reason="Windows 目录权限位语义不同"
)
@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root 用户可绕过目录读权限",
)
def test_real_unreadable_root_dir_raises_collector_error(tmp_path: Path):
    # sense 有真实文件，确保失败来自权限而非整体为空
    sense_dir = _mkdir(tmp_path, "sense")
    (sense_dir / "ok.txt").write_text("ok", encoding="utf-8")

    locked = _mkdir(tmp_path, "action")
    (locked / "secret.txt").write_text("nope", encoding="utf-8")

    paths = {"sense": sense_dir, "action": locked}
    for cat in ("thought", "skill", "credentials"):
        paths[cat] = _mkdir(tmp_path, cat)
    config = _make_config(paths)

    os.chmod(locked, 0o000)
    try:
        with pytest.raises(CollectorError) as exc_info:
            run(config)
        assert exc_info.value.exit_code == 1
        assert "action" in str(exc_info.value)
    finally:
        # 恢复权限以便 tmp_path 清理
        os.chmod(locked, 0o755)


# ---------------------------------------------------------------------------
# R9-AC2：非 local 数据源条目被跳过并在 stderr 报错
# ---------------------------------------------------------------------------


def test_non_local_entry_skipped_with_stderr_error(tmp_path: Path, capsys):
    sense_dir = _mkdir(tmp_path, "sense")
    (sense_dir / "ok.txt").write_text("local-data", encoding="utf-8")  # 避免整体为空

    # credentials 指向一个含文件的存在目录，但 type 非 local → 应被跳过
    cloud_dir = _mkdir(tmp_path, "credentials")
    (cloud_dir / "should_not_be_collected.txt").write_text(
        "remote", encoding="utf-8"
    )

    paths = {
        "sense": sense_dir,
        "action": _mkdir(tmp_path, "action"),
        "thought": _mkdir(tmp_path, "thought"),
        "skill": _mkdir(tmp_path, "skill"),
        "credentials": cloud_dir,
    }
    config = _make_config(paths, types={"credentials": "cloud"})

    result = run(config)

    captured = capsys.readouterr()
    # stderr 指出违规条目名称与 "MVP 仅支持 local 数据源"
    assert "credentials" in captured.err
    assert "cloud" in captured.err
    assert "MVP 仅支持" in captured.err

    # 非 local 条目被跳过：其目录内文件未被采集
    collected_names = {os.path.basename(f.absolute_path) for f in result.files}
    assert "should_not_be_collected.txt" not in collected_names
    assert collected_names == {"ok.txt"}

    # 摘要中 credentials 计为 0
    assert result.summary["credentials"] == (0, 0)
    assert "credentials: 0 files, 0 bytes" in _summary_lines(captured.out)


def test_all_non_local_entries_trigger_empty_data_error(tmp_path: Path, capsys):
    # 五类全部为非 local 条目 → 全部跳过 → 整体无数据 → R4-AC7 退出码 1
    paths = {cat: _mkdir(tmp_path, cat) for cat in CATEGORY_ORDER}
    for cat in CATEGORY_ORDER:
        (paths[cat] / "data.txt").write_text("x", encoding="utf-8")
    types = {cat: "cloud" for cat in CATEGORY_ORDER}
    config = _make_config(paths, types=types)

    with pytest.raises(CollectorError) as exc_info:
        run(config)

    assert exc_info.value.exit_code == 1
    err = capsys.readouterr().err
    # 每个非 local 条目均在 stderr 报错
    for cat in CATEGORY_ORDER:
        assert cat in err
