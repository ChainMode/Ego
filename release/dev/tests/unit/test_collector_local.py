"""`ego_pod.collector.local.LocalDirectoryAdapter` 的单元测试。

覆盖设计文档 §Components and Interfaces / 3. Collector 与需求 R4：

* R4-AC2：递归读取常规文件，保留隐藏文件与 0 字节空文件，排除目录、
  符号链接、设备文件、套接字、FIFO 管道；
* R4-AC3：路径不存在返回空迭代器（跳过警告由编排层负责，本层不报错）；
* R4-AC5：每个常规文件记录绝对路径、字节大小、Data_Category 标签与
  ISO 8601 UTC 秒精度时间戳，0 字节文件同样记录全部字段。

测试以 ``tmp_path`` 构建临时目录树，不依赖任何外部资源或网络。
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path

import pytest

from ego_pod.collector.base import CollectedFile
from ego_pod.collector.local import LocalDirectoryAdapter
from ego_pod.config.schema import SourceEntry

# ISO 8601 UTC 秒精度（Z 后缀、无小数秒），例如 2025-11-25T08:00:00Z
READ_AT_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _entry(path: Path) -> SourceEntry:
    """构造一个指向 ``path`` 的本地数据源条目。"""

    return SourceEntry(type="local", path=str(path))


def _collect(path: Path, category: str = "thought") -> list[CollectedFile]:
    """运行适配器并物化结果列表，便于断言。"""

    adapter = LocalDirectoryAdapter()
    return list(adapter.collect(_entry(path), category))


def _basenames(files: list[CollectedFile]) -> set[str]:
    return {os.path.basename(f.absolute_path) for f in files}


# ---------------------------------------------------------------------------
# 基本采集与递归
# ---------------------------------------------------------------------------


def test_collects_regular_files(tmp_path: Path):
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "b.md").write_text("world", encoding="utf-8")

    files = _collect(tmp_path)

    assert _basenames(files) == {"a.txt", "b.md"}


def test_recursive_collection(tmp_path: Path):
    nested = tmp_path / "sub" / "deep"
    nested.mkdir(parents=True)
    (tmp_path / "top.txt").write_text("t", encoding="utf-8")
    (tmp_path / "sub" / "mid.txt").write_text("m", encoding="utf-8")
    (nested / "leaf.txt").write_text("l", encoding="utf-8")

    files = _collect(tmp_path)

    assert _basenames(files) == {"top.txt", "mid.txt", "leaf.txt"}


# ---------------------------------------------------------------------------
# R4-AC2：隐藏文件保留
# ---------------------------------------------------------------------------


def test_hidden_files_retained(tmp_path: Path):
    (tmp_path / ".hidden").write_text("secret-note", encoding="utf-8")
    (tmp_path / "visible.txt").write_text("v", encoding="utf-8")
    # 隐藏子目录下的隐藏文件也应被递归采集
    hidden_dir = tmp_path / ".config"
    hidden_dir.mkdir()
    (hidden_dir / ".env_note").write_text("x", encoding="utf-8")

    files = _collect(tmp_path)

    assert _basenames(files) == {".hidden", "visible.txt", ".env_note"}


# ---------------------------------------------------------------------------
# R4-AC2 / R4-AC5：0 字节文件保留且记录全部字段
# ---------------------------------------------------------------------------


def test_zero_byte_files_retained(tmp_path: Path):
    empty = tmp_path / "empty.txt"
    empty.touch()
    assert empty.stat().st_size == 0

    files = _collect(tmp_path)

    assert len(files) == 1
    collected = files[0]
    assert os.path.basename(collected.absolute_path) == "empty.txt"
    assert collected.size_bytes == 0
    # 0 字节文件同样记录全部元数据字段
    assert collected.category == "thought"
    assert READ_AT_RE.match(collected.read_at)


# ---------------------------------------------------------------------------
# R4-AC2：符号链接排除（文件与目录两种）
# ---------------------------------------------------------------------------


def test_file_symlink_excluded(tmp_path: Path):
    real = tmp_path / "real.txt"
    real.write_text("data", encoding="utf-8")
    link = tmp_path / "link.txt"
    link.symlink_to(real)

    files = _collect(tmp_path)

    # 仅采集到真实文件一次，符号链接被排除（不重复计数）
    assert _basenames(files) == {"real.txt"}
    assert all(not Path(f.absolute_path).is_symlink() for f in files)


def test_dir_symlink_not_followed(tmp_path: Path):
    # 目标目录置于扫描根之外，其内文件不应通过目录符号链接被采集
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("should-not-appear", encoding="utf-8")

    root = tmp_path / "root"
    root.mkdir()
    (root / "keep.txt").write_text("k", encoding="utf-8")
    (root / "link_dir").symlink_to(outside, target_is_directory=True)

    files = _collect(root)

    assert _basenames(files) == {"keep.txt"}
    assert "secret.txt" not in _basenames(files)


# ---------------------------------------------------------------------------
# R4-AC2：FIFO 管道排除（在不支持 os.mkfifo 的平台上跳过）
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not hasattr(os, "mkfifo"), reason="平台不支持 os.mkfifo（如 Windows）"
)
def test_fifo_excluded(tmp_path: Path):
    (tmp_path / "regular.txt").write_text("r", encoding="utf-8")
    fifo_path = tmp_path / "pipe.fifo"
    os.mkfifo(fifo_path)

    files = _collect(tmp_path)

    # FIFO 不是常规文件，应被排除；常规文件保留
    assert _basenames(files) == {"regular.txt"}
    assert "pipe.fifo" not in _basenames(files)


# ---------------------------------------------------------------------------
# R4-AC3：不存在路径返回空迭代器（不报错）
# ---------------------------------------------------------------------------


def test_nonexistent_path_returns_empty(tmp_path: Path):
    missing = tmp_path / "does_not_exist"
    assert not missing.exists()

    files = _collect(missing)

    assert files == []


def test_empty_directory_returns_empty(tmp_path: Path):
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()

    files = _collect(empty_dir)

    assert files == []


def test_directory_with_only_subdirs_returns_empty(tmp_path: Path):
    (tmp_path / "sub1").mkdir()
    (tmp_path / "sub2" / "sub3").mkdir(parents=True)

    files = _collect(tmp_path)

    assert files == []


# ---------------------------------------------------------------------------
# R4-AC5：CollectedFile 字段完整且正确
# ---------------------------------------------------------------------------


def test_collected_file_fields_populated(tmp_path: Path):
    content = "采集元数据字段验证"
    target = tmp_path / "note.txt"
    target.write_text(content, encoding="utf-8")
    expected_size = target.stat().st_size

    files = _collect(tmp_path, category="sense")

    assert len(files) == 1
    f = files[0]

    # absolute_path：绝对路径且指向正确文件
    assert os.path.isabs(f.absolute_path)
    assert Path(f.absolute_path) == target.resolve() or Path(
        f.absolute_path
    ) == target
    assert os.path.basename(f.absolute_path) == "note.txt"

    # size_bytes：与真实字节大小一致
    assert f.size_bytes == expected_size

    # category：透明传递调用方传入的标签
    assert f.category == "sense"

    # read_at：ISO 8601 UTC 秒精度，可被解析为带时区的 datetime
    assert READ_AT_RE.match(f.read_at)
    parsed = datetime.fromisoformat(f.read_at.replace("Z", "+00:00"))
    assert parsed.tzinfo is not None
    assert parsed.microsecond == 0


def test_category_label_propagated_to_all_files(tmp_path: Path):
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")

    files = _collect(tmp_path, category="credentials")

    assert len(files) == 2
    assert all(f.category == "credentials" for f in files)


def test_size_bytes_matches_content_length(tmp_path: Path):
    payload = b"0123456789"  # 10 bytes
    target = tmp_path / "ten.bin"
    target.write_bytes(payload)

    files = _collect(tmp_path)

    assert len(files) == 1
    assert files[0].size_bytes == 10


def test_relative_entry_path_yields_absolute_paths(tmp_path: Path, monkeypatch):
    # 以相对路径作为 SourceEntry.path，验证产出的 absolute_path 仍为绝对路径
    workdir = tmp_path / "work"
    data = workdir / "data"
    data.mkdir(parents=True)
    (data / "rel.txt").write_text("r", encoding="utf-8")
    monkeypatch.chdir(workdir)

    adapter = LocalDirectoryAdapter()
    files = list(adapter.collect(SourceEntry(type="local", path="data"), "skill"))

    assert len(files) == 1
    assert os.path.isabs(files[0].absolute_path)
    assert os.path.basename(files[0].absolute_path) == "rel.txt"
