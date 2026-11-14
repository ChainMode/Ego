"""`ego_pod.commands.init` 的单元测试（任务 4.5）。

本文件覆盖 Requirement 2（``ego init`` 初始化工作目录）的验收标准：

- **R2-AC1 / AC8**：在不含 ``ego-config.yaml`` 的目录成功创建文件（UTF-8），
  在标准输出打印生成文件的绝对路径并以退出码 0 结束。
- **R2-AC2..AC5**：生成文件字段完备性——顶层四项、五个 Data_Category 的
  ``local`` 条目、仅 ``normal`` edition（base=qwen3.5-0.8B）、QLoRA 五个数值参数。
- **R2-AC6**：目录已含 ``ego-config.yaml`` 且未提供 ``--force`` 时拒绝写入，
  抛 :class:`~ego_pod.exit_codes.EgoError`（退出码 1），原文件保持不变。
- **R2-AC7**：``--force`` 时用默认配置覆盖原文件。
- **R2-AC9**：写入过程中发生 I/O / 权限错误时不留下任何部分写入的
  ``ego-config.yaml``（清理半成品），抛 :class:`~ego_pod.exit_codes.EgoError`
  （退出码 1）。

测试约定：:func:`ego_pod.commands.init.run_init` 接收一个 argparse 命名空间
（仅用到 ``.force``），成功时返回 :data:`~ego_pod.exit_codes.EXIT_OK`，失败时
**抛出** :class:`~ego_pod.exit_codes.EgoError`（由 CLI 顶层映射退出码，本函数不
自行写 stderr）。所有用例通过 ``monkeypatch.chdir(tmp_path)`` 将当前工作目录切到
临时目录，使 ``run_init`` 在隔离的文件系统位置生成配置。
"""

from __future__ import annotations

import argparse
import os

import pytest
import yaml

from ego_pod.commands.init import CONFIG_FILENAME, run_init
from ego_pod.config.defaults import (
    DEFAULT_BASE_MODEL,
    DEFAULT_EDITION_NAME,
    render_default_yaml,
)
from ego_pod.exit_codes import EXIT_OK, EXIT_RUNTIME_FAILURE, EgoError

# 五个 Data_Category 的固定顺序（与 defaults 模块一致）。
EXPECTED_CATEGORIES = ["sense", "action", "thought", "skill", "credentials"]


def _make_args(force: bool = False) -> argparse.Namespace:
    """构造 ``run_init`` 所需的 argparse 命名空间（仅 ``force`` 字段）。"""
    return argparse.Namespace(force=force)


# ===========================================================================
# R2-AC1 / AC8：成功创建 → 退出码 0，文件存在，打印绝对路径
# ===========================================================================


def test_init_creates_config_and_returns_ok(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    rc = run_init(_make_args(force=False))
    assert rc == EXIT_OK

    target = tmp_path / CONFIG_FILENAME
    assert target.exists()


def test_init_prints_absolute_path(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    run_init(_make_args(force=False))

    out = capsys.readouterr().out.strip()
    expected = str((tmp_path / CONFIG_FILENAME).resolve())
    # R2-AC8：标准输出打印生成文件的绝对路径。
    assert out == expected
    assert os.path.isabs(out)


def test_init_writes_utf8_without_bom(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    run_init(_make_args(force=False))

    target = tmp_path / CONFIG_FILENAME
    raw = target.read_bytes()
    # R2-AC1：UTF-8 编码且不含 BOM。
    assert not raw.startswith(b"\xef\xbb\xbf")
    # 内容与默认渲染文本一致。
    assert target.read_text(encoding="utf-8") == render_default_yaml()


# ===========================================================================
# R2-AC2..AC5：生成文件字段完备性
# ===========================================================================


def _load_generated(tmp_path) -> dict:
    """解析已生成的 ego-config.yaml 为字典，供字段断言复用。"""
    data = yaml.safe_load((tmp_path / CONFIG_FILENAME).read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def test_generated_has_top_level_fields(tmp_path, monkeypatch, capsys):
    # R2-AC2：顶层包含 sources / training / editions / output。
    monkeypatch.chdir(tmp_path)
    run_init(_make_args())
    data = _load_generated(tmp_path)
    for field in ("sources", "training", "editions", "output"):
        assert field in data, f"缺少顶层字段 {field}"


def test_generated_has_five_local_sources(tmp_path, monkeypatch, capsys):
    # R2-AC3：五个 Data_Category 各一条目，含 type: local 与非空路径。
    monkeypatch.chdir(tmp_path)
    run_init(_make_args())
    sources = _load_generated(tmp_path)["sources"]
    assert list(sources.keys()) == EXPECTED_CATEGORIES
    for name, entry in sources.items():
        assert entry["type"] == "local", f"{name} 的 type 应为 local"
        assert isinstance(entry["path"], str) and entry["path"].strip()


def test_generated_editions_only_normal(tmp_path, monkeypatch, capsys):
    # R2-AC4：仅声明 normal edition，base 为 qwen3.5-0.8B。
    monkeypatch.chdir(tmp_path)
    run_init(_make_args())
    editions = _load_generated(tmp_path)["editions"]
    assert list(editions.keys()) == [DEFAULT_EDITION_NAME]
    assert editions[DEFAULT_EDITION_NAME]["base"] == DEFAULT_BASE_MODEL


def test_generated_training_qlora_params(tmp_path, monkeypatch, capsys):
    # R2-AC5：method=qlora 且五个数值参数完备并取默认值。
    monkeypatch.chdir(tmp_path)
    run_init(_make_args())
    training = _load_generated(tmp_path)["training"]
    assert training["method"] == "qlora"
    assert training["lora_rank"] == 16
    assert training["lora_alpha"] == 32
    assert training["learning_rate"] == pytest.approx(2e-4)
    assert training["epochs"] == 3
    assert training["batch_size"] == 4
    # 整数参数为正整数，learning_rate 为大于 0 的浮点数。
    for int_param in ("lora_rank", "lora_alpha", "epochs", "batch_size"):
        value = training[int_param]
        assert isinstance(value, int) and not isinstance(value, bool)
        assert value > 0
    assert isinstance(training["learning_rate"], float)
    assert training["learning_rate"] > 0


# ===========================================================================
# R2-AC6：已存在且无 --force → 拒绝写入，退出码 1，原文件不变
# ===========================================================================


def test_init_rejects_existing_without_force(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / CONFIG_FILENAME
    sentinel = "original: content\n"
    target.write_text(sentinel, encoding="utf-8")

    with pytest.raises(EgoError) as excinfo:
        run_init(_make_args(force=False))

    # 退出码 1，错误信息含"已存在"提示。
    assert excinfo.value.exit_code == EXIT_RUNTIME_FAILURE
    assert "已存在" in str(excinfo.value)
    # 原文件内容保持不变（未被覆盖）。
    assert target.read_text(encoding="utf-8") == sentinel


def test_init_rejects_existing_does_not_print_path(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / CONFIG_FILENAME
    target.write_text("original: content\n", encoding="utf-8")

    with pytest.raises(EgoError):
        run_init(_make_args(force=False))

    # 拒绝写入路径下不应在标准输出打印成功路径。
    assert capsys.readouterr().out == ""


# ===========================================================================
# R2-AC7：--force → 覆盖原文件
# ===========================================================================


def test_init_force_overwrites_existing(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / CONFIG_FILENAME
    target.write_text("stale: data\n", encoding="utf-8")

    rc = run_init(_make_args(force=True))
    assert rc == EXIT_OK

    content = target.read_text(encoding="utf-8")
    # 旧内容被替换为默认配置。
    assert "stale: data" not in content
    assert "sources:" in content
    assert content == render_default_yaml()


def test_init_force_on_empty_dir_creates(tmp_path, monkeypatch, capsys):
    # --force 在不存在文件时同样正常创建（覆盖语义对"不存在"是创建）。
    monkeypatch.chdir(tmp_path)
    rc = run_init(_make_args(force=True))
    assert rc == EXIT_OK
    assert (tmp_path / CONFIG_FILENAME).exists()


# ===========================================================================
# R2-AC9：写入 I/O 错误 → 不留部分写入文件，退出码 1
# ===========================================================================


def test_init_io_error_leaves_no_partial_file(tmp_path, monkeypatch):
    # 模拟原子写入函数在写入过程中抛 OSError。
    monkeypatch.chdir(tmp_path)

    def _boom(target, text):
        raise OSError("disk full")

    monkeypatch.setattr("ego_pod.commands.init._atomic_write_text", _boom)

    with pytest.raises(EgoError) as excinfo:
        run_init(_make_args(force=False))

    # R2-AC9：退出码 1，错误信息携带原因。
    assert excinfo.value.exit_code == EXIT_RUNTIME_FAILURE
    # 目标目录不应留下任何（部分写入的）ego-config.yaml。
    assert not (tmp_path / CONFIG_FILENAME).exists()


def test_init_io_error_cleans_temp_files(tmp_path, monkeypatch):
    # 让底层 os.replace 失败，触发 _atomic_write_text 的 except 清理分支：
    # 临时文件已写入但 rename 失败 → 应清理临时文件，且不留下 ego-config.yaml。
    monkeypatch.chdir(tmp_path)

    real_replace = os.replace

    def _failing_replace(src, dst, *args, **kwargs):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(os, "replace", _failing_replace)

    with pytest.raises(EgoError) as excinfo:
        run_init(_make_args(force=False))
    assert excinfo.value.exit_code == EXIT_RUNTIME_FAILURE

    # 还原 os.replace 以免影响后续清理。
    monkeypatch.setattr(os, "replace", real_replace)

    # 不留下最终配置文件，也不残留任何 .tmp 临时文件。
    assert not (tmp_path / CONFIG_FILENAME).exists()
    leftover = [p.name for p in tmp_path.iterdir()]
    assert leftover == [], f"残留临时文件: {leftover}"


@pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0,
    reason="root 会绕过目录写权限检查",
)
def test_init_readonly_directory_leaves_no_partial_file(tmp_path, monkeypatch):
    # 真实场景：当前工作目录只读 → 在其中创建临时文件失败（PermissionError）。
    monkeypatch.chdir(tmp_path)
    os.chmod(tmp_path, 0o500)
    try:
        with pytest.raises(EgoError) as excinfo:
            run_init(_make_args(force=False))
        assert excinfo.value.exit_code == EXIT_RUNTIME_FAILURE
        assert not (tmp_path / CONFIG_FILENAME).exists()
    finally:
        # 还原权限以便 tmp_path 清理。
        os.chmod(tmp_path, 0o700)
