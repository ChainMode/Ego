"""``ego train`` 失败路径集成测试（任务 12.2）。

本文件验证 ``ego train`` 在配置错误与各数据/语料异常场景下的退出码、标准错误
内容与「未产出 ``ego-output/``」不变量，覆盖：

* **R3 配置错误（退出码 2）**：
  - R3-AC2：当前工作目录无 ``ego-config.yaml``；
  - R3-AC5：顶层缺失 ``sources``；
  - R3-AC6：``editions`` 未声明 ``normal``；
  - R3-AC7：``editions.normal.base`` 非 ``qwen3.5-0.8B``；
  - R3-AC8：存在非 ``local`` 数据源条目；
  - R3-AC9：``training.method`` 非 ``qlora``。
  对应 R11-AC6：配置加载/校验失败 → 不启动 Collector、退出码 2。
* **R4-AC7 数据全空（退出码 1）**：五个 Data_Category 本地目录全部不存在。
* **R7-AC6 语料为空（退出码 1）**：数据通过采集但标准化后样本数为 0
  （0 字节文件经模板分类得到空 output 被丢弃）。
* **R7-AC7 无法补足 16（退出码 1）**：过滤后样本数 ``0 < n < 16`` 且对全部
  预定义增强模板应用后仍无法补足至 16。

每个用例均额外断言 ``ego-output/`` 未被创建（R11-AC2：失败阶段终止后续阶段，
Vault 从未运行）。所有用例通过 ``conftest`` 的 autouse ``force_stub_backend``
夹具强制 Stub_Backend，并以 ``tmp_path`` 为工作目录，不依赖网络。
"""

from __future__ import annotations

import pytest

from ego_pod.exit_codes import EXIT_RUNTIME_FAILURE, EXIT_USAGE_OR_CONFIG


def _assert_no_output(tmp_path) -> None:
    """断言 ``ego-output/`` 未被创建（失败路径不得留下产物）。"""
    assert not (tmp_path / "ego-output").exists(), "失败路径不应创建 ego-output/"


# ===========================================================================
# R3 配置错误：退出码 2（R11-AC6）
# ===========================================================================


def test_missing_config_file_exit_2(tmp_path, run_train):
    """R3-AC2：当前目录无 ego-config.yaml → 退出码 2，stderr 提示先 init。"""
    # 不写任何 ego-config.yaml。
    rc, out, err = run_train()

    assert rc == EXIT_USAGE_OR_CONFIG
    assert "未找到" in err
    assert "ego-config.yaml" in err
    _assert_no_output(tmp_path)


def test_missing_top_level_sources_exit_2(
    tmp_path, make_category_dirs, write_config, run_train
):
    """R3-AC5：顶层缺失 sources → 退出码 2，stderr 指出缺失字段。"""
    sources = make_category_dirs(tmp_path, {"sense": [("a.txt", "有效内容文本")]})
    write_config(tmp_path / "ego-config.yaml", sources, drop_top=["sources"])

    rc, out, err = run_train()

    assert rc == EXIT_USAGE_OR_CONFIG
    assert "sources" in err
    assert "缺少" in err
    _assert_no_output(tmp_path)


def test_editions_without_normal_exit_2(
    tmp_path, make_category_dirs, write_config, run_train
):
    """R3-AC6：editions 未声明 normal → 退出码 2。"""
    sources = make_category_dirs(tmp_path, {"sense": [("a.txt", "有效内容文本")]})
    write_config(
        tmp_path / "ego-config.yaml",
        sources,
        editions={"plus": {"base": "qwen3.5-0.8B"}},
    )

    rc, out, err = run_train()

    assert rc == EXIT_USAGE_OR_CONFIG
    assert "MVP 仅支持 normal" in err
    _assert_no_output(tmp_path)


def test_illegal_base_model_exit_2(
    tmp_path, make_category_dirs, write_config, run_train
):
    """R3-AC7：editions.normal.base 非 qwen3.5-0.8B → 退出码 2。"""
    sources = make_category_dirs(tmp_path, {"sense": [("a.txt", "有效内容文本")]})
    write_config(
        tmp_path / "ego-config.yaml",
        sources,
        editions={"normal": {"base": "gpt-4"}},
    )

    rc, out, err = run_train()

    assert rc == EXIT_USAGE_OR_CONFIG
    assert "qwen3.5-0.8B" in err
    _assert_no_output(tmp_path)


def test_non_local_source_exit_2(
    tmp_path, make_category_dirs, write_config, run_train
):
    """R3-AC8：存在非 local 数据源条目 → 退出码 2。"""
    sources = make_category_dirs(tmp_path, {"sense": [("a.txt", "有效内容文本")]})
    write_config(
        tmp_path / "ego-config.yaml",
        sources,
        source_types={"sense": "cloud"},
    )

    rc, out, err = run_train()

    assert rc == EXIT_USAGE_OR_CONFIG
    assert "local 数据源" in err
    _assert_no_output(tmp_path)


def test_non_qlora_method_exit_2(
    tmp_path, make_category_dirs, write_config, run_train
):
    """R3-AC9：training.method 非 qlora → 退出码 2。"""
    sources = make_category_dirs(tmp_path, {"sense": [("a.txt", "有效内容文本")]})
    write_config(
        tmp_path / "ego-config.yaml",
        sources,
        training={
            "method": "full",
            "quantization": "4bit",
            "lora_rank": 16,
            "lora_alpha": 32,
            "learning_rate": 0.0002,
            "epochs": 3,
            "batch_size": 4,
        },
    )

    rc, out, err = run_train()

    assert rc == EXIT_USAGE_OR_CONFIG
    assert "qlora" in err
    _assert_no_output(tmp_path)


# ===========================================================================
# R4-AC7：五个 Data_Category 全空 → 退出码 1
# ===========================================================================


def test_all_data_empty_exit_1(
    tmp_path, make_category_dirs, write_config, run_train
):
    """R4-AC7：五类本地目录全部不存在 → Collector 退出码 1。"""
    # 不为任何类别创建目录 → 五个路径均不存在。
    sources = make_category_dirs(tmp_path, {})
    write_config(tmp_path / "ego-config.yaml", sources)

    rc, out, err = run_train()

    assert rc == EXIT_RUNTIME_FAILURE
    assert "无可用训练数据" in err
    _assert_no_output(tmp_path)


# ===========================================================================
# R7-AC6：过滤后语料为空 → 退出码 1
# ===========================================================================


def test_empty_corpus_exit_1(
    tmp_path, make_category_dirs, write_config, run_train
):
    """R7-AC6：仅含 0 字节文件 → 采集通过但标准化后样本数为 0。

    0 字节文件经模板分类得到空 output（template 的 ``{text}`` 替换为空串），
    按 R6-AC6 在分类阶段被丢弃，过滤后样本数为 0，触发 R7-AC6。
    """
    # sense 目录含两个 0 字节文件（采集计数 > 0，避免落入 R4-AC7）。
    sources = make_category_dirs(
        tmp_path, {"sense": [("empty1.txt", ""), ("empty2.txt", "")]}
    )
    write_config(tmp_path / "ego-config.yaml", sources)

    rc, out, err = run_train()

    assert rc == EXIT_RUNTIME_FAILURE
    assert "可用语料为空" in err
    _assert_no_output(tmp_path)


# ===========================================================================
# R7-AC7：无法补足最小样本数 16 → 退出码 1
# ===========================================================================


def test_cannot_reach_sixteen_exit_1(
    tmp_path, make_category_dirs, write_config, run_train
):
    """R7-AC7：过滤后样本数 0 < n < 16 且增强后仍无法达到 16 → 退出码 1。

    仅 2 条互异样本：增强阶段对每条样本套用全部 5 条预定义模板，最多得
    2 + 2×5 = 12 条，仍不足 16，触发 R7-AC7。
    """
    # thought 目录含两个内容互异、非空、足够长的文件（避免被去重或过短过滤）。
    sources = make_category_dirs(
        tmp_path,
        {
            "thought": [
                ("note1.txt", "第一条互异的笔记内容，足够长以通过过滤。"),
                ("note2.txt", "第二条与第一条不同的笔记内容。"),
            ]
        },
    )
    write_config(tmp_path / "ego-config.yaml", sources)

    rc, out, err = run_train()

    assert rc == EXIT_RUNTIME_FAILURE
    assert "无法满足最小样本数 16" in err
    _assert_no_output(tmp_path)
