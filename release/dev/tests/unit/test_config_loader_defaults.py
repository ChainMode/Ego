"""`ego_pod.config.loader` 与 `ego_pod.config.defaults` 的单元测试。

本文件聚焦"加载/解析"与"默认配置渲染/写入"两个纯粹的 I/O 关注点，
对应任务 2.6，覆盖以下需求条款：

- R3-AC2：当前工作目录不存在 ``ego-config.yaml`` 时的 not-found 提示。
- R3-AC3：文件存在但无读权限，或存在性检查与读取之间 / 读取过程中发生
  任何 I/O 错误（被删除、被替换为目录等）时的错误处理。
- R3-AC4：非法 YAML 文档报告解析错误所在的行号与列号。
- R2-AC2 / AC3 / AC4 / AC5：``defaults.render_default_yaml`` /
  ``defaults.write_default`` 渲染产物的字段完备性与默认值
  （lora_rank=16, lora_alpha=32, lr=2e-4, epochs=3, batch_size=4；五个
  Data_Category 数据源；仅 normal edition，base 为 qwen3.5-0.8B）。

加载相关测试与 ``test_config_validators.py`` 中的少量 loader 样例互补：本文件
以更细的粒度断言行列号数值、I/O 错误分类，以及 loader 与 defaults 的端到端
往返（渲染 → 写盘 → 加载 → EgoConfig）。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from ego_pod.config.defaults import (
    DEFAULT_BASE_MODEL,
    DEFAULT_EDITION_NAME,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SOURCE_PATHS,
    DEFAULT_TRAINING,
    render_default_yaml,
    write_default,
)
from ego_pod.config.loader import load
from ego_pod.config.schema import EgoConfig
from ego_pod.exit_codes import EXIT_USAGE_OR_CONFIG, ConfigError

CONFIG_NAME = "ego-config.yaml"

# 任务 2.6 与设计文档约定的五个 Data_Category 固定顺序。
EXPECTED_CATEGORIES = ["sense", "action", "thought", "skill", "credentials"]


# ===========================================================================
# R3-AC2：文件不存在
# ===========================================================================


def test_load_missing_file_raises_not_found(tmp_path):
    missing = tmp_path / CONFIG_NAME
    with pytest.raises(ConfigError) as exc:
        load(missing)
    # 提示应引导用户执行 ego init（R3-AC2 固定文案）。
    assert "ego init" in str(exc.value)
    assert exc.value.exit_code == EXIT_USAGE_OR_CONFIG


def test_load_missing_file_for_relative_path(tmp_path, monkeypatch):
    # 以相对路径加载且当前工作目录无配置文件时同样按 not-found 处理。
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ConfigError) as exc:
        load(CONFIG_NAME)
    assert "ego init" in str(exc.value)
    assert exc.value.exit_code == EXIT_USAGE_OR_CONFIG


# ===========================================================================
# R3-AC3：I/O 与权限错误
# ===========================================================================


def test_load_path_is_directory_is_io_error(tmp_path):
    # 路径存在但为目录 → 读取阶段抛 IsADirectoryError(OSError) → ConfigError。
    target = tmp_path / CONFIG_NAME
    target.mkdir()
    with pytest.raises(ConfigError) as exc:
        load(target)
    message = str(exc.value)
    assert exc.value.exit_code == EXIT_USAGE_OR_CONFIG
    # 错误信息应包含出错文件的绝对路径，便于用户定位。
    assert str(target.resolve()) in message or str(target) in message
    # 目录读取错误归类为 I/O 错误，而非 not-found 提示。
    assert "ego init" not in message


@pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0,
    reason="root 会绕过文件权限检查",
)
def test_load_permission_error(tmp_path):
    target = tmp_path / CONFIG_NAME
    target.write_text("sources: {}\n", encoding="utf-8")
    target.chmod(0o000)
    try:
        with pytest.raises(ConfigError) as exc:
            load(target)
        message = str(exc.value)
        assert exc.value.exit_code == EXIT_USAGE_OR_CONFIG
        # 无读权限不应被误判为文件不存在。
        assert "ego init" not in message
    finally:
        # 还原权限以便 tmp_path 清理。
        target.chmod(0o600)


def test_load_io_error_preserves_cause(tmp_path):
    # ConfigError 应链接底层 OSError 作为 __cause__，保留可诊断的异常链。
    target = tmp_path / CONFIG_NAME
    target.mkdir()
    with pytest.raises(ConfigError) as exc:
        load(target)
    assert isinstance(exc.value.__cause__, OSError)


# ===========================================================================
# R3-AC4：非法 YAML 报告行号与列号
# ===========================================================================


def test_load_invalid_yaml_reports_line_and_column(tmp_path):
    target = tmp_path / CONFIG_NAME
    # 未闭合的流式集合会触发带位置标记的解析错误。
    target.write_text("sources: {unclosed: [1, 2\n", encoding="utf-8")
    with pytest.raises(ConfigError) as exc:
        load(target)
    message = str(exc.value)
    assert exc.value.exit_code == EXIT_USAGE_OR_CONFIG
    assert "行" in message and "列" in message


def test_load_invalid_yaml_reports_specific_position(tmp_path):
    # 错误缩进在第 2 行；断言报告的行列号与 PyYAML 标记（1 基）一致。
    target = tmp_path / CONFIG_NAME
    target.write_text("a: 1\n  b: 2\n", encoding="utf-8")
    with pytest.raises(ConfigError) as exc:
        load(target)
    message = str(exc.value)
    assert "行 2" in message
    assert "列 4" in message


def test_load_invalid_yaml_one_based_line(tmp_path):
    # 首行即出错时应报告"行 1"（验证 0 基 → 1 基转换）。
    # 流式序列以 `}` 错误闭合 → PyYAML 在第 1 行第 6 列标记问题。
    target = tmp_path / CONFIG_NAME
    target.write_text("[a, b}\n", encoding="utf-8")
    with pytest.raises(ConfigError) as exc:
        load(target)
    message = str(exc.value)
    assert "行 1" in message
    assert "列 6" in message


# ===========================================================================
# R2-AC2..AC5：render_default_yaml 字段完备性与默认值
# ===========================================================================


def _parsed_default() -> dict:
    """渲染默认 YAML 并解析为原始映射，供字段断言复用。"""

    data = yaml.safe_load(render_default_yaml())
    assert isinstance(data, dict)
    return data


def test_render_default_top_level_fields():
    # R2-AC2：顶层包含 sources / training / editions / output。
    data = _parsed_default()
    for field in ("sources", "training", "editions", "output"):
        assert field in data, f"缺少顶层字段 {field}"


def test_render_default_sources_five_categories():
    # R2-AC3：五个 Data_Category 各一个条目，含 type: local 与非空路径。
    data = _parsed_default()
    sources = data["sources"]
    assert list(sources.keys()) == EXPECTED_CATEGORIES
    for name, entry in sources.items():
        assert entry["type"] == "local", f"{name} 的 type 应为 local"
        assert isinstance(entry["path"], str)
        assert entry["path"].strip(), f"{name} 的 path 不应为空"


def test_render_default_sources_paths_match_constant():
    data = _parsed_default()
    for name in EXPECTED_CATEGORIES:
        assert data["sources"][name]["path"] == DEFAULT_SOURCE_PATHS[name]


def test_render_default_editions_only_normal():
    # R2-AC4：仅声明 normal edition，base 为 qwen3.5-0.8B。
    data = _parsed_default()
    editions = data["editions"]
    assert list(editions.keys()) == [DEFAULT_EDITION_NAME]
    assert editions[DEFAULT_EDITION_NAME]["base"] == DEFAULT_BASE_MODEL


def test_render_default_training_method_and_quantization():
    # R2-AC5：training.method 为 qlora，quantization 为 4bit。
    data = _parsed_default()
    training = data["training"]
    assert training["method"] == "qlora"
    assert training["quantization"] == "4bit"


def test_render_default_training_numeric_values():
    # R2-AC5：五个数值参数的默认值。
    data = _parsed_default()
    training = data["training"]
    assert training["lora_rank"] == 16
    assert training["lora_alpha"] == 32
    assert training["learning_rate"] == pytest.approx(2e-4)
    assert training["epochs"] == 3
    assert training["batch_size"] == 4


def test_render_default_training_numeric_types():
    # 数值参数解析后类型应正确：整数参数为 int、learning_rate 为 float
    # （而非被 YAML 解析成字符串）。
    data = _parsed_default()
    training = data["training"]
    for int_param in ("lora_rank", "lora_alpha", "epochs", "batch_size"):
        value = training[int_param]
        assert isinstance(value, int) and not isinstance(value, bool)
    assert isinstance(training["learning_rate"], float)


def test_render_default_values_match_constants():
    # 渲染产物应与模块级默认常量保持一致，避免常量与渲染漂移。
    data = _parsed_default()
    training = data["training"]
    for key, expected in DEFAULT_TRAINING.items():
        if key == "learning_rate":
            assert training[key] == pytest.approx(float(expected))
        else:
            assert training[key] == expected
    assert data["output"]["dir"] == DEFAULT_OUTPUT_DIR


def test_render_default_is_utf8_without_bom():
    text = render_default_yaml()
    assert not text.startswith("\ufeff")
    # 文本应能以 UTF-8 编码且不含 BOM 字节。
    encoded = text.encode("utf-8")
    assert not encoded.startswith(b"\xef\xbb\xbf")


# ===========================================================================
# write_default：写盘行为与编码
# ===========================================================================


def test_write_default_creates_utf8_file(tmp_path):
    target = tmp_path / CONFIG_NAME
    returned = write_default(target)
    assert target.exists()
    # 返回写入文件的绝对路径。
    assert returned == target.resolve()
    raw = target.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")  # 无 BOM
    # 内容应与渲染文本完全一致。
    assert target.read_text(encoding="utf-8") == render_default_yaml()


def test_write_default_overwrites_existing(tmp_path):
    target = tmp_path / CONFIG_NAME
    target.write_text("stale content\n", encoding="utf-8")
    write_default(target)
    assert "stale content" not in target.read_text(encoding="utf-8")
    assert "sources:" in target.read_text(encoding="utf-8")


# ===========================================================================
# 端到端：defaults 渲染 → 写盘 → loader 加载 → EgoConfig 字段一致
# ===========================================================================


def test_default_config_round_trips_through_loader(tmp_path):
    target = tmp_path / CONFIG_NAME
    write_default(target)

    config, abs_path = load(target)

    assert isinstance(config, EgoConfig)
    assert abs_path == str(target.resolve())
    assert os.path.isabs(abs_path)

    # sources：五类齐全且均为 local + 非空路径。
    assert list(config.sources.keys()) == EXPECTED_CATEGORIES
    for name in EXPECTED_CATEGORIES:
        entry = config.sources[name]
        assert entry.type == "local"
        assert entry.path == DEFAULT_SOURCE_PATHS[name]

    # training：方法、量化与五个数值参数。
    training = config.training
    assert training.method == "qlora"
    assert training.quantization == "4bit"
    assert training.lora_rank == 16
    assert training.lora_alpha == 32
    assert training.learning_rate == pytest.approx(2e-4)
    assert training.epochs == 3
    assert training.batch_size == 4

    # editions：仅 normal，base 正确。
    assert list(config.editions.keys()) == [DEFAULT_EDITION_NAME]
    assert config.editions[DEFAULT_EDITION_NAME].base == DEFAULT_BASE_MODEL
