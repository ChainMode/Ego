"""`ego_pod.config.validators` 配置校验的单元测试。

覆盖需求 R3 的全部错误条款与成功语义：

- R3-AC2 / AC3 / AC4：文件加载与 YAML 解析错误（由 ``loader`` 负责，本文件以
  少量样例确认其行为，从而完整覆盖 R3 错误条款 AC2–AC9）。
- R3-AC5：顶层 ``sources`` → ``training`` → ``editions`` 缺失检测，仅报告首个。
- R3-AC6：``editions`` 必须声明 ``normal``。
- R3-AC7：``editions.normal.base`` 必须为 ``qwen3.5-0.8B``。
- R3-AC8：``sources`` 不得存在 ``type`` 非 ``local`` 的条目。
- R3-AC9：``training.method`` 必须为 ``qlora``。
- R3-AC10：全部通过时打印 ``Config validated: <绝对路径>``；任一失败分支抑制
  该确认信息（成功输出抑制）。
- R3-AC11：按 AC 编号升序（5 → 6 → 7 → 8 → 9）短路，仅报告首个违规。

``validators.validate`` 作用于 ``loader`` 解析得到的**原始映射**（而非已构造的
``EgoConfig``），因此本文件直接以 dict 构造输入。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ego_pod.config.loader import load
from ego_pod.config.validators import validate
from ego_pod.exit_codes import EXIT_USAGE_OR_CONFIG, ConfigError


# ---------------------------------------------------------------------------
# 测试夹具：合法配置的原始映射
# ---------------------------------------------------------------------------


def _valid_mapping() -> dict:
    """返回一份通过全部 R3/R8-AC9 校验的原始配置映射。"""

    return {
        "sources": {
            "sense": {"type": "local", "path": "./data/sense"},
            "action": {"type": "local", "path": "./data/action"},
            "thought": {"type": "local", "path": "./data/thought"},
            "skill": {"type": "local", "path": "./data/skill"},
            "credentials": {"type": "local", "path": "./data/credentials"},
        },
        "training": {
            "method": "qlora",
            "quantization": "4bit",
            "lora_rank": 16,
            "lora_alpha": 32,
            "learning_rate": 2e-4,
            "epochs": 3,
            "batch_size": 4,
        },
        "editions": {
            "normal": {"base": "qwen3.5-0.8B", "description": ""},
        },
        "output": {},
    }


CONFIG_PATH = "ego-config.yaml"


# ---------------------------------------------------------------------------
# 成功路径与 R3-AC10 成功确认信息
# ---------------------------------------------------------------------------


def test_validate_success_returns_none_and_prints_confirmation(capsys):
    result = validate(_valid_mapping(), CONFIG_PATH)
    captured = capsys.readouterr()
    assert result is None
    assert captured.out.strip() == f"Config validated: {Path(CONFIG_PATH).resolve()}"


def test_validate_success_prints_absolute_path(capsys):
    # 即便传入相对路径，确认信息也应展示解析后的绝对路径（R3-AC10）。
    validate(_valid_mapping(), "subdir/ego-config.yaml")
    out = capsys.readouterr().out
    printed = out.split("Config validated: ", 1)[1].strip()
    assert os.path.isabs(printed)
    assert printed == str(Path("subdir/ego-config.yaml").resolve())


def test_validate_success_writes_to_custom_stream():
    import io

    stream = io.StringIO()
    validate(_valid_mapping(), CONFIG_PATH, stream=stream)
    assert stream.getvalue().strip() == f"Config validated: {Path(CONFIG_PATH).resolve()}"


# ---------------------------------------------------------------------------
# R3-AC5：顶层字段缺失（sources → training → editions 顺序，仅报告首个）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("missing", ["sources", "training", "editions"])
def test_missing_top_level_field_reported(missing):
    data = _valid_mapping()
    del data[missing]
    with pytest.raises(ConfigError) as exc:
        validate(data, CONFIG_PATH)
    assert missing in str(exc.value)
    assert exc.value.exit_code == EXIT_USAGE_OR_CONFIG


def test_top_level_none_value_treated_as_missing():
    # YAML 中写成空的 `sources:` 会解析为 None，应视为缺失（R3-AC5）。
    data = _valid_mapping()
    data["sources"] = None
    with pytest.raises(ConfigError) as exc:
        validate(data, CONFIG_PATH)
    assert "sources" in str(exc.value)


def test_top_level_order_reports_sources_first():
    # 三个顶层字段同时缺失 → 仅报告固定顺序中的首个 `sources`（R3-AC5/AC11）。
    data = _valid_mapping()
    del data["sources"]
    del data["training"]
    del data["editions"]
    with pytest.raises(ConfigError) as exc:
        validate(data, CONFIG_PATH)
    assert "sources" in str(exc.value)
    assert "training" not in str(exc.value)
    assert "editions" not in str(exc.value)


def test_top_level_order_reports_training_before_editions():
    data = _valid_mapping()
    del data["training"]
    del data["editions"]
    with pytest.raises(ConfigError) as exc:
        validate(data, CONFIG_PATH)
    assert "training" in str(exc.value)
    assert "editions" not in str(exc.value)


# ---------------------------------------------------------------------------
# R3-AC6：editions 未声明 normal
# ---------------------------------------------------------------------------


def test_missing_normal_edition():
    data = _valid_mapping()
    data["editions"] = {"lite": {"base": "qwen3.5-0.8B"}}
    with pytest.raises(ConfigError) as exc:
        validate(data, CONFIG_PATH)
    assert "normal" in str(exc.value)
    assert exc.value.exit_code == EXIT_USAGE_OR_CONFIG


# ---------------------------------------------------------------------------
# R3-AC7：editions.normal.base 非 qwen3.5-0.8B
# ---------------------------------------------------------------------------


def test_wrong_base_model():
    data = _valid_mapping()
    data["editions"]["normal"]["base"] = "gemma-2b"
    with pytest.raises(ConfigError) as exc:
        validate(data, CONFIG_PATH)
    assert "qwen3.5-0.8B" in str(exc.value)
    assert exc.value.exit_code == EXIT_USAGE_OR_CONFIG


# ---------------------------------------------------------------------------
# R3-AC8：sources 含非 local 条目（报告首个违规条目名称）
# ---------------------------------------------------------------------------


def test_non_local_source_reports_entry_name():
    data = _valid_mapping()
    data["sources"]["thought"]["type"] = "url"
    with pytest.raises(ConfigError) as exc:
        validate(data, CONFIG_PATH)
    message = str(exc.value)
    assert "thought" in message
    assert "local" in message
    assert exc.value.exit_code == EXIT_USAGE_OR_CONFIG


def test_non_local_source_reports_first_violation_in_order():
    # 多个非 local 条目时，按 sources 插入顺序报告首个（sense 在 skill 之前）。
    data = _valid_mapping()
    data["sources"]["sense"]["type"] = "s3"
    data["sources"]["skill"]["type"] = "url"
    with pytest.raises(ConfigError) as exc:
        validate(data, CONFIG_PATH)
    assert "sense" in str(exc.value)
    assert "skill" not in str(exc.value)


# ---------------------------------------------------------------------------
# R3-AC9：training.method 非 qlora
# ---------------------------------------------------------------------------


def test_wrong_training_method():
    data = _valid_mapping()
    data["training"]["method"] = "full"
    with pytest.raises(ConfigError) as exc:
        validate(data, CONFIG_PATH)
    assert "qlora" in str(exc.value)
    assert exc.value.exit_code == EXIT_USAGE_OR_CONFIG


# ---------------------------------------------------------------------------
# R8-AC9：五个 LoRA 训练参数完备性/合法性预检（validate 末步）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "param,bad_value",
    [
        ("lora_rank", 0),
        ("lora_rank", -1),
        ("lora_rank", 1.5),
        ("lora_alpha", 0),
        ("learning_rate", 0),
        ("learning_rate", -0.1),
        ("epochs", 0),
        ("batch_size", -2),
    ],
)
def test_invalid_training_param(param, bad_value):
    data = _valid_mapping()
    data["training"][param] = bad_value
    with pytest.raises(ConfigError) as exc:
        validate(data, CONFIG_PATH)
    assert param in str(exc.value)


@pytest.mark.parametrize(
    "param", ["lora_rank", "lora_alpha", "learning_rate", "epochs", "batch_size"]
)
def test_missing_training_param(param):
    data = _valid_mapping()
    del data["training"][param]
    with pytest.raises(ConfigError) as exc:
        validate(data, CONFIG_PATH)
    assert param in str(exc.value)


def test_bool_not_accepted_as_positive_int():
    # bool 是 int 的子类，但不应被当作合法正整数参数。
    data = _valid_mapping()
    data["training"]["lora_rank"] = True
    with pytest.raises(ConfigError) as exc:
        validate(data, CONFIG_PATH)
    assert "lora_rank" in str(exc.value)


# ---------------------------------------------------------------------------
# R3-AC11：按 AC 编号升序短路，仅报告首个违规
# ---------------------------------------------------------------------------


def test_short_circuit_ac6_before_ac9():
    # 同时违反 AC6（无 normal）与 AC9（method 非 qlora）→ 报告 AC6。
    data = _valid_mapping()
    data["editions"] = {"lite": {"base": "qwen3.5-0.8B"}}
    data["training"]["method"] = "full"
    with pytest.raises(ConfigError) as exc:
        validate(data, CONFIG_PATH)
    assert "normal" in str(exc.value)
    assert "qlora" not in str(exc.value)


def test_short_circuit_ac7_before_ac8():
    # 同时违反 AC7（base 错）与 AC8（非 local）→ 报告 AC7。
    data = _valid_mapping()
    data["editions"]["normal"]["base"] = "gemma-2b"
    data["sources"]["sense"]["type"] = "url"
    with pytest.raises(ConfigError) as exc:
        validate(data, CONFIG_PATH)
    assert "qwen3.5-0.8B" in str(exc.value)
    assert "数据源" not in str(exc.value)


def test_short_circuit_ac8_before_ac9():
    # 同时违反 AC8（非 local）与 AC9（method 非 qlora）→ 报告 AC8。
    data = _valid_mapping()
    data["sources"]["action"]["type"] = "url"
    data["training"]["method"] = "full"
    with pytest.raises(ConfigError) as exc:
        validate(data, CONFIG_PATH)
    assert "action" in str(exc.value)
    assert "qlora" not in str(exc.value)


def test_short_circuit_ac5_before_ac6():
    # 缺失 editions（AC5）优先于 normal 缺失（AC6）的语义被报告。
    data = _valid_mapping()
    del data["editions"]
    with pytest.raises(ConfigError) as exc:
        validate(data, CONFIG_PATH)
    assert "editions" in str(exc.value)


# ---------------------------------------------------------------------------
# R3-AC10：任一校验失败时抑制成功确认信息
# ---------------------------------------------------------------------------


def _broken_mappings() -> list[dict]:
    """构造覆盖 AC5–AC9 与 R8-AC9 的各类违规映射。"""

    cases = []

    m = _valid_mapping()
    del m["sources"]
    cases.append(m)  # AC5

    m = _valid_mapping()
    m["editions"] = {"lite": {"base": "qwen3.5-0.8B"}}
    cases.append(m)  # AC6

    m = _valid_mapping()
    m["editions"]["normal"]["base"] = "gemma-2b"
    cases.append(m)  # AC7

    m = _valid_mapping()
    m["sources"]["sense"]["type"] = "url"
    cases.append(m)  # AC8

    m = _valid_mapping()
    m["training"]["method"] = "full"
    cases.append(m)  # AC9

    m = _valid_mapping()
    m["training"]["lora_rank"] = 0
    cases.append(m)  # R8-AC9

    return cases


@pytest.mark.parametrize("data", _broken_mappings())
def test_failure_suppresses_success_confirmation(data, capsys):
    with pytest.raises(ConfigError):
        validate(data, CONFIG_PATH)
    captured = capsys.readouterr()
    assert "Config validated" not in captured.out


# ---------------------------------------------------------------------------
# R3-AC2 / AC3 / AC4：加载与解析错误（由 loader 负责，完整覆盖 R3 错误条款）
# ---------------------------------------------------------------------------


def test_ac2_config_file_not_found(tmp_path):
    missing = tmp_path / "ego-config.yaml"
    with pytest.raises(ConfigError) as exc:
        load(missing)
    assert "ego init" in str(exc.value)
    assert exc.value.exit_code == EXIT_USAGE_OR_CONFIG


def test_ac3_io_error_when_path_is_directory(tmp_path):
    # 路径存在但为目录 → 读取阶段抛 IsADirectoryError（OSError）→ ConfigError。
    target = tmp_path / "ego-config.yaml"
    target.mkdir()
    with pytest.raises(ConfigError) as exc:
        load(target)
    assert exc.value.exit_code == EXIT_USAGE_OR_CONFIG


@pytest.mark.skipif(os.geteuid() == 0, reason="root 会绕过文件权限检查")
def test_ac3_permission_error(tmp_path):
    target = tmp_path / "ego-config.yaml"
    target.write_text("sources: {}\n", encoding="utf-8")
    target.chmod(0o000)
    try:
        with pytest.raises(ConfigError) as exc:
            load(target)
        assert exc.value.exit_code == EXIT_USAGE_OR_CONFIG
    finally:
        target.chmod(0o600)  # 还原以便清理


def test_ac4_invalid_yaml_reports_line_and_column(tmp_path):
    target = tmp_path / "ego-config.yaml"
    # 非法 YAML：未闭合的流式映射会触发带位置标记的解析错误。
    target.write_text("sources: {unclosed: [1, 2\n", encoding="utf-8")
    with pytest.raises(ConfigError) as exc:
        load(target)
    message = str(exc.value)
    assert exc.value.exit_code == EXIT_USAGE_OR_CONFIG
    assert "行" in message and "列" in message
