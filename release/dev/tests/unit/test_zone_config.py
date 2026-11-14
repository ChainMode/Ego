"""Zone_Config 加载与校验边界的单元测试（任务 1.6）。

本文件以**具体示例/边界**补充 ``tests/property/test_zone_config_validation_pbt.py``
中的 Property 5（充要性属性测试）：属性测试以 hypothesis 在随机输入空间上锁定
"非法 ⇔ 抛错"的整体性质，本文件则针对**加载侧**（``loader.load`` 把 YAML 顶层
``zones:`` 块宽松构造为 :class:`~ego_pod.config.schema.ZoneConfig`）与**校验侧**
（``validators.validate`` 推进到 ``_check_zone_config`` 的若干代表性边界）给出确定性
单测，覆盖以下需求条款：

- R1-AC8：未声明 ``zones:`` ⇒ ``EgoConfig.zones is None``（下游回退默认基线）。
- R1-AC9：声明 ``zones:`` ⇒ ``EgoConfig.zones`` 为含声明分区元组与 routing 字典的
  :class:`ZoneConfig`；宽松构造下缺失子字段（如无 ``routing:``）不在加载期报错。
- R1-AC10：非法 Zone_Name（不匹配 ``^[a-z0-9-]+$``）/ 重复 Zone_Name / routing 取值
  指向未声明分区 ⇒ :class:`ConfigError`、退出码 2，且消息含首个违规名；退出码语义
  为"用法/配置错误"（2），**不**映射到运行期失败（1）。

加载侧测试把 YAML 写入 ``tmp_path`` 后调用 ``loader.load``；校验侧测试复用
``test_config_validators.py`` 的合法基底映射模式，仅替换其 ``zones`` 子块，使
``validate`` 恰好推进到 ``_check_zone_config``。
"""

from __future__ import annotations

import io

import pytest

from ego_pod.config.loader import load
from ego_pod.config.schema import EgoConfig, ZoneConfig
from ego_pod.config.validators import validate
from ego_pod.exit_codes import (
    EXIT_RUNTIME_FAILURE,
    EXIT_USAGE_OR_CONFIG,
    ConfigError,
)

CONFIG_NAME = "ego-config.yaml"
CONFIG_PATH = "ego-config.yaml"


# ---------------------------------------------------------------------------
# 测试夹具：合法基底（YAML 文本 / 原始映射）
# ---------------------------------------------------------------------------

#: 一份不含 ``zones:`` 块、可被 loader 宽松构造的最小有效 YAML 文本。
_BASE_YAML = """\
sources:
  sense:
    type: local
    path: ./data/sense
  action:
    type: local
    path: ./data/action
  thought:
    type: local
    path: ./data/thought
  skill:
    type: local
    path: ./data/skill
  credentials:
    type: local
    path: ./data/credentials
training:
  method: qlora
  quantization: 4bit
  lora_rank: 16
  lora_alpha: 32
  learning_rate: 0.0002
  epochs: 3
  batch_size: 4
editions:
  normal:
    base: qwen3.5-0.8B
    description: ""
output: {}
"""


def _write_config(tmp_path, body: str):
    """把 ``_BASE_YAML`` + ``body``（追加的 ``zones:`` 块等）写入 tmp_path 并返回路径。"""

    target = tmp_path / CONFIG_NAME
    target.write_text(_BASE_YAML + body, encoding="utf-8")
    return target


def _valid_mapping() -> dict:
    """返回一份通过全部 R3 / R8-AC9 前置校验的合法原始配置映射。

    仅其 ``zones`` 子块在各测试中被替换/移除，从而使 :func:`validate` 恰好推进到
    ``_check_zone_config`` 这一步并对 Zone_Config 施加校验。
    """

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


def _validate_silent(data: dict) -> None:
    """调用 ``validate`` 并把成功确认信息导入临时流以静默（仅关注是否抛错）。"""

    validate(data, CONFIG_PATH, stream=io.StringIO())


# ===========================================================================
# Loader：zones 块的宽松构造（R1-AC8 / R1-AC9）
# ===========================================================================


def test_load_without_zones_block_yields_none(tmp_path):
    # R1-AC8：YAML 不含 zones: ⇒ EgoConfig.zones is None（下游回退默认基线）。
    target = _write_config(tmp_path, "")
    config, abs_path = load(target)
    assert isinstance(config, EgoConfig)
    assert config.zones is None


def test_load_with_zones_block_builds_zone_config(tmp_path):
    # R1-AC9：声明 zones: ⇒ EgoConfig.zones 为 ZoneConfig，含声明分区元组与 routing 字典。
    body = """\
zones:
  zones: [coding, knowledge]
  routing:
    skill: coding
    action: coding
    thought: knowledge
"""
    target = _write_config(tmp_path, body)
    config, _ = load(target)

    assert isinstance(config.zones, ZoneConfig)
    # 声明分区保持声明顺序并以 tuple 表达。
    assert config.zones.zones == ("coding", "knowledge")
    # routing 为 Data_Category → Zone_Name 的字典。
    assert config.zones.routing == {
        "skill": "coding",
        "action": "coding",
        "thought": "knowledge",
    }


def test_load_zones_block_missing_routing_defaults_to_empty(tmp_path):
    # 宽松构造：zones: 块只有 zones 列表、无 routing ⇒ routing 默认空 dict，加载期不报错。
    body = """\
zones:
  zones: [coding, knowledge]
"""
    target = _write_config(tmp_path, body)
    config, _ = load(target)

    assert isinstance(config.zones, ZoneConfig)
    assert config.zones.zones == ("coding", "knowledge")
    assert config.zones.routing == {}


def test_load_zones_block_missing_zones_list_defaults_to_empty_tuple(tmp_path):
    # 宽松构造：zones: 块只有 routing、无 zones 列表 ⇒ zones 默认空 tuple，加载期不报错。
    body = """\
zones:
  routing:
    skill: coding
"""
    target = _write_config(tmp_path, body)
    config, _ = load(target)

    assert isinstance(config.zones, ZoneConfig)
    assert config.zones.zones == ()
    assert config.zones.routing == {"skill": "coding"}


def test_load_empty_zones_block_builds_empty_zone_config(tmp_path):
    # zones: 块整体为空值（解析为 None）⇒ 视为缺省 ⇒ zones is None（R1-AC8）。
    body = "zones:\n"
    target = _write_config(tmp_path, body)
    config, _ = load(target)
    assert config.zones is None


# ===========================================================================
# Validator：zones 缺省与合法配置（R1-AC8 / R1-AC9）
# ===========================================================================


def test_validate_absent_zones_block_no_error():
    # R1-AC8：缺省 zones 块 ⇒ 跳过 Zone_Config 校验，不抛错。
    data = _valid_mapping()
    assert "zones" not in data
    _validate_silent(data)  # 不应抛错


def test_validate_zones_none_no_error():
    # zones 显式为 None（YAML 空块）⇒ 同样视为缺省，不抛错。
    data = _valid_mapping()
    data["zones"] = None
    _validate_silent(data)


def test_validate_valid_zones_and_routing_no_error():
    # 全部 Zone_Name 合法且唯一、routing 取值均指向已声明分区 ⇒ 不抛错。
    data = _valid_mapping()
    data["zones"] = {
        "zones": ["coding", "knowledge"],
        "routing": {"skill": "coding", "action": "coding", "thought": "knowledge"},
    }
    _validate_silent(data)


# ===========================================================================
# Validator：非法 Zone_Name（R1-AC10）
# ===========================================================================


@pytest.mark.parametrize("bad_name", ["Bad", "a_b", "", "x y"])
def test_validate_invalid_zone_name_raises_config_error(bad_name):
    # R1-AC10：首个不匹配 ^[a-z0-9-]+$ 的 Zone_Name ⇒ ConfigError，退出码 2，消息含违规名。
    data = _valid_mapping()
    data["zones"] = {"zones": [bad_name], "routing": {}}
    with pytest.raises(ConfigError) as exc:
        _validate_silent(data)
    assert exc.value.exit_code == EXIT_USAGE_OR_CONFIG
    # 退出码为"用法/配置错误"(2)，绝不映射到运行期失败(1)。
    assert exc.value.exit_code != EXIT_RUNTIME_FAILURE
    # 消息含违规 Zone_Name（校验器以 repr 格式化）。
    assert repr(bad_name) in str(exc.value)


def test_validate_reports_first_invalid_zone_name():
    # 多个非法名时，按声明顺序报告首个（"Bad" 在 "also_bad" 之前）。
    data = _valid_mapping()
    data["zones"] = {"zones": ["Bad", "also_bad"], "routing": {}}
    with pytest.raises(ConfigError) as exc:
        _validate_silent(data)
    message = str(exc.value)
    assert repr("Bad") in message
    assert repr("also_bad") not in message


# ===========================================================================
# Validator：重复 Zone_Name（R1-AC10）
# ===========================================================================


def test_validate_duplicate_zone_name_raises_config_error():
    # R1-AC10：重复 Zone_Name ⇒ ConfigError，退出码 2，消息含首个重复名。
    data = _valid_mapping()
    data["zones"] = {"zones": ["coding", "coding"], "routing": {}}
    with pytest.raises(ConfigError) as exc:
        _validate_silent(data)
    assert exc.value.exit_code == EXIT_USAGE_OR_CONFIG
    assert "coding" in str(exc.value)


def test_validate_reports_first_duplicate_name():
    # 多组重复时报告首个出现的重复名（knowledge 先于 coding 第二次出现）。
    data = _valid_mapping()
    data["zones"] = {
        "zones": ["knowledge", "coding", "knowledge", "coding"],
        "routing": {},
    }
    with pytest.raises(ConfigError) as exc:
        _validate_silent(data)
    assert "knowledge" in str(exc.value)


# ===========================================================================
# Validator：routing 取值指向未声明分区（R1-AC10）
# ===========================================================================


def test_validate_routing_to_undeclared_zone_raises_config_error():
    # R1-AC10：routing 值指向未声明 Zone ⇒ ConfigError，退出码 2，消息含未声明的目标名。
    data = _valid_mapping()
    data["zones"] = {"zones": ["coding"], "routing": {"skill": "undeclared"}}
    with pytest.raises(ConfigError) as exc:
        _validate_silent(data)
    assert exc.value.exit_code == EXIT_USAGE_OR_CONFIG
    assert exc.value.exit_code != EXIT_RUNTIME_FAILURE
    assert repr("undeclared") in str(exc.value)


# ===========================================================================
# Validator：routing 键不在 Data_Category 五元集 ⇒ 非致命（仅告警）
# ===========================================================================


def test_validate_routing_key_outside_data_categories_is_non_fatal(capsys):
    # routing 键不属于 Data_Category 五元集，但其值指向已声明分区 ⇒ 不因该键抛错。
    data = _valid_mapping()
    data["zones"] = {
        "zones": ["coding"],
        # "notacategory" 不在 {sense,action,thought,skill,credentials} 中 ⇒ 永不命中样本。
        "routing": {"notacategory": "coding"},
    }
    # 不应因该越界键而抛 ConfigError（仅可能向 stderr 打印告警）。
    validate(data, CONFIG_PATH, stream=io.StringIO())
    # 该越界键以告警形式出现在 stderr（非致命）。
    captured = capsys.readouterr()
    assert "notacategory" in captured.err


def test_validate_routing_key_outside_categories_with_valid_target_succeeds():
    # 越界键 + 指向已声明分区 ⇒ 校验整体通过（越界键不计入违规）。
    data = _valid_mapping()
    data["zones"] = {
        "zones": ["coding", "knowledge"],
        "routing": {"skill": "coding", "weird_key": "knowledge"},
    }
    # 仅 weird_key 越界但其值合法 ⇒ 不抛错。
    _validate_silent(data)
