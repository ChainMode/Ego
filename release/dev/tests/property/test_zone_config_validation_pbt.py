"""Zone_Config 校验的属性测试（hypothesis）。

Feature: ego-layered-adapter, Property 5

本文件实现 design.md「Correctness Properties」中的 **Property 5: Zone_Config
校验**，对应 tasks.md 任务 1.4：

    *For any* 随机 Zone_Config：若其包含任何不匹配 ``^[a-z0-9-]+$`` 的 Zone_Name、
    或包含重复的 Zone_Name、或 ``routing`` 的某个值指向未声明的分区，则配置校验
    抛出 ``ConfigError``（退出码 2），消息含首个违规的 Zone_Name 或具体原因；若全部
    Zone_Name 合法且唯一、且 ``routing`` 取值均指向已声明分区（``zones`` 块缺省亦视为
    合法），则校验在 Zone_Config 这一环节通过。

测试在**无 ML 依赖**的最小环境（``pytest`` + ``hypothesis``）下运行：
:func:`ego_pod.config.validators.validate` 是纯 Python 校验，作用于 ``loader``
解析得到的**原始映射**（而非已构造的 ``EgoConfig``），因此本文件直接以 dict
构造输入——以一份通过全部前置校验（R3/R8-AC9）的合法基底映射打底，仅改变其
``zones`` 子块，使校验恰好推进到 ``_check_zone_config`` 这一步。

iff（充要）方向由 :func:`_expected_first_offender` 精确镜像校验器的判定与短路
顺序得到：先按声明顺序检测 Zone_Name 正则与重复，再按插入顺序检测 ``routing``
取值是否指向已声明分区。注意 ``routing`` 中**不属于** Data_Category 五元集的键
仅产生（非致命）告警而不致校验失败，故这类键不被计入违规。

**Validates: Requirements 1.3, 1.10**
"""

from __future__ import annotations

import io
import re
import string
from typing import Any

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from ego_pod.config.validators import validate
from ego_pod.exit_codes import EXIT_USAGE_OR_CONFIG, ConfigError

# ---------------------------------------------------------------------------
# 公共常量与合法基底映射
# ---------------------------------------------------------------------------

CONFIG_PATH = "ego-config.yaml"

#: 与 ``validators._check_zone_config`` 完全一致的 Zone_Name 取值规则（R1-AC3）。
_ZONE_NAME_PATTERN = re.compile(r"^[a-z0-9-]+$")

#: 固定五元 Data_Category 集合；``routing`` 中落在集合外的键仅告警、不致命。
_DATA_CATEGORIES = ["sense", "action", "thought", "skill", "credentials"]


def _valid_mapping() -> dict:
    """返回一份通过全部 R3 / R8-AC9 前置校验的合法原始配置映射。

    仅其 ``zones`` 子块在各属性中被替换/移除，从而使 :func:`validate` 恰好推进到
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


# ---------------------------------------------------------------------------
# 期望判定：精确镜像 _check_zone_config 的判定规则与短路顺序
# ---------------------------------------------------------------------------


def _expected_first_offender(zones_block: Any) -> tuple[str, Any] | None:
    """返回校验器应当报告的首个违规 ``(kind, name)``，或 ``None`` 表示合法。

    ``kind`` ∈ {``"regex"``, ``"dup"``, ``"routing"``}，与 ``validators``
    模块 ``_check_zone_config`` 的检测顺序（声明顺序的正则/去重 → 插入顺序的
    routing 取值合法性）逐字对应；``zones`` 块缺省（``None`` 或非映射）⇒ 合法。
    """

    if not isinstance(zones_block, dict):
        return None

    declared_raw = zones_block.get("zones")
    declared = list(declared_raw) if isinstance(declared_raw, (list, tuple)) else []

    seen: set[str] = set()
    for name in declared:
        if not isinstance(name, str) or not _ZONE_NAME_PATTERN.fullmatch(name):
            return ("regex", name)
        if name in seen:
            return ("dup", name)
        seen.add(name)

    routing = zones_block.get("routing")
    if isinstance(routing, dict):
        for _category, target in routing.items():
            if target not in seen:
                return ("routing", target)

    return None


# ---------------------------------------------------------------------------
# 生成器：在合法/非法 Zone_Name 与 routing 取值上取值
# ---------------------------------------------------------------------------

#: 合法 Zone_Name 字符集（与 ``^[a-z0-9-]+$`` 一致）。
_VALID_ZONE_ALPHABET = string.ascii_lowercase + string.digits + "-"
_valid_zone_name = st.text(alphabet=_VALID_ZONE_ALPHABET, min_size=1, max_size=6)

#: 非法 Zone_Name：在更宽字符集（含大写、下划线、空格、点、空串）上取值后，
#: 过滤出不完整匹配正则的串——它们必然触发正则违规。
_BROAD_ALPHABET = string.ascii_letters + string.digits + "-_ ."
_invalid_zone_name = st.text(
    alphabet=_BROAD_ALPHABET, min_size=0, max_size=6
).filter(lambda s: not _ZONE_NAME_PATTERN.fullmatch(s))

#: 小型合法分区名池：用 ``sampled_from`` 取值以提高生成重复名（去重违规）的概率。
_SMALL_POOL = ["coding", "knowledge", "social", "a", "b1", "x-y"]

#: 单个声明分区名：合法池 / 任意合法 / 任意非法的混合，覆盖正则与去重两类违规。
_declared_elem = st.one_of(
    st.sampled_from(_SMALL_POOL),
    _valid_zone_name,
    _invalid_zone_name,
)

#: routing 键：五元 Data_Category（命中样本）或任意小写串（不命中、仅告警）。
_routing_key = st.one_of(
    st.sampled_from(_DATA_CATEGORIES),
    st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=5),
)


@st.composite
def _zones_block(draw: st.DrawFn) -> dict:
    """生成一个 ``zones`` 子块：声明分区列表 + routing 映射。

    routing 取值在"任意合法 Zone_Name"与"恰好取自已声明分区"之间混合，使
    routing 既可能指向已声明分区（合法）也可能指向未声明分区（违规），从而充分
    行使 Property 5 的两个方向。
    """

    declared = draw(st.lists(_declared_elem, min_size=0, max_size=5))
    valid_targets = [n for n in declared if isinstance(n, str) and n]

    value_strategies = [_valid_zone_name]
    if valid_targets:
        value_strategies.append(st.sampled_from(valid_targets))

    routing = draw(
        st.dictionaries(_routing_key, st.one_of(*value_strategies), max_size=4)
    )
    return {"zones": declared, "routing": routing}


#: 完整输入空间：``None``（``zones`` 块缺省 ⇒ 合法）或一个生成的 ``zones`` 子块。
_zones_block_or_none = st.one_of(st.none(), _zones_block())


# ===========================================================================
# Property 5: Zone_Config 校验（充要 / iff）
# ===========================================================================


@given(zones_block=_zones_block_or_none)
@settings(max_examples=200, deadline=None)
# 显式锁定五类代表性输入，保证两个方向均被确定性覆盖。
@example(zones_block=None)  # zones 块缺省 ⇒ 合法
@example(  # 全部合法且唯一、routing 取值均指向已声明分区 ⇒ 合法
    zones_block={
        "zones": ["coding", "knowledge"],
        "routing": {"skill": "coding", "thought": "knowledge"},
    }
)
@example(zones_block={"zones": ["Bad"], "routing": {}})  # 正则违规
@example(zones_block={"zones": ["a", "a"], "routing": {}})  # 去重违规
@example(  # routing 取值指向未声明分区
    zones_block={"zones": ["coding"], "routing": {"skill": "undeclared"}}
)
def test_zone_config_validation_property(zones_block: Any) -> None:
    """``validate`` 在 Zone_Config 这一环节抛 ``ConfigError`` 当且仅当 ``zones``
    块非法（非法 Zone_Name / 重复 Zone_Name / routing 指向未声明分区）。

    合法（含 ``zones`` 缺省）时不在 Zone_Config 这一环节抛错；非法时以退出码 2
    抛 :class:`ConfigError`，且消息包含首个违规的 Zone_Name 或具体原因。

    **Validates: Requirements 1.3, 1.10**
    """

    data = _valid_mapping()
    # ``zones=None`` 表示该块缺省（与键不存在同义，校验器据此跳过并采用默认基线）。
    data["zones"] = zones_block

    offender = _expected_first_offender(zones_block)

    if offender is None:
        # 合法：校验在 Zone_Config 环节不得抛错（成功确认信息导入临时流以静默）。
        validate(data, CONFIG_PATH, stream=io.StringIO())
        return

    # 非法：以退出码 2 抛 ConfigError。
    with pytest.raises(ConfigError) as exc:
        validate(data, CONFIG_PATH, stream=io.StringIO())

    assert exc.value.exit_code == EXIT_USAGE_OR_CONFIG

    # 消息含首个违规名 / 原因，且与校验器的格式化方式（正则/routing 用 ``repr``、
    # 去重用裸名）一致。
    kind, name = offender
    message = str(exc.value)
    if kind == "regex":
        assert repr(name) in message
    elif kind == "dup":
        assert name in message
    else:  # routing
        assert repr(name) in message
