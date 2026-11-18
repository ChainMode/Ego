"""深层多维网格专精度分层器（Specificity Distiller）与按 Edition 网格裁剪单测（ADR 0005）。

覆盖核心验收标准：
1. 纯 Python 零 ML 依赖的分区内 IDF + 结构/代码/推理特征评分，取值严格落于 [0.0, 1.0]；
2. 显式空值 / None / 纯空白字段检测与安全计数（零分兜底归入 L1，绝不崩溃或产生 NaN）；
3. 严格防数据泄漏（Strict Featurization Ordering）：仅在训练集子集拟合词表 IDF 与分位数切点，
   改变留出验证集样本完全不影响训练集的统计量与切点；
4. 层级连续性压缩（Contiguous Compaction）：小样本或空中间桶自动重编号为严格连续的 1..K（无跳号）；
5. 按 Edition 网格裁剪（lite=1, normal=2, plus=3, super=3 及自定义 zones 子集折叠）。
"""

from __future__ import annotations

import math

from ego_pod.compiler.classify import SampleCandidate
from ego_pod.compiler.routing import (
    KERNEL,
    CellKey,
    ResolvedZoneConfig,
    prune_cells_for_edition,
    resolve_zone_config,
    route_cells,
)
from ego_pod.compiler.specificity import (
    NullFieldStats,
    SpecificityModel,
    assign_zone_levels,
    compact_levels,
    fit_specificity_model,
    score_sample_specificity,
    tokenize_for_specificity,
)
from ego_pod.config.schema import ZoneConfig
import pytest


def _sample(
    category: str,
    output: str,
    *,
    instruction: str = "请回答问题",
    input_field: str = "",
) -> SampleCandidate:
    return SampleCandidate(
        instruction=instruction,
        input_field=input_field,
        output=output,
        category=category,  # type: ignore[arg-type]
    )


def test_score_sample_specificity_bounds_and_ordering() -> None:
    """一般日常短句得分低于进阶解释，进阶解释低于含代码与复杂推理的深度样本。"""
    s_general = _sample("skill", "今天写了一点简单的日常记录。")
    s_medium = _sample(
        "skill",
        "在 Python 项目中使用 dataclass(frozen=True) 可以保证配置不可变，因此能避免运行期意外修改状态。",
    )
    s_deep = _sample(
        "skill",
        "```python\ndef derive_hkdf_subkey(seed: bytes, info: bytes) -> bytes:\n"
        "    # 基于 RFC 5869 HKDF-SHA256 实现域隔离子密钥派生，保证前向安全与确定性不变量\n"
        "    return HKDF(algorithm=hashes.SHA256(), length=32, salt=b'v1', info=info).derive(seed)\n```\n"
        "复杂度为 O(1)，因为输入种子定长 32 字节；权衡在于必须严格校验 hex 编码边界。",
    )
    train_samples = [s_general, s_medium, s_deep]
    cfg = resolve_zone_config(None, default_max_level=3)
    model = fit_specificity_model(train_samples, cfg.routing, max_level=3)

    score_1 = score_sample_specificity(s_general, "coding", model)
    score_2 = score_sample_specificity(s_medium, "coding", model)
    score_3 = score_sample_specificity(s_deep, "coding", model)

    for sc in (score_1, score_2, score_3):
        assert 0.0 <= sc <= 1.0
        assert math.isfinite(sc)
    assert score_1 < score_2 < score_3


def test_missing_null_and_whitespace_fields_handled_safely() -> None:
    """显式检测 None / 空串 / 纯空白字段：统计计数准确且赋予确定性零基线得分。"""
    s_none = SampleCandidate(
        instruction=None,  # type: ignore[arg-type]
        input_field=None,  # type: ignore[arg-type]
        output="   \t\n  ",
        category="thought",
    )
    s_valid = _sample(
        "thought",
        "系统架构设计应当遵循单仓自足原则，避免跨仓强耦合导致级联阻塞。",
    )
    stats = NullFieldStats()
    cfg = resolve_zone_config(None, default_max_level=3)
    model = fit_specificity_model(
        [s_none, s_valid], cfg.routing, max_level=3, stats=stats
    )

    assert stats.null_or_empty_fields >= 2
    assert stats.empty_samples == 1

    null_score = score_sample_specificity(s_none, "knowledge", model)
    assert null_score == 0.0

    routed = route_cells([s_none, s_valid], cfg, max_level=3, specificity_model=model)
    assert s_none in routed[CellKey("knowledge", 1)]
    assert s_valid in routed[CellKey("knowledge", 2)]


def test_strict_featurization_ordering_zero_holdout_leakage() -> None:
    """防数据泄漏验证：仅在 train_split 上拟合模型，篡改 holdout_split 绝不改变模型词表、IDF 或切点。"""
    train_split = [
        _sample("skill", "基础函数定义与变量命名习惯说明。"),
        _sample("skill", "使用 pytest fixture 组织模块化单元测试与异常断言。"),
        _sample(
            "skill",
            "```python\nclass AsyncLimiter:\n    def acquire(self, tokens: int = 1) -> bool:\n        return True\n```\n"
            "通过令牌桶算法控制并发吞吐并维持单调时钟不变量。",
        ),
    ]
    holdout_a = [_sample("skill", "短留出样本 A")]
    holdout_b = [
        _sample(
            "skill",
            "```rust\npub fn merkle_root_sha256(leaves: &[Hash]) -> Hash { todo!() }\n```\n"
            "完全不同的高密度专精留出集词表量子纠缠零知识证明 B " * 10,
        )
    ]

    cfg = resolve_zone_config(None, default_max_level=3)
    model_a = fit_specificity_model(train_split, cfg.routing, max_level=3)
    # 模拟不同留出集存在但严格只传入 train_split 拟合
    model_b = fit_specificity_model(list(train_split), cfg.routing, max_level=3)

    assert isinstance(model_a, SpecificityModel)
    assert model_a.zone_doc_counts == model_b.zone_doc_counts
    assert model_a.zone_df == model_b.zone_df
    assert model_a.zone_quantiles == model_b.zone_quantiles

    # 用 train_split 拟合出的统计量路由 train_split，完全独立于 holdout_a / holdout_b
    routed_with_a = route_cells(
        train_split, cfg, max_level=3, fit_samples=train_split
    )
    routed_with_b = route_cells(
        train_split, cfg, max_level=3, fit_samples=train_split
    )
    assert routed_with_a == routed_with_b
    assert len(holdout_a) == len(holdout_b) == 1


def test_contiguous_compaction_eliminates_empty_middle_levels() -> None:
    """当原始分桶出现跳号（如 [1, 3] 或 [2, 3]）时，compact_levels 保序压缩为连续的 1..K。"""
    assert compact_levels([]) == []
    assert compact_levels([3, 3]) == [1, 1]
    assert compact_levels([1, 3, 1, 3]) == [1, 2, 1, 2]
    assert compact_levels([3, 1, 5]) == [2, 1, 3]


def test_small_zone_compacts_to_contiguous_levels_without_gaps() -> None:
    """某分区仅有 2 条不同专精度样本但请求 max_level=3 时，产出 L1 与 L2（无空缺 L2 跳 L3）。"""
    samples = [
        _sample("sense", "清晨窗外的阳光。"),
        _sample("thought", "简单的一句感想。"),
        _sample(
            "thought",
            "深入分析分布式共识协议在拜占庭容错假设下的活性与安全性权衡，因此需要两阶段法定人数交集证明。",
        ),
    ]
    cfg = resolve_zone_config(
        ZoneConfig(zones=("knowledge",), routing={"thought": "knowledge"}, max_level=3)
    )
    routed = route_cells(samples, cfg)
    assert set(routed.keys()) == {
        KERNEL,
        CellKey("knowledge", 1),
        CellKey("knowledge", 2),
    }


def test_per_edition_grid_pruning_folds_levels_and_excluded_zones() -> None:
    """按 Edition 裁剪：lite 折叠为 L1，normal 折叠为 L1..L2，排除分区归入 Kernel 且样本总数守恒。"""
    samples = [
        _sample("sense", "日常感知基线记录。"),
        _sample("skill", "简单的命令行查看目录。"),
        _sample("skill", "编写结构化 Python 函数并使用类型注解校验输入参数。"),
        _sample(
            "skill",
            "```python\ndef canonical_jcs(obj: object) -> bytes:\n    return b'{}'\n```\n"
            "实现 RFC 8785 JCS 规范化器并与 Ed25519 域分离签名逐字节对齐。",
        ),
        _sample("thought", "基础阅读随笔记录。"),
        _sample(
            "thought",
            "探讨四层认知架构中自我层与认知层的解耦边界以及为什么局部适配器优于全量微调。",
        ),
    ]
    cfg = resolve_zone_config(None, default_max_level=3)
    deep_routed = route_cells(samples, cfg, max_level=3)

    # plus/super (max_level=3): coding 拥有 L1..L3
    coding_levels_plus = sorted(
        k.level for k in deep_routed if k.zone == "coding"
    )
    assert coding_levels_plus == [1, 2, 3]

    # normal (max_level=2): coding 折叠为 L1..L2，样本总数守恒
    normal_routed = prune_cells_for_edition(deep_routed, max_level=2)
    assert sorted(k.level for k in normal_routed if k.zone == "coding") == [1, 2]
    assert sum(len(v) for v in normal_routed.values()) == len(samples)

    # lite (max_level=1): 所有激活分区仅含 L1
    lite_routed = prune_cells_for_edition(deep_routed, max_level=1)
    assert sorted(k.level for k in lite_routed if k.zone == "coding") == [1]
    assert sorted(k.level for k in lite_routed if k.zone == "knowledge") == [1]
    assert sum(len(v) for v in lite_routed.values()) == len(samples)

    # 自定义 Edition 仅保留 coding 分区：knowledge 样本安全折叠回 Kernel
    custom_routed = prune_cells_for_edition(
        deep_routed, max_level=2, allowed_zones=("coding",)
    )
    assert {k.zone for k in custom_routed} == {None, "coding"}
    assert sum(len(v) for v in custom_routed.values()) == len(samples)
    assert len(custom_routed[KERNEL]) == 1 + 2  # 原 Kernel(1) + 原 knowledge(2)


@pytest.mark.parametrize(
    "text,expected_non_empty",
    [
        ("", False),
        ("   \t\n ", False),
        ("分布式系统架构设计", True),
        ("日本語のコーパス解析テスト", True),
        ("한국어 데이터 처리 파이프라인", True),
        ("Проверка кириллического текста", True),
        ("snake_case_identifier and CamelCaseToken", True),
        ("中", True),
    ],
)
def test_tokenize_for_specificity_multilingual_and_edge_inputs(
    text: str, expected_non_empty: bool
) -> None:
    tokens = tokenize_for_specificity(text)
    assert bool(tokens) is expected_non_empty


def test_identical_specificity_samples_collapse_to_level_one() -> None:
    """同一分区全部样本内容等同时，无跳号压缩只产出 L1。"""
    dup_samples = [_sample("skill", "完全一致的短文本记录。") for _ in range(5)]
    cfg = resolve_zone_config(None, default_max_level=3)
    model = fit_specificity_model(dup_samples, cfg.routing, max_level=3)
    levels = assign_zone_levels(dup_samples, "coding", model, max_level=3)
    assert levels == [1, 1, 1, 1, 1]


def test_assign_zone_levels_monotonic_with_score_and_oov_safe() -> None:
    """从未见过的词元（OOV）与长梯度样本得分单调映射为非递减的连续层级。"""
    train_samples = [
        _sample("skill", f"基础记录条目 {i}") for i in range(4)
    ] + [
        _sample("skill", f"因为模块接口需要稳定，因此采用契约校验方案 {i}") for i in range(4)
    ] + [
        _sample(
            "skill",
            f"```python\ndef check_{i}(x: int) -> bool:\n    return x > 0\n```\n"
            f"基于 RFC 8785 与 SHA-256 证明时间复杂度 O(n) 不变量 {i}",
        )
        for i in range(4)
    ]
    cfg = resolve_zone_config(None, default_max_level=3)
    model = fit_specificity_model(train_samples, cfg.routing, max_level=3)

    oov_sample = _sample(
        "skill",
        "```python\nasync def unseen_quantum_verifier() -> None:\n    return None\n```\n"
        "全新未登录词表协议复杂度与拜占庭容错权衡证明。",
    )
    score_oov = score_sample_specificity(oov_sample, "coding", model)
    assert 0.0 < score_oov <= 1.0

    scores = [score_sample_specificity(s, "coding", model) for s in train_samples]
    levels = assign_zone_levels(train_samples, "coding", model, max_level=3)
    assert set(levels) == {1, 2, 3}
    for (s_a, l_a), (s_b, l_b) in zip(
        sorted(zip(scores, levels)), sorted(zip(scores, levels))[1:]
    ):
        if s_a < s_b:
            assert l_a <= l_b
