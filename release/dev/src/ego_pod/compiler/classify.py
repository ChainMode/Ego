"""Compiler 模板分类 — 数据 → instruction/output 标准化。

本模块实现语料构建管线中"标准化与分类"子阶段，对应需求 R6。它把来源样本
（已脱敏、已规范化）转换为统一的对话格式候选样本
（:class:`SampleCandidate`），并在此过程中执行三条丢弃规则与字段长度约束。

转换规则（design.md §4.3）：

* **已具备对话结构的来源样本**（即上游 reader 已能将其直接映射为
  instruction/output 对，例如问答对、对话轮次或等价结构）—— **不**查找、
  **不**应用任何 Data_Category 模板，直接把其 instruction 与 output 映射到
  输出样本（R6-AC3 后半句）。
* **非对话结构的来源样本** —— 按其 Data_Category 从 :data:`TEMPLATES` 中
  选取对应模板，将原文转换为 instruction/output 对（R6-AC3 前半句）。

丢弃路径：

* **R6-AC6（任一字段为空即丢弃）**：标准化后只要 ``instruction`` 字段为
  空字符串、``output`` 字段为空字符串、或两者同时为空，样本即被丢弃；
  字段超长先截断到 :data:`MAX_FIELD_LEN`（32768）**再**判空。
* **R6-AC7（仅模板缺失触发）**：当且仅当一条 *非对话结构* 样本的
  Data_Category 在 :data:`TEMPLATES` 中无对应条目时才走"模板缺失丢弃"
  路径；脱敏、规范化、过滤等其它阶段的处理错误**不**触发本路径；本路径
  **不**适用于已具备对话结构的样本。
* **R6-AC8**：样本的 ``category`` 在标准化（小写化）后取值不属于
  ``{sense, action, thought, skill, credentials}`` 时丢弃。

类别透明性（Property 8）：若来源样本的 Data_Category 属于五元集，则被保留
的输出样本的 ``category`` 字段严格等于该值的小写形式（R6-AC1/AC2）。

与 :mod:`ego_pod.compiler.redact` 一致，本模块为**纯函数**实现：不直接写
标准输出/错误流，而是把丢弃计数与原因明细收集进 :class:`ClassifyStats`，
由 Compiler ``pipeline.py`` 负责按 R6-AC6/AC7/AC8 输出到 stderr。
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Final, Literal, Optional, cast

__all__ = [
    "VALID_CATEGORIES",
    "MAX_FIELD_LEN",
    "TEXT_PLACEHOLDER",
    "Category",
    "Template",
    "TEMPLATES",
    "SourceSample",
    "SampleCandidate",
    "DiscardReason",
    "ClassifyResult",
    "ClassifyStats",
    "truncate_field",
    "classify_sample",
    "classify_samples",
]

# ---------------------------------------------------------------------------
# 常量与类型
# ---------------------------------------------------------------------------

#: 合法 category 取值（需求 R6-AC1/AC2/AC8 的五元集，全小写）。
VALID_CATEGORIES: Final[frozenset[str]] = frozenset(
    {"sense", "action", "thought", "skill", "credentials"}
)

#: 单个字符串字段的长度上限（需求 R6-AC1）。超出则截断到此长度后再判空。
MAX_FIELD_LEN: Final[int] = 32768

#: 模板 output 串中代表"原文"的占位符。
TEXT_PLACEHOLDER: Final[str] = "{text}"

#: ``category`` 字段的字面量类型（与 ``config.schema`` 中的取值集合一致）。
Category = Literal["sense", "action", "thought", "skill", "credentials"]


# ---------------------------------------------------------------------------
# 模板表（需求 R6-AC3）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Template:
    """一条 Data_Category 模板：固定 instruction + 含占位符的 output 模板。

    ``output_template`` 中的 :data:`TEXT_PLACEHOLDER`（``{text}``）会在
    :meth:`render` 时被替换为来源样本原文。
    """

    instruction: str
    output_template: str = TEXT_PLACEHOLDER

    def render(self, text: str) -> tuple[str, str]:
        """用原文 ``text`` 渲染模板，返回 ``(instruction, output)`` 二元组。"""
        return self.instruction, self.output_template.replace(TEXT_PLACEHOLDER, text)


#: 五个 Data_Category 各自的模板列表。MVP 中分类阶段使用每个类别的首条
#: （主）模板把非对话结构原文转换为 instruction/output 对；列表结构为后续
#: 扩展多模板预留。所有五个合法类别均有条目，因此 R6-AC7"模板缺失"路径在
#: 完整表下不会被触发，但保留该防御性分支以满足"模板缺失即丢弃"的语义。
TEMPLATES: Final[dict[str, list[Template]]] = {
    "sense": [Template("总结以下感知数据：")],
    "action": [Template("描述以下行为记录：")],
    "thought": [Template("将以下笔记整理为要点：")],
    "skill": [Template("说明以下技能要点：")],
    "credentials": [Template("解释以下凭证信息的用途：")],
}


# ---------------------------------------------------------------------------
# 输入 / 输出数据结构
# ---------------------------------------------------------------------------


@dataclass
class SourceSample:
    """进入分类阶段的来源样本（已脱敏、已规范化）。

    一条样本被视为 **已具备对话结构** 当且仅当其 :attr:`instruction` 与
    :attr:`output` 均非 ``None``（即上游 reader 已将问答对/对话轮次等结构
    映射为显式的 instruction/output 对）；此时分类阶段直接采用这两个字段、
    不查找模板。否则视为 **非对话结构**，将依据 :attr:`category` 套用
    :data:`TEMPLATES` 中的模板，把 :attr:`text` 原文转换为 instruction/output。
    """

    #: 来源 Data_Category 名称（可为任意大小写；分类阶段会小写化后校验）。
    category: str
    #: 非对话结构样本的原文内容。
    text: str = ""
    #: 已具备对话结构时的 instruction；为 ``None`` 表示非对话结构。
    instruction: Optional[str] = None
    #: 已具备对话结构时的 output；为 ``None`` 表示非对话结构。
    output: Optional[str] = None
    #: 可选的 input 字段内容（对应输出样本的 JSON ``input`` 字段）。
    input_field: str = ""

    @property
    def is_conversation(self) -> bool:
        """该样本是否已具备可直接映射的对话结构。"""
        return self.instruction is not None and self.output is not None


@dataclass
class SampleCandidate:
    """标准化后的对话格式候选样本（design.md §4.3）。

    ``input_field`` 对应中间文件中的 JSON 字段名 ``input``；``augmented``
    标记该样本是否由增强阶段产生（分类阶段产出的原始样本恒为 ``False``）。
    """

    instruction: str
    input_field: str
    output: str
    category: Category
    augmented: bool = False


class DiscardReason(enum.Enum):
    """分类阶段的三类丢弃原因，分别对应一条验收标准。"""

    #: R6-AC6：instruction / output 任一为空字符串。
    EMPTY_FIELD = "empty_field"
    #: R6-AC7：非对话结构样本的 Data_Category 在 TEMPLATES 中无对应模板。
    MISSING_TEMPLATE = "missing_template"
    #: R6-AC8：category 取值不属于五元合法集。
    ILLEGAL_CATEGORY = "illegal_category"


@dataclass
class ClassifyResult:
    """单条样本的分类结果。

    ``candidate`` 在样本被保留时为对应的 :class:`SampleCandidate`，被丢弃时
    为 ``None``。``reason`` 与 ``detail`` 仅在丢弃时有意义：``detail`` 对
    :attr:`DiscardReason.MISSING_TEMPLATE` 记录小写化后的类别名，对
    :attr:`DiscardReason.ILLEGAL_CATEGORY` 记录原始非法 category 取值。
    """

    candidate: Optional[SampleCandidate]
    discarded: bool
    reason: Optional[DiscardReason] = None
    detail: str = ""


@dataclass
class ClassifyStats:
    """分类阶段的丢弃计数与原因明细，供 pipeline 输出到 stderr。

    与 :class:`ego_pod.compiler.redact.RedactCounters` 同理，本结构只收集
    信息、不做 I/O。pipeline 据此满足 R6-AC6（丢弃计数）、R6-AC7（模板缺失
    原因）、R6-AC8（非法 category 值）的标准错误输出要求。
    """

    #: R6-AC6：因任一字段为空被丢弃的样本数。
    empty_field_discards: int = 0
    #: R6-AC7：因模板缺失被丢弃的样本数。
    missing_template_discards: int = 0
    #: R6-AC8：因非法 category 被丢弃的样本数。
    illegal_category_discards: int = 0
    #: R6-AC7：被丢弃样本对应的（小写化）类别名清单，保持输入顺序。
    missing_template_categories: list[str] = field(default_factory=list)
    #: R6-AC8：被丢弃样本的原始非法 category 取值清单，保持输入顺序。
    illegal_category_values: list[str] = field(default_factory=list)

    @property
    def total_discards(self) -> int:
        """三类丢弃的样本总数。"""
        return (
            self.empty_field_discards
            + self.missing_template_discards
            + self.illegal_category_discards
        )

    def record(self, result: ClassifyResult) -> None:
        """按 ``result`` 的丢弃原因累加计数与明细；保留的结果不计入。"""
        if not result.discarded:
            return
        if result.reason is DiscardReason.EMPTY_FIELD:
            self.empty_field_discards += 1
        elif result.reason is DiscardReason.MISSING_TEMPLATE:
            self.missing_template_discards += 1
            self.missing_template_categories.append(result.detail)
        elif result.reason is DiscardReason.ILLEGAL_CATEGORY:
            self.illegal_category_discards += 1
            self.illegal_category_values.append(result.detail)

    # -- 供 pipeline 输出到 stderr 的消息构造（措辞集中在分类模块） --------

    def empty_field_summary_line(self) -> str:
        """R6-AC6：标准化阶段丢弃数量的单行摘要。"""
        return f"标准化阶段丢弃 {self.empty_field_discards} 条样本（instruction/output 字段为空）"

    def missing_template_messages(self) -> list[str]:
        """R6-AC7：每条因模板缺失被丢弃样本的说明行。"""
        return [
            f"丢弃样本：Data_Category「{category}」在预定义模板集合中无对应模板"
            for category in self.missing_template_categories
        ]

    def illegal_category_messages(self) -> list[str]:
        """R6-AC8：每条因非法 category 被丢弃样本的说明行。"""
        return [
            f"丢弃样本：非法 category 值「{value}」"
            for value in self.illegal_category_values
        ]


# ---------------------------------------------------------------------------
# 分类主逻辑
# ---------------------------------------------------------------------------


def truncate_field(value: str) -> str:
    """将超过 :data:`MAX_FIELD_LEN` 的字段截断到上限；其余原样返回。"""
    if len(value) > MAX_FIELD_LEN:
        return value[:MAX_FIELD_LEN]
    return value


def classify_sample(sample: SourceSample) -> ClassifyResult:
    """对单条来源样本执行标准化与分类，返回 :class:`ClassifyResult`。

    处理顺序：

    1. 小写化 ``category``；若不属于五元合法集 → 按 R6-AC8 丢弃
       （``detail`` 记录原始取值）。
    2. 已具备对话结构 → 直接采用其 ``instruction`` / ``output``，不查模板；
       非对话结构 → 若类别在 :data:`TEMPLATES` 无条目则按 R6-AC7 丢弃，
       否则用该类别首条模板渲染 ``text`` 得到 instruction/output。
    3. 对 ``instruction`` / ``output`` / ``input`` 三字段截断到
       :data:`MAX_FIELD_LEN`，**再**判空：``instruction`` 或 ``output`` 为
       空字符串 → 按 R6-AC6 丢弃。
    4. 否则返回保留结果，输出样本 ``category`` 为小写化后的合法值
       （Property 8 类别透明传递）。
    """
    normalized_category = sample.category.lower()

    # R6-AC8：非法 category（先于模板缺失判定，因为更根本）。
    if normalized_category not in VALID_CATEGORIES:
        return ClassifyResult(
            candidate=None,
            discarded=True,
            reason=DiscardReason.ILLEGAL_CATEGORY,
            detail=sample.category,
        )

    if sample.is_conversation:
        # 对话结构样本：直接映射，不查找、不应用模板（R6-AC3 后半句、R6-AC7 不适用）。
        instruction = sample.instruction or ""
        output = sample.output or ""
    else:
        # 非对话结构样本：按类别套用模板（R6-AC3 前半句）。
        templates = TEMPLATES.get(normalized_category)
        if not templates:
            # R6-AC7：仅"该类别在 TEMPLATES 中无条目"触发本路径。
            return ClassifyResult(
                candidate=None,
                discarded=True,
                reason=DiscardReason.MISSING_TEMPLATE,
                detail=normalized_category,
            )
        instruction, output = templates[0].render(sample.text)

    # R6-AC1：字段超长先截断到 32768，再判空（R6-AC6）。
    instruction = truncate_field(instruction)
    output = truncate_field(output)
    input_field = truncate_field(sample.input_field)

    if instruction == "" or output == "":
        return ClassifyResult(
            candidate=None,
            discarded=True,
            reason=DiscardReason.EMPTY_FIELD,
            detail=normalized_category,
        )

    candidate = SampleCandidate(
        instruction=instruction,
        input_field=input_field,
        output=output,
        category=cast(Category, normalized_category),
    )
    return ClassifyResult(candidate=candidate, discarded=False)


def classify_samples(
    samples: list[SourceSample],
) -> tuple[list[SampleCandidate], ClassifyStats]:
    """批量分类来源样本，返回 ``(保留样本列表, 丢弃统计)``。

    保留样本按输入顺序排列；被丢弃的样本不进入结果列表，其原因与明细累计在
    :class:`ClassifyStats` 中，供 pipeline 按 R6-AC6/AC7/AC8 输出 stderr。
    """
    kept: list[SampleCandidate] = []
    stats = ClassifyStats()
    for sample in samples:
        result = classify_sample(sample)
        stats.record(result)
        if result.candidate is not None:
            kept.append(result.candidate)
    return kept, stats
