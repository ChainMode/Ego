"""不足 16 条时的模板增强（design.md §4.5）。

本模块实现语料构建管线中"过滤与增强"子阶段的 **增强** 部分，对应需求
R7-AC4 / R7-AC5 / R7-AC7。它接收过滤与去重阶段（:mod:`ego_pod.compiler.filter_dedup`）
产出的候选样本（:class:`ego_pod.compiler.classify.SampleCandidate`），在样本
总数不足 :data:`TARGET_SAMPLE_COUNT`（16）时，通过对每条剩余样本依次应用
一组数量预定义且固定的指令改写模板（:data:`AUGMENT_TEMPLATES`）生成增强样本。

算法（design.md §4.5）：

* **仅当过滤后样本数 ``0 < n < 16`` 时启动增强**。``n >= 16`` 时为 no-op，
  输出严格等于输入（Property 9：不必要不增强）；``n == 0`` 的"语料为空"
  错误由 ``pipeline.py`` 先于本阶段按 R7-AC6 处理，本模块对 ``n == 0`` 同样
  不产生增强样本、也**不**触发 R7-AC7 的"无法补足 16"错误（R7-AC7 仅在
  ``n > 0`` 时适用）。
* 启动后，**对每条原始样本依次套用全部模板**；每生成一条增强样本立即检查
  总数，一旦达到 :data:`TARGET_SAMPLE_COUNT` **立即停止**（R7-AC4）。
* 每条增强样本的 ``instruction`` 在改写后再加上固定可识别前缀
  :data:`AUGMENT_PREFIX`（``[AUG]``），使其在中间文件中可与原始样本准确
  区分（R7-AC5、Property 10）；该前缀绝不出现在任何原始样本的
  ``instruction`` 开头——这一点由分类阶段的模板（均不以 ``[AUG]`` 起始）保证。
* 增强样本仅改写 ``instruction``，其 ``output`` / ``input`` / ``category``
  与来源样本保持一致，``augmented`` 标记置为 ``True``。
* **若把全部预定义模板应用于全部剩余样本后总数仍 ``< 16``**，则按 R7-AC7
  抛出 :class:`ego_pod.exit_codes.CompilerError`（携带说明文案，由 CLI 顶层
  输出到标准错误并以非零退出码结束本次训练）。

与 :mod:`ego_pod.compiler.classify`、:mod:`ego_pod.compiler.filter_dedup`
一致，本模块为**纯函数**实现：不直接写标准输出/错误流，增强计数收集进
:class:`AugmentStats`；R7-AC7 的失败语义通过抛出 ``CompilerError`` 表达，
实际的 stderr 文案与退出码由 CLI 顶层异常处理统一负责。

正确性不变量（design.md "Correctness Properties"）：

* **Property 9（不必要不增强）** —— 若过滤后样本数 ``>= 16``，增强阶段不产生
  任何新样本，输出等于输入。
* **Property 10（来源可识别）** —— 增强阶段产生的每条样本，其 ``instruction``
  字段以固定前缀 ``[AUG]`` 开头；任何原始样本的 ``instruction`` 都不以
  ``[AUG]`` 开头。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Iterable

from ego_pod.compiler.classify import SampleCandidate
from ego_pod.exit_codes import CompilerError

__all__ = [
    "TARGET_SAMPLE_COUNT",
    "AUGMENT_PREFIX",
    "AUGMENT_TEMPLATES",
    "AugmentStats",
    "is_augmented_instruction",
    "make_augmented_instruction",
    "make_augmented_sample",
    "augment_samples",
]

#: 增强后语料必须达到的最小样本总数（需求 R7-AC4）。
TARGET_SAMPLE_COUNT: Final[int] = 16

#: 增强样本 ``instruction`` 的固定可识别前缀（需求 R7-AC5）。该前缀绝不出现
#: 在任何原始样本的 ``instruction`` 开头，从而保证来源可识别（Property 10）。
AUGMENT_PREFIX: Final[str] = "[AUG]"

#: 预定义且固定数量的指令改写模板。每个模板含 ``{instr}`` 占位符，渲染时被
#: 替换为来源样本的原始 ``instruction``。模板数量固定，与 R7-AC7 的"应用全部
#: 预定义模板后仍无法补足 16"判定直接相关。
AUGMENT_TEMPLATES: Final[tuple[str, ...]] = (
    "请用不同的方式表达：{instr}",
    "换一种说法：{instr}",
    "改写下面这句话：{instr}",
    "请重新组织以下指令：{instr}",
    "用更口语化的方式重写：{instr}",
)


# ---------------------------------------------------------------------------
# 增强统计
# ---------------------------------------------------------------------------


@dataclass
class AugmentStats:
    """增强阶段的样本计数。

    满足不变量 ``final_count == original_count + augmented_count``：增强阶段
    只在原始样本之后追加增强样本，不丢弃任何原始样本。
    """

    #: 进入增强阶段的（过滤后）原始样本数。
    original_count: int = 0
    #: 本阶段新生成的增强样本数。
    augmented_count: int = 0
    #: 增强后的样本总数。
    final_count: int = 0

    def summary_line(self) -> str:
        """构造可被外部捕获的单行增强摘要文本（含三项计数）。"""
        return (
            "增强摘要: "
            f"original={self.original_count} "
            f"augmented={self.augmented_count} "
            f"final={self.final_count}"
        )


# ---------------------------------------------------------------------------
# 增强样本构造
# ---------------------------------------------------------------------------


def is_augmented_instruction(instruction: str) -> bool:
    """判断一条 ``instruction`` 是否带有增强前缀 :data:`AUGMENT_PREFIX`。

    供测试与下游按 R7-AC5 区分原始样本与增强样本使用。
    """
    return instruction.startswith(AUGMENT_PREFIX)


def make_augmented_instruction(template: str, instruction: str) -> str:
    """用 ``template`` 改写 ``instruction`` 并加上 :data:`AUGMENT_PREFIX` 前缀。

    ``template`` 中的 ``{instr}`` 占位符被替换为原始 ``instruction``，随后整体
    前置 ``[AUG]`` 前缀，保证产出以该前缀开头（Property 10）。
    """
    return AUGMENT_PREFIX + template.format(instr=instruction)


def make_augmented_sample(sample: SampleCandidate, template: str) -> SampleCandidate:
    """基于来源 ``sample`` 与一条改写 ``template`` 生成一条增强样本。

    仅改写 ``instruction``（加 ``[AUG]`` 前缀）；``output`` / ``input`` /
    ``category`` 与来源样本一致，``augmented`` 置为 ``True``（design.md §4.5）。
    """
    return SampleCandidate(
        instruction=make_augmented_instruction(template, sample.instruction),
        input_field=sample.input_field,
        output=sample.output,
        category=sample.category,
        augmented=True,
    )


# ---------------------------------------------------------------------------
# 增强主逻辑
# ---------------------------------------------------------------------------


def augment_samples(
    samples: Iterable[SampleCandidate],
) -> tuple[list[SampleCandidate], AugmentStats]:
    """按 R7-AC4 / R7-AC5 / R7-AC7 对样本执行模板增强。

    返回 ``(增强后样本列表, AugmentStats)``。增强后列表以全部原始样本（保持
    输入顺序）开头，其后追加增强样本（按"原始样本 × 模板"的嵌套顺序生成）。

    行为分支：

    * ``n >= 16`` —— 不启动增强，原样返回输入（Property 9）。
    * ``n == 0`` —— 不产生增强样本，返回空列表；"语料为空"由 pipeline 按
      R7-AC6 在本阶段之前处理，本函数不就 ``n == 0`` 抛出 R7-AC7 错误。
    * ``0 < n < 16`` —— 对每条原始样本依次套用全部模板，每生成一条立即检查
      总数，达到 16 立即停止。

    :raises ego_pod.exit_codes.CompilerError: 当 ``0 < n < 16`` 且对全部剩余
        样本应用全部预定义模板后总数仍 ``< 16`` 时（R7-AC7）；该异常由 CLI
        顶层映射为 stderr 错误信息与非零退出码。
    """
    originals = list(samples)
    original_count = len(originals)

    # Property 9：n >= 16 不必要不增强，输出等于输入。
    if original_count >= TARGET_SAMPLE_COUNT:
        return originals, AugmentStats(
            original_count=original_count,
            augmented_count=0,
            final_count=original_count,
        )

    result: list[SampleCandidate] = list(originals)

    # n == 0：无可增强的来源样本（pipeline 已按 R7-AC6 先行处理空语料）。
    # 跳过增强循环；下方 R7-AC7 判定仅在 n > 0 时才有意义，故 n == 0 直接返回。
    if original_count > 0:
        # 对每条原始样本依次套用全部模板（sample-outer / template-inner）。
        for sample in originals:
            if len(result) >= TARGET_SAMPLE_COUNT:
                break
            for template in AUGMENT_TEMPLATES:
                if len(result) >= TARGET_SAMPLE_COUNT:
                    break
                result.append(make_augmented_sample(sample, template))

        # R7-AC7：应用全部模板后仍无法补足至 16 → 报错并以非零退出码结束。
        if len(result) < TARGET_SAMPLE_COUNT:
            raise CompilerError(
                "无法满足最小样本数 16："
                f"过滤后样本数 {original_count}，"
                f"对每条样本应用全部 {len(AUGMENT_TEMPLATES)} 条增强模板后"
                f"仅得 {len(result)} 条，仍不足 {TARGET_SAMPLE_COUNT} 条"
            )

    stats = AugmentStats(
        original_count=original_count,
        augmented_count=len(result) - original_count,
        final_count=len(result),
    )
    return result, stats
