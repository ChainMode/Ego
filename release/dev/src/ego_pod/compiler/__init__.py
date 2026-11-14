"""Compiler 模块 — 语料构建。

本模块负责把 Collector 采集到的原始文件流加工为训练用的对话格式语料
（Conversation_Corpus）。按设计文档 §4，加工流程顺序为：

    read → redact → normalize → classify → filter_dedup → augment → 写 JSONL

各子流程分别由独立模块实现：

- :mod:`ego_pod.compiler.redact`       凭证脱敏（Credential_Token 模式）
- :mod:`ego_pod.compiler.normalize`    控制字符移除与空白压缩
- :mod:`ego_pod.compiler.classify`     数据 → instruction/output 模板转换
- :mod:`ego_pod.compiler.filter_dedup` 过短过滤 + SHA-256 去重
- :mod:`ego_pod.compiler.augment`      不足 16 条时的模板增强
- :mod:`ego_pod.compiler.pipeline`     Compiler 编排，输出 JSONL

当前 :mod:`ego_pod.compiler.redact`、:mod:`ego_pod.compiler.normalize`、
:mod:`ego_pod.compiler.classify`、:mod:`ego_pod.compiler.filter_dedup`、
:mod:`ego_pod.compiler.augment`、:mod:`ego_pod.compiler.pipeline` 均已实现。
"""

from ego_pod.compiler.augment import (
    AUGMENT_PREFIX,
    AUGMENT_TEMPLATES,
    TARGET_SAMPLE_COUNT,
    AugmentStats,
    augment_samples,
    is_augmented_instruction,
    make_augmented_instruction,
    make_augmented_sample,
)
from ego_pod.compiler.classify import (
    MAX_FIELD_LEN,
    TEMPLATES,
    Category,
    ClassifyResult,
    ClassifyStats,
    DiscardReason,
    SampleCandidate,
    SourceSample,
    Template,
    VALID_CATEGORIES,
    classify_sample,
    classify_samples,
    truncate_field,
)
from ego_pod.compiler.filter_dedup import (
    FilterStats,
    MIN_CONTENT_LEN,
    dedup,
    dedup_key,
    extract_text_field,
    filter_and_dedup,
    filter_short,
    is_valid_sample,
    print_filter_summary,
)
from ego_pod.compiler.normalize import (
    collapse_whitespace,
    normalize_text,
    remove_control_chars,
)
from ego_pod.compiler.pipeline import (
    CORPUS_FILENAME,
    EGO_DIR_NAME,
    CompilationResult,
    parse_corpus,
    parse_corpus_line,
    parse_sample_dict,
    read_source_samples,
    render_corpus,
    run_compilation,
    sample_to_dict,
    serialize_sample,
    write_corpus,
)
from ego_pod.compiler.redact import (
    PATTERN_BASE64,
    PATTERN_PEM,
    PATTERN_SK,
    PATTERNS,
    PLACEHOLDER,
    RedactCounters,
    redact_text,
)

__all__ = [
    # redact
    "PATTERN_BASE64",
    "PATTERN_PEM",
    "PATTERN_SK",
    "PATTERNS",
    "PLACEHOLDER",
    "RedactCounters",
    "redact_text",
    # normalize
    "collapse_whitespace",
    "normalize_text",
    "remove_control_chars",
    # classify
    "MAX_FIELD_LEN",
    "TEMPLATES",
    "Category",
    "ClassifyResult",
    "ClassifyStats",
    "DiscardReason",
    "SampleCandidate",
    "SourceSample",
    "Template",
    "VALID_CATEGORIES",
    "classify_sample",
    "classify_samples",
    "truncate_field",
    # filter_dedup
    "FilterStats",
    "MIN_CONTENT_LEN",
    "dedup",
    "dedup_key",
    "extract_text_field",
    "filter_and_dedup",
    "filter_short",
    "is_valid_sample",
    "print_filter_summary",
    # augment
    "AUGMENT_PREFIX",
    "AUGMENT_TEMPLATES",
    "TARGET_SAMPLE_COUNT",
    "AugmentStats",
    "augment_samples",
    "is_augmented_instruction",
    "make_augmented_instruction",
    "make_augmented_sample",
    # pipeline
    "CORPUS_FILENAME",
    "EGO_DIR_NAME",
    "CompilationResult",
    "parse_corpus",
    "parse_corpus_line",
    "parse_sample_dict",
    "read_source_samples",
    "render_corpus",
    "run_compilation",
    "sample_to_dict",
    "serialize_sample",
    "write_corpus",
]
