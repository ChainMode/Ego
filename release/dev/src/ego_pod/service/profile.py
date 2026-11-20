"""`Human Prime` 认知画像（`CognitiveProfile`）、红线守卫与 PII 脱敏引擎。

本模块为 `Ego` 三合一本地认知服务（`ego serve`）、子模型切片（`ego slice`）与
`Ego School` 养成机制提供统一的结构化人格锚点与隐私安全底座：

1. **PII 与凭证双重脱敏（:func:`scrub_pii_and_secrets`）**：
   先调用 :func:`ego_pod.compiler.redact.redact_text`（擦除 PEM、``sk-`` 与长 base64），
   再擦除邮箱、手机号、SSN、Bearer / GitHub / AWS / Slack 令牌与助记词片段，统一替换为
   ``[REDACTED]``。
2. **结构化认知画像（:class:`CognitiveProfile`）**：
   持久化于模型包根目录 ``cognitive_profile.json``（并通过 ``manifest.cognitive_profile_sha256``
   纳入 RFC 8785 JCS + Ed25519 签名保护），承载：
   - ``style_anchors``：表达风格锚点；
   - ``decision_heuristics``：决策启发式准则；
   - ``redlines``：不可逾越的硬红线（内置安全底线 + 人类原型显式红线 + 纠偏提炼红线）；
   - ``anti_tone_markers``：外部大模型常见机械套话/违背主人语气的禁用短语；
   - ``exemplars``：高置信直觉快反样本（特别是 ``human_correction`` 高权重偏好对）。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from ego_pod.compiler.redact import PLACEHOLDER, RedactCounters, redact_text

__all__ = [
    "COGNITIVE_PROFILE_FILENAME",
    "COGNITIVE_PROFILE_SCHEMA_VERSION",
    "ScrubResult",
    "RedlineRule",
    "IntuitionExemplar",
    "CognitiveProfile",
    "scrub_pii_and_secrets",
    "default_redline_rules",
    "default_anti_tone_markers",
    "build_default_cognitive_profile",
    "extract_rules_from_text",
    "compile_cognitive_profile",
    "prune_cognitive_profile_for_domains",
    "profile_to_dict",
    "profile_from_dict",
    "write_cognitive_profile",
    "read_cognitive_profile",
]

COGNITIVE_PROFILE_FILENAME = "cognitive_profile.json"
COGNITIVE_PROFILE_SCHEMA_VERSION = "ego-cognitive-profile/1.0"

# ---------------------------------------------------------------------------
# PII 与凭证双重脱敏正则（在 compiler.redact 三类凭证模式之上补充个人隐私脱敏）
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
# 中国大陆 11 位手机号（前后非数字边界）与国际常见带分隔符电话号码
_CN_PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
_INTL_PHONE_RE = re.compile(
    r"(?<!\w)(?:\+\d{1,3}[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]\d{3}[\s.-]\d{4}(?!\d)"
)
_CLOUD_TOKEN_RE = re.compile(
    r"(?:"
    r"gh[pousr]_[A-Za-z0-9]{20,}"
    r"|github_pat_[A-Za-z0-9_]{20,}"
    r"|AKIA[0-9A-Z]{16}"
    r"|xox[baprs]-[A-Za-z0-9-]{10,}"
    r"|sk-ant-[A-Za-z0-9_-]{16,}"
    r"|Bearer\s+[A-Za-z0-9._\-]{16,}"
    r")"
)

_PII_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("cloud_token", _CLOUD_TOKEN_RE),
    ("email", _EMAIL_RE),
    ("ssn", _SSN_RE),
    ("cn_phone", _CN_PHONE_RE),
    ("intl_phone", _INTL_PHONE_RE),
)


@dataclass(frozen=True)
class ScrubResult:
    """一次凭证 + PII 综合脱敏的结果与命中统计。"""

    text: str
    credential_counters: RedactCounters
    pii_counts: dict[str, int]

    @property
    def total_redactions(self) -> int:
        return self.credential_counters.total + sum(self.pii_counts.values())


def scrub_pii_and_secrets(text: str, category: str = "thought") -> ScrubResult:
    """依次执行凭证脱敏（PEM/sk-/base64）与个人 PII 脱敏（邮箱/电话/SSN/令牌）。"""

    redacted, cred_counters = redact_text(text, category)
    pii_counts: dict[str, int] = {}
    current = redacted
    for label, pattern in _PII_PATTERNS:
        current, count = pattern.subn(PLACEHOLDER, current)
        if count > 0:
            pii_counts[label] = pii_counts.get(label, 0) + count
    return ScrubResult(
        text=current,
        credential_counters=cred_counters,
        pii_counts=pii_counts,
    )


# ---------------------------------------------------------------------------
# 结构化认知画像数据模型
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RedlineRule:
    """单条不可逾越的 `Human Prime` 红线规则。"""

    rule_id: str
    description: str
    patterns: tuple[str, ...]
    source: str = "builtin"
    domain: str = "global"

    def matches(self, text: str) -> bool:
        lowered = text.lower()
        for pat in self.patterns:
            if not pat:
                continue
            try:
                if re.search(pat, lowered, flags=re.IGNORECASE):
                    return True
            except re.error:
                if pat.lower() in lowered:
                    return True
        return False


@dataclass(frozen=True)
class IntuitionExemplar:
    """存储在 Pod 内的轻量直觉快反样本（无原始隐私泄露，用于 `<20ms` 本地直觉判别）。"""

    prompt: str
    reply: str
    domain: str = "general"
    weight: float = 1.0
    source: str = "prime_turn"


@dataclass(frozen=True)
class CognitiveProfile:
    """`Ego` Pod 内置的人类原型认知画像。"""

    schema_version: str = COGNITIVE_PROFILE_SCHEMA_VERSION
    persona_summary: str = (
        "Direct, pragmatic, privacy-first thinker; prefers concise conclusions, "
        "explicit trade-offs, and deterministic verification before action."
    )
    style_anchors: tuple[str, ...] = field(default_factory=tuple)
    decision_heuristics: tuple[str, ...] = field(default_factory=tuple)
    redlines: tuple[RedlineRule, ...] = field(default_factory=tuple)
    anti_tone_markers: tuple[str, ...] = field(default_factory=tuple)
    exemplars: tuple[IntuitionExemplar, ...] = field(default_factory=tuple)


def default_redline_rules() -> tuple[RedlineRule, ...]:
    """返回每份 `iFay` / `Ego` 默认守护的三条核心人类原型红线。"""

    return (
        RedlineRule(
            rule_id="redline.secret_exfiltration",
            description=(
                "Never leak, export, print, or transmit private keys, mnemonics, "
                "seed phrases, passwords, or raw API credentials."
            ),
            patterns=(
                r"(?:leak|export|dump|print|send|exfiltrat\w+|share|reveal)\b.{0,40}\b(?:private\s+key|mnemonic|seed\s+phrase|secret\s+key|api\s+key|password|credentials?)",
                r"(?:导出|泄露|发送|公开|打印|告诉我).{0,16}(?:助记词|私钥|根密钥|密码|密钥|api\s*key)",
            ),
            source="builtin",
            domain="security",
        ),
        RedlineRule(
            rule_id="redline.irreversible_destruction",
            description=(
                "Never execute or approve unconfirmed irreversible destructive "
                "filesystem, database, or primary-ledger wipe operations."
            ),
            patterns=(
                r"\brm\s+-[a-z]*r[a-z]*f[a-z]*\s+/(?:[\s`'\"|;&)*]|$)",
                r"\brm\s+-[a-z]*f[a-z]*r[a-z]*\s+/(?:[\s`'\"|;&)*]|$)",
                r"\bdrop\s+(?:database|table|schema)\b",
                r"\btruncate\s+table\b",
                r"\bmkfs(?:\.[a-z0-9]+)?\b",
                r"(?:清空|格式化|彻底删除|销毁).{0,12}(?:生产数据库|主账本|根目录|全部数据|硬盘)",
            ),
            source="builtin",
            domain="safety",
        ),
        RedlineRule(
            rule_id="redline.unauthorized_commitment_or_spoof",
            description=(
                "Never forge FayID signatures, spoof Human Prime identity, or sign "
                "binding financial/legal transfers without explicit Human Prime confirmation."
            ),
            patterns=(
                r"(?:forge|spoof|bypass|fake)\b.{0,30}\b(?:fayid|ed25519|signature|identity)",
                r"(?:transfer|wire|send)\b.{0,25}\b(?:all\s+funds|money|crypto|meritoken)\b.{0,25}\bwithout\s+(?:asking|confirmation|approval)",
                r"(?:伪造|冒充|绕过).{0,12}(?:签名|身份|fayid|授权)",
                r"未经.{0,6}(?:确认|授权|同意).{0,12}(?:转账|汇款|签约|打款)",
            ),
            source="builtin",
            domain="governance",
        ),
    )


def default_anti_tone_markers() -> tuple[str, ...]:
    """返回外部通用大模型常见的机械客套/AI 腔前缀（供 `/v1/guard-rewrite` 清洗）。"""

    return (
        "As an AI language model,",
        "As an AI language model",
        "As an AI assistant,",
        "As an AI assistant",
        "As a large language model,",
        "Certainly! I'd be happy to help with that.",
        "Certainly! I would be happy to help!",
        "Certainly! Here is",
        "Sure thing! I'd love to help you with that!",
        "Great question! Let's dive right in.",
        "I hope this helps! Let me know if you have any other questions.",
        "作为一个人工智能语言模型，",
        "作为一个AI助手，",
        "作为一名AI语言模型，",
        "当然可以！我很乐意为您提供帮助。",
        "没问题！下面为您详细介绍：",
        "希望以上内容对您有所帮助！如果您还有其他问题，请随时告诉我。",
    )


def build_default_cognitive_profile() -> CognitiveProfile:
    """构造开箱可用的默认 `Human Prime` 认知画像。"""

    return CognitiveProfile(
        schema_version=COGNITIVE_PROFILE_SCHEMA_VERSION,
        persona_summary=(
            "Direct, concise, local-first thinker; leads with conclusions, "
            "states concrete trade-offs, and enforces strict privacy & safety redlines."
        ),
        style_anchors=(
            "Lead with the direct conclusion or decision before supporting details.",
            "Keep sentences concise and concrete; avoid generic AI filler or sycophancy.",
            "Preserve technical identifiers, commands, and file paths verbatim in code spans.",
        ),
        decision_heuristics=(
            "Prefer local, deterministic, reversible actions over opaque external side effects.",
            "Verify cryptographic signatures and invariants before promoting or executing artifacts.",
            "Escalate open-ended heavy synthesis to System-2 with explicit persona constraints.",
        ),
        redlines=default_redline_rules(),
        anti_tone_markers=default_anti_tone_markers(),
        exemplars=(),
    )


# ---------------------------------------------------------------------------
# 从人类原型语料与纠偏对中提炼规则、风格与红线
# ---------------------------------------------------------------------------

_REDLINE_TRIGGER_RE = re.compile(
    r"(?:^|[。！？.!?\n])\s*("
    r"(?:严禁|禁止|绝不|绝对不要|千万不要|切勿|不得)[^。！？.!?\n]{4,120}"
    r"|(?:never|do\s+not|don't|must\s+not|strictly\s+forbid\w*)\s+[^.!?\n]{6,140}"
    r")",
    flags=re.IGNORECASE,
)

_PREFERENCE_TRIGGER_RE = re.compile(
    r"(?:^|[。！？.!?\n])\s*("
    r"(?:优先|偏好|一律|默认使用|总是|习惯|请保持|保持)[^。！？.!?\n]{4,120}"
    r"|(?:prefer|always\s+use|always\s+keep|default\s+to|make\s+sure\s+to)\s+[^.!?\n]{6,140}"
    r")",
    flags=re.IGNORECASE,
)


def _extract_redline_keywords_pattern(statement: str) -> str | None:
    """从自然语言红线声明中提取供运行期匹配的关键词正则模式。"""

    cleaned = statement.strip()
    # 中文“严禁/禁止/绝不/不要 <动作短语>”
    m_zh = re.search(
        r"(?:严禁|禁止|绝不(?:允许|可)?|绝对不要|千万不要|切勿|不得)\s*(.+)$",
        cleaned,
    )
    if m_zh:
        phrase = re.sub(r"[，,。！？!?\s]+$", "", m_zh.group(1)).strip()
        if 2 <= len(phrase) <= 48:
            return re.escape(phrase)

    m_en = re.search(
        r"(?:never|do\s+not|don't|must\s+not|forbid\s+\w+\s+to)\s+(.+)$",
        cleaned,
        flags=re.IGNORECASE,
    )
    if m_en:
        phrase = re.sub(r"[,.!?;:\s]+$", "", m_en.group(1)).strip()
        # 取前 2~5 个实词构造紧凑短语匹配，避免整句过长导致漏检
        words = [w for w in re.split(r"\s+", phrase) if w]
        if len(words) >= 2:
            core = r"\s+".join(re.escape(w) for w in words[: min(4, len(words))])
            return core
    return None


def extract_rules_from_text(
    text: str,
    *,
    domain: str = "general",
    source: str = "prime_note",
) -> tuple[list[RedlineRule], list[str]]:
    """从单段已脱敏的人类原型文本中提炼显式红线规则与偏好锚点。"""

    scrubbed = scrub_pii_and_secrets(text).text
    redlines: list[RedlineRule] = []
    preferences: list[str] = []

    for match in _REDLINE_TRIGGER_RE.finditer(scrubbed):
        stmt = " ".join(match.group(1).strip().split())
        if not stmt:
            continue
        pat = _extract_redline_keywords_pattern(stmt)
        if pat:
            digest = hashlib.sha256(stmt.encode("utf-8")).hexdigest()[:10]
            redlines.append(
                RedlineRule(
                    rule_id=f"redline.custom.{digest}",
                    description=stmt,
                    patterns=(pat,),
                    source=source,
                    domain=domain,
                )
            )

    for match in _PREFERENCE_TRIGGER_RE.finditer(scrubbed):
        stmt = " ".join(match.group(1).strip().split())
        if stmt and stmt not in preferences:
            preferences.append(stmt)

    return redlines, preferences


def _category_to_domain(category: str) -> str:
    cat = category.lower().strip()
    if cat in ("skill", "action"):
        return "coding"
    if cat == "thought":
        return "knowledge"
    if cat == "sense":
        return "sense"
    return "general"


def compile_cognitive_profile(
    records: Iterable[Mapping[str, Any]],
    *,
    base_profile: CognitiveProfile | None = None,
    max_exemplars: int = 64,
) -> CognitiveProfile:
    """从语料记录与 `Ego School` 纠偏样本编译完整的 `CognitiveProfile`。

    每条 ``record`` 支持字段：
    - ``instruction`` / ``prompt``
    - ``input``
    - ``output`` / ``reply``
    - ``category`` / ``domain``
    - ``weight``（``human_correction`` 默认 ``>= 3.0``）
    - ``source``（``human_correction`` / ``prime_turn`` / ``prime_note``）
    - ``rejected_draft``（可选：被人类原型否决的草稿片段，提炼进 ``anti_tone_markers``）
    - ``redline``（可选：显式红线描述）
    """

    seed = base_profile if base_profile is not None else build_default_cognitive_profile()
    style_list: list[str] = list(seed.style_anchors)
    heuristic_list: list[str] = list(seed.decision_heuristics)
    redline_by_id: dict[str, RedlineRule] = {r.rule_id: r for r in seed.redlines}
    anti_markers: list[str] = list(seed.anti_tone_markers)
    exemplars: list[IntuitionExemplar] = list(seed.exemplars)
    seen_exemplar_keys: set[str] = {
        f"{ex.domain}:{ex.prompt.strip().lower()}" for ex in exemplars
    }

    candidate_exemplars: list[IntuitionExemplar] = []

    for rec in records:
        raw_instruction = str(rec.get("instruction") or rec.get("prompt") or "").strip()
        if raw_instruction.startswith("[AUG]"):
            # 跳过模板增强副本，仅从真实人类原型表达中编译画像
            continue
        raw_input = str(rec.get("input") or "").strip()
        raw_output = str(
            rec.get("output") or rec.get("corrected_reply") or rec.get("reply") or ""
        ).strip()
        if not raw_output:
            continue

        domain = str(
            rec.get("domain") or _category_to_domain(str(rec.get("category", "")))
        ).strip() or "general"
        source = str(rec.get("source") or "prime_turn").strip()
        weight = float(rec.get("weight", 3.0 if source == "human_correction" else 1.0))

        clean_prompt = scrub_pii_and_secrets(
            f"{raw_instruction}\n{raw_input}".strip() if raw_input else raw_instruction
        ).text
        clean_output = scrub_pii_and_secrets(raw_output).text

        # 显式红线字段（例如来自 ifay-ledger human_correction）
        explicit_redline = str(rec.get("redline") or "").strip()
        if explicit_redline:
            clean_rl = scrub_pii_and_secrets(explicit_redline).text
            pat = _extract_redline_keywords_pattern(clean_rl) or re.escape(clean_rl[:36])
            rl_id = f"redline.ledger.{hashlib.sha256(clean_rl.encode('utf-8')).hexdigest()[:10]}"
            redline_by_id[rl_id] = RedlineRule(
                rule_id=rl_id,
                description=clean_rl,
                patterns=(pat,),
                source="human_correction",
                domain=domain,
            )

        extracted_rls, extracted_prefs = extract_rules_from_text(
            clean_output, domain=domain, source=source
        )
        for rl in extracted_rls:
            redline_by_id[rl.rule_id] = rl
        for pref in extracted_prefs:
            if domain in ("coding", "knowledge") and pref not in heuristic_list:
                if len(heuristic_list) < 24:
                    heuristic_list.append(pref)
            elif pref not in style_list and len(style_list) < 24:
                style_list.append(pref)

        # 若纠偏记录带有简短明确的禁用套话短语（rejected_phrase），加入 anti_tone_markers
        rejected_phrase = str(rec.get("rejected_phrase") or "").strip()
        if rejected_phrase and 4 <= len(rejected_phrase) <= 80:
            clean_rej = scrub_pii_and_secrets(rejected_phrase).text
            if clean_rej not in anti_markers:
                anti_markers.append(clean_rej)

        if clean_prompt and clean_output:
            dedup_key = f"{domain}:{clean_prompt.lower()}"
            if dedup_key not in seen_exemplar_keys:
                seen_exemplar_keys.add(dedup_key)
                candidate_exemplars.append(
                    IntuitionExemplar(
                        prompt=clean_prompt[:512],
                        reply=clean_output[:1024],
                        domain=domain,
                        weight=weight,
                        source=source,
                    )
                )

    # 高权重（如 human_correction）优先保留
    candidate_exemplars.sort(key=lambda ex: (-ex.weight, ex.domain, ex.prompt))
    merged_exemplars = (exemplars + candidate_exemplars)[:max_exemplars]

    return CognitiveProfile(
        schema_version=COGNITIVE_PROFILE_SCHEMA_VERSION,
        persona_summary=seed.persona_summary,
        style_anchors=tuple(style_list[:24]),
        decision_heuristics=tuple(heuristic_list[:24]),
        redlines=tuple(redline_by_id.values()),
        anti_tone_markers=tuple(anti_markers[:40]),
        exemplars=tuple(merged_exemplars),
    )


def prune_cognitive_profile_for_domains(
    profile: CognitiveProfile,
    selected_domains: Sequence[str],
) -> CognitiveProfile:
    """为 `ego slice` 裁剪认知画像：始终保留全部安全红线与全局锚点，仅保留目标领域样本。"""

    allowed = {d.strip().lower() for d in selected_domains if d.strip()} | {
        "global",
        "general",
        "security",
        "safety",
        "governance",
    }
    if "drone-photo" in allowed:
        allowed.add("sense")

    pruned_exemplars = tuple(
        ex for ex in profile.exemplars if ex.domain.lower() in allowed
    )
    return CognitiveProfile(
        schema_version=profile.schema_version,
        persona_summary=profile.persona_summary,
        style_anchors=profile.style_anchors,
        decision_heuristics=profile.decision_heuristics,
        redlines=profile.redlines,  # 人类原型红线随 Kernel 恒定继承，绝不因切片被削弱
        anti_tone_markers=profile.anti_tone_markers,
        exemplars=pruned_exemplars,
    )


# ---------------------------------------------------------------------------
# JSON 序列化与落盘读写
# ---------------------------------------------------------------------------


def profile_to_dict(profile: CognitiveProfile) -> dict[str, Any]:
    """将 :class:`CognitiveProfile` 序列化为纯 JSON 兼容字典。"""

    return {
        "schema_version": profile.schema_version,
        "persona_summary": profile.persona_summary,
        "style_anchors": list(profile.style_anchors),
        "decision_heuristics": list(profile.decision_heuristics),
        "redlines": [
            {
                "rule_id": r.rule_id,
                "description": r.description,
                "patterns": list(r.patterns),
                "source": r.source,
                "domain": r.domain,
            }
            for r in profile.redlines
        ],
        "anti_tone_markers": list(profile.anti_tone_markers),
        "exemplars": [
            {
                "prompt": ex.prompt,
                "reply": ex.reply,
                "domain": ex.domain,
                "weight": ex.weight,
                "source": ex.source,
            }
            for ex in profile.exemplars
        ],
    }


def profile_from_dict(payload: Mapping[str, Any]) -> CognitiveProfile:
    """从 JSON 字典还原 :class:`CognitiveProfile`（并自动补齐内置基础红线）。"""

    base = build_default_cognitive_profile()
    redline_map: dict[str, RedlineRule] = {r.rule_id: r for r in base.redlines}
    raw_redlines = payload.get("redlines", [])
    if isinstance(raw_redlines, list):
        for item in raw_redlines:
            if isinstance(item, Mapping) and item.get("rule_id") and item.get("description"):
                pats = item.get("patterns", [])
                rl = RedlineRule(
                    rule_id=str(item["rule_id"]),
                    description=str(item["description"]),
                    patterns=tuple(str(p) for p in pats) if isinstance(pats, list) else (),
                    source=str(item.get("source", "custom")),
                    domain=str(item.get("domain", "global")),
                )
                redline_map[rl.rule_id] = rl

    exemplars: list[IntuitionExemplar] = []
    raw_exemplars = payload.get("exemplars", [])
    if isinstance(raw_exemplars, list):
        for item in raw_exemplars:
            if isinstance(item, Mapping) and item.get("prompt") and item.get("reply"):
                exemplars.append(
                    IntuitionExemplar(
                        prompt=str(item["prompt"]),
                        reply=str(item["reply"]),
                        domain=str(item.get("domain", "general")),
                        weight=float(item.get("weight", 1.0)),
                        source=str(item.get("source", "prime_turn")),
                    )
                )

    raw_styles = payload.get("style_anchors")
    styles = (
        tuple(str(s) for s in raw_styles if str(s).strip())
        if isinstance(raw_styles, list) and raw_styles
        else base.style_anchors
    )
    raw_heuristics = payload.get("decision_heuristics")
    heuristics = (
        tuple(str(h) for h in raw_heuristics if str(h).strip())
        if isinstance(raw_heuristics, list) and raw_heuristics
        else base.decision_heuristics
    )
    raw_anti = payload.get("anti_tone_markers")
    anti = (
        tuple(str(a) for a in raw_anti if str(a).strip())
        if isinstance(raw_anti, list) and raw_anti
        else base.anti_tone_markers
    )

    return CognitiveProfile(
        schema_version=str(
            payload.get("schema_version", COGNITIVE_PROFILE_SCHEMA_VERSION)
        ),
        persona_summary=str(payload.get("persona_summary", base.persona_summary)),
        style_anchors=styles,
        decision_heuristics=heuristics,
        redlines=tuple(redline_map.values()),
        anti_tone_markers=anti,
        exemplars=tuple(exemplars),
    )


def write_cognitive_profile(path: str | Path, profile: CognitiveProfile) -> str:
    """将 :class:`CognitiveProfile` 原子写入 ``path`` 并返回 ``sha256:<64hex>`` 摘要。"""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(profile_to_dict(profile), ensure_ascii=False, indent=2) + "\n"
    raw_bytes = text.encode("utf-8")
    tmp = target.with_suffix(".json.tmp")
    tmp.write_bytes(raw_bytes)
    tmp.replace(target)
    return f"sha256:{hashlib.sha256(raw_bytes).hexdigest()}"


def read_cognitive_profile(path: str | Path) -> CognitiveProfile:
    """从 JSON 文件加载 :class:`CognitiveProfile`。"""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"cognitive_profile 顶层必须为 JSON 对象：{path}")
    return profile_from_dict(raw)
