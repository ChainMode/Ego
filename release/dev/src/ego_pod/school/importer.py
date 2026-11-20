"""`Ego School` 冷启动与回流导入器（`ego school import`）。

支持四种标准输入格式：
- ``markdown``：个人 Markdown 笔记目录或带角色标记的对话记录；
- ``jsonl``：通用对话/指令 JSONL；
- ``openai``：ChatGPT 官方导出的 ``conversations.json`` 树状对话历史；
- ``ifay-ledger``：`iFay` 对等主体回流的终端交互与 ``human_correction`` 纠偏账本。

核心铁律：
1. **只学主人，不学通用 AI 废话**：仅提取 ``Human Prime`` 本人撰写的表达、决策，以及
   **人类原型纠正助手/iFay 草稿的瞬间（``human_correction`` -> 高权重偏好与红线训练样本）**；
2. **落盘前强制脱敏**：全量经过 ``redact_text``（PEM / ``sk-`` / base64）+ 个人 PII
   （邮箱 / 手机号 / SSN / 云令牌）双重擦除，零明文隐私写入工作区；
3. **幂等增量去重**：通过 SHA-256 水位游标（``.ego/school/ledger_cursor.json``）避免重复导入。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from ego_pod.compiler.redact import RedactCounters
from ego_pod.config.defaults import DEFAULT_SOURCE_PATHS, write_default
from ego_pod.exit_codes import CollectorError, ConfigError
from ego_pod.service.profile import (
    COGNITIVE_PROFILE_FILENAME,
    compile_cognitive_profile,
    scrub_pii_and_secrets,
    write_cognitive_profile,
)

__all__ = [
    "SUPPORTED_IMPORT_FORMATS",
    "SCHOOL_DIRNAME",
    "IMPORTED_RECORDS_FILENAME",
    "LEDGER_CURSOR_FILENAME",
    "SchoolImportResult",
    "import_school_data",
    "load_imported_school_records",
]

SUPPORTED_IMPORT_FORMATS: tuple[str, ...] = (
    "markdown",
    "jsonl",
    "openai",
    "ifay-ledger",
)

SCHOOL_DIRNAME = "school"
IMPORTED_RECORDS_FILENAME = "imported_records.jsonl"
LEDGER_CURSOR_FILENAME = "ledger_cursor.json"

_HUMAN_ROLES = frozenset({"user", "human", "human_prime", "owner", "prime"})
_ASSISTANT_ROLES = frozenset({"assistant", "ai", "bot", "model", "ifay", "cofay"})

_CORRECTION_CUE_RE = re.compile(
    r"(?:"
    r"^\s*(?:no[,.\s!]|wrong[,.\s!]|incorrect[,.\s!]|stop[,.\s!]|actually[,.\s]|instead[,.\s])"
    r"|\b(?:don't\s+use|do\s+not\s+use|never\s+use|never\s+do|instead\s+of|rewrite\s+it\s+as|too\s+verbose|keep\s+it\s+short)\b"
    r"|(?:不对|错了|不要这样|不要用|别用|严禁|绝不|太啰嗦|改成|改用|以后记住|重来|按我说的)"
    r")",
    flags=re.IGNORECASE,
)

_MD_ROLE_LINE_RE = re.compile(
    r"^(?:#{1,4}\s*|\*\*)?(user|human\s*prime|human|owner|assistant|ai|ifay|system)(?:\*\*)?\s*[:：]\s*(.*)$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class SchoolImportResult:
    """一次 `ego school import` 的结构化导入统计。"""

    fmt: str
    workspace: Path
    imported_turns: int
    correction_pairs: int
    skipped_duplicates: int
    skipped_non_human: int
    total_redactions: int
    redlines_count: int
    profile_path: Path


def _infer_category_and_domain(text: str, explicit_domain: str | None = None) -> tuple[str, str]:
    lowered = text.lower()
    if explicit_domain:
        dom = explicit_domain.strip().lower()
        if dom == "coding":
            return "skill", "coding"
        if dom in ("drone-photo", "sense"):
            return "sense", dom
        if dom in ("social", "creative", "knowledge"):
            return "thought", dom

    if any(
        kw in lowered
        for kw in (
            "def ",
            "class ",
            "import ",
            "pytest",
            "python",
            "git ",
            "typescript",
            "pip ",
            "函数",
            "代码",
            "重构",
            "测试",
            "接口",
        )
    ):
        return "skill", "coding"
    if any(
        kw in lowered
        for kw in (
            "drone",
            "camera",
            "iso",
            "shutter",
            "gimbal",
            "sensor",
            "航拍",
            "无人机",
            "快门",
            "相机",
            "感知",
            "观察",
        )
    ):
        return "sense", "drone-photo" if ("drone" in lowered or "无人机" in lowered or "航拍" in lowered) else "sense"
    if any(kw in lowered for kw in ("run ", "execute", "deploy", "执行", "操作", "流程", "步骤")):
        return "action", "coding"
    return "thought", "knowledge"


def _is_correction_turn(human_text: str) -> bool:
    return bool(_CORRECTION_CUE_RE.search(human_text.strip()))


def _ensure_workspace_scaffold(workspace: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    cfg_path = workspace / "ego-config.yaml"
    if not cfg_path.is_file():
        write_default(cfg_path)
    for rel in DEFAULT_SOURCE_PATHS.values():
        (workspace / rel).mkdir(parents=True, exist_ok=True)


def _load_cursor(cursor_path: Path) -> set[str]:
    if not cursor_path.is_file():
        return set()
    try:
        obj = json.loads(cursor_path.read_text(encoding="utf-8"))
        seen = obj.get("seen_digests", []) if isinstance(obj, Mapping) else []
        return {str(x) for x in seen if isinstance(x, str)}
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return set()


def _save_cursor(cursor_path: Path, seen_digests: set[str]) -> None:
    cursor_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "ego-school-cursor/1.0",
        "seen_digests": sorted(seen_digests),
    }
    tmp = cursor_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(cursor_path)


def load_imported_school_records(workspace: str | Path) -> list[dict[str, Any]]:
    """读取工作区 `.ego/school/imported_records.jsonl` 中已累积的全部已脱敏记录。"""

    rec_path = Path(workspace) / ".ego" / SCHOOL_DIRNAME / IMPORTED_RECORDS_FILENAME
    if not rec_path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in rec_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                records.append(obj)
        except json.JSONDecodeError:
            continue
    return records


# ---------------------------------------------------------------------------
# 各格式解析器 -> 产出统一的原始候选事件列表
# ---------------------------------------------------------------------------


def _extract_turns_to_candidates(
    turns: list[tuple[str, str]],
    *,
    title: str = "",
) -> tuple[list[dict[str, Any]], int]:
    """从按时间序排列的 ``(role, content)`` 轮次序列中仅提取人类原型表达与纠偏对。"""

    candidates: list[dict[str, Any]] = []
    skipped_non_human = 0
    prev_user_prompt = ""
    prev_assistant_draft = ""

    for role_raw, text_raw in turns:
        role = role_raw.strip().lower().replace(" ", "_")
        content = text_raw.strip()
        if not content:
            continue

        if role in _ASSISTANT_ROLES:
            prev_assistant_draft = content
            skipped_non_human += 1
            continue

        if role not in _HUMAN_ROLES:
            skipped_non_human += 1
            continue

        # 当前轮是 Human Prime 本人发言！
        if prev_assistant_draft and _is_correction_turn(content):
            anchor_prompt = prev_user_prompt or title or "Follow Human Prime preference"
            cat, dom = _infer_category_and_domain(f"{anchor_prompt}\n{content}")
            candidates.append(
                {
                    "prompt": anchor_prompt,
                    "rejected_draft": prev_assistant_draft[:300],
                    "output": content,
                    "category": cat,
                    "domain": dom,
                    "source": "human_correction",
                    "weight": 3.5,
                }
            )
            prev_assistant_draft = ""
        else:
            # 普通人类原型发言/指示/观点记录
            if len(content) >= 6:
                cat, dom = _infer_category_and_domain(content)
                prompt_label = title if title else f"Human Prime {dom} directive"
                candidates.append(
                    {
                        "prompt": prompt_label,
                        "output": content,
                        "category": cat,
                        "domain": dom,
                        "source": "prime_turn",
                        "weight": 1.0,
                    }
                )
            prev_user_prompt = content
            prev_assistant_draft = ""

    return candidates, skipped_non_human


def _parse_openai_conversations(input_path: Path) -> tuple[list[dict[str, Any]], int]:
    try:
        raw = json.loads(input_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise CollectorError(f"读取 OpenAI conversations.json 失败：{exc}") from exc

    conv_list: list[Mapping[str, Any]] = []
    if isinstance(raw, list):
        conv_list = [item for item in raw if isinstance(item, Mapping)]
    elif isinstance(raw, Mapping):
        if isinstance(raw.get("conversations"), list):
            conv_list = [
                item for item in raw["conversations"] if isinstance(item, Mapping)
            ]
        else:
            conv_list = [raw]
    else:
        raise CollectorError("OpenAI 导入文件顶层必须为 JSON 数组或对象")

    all_candidates: list[dict[str, Any]] = []
    total_skipped = 0

    for conv in conv_list:
        title = str(conv.get("title") or "").strip()
        turns: list[tuple[str, str]] = []

        mapping = conv.get("mapping")
        if isinstance(mapping, Mapping):
            # 按父子拓扑序（或 create_time）遍历 ChatGPT 树状 mapping
            roots = [
                nid
                for nid, node in mapping.items()
                if isinstance(node, Mapping)
                and (node.get("parent") is None or node.get("parent") not in mapping)
            ]
            visited: set[str] = set()
            stack: list[str] = list(reversed(roots))
            while stack:
                nid = stack.pop()
                if nid in visited:
                    continue
                visited.add(nid)
                node = mapping.get(nid)
                if not isinstance(node, Mapping):
                    continue
                msg = node.get("message")
                if isinstance(msg, Mapping):
                    author_role = str(
                        (msg.get("author") or {}).get("role") or ""
                    ).strip()
                    content_obj = msg.get("content") or {}
                    parts = (
                        content_obj.get("parts")
                        if isinstance(content_obj, Mapping)
                        else None
                    )
                    if isinstance(parts, list):
                        text = "\n".join(
                            str(p)
                            for p in parts
                            if isinstance(p, str) and p.strip()
                        ).strip()
                    elif isinstance(content_obj, Mapping) and isinstance(
                        content_obj.get("text"), str
                    ):
                        text = content_obj["text"].strip()
                    else:
                        text = ""
                    if author_role and text:
                        turns.append((author_role, text))
                children = node.get("children")
                if isinstance(children, list):
                    for child_id in reversed(children):
                        if isinstance(child_id, str):
                            stack.append(child_id)
        elif isinstance(conv.get("messages"), list):
            for msg in conv["messages"]:
                if isinstance(msg, Mapping):
                    turns.append(
                        (
                            str(msg.get("role", "")),
                            str(msg.get("content", "")),
                        )
                    )

        cands, skipped = _extract_turns_to_candidates(turns, title=title)
        all_candidates.extend(cands)
        total_skipped += skipped

    return all_candidates, total_skipped


def _parse_ifay_ledger(input_path: Path) -> tuple[list[dict[str, Any]], int]:
    target_path = input_path / "ledger.json" if input_path.is_dir() and (input_path / "ledger.json").is_file() else input_path
    try:
        raw_text = target_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise CollectorError(f"读取 iFay 账本文件失败：{exc}") from exc

    events: list[Mapping[str, Any]] = []
    if raw_text.startswith("["):
        parsed = json.loads(raw_text)
        if isinstance(parsed, list):
            events = [e for e in parsed if isinstance(e, Mapping)]
    elif raw_text.startswith("{") and ("\n" not in raw_text or '"events"' in raw_text):
        try:
            parsed_obj = json.loads(raw_text)
            if isinstance(parsed_obj, Mapping) and isinstance(parsed_obj.get("events"), list):
                events = [e for e in parsed_obj["events"] if isinstance(e, Mapping)]
            elif isinstance(parsed_obj, Mapping):
                events = [parsed_obj]
        except json.JSONDecodeError:
            pass

    if not events and not raw_text.startswith("["):
        for line in raw_text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, Mapping):
                    events.append(obj)
            except json.JSONDecodeError as exc:
                raise CollectorError(f"iFay 账本 JSONL 解析失败：{exc}") from exc

    candidates: list[dict[str, Any]] = []
    skipped_non_human = 0

    for ev in events:
        ev_type = str(ev.get("event_type") or ev.get("type") or "").strip().lower()
        event_id = str(ev.get("event_id") or ev.get("event_hash") or ev.get("id") or "").strip()
        payload = ev.get("payload") if isinstance(ev.get("payload"), Mapping) else {}

        # 0) @ifay/soul 创世主体事件（soul_genesis）：提取初始六维画像中的人类原型 Thought 条目
        if ev_type == "soul_genesis" and isinstance(payload.get("initial_profile"), Mapping):
            thought_map = payload["initial_profile"].get("thought")
            if isinstance(thought_map, Mapping):
                for subtype, items in thought_map.items():
                    if not isinstance(items, list):
                        continue
                    for idx, item in enumerate(items):
                        if not isinstance(item, Mapping):
                            continue
                        txt = str(item.get("text") or "").strip()
                        if not txt:
                            continue
                        title = str(item.get("title") or f"Genesis {subtype}").strip()
                        cat, dom = _infer_category_and_domain(
                            f"{title}\n{txt}",
                            explicit_domain=str(item.get("domain") or "") or None,
                        )
                        candidates.append(
                            {
                                "event_id": f"{event_id}:genesis:{subtype}:{idx}",
                                "prompt": title,
                                "output": txt,
                                "redline": "",
                                "category": cat,
                                "domain": dom,
                                "source": "prime_turn",
                                "weight": 1.5,
                            }
                        )
            continue

        # 1) 显式纠偏事件（human_correction / draft_corrected / guard_override，兼容顶层与 @ifay/soul payload）
        is_correction = ev_type in (
            "human_correction",
            "draft_corrected",
            "guard_override",
            "preference_correction",
        ) or ("corrected_reply" in ev or "human_correction" in ev)

        if is_correction:
            prompt = str(
                ev.get("prompt")
                or payload.get("prompt_context")
                or payload.get("prompt")
                or ev.get("instruction")
                or ev.get("query")
                or "Human Prime preference"
            ).strip()
            corrected = str(
                ev.get("corrected_reply")
                or ev.get("human_correction")
                or ev.get("preferred_reply")
                or payload.get("preferred_output")
                or payload.get("corrected_reply")
                or ev.get("feedback")
                or ev.get("output")
                or ""
            ).strip()
            if not corrected:
                continue
            cat, dom = _infer_category_and_domain(
                f"{prompt}\n{corrected}",
                explicit_domain=str(ev.get("domain") or payload.get("domain") or "") or None,
            )
            candidates.append(
                {
                    "event_id": event_id,
                    "prompt": prompt,
                    "rejected_draft": str(
                        ev.get("draft_reply")
                        or ev.get("rejected_reply")
                        or payload.get("rejected_output")
                        or ""
                    )[:300],
                    "rejected_phrase": str(
                        ev.get("rejected_phrase") or payload.get("rejected_phrase") or ""
                    ),
                    "redline": str(
                        ev.get("redline") or payload.get("extracted_rule") or ""
                    ),
                    "output": corrected,
                    "category": cat,
                    "domain": dom,
                    "source": "human_correction",
                    "weight": float(ev.get("weight", 4.0)),
                }
            )
            continue

        # 1b) @ifay/soul 主体新增思维与 DTP 语境归集事件（thought_added / dtp_context_ingested）
        if ev_type in ("thought_added", "dtp_context_ingested") and isinstance(
            payload.get("thought"), Mapping
        ):
            th = payload["thought"]
            txt = str(th.get("text") or "").strip()
            if txt:
                title = str(th.get("title") or f"Human Prime {ev_type}").strip()
                cat, dom = _infer_category_and_domain(
                    f"{title}\n{txt}",
                    explicit_domain=str(th.get("domain") or "") or None,
                )
                candidates.append(
                    {
                        "event_id": event_id,
                        "prompt": title,
                        "output": txt,
                        "redline": "",
                        "category": cat,
                        "domain": dom,
                        "source": "prime_turn",
                        "weight": 1.5,
                    }
                )
                continue

        # 2) 主体回流的人类原型亲自输入/决策/笔记事件
        author = str(ev.get("author") or ev.get("role") or "").strip().lower()
        if author in _ASSISTANT_ROLES and ev_type not in (
            "human_turn",
            "human_decision",
            "thought_note",
            "terminal_interaction",
        ):
            skipped_non_human += 1
            continue

        if ev_type in (
            "human_turn",
            "human_decision",
            "terminal_interaction",
            "thought_note",
            "skill_note",
            "sense_observation",
            "cap_policy_decision",
        ) or author in _HUMAN_ROLES:
            human_out = str(
                ev.get("human_reply")
                or ev.get("decision")
                or ev.get("content")
                or ev.get("note")
                or ev.get("output")
                or ""
            ).strip()
            if not human_out:
                skipped_non_human += 1
                continue
            prompt = str(
                ev.get("prompt")
                or ev.get("context_summary")
                or ev.get("topic")
                or f"Human Prime {ev_type or 'decision'}"
            ).strip()
            cat, dom = _infer_category_and_domain(
                f"{prompt}\n{human_out}",
                explicit_domain=str(ev.get("domain") or "") or None,
            )
            candidates.append(
                {
                    "event_id": event_id,
                    "prompt": prompt,
                    "output": human_out,
                    "redline": str(ev.get("redline") or ""),
                    "category": cat,
                    "domain": dom,
                    "source": "prime_turn",
                    "weight": float(ev.get("weight", 1.5)),
                }
            )
        else:
            skipped_non_human += 1

    return candidates, skipped_non_human


def _parse_generic_jsonl(input_path: Path) -> tuple[list[dict[str, Any]], int]:
    try:
        lines = input_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise CollectorError(f"读取 JSONL 文件失败：{exc}") from exc

    candidates: list[dict[str, Any]] = []
    skipped_non_human = 0
    rolling_turns: list[tuple[str, str]] = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CollectorError(f"JSONL 行解析失败：{exc}") from exc
        if not isinstance(obj, Mapping):
            continue

        if isinstance(obj.get("messages"), list):
            msg_turns = [
                (str(m.get("role", "")), str(m.get("content", "")))
                for m in obj["messages"]
                if isinstance(m, Mapping)
            ]
            cands, skipped = _extract_turns_to_candidates(msg_turns)
            candidates.extend(cands)
            skipped_non_human += skipped
        elif "corrected_reply" in obj or "human_correction" in obj:
            out = str(obj.get("corrected_reply") or obj.get("human_correction") or "").strip()
            if out:
                prompt = str(obj.get("prompt") or obj.get("instruction") or "Preference correction").strip()
                cat, dom = _infer_category_and_domain(f"{prompt}\n{out}")
                candidates.append(
                    {
                        "prompt": prompt,
                        "rejected_draft": str(obj.get("draft_reply") or "")[:300],
                        "output": out,
                        "category": cat,
                        "domain": dom,
                        "source": "human_correction",
                        "weight": 3.5,
                    }
                )
        elif "role" in obj and "content" in obj:
            rolling_turns.append((str(obj["role"]), str(obj["content"])))
        else:
            author = str(obj.get("author") or "human_prime").strip().lower()
            if author in _ASSISTANT_ROLES:
                skipped_non_human += 1
                continue
            out = str(obj.get("output") or obj.get("text") or "").strip()
            if out:
                prompt = str(obj.get("instruction") or obj.get("prompt") or "Human Prime note").strip()
                cat, dom = _infer_category_and_domain(f"{prompt}\n{out}")
                candidates.append(
                    {
                        "prompt": prompt,
                        "output": out,
                        "category": str(obj.get("category") or cat),
                        "domain": dom,
                        "source": "prime_turn",
                        "weight": 1.0,
                    }
                )

    if rolling_turns:
        cands, skipped = _extract_turns_to_candidates(rolling_turns)
        candidates.extend(cands)
        skipped_non_human += skipped

    return candidates, skipped_non_human


def _parse_markdown_source(input_path: Path) -> tuple[list[dict[str, Any]], int]:
    md_files: list[Path] = []
    if input_path.is_dir():
        md_files = sorted(
            p
            for p in input_path.rglob("*")
            if p.is_file() and p.suffix.lower() in (".md", ".markdown", ".txt")
        )
    elif input_path.is_file():
        md_files = [input_path]
    else:
        raise CollectorError(f"未找到 Markdown 输入路径：{input_path}")

    candidates: list[dict[str, Any]] = []
    skipped_non_human = 0

    for md_file in md_files:
        text = md_file.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()

        # 检测是否为带角色对话记录
        turns: list[tuple[str, str]] = []
        current_role: str | None = None
        current_buf: list[str] = []
        for line in lines:
            m = _MD_ROLE_LINE_RE.match(line.strip())
            if m:
                if current_role is not None and "".join(current_buf).strip():
                    turns.append((current_role, "\n".join(current_buf).strip()))
                current_role = m.group(1)
                first_line = m.group(2).strip()
                current_buf = [first_line] if first_line else []
            elif current_role is not None:
                current_buf.append(line)
        if current_role is not None and "".join(current_buf).strip():
            turns.append((current_role, "\n".join(current_buf).strip()))

        if turns:
            cands, skipped = _extract_turns_to_candidates(turns, title=md_file.stem)
            candidates.extend(cands)
            skipped_non_human += skipped
            continue

        # 纯个人笔记：按空行/标题切分语义段落作为 prime_note
        blocks = [b.strip() for b in re.split(r"\n\s*\n+", text) if len(b.strip()) >= 8]
        for blk in blocks:
            cat, dom = _infer_category_and_domain(blk)
            candidates.append(
                {
                    "prompt": f"Note ({md_file.stem})",
                    "output": blk,
                    "category": cat,
                    "domain": dom,
                    "source": "prime_note",
                    "weight": 1.0,
                }
            )

    return candidates, skipped_non_human


# ---------------------------------------------------------------------------
# 统一脱敏、幂等落盘与画像更新主入口
# ---------------------------------------------------------------------------


def import_school_data(
    input_path: str | Path,
    *,
    fmt: str,
    workspace: str | Path,
) -> SchoolImportResult:
    """执行 `Ego School` 冷启动/增量导入，自动擦除 PII 并生成分类语料与认知画像。"""

    normalized_fmt = fmt.strip().lower()
    if normalized_fmt not in SUPPORTED_IMPORT_FORMATS:
        raise ConfigError(
            f"不支持的 --format {fmt!r}；支持：{list(SUPPORTED_IMPORT_FORMATS)}"
        )

    src = Path(input_path)
    if not src.exists():
        raise CollectorError(f"输入路径不存在：{src}")

    ws = Path(workspace)
    _ensure_workspace_scaffold(ws)

    if normalized_fmt == "openai":
        raw_candidates, skipped_non_human = _parse_openai_conversations(src)
    elif normalized_fmt == "ifay-ledger":
        raw_candidates, skipped_non_human = _parse_ifay_ledger(src)
    elif normalized_fmt == "jsonl":
        raw_candidates, skipped_non_human = _parse_generic_jsonl(src)
    else:
        raw_candidates, skipped_non_human = _parse_markdown_source(src)

    school_dir = ws / ".ego" / SCHOOL_DIRNAME
    school_dir.mkdir(parents=True, exist_ok=True)
    cursor_path = school_dir / LEDGER_CURSOR_FILENAME
    records_path = school_dir / IMPORTED_RECORDS_FILENAME
    seen_digests = _load_cursor(cursor_path)

    total_redactions = 0
    merged_cred_counters = RedactCounters()
    imported_turns = 0
    correction_pairs = 0
    skipped_duplicates = 0
    newly_appended_lines: list[str] = []

    for item in raw_candidates:
        cat = str(item.get("category") or "thought").lower()
        if cat not in DEFAULT_SOURCE_PATHS:
            cat = "thought"

        scrub_prompt = scrub_pii_and_secrets(str(item.get("prompt") or ""), category=cat)
        scrub_output = scrub_pii_and_secrets(str(item.get("output") or ""), category=cat)
        scrub_rejected = scrub_pii_and_secrets(
            str(item.get("rejected_draft") or ""), category=cat
        )
        scrub_redline = scrub_pii_and_secrets(
            str(item.get("redline") or ""), category=cat
        )
        scrub_rej_phrase = scrub_pii_and_secrets(
            str(item.get("rejected_phrase") or ""), category=cat
        )

        for sc in (
            scrub_prompt,
            scrub_output,
            scrub_rejected,
            scrub_redline,
            scrub_rej_phrase,
        ):
            total_redactions += sc.total_redactions
            merged_cred_counters.merge(sc.credential_counters)

        clean_prompt = scrub_prompt.text.strip()
        clean_output = scrub_output.text.strip()
        if not clean_output:
            continue

        dedup_basis = (
            str(item.get("event_id") or "").strip()
            or f"{item.get('source')}|{clean_prompt}|{clean_output}|{scrub_redline.text}"
        )
        digest = hashlib.sha256(dedup_basis.encode("utf-8")).hexdigest()
        if digest in seen_digests:
            skipped_duplicates += 1
            continue
        seen_digests.add(digest)

        source_kind = str(item.get("source") or "prime_turn")
        weight = float(item.get("weight", 3.5 if source_kind == "human_correction" else 1.0))

        clean_record: dict[str, Any] = {
            "digest": digest,
            "prompt": clean_prompt,
            "output": clean_output,
            "category": cat,
            "domain": str(item.get("domain") or "general"),
            "source": source_kind,
            "weight": weight,
        }
        if scrub_rejected.text.strip():
            clean_record["rejected_draft"] = scrub_rejected.text.strip()
        if scrub_redline.text.strip():
            clean_record["redline"] = scrub_redline.text.strip()
        if scrub_rej_phrase.text.strip():
            clean_record["rejected_phrase"] = scrub_rej_phrase.text.strip()

        newly_appended_lines.append(
            json.dumps(clean_record, ensure_ascii=False, separators=(",", ":"))
        )

        # 物化为 ego-data/<category>/ 下的本地训练源文件供 Dojo Collector/Compiler 消费
        cat_dir = ws / DEFAULT_SOURCE_PATHS[cat]
        cat_dir.mkdir(parents=True, exist_ok=True)
        short_hash = digest[:16]

        if source_kind == "human_correction":
            correction_pairs += 1
            # 高权重纠偏样本：同时写入主偏好准则与纠偏边界对照样本，提升 SFT 训练权重！
            primary_doc = (
                f"[Human Prime Correction | domain={clean_record['domain']}]\n"
                f"Context: {clean_prompt}\n"
                f"Preferred Response: {clean_output}\n"
            )
            if clean_record.get("redline"):
                primary_doc += f"Redline: {clean_record['redline']}\n"
            (cat_dir / f"school_corr_{short_hash}_a.txt").write_text(
                primary_doc, encoding="utf-8"
            )
            boundary_doc = (
                f"When handling '{clean_prompt}', follow the Human Prime's exact correction: "
                f"{clean_output}"
            )
            (cat_dir / f"school_corr_{short_hash}_b.txt").write_text(
                boundary_doc + "\n", encoding="utf-8"
            )
        else:
            imported_turns += 1
            turn_doc = f"{clean_prompt}: {clean_output}\n" if clean_prompt else f"{clean_output}\n"
            (cat_dir / f"school_turn_{short_hash}.txt").write_text(
                turn_doc, encoding="utf-8"
            )

    if newly_appended_lines:
        with records_path.open("a", encoding="utf-8") as fh:
            for line in newly_appended_lines:
                fh.write(line + "\n")
    _save_cursor(cursor_path, seen_digests)

    all_records = load_imported_school_records(ws)
    profile = compile_cognitive_profile(all_records)
    profile_path = ws / ".ego" / COGNITIVE_PROFILE_FILENAME
    write_cognitive_profile(profile_path, profile)

    # 确保 Kernel（sense/credentials）始终沉淀核心人格锚点与红线，满足 Sparse_Grid Kernel-Always 不变量
    sense_dir = ws / DEFAULT_SOURCE_PATHS["sense"]
    cred_dir = ws / DEFAULT_SOURCE_PATHS["credentials"]
    has_kernel_file = any(
        p.is_file() for d in (sense_dir, cred_dir) for p in d.iterdir()
    )
    if not has_kernel_file:
        kernel_lines = [
            f"Human Prime Core Persona: {profile.persona_summary}",
            *[f"Style Anchor: {s}" for s in profile.style_anchors[:3]],
            *[f"Redline Boundary: {r.description}" for r in profile.redlines[:5]],
        ]
        (sense_dir / "school_kernel_foundation.txt").write_text(
            "\n".join(kernel_lines) + "\n", encoding="utf-8"
        )

    return SchoolImportResult(
        fmt=normalized_fmt,
        workspace=ws.resolve(),
        imported_turns=imported_turns,
        correction_pairs=correction_pairs,
        skipped_duplicates=skipped_duplicates,
        skipped_non_human=skipped_non_human,
        total_redactions=total_redactions,
        redlines_count=len(profile.redlines),
        profile_path=profile_path.resolve(),
    )
