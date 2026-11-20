"""`EgoCognitiveEngine` —— 快慢双系统三合一本地认知引擎。

职责（对齐《15-mvp-product-spec-aligned.md》）：
1. 启动期强制校验 ``.egopod`` 的 Tier 1 ``cell_digests``、``cognitive_profile_sha256`` 与
   Tier 2 RFC 8785 JCS + Ed25519 签名（默认拒绝未签名或被篡改的模型包，仅在显式开启
   ``allow_unsigned_dev=True`` 时放行未签名开发包，且篡改包一律拒绝）；
2. 同时支持已加载权重的真实神经后端（``peft-*`` / ``mlx`` / ``unsloth``）与开箱 `<20ms` 极速响应的
   **画像锚定本地直觉引擎（Profile-Grounded Local Intuition Engine）**；
3. 暴露三合一标准认知能力：
   - :meth:`EgoCognitiveEngine.health`（``GET /health``）
   - :meth:`EgoCognitiveEngine.intuition`（``POST /v1/intuition`` 直觉快反判别 + 红线秒拒）
   - :meth:`EgoCognitiveEngine.dynamic_context`（``POST /v1/dynamic-context`` 外部大模型动态上下文生成）
   - :meth:`EgoCognitiveEngine.guard_rewrite`（``POST /v1/guard-rewrite`` 外部草稿输出把关改写）
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
import re
from typing import Any

from ego_pod.exit_codes import EgoError, RuntimeError_, VaultError
from ego_pod.runtime.inference import (
    BackendInferenceSession,
    CONFIG_FILENAME,
    DEFAULT_BASE_MODEL,
    DEFAULT_EDITION,
    DEFAULT_QUANTIZATION,
    MANIFEST_FILENAME,
    OUTPUT_DIRNAME,
    STUB_TRAINING_BACKEND,
    _coerce_positive_int,
    _validate_adapter_file,
)
from ego_pod.runtime.selection import CellKey, LoadSelection, select_cells
from ego_pod.service.profile import (
    COGNITIVE_PROFILE_FILENAME,
    CognitiveProfile,
    IntuitionExemplar,
    RedlineRule,
    build_default_cognitive_profile,
    compile_cognitive_profile,
    read_cognitive_profile,
    scrub_pii_and_secrets,
)
from ego_pod.vault.grid_descriptor import parse_grid_descriptor
from ego_pod.vault.signing import (
    ENV_TRUSTED_PUBKEY_HEX,
    _sha256_file_prefixed,
    verify_package_signature,
)

__all__ = [
    "SERVICE_SCHEMA_VERSION",
    "PROFILE_INTUITION_BACKEND",
    "EgoCognitiveEngine",
]

SERVICE_SCHEMA_VERSION = "ego-service/1.0"
PROFILE_INTUITION_BACKEND = "profile-intuition"

_WORD_TOKEN_RE = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]{1,2}", flags=re.IGNORECASE)

_DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "drone-photo": (
        "drone",
        "aerial",
        "gimbal",
        "shutter",
        "iso",
        "aperture",
        "altitude",
        "hover",
        "航拍",
        "无人机",
        "云台",
        "快门",
        "曝光",
        "悬停",
        "构图",
    ),
    "sense": (
        "camera",
        "sensor",
        "battery",
        "gps",
        "ambient",
        "temperature",
        "感知",
        "传感器",
        "环境",
        "电量",
        "观测",
    ),
    "coding": (
        "python",
        "typescript",
        "rust",
        "git",
        "pytest",
        "refactor",
        "function",
        "class",
        "bug",
        "api",
        "sql",
        "代码",
        "函数",
        "重构",
        "单测",
        "接口",
        "模块",
    ),
    "social": (
        "reply",
        "email",
        "message",
        "chat",
        "invite",
        "polite",
        "decline",
        "回复",
        "消息",
        "邮件",
        "群聊",
        "婉拒",
        "措辞",
    ),
    "creative": (
        "story",
        "poem",
        "script",
        "podcast",
        "essay",
        "slogan",
        "文案",
        "脚本",
        "播客",
        "故事",
        "创作",
    ),
    "knowledge": (
        "architecture",
        "principle",
        "summary",
        "note",
        "concept",
        "trade-off",
        "笔记",
        "原理",
        "架构",
        "总结",
        "概念",
    ),
}

_HEAVY_SYSTEM2_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:design\s+a\s+complete|write\s+a\s+full|comprehensive\s+analysis|"
        r"formal\s+proof|prove\s+that|compare\s+and\s+contrast|multi-step|"
        r"implement\s+from\s+scratch|literature\s+review|benchmark\s+report)\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"(?:从零设计|完整实现一套|深度调研|形式化证明|数学推导|全方位对比分析|撰写完整的架构方案|长篇报告)"
    ),
)


def _format_cell_label(cell: CellKey) -> str:
    if cell.is_kernel:
        return "kernel"
    return f"{cell.zone}:L{cell.level}"


def _tokenize_for_match(text: str) -> set[str]:
    cleaned = " ".join(text.lower().strip().split())
    if not cleaned:
        return set()
    tokens = set(_WORD_TOKEN_RE.findall(cleaned))
    # 补充字符二元组以兼顾中英文短句相似度
    compact = re.sub(r"\s+", "", cleaned)
    for idx in range(max(0, len(compact) - 1)):
        tokens.add(compact[idx : idx + 2])
    return tokens


def _similarity_score(query: str, target: str) -> float:
    q_norm = " ".join(query.lower().strip().split())
    t_norm = " ".join(target.lower().strip().split())
    if not q_norm or not t_norm:
        return 0.0
    if q_norm == t_norm:
        return 1.0
    q_tokens = _tokenize_for_match(q_norm)
    t_tokens = _tokenize_for_match(t_norm)
    if not q_tokens or not t_tokens:
        return 0.0
    jaccard = len(q_tokens & t_tokens) / len(q_tokens | t_tokens)
    # 子串包含加成（如用户问句完全命中存储的偏好主题）
    containment = 0.0
    if len(q_norm) >= 4 and (q_norm in t_norm or t_norm in q_norm):
        containment = 0.28
    overlap_recall = len(q_tokens & t_tokens) / max(1, min(len(q_tokens), len(t_tokens)))
    blended = 0.55 * jaccard + 0.45 * overlap_recall + containment
    return min(0.98, blended)


def _infer_domain(prompt: str, explicit_hint: str | None = None) -> str:
    if explicit_hint and explicit_hint.strip():
        return explicit_hint.strip().lower()
    lowered = prompt.lower()
    best_domain = "general"
    best_hits = 0
    for dom, kw_list in _DOMAIN_KEYWORDS.items():
        hits = sum(1 for kw in kw_list if kw in lowered)
        if hits > best_hits:
            best_hits = hits
            best_domain = dom
    return best_domain


def _resolve_package_and_workspace(
    package_or_workspace: str | Path,
) -> tuple[Path, Path | None]:
    target = Path(package_or_workspace)
    if (target / MANIFEST_FILENAME).is_file():
        ws = target.parent if target.name == OUTPUT_DIRNAME else None
        return target, ws
    if (target / OUTPUT_DIRNAME / MANIFEST_FILENAME).is_file():
        return target / OUTPUT_DIRNAME, target
    raise RuntimeError_(
        f"未找到可服务的 Ego_Package（缺少 manifest.json）：{target}"
    )


class EgoCognitiveEngine:
    """`Ego` 三合一本地认知引擎（可由 Python 直接调用，亦可挂载于 `ego serve` HTTP 服务）。"""

    def __init__(
        self,
        package_or_workspace: str | Path,
        *,
        edition: str | None = None,
        selection: LoadSelection | None = None,
        allow_unsigned_dev: bool = False,
        trusted_public_key_hex: str | None = None,
        backend: Any | None = None,
        prefer_neural: bool = True,
    ) -> None:
        self.package_root, self.workspace_root = _resolve_package_and_workspace(
            package_or_workspace
        )
        manifest_path = self.package_root / MANIFEST_FILENAME
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RuntimeError_(
                f"Ego_Package manifest.json 损坏或不可读：{manifest_path}；{exc}"
            ) from exc
        if not isinstance(manifest, dict):
            raise RuntimeError_(f"manifest.json 顶层必须为 JSON 对象：{manifest_path}")
        self.manifest = manifest

        editions = [str(e) for e in manifest.get("editions", []) if isinstance(e, str)]
        if not editions:
            raise RuntimeError_("manifest.editions 为空，无法加载模型包")

        if edition is not None:
            if edition not in editions:
                raise RuntimeError_(
                    f"模型包不包含请求的 Edition {edition!r}（可用：{editions}）"
                )
            resolved_edition = edition
        else:
            slice_ed = (manifest.get("slice_descriptor") or {}).get("selected_edition")
            if isinstance(slice_ed, str) and slice_ed in editions:
                resolved_edition = slice_ed
            elif DEFAULT_EDITION in editions:
                resolved_edition = DEFAULT_EDITION
            else:
                resolved_edition = editions[0]
        self.edition = resolved_edition

        edition_dir = self.package_root / resolved_edition
        cfg_path = edition_dir / CONFIG_FILENAME
        if not cfg_path.is_file():
            raise RuntimeError_(f"缺少 Edition 配置文件：{cfg_path}")
        try:
            config = json.loads(cfg_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RuntimeError_(f"config.json 损坏：{cfg_path}；{exc}") from exc
        if not isinstance(config, dict):
            raise RuntimeError_(f"config.json 顶层必须为 JSON 对象：{cfg_path}")
        self.config = config

        grid = parse_grid_descriptor(manifest.get("grid"), resolved_edition)
        selected_keys = select_cells(grid, selection)

        relpath_by_cell: dict[CellKey, str] = {grid.kernel.cell: grid.kernel.relpath}
        for z_cells in grid.zones.values():
            for gc in z_cells:
                relpath_by_cell[gc.cell] = gc.relpath

        cells_with_paths: list[tuple[CellKey, Path]] = []
        selected_rel_paths: list[str] = []
        for ck in selected_keys:
            rel = relpath_by_cell[ck]
            abs_adapter = edition_dir / rel
            _validate_adapter_file(abs_adapter, f"{_format_cell_label(ck)} -> {rel}")
            cells_with_paths.append((ck, abs_adapter))
            selected_rel_paths.append(Path(resolved_edition, rel).as_posix())

        # 启动期强制签名与防篡改校验
        eff_pubkey = (
            trusted_public_key_hex
            if trusted_public_key_hex is not None
            else (os.environ.get(ENV_TRUSTED_PUBKEY_HEX) or None)
        )
        try:
            ver = verify_package_signature(
                self.package_root,
                trusted_public_key_hex=eff_pubkey,
                selected_cell_rel_paths=selected_rel_paths,
            )
        except VaultError as exc:
            raise RuntimeError_(
                f"拒绝启动：Ego_Package 签名或防篡改完整性校验失败 —— {exc}"
            ) from exc

        if ver.status == "unsigned":
            if not allow_unsigned_dev:
                raise RuntimeError_(
                    "拒绝启动：Ego_Package 缺少 Ed25519 签名（生产默认强制验签；"
                    "若处于本地开发调试请显式传入 --allow-unsigned-dev）"
                )
            # 即便显式允许未签名开发包，若包内已有 integrity.cell_digests 仍强制核对哈希防损坏/篡改
            integrity_block = manifest.get("integrity")
            if isinstance(integrity_block, Mapping) and isinstance(
                integrity_block.get("cell_digests"), Mapping
            ):
                digests = integrity_block["cell_digests"]
                for rel_posix in selected_rel_paths:
                    expected = digests.get(rel_posix)
                    if expected is not None:
                        actual = _sha256_file_prefixed(self.package_root / rel_posix)
                        if actual != expected:
                            raise RuntimeError_(
                                f"拒绝启动：开发包适配器摘要已被篡改（{rel_posix}）"
                            )
            self.signer_fay_id = "unsigned-dev"
        else:
            self.signer_fay_id = ver.signer_fay_id or "verified"

        self.active_cells: tuple[CellKey, ...] = tuple(selected_keys)
        self.active_cell_labels: tuple[str, ...] = tuple(
            _format_cell_label(ck) for ck in selected_keys
        )
        self.slice_descriptor: dict[str, Any] | None = (
            dict(manifest["slice_descriptor"])
            if isinstance(manifest.get("slice_descriptor"), Mapping)
            else None
        )

        # 加载结构化认知画像
        self.profile: CognitiveProfile = self._load_cognitive_profile()

        # 按需连接真实神经后端（peft / mlx），若基座未下载或为 stub 则零延迟启用直觉画像引擎
        self.neural_session: BackendInferenceSession | None = None
        self.backend_name: str = PROFILE_INTUITION_BACKEND
        training_backend = str(manifest.get("training_backend", STUB_TRAINING_BACKEND))

        if prefer_neural and (
            backend is not None or training_backend != STUB_TRAINING_BACKEND
        ):
            lora_rank = _coerce_positive_int(
                config.get("lora_rank", 16), "lora_rank", str(cfg_path)
            )
            lora_alpha = _coerce_positive_int(
                config.get("lora_alpha", 32), "lora_alpha", str(cfg_path)
            )
            try:
                self.neural_session = BackendInferenceSession(
                    base_model=str(config.get("base_model", DEFAULT_BASE_MODEL)),
                    quantization=str(config.get("quantization", DEFAULT_QUANTIZATION)),
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    cells=cells_with_paths,
                    identifier=(
                        training_backend
                        if training_backend != STUB_TRAINING_BACKEND
                        else None
                    ),
                    backend=backend,
                )
                self.backend_name = str(
                    getattr(self.neural_session.backend, "identifier", None)
                    or training_backend
                )
            except (EgoError, ImportError, OSError, RuntimeError):
                if backend is not None:
                    # 显式注入了后端对象却加载失败时向上抛错
                    raise
                self.neural_session = None
                self.backend_name = PROFILE_INTUITION_BACKEND

    def _load_cognitive_profile(self) -> CognitiveProfile:
        pod_prof = self.package_root / COGNITIVE_PROFILE_FILENAME
        if pod_prof.is_file():
            return read_cognitive_profile(pod_prof)

        if self.workspace_root is not None:
            ws_prof = self.workspace_root / ".ego" / COGNITIVE_PROFILE_FILENAME
            if ws_prof.is_file():
                return read_cognitive_profile(ws_prof)
            corpus_file = self.workspace_root / ".ego" / "corpus.jsonl"
            if corpus_file.is_file():
                records: list[dict[str, Any]] = []
                for line in corpus_file.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            records.append(obj)
                    except json.JSONDecodeError:
                        continue
                if records:
                    return compile_cognitive_profile(records)

        return build_default_cognitive_profile()

    def close(self) -> None:
        """释放可能挂载的神经推理会话资源。"""

        if self.neural_session is not None:
            self.neural_session.close()
            self.neural_session = None

    # -----------------------------------------------------------------------
    # 1. GET /health
    # -----------------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        """返回标准健康状态与当前挂载规格、签名者、激活 Cell 清单。"""

        return {
            "status": "ok",
            "schema_version": SERVICE_SCHEMA_VERSION,
            "signer_fay_id": self.signer_fay_id,
            "edition": self.edition,
            "backend": self.backend_name,
            "active_cells": list(self.active_cell_labels),
        }

    # -----------------------------------------------------------------------
    # 2. POST /v1/intuition —— ① 直觉快反判别
    # -----------------------------------------------------------------------

    def _check_redlines(self, text: str) -> list[RedlineRule]:
        return [rule for rule in self.profile.redlines if rule.matches(text)]

    def _is_heavy_system2_task(self, prompt: str) -> bool:
        stripped = prompt.strip()
        if len(stripped) > 360 or stripped.count("\n") >= 8:
            return True
        return any(pat.search(stripped) for pat in _HEAVY_SYSTEM2_PATTERNS)

    def _match_best_exemplar(
        self, prompt: str, domain_hint: str
    ) -> tuple[IntuitionExemplar | None, float]:
        best_ex: IntuitionExemplar | None = None
        best_score = 0.0
        for ex in self.profile.exemplars:
            raw_sim = _similarity_score(prompt, ex.prompt)
            if raw_sim <= 0.0:
                continue
            domain_bonus = (
                0.06
                if (ex.domain == domain_hint and domain_hint != "general")
                else 0.0
            )
            weight_bonus = min(0.08, max(0.0, (ex.weight - 1.0) * 0.03))
            score = min(0.99, raw_sim + domain_bonus + weight_bonus)
            if score > best_score:
                best_score = score
                best_ex = ex
        return best_ex, best_score

    def intuition(
        self,
        prompt: str,
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """执行 System-1 直觉快反判别：红线秒拒 / 本地高置信秒回 / 复杂任务上浮外部大模型。"""

        clean_prompt = str(prompt or "").strip()
        ctx = dict(context) if isinstance(context, Mapping) else {}
        if not clean_prompt:
            return {
                "handled_locally": True,
                "confidence": 1.0,
                "reply": "",
                "matched_domain": "general",
                "escalate_to_llm": False,
                "refused": False,
                "reason": "empty_prompt",
            }

        # 1) 硬红线秒拒（绝不外发给外部大模型，离线/在线一致）
        combined_check_text = clean_prompt
        if isinstance(ctx.get("command"), str):
            combined_check_text = f"{clean_prompt}\n{ctx['command']}"
        triggered = self._check_redlines(combined_check_text)
        if triggered:
            primary = triggered[0]
            return {
                "handled_locally": True,
                "confidence": 0.99,
                "reply": (
                    f"[Ego Redline Guard] Refused per Human Prime redline "
                    f"({primary.rule_id}): {primary.description}"
                ),
                "matched_domain": "redline",
                "escalate_to_llm": False,
                "refused": True,
                "reason": f"redline_violation:{primary.rule_id}",
                "triggered_redlines": [r.rule_id for r in triggered],
            }

        explicit_domain = (
            str(ctx["domain"])
            if isinstance(ctx.get("domain"), str)
            else (str(ctx["task_type"]) if isinstance(ctx.get("task_type"), str) else None)
        )
        matched_domain = _infer_domain(clean_prompt, explicit_domain)

        # 2) 若当前运行的是面向受限终端的独立切片包，且请求落入切片之外的其他专精领域 -> 上浮主体/外部大模型
        if self.slice_descriptor is not None:
            sliced_domains = {
                str(d).lower()
                for d in self.slice_descriptor.get("selected_domains", [])
            }
            expanded_allowed = set(sliced_domains) | {"general"}
            if "drone-photo" in expanded_allowed:
                expanded_allowed.add("sense")
            if (
                matched_domain not in expanded_allowed
                and matched_domain in _DOMAIN_KEYWORDS
            ):
                return {
                    "handled_locally": False,
                    "confidence": 0.15,
                    "reply": "",
                    "matched_domain": matched_domain,
                    "escalate_to_llm": True,
                    "refused": False,
                    "reason": (
                        f"domain_outside_slice:{matched_domain} not in {sorted(sliced_domains)}"
                    ),
                }

        # 3) 检索 Pod 内编译的高权重个人偏好与纠偏样本（human_correction / prime_turn）
        best_ex, ex_score = self._match_best_exemplar(clean_prompt, matched_domain)
        is_heavy = self._is_heavy_system2_task(clean_prompt)

        # 若为极其明确的纠偏对命中（>= 0.80），即便句子稍长也直接按主人纠偏定论秒回
        if best_ex is not None and (
            ex_score >= 0.80 or (not is_heavy and ex_score >= 0.52)
        ):
            return {
                "handled_locally": True,
                "confidence": round(ex_score, 4),
                "reply": best_ex.reply,
                "matched_domain": best_ex.domain,
                "escalate_to_llm": False,
                "refused": False,
                "reason": f"matched_{best_ex.source}",
            }

        # 4) 重度 System-2 推理/跨领域综合长任务 -> 交由外部大模型 + dynamic-context 引导
        if is_heavy:
            return {
                "handled_locally": False,
                "confidence": round(min(0.35, ex_score), 4),
                "reply": "",
                "matched_domain": matched_domain,
                "escalate_to_llm": True,
                "refused": False,
                "reason": "requires_system2_deep_reasoning",
            }

        # 5) 常规个人风格/准则直觉询问（命中 style_anchors 或 decision_heuristics）
        best_rule: str | None = None
        best_rule_score = 0.0
        for anchor in (*self.profile.style_anchors, *self.profile.decision_heuristics):
            sc = _similarity_score(clean_prompt, anchor)
            if sc > best_rule_score:
                best_rule_score = sc
                best_rule = anchor
        if best_rule is not None and best_rule_score >= 0.55:
            return {
                "handled_locally": True,
                "confidence": round(max(0.72, best_rule_score), 4),
                "reply": best_rule,
                "matched_domain": matched_domain,
                "escalate_to_llm": False,
                "refused": False,
                "reason": "matched_personal_heuristic",
            }

        # 6) 若真实神经后端权重已加载且请求属于当前激活网格内的短平快日常交互，直接本地神经生成！
        active_zones = {ck.zone for ck in self.active_cells if ck.zone is not None}
        if self.neural_session is not None and (
            matched_domain in active_zones
            or matched_domain in ("general", "sense")
            or bool(ctx.get("offline"))
        ):
            raw_reply = self.neural_session.generate(clean_prompt)
            guarded = self.guard_rewrite(clean_prompt, raw_reply)
            return {
                "handled_locally": True,
                "confidence": 0.84,
                "reply": guarded["rewritten_reply"],
                "matched_domain": matched_domain,
                "escalate_to_llm": False,
                "refused": not guarded["approved"] and any(
                    v.get("type") == "redline" for v in guarded["violations"]
                ),
                "reason": f"neural_local_inference:{self.backend_name}",
            }

        # 7) 离线断网兜底模式（context.offline == True）：利用本地人格准则给出确定性离线连续性回复
        if bool(ctx.get("offline")):
            anchor_tip = (
                self.profile.decision_heuristics[0]
                if self.profile.decision_heuristics
                else self.profile.persona_summary
            )
            return {
                "handled_locally": True,
                "confidence": 0.65,
                "reply": f"[Offline Ego Continuity] {anchor_tip}",
                "matched_domain": matched_domain,
                "escalate_to_llm": False,
                "refused": False,
                "reason": "offline_continuity_fallback",
            }

        # 8) 未命中本地高置信直觉 -> 诚实上浮至外部大模型
        return {
            "handled_locally": False,
            "confidence": round(ex_score, 4),
            "reply": "",
            "matched_domain": matched_domain,
            "escalate_to_llm": True,
            "refused": False,
            "reason": "unfamiliar_query_escalate_to_external_llm",
        }

    # -----------------------------------------------------------------------
    # 3. POST /v1/dynamic-context —— ② 外部大模型动态上下文生成
    # -----------------------------------------------------------------------

    def dynamic_context(
        self,
        prompt: str,
        task_type: str | None = None,
    ) -> dict[str, Any]:
        """在调用外部大模型前生成结构化的 `Human Prime` 思维锚点与红线上下文。"""

        clean_prompt = scrub_pii_and_secrets(str(prompt or "")).text
        domain = _infer_domain(clean_prompt, task_type)

        # 按与当前任务的相关度对 style_anchors 与 decision_heuristics 排序
        ranked_styles = sorted(
            self.profile.style_anchors,
            key=lambda s: (-_similarity_score(clean_prompt, s), s),
        )[:6]
        ranked_heuristics = sorted(
            self.profile.decision_heuristics,
            key=lambda h: (-_similarity_score(clean_prompt, h), h),
        )[:6]

        # 提取与当前领域最相关的最多 3 条高权重偏好（如 human_correction 教训）注入启发式准则
        domain_exemplars = [
            ex
            for ex in self.profile.exemplars
            if ex.domain in (domain, "general") or ex.source == "human_correction"
        ]
        domain_exemplars.sort(
            key=lambda ex: (
                -(_similarity_score(clean_prompt, ex.prompt) + ex.weight * 0.1),
                ex.prompt,
            )
        )
        for ex in domain_exemplars[:3]:
            note = f"Learned preference ({ex.domain}): {ex.reply}"
            if len(note) <= 220 and note not in ranked_heuristics:
                ranked_heuristics.append(note)

        redline_texts = [
            scrub_pii_and_secrets(r.description).text for r in self.profile.redlines
        ]

        directive_lines = [
            f"[iFay Ego Persona Steering | signer={self.signer_fay_id} | domain={domain}]",
            f"Persona: {self.profile.persona_summary}",
            "Style Anchors:",
            *[f"- {item}" for item in ranked_styles],
            "Decision Heuristics:",
            *[f"- {item}" for item in ranked_heuristics],
            "Non-Negotiable Redlines (MUST NOT violate):",
            *[f"- {rl}" for rl in redline_texts],
        ]

        return {
            "system_directive": "\n".join(directive_lines),
            "style_anchors": ranked_styles,
            "decision_heuristics": ranked_heuristics,
            "redlines": redline_texts,
            "active_cells": list(self.active_cell_labels),
        }

    # -----------------------------------------------------------------------
    # 4. POST /v1/guard-rewrite —— ③ 输出把关改写
    # -----------------------------------------------------------------------

    def guard_rewrite(
        self,
        prompt: str,
        draft_reply: str,
    ) -> dict[str, Any]:
        """审计外部大模型的草稿回复：拦截红线越界、自动擦除隐私密钥泄露并去除非主人 AI 腔套话。"""

        raw_prompt = str(prompt or "")
        raw_draft = str(draft_reply or "")
        violations: list[dict[str, str]] = []

        # 1) 检查草稿自身是否触犯红线，或原请求触犯红线而外部大模型草稿未拒绝
        draft_redlines = self._check_redlines(raw_draft)
        prompt_redlines = self._check_redlines(raw_prompt)
        draft_already_refuses = bool(
            re.search(
                r"(?:\brefuse\b|\bcannot\b|\bcan't\b|\bwon't\b|拒绝|无法执行|不能执行|红线)",
                raw_draft,
                flags=re.IGNORECASE,
            )
        )

        active_redline_hits = list(draft_redlines)
        if prompt_redlines and not draft_already_refuses:
            for pr in prompt_redlines:
                if all(pr.rule_id != r.rule_id for r in active_redline_hits):
                    active_redline_hits.append(pr)

        if active_redline_hits:
            for rl in active_redline_hits:
                violations.append(
                    {
                        "type": "redline",
                        "rule_id": rl.rule_id,
                        "description": rl.description,
                    }
                )
            primary = active_redline_hits[0]
            safe_refusal = (
                f"[Blocked by Ego Guard] Draft violated Human Prime redline "
                f"({primary.rule_id}): {primary.description}"
            )
            return {
                "approved": False,
                "rewritten_reply": safe_refusal,
                "violations": violations,
            }

        rewritten = raw_draft

        # 2) 凭证与个人隐私（PII）脱敏把关
        scrub = scrub_pii_and_secrets(rewritten, category="credentials")
        if scrub.total_redactions > 0:
            violations.append(
                {
                    "type": "pii_hygiene",
                    "rule_id": "hygiene.pii_and_secrets",
                    "description": (
                        f"Redacted {scrub.total_redactions} sensitive credential/PII item(s)"
                    ),
                }
            )
            rewritten = scrub.text

        # 3) 语气与反 AI 腔（anti_tone_markers）把关改写
        matched_markers: list[str] = []
        for marker in self.profile.anti_tone_markers:
            if not marker:
                continue
            pattern = re.compile(re.escape(marker), flags=re.IGNORECASE)
            if pattern.search(rewritten):
                matched_markers.append(marker)
                rewritten = pattern.sub("", rewritten)

        if matched_markers:
            # 清理剥离套话后可能残留的前导标点与多余空行
            rewritten = re.sub(r"^[\s,;:!，。！？：；]+", "", rewritten).strip()
            if rewritten and rewritten[0].islower() and rewritten[0].isascii():
                rewritten = rewritten[0].upper() + rewritten[1:]
            violations.append(
                {
                    "type": "tone_mismatch",
                    "rule_id": "style.anti_tone_marker",
                    "description": (
                        f"Stripped non-Prime boilerplate ({len(matched_markers)} marker(s))"
                    ),
                }
            )

        return {
            "approved": len(violations) == 0,
            "rewritten_reply": rewritten,
            "violations": violations,
        }
