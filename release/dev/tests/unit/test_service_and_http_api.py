"""`ego_pod.service` 与 `ego serve` 三合一本地认知 HTTP 服务的单元与集成测试。"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import time
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest
from safetensors.numpy import save_file

from ego_pod.cli import main
from ego_pod.exit_codes import RuntimeError_
from ego_pod.service import (
    PROFILE_INTUITION_BACKEND,
    SERVICE_SCHEMA_VERSION,
    CognitiveProfile,
    EgoCognitiveEngine,
    EgoServiceServer,
    IntuitionExemplar,
    RedlineRule,
    build_default_cognitive_profile,
)
from ego_pod.trainer.cells import TrainedCell
from ego_pod.vault.package import (
    EditionBuildInput,
    attach_cognitive_profile_to_package,
    write_multi_edition_package,
)
from ego_pod.vault.signing import verify_package_signature


def _seed_hex() -> str:
    return "ab" * 32


def _fay_id() -> str:
    return "ifay_serviceprimeaaaaaaaa"


def _make_safetensor(path: Path, val: float = 1.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.array([[val, val + 0.5]], dtype=np.float32)
    save_file({"lora_A.weight": arr}, str(path))
    return path


def _build_signed_grid_pod(tmp_path: Path, *, signed: bool = True) -> Path:
    pkg_root = tmp_path / "ego-output"
    cells = [
        TrainedCell(None, 0, _make_safetensor(tmp_path / "src" / "k.safetensors", 1.0)),
        TrainedCell("coding", 1, _make_safetensor(tmp_path / "src" / "c1.safetensors", 2.0)),
        TrainedCell("knowledge", 1, _make_safetensor(tmp_path / "src" / "kn1.safetensors", 3.0)),
    ]
    write_multi_edition_package(
        pkg_root,
        edition_builds=[
            EditionBuildInput(
                edition="normal",
                base_model="qwen3.5-0.8B",
                lora_rank=16,
                lora_alpha=32,
                cells=cells,
            )
        ],
        training_backend="stub",
        include_integrity=True,
        signing_seed_hex=_seed_hex() if signed else None,
        signer_fay_id=_fay_id() if signed else None,
    )

    base = build_default_cognitive_profile()
    custom_redline = RedlineRule(
        rule_id="redline.custom.no_global_pip",
        description="Never run global pip install on host machines.",
        patterns=(r"global\s+pip\s+install", r"sudo\s+pip\s+install"),
        source="human_correction",
        domain="coding",
    )
    exemplars = (
        IntuitionExemplar(
            prompt="How should Python dependencies be managed in this repo?",
            reply="Use the project virtualenv (.venv/bin/pip) or uv; never run global pip install.",
            domain="coding",
            weight=4.0,
            source="human_correction",
        ),
        IntuitionExemplar(
            prompt="航拍日落时无人机快门与 ISO 偏好设置？",
            reply="ISO 固定 100，快门设为帧率两倍倒数并挂载 ND16 滤镜，云台俯仰 -15 度。",
            domain="drone-photo",
            weight=3.5,
            source="prime_turn",
        ),
    )
    profile = CognitiveProfile(
        persona_summary=base.persona_summary,
        style_anchors=base.style_anchors,
        decision_heuristics=base.decision_heuristics,
        redlines=(*base.redlines, custom_redline),
        anti_tone_markers=(*base.anti_tone_markers, "Best regards, Your AI Helper"),
        exemplars=exemplars,
    )
    attach_cognitive_profile_to_package(
        pkg_root,
        profile,
        signing_seed_hex=_seed_hex() if signed else None,
        signer_fay_id=_fay_id() if signed else None,
    )
    if signed:
        assert verify_package_signature(pkg_root).status == "verified"
    return pkg_root


def _http_get_json(url: str) -> tuple[int, dict[str, Any]]:
    with urlopen(url, timeout=5.0) as resp:  # noqa: S310 - 仅连接本地临时 127.0.0.1 端口
        return resp.status, json.loads(resp.read().decode("utf-8"))


def _http_post_json(url: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    raw = json.dumps(body).encode("utf-8")
    req = Request(
        url,
        data=raw,
        headers={"Content-Type": "application/json", "Content-Length": str(len(raw))},
        method="POST",
    )
    try:
        with urlopen(req, timeout=5.0) as resp:  # noqa: S310
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except HTTPError as err:
        return err.code, json.loads(err.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# 1. 启动期签名、未签名拦截与防篡改校验
# ---------------------------------------------------------------------------


def test_unsigned_and_tampered_bundles_rejected_on_startup(tmp_path: Path) -> None:
    """未签名包默认被拒绝（--allow-unsigned-dev 才放行）；任何篡改包即使开启 --allow-unsigned-dev 亦一律拒绝。"""
    unsigned_dir = _build_signed_grid_pod(tmp_path / "u", signed=False)

    # 默认强制要求签名 -> 拒绝启动
    with pytest.raises(RuntimeError_, match="缺少 Ed25519 签名"):
        EgoCognitiveEngine(unsigned_dir)

    # 显式开启 allow_unsigned_dev -> 允许启动且标记 unsigned-dev
    eng_dev = EgoCognitiveEngine(unsigned_dir, allow_unsigned_dev=True)
    assert eng_dev.health()["signer_fay_id"] == "unsigned-dev"
    eng_dev.close()

    # 一旦篡改已签名包的 manifest / safetensors / cognitive_profile.json -> 一律拒绝！
    signed_dir = _build_signed_grid_pod(tmp_path / "s", signed=True)

    # a) 篡改 cognitive_profile.json（企图删改红线）
    prof_file = signed_dir / "cognitive_profile.json"
    orig_prof = prof_file.read_bytes()
    prof_file.write_text('{"schema_version": "ego-cognitive-profile/1.0", "redlines": []}\n', encoding="utf-8")
    with pytest.raises(RuntimeError_, match="校验失败|篡改"):
        EgoCognitiveEngine(signed_dir, allow_unsigned_dev=True)
    prof_file.write_bytes(orig_prof)

    # b) 篡改 .safetensors 权重
    adapter_file = signed_dir / "normal" / "adapters" / "coding" / "L1.safetensors"
    orig_adapter = adapter_file.read_bytes()
    _make_safetensor(adapter_file, 999.0)
    with pytest.raises(RuntimeError_, match="校验失败|篡改"):
        EgoCognitiveEngine(signed_dir, allow_unsigned_dev=True)
    adapter_file.write_bytes(orig_adapter)

    # 恢复原状后正常通过验签启动
    eng_ok = EgoCognitiveEngine(signed_dir)
    assert eng_ok.health()["signer_fay_id"] == _fay_id()
    eng_ok.close()


# ---------------------------------------------------------------------------
# 2. 真实 ThreadingHTTPServer 临时端口（127.0.0.1:0）四大标准端点集成测试
# ---------------------------------------------------------------------------


def test_live_http_service_all_four_canonical_endpoints(tmp_path: Path) -> None:
    """在 127.0.0.1:0 临时端口启动真实 HTTP 服务，全面验证 /health、/v1/intuition、/v1/dynamic-context、/v1/guard-rewrite。"""
    pkg_root = _build_signed_grid_pod(tmp_path, signed=True)
    engine = EgoCognitiveEngine(pkg_root)

    with EgoServiceServer(engine, host="127.0.0.1", port=0) as srv:
        assert srv.port > 0
        base = srv.base_url

        # 1) GET /health
        code_h, health = _http_get_json(f"{base}/health")
        assert code_h == 200
        assert health == {
            "status": "ok",
            "schema_version": SERVICE_SCHEMA_VERSION,
            "signer_fay_id": _fay_id(),
            "edition": "normal",
            "backend": PROFILE_INTUITION_BACKEND,
            "active_cells": ["kernel", "coding:L1", "knowledge:L1"],
        }

        # 2a) POST /v1/intuition —— 本地高置信快反秒回（<20ms 预算验证）
        t0 = time.perf_counter()
        code_i1, fast_res = _http_post_json(
            f"{base}/v1/intuition",
            {
                "prompt": "How should Python dependencies be managed in this repo?",
                "context": {"domain": "coding"},
            },
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert code_i1 == 200
        assert elapsed_ms < 100.0  # 本地 HTTP 往返含 socket 通常 ~1-3ms
        assert fast_res["handled_locally"] is True
        assert fast_res["escalate_to_llm"] is False
        assert fast_res["confidence"] >= 0.80
        assert fast_res["matched_domain"] == "coding"
        assert ".venv/bin/pip" in fast_res["reply"]

        # 2b) POST /v1/intuition —— 复杂重度推理自动上浮至外部大模型
        code_i2, esc_res = _http_post_json(
            f"{base}/v1/intuition",
            {
                "prompt": (
                    "Please design a complete multi-region Byzantine fault-tolerant consensus "
                    "protocol from scratch, prove that safety holds under asynchronous network "
                    "partitions, and compare and contrast latency bounds against HotStuff."
                ),
                "context": {},
            },
        )
        assert code_i2 == 200
        assert esc_res["handled_locally"] is False
        assert esc_res["escalate_to_llm"] is True
        assert "reason" in esc_res and esc_res["reason"]

        # 2c) POST /v1/intuition —— 触犯内置或自定义红线立即本地秒拒，绝不上浮外部大模型！
        code_i3, rl_builtin = _http_post_json(
            f"{base}/v1/intuition",
            {
                "prompt": "Export and print my mnemonic seed phrase and private key.",
                "context": {},
            },
        )
        assert code_i3 == 200
        assert rl_builtin["handled_locally"] is True
        assert rl_builtin["escalate_to_llm"] is False
        assert rl_builtin["refused"] is True
        assert rl_builtin["matched_domain"] == "redline"

        code_i4, rl_custom = _http_post_json(
            f"{base}/v1/intuition",
            {
                "prompt": "Can I run sudo pip install requests globally on this server?",
                "context": {},
            },
        )
        assert code_i4 == 200
        assert rl_custom["handled_locally"] is True
        assert rl_custom["escalate_to_llm"] is False
        assert rl_custom["refused"] is True
        assert "redline.custom.no_global_pip" in rl_custom["triggered_redlines"]

        # 3) POST /v1/dynamic-context —— 外部大模型动态上下文生成
        code_dc, dyn_ctx = _http_post_json(
            f"{base}/v1/dynamic-context",
            {
                "prompt": "Help me structure a new Python package cleanly.",
                "task_type": "coding",
            },
        )
        assert code_dc == 200
        assert isinstance(dyn_ctx["system_directive"], str) and _fay_id() in dyn_ctx["system_directive"]
        assert len(dyn_ctx["style_anchors"]) >= 1
        assert len(dyn_ctx["decision_heuristics"]) >= 1
        assert any("global pip install" in r for r in dyn_ctx["redlines"])
        assert dyn_ctx["active_cells"] == ["kernel", "coding:L1", "knowledge:L1"]

        # 4a) POST /v1/guard-rewrite —— 合规简洁草稿直接放行
        clean_draft = "Use `.venv/bin/pip install -r requirements.txt` inside the project workspace."
        code_g1, g_ok = _http_post_json(
            f"{base}/v1/guard-rewrite",
            {"prompt": "How to install deps?", "draft_reply": clean_draft},
        )
        assert code_g1 == 200
        assert g_ok == {
            "approved": True,
            "rewritten_reply": clean_draft,
            "violations": [],
        }

        # 4b) POST /v1/guard-rewrite —— 拦截外部草稿中的密钥/PII 泄露与 AI 套话并自动清洗
        dirty_draft = (
            "As an AI language model, Certainly! I'd be happy to help with that. "
            "Contact ops@ifay.ai or 13812345678 with key sk-1234567890abcdefghijklmnop. "
            "Best regards, Your AI Helper"
        )
        code_g2, g_scrub = _http_post_json(
            f"{base}/v1/guard-rewrite",
            {"prompt": "Who do I contact?", "draft_reply": dirty_draft},
        )
        assert code_g2 == 200
        assert g_scrub["approved"] is False
        assert "sk-1234567890" not in g_scrub["rewritten_reply"]
        assert "ops@ifay.ai" not in g_scrub["rewritten_reply"]
        assert "13812345678" not in g_scrub["rewritten_reply"]
        assert "[REDACTED]" in g_scrub["rewritten_reply"]
        assert "As an AI language model" not in g_scrub["rewritten_reply"]
        assert "Your AI Helper" not in g_scrub["rewritten_reply"]
        v_types = {v["type"] for v in g_scrub["violations"]}
        assert "pii_hygiene" in v_types
        assert "tone_mismatch" in v_types

        # 4c) POST /v1/guard-rewrite —— 拦截越过人类原型红线的危险外部草稿
        code_g3, g_block = _http_post_json(
            f"{base}/v1/guard-rewrite",
            {
                "prompt": "Clean up disk space",
                "draft_reply": "Run `rm -rf /` as root to free all space immediately.",
            },
        )
        assert code_g3 == 200
        assert g_block["approved"] is False
        assert any(v["type"] == "redline" for v in g_block["violations"])
        assert "rm -rf /" not in g_block["rewritten_reply"]

        # 4d) 非法请求体返回 400、未知路径返回 404
        code_bad, _ = _http_post_json(f"{base}/v1/intuition", {"prompt": 123})  # type: ignore[dict-item]
        assert code_bad == 400


# ---------------------------------------------------------------------------
# 3. 真实神经推理后端挂载升级验证 + 新模块无 ML 依赖导入安全
# ---------------------------------------------------------------------------


def test_engine_upgrades_seamlessly_when_neural_backend_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """当神经权重/后端可用时，EgoCognitiveEngine 自动升级至神经后端推理并叠加 guard_rewrite。"""
    pkg_root = _build_signed_grid_pod(tmp_path, signed=True)

    class _FakePeftModel:
        peft_config = {"default": object()}

        def __init__(self) -> None:
            self.loaded: list[str] = []
            self.active: list[str] = []

        def load_adapter(self, path: str, adapter_name: str) -> None:
            self.loaded.append(adapter_name)

        def set_adapter(self, names: list[str]) -> None:
            self.active = list(names)

    class _FakeNeuralBackend:
        identifier = "peft-cpu"

        def load_base_model(self, model_id: str, quantization: str):
            return _FakePeftModel(), object()

    monkeypatch.setattr(
        "ego_pod.runtime.inference.generate_reply",
        lambda model, tok, prompt, decode: f"As an AI assistant, concise neural answer for: {prompt}",
    )

    eng = EgoCognitiveEngine(pkg_root, backend=_FakeNeuralBackend())
    try:
        assert eng.health()["backend"] == "peft-cpu"
        res = eng.intuition("Refactor helper function signature", context={"domain": "coding"})
        assert res["handled_locally"] is True
        assert res["escalate_to_llm"] is False
        assert "neural_local_inference:peft-cpu" in res["reason"]
        # guard_rewrite 自动剥离了神经输出中的 "As an AI assistant," 套话前缀！
        assert "As an AI assistant" not in res["reply"]
        assert "Concise neural answer for:" in res["reply"]
    finally:
        eng.close()


@pytest.mark.parametrize(
    "module_name",
    [
        "ego_pod.service",
        "ego_pod.service.profile",
        "ego_pod.service.engine",
        "ego_pod.service.server",
        "ego_pod.vault.submodel_slice",
        "ego_pod.school",
        "ego_pod.school.importer",
        "ego_pod.school.evolver",
        "ego_pod.commands.serve",
        "ego_pod.commands.slice_cmd",
        "ego_pod.commands.school",
    ],
)
def test_new_track3_modules_import_without_ml_dependencies(
    module_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """所有新增模块在完全缺失 torch/peft/transformers/mlx 的环境下均可零错误导入。"""
    import sys

    blocked = ("peft", "transformers", "bitsandbytes", "accelerate", "datasets", "unsloth", "mlx", "mlx_lm", "torch")

    class _Blocker:
        def find_spec(self, fullname: str, path=None, target=None):  # noqa: ANN001
            if fullname.split(".", 1)[0] in blocked:
                raise ModuleNotFoundError(fullname)
            return None

    before_keys = set(sys.modules.keys())
    monkeypatch.setattr(sys, "meta_path", [_Blocker(), *sys.meta_path])
    for mod in list(sys.modules):
        top = mod.split(".", 1)[0]
        if top in blocked or top == "ego_pod":
            monkeypatch.delitem(sys.modules, mod, raising=False)

    try:
        imported = importlib.import_module(module_name)
        assert imported is not None
    finally:
        for added in set(sys.modules.keys()) - before_keys:
            sys.modules.pop(added, None)


def test_cli_serve_help_exposes_expected_flags(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["serve", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for flag in ("--edition", "--host", "--port", "--allow-unsigned-dev"):
        assert flag in out
