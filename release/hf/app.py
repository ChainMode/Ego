"""Hugging Face Space interactive demo for Ego (`ego-pod` v0.2.0).

Provides four interactive tabs:
1. `.egopod` / `manifest.json` Inspector (aligned byte-for-byte with `ego_pod.vault`)
2. 6-Axis Behavioral Profile & `L1..L3` Sparse-Grid Explorer
3. Two-Tier Cryptographic Signature Verifier (`RFC 8785 JCS + Ed25519`, `ifay_` FayID)
4. Stub vs. LoRA Adapter (`Kernel-only` vs. `Full_In_Load`) Comparison
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
import re
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

SIGNING_DOMAIN = "ifay.ego.package-signing.v1"
HKDF_SALT = b"ifay.fayid.hkdf.v1"
HKDF_INFO = b"ifay.ego.package-signing.v1"
CANONICAL_IFAY_ID_RE = re.compile(r"^ifay_[a-z2-7]{16,}$")

SIX_AXES = (
    (
        "D1 — Tone & Expression",
        "sense / thought",
        "Kernel (L0)",
        "Concise, structured prose; direct conclusions before supporting detail.",
    ),
    (
        "D2 — Habits & Routines",
        "sense / action",
        "Kernel (L0) + L1",
        "07:00 black coffee, uninterrupted deep-work block until 10:30, async batch reviews.",
    ),
    (
        "D3 — Reasoning Style",
        "thought",
        "knowledge (L1..L3)",
        "Explicit trade-off matrices, invariants first, deterministic failure modes.",
    ),
    (
        "D4 — Domain Knowledge",
        "thought",
        "knowledge (L1..L3)",
        "Distributed systems, local-first cryptography, privacy-preserving ML pipelines.",
    ),
    (
        "D5 — Technical & Craft Skills",
        "skill / action",
        "coding (L1..L3)",
        "Python 3.12 type-safe dataclasses, Hypothesis property tests, SafeTensors-only weights.",
    ),
    (
        "D6 — Boundary & CAP Stance",
        "credentials (PII-scrubbed)",
        "Kernel (L0)",
        "Never leak secrets; external transfers or destructive ops require explicit Human Prime CAP approval.",
    ),
)


def _format_jcs_float(value: float) -> str:
    if math.isnan(value) or math.isinf(value):
        raise ValueError("RFC 8785 JCS disallows NaN/Inf floats")
    if value == 0.0:
        return "0"
    sign = "-" if value < 0.0 else ""
    text = repr(abs(value)).lower()
    if "e" in text:
        mantissa, exp_str = text.split("e", 1)
        exp = int(exp_str)
    else:
        mantissa, exp = text, 0
    if "." in mantissa:
        int_part, frac_part = mantissa.split(".", 1)
        raw_digits = int_part + frac_part
        exp -= len(frac_part)
    else:
        raw_digits = mantissa
    stripped = raw_digits.lstrip("0")
    digits = stripped.rstrip("0")
    exp += len(stripped) - len(digits)
    k = len(digits)
    n = exp + k
    if k <= n <= 21:
        body = digits + ("0" * (n - k))
    elif 0 < n <= 21:
        body = f"{digits[:n]}.{digits[n:]}"
    elif -6 < n <= 0:
        body = "0." + ("0" * (-n)) + digits
    else:
        exponent_val = n - 1
        exp_sign = "+" if exponent_val >= 0 else "-"
        coeff = digits if k == 1 else f"{digits[0]}.{digits[1:]}"
        body = f"{coeff}e{exp_sign}{abs(exponent_val)}"
    return f"{sign}{body}"


def canonicalize_jcs(obj: Any) -> bytes:
    """Serialize a JSON-compatible object to RFC 8785 JCS UTF-8 bytes."""

    def _ser(val: Any) -> str:
        if val is None:
            return "null"
        if isinstance(val, bool):
            return "true" if val else "false"
        if isinstance(val, int):
            return str(val)
        if isinstance(val, float):
            return _format_jcs_float(val)
        if isinstance(val, str):
            return json.dumps(val, ensure_ascii=False)
        if isinstance(val, (list, tuple)):
            return "[" + ",".join(_ser(item) for item in val) + "]"
        if isinstance(val, Mapping):
            items = [(str(k).encode("utf-16-be"), str(k), v) for k, v in val.items()]
            items.sort(key=lambda e: e[0])
            return (
                "{"
                + ",".join(
                    f"{json.dumps(k, ensure_ascii=False)}:{_ser(v)}"
                    for _, k, v in items
                )
                + "}"
            )
        raise TypeError(f"Unsupported JCS type: {type(val).__name__}")

    return _ser(obj).encode("utf-8")


def build_demo_signed_manifest() -> dict[str, Any]:
    """Construct a valid, cryptographically signed multi-edition Ego `manifest.json`."""

    demo_seed = bytes.fromhex("2a" * 32)
    subkey = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=HKDF_SALT,
        info=HKDF_INFO,
    ).derive(demo_seed)
    priv = Ed25519PrivateKey.from_private_bytes(subkey)
    pub_hex = priv.public_key().public_bytes_raw().hex()
    signer_fay_id = "ifay_ed234567abcdefghij"

    def _digest(tag: str) -> str:
        return "sha256:" + hashlib.sha256(tag.encode("utf-8")).hexdigest()

    def _zone_levels(max_lvl: int, zone: str) -> list[dict[str, Any]]:
        return [
            {"level": lvl, "path": f"adapters/{zone}/L{lvl}.safetensors"}
            for lvl in range(1, max_lvl + 1)
        ]

    payload: dict[str, Any] = {
        "version": "0.2.0",
        "created_at": "2026-11-18T14:51:42Z",
        "base_model": "qwen2.5-0.5B",
        "editions": ["lite", "normal", "plus"],
        "training_backend": "peft-cpu",
        "grid": {
            "lite": {
                "kernel": {"path": "adapters/kernel.safetensors"},
                "zones": {
                    "coding": _zone_levels(1, "coding"),
                    "knowledge": _zone_levels(1, "knowledge"),
                },
            },
            "normal": {
                "kernel": {"path": "adapters/kernel.safetensors"},
                "zones": {
                    "coding": _zone_levels(2, "coding"),
                    "knowledge": _zone_levels(2, "knowledge"),
                },
            },
            "plus": {
                "kernel": {"path": "adapters/kernel.safetensors"},
                "zones": {
                    "coding": _zone_levels(3, "coding"),
                    "knowledge": _zone_levels(3, "knowledge"),
                },
            },
        },
        "fidelity_report": {
            "schema_version": "1.0",
            "status": "measured",
            "holdout_sample_count": 3,
            "holdout_loss": 3.073676,
            "holdout_perplexity": 21.621236,
            "cell_consistency_score": 1.0,
            "probe_win_rate": None,
            "drift_from_baseline": None,
        },
        "integrity": {
            "schema_version": "1.0",
            "base_model_digest": "unverified_offline",
            "cell_digests": {
                "lite/adapters/coding/L1.safetensors": _digest("lite-c1"),
                "lite/adapters/kernel.safetensors": _digest("lite-k"),
                "lite/adapters/knowledge/L1.safetensors": _digest("lite-k1"),
                "normal/adapters/coding/L1.safetensors": _digest("norm-c1"),
                "normal/adapters/coding/L2.safetensors": _digest("norm-c2"),
                "normal/adapters/kernel.safetensors": _digest("norm-k"),
                "normal/adapters/knowledge/L1.safetensors": _digest("norm-k1"),
                "normal/adapters/knowledge/L2.safetensors": _digest("norm-k2"),
                "plus/adapters/coding/L1.safetensors": _digest("plus-c1"),
                "plus/adapters/coding/L2.safetensors": _digest("plus-c2"),
                "plus/adapters/coding/L3.safetensors": _digest("plus-c3"),
                "plus/adapters/kernel.safetensors": _digest("plus-k"),
                "plus/adapters/knowledge/L1.safetensors": _digest("plus-k1"),
                "plus/adapters/knowledge/L2.safetensors": _digest("plus-k2"),
                "plus/adapters/knowledge/L3.safetensors": _digest("plus-k3"),
            },
        },
    }

    jcs_bytes = canonicalize_jcs(payload)
    jcs_sha256 = hashlib.sha256(jcs_bytes).hexdigest()
    sig_hex = priv.sign(SIGNING_DOMAIN.encode("ascii") + b":" + jcs_bytes).hex()

    signed = dict(payload)
    signed["signature"] = {
        "schema_version": "1.0",
        "alg": "Ed25519",
        "domain": SIGNING_DOMAIN,
        "signer_fay_id": signer_fay_id,
        "key_id": f"{signer_fay_id}#ego-sign-1",
        "public_key_hex": pub_hex,
        "payload_jcs_sha256": jcs_sha256,
        "sig_hex": sig_hex,
    }
    return signed


def inspect_manifest_json(raw_json: str) -> str:
    """Parse a real or demo `manifest.json` string and return a Markdown audit report."""

    try:
        manifest = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        return f"### ❌ Invalid JSON\n\n`{exc}`"
    if not isinstance(manifest, dict):
        return "### ❌ Invalid Manifest\n\nTop-level JSON value must be an object."

    editions = manifest.get("editions", [])
    backend = manifest.get("training_backend", "unknown")
    base_model = manifest.get("base_model", "multi-base")
    fidelity = manifest.get("fidelity_report") or {}
    integrity = manifest.get("integrity") or {}
    cell_digests = integrity.get("cell_digests") or {}
    sig = manifest.get("signature")

    consistency = fidelity.get("cell_consistency_score")
    gate_badge = (
        f"✅ `{consistency:.4f}` (passes FayGer `>= 0.70` gate)"
        if isinstance(consistency, (int, float)) and consistency >= 0.70
        else (
            f"⚠️ `{consistency}` (below `0.70` production gate)"
            if isinstance(consistency, (int, float))
            else "ℹ️ `null` (Stub or flat Kernel-only package)"
        )
    )

    lines = [
        "### `.egopod` Manifest Audit Summary",
        "",
        f"- **Package Version**: `{manifest.get('version', 'unknown')}`",
        f"- **Primary Base Model**: `{base_model}`",
        f"- **Training Backend**: `{backend}`",
        f"- **Editions (`{len(editions)}`)**: {', '.join(f'`{e}`' for e in editions) or '_none_'}",
        f"- **Protected `.safetensors` Cells**: `{len(cell_digests)}`",
        f"- **Holdout Samples**: `{fidelity.get('holdout_sample_count', 'N/A')}`",
        f"- **Holdout Loss / Perplexity**: `{fidelity.get('holdout_loss', 'N/A')}` / `{fidelity.get('holdout_perplexity', 'N/A')}`",
        f"- **Cross-Cell Consistency**: {gate_badge}",
        f"- **Signature Present**: `{'yes (' + str(sig.get('signer_fay_id')) + ')' if isinstance(sig, dict) else 'unsigned'}`",
    ]
    return "\n".join(lines)


def verify_manifest_signature_json(
    raw_json: str, trusted_pubkey_hex: str = ""
) -> str:
    """Verify RFC 8785 JCS + Ed25519 signature & canonical `ifay_` signer ID."""

    try:
        manifest = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        return f"### ❌ Verification Failed\n\nInvalid JSON: `{exc}`"
    if not isinstance(manifest, dict):
        return "### ❌ Verification Failed\n\nManifest root must be a JSON object."

    sig = manifest.get("signature")
    if not isinstance(sig, dict):
        return "### ⚠️ Unsigned Package\n\n`manifest.signature` is absent."

    signer_id = str(sig.get("signer_fay_id", ""))
    if not CANONICAL_IFAY_ID_RE.match(signer_id):
        return (
            "### ❌ Invalid Signer FayID\n\n"
            f"`{signer_id}` does not match canonical `^ifay_[a-z2-7]{{16,}}$`."
        )
    if sig.get("alg") != "Ed25519" or sig.get("domain") != SIGNING_DOMAIN:
        return "### ❌ Unsupported Signature Algorithm or Domain"

    pub_hex = str(sig.get("public_key_hex", "")).strip().lower()
    expected_jcs_sha = str(sig.get("payload_jcs_sha256", "")).strip().lower()
    sig_hex = str(sig.get("sig_hex", "")).strip().lower()

    if trusted_pubkey_hex.strip():
        clean_trusted = trusted_pubkey_hex.strip().lower()
        if clean_trusted != pub_hex:
            return (
                "### ❌ Public Key Mismatch\n\n"
                f"- **Manifest Key**: `{pub_hex}`\n"
                f"- **Trusted Key**: `{clean_trusted}`"
            )

    payload = {k: v for k, v in manifest.items() if k != "signature"}
    jcs_bytes = canonicalize_jcs(payload)
    actual_jcs_sha = hashlib.sha256(jcs_bytes).hexdigest()
    if actual_jcs_sha != expected_jcs_sha:
        return (
            "### ❌ Tamper Detected (JCS Digest Mismatch)\n\n"
            f"- **Signed `payload_jcs_sha256`**: `{expected_jcs_sha}`\n"
            f"- **Recomputed RFC 8785 JCS SHA-256**: `{actual_jcs_sha}`"
        )

    try:
        pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(pub_hex))
        pub.verify(
            bytes.fromhex(sig_hex),
            SIGNING_DOMAIN.encode("ascii") + b":" + jcs_bytes,
        )
    except (ValueError, InvalidSignature):
        return "### ❌ Invalid Ed25519 Signature\n\nCryptographic verification failed over `ifay.ego.package-signing.v1:<jcs_bytes>`."

    cell_digests = (manifest.get("integrity") or {}).get("cell_digests") or {}
    return "\n".join(
        [
            "### ✅ Cryptographic Signature Verified",
            "",
            f"- **Signer `FayID`**: `{signer_fay_id_str(signer_id)}`",
            f"- **Key ID**: `{sig.get('key_id')}`",
            f"- **Domain**: `{SIGNING_DOMAIN}`",
            f"- **Ed25519 Public Key**: `{pub_hex}`",
            f"- **RFC 8785 JCS SHA-256**: `{actual_jcs_sha}`",
            f"- **Tier-1 Protected Adapter Cells**: `{len(cell_digests)}`",
        ]
    )


def signer_fay_id_str(val: str) -> str:
    return val


def simulate_tamper(raw_json: str) -> str:
    """Mutate `fidelity_report.cell_consistency_score` without re-signing to demonstrate tamper detection."""

    obj = json.loads(raw_json)
    fid = dict(obj.get("fidelity_report") or {})
    fid["cell_consistency_score"] = 0.55
    obj["fidelity_report"] = fid
    return json.dumps(obj, indent=2, ensure_ascii=False)


def explore_sparse_grid(
    edition: str, kernel_only: bool, coding_depth: int, knowledge_depth: int
) -> str:
    """Compute the deterministic `Coexist_Activate` adapter stack for an edition & selection."""

    manifest = build_demo_signed_manifest()
    ed_grid = manifest["grid"][edition]
    kernel_rel = f"{edition}/{ed_grid['kernel']['path']}"
    edition_bases = {
        "lite": "qwen2.5-0.5B (Qwen/Qwen2.5-0.5B-Instruct)",
        "normal": "qwen3.5-0.8B (Qwen/Qwen3.5-0.8B)",
        "plus": "qwen3.5-4B (Qwen/Qwen3.5-4B)",
    }

    active_cells = [f"1. `kernel` → `{kernel_rel}` _(always loaded, Level 0)_"]
    if not kernel_only:
        for zone_name, req_depth in (
            ("coding", coding_depth),
            ("knowledge", knowledge_depth),
        ):
            entries = ed_grid["zones"].get(zone_name, [])
            by_level = {int(item["level"]): item["path"] for item in entries}
            max_avail = max(by_level.keys(), default=0)
            eff_depth = min(int(req_depth), max_avail)
            for lvl in range(1, eff_depth + 1):
                rel = by_level.get(lvl)
                if rel:
                    idx = len(active_cells) + 1
                    active_cells.append(
                        f"{idx}. `{zone_name}_L{lvl}` → `{edition}/{rel}`"
                    )

    cli_cmd = f"ego run --edition {edition}"
    if kernel_only:
        cli_cmd += " --kernel-only"
    else:
        if coding_depth > 0:
            cli_cmd += f" --zone coding:{coding_depth}"
        if knowledge_depth > 0:
            cli_cmd += f" --zone knowledge:{knowledge_depth}"

    axis_rows = "\n".join(
        f"| {axis} | `{stream}` | `{placement}` | {desc} |"
        for axis, stream, placement, desc in SIX_AXES
    )
    return "\n".join(
        [
            f"### Active `Coexist_Activate` Stack — Edition `{edition}` (`{edition_bases[edition]}`)",
            "",
            f"**Equivalent CLI Command**: `{cli_cmd}`",
            "",
            *active_cells,
            "",
            "### 6D Behavioral Profile Mapping",
            "",
            "| Axis | Source Stream | Sparse-Grid Cell | Distilled Trait |",
            "| --- | --- | --- | --- |",
            axis_rows,
        ]
    )


def compare_stub_vs_adapter(prompt_choice: str) -> tuple[str, str, str]:
    """Return Stub reply, Kernel-only LoRA reply, and Full Sparse-Grid LoRA reply."""

    stub_reply = "[STUB MODE] This is a placeholder reply from a stub-trained Ego."
    catalog = {
        "Summarize my morning focus protocol and meeting rules.": (
            "Start at 07:00 with black coffee, keep 07:30–10:30 strictly meeting-free for deep work, and batch review notifications at 11:00.",
            "07:00 black coffee -> 07:30–10:30 uninterrupted deep-work block (notifications muted) -> 11:00 async PR & design doc triage. Decline ad-hoc morning calls unless P0.",
        ),
        "How should we store API keys and model weights in the new service?": (
            "Never commit plaintext secrets to config files; read credentials from environment variables and verify artifact signatures before loading.",
            "Enforce three invariants: (1) reject plaintext keys in YAML configs with exit code 2 (`EGO_CLOUD_*` env vars only), (2) serialize adapters strictly as `.safetensors` (zero pickle), and (3) verify RFC 8785 JCS + Ed25519 digests before activation.",
        ),
        "Can you wire $500 to the vendor invoice automatically?": (
            "I can prepare the payment summary, but external transfers require your explicit confirmation first.",
            "[CAP Boundary] I have drafted the invoice verification checklist, but LoRA persona weights never grant financial transfer authority. Please review and sign the transaction directly as the Human Prime.",
        ),
    }
    kernel_reply, full_reply = catalog.get(
        prompt_choice,
        (
            "Kernel persona aligned: concise, privacy-first response.",
            "Full Sparse-Grid (`Kernel + coding + knowledge`) aligned response.",
        ),
    )
    return stub_reply, kernel_reply, full_reply


def create_demo_app():
    """Build the Gradio Blocks application."""

    import gradio as gr

    default_manifest_str = json.dumps(
        build_demo_signed_manifest(), indent=2, ensure_ascii=False
    )

    with gr.Blocks(title="Ego — Sovereign Personal AI Pod Inspector") as demo:
        gr.Markdown(
            "# 🧬 Ego — Sovereign Personal AI Pod Inspector\n"
            "Inspect multi-edition `.egopod` manifests, explore the **6-Axis Behavioral & `L1..L3` Sparse Grid**, "
            "verify **RFC 8785 JCS + Ed25519 (`ifay_`)** bundle signatures, and compare **Stub vs. LoRA Adapter** activation."
        )

        with gr.Tabs():
            with gr.Tab("1. Manifest Inspector"):
                manifest_in = gr.Code(
                    value=default_manifest_str,
                    language="json",
                    label="ego-output/manifest.json",
                    lines=22,
                )
                inspect_btn = gr.Button("Audit Manifest", variant="primary")
                inspect_out = gr.Markdown(inspect_manifest_json(default_manifest_str))
                inspect_btn.click(
                    inspect_manifest_json, inputs=[manifest_in], outputs=[inspect_out]
                )

            with gr.Tab("2. 6D Behavioral & L1..L3 Grid Explorer"):
                with gr.Row():
                    ed_sel = gr.Dropdown(
                        choices=["lite", "normal", "plus"],
                        value="normal",
                        label="Edition",
                    )
                    kernel_chk = gr.Checkbox(
                        value=False, label="Kernel-Only (--kernel-only)"
                    )
                    code_slider = gr.Slider(
                        0, 3, value=2, step=1, label="coding Zone Depth (L0..L3)"
                    )
                    know_slider = gr.Slider(
                        0, 3, value=2, step=1, label="knowledge Zone Depth (L0..L3)"
                    )
                grid_out = gr.Markdown(explore_sparse_grid("normal", False, 2, 2))
                for ctrl in (ed_sel, kernel_chk, code_slider, know_slider):
                    ctrl.change(
                        explore_sparse_grid,
                        inputs=[ed_sel, kernel_chk, code_slider, know_slider],
                        outputs=[grid_out],
                    )

            with gr.Tab("3. Ed25519 Signature Verifier"):
                sig_json = gr.Code(
                    value=default_manifest_str,
                    language="json",
                    label="Signed manifest.json",
                    lines=18,
                )
                trusted_key = gr.Textbox(
                    value="",
                    placeholder="Optional 64-hex Ed25519 public key pin",
                    label="Trusted Public Key Hex (optional)",
                )
                with gr.Row():
                    verify_btn = gr.Button(
                        "Verify JCS + Ed25519 Signature", variant="primary"
                    )
                    tamper_btn = gr.Button("Simulate Tamper Attack")
                    reset_btn = gr.Button("Reset Valid Manifest")
                verify_out = gr.Markdown(
                    verify_manifest_signature_json(default_manifest_str)
                )
                verify_btn.click(
                    verify_manifest_signature_json,
                    inputs=[sig_json, trusted_key],
                    outputs=[verify_out],
                )
                tamper_btn.click(
                    simulate_tamper, inputs=[sig_json], outputs=[sig_json]
                ).then(
                    verify_manifest_signature_json,
                    inputs=[sig_json, trusted_key],
                    outputs=[verify_out],
                )
                reset_btn.click(
                    lambda: default_manifest_str, outputs=[sig_json]
                ).then(
                    verify_manifest_signature_json,
                    inputs=[sig_json, trusted_key],
                    outputs=[verify_out],
                )

            with gr.Tab("4. Stub vs. LoRA Adapter Comparison"):
                prompt_dd = gr.Dropdown(
                    choices=[
                        "Summarize my morning focus protocol and meeting rules.",
                        "How should we store API keys and model weights in the new service?",
                        "Can you wire $500 to the vendor invoice automatically?",
                    ],
                    value="Summarize my morning focus protocol and meeting rules.",
                    label="Prompt",
                )
                s0, k0, f0 = compare_stub_vs_adapter(
                    "Summarize my morning focus protocol and meeting rules."
                )
                with gr.Row():
                    stub_box = gr.Textbox(
                        value=s0, label="Stub Backend (Dry-Run)", lines=5
                    )
                    kern_box = gr.Textbox(
                        value=k0, label="Real LoRA — Kernel-Only (L0)", lines=5
                    )
                    full_box = gr.Textbox(
                        value=f0,
                        label="Real LoRA — Full Sparse Grid (Kernel + L1..L2)",
                        lines=5,
                    )
                prompt_dd.change(
                    compare_stub_vs_adapter,
                    inputs=[prompt_dd],
                    outputs=[stub_box, kern_box, full_box],
                )

    return demo


if __name__ == "__main__":
    app = create_demo_app()
    app.launch()
