---
title: Ego Sovereign Pod Inspector
emoji: 🧬
colorFrom: indigo
colorTo: cyan
sdk: gradio
sdk_version: "5.12.0"
app_file: app.py
pinned: false
license: mit
short_description: Inspect signed .egopod manifests, L1..L3 Sparse Grids & Ed25519 proofs
---

# Ego — Sovereign Personal AI Pod Inspector

**Ego** (`ego-pod`) is a local-first, privacy-preserving cultivation pipeline that distills a **Human Prime**'s own notes, conversations, preferences, and workflows into multi-edition, cryptographically signed **Sparse-Grid LoRA packages** (`ego-output/` / `.egopod`) for on-demand loading on **Host** devices (`Mac`, `Linux`, edge endpoints, and [FayGer](https://ifay.ai) runtimes).

- **GitHub Repository**: [https://github.com/ChainMode/Ego](https://github.com/ChainMode/Ego)
- **Ecosystem Portal**: [https://ifay.ai](https://ifay.ai)
- **CLI & Real Fine-Tuning Guide**: [../dev/README.md](../dev/README.md) · [../dev/REAL_TRAINING.md](../dev/REAL_TRAINING.md)
- **Adapter Model Card Template**: [./MODEL_CARD.md](./MODEL_CARD.md)

## What This Space Demonstrates

1. **`.egopod` / `manifest.json` Inspector**: Parse and audit multi-edition Ego manifests (`lite`, `normal`, `plus`, `super`, or custom editions), base model mappings, and privacy-safe `fidelity_report` scalar summaries (`holdout_loss`, `cell_consistency_score`).
2. **6-Axis Behavioral & Sparse-Grid Explorer (`Kernel + L1..L3`)**: Explore how a Human Prime's five raw data streams (`sense`, `action`, `thought`, `skill`, `credentials`) map across core personality axes and selective functional zones (`coding`, `knowledge`, etc.) from `Level 0` (`Kernel`) through `L1..L3` specificity depth.
3. **Two-Tier Cryptographic Verification (`JCS + Ed25519`)**: Verify `ifay_` FayID `HKDF-SHA256` subkey signatures (`domain = "ifay.ego.package-signing.v1"`, RFC 8785 JSON Canonicalization Scheme) and `sha256:` per-Cell `.safetensors` digests directly in your browser—and see tamper detection in action.
4. **Stub vs. LoRA Adapter Response Comparison**: Compare deterministic `[STUB MODE]` dry-run pipeline responses against real LoRA adapter activation (`Kernel-only` vs. `Full_In_Load` `Coexist_Activate`).

## Zero-Leakage Privacy Guarantees

- **Local Cultivation Only**: Real personal corpora are collected, PII-scrubbed (`[REDACTED]`), compiled, and fine-tuned strictly on your own **Host** hardware (`ego train`).
- **Adapter-Only SafeTensors**: Exported `.safetensors` files contain *only* low-rank LoRA delta matrices (`lora_A`, `lora_B`)—never base weights or raw personal text.
- **Aggregate Fidelity Scalars Only**: `manifest.json` records scalar metrics (`holdout_loss`, `cell_consistency_score`) without storing holdout prompts or generated text.

## Running Locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r release/hf/requirements.txt
python release/hf/app.py
```
