# Ego

[![Docs & Link Integrity CI](https://github.com/ChainMode/Ego/actions/workflows/docs-ci.yml/badge.svg)](https://github.com/ChainMode/Ego/actions/workflows/docs-ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB.svg?logo=python&logoColor=white)](./release/dev/pyproject.toml)
[![Website: ifay.ai](https://img.shields.io/badge/Website-ifay.ai-0F766E.svg)](https://ifay.ai)

**Distill your own notes and conversations into a signed, local-first personal AI pod on your Mac or Linux machine — zero cloud lock-in.**

**Ego** (`ego-pod` v0.2.0) is the personal model cultivation pipeline for the [iFay](https://ifay.ai) ecosystem (`Fay = iFay + coFay`). Starting from a **Human Prime**'s local notes, habits, code idioms, and workflows, Ego scrubs secrets locally, compiles instruction corpora, fine-tunes multi-edition **Sparse-Grid LoRA** adapters on ungated open weights (`Qwen`, `Gemma 4`, `Phi-4`), and seals the resulting `ego-output/` package with **RFC 8785 JCS + Ed25519** provenance signatures ready for on-demand loading on **Host** devices.

## 60-Second Quickstart

```bash
# Install ego-pod with Hugging Face PEFT real training support (macOS / Linux)
pip install "./release/dev[peft]"

# 1. Initialize workspace config (ego-config.yaml)
ego init

# 2. Cultivate Sparse-Grid LoRA adapters + measure cross-Cell consistency
ego train

# 3. Chat locally with full Sparse Grid or selective zone depth
ego run --edition normal
ego run --edition normal --zone coding:1 --kernel-only

# 4. Launch 3-in-1 Local Cognitive Service (Intuition + Dynamic Context + Guard Rewrite)
ego serve ./ego-output --edition normal --host 127.0.0.1 --port 7980

# 5. Export a size-bounded standalone sub-model slice for constrained FayGer hosts
ego slice ./ego-output --domain drone-photo --out ./drone-slice --max-size-mb 16 --signing-key "$EGO_SIGNING_SEED_HEX"

# 6. Cold-start import & companion incremental distillation (Ego School)
ego school import --format openai --input ./conversations.json --workspace .
ego school evolve . --ledger ./primary_body_events.jsonl
```

- **CLI Reference & Architecture**: [release/dev/README.md](./release/dev/README.md)
- **Verified Non-Stub Fine-Tuning Benchmark (Apple Silicon)**: [release/dev/REAL_TRAINING.md](./release/dev/REAL_TRAINING.md)
- **PyPI Packaging & Trusted Publisher Guide**: [release/dev/PUBLISHING.md](./release/dev/PUBLISHING.md)
- **Hugging Face Space Inspector & Adapter Model Card**: [release/hf/README.md](./release/hf/README.md) · [release/hf/MODEL_CARD.md](./release/hf/MODEL_CARD.md)
- **Multilingual Architectural Blueprints (9 Locales)**: [docs/](./docs)

## Core Capabilities (`ego-pod` v0.2.0)

- **Deep Multi-Dimensional Sparse-Grid Distillation (`Kernel + L1..L3`)**: Leakage-free within-zone IDF + structural/code/reasoning specificity scoring with contiguous level compaction (`1..K`) and per-Edition grid pruning.
- **Multi-Edition & Real Ungated Base Models**: Built-in `lite`, `normal`, `plus`, `super`, and custom Edition presets mapped directly to verified Hugging Face checkpoints (`qwen2.5-0.5B` → `Qwen/Qwen2.5-0.5B-Instruct`, `qwen3.5-0.8B..27B`, `gemma4-2B`, `phi4-mini`, `phi4-14B`), with automatic OOM `batch_size` halving retry and size ceilings.
- **Pluggable & Incremental Data Sources**: Local directories, allowlist-guarded HTTPS URLs with ETag/Range resume, environment-credentialed S3/OSS buckets, Python `entry_points` (`ego_pod.collectors`) plugins, and `ego train --update` content-SHA-256 incremental compilation.
- **Versioned & Signed Vault**: Two-tier package integrity (`cell_digests` SHA-256 + RFC 8785 JCS Ed25519 signature via `ifay_` FayID `HKDF-SHA256` subkey), post-training `cell_consistency_score` probing, `.history/` snapshot retention, and verified rollback (`ego vault list | rollback | purge`).
- **On-Demand Runtime Activation**: Selective zone and depth loading via `ego run --edition <name> --zone <zone>:<depth>` or `ego run --kernel-only`.
- **3-in-1 Local Cognitive Service (`ego serve` & `EgoCognitiveEngine`)**: Zero-dependency stdlib `ThreadingHTTPServer` JSON API verifying `Ed25519` bundle provenance on startup and serving `<20ms` System-1 fast intuition (`POST /v1/intuition`), external System-2 persona steering (`POST /v1/dynamic-context`), and post-generation redline/PII/tone auditing (`POST /v1/guard-rewrite`), plus `GET /health`.
- **Standalone Sub-Model Slicing (`ego slice`, `ego-submodel-slice/1.0`)**: Prunes a parent `.egopod` down to `Kernel` + target domain cells (`coding`, `social`, `knowledge`, `creative`, `sense`, `drone-photo`, or custom zones) within `--max-size-mb`, recomputes `cell_digests`, embeds `slice_descriptor`, and signs a standalone runnable `.egopod` for constrained `FayGer` edge devices.
- **`Ego School` Cold-Start & Companion Evolution (`ego school`)**: Scrubs PII/secrets automatically while extracting **Human Prime**-authored turns and high-weight `human_correction` pairs from Markdown, JSONL, OpenAI `conversations.json`, and `iFay` Primary Body ledgers (`ego school import`), then runs consistency-gated incremental Dojo updates with automatic Vault snapshot archival and rollback (`ego school evolve`).

## Governance & Security

- Contribution guidelines & DCO 1.1 sign-off: [CONTRIBUTING.md](./CONTRIBUTING.md)
- Private vulnerability reporting: [SECURITY.md](./SECURITY.md)
- Community standards: [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md)

## License

This repository uses a dual-track open-source license (`Copyright (c) 2025-2026 ChainMode contributors`):

- **Source Code, Schemas & Training Pipeline** (`release/dev/`, `release/hf/`, `schema/`): Licensed under the **MIT License** — see [LICENSE](./LICENSE).
- **Specifications & Architectural Blueprints** (`docs/`): Licensed under **Creative Commons Attribution 4.0 International (CC BY 4.0)** — see [LICENSE-DOCS](./LICENSE-DOCS).
