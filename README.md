# Ego

Local, privacy-first personalized AI model cultivation pipeline (`ego_pod` v0.2.0) for iFay — instantiating a Human Prime's cognition, knowledge, and habits into multi-edition, cryptographically signed two-dimensional Sparse Grid LoRA packages ready for on-demand loading on Host devices.

## Capabilities (`ego_pod` v0.2.0)

- **Deep Multi-Dimensional Sparse-Grid Distillation (`L1..L3`)**: Leakage-free within-zone IDF + structural/code/reasoning complexity scoring with automatic contiguous level compaction (`1..K`) and per-Edition grid pruning.
- **Multi-Edition & Multi-Base Cultivation**: Built-in `lite`, `normal`, `plus`, `super`, and custom Edition presets across controlled Qwen 3.5, Gemma 4, and Phi-4 base models, with automatic OOM `batch_size` halving retry and artifact size ceilings.
- **Pluggable & Incremental Data Sources**: Local directories, allowlist-guarded HTTPS URLs with ETag/Range checkpoint resume, environment-credentialed S3/OSS object stores, Python `entry_points` (`ego_pod.collectors`) plugins, and `ego train --update` content-SHA-256 incremental compilation.
- **Versioned & Signed Vault**: Two-tier package integrity (`cell_digests` SHA-256 + RFC 8785 JCS Ed25519 signature via FayID HKDF subkey), automatic `.history/` snapshot retention, and verified rollback (`ego vault list | rollback | purge`).
- **On-Demand Runtime Activation**: Selective zone and depth loading via `ego run --edition <name> --zone <zone>:<depth>` or `ego run --kernel-only`.

See [release/dev/README.md](./release/dev/README.md) for CLI installation and usage, and [docs/](./docs) for multilingual architectural blueprints.

## Governance & Security

- Contribution guidelines & DCO 1.1 sign-off: [CONTRIBUTING.md](./CONTRIBUTING.md)
- Private vulnerability reporting: [SECURITY.md](./SECURITY.md)
- Community standards: [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md)

## License

This repository uses a dual-track open-source license (`Copyright (c) 2025-2026 ChainMode contributors`):

- **Source Code, Schemas & Training Pipeline** (`release/dev/`, `schema/`): Licensed under the **MIT License** — see [LICENSE](./LICENSE).
- **Specifications & Architectural Blueprints** (`docs/`): Licensed under **Creative Commons Attribution 4.0 International (CC BY 4.0)** — see [LICENSE-DOCS](./LICENSE-DOCS).
