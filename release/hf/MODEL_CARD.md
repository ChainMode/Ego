---
library_name: peft
base_model: Qwen/Qwen2.5-0.5B-Instruct
license: mit
pipeline_tag: text-generation
tags:
  - ego
  - egopod
  - ifay
  - lora
  - qlora
  - safetensors
  - sparse-grid
  - personal-ai
  - local-first
---

# Model Card Template — `<ego-handle>.egopod` (Sovereign Ego Adapter Package)

> Reusable Hugging Face Model Card template for user-exported **Ego** (`ego-pod`) Sparse-Grid LoRA packages. Replace bracketed placeholders (`<...>`) with values from your `ego-output/manifest.json` and `ego-output/<edition>/config.json` before publishing a public or team-shared adapter bundle. **Never publish an Ego trained on private unconsented personal records.**

## Package Summary

| Field | Value |
| --- | --- |
| **Package Format** | `ego-output/` / `.egopod` (`manifest.json` + SafeTensors Sparse-Grid Cells) |
| **Cultivated By** | `ego-pod v0.2.0` ([ChainMode/Ego](https://github.com/ChainMode/Ego)) |
| **Editions Included** | `<lite, normal, plus, super>` |
| **Base Model (`HF Repo ID`)** | `<Qwen/Qwen2.5-0.5B-Instruct \| Qwen/Qwen3.5-0.8B \| Qwen/Qwen3.5-4B>` |
| **Training Backend** | `<peft-cpu \| mlx \| peft-cuda \| unsloth>` |
| **Signer `FayID`** | `<ifay_...>` (`Ed25519` over RFC 8785 JCS, domain `ifay.ego.package-signing.v1`) |
| **Cell Consistency Score** | `<0.8500>` (FayGer production gate requires `>= 0.70`) |

## 6D Behavioral Profile & `L1..L3` Sparse-Grid Topology

An `Ego` package separates always-on core persona alignment (**Level 0 `Kernel`**) from on-demand domain specialization (**`L1..L3` Functional Zone Cells**), spanning six behavioral dimensions:

| Axis | Behavioral Dimension | Primary Data Stream | Sparse-Grid Placement |
| --- | --- | --- | --- |
| **D1 — Tone & Expression** | Phrasing cadence, conciseness, bilingual register | `sense` / `thought` | `Kernel (L0)` |
| **D2 — Habits & Routines** | Daily scheduling preferences, operational defaults | `sense` / `action` | `Kernel (L0)` + `L1` |
| **D3 — Reasoning Style** | Trade-off framing, first-principles decomposition | `thought` | `knowledge (L1..L3)` |
| **D4 — Domain Knowledge** | Curated reading notes, personal terminology & facts | `thought` | `knowledge (L1..L3)` |
| **D5 — Technical & Craft Skills** | Coding idioms, tool workflows, structured execution | `skill` / `action` | `coding (L1..L3)` |
| **D6 — Boundary & Delegation Stance** | Redacted capability scope & escalation habits | `credentials` (PII-scrubbed) | `Kernel (L0)` |

### Included Adapter Cells (`integrity.cell_digests`)

```text
<edition>/
├── config.json
└── adapters/
    ├── kernel.safetensors          # Level 0: Core tone, values & boundary stance
    ├── coding/
    │   ├── L1.safetensors          # General coding & workflow habits
    │   └── L2.safetensors          # Deep framework/architecture idioms
    └── knowledge/
        ├── L1.safetensors          # Broad synthesis & reading patterns
        └── L2.safetensors          # Specialized domain reasoning
```

## Sovereignty, Privacy & CAP Authorization Boundaries

1. **Adapter-Only Weights**: Every `.safetensors` file contains strictly LoRA `lora_A` / `lora_B` low-rank matrices—zero pickle payloads and zero raw training samples.
2. **Cryptographic Provenance**: Verify package integrity prior to loading on any **Host** runtime:
   ```python
   from ego_pod.vault.signing import verify_package_signature

   result = verify_package_signature("./ego-output")
   assert result.status == "verified"
   print("Verified signer:", result.signer_fay_id, "cells:", result.verified_cells)
   ```
3. **CAP (Capability & Authorization Policy) Boundaries**:
   - An **Ego** adapter encodes cognitive style, preferences, and domain skills for a **Human Prime**'s personal **iFay** avatar (note: organizational **coFay** avatars operate under a `RoleCharter` without an `Ego` layer).
   - **Model weights are NOT authority**: LoRA weights must never be treated as cryptographic authorization for high-impact external actions (payments, credential disclosure, irreversible deletions, or legal commitments). All side-effecting actions on a **Host** runtime must be gated by explicit runtime capability policies (`CAP`) and deterministic **Human Prime** confirmation.

## Loading & Selective Zone Activation

### Using `ego-pod` CLI

```bash
pip install "ego-pod[peft]"

# Load full Sparse Grid for an edition
ego run --edition normal

# Load Kernel + specific zone depth only (minimal memory footprint)
ego run --edition normal --zone coding:1

# Load Level-0 Kernel persona only
ego run --edition normal --kernel-only
```

## Limitations & Intended Use

- Designed for personal assistant alignment on authorized **Host** devices where the matching base model checkpoint is installed.
- Switching base model families (e.g., `qwen2.5-0.5B` to `qwen3.5-4B`) requires re-running `ego train` against the target base checkpoint; LoRA matrices are dimension-coupled to their base architecture.
