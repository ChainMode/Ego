# Real Non-Stub Fine-Tuning & Inference Guide (`ego-pod` v0.2.0)

This guide documents reproducible, **non-stub** LoRA fine-tuning, post-training `cell_consistency_score` probing, cryptographic signing, and local multi-adapter inference using `ego-pod` on macOS (Apple Silicon) or Linux—**with zero Hugging Face login token required**.

---

## 1. Ungated Hugging Face Base Model Registry

`ego_pod.config.base_models.resolve_hf_repo_id` maps friendly `ego-config.yaml` model names to verified, ungated (`gated: false`) Hugging Face Hub checkpoints:

| Friendly Name (`ego-config.yaml`) | Hugging Face Repo ID | Approx. Weights Size | Notes |
| --- | --- | --- | --- |
| `qwen2.5-0.5B` | `Qwen/Qwen2.5-0.5B-Instruct` | `~0.99 GB` | **Fastest CPU / Apple Silicon starter** (`Qwen2ForCausalLM` SDPA) |
| `qwen3.5-0.8B` | `Qwen/Qwen3.5-0.8B` | `~1.75 GB` | Default `ego init` & `lite` preset (`qwen3_5`; auto-resolves `["q_proj", "v_proj"]` LoRA targets) |
| `qwen3.5-2B` | `Qwen/Qwen3.5-2B` | `~4.55 GB` | Default `normal` multi-edition preset |
| `qwen3.5-4B` | `Qwen/Qwen3.5-4B` | `~9.32 GB` | Default `plus` preset |
| `qwen3.5-9B` | `Qwen/Qwen3.5-9B` | `~18.60 GB` | Mid-scale desktop / GPU workstation |
| `qwen3.5-27B` | `Qwen/Qwen3.5-27B` | `~55.56 GB` | Default `super` preset |
| `gemma4-2B` | `google/gemma-4-E2B-it` | `~10.25 GB` | Gemma 4 `E2B` instruction-tuned checkpoint |
| `phi4-mini` | `microsoft/Phi-4-mini-instruct` | `~7.67 GB` | Phi-4 Mini instruct |
| `phi4-14B` | `microsoft/phi-4` | `~29.32 GB` | Phi-4 14B |

> **Note on `gemma4-9B` and `gemma4-27B`**: Kept valid in `ego-config.yaml` for backward compatibility and `Stub` dry-runs, because Google published `E2B`, `E4B`, `26B-A4B`, and `31B` checkpoints rather than `9B`/`27B`. Selecting them on a real backend raises an immediate `TrainerError` listing all working ungated checkpoints. You can also pass any Hugging Face `org/repo` ID or local checkpoint directory path directly in `editions.<name>.base`.

---

## 2. Step-by-Step Reproducible Local Cultivation

### Step 1 — Install `ego-pod` with a Real Training Backend

Pick the optional dependency group matching your **Host** hardware:

```bash
# Cross-platform CPU / macOS PyTorch + Hugging Face PEFT
pip install "./release/dev[peft]"

# Or Apple Silicon native MLX LoRA training (macOS arm64)
pip install "./release/dev[mlx]"

# Or NVIDIA CUDA 4-bit NF4 QLoRA
pip install "./release/dev[cuda]"
```

### Step 2 — Initialize Workspace & Add Synthetic or Personal Notes

```bash
mkdir my-ego-pod && cd my-ego-pod
ego init
```

Place `.md` or `.txt` files under `./ego-data/{sense,action,thought,skill,credentials}/`. Any embedded `sk-...`, PEM private keys, or high-entropy credential tokens are automatically scrubbed to `[REDACTED]` during `COMPILER` before SFT tokenization.

For a fast ~1–4 minute local smoke run on CPU or Apple Silicon without GPU requirements, set `editions.normal.base: qwen2.5-0.5B`, `epochs: 1`, and `lora_rank: 4` in `ego-config.yaml`:

```yaml
sources:
  sense:       { type: local, path: ./ego-data/sense }
  action:      { type: local, path: ./ego-data/action }
  thought:     { type: local, path: ./ego-data/thought }
  skill:       { type: local, path: ./ego-data/skill }
  credentials: { type: local, path: ./ego-data/credentials }

training:
  method: qlora
  quantization: 4bit
  lora_rank: 4
  lora_alpha: 8
  learning_rate: 0.0002
  epochs: 1
  batch_size: 4

editions:
  normal:
    base: qwen2.5-0.5B
    description: "Compact real PEFT LoRA edition (Qwen/Qwen2.5-0.5B-Instruct)"
    lora_rank: 4
    lora_alpha: 8
    max_level: 1

output:
  dir: ./ego-output
  max_history: 5
```

### Step 3 — Sign, Fine-Tune & Probe Consistency (`ego train`)

```bash
# Generate or export your 32-byte hex signing seed + canonical ifay_ FayID
export EGO_SIGNING_SEED_HEX="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
export EGO_SIGNER_FAY_ID="ifay_ed234567abcdefghij"

ego train
```

### Step 4 — Verify Signature & Run Interactive Multi-Cell Inference (`ego run`)

```bash
# Verify Tier-1 SHA-256 adapter digests + Tier-2 RFC 8785 JCS Ed25519 signature
python3 -c '
from ego_pod.vault.signing import verify_package_signature
res = verify_package_signature("./ego-output")
print(res.status, res.signer_fay_id, res.verified_cells)
'

# Launch interactive REPL with Coexist_Activate across Kernel + coding + knowledge
ego run --edition normal
```

---

## 3. Verified Real Benchmark Log (macOS Apple Silicon, Non-Stub)

Executed end-to-end on **2026-11-18** on an Apple Silicon Mac (`darwin arm64`, Python `3.12.7`, `torch 2.11.0`, `transformers 5.5.3`, `peft 0.18.1`, `safetensors 0.7.0`, `cryptography 43.0.0`) with **zero Hugging Face token** (`unauthenticated` Hub download):

### Truthful Run Metrics

| Metric | Recorded Value |
| --- | --- |
| **Configured Friendly Base** | `qwen2.5-0.5B` |
| **Resolved Hugging Face Repo ID** | `Qwen/Qwen2.5-0.5B-Instruct` (`gated: false`) |
| **HF Cache Weight Footprint** | `999,586,347 bytes` (`999.59 MB` in `~/.cache/huggingface/hub`) |
| **Selected Backend & Precision** | `peft-cpu` (`HuggingFace PEFT (CPU)`, `precision=fp32`, `lora_rank=4`, `lora_alpha=8`) |
| **Input Corpus & Redaction** | `10` local Markdown files across all 5 categories (`1` `sk-...` secret auto-scrubbed → `16` compiled SFT turns: `Kernel=9`, `coding/L1=4`, `knowledge/L1=3`) |
| **Per-Cell Final Training Losses (1 Epoch)** | `Kernel`: `4.947380` (3 steps) · `(coding, L1)`: `5.257201` (1 step) · `(knowledge, L1)`: `6.367732` (1 step) |
| **Wall-Clock Time (`ego train` total)** | `248.56 s` total (`197.14 s` trainer incl. first-time `999.59 MB` Hub download + `49.93 s` Vault signing & 2-pass `consistency_probe` inference) |
| **Exported `.safetensors` Adapter Sizes** | `1,092,064 bytes` (`1.04 MiB`) × 3 adapter-only files (`kernel.safetensors`, `coding/L1.safetensors`, `knowledge/L1.safetensors`) |
| **Holdout Metrics (`fidelity_report`)** | `status="measured"`, `holdout_sample_count=3`, `holdout_loss=3.073676`, `holdout_perplexity=21.621236` |
| **Post-Training `cell_consistency_score`** | **`1.0000`** (`min over 1 edition(s) x 2 holdout prompt(s)`; passes FayGer `>= 0.70` production gate) |
| **Cryptographic Bundle Signature** | **`verified`** (`Ed25519` over `ifay.ego.package-signing.v1`, `signer_fay_id="ifay_ed234567abcdefghij"`, `payload_jcs_sha256="fdc8e701790496bd06cb0930bd76c83cdbad2f674277a96025eacf113e828b28"`) |
| **Wall-Clock Time (`ego run --edition normal`)** | `79.32 s` (cold process model load + 3-Cell `Coexist_Activate` + greedy generation up to `256` new tokens) |

### Verbatim `ego train` Output

```text
Config validated: .../ego-real-smoke/ego-config.yaml
[2026-11-18T22:48:25+08:00] COLLECTOR START
sense: 2 files, 226 bytes
action: 2 files, 254 bytes
thought: 3 files, 398 bytes
skill: 2 files, 249 bytes
credentials: 1 files, 176 bytes
[2026-11-18T22:48:25+08:00] COLLECTOR END
[dojo progress] 25% COLLECTOR completed (10 files, 4ms)
[2026-11-18T22:48:25+08:00] COMPILER START
redact summary: total=1 sk=1 base64=0 pem=0
filter summary: dropped_short=0 merged_dup=0 total_before=10 total_after=10
增强摘要: original=10 augmented=6 final=16
grid summary: Kernel=9 (coding, L1)=4 (knowledge, L1)=3
[2026-11-18T22:48:25+08:00] COMPILER END
[dojo progress] 50% COMPILER completed (16 samples, 0ms)
[2026-11-18T22:48:25+08:00] TRAINER START
Selected backend: HuggingFace PEFT (CPU)
Loaded base model: model_id=qwen2.5-0.5B hf_repo=Qwen/Qwen2.5-0.5B-Instruct device=CPU quantization=none (precision=fp32)
[progress] epoch=1 step=3 loss=4.9474
Training complete: epochs=1 steps=3 final_loss=4.947380
Loaded base model: model_id=qwen2.5-0.5B hf_repo=Qwen/Qwen2.5-0.5B-Instruct device=CPU quantization=none (precision=fp32)
[progress] epoch=1 step=1 loss=5.2572
Training complete: epochs=1 steps=1 final_loss=5.257201
Loaded base model: model_id=qwen2.5-0.5B hf_repo=Qwen/Qwen2.5-0.5B-Instruct device=CPU quantization=none (precision=fp32)
[progress] epoch=1 step=1 loss=6.3677
Training complete: epochs=1 steps=1 final_loss=6.367732
[2026-11-18T22:51:42+08:00] TRAINER END
[dojo progress] 85% TRAINER completed (editions=normal, 197140ms)
[2026-11-18T22:51:42+08:00] VAULT START
[VAULT] ego-output directory created: .../ego-real-smoke/ego-output
Loaded base model: model_id=qwen2.5-0.5B hf_repo=Qwen/Qwen2.5-0.5B-Instruct device=CPU quantization=none (precision=fp32)
Loaded base model: model_id=qwen2.5-0.5B hf_repo=Qwen/Qwen2.5-0.5B-Instruct device=CPU quantization=none (precision=fp32)
[fidelity] cell_consistency_score=1.0000 (min over 1 edition(s) x 2 holdout prompt(s))
[2026-11-18T22:52:32+08:00] VAULT END
[dojo progress] 100% VAULT completed (49933ms)
cultivation summary: editions=normal samples=16 grid=[normal:3cells] fidelity=3.0737 elapsed_ms={collector:4,compiler:0,trainer:197140,vault:49933}
```

### Sample Prompt & Real Non-Stub `ego run --edition normal` Output

- **Prompt**: `"Summarize my weekday morning routine and how I structure technical memos."`
- **Loaded Cells (`Coexist_Activate`)**: `kernel` + `coding_L1` + `knowledge_L1` on `Qwen/Qwen2.5-0.5B-Instruct`
- **Generated Output (verbatim excerpt)**:
  > `1. Start with a clear introduction: Begin your memo with a concise summary of the problem or issue you're addressing. This should be brief enough to convey the core message without being overly verbose.`
  > `2. Provide context: Include any relevant background information that will help readers understand the importance of the issue and its impact.`
  > `3. Outline steps to resolve the issue: Clearly outline the actions you plan to take to address the problem. Be specific about what needs to be done, when it's due, and who is responsible for each step.`
  > `4. Include timelines ... 5. Address potential risks ... 6. Conclude with a call-to-action ...`
