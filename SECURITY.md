# Security Policy

## Supported Versions

Security fixes and advisories cover the active `ego-pod` pipeline and specification baseline on `main`:

| Component / Artifact | Version / Branch | Supported |
| --- | --- | --- |
| `ego-pod` Python Pipeline (`release/dev`) & Schemas (`schema/`) | `main` / current `0.x` baseline | Yes |
| Architectural Blueprints (`docs/`) | `main` | Yes |
| Unmaintained experimental snapshots | Pre-baseline commits | No |

## Security Scope

`Ego` processes highly sensitive personal data belonging to a **Human Prime** to train local personality adapters for an `iFay`. In-scope security and privacy vulnerabilities include:

1. **Local Privacy Sanitization & Offline Isolation (`ego_pod`)**:
   - Unintended network egress of personal corpus data during collection, compilation, training, or inference, or bypasses in PII/secret scrubbing filters during corpus compilation.
2. **Zero Raw-Sample Leakage in Exported Artifacts & Metrics**:
   - Leakage of raw Human Prime text samples, prompts, or private identifiers into exported `manifest.json` metadata, `fidelity_report` scalar summaries, logs, or Sparse Grid routing metadata.
3. **Two-Tier `Ego_Package` Supply-Chain Signing (`ego_pod.vault.signing`)**:
   - Signature forgery, RFC 8785 JCS canonicalization mismatches, exposure of root `FayID` seeds instead of isolated `ifay.ego.package-signing.v1` `HKDF-SHA256` subkeys, or bypass of per-Cell `SHA-256` integrity verification during Vault loading and on-demand Sparse Grid Cell inference.
4. **Safe Model & Adapter Serialization**:
   - Unsafe deserialization vectors (for example, arbitrary code execution via untrusted pickle payloads instead of constrained `SafeTensors` weights and validated JSON/YAML manifests) or path traversal when unpacking `.ego` archives.

## Reporting a Vulnerability

**Please do NOT report undisclosed privacy leaks, signature bypasses, or deserialization vulnerabilities via public GitHub Issues or Pull Requests.**

- **Primary Channel (Recommended)**: Submit a report privately via **GitHub Private Vulnerability Reporting (Security Advisories)**:
  - https://github.com/ChainMode/Ego/security/advisories/new
- **Fallback Email Channel**: If GitHub Security Advisories is unavailable to you, email **[jah.g@ifay.ai](mailto:jah.g@ifay.ai)** with `[SECURITY][Ego]` in the subject line.

Please include affected `ego_pod` modules, minimal synthetic reproduction data (never include real personal data), and impact details.

## Coordinated Disclosure Policy

- **Acknowledgment**: Within **72 hours** of receiving your report.
- **Triage & Assessment**: Initial validation and remediation timeline within **7 calendar days**.
- **Coordinated Disclosure Window**: **90 days** from report submission to coordinated public disclosure.
- **Credit**: Security researchers receive credit in the published advisory unless anonymity is requested.
