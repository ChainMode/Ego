# Contributing to Ego

Thank you for contributing to **Ego**! All contributors are expected to abide by our [Code of Conduct](./CODE_OF_CONDUCT.md). Please report privacy, model-signing, or security vulnerabilities privately via [SECURITY.md](./SECURITY.md).

## Local Setup & Verification

The `ego-pod` training and inference pipeline resides in `release/dev` and requires **Python `>= 3.10`**.

```bash
# Clone your fork
git clone https://github.com/ChainMode/Ego.git
cd Ego/release/dev

# Create a virtual environment and install editable package + test dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"

# Run the unit and property-based test suite
pytest
```

Never include real personal data in test fixtures, sample corpora, or Pull Request descriptions; use deterministic synthetic fixtures only.

## Contribution Workflow

1. **Fork & Branch**: Fork `https://github.com/ChainMode/Ego.git` and create a topic branch from `main`.
2. **Conventional Commits**: Follow Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`).
3. **Sign Off Your Commits (DCO 1.1)**: Use `git commit -s` so every commit includes a `Signed-off-by` line.
4. **Pull Request**: Open a Pull Request against `main` detailing privacy/security considerations and `pytest` results.

## Developer Certificate of Origin v1.1 (DCO 1.1) & Licensing

We use the **Developer Certificate of Origin Version 1.1 (DCO 1.1)** (`https://developercertificate.org/`). **No separate CLA is required.**

Sign off every commit with `git commit -s`:

```text
Signed-off-by: Jane Doe <jane.doe@example.com>
```

Under **Inbound = Outbound**, your contributions are licensed under our dual-track open-source licenses (`Copyright (c) 2025-2026 ChainMode contributors`):

- **Source Code, Schemas & Pipeline Package** (`release/dev/`, `schema/`): Licensed under the **MIT License** ([LICENSE](./LICENSE)).
- **Specifications & Architectural Blueprints** (`docs/`): Licensed under **Creative Commons Attribution 4.0 International (CC BY 4.0)** ([LICENSE-DOCS](./LICENSE-DOCS)).

## Ecosystem Terminology Alignment

Please maintain canonical ecosystem terminology across code, schemas, and documentation:

- **`Human Prime` (`人类原型`)**: Always use `Human Prime` (`人类原型`) for the living human individual whose traits an `Ego` model aligns with.
- **`Host` (`宿主`)**: Use `Host` (`宿主`) strictly for terminal hardware devices, operating systems, or local runtime environments executing training or inference.
- **`Fay = iFay + coFay`**: Use `Fay` as the umbrella term, `iFay` for an individual's personal avatar (which loads an `Ego`), and `coFay` for a public/organizational role avatar (which operates under a `RoleCharter` without an `Ego` layer).
- **`GMChain` (`公誉链`) & `MeriToken (μ)` (`美誉值`)**: Use `GMChain` (`公誉链`) and non-monetary `MeriToken` (`μ` / `Merit Unit`) when referencing ecosystem social contribution records.
