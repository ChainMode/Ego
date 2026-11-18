# Publishing `ego-pod` to PyPI

`ego-pod` is packaged as a PEP 621 / `setuptools` distribution rooted at [pyproject.toml](./pyproject.toml).

## 1. Pre-Release Checklist

1. Bump `__version__` in [src/ego_pod/\_\_init\_\_.py](./src/ego_pod/__init__.py) (dynamically read via `[tool.setuptools.dynamic]`).
2. Run the full unit, property-based, and integration test suite:
   ```bash
   cd release/dev
   pytest
   ```
3. Run the repository documentation and link integrity gate from the repo root:
   ```bash
   # Verifies tracked Markdown links, fences, and canonical terminology
   git status
   ```

## 2. Local Build & Metadata Verification (`build` + `twine check`)

Use an isolated virtual environment to build both source distribution (`sdist`) and wheel (`bdist_wheel`) artifacts and validate PyPI rendering metadata:

```bash
cd release/dev
python3 -m venv .venv-release
source .venv-release/bin/activate
pip install --upgrade pip build twine

# Clean any stale artifacts
rm -rf dist/ build/

# Build sdist (.tar.gz) and universal wheel (.whl)
python -m build

# Validate Core Metadata 2.1+ and README Markdown rendering
twine check --strict dist/*
```

Inspect the wheel contents before publishing to confirm zero test fixtures or local `.ego/` / `ego-output/` directories are bundled:

```bash
python -m zipfile -l dist/ego_pod-*.whl
```

## 3. Publishing via GitHub Actions OIDC Trusted Publisher (Recommended)

`ego-pod` is designed for **PyPI Trusted Publishing** (OpenID Connect — zero long-lived API tokens stored in GitHub Secrets):

1. **Configure Trusted Publisher on PyPI / TestPyPI**:
   - **PyPI Project Name**: `ego-pod`
   - **Owner**: `ChainMode`
   - **Repository**: `Ego`
   - **Workflow Filename**: `release-pypi.yml`
   - **Environment Name**: `pypi` (and `testpypi` for staging)
2. **Workflow Job Permissions**:
   ```yaml
   permissions:
     id-token: write
     contents: read
   steps:
     - uses: actions/checkout@v4
     - uses: actions/setup-python@v5
       with:
         python-version: "3.12"
     - name: Build distribution
       working-directory: release/dev
       run: |
         python -m pip install --upgrade build twine
         python -m build
         twine check --strict dist/*
     - name: Publish to PyPI via Trusted Publisher
       uses: pypa/gh-action-pypi-publish@release/v1
       with:
         packages-dir: release/dev/dist/
   ```

## 4. Clean Up Temporary Environments

After verifying locally, remove temporary release build artifacts (already gitignored):

```bash
rm -rf release/dev/dist release/dev/build release/dev/.venv-release
```
