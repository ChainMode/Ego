"""Friendly base-model names → real Hugging Face repository IDs.

Ego's config files use short, stable *friendly* base names (``qwen2.5-0.5B``,
``qwen3.5-0.8B``, ``phi4-mini`` …, see
:data:`ego_pod.config.schema.SUPPORTED_BASE_MODELS`). Those names are **not**
Hugging Face repo IDs, so every real training/inference backend translates them
through :func:`resolve_hf_repo_id` before calling ``from_pretrained`` /
``mlx_lm.load``. This module is the single place where that mapping lives.

Each repo ID below was confirmed to exist on the Hugging Face Hub via
``https://huggingface.co/api/models/<org>/<name>`` on 2026-11-18, together with its
gating status (``gated: false`` means it downloads without license acceptance or
authentication token).

Friendly names whose family/size has no real published equivalent are kept
*accepted* by config validation (so existing ``ego-config.yaml`` files and the
Stub backend keep working), but real backends fail fast with
:class:`BaseModelResolutionError` (wrapped as :class:`~ego_pod.exit_codes.TrainerError`),
whose message lists the working choices.

Values that already look like a Hugging Face repo ID (``org/name``) or an existing
local directory are passed through unchanged, so advanced users and tests can point
backends at any checkpoint directly.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

__all__ = [
    "BaseModelInfo",
    "BaseModelResolutionError",
    "BASE_MODEL_REGISTRY",
    "RECOMMENDED_DEFAULT_BASE",
    "COMPACT_FAST_BASE",
    "is_hf_repo_id_syntax",
    "is_valid_base_model_spec",
    "real_base_model_choices",
    "resolve_hf_repo_id",
]

_HF_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class BaseModelInfo:
    """One friendly base name and its real Hugging Face counterpart (if any)."""

    #: Friendly name used in ``ego-config.yaml``.
    name: str
    #: Real Hugging Face repo ID, or ``None`` when no real equivalent exists.
    hf_repo_id: str | None
    #: ``True`` when the repo requires accepting a license on the Hub first.
    gated: bool = False
    #: Approximate size of the bf16/fp16 weights download, in GB (informational).
    approx_download_gb: float | None = None
    #: Short human note (e.g. architecture or why there is no equivalent).
    note: str = ""


#: Default friendly model written by ``ego init`` and the ``lite`` preset.
RECOMMENDED_DEFAULT_BASE = "qwen3.5-0.8B"

#: Smallest standard Transformer checkpoint (~0.99 GB) for fast CPU/MPS training.
COMPACT_FAST_BASE = "qwen2.5-0.5B"

#: Friendly name → real repo mapping (verified 2026-11-18 against the HF API).
BASE_MODEL_REGISTRY: dict[str, BaseModelInfo] = {
    info.name: info
    for info in (
        BaseModelInfo(
            "qwen2.5-0.5B",
            "Qwen/Qwen2.5-0.5B-Instruct",
            approx_download_gb=0.99,
            note="Compact ~0.99 GB standard Qwen2 Transformer; fastest CPU/MPS LoRA fine-tuning.",
        ),
        BaseModelInfo("qwen3.5-0.8B", "Qwen/Qwen3.5-0.8B", approx_download_gb=1.75),
        BaseModelInfo("qwen3.5-2B", "Qwen/Qwen3.5-2B", approx_download_gb=4.55),
        BaseModelInfo("qwen3.5-4B", "Qwen/Qwen3.5-4B", approx_download_gb=9.32),
        BaseModelInfo("qwen3.5-9B", "Qwen/Qwen3.5-9B", approx_download_gb=18.60),
        BaseModelInfo("qwen3.5-27B", "Qwen/Qwen3.5-27B", approx_download_gb=55.56),
        BaseModelInfo(
            "gemma4-2B",
            "google/gemma-4-E2B-it",
            approx_download_gb=10.25,
            note="Gemma 4 'E2B' (effective 2B) instruction-tuned checkpoint.",
        ),
        BaseModelInfo(
            "gemma4-9B",
            None,
            note="Gemma 4 has no 9B checkpoint on the Hub (published: E2B, E4B, 26B-A4B, 31B).",
        ),
        BaseModelInfo(
            "gemma4-27B",
            None,
            note="Gemma 4 has no 27B checkpoint on the Hub (published: E2B, E4B, 26B-A4B, 31B).",
        ),
        BaseModelInfo(
            "phi4-mini",
            "microsoft/Phi-4-mini-instruct",
            approx_download_gb=7.67,
        ),
        BaseModelInfo("phi4-14B", "microsoft/phi-4", approx_download_gb=29.32),
    )
}


class BaseModelResolutionError(ValueError):
    """A friendly base name has no real Hugging Face equivalent."""


def is_hf_repo_id_syntax(value: str) -> bool:
    """Return ``True`` when ``value`` matches standard Hugging Face ``org/repo`` syntax."""

    return isinstance(value, str) and bool(_HF_REPO_ID_RE.fullmatch(value))


def is_valid_base_model_spec(value: str) -> bool:
    """Return ``True`` for registered friendly names, ``org/repo`` IDs, or local model dirs."""

    if not isinstance(value, str) or not value.strip():
        return False
    if value in BASE_MODEL_REGISTRY:
        return True
    if os.path.isdir(value):
        return True
    return is_hf_repo_id_syntax(value)


def real_base_model_choices() -> tuple[str, ...]:
    """Friendly names that resolve to a real, ungated Hugging Face repo."""

    return tuple(
        name
        for name, info in BASE_MODEL_REGISTRY.items()
        if info.hf_repo_id is not None and not info.gated
    )


def resolve_hf_repo_id(name: str) -> str:
    """Translate a friendly base name into the identifier passed to the loader.

    * Registered friendly name with a real repo → that repo ID.
    * Registered friendly name without a real repo → :class:`BaseModelResolutionError`
      listing the working choices.
    * Anything else (``org/name`` repo IDs, local directories, unknown names) is
      returned unchanged so the underlying loader reports its own error.

    :param name: friendly name, repo ID or local path.
    :returns: the identifier to hand to ``from_pretrained`` / ``mlx_lm.load``.
    :raises BaseModelResolutionError: friendly name without a real equivalent.
    """

    if not isinstance(name, str):
        return name
    if os.path.isdir(name):
        return name
    info = BASE_MODEL_REGISTRY.get(name)
    if info is None:
        return name
    if info.hf_repo_id is None:
        choices = ", ".join(
            f"{n} ({BASE_MODEL_REGISTRY[n].hf_repo_id})" for n in real_base_model_choices()
        )
        raise BaseModelResolutionError(
            f"base model {name!r} has no real Hugging Face checkpoint "
            f"({info.note or 'no published equivalent'}); it only works with the "
            f"Stub backend. Pick one of: {choices}"
        )
    return info.hf_repo_id
