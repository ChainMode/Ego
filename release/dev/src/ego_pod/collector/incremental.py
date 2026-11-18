"""基于内容 SHA-256 的增量采集清单（``.ego/collection-manifest.json``）与缓存复用（ADR 0006 §5）。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ego_pod.collector.base import CollectedFile
from ego_pod.compiler.classify import SourceSample
from ego_pod.compiler.normalize import normalize_text
from ego_pod.compiler.redact import RedactCounters, redact_text

__all__ = [
    "COLLECTION_MANIFEST_FILENAME",
    "IncrementalCollectionPlan",
    "compute_file_sha256",
    "compile_sources_incrementally",
]

COLLECTION_MANIFEST_FILENAME = "collection-manifest.json"


@dataclass(frozen=True)
class IncrementalCollectionPlan:
    """一次全量或 ``--update`` 增量源文本预处理的结果与差异统计。"""

    unchanged_count: int
    changed_or_new_count: int
    removed_count: int
    source_samples: list[SourceSample]
    redact_counters: RedactCounters


def compute_file_sha256(path: str | Path) -> str:
    """流式计算文件的十六进制 SHA-256 摘要。"""

    hasher = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            chunk = fh.read(65536)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_manifest_entries(manifest_path: Path) -> dict[str, dict[str, Any]]:
    if not manifest_path.is_file():
        return {}
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = raw.get("entries", {}) if isinstance(raw, dict) else {}
        return dict(entries) if isinstance(entries, dict) else {}
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}


def _save_manifest_entries(
    manifest_path: Path, entries: dict[str, dict[str, Any]]
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "entries": entries,
    }
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def compile_sources_incrementally(
    files: Sequence[CollectedFile],
    *,
    cwd: Path | None = None,
    update: bool = False,
) -> IncrementalCollectionPlan:
    """对采集文件列表执行内容哈希比对，在 ``update=True`` 时复用未变更文件的脱敏与规范化缓存。"""

    base_cwd = Path.cwd() if cwd is None else Path(cwd)
    manifest_path = base_cwd / ".ego" / COLLECTION_MANIFEST_FILENAME
    previous_entries = _load_manifest_entries(manifest_path) if update else {}

    new_entries: dict[str, dict[str, Any]] = {}
    source_samples: list[SourceSample] = []
    merged_counters = RedactCounters()
    unchanged = 0
    changed_or_new = 0

    for collected in files:
        file_path = Path(collected.absolute_path)
        digest = compute_file_sha256(file_path)
        entry_key = f"{collected.category}:{file_path.resolve()}"
        cached = previous_entries.get(entry_key)

        if (
            update
            and isinstance(cached, dict)
            and cached.get("sha256") == digest
            and cached.get("category") == collected.category
            and isinstance(cached.get("normalized_text"), str)
        ):
            unchanged += 1
            norm_text = str(cached["normalized_text"])
            raw_counts = cached.get("redact_counts", {})
            if isinstance(raw_counts, dict):
                file_counters = RedactCounters()
                for cat_k, bucket in raw_counts.items():
                    if isinstance(bucket, dict):
                        file_counters.counts[str(cat_k)] = {
                            str(pk): int(pv) for pk, pv in bucket.items()
                        }
                merged_counters.merge(file_counters)
            new_entries[entry_key] = cached
            source_samples.append(
                SourceSample(category=collected.category, text=norm_text)
            )
            continue

        changed_or_new += 1
        raw_text = file_path.read_bytes().decode("utf-8", errors="replace")
        redacted, file_counters = redact_text(raw_text, collected.category)
        merged_counters.merge(file_counters)
        norm_text = normalize_text(redacted)

        new_entries[entry_key] = {
            "category": collected.category,
            "path": str(file_path.resolve()),
            "size_bytes": int(collected.size_bytes),
            "sha256": digest,
            "normalized_text": norm_text,
            "redact_counts": file_counters.counts,
        }
        source_samples.append(
            SourceSample(category=collected.category, text=norm_text)
        )

    removed_count = (
        len(set(previous_entries.keys()) - set(new_entries.keys()))
        if update
        else 0
    )
    _save_manifest_entries(manifest_path, new_entries)

    return IncrementalCollectionPlan(
        unchanged_count=unchanged,
        changed_or_new_count=changed_or_new,
        removed_count=removed_count,
        source_samples=source_samples,
        redact_counters=merged_counters,
    )
