"""Vault 版本历史归档、列表查询、完整性验签回滚与彻底清理（ADR 0006 §6）。

所有历史快照均保存在 ``ego-output/.history/<version>-<seq>/`` 下，满足：
1. **FIFO 保留上限**：默认最多保留最近 ``max_history = 5`` 份历史快照，超出时自动淘汰最旧快照。
2. **回滚前强制完整性与签名核验**：调用 :func:`rollback_vault_snapshot` 时，在触碰活动
   ``ego-output/`` 之前先对目标快照做 Tier 2 Ed25519 签名（若已签名或指定公钥）与 Tier 1
   ``cell_digests`` 全量校验；校验失败立即拒绝且零修改。
3. **原子可逆恢复**：快照恢复经 ``.tmp-rollback`` 暂存与旧目录备份交换完成，遭遇 I/O 错误时
   自动恢复原状。
4. **隐私彻底清除**：:func:`purge_vault_and_cache` 一键清空 ``ego-output/``（含 ``.history/``）
   与 ``.ego/`` 中间缓存。
"""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from ego_pod.exit_codes import VaultError
from ego_pod.vault.signing import _sha256_file_prefixed, verify_package_signature

__all__ = [
    "HISTORY_DIRNAME",
    "DEFAULT_MAX_HISTORY",
    "SnapshotInfo",
    "archive_existing_package",
    "list_vault_snapshots",
    "rollback_vault_snapshot",
    "purge_vault_and_cache",
]

HISTORY_DIRNAME = ".history"
DEFAULT_MAX_HISTORY = 5

_SNAPSHOT_SEQ_RE = re.compile(r"^(?P<ver>[A-Za-z0-9._-]+)-(?P<seq>\d{4,})$")
_SAFE_VER_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class SnapshotInfo:
    """单份历史快照的元数据摘要。"""

    snapshot_id: str
    path: Path
    version: str
    created_at: str
    editions: tuple[str, ...]
    signed: bool
    seq: int = 0


def _sanitize_version(raw_ver: str) -> str:
    cleaned = _SAFE_VER_RE.sub("_", raw_ver.strip())
    return cleaned or "0.0.0"


def _parse_snapshot_dir(snap_dir: Path) -> SnapshotInfo | None:
    manifest_file = snap_dir / "manifest.json"
    if not manifest_file.is_file():
        return None
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(manifest, Mapping):
        return None

    m = _SNAPSHOT_SEQ_RE.match(snap_dir.name)
    seq = int(m.group("seq")) if m else 0
    editions_raw = manifest.get("editions")
    editions = (
        tuple(str(e) for e in editions_raw)
        if isinstance(editions_raw, list)
        else ()
    )
    return SnapshotInfo(
        snapshot_id=snap_dir.name,
        path=snap_dir.resolve(),
        version=str(manifest.get("version", "unknown")),
        created_at=str(manifest.get("created_at", "")),
        editions=editions,
        signed=isinstance(manifest.get("signature"), Mapping),
        seq=seq,
    )


def list_vault_snapshots(output_root: str | Path) -> list[SnapshotInfo]:
    """按归档时间升序（旧 -> 新）列出 ``ego-output/.history/`` 下的有效快照。"""

    hist_root = Path(output_root) / HISTORY_DIRNAME
    if not hist_root.is_dir():
        return []

    snapshots: list[SnapshotInfo] = []
    for child in hist_root.iterdir():
        if child.name.startswith(".") or not child.is_dir() or child.is_symlink():
            continue
        info = _parse_snapshot_dir(child)
        if info is not None:
            snapshots.append(info)

    snapshots.sort(key=lambda s: (s.seq, s.snapshot_id))
    return snapshots


def archive_existing_package(
    output_root: str | Path,
    *,
    max_history: int = DEFAULT_MAX_HISTORY,
) -> SnapshotInfo | None:
    """在覆盖活动 ``ego-output/`` 前将其完整归档至 ``.history/<version>-<seq>/``。"""

    root = Path(output_root)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return None

    hist_root = root / HISTORY_DIRNAME
    if max_history <= 0:
        shutil.rmtree(hist_root, ignore_errors=True)
        return None

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(manifest, Mapping):
        return None

    existing = list_vault_snapshots(root)
    next_seq = (max((s.seq for s in existing), default=0)) + 1
    safe_ver = _sanitize_version(str(manifest.get("version", "0.0.0")))
    snapshot_id = f"{safe_ver}-{next_seq:04d}"

    hist_root.mkdir(parents=True, exist_ok=True)
    staging = hist_root / f".staging-{snapshot_id}-{os.getpid()}"
    final_snap = hist_root / snapshot_id
    shutil.rmtree(staging, ignore_errors=True)

    skip_names = {HISTORY_DIRNAME, ".tmp"}
    try:
        staging.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest_path, staging / "manifest.json")
        for child in root.iterdir():
            if (
                child.name in skip_names
                or child.name == "manifest.json"
                or child.name.startswith(".old")
                or ".old." in child.name
            ):
                continue
            if child.is_dir() and not child.is_symlink():
                shutil.copytree(child, staging / child.name)
        os.replace(staging, final_snap)
    except OSError:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    # FIFO 淘汰超过 max_history 的最旧快照
    updated_snapshots = list_vault_snapshots(root)
    while len(updated_snapshots) > max_history:
        oldest = updated_snapshots.pop(0)
        shutil.rmtree(oldest.path, ignore_errors=True)

    return _parse_snapshot_dir(final_snap)


def _verify_snapshot_before_restore(
    snap_dir: Path,
    *,
    trusted_public_key_hex: str | None = None,
) -> Mapping[str, object]:
    """在触碰活动产物前对目标快照执行完整的签名与 SHA-256 摘要校验。"""

    manifest_file = snap_dir / "manifest.json"
    if not manifest_file.is_file():
        raise VaultError(f"快照已损坏：缺少 manifest.json（{snap_dir}）")

    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise VaultError(f"快照 manifest.json 解析失败：{exc}") from exc
    if not isinstance(manifest, Mapping):
        raise VaultError("快照 manifest.json 顶层必须为 JSON 对象")

    if manifest.get("signature") is not None or trusted_public_key_hex is not None:
        verify_package_signature(
            snap_dir, trusted_public_key_hex=trusted_public_key_hex
        )
        return manifest

    integrity = manifest.get("integrity")
    if isinstance(integrity, Mapping):
        cell_digests = integrity.get("cell_digests")
        if not isinstance(cell_digests, Mapping) or not cell_digests:
            raise VaultError("快照 integrity.cell_digests 缺失或为空")
        for rel_posix, expected_digest in cell_digests.items():
            target_file = snap_dir / str(rel_posix)
            if not target_file.is_file():
                raise VaultError(f"快照适配器文件缺失：{rel_posix}")
            actual = _sha256_file_prefixed(target_file)
            if actual != expected_digest:
                raise VaultError(
                    f"快照适配器文件 SHA-256 校验不匹配（疑似损坏或篡改）：{rel_posix}"
                )
        return manifest

    editions = manifest.get("editions")
    if not isinstance(editions, list) or not editions:
        raise VaultError("快照 manifest.editions 非法或为空")
    for ed in editions:
        ed_dir = snap_dir / str(ed)
        if not (ed_dir / "config.json").is_file():
            raise VaultError(f"快照缺少 {ed}/config.json")
        safetensors = list((ed_dir / "adapters").rglob("*.safetensors")) if (ed_dir / "adapters").is_dir() else []
        if not safetensors:
            raise VaultError(f"快照规格 {ed} 下缺少任何 .safetensors 适配器文件")
    return manifest


def rollback_vault_snapshot(
    output_root: str | Path,
    snapshot_id: str,
    *,
    trusted_public_key_hex: str | None = None,
) -> SnapshotInfo:
    """校验指定历史快照并将其原子恢复为当前活动 ``ego-output/``。"""

    if (
        not snapshot_id
        or "/" in snapshot_id
        or "\\" in snapshot_id
        or ".." in snapshot_id
    ):
        raise VaultError(f"非法的快照标识：{snapshot_id!r}")

    root = Path(output_root)
    available = list_vault_snapshots(root)
    available_ids = [s.snapshot_id for s in available]
    snap_dir = root / HISTORY_DIRNAME / snapshot_id
    if not snap_dir.is_dir():
        avail_text = ", ".join(available_ids) if available_ids else "（当前无任何历史快照）"
        raise VaultError(
            f"未找到历史快照 '{snapshot_id}'；可用快照列表：{avail_text}"
        )

    manifest = _verify_snapshot_before_restore(
        snap_dir, trusted_public_key_hex=trusted_public_key_hex
    )
    info = _parse_snapshot_dir(snap_dir)
    if info is None:
        raise VaultError(f"无法读取快照元数据：{snapshot_id}")

    editions = [str(e) for e in manifest.get("editions", [])]  # type: ignore[arg-type]
    staging = root / f".tmp-rollback-{os.getpid()}"
    backup = root / f".backup-rollback-{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)
    shutil.rmtree(backup, ignore_errors=True)

    skip_names = {
        HISTORY_DIRNAME,
        ".tmp",
        staging.name,
        backup.name,
    }

    moved_to_backup: list[str] = []
    placed_from_staging: list[str] = []
    try:
        staging.mkdir(parents=True, exist_ok=True)
        shutil.copy2(snap_dir / "manifest.json", staging / "manifest.json")
        for ed in editions:
            shutil.copytree(snap_dir / ed, staging / ed)

        backup.mkdir(parents=True, exist_ok=True)
        for child in list(root.iterdir()):
            if child.name in skip_names:
                continue
            os.replace(child, backup / child.name)
            moved_to_backup.append(child.name)

        for ed in editions:
            os.replace(staging / ed, root / ed)
            placed_from_staging.append(ed)
        os.replace(staging / "manifest.json", root / "manifest.json")
        placed_from_staging.append("manifest.json")

        shutil.rmtree(backup, ignore_errors=True)
    except OSError as exc:
        # 逆向还原以保证活动 ego-output 绝不处于半损毁态
        for name in placed_from_staging:
            target = root / name
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
            else:
                target.unlink(missing_ok=True)
        for name in moved_to_backup:
            src_bak = backup / name
            if src_bak.exists():
                try:
                    os.replace(src_bak, root / name)
                except OSError:
                    pass
        shutil.rmtree(backup, ignore_errors=True)
        raise VaultError(f"回滚快照 '{snapshot_id}' 时发生 I/O 错误：{exc}") from exc
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    return info


def _dir_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    for item in path.rglob("*"):
        if item.is_file() and not item.is_symlink():
            try:
                total += item.stat().st_size
            except OSError:
                pass
    return total


def purge_vault_and_cache(
    cwd: str | Path,
    *,
    output_root: str | Path | None = None,
) -> tuple[list[Path], int]:
    """彻底清除 ``ego-output/``（含 ``.history/``）与 ``.ego/`` 中间缓存目录。"""

    base_cwd = Path(cwd)
    out_dir = base_cwd / "ego-output" if output_root is None else Path(output_root)
    ego_cache_dir = base_cwd / ".ego"

    deleted: list[Path] = []
    freed_bytes = 0
    for target in (out_dir, ego_cache_dir):
        if target.exists():
            freed_bytes += _dir_size_bytes(target)
            resolved = target.resolve()
            shutil.rmtree(target)
            deleted.append(resolved)
    return deleted, freed_bytes
