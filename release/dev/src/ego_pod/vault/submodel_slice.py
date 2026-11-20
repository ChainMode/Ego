"""嵌套/切分子模型协议（`ego-submodel-slice/1.0`）与独立可运行 `.egopod` 切片导出器（`ego slice`）。

对应《iFay System 全栈可用 MVP 对齐规格备忘录》：
- 在协议/清单层正式定义 ``ego-submodel-slice/1.0`` 描述符（预留未来 Matryoshka 参数级嵌套子网络槽位）；
- MVP 阶段提供开箱可用的按场景/领域独立裁剪导出：仅提取 ``Kernel`` + 指定领域/深度 Cell，
  按 ``max_size_mb`` 自动削减深层与无关规格，重算独立 SHA-256 ``cell_digests``，写入
  ``slice_descriptor`` 并完成 RFC 8785 JCS + Ed25519 签名，使无人机相机或智能手表上的
  受限 ``FayGer`` 可直接独立验签并挂载该微型切片包。
"""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any

from ego_pod.exit_codes import ConfigError, VaultError
from ego_pod.runtime.selection import KERNEL, CellKey, GridCell
from ego_pod.service.profile import (
    COGNITIVE_PROFILE_FILENAME,
    build_default_cognitive_profile,
    prune_cognitive_profile_for_domains,
    read_cognitive_profile,
    write_cognitive_profile,
)
from ego_pod.vault.grid_descriptor import (
    FLAT_KERNEL_RELPATH,
    KERNEL_RELPATH,
    parse_grid_descriptor,
    zone_cell_relpath,
)
from ego_pod.vault.signing import (
    ENV_SIGNER_FAY_ID,
    canonicalize_jcs,
    compute_package_integrity,
    derive_ego_signing_subkey,
    sign_manifest,
    verify_package_signature,
)

__all__ = [
    "SLICE_SCHEMA_VERSION",
    "BUILTIN_SLICE_DOMAINS",
    "DEFAULT_DEVICE_CLASS_BY_DOMAIN",
    "SliceArtifacts",
    "build_slice_descriptor",
    "validate_slice_descriptor",
    "resolve_signing_material",
    "export_submodel_slice",
]

SLICE_SCHEMA_VERSION = "ego-submodel-slice/1.0"

BUILTIN_SLICE_DOMAINS: tuple[str, ...] = (
    "coding",
    "social",
    "knowledge",
    "creative",
    "sense",
    "drone-photo",
)

DEFAULT_DEVICE_CLASS_BY_DOMAIN: dict[str, str] = {
    "drone-photo": "drone-camera",
    "sense": "wearable-sensor",
    "social": "mobile-companion",
    "creative": "studio-edge",
    "coding": "workstation-dev",
    "knowledge": "tablet-reader",
}

_DOMAIN_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,31}$")
_SHA256_PREF_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True)
class SliceArtifacts:
    """一次 `ego slice` 成功导出的独立微型 `.egopod` 产物摘要。"""

    root: Path
    manifest_path: Path
    edition: str
    selected_domains: tuple[str, ...]
    included_cells: tuple[str, ...]
    actual_size_bytes: int
    signer_fay_id: str
    public_key_hex: str


def build_slice_descriptor(
    *,
    parent_manifest_sha256: str,
    selected_domains: Sequence[str],
    selected_edition: str,
    target_device_class: str,
    actual_size_bytes: int,
    max_size_mb: float | None = None,
    max_depth_by_domain: Mapping[str, int] | None = None,
    effective_lora_rank: int = 8,
     standalone_runnable: bool = True,
) -> dict[str, Any]:
    """构造符合 `ego-submodel-slice/1.0` 标准协议的 ``slice_descriptor`` 字典。"""

    domains = [str(d).strip() for d in selected_domains if str(d).strip()]
    descriptor: dict[str, Any] = {
        "schema_version": SLICE_SCHEMA_VERSION,
        "parent_manifest_sha256": parent_manifest_sha256,
        "selected_domains": domains,
        "selected_edition": selected_edition,
        "max_depth_by_domain": dict(max_depth_by_domain or {}),
        "slicing_mode": "cell-grid-prune",
        "matryoshka_parameter_slice": {
            "enabled": False,
            "effective_lora_rank": int(effective_lora_rank),
            "rank_prefix_ratio": 1.0,
        },
        "standalone_runnable": bool(standalone_runnable),
        "target_device_class": target_device_class,
        "max_size_mb": None if max_size_mb is None else round(float(max_size_mb), 4),
        "actual_size_bytes": int(actual_size_bytes),
    }
    validate_slice_descriptor(descriptor)
    return descriptor


def validate_slice_descriptor(descriptor: Mapping[str, Any]) -> None:
    """校验 ``manifest.slice_descriptor`` 是否满足 `ego-submodel-slice/1.0` 契约。"""

    if not isinstance(descriptor, Mapping):
        raise VaultError("slice_descriptor 必须为 JSON 对象")
    if descriptor.get("schema_version") != SLICE_SCHEMA_VERSION:
        raise VaultError(
            f"slice_descriptor.schema_version 必须为 {SLICE_SCHEMA_VERSION!r}，"
            f"实际为 {descriptor.get('schema_version')!r}"
        )
    parent_sha = descriptor.get("parent_manifest_sha256")
    if not isinstance(parent_sha, str) or not _SHA256_PREF_RE.match(parent_sha):
        raise VaultError(
            "slice_descriptor.parent_manifest_sha256 必须为 'sha256:<64-lowercase-hex>'"
        )
    domains = descriptor.get("selected_domains")
    if (
        not isinstance(domains, list)
        or not domains
        or any(not isinstance(d, str) or not _DOMAIN_NAME_RE.match(d) for d in domains)
    ):
        raise VaultError("slice_descriptor.selected_domains 必须为非空合法领域名数组")
    if not isinstance(descriptor.get("standalone_runnable"), bool):
        raise VaultError("slice_descriptor.standalone_runnable 必须为布尔值")
    target_cls = descriptor.get("target_device_class")
    if not isinstance(target_cls, str) or not target_cls.strip():
        raise VaultError("slice_descriptor.target_device_class 必须为非空字符串")
    actual_bytes = descriptor.get("actual_size_bytes")
    if isinstance(actual_bytes, bool) or not isinstance(actual_bytes, int) or actual_bytes < 0:
        raise VaultError("slice_descriptor.actual_size_bytes 必须为非负整数")


def _derive_fallback_fay_id(subkey_bytes: bytes) -> str:
    """当调用方仅传入签名密钥而未指定 FayID 时，从公钥确定性派生规范 `ifay_<base32>` 标识。"""

    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    pub_raw = Ed25519PrivateKey.from_private_bytes(subkey_bytes).public_key().public_bytes_raw()
    b32 = base64.b32encode(pub_raw).decode("ascii").lower().rstrip("=")
    return f"ifay_{b32[:32]}"


def resolve_signing_material(
    signing_key: str,
    *,
    signer_fay_id: str | None = None,
    inherited_fay_id: str | None = None,
) -> tuple[bytes, str]:
    """解析 `--signing-key`（支持 64 位 hex 种子、`subkey:<64hex>`、密钥文件、JSON 密钥包或 PEM）与签名者 `FayID`。

    返回 ``(32字节 Ed25519 子密钥 bytes, 规范 signer_fay_id)``。
    """

    if not isinstance(signing_key, str) or not signing_key.strip():
        raise ConfigError("--signing-key 不能为空")

    raw_arg = signing_key.strip()
    key_file_fay_id: str | None = None
    subkey_bytes: bytes | None = None

    if raw_arg.startswith("subkey:"):
        hex_part = raw_arg.split(":", 1)[1].strip()
        try:
            subkey_bytes = bytes.fromhex(hex_part)
        except ValueError as exc:
            raise VaultError("--signing-key subkey:<hex> 不是合法十六进制串") from exc
        if len(subkey_bytes) != 32:
            raise VaultError("--signing-key subkey 长度必须为 32 字节（64 位 hex）")
    elif re.fullmatch(r"[0-9a-fA-F]{64}", raw_arg):
        subkey_bytes = derive_ego_signing_subkey(raw_arg.lower())
    else:
        key_path = Path(raw_arg)
        if not key_path.is_file():
            raise VaultError(
                f"无法解析 --signing-key：既不是 64 位十六进制种子，也不是已存在的密钥文件（{raw_arg}）"
            )
        content = key_path.read_text(encoding="utf-8").strip()
        if content.startswith("{"):
            try:
                obj = json.loads(content)
            except json.JSONDecodeError as exc:
                raise VaultError(f"密钥 JSON 文件解析失败：{key_path}") from exc
            if not isinstance(obj, Mapping):
                raise VaultError(f"密钥 JSON 顶层必须为对象：{key_path}")
            if isinstance(obj.get("signer_fay_id"), str):
                key_file_fay_id = obj["signer_fay_id"].strip()
            elif isinstance(obj.get("fay_id"), str):
                key_file_fay_id = obj["fay_id"].strip()
            if isinstance(obj.get("subkey_hex"), str):
                subkey_bytes = bytes.fromhex(obj["subkey_hex"].strip())
            elif isinstance(obj.get("seed_hex"), str):
                subkey_bytes = derive_ego_signing_subkey(obj["seed_hex"].strip())
            else:
                raise VaultError(f"密钥 JSON 缺少 seed_hex 或 subkey_hex：{key_path}")
        elif "BEGIN" in content:
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                Ed25519PrivateKey,
            )

            priv = serialization.load_pem_private_key(
                content.encode("utf-8"), password=None
            )
            if not isinstance(priv, Ed25519PrivateKey):
                raise VaultError(f"PEM 密钥必须为 Ed25519 私钥：{key_path}")
            subkey_bytes = priv.private_bytes_raw()
        elif re.fullmatch(r"[0-9a-fA-F]{64}", content):
            subkey_bytes = derive_ego_signing_subkey(content.lower())
        else:
            raise VaultError(f"不支持的密钥文件格式：{key_path}")

    assert subkey_bytes is not None
    resolved_fay_id = (
        (signer_fay_id.strip() if isinstance(signer_fay_id, str) and signer_fay_id.strip() else None)
        or key_file_fay_id
        or (os.environ.get(ENV_SIGNER_FAY_ID, "").strip() or None)
        or (inherited_fay_id.strip() if isinstance(inherited_fay_id, str) and inherited_fay_id.strip() else None)
        or _derive_fallback_fay_id(subkey_bytes)
    )
    return subkey_bytes, resolved_fay_id


def _resolve_source_package_root(source_package: str | Path) -> tuple[Path, Path | None]:
    candidate = Path(source_package)
    if (candidate / "manifest.json").is_file():
        ws = candidate.parent if candidate.name == "ego-output" else None
        return candidate, ws
    if (candidate / "ego-output" / "manifest.json").is_file():
        return candidate / "ego-output", candidate
    raise VaultError(
        f"未找到源 Ego_Package（缺少 manifest.json）：{candidate}"
    )


def _normalize_domains(domains: Sequence[str]) -> list[str]:
    result: list[str] = []
    for raw in domains:
        for part in str(raw).split(","):
            item = part.strip().lower()
            if not item:
                continue
            if item == "custom":
                if item not in result:
                    result.append(item)
                continue
            if not _DOMAIN_NAME_RE.match(item):
                raise ConfigError(
                    f"非法的切片领域名 {item!r}（须匹配 ^[a-z0-9][a-z0-9-]{{0,31}}$）"
                )
            if item not in result:
                result.append(item)
    if not result:
        raise ConfigError("ego slice 至少需要指定一个 --domain")
    return result


def _map_domains_to_zones(
    selected_domains: Sequence[str],
    available_zones: Mapping[str, list[GridCell]],
) -> list[str]:
    """将请求领域映射为源包网格中实际需保留的 Functional_Zone 列表（无关分区一律剔除）。"""

    matched_zones: list[str] = []
    for dom in selected_domains:
        if dom in available_zones and available_zones[dom]:
            if dom not in matched_zones:
                matched_zones.append(dom)
            continue

        # 内置语义领域的优雅映射：例如 drone-photo / sense 的感知直觉默认蒸馏在 Kernel（或 sense 分区）
        if dom in ("drone-photo", "sense"):
            if "drone-photo" in available_zones and available_zones["drone-photo"]:
                if "drone-photo" not in matched_zones:
                    matched_zones.append("drone-photo")
            elif "sense" in available_zones and available_zones["sense"]:
                if "sense" not in matched_zones:
                    matched_zones.append("sense")
            # 否则仅保留 Kernel（sense 原生位于 Kernel），裁剪掉 coding / knowledge 等重分区！
            continue

        if dom in ("social", "creative"):
            if dom in available_zones and available_zones[dom]:
                if dom not in matched_zones:
                    matched_zones.append(dom)
            elif "knowledge" in available_zones and available_zones["knowledge"]:
                if "knowledge" not in matched_zones:
                    matched_zones.append("knowledge")
            continue

        if dom in ("coding", "knowledge", "custom"):
            # 若源包恰好为 Flat_Package（无分区），允许退化为仅含 Kernel 的独立切片
            continue

        # 非内置的自定义领域名若在源包网格中根本不存在，显式报错防止拼写误切
        raise VaultError(
            f"源包网格未包含请求的自定义领域分区 {dom!r}；"
            f"源包可用分区：{sorted(available_zones.keys())}，"
            f"内置语义领域：{list(BUILTIN_SLICE_DOMAINS)}"
        )

    return matched_zones


def _edition_kernel_size(pkg_root: Path, manifest: Mapping[str, Any], edition: str) -> int:
    grid = parse_grid_descriptor(manifest.get("grid"), edition)
    kernel_file = pkg_root / edition / grid.kernel.relpath
    return kernel_file.stat().st_size if kernel_file.is_file() else 0


def _pick_source_edition(
    pkg_root: Path,
    manifest: Mapping[str, Any],
    *,
    requested_edition: str | None,
    selected_domains: Sequence[str],
    max_size_mb: float | None,
) -> str:
    editions = [str(e) for e in manifest.get("editions", []) if isinstance(e, str)]
    if not editions:
        raise VaultError("源包 manifest.editions 为空，无法导出切片")

    if requested_edition is not None:
        if requested_edition not in editions:
            raise VaultError(
                f"源包不包含请求的规格 {requested_edition!r}（可用规格：{editions}）"
            )
        return requested_edition

    # 若面向无人机/穿戴传感器或显式指定了体积上限，优先挑体积最小的规格（如 lite）
    edge_oriented = max_size_mb is not None or any(
        d in ("drone-photo", "sense") for d in selected_domains
    )
    if edge_oriented and len(editions) > 1:
        return min(editions, key=lambda ed: _edition_kernel_size(pkg_root, manifest, ed))

    if "normal" in editions:
        return "normal"
    return editions[0]


def _dir_total_bytes(root: Path) -> int:
    return sum(
        p.stat().st_size
        for p in root.rglob("*")
        if p.is_file() and not p.is_symlink()
    )


def export_submodel_slice(
    source_package: str | Path,
    *,
    domains: Sequence[str],
    out_dir: str | Path,
    signing_key: str,
    signer_fay_id: str | None = None,
    edition: str | None = None,
    max_depth: int | None = None,
    max_size_mb: float | None = None,
    target_device_class: str | None = None,
) -> SliceArtifacts:
    """从源 `Ego_Package` 裁剪、重算摘要并签发独立可运行的微型 `.egopod` 切片包。"""

    from ego_pod.vault.package import (
        CONFIG_FILENAME,
        MANIFEST_FILENAME,
        TMP_DIRNAME,
        _atomic_replace_dir,
        _remove_orphans,
        _write_json,
        build_manifest,
    )

    if max_size_mb is not None and max_size_mb <= 0:
        raise ConfigError(f"--max-size-mb 必须大于 0，实际为 {max_size_mb}")
    if max_depth is not None and max_depth < 1:
        raise ConfigError(f"--max-depth 必须 >= 1，实际为 {max_depth}")

    selected_domains = _normalize_domains(domains)
    src_root, ws_root = _resolve_source_package_root(source_package)
    src_manifest_path = src_root / MANIFEST_FILENAME
    try:
        src_manifest = json.loads(src_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise VaultError(f"读取源包 manifest.json 失败：{exc}") from exc
    if not isinstance(src_manifest, Mapping):
        raise VaultError("源包 manifest.json 顶层必须为 JSON 对象")

    # 若源包本身已签名，导出切片前强制先验签，拒绝从被篡改的母包派生切片！
    if src_manifest.get("signature") is not None:
        verify_package_signature(src_root)

    inherited_fay_id: str | None = None
    sig_block = src_manifest.get("signature")
    if isinstance(sig_block, Mapping) and isinstance(sig_block.get("signer_fay_id"), str):
        inherited_fay_id = sig_block["signer_fay_id"]

    subkey_bytes, resolved_fay_id = resolve_signing_material(
        signing_key,
        signer_fay_id=signer_fay_id,
        inherited_fay_id=inherited_fay_id,
    )

    parent_payload_no_sig = {k: v for k, v in src_manifest.items() if k != "signature"}
    parent_manifest_sha256 = (
        "sha256:" + hashlib.sha256(canonicalize_jcs(parent_payload_no_sig)).hexdigest()
    )

    chosen_edition = _pick_source_edition(
        src_root,
        src_manifest,
        requested_edition=edition,
        selected_domains=selected_domains,
        max_size_mb=max_size_mb,
    )
    src_edition_dir = src_root / chosen_edition
    src_cfg_path = src_edition_dir / CONFIG_FILENAME
    if not src_cfg_path.is_file():
        raise VaultError(f"源包缺少规格配置：{src_cfg_path}")
    src_cfg = json.loads(src_cfg_path.read_text(encoding="utf-8"))

    src_grid = parse_grid_descriptor(src_manifest.get("grid"), chosen_edition)
    matched_zones = _map_domains_to_zones(selected_domains, src_grid.zones)

    # 为每个匹配分区挑选初始层级前缀 1..depth
    zone_cells_selected: dict[str, list[GridCell]] = {}
    for z in matched_zones:
        cells = list(src_grid.zones.get(z, []))
        if max_depth is not None:
            cells = [c for c in cells if c.cell.level <= max_depth]
        if cells:
            zone_cells_selected[z] = cells

    # 读取或构造认知画像并按领域裁剪（保留全部红线）
    src_profile_path = src_root / COGNITIVE_PROFILE_FILENAME
    ws_profile_path = (
        ws_root / ".ego" / COGNITIVE_PROFILE_FILENAME if ws_root is not None else None
    )
    if src_profile_path.is_file():
        base_profile = read_cognitive_profile(src_profile_path)
    elif ws_profile_path is not None and ws_profile_path.is_file():
        base_profile = read_cognitive_profile(ws_profile_path)
    else:
        base_profile = build_default_cognitive_profile()
    sliced_profile = prune_cognitive_profile_for_domains(base_profile, selected_domains)

    # 计算并按 max_size_mb 贪心削减最深层分区 Cell（保障 1..K 连续前缀与 Kernel 必含）
    kernel_src_file = src_edition_dir / src_grid.kernel.relpath
    if not kernel_src_file.is_file():
        raise VaultError(f"源包缺少 Kernel 适配器文件：{kernel_src_file}")

    limit_bytes = int(max_size_mb * 1024 * 1024) if max_size_mb is not None else None
    overhead_reserve_bytes = 8192  # 预留 manifest.json + config.json + cognitive_profile.json 元数据余量

    def _current_adapter_bytes() -> int:
        total = kernel_src_file.stat().st_size
        for z_list in zone_cells_selected.values():
            for gc in z_list:
                total += (src_edition_dir / gc.relpath).stat().st_size
        return total

    if limit_bytes is not None:
        while (
            zone_cells_selected
            and (_current_adapter_bytes() + overhead_reserve_bytes) > limit_bytes
        ):
            # 优先剥离当前层级最深（同深则文件更大）的分区末端 Cell
            deepest_zone = max(
                zone_cells_selected.keys(),
                key=lambda zn: (
                    zone_cells_selected[zn][-1].cell.level,
                    (src_edition_dir / zone_cells_selected[zn][-1].relpath).stat().st_size,
                ),
            )
            zone_cells_selected[deepest_zone].pop()
            if not zone_cells_selected[deepest_zone]:
                del zone_cells_selected[deepest_zone]

    resolved_device_class = (
        target_device_class.strip()
        if isinstance(target_device_class, str) and target_device_class.strip()
        else DEFAULT_DEVICE_CLASS_BY_DOMAIN.get(
            selected_domains[0], "constrained-edge"
        )
    )

    dest_root = Path(out_dir)
    dest_root.mkdir(parents=True, exist_ok=True)
    tmp_dir = dest_root / TMP_DIRNAME
    shutil.rmtree(tmp_dir, ignore_errors=True)

    is_flat_source = src_manifest.get("grid") is None and not zone_cells_selected

    try:
        tmp_ed = tmp_dir / chosen_edition
        tmp_ed.mkdir(parents=True, exist_ok=True)
        _write_json(tmp_ed / CONFIG_FILENAME, src_cfg)

        included_cell_labels: list[str] = ["kernel"]
        if is_flat_source:
            kernel_dst = tmp_ed / FLAT_KERNEL_RELPATH
            kernel_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(kernel_src_file, kernel_dst)
            grid_payload: dict[str, Any] | None = None
        else:
            kernel_dst = tmp_ed / KERNEL_RELPATH
            kernel_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(kernel_src_file, kernel_dst)
            zones_desc: dict[str, list[dict[str, Any]]] = {}
            for zn in sorted(zone_cells_selected.keys()):
                entries: list[dict[str, Any]] = []
                for gc in zone_cells_selected[zn]:
                    rel = zone_cell_relpath(zn, gc.cell.level)
                    dst_cell = tmp_ed / rel
                    dst_cell.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_edition_dir / gc.relpath, dst_cell)
                    entries.append({"level": gc.cell.level, "path": rel})
                    included_cell_labels.append(f"{zn}:L{gc.cell.level}")
                zones_desc[zn] = entries
            grid_payload = {
                chosen_edition: {
                    "kernel": {"path": KERNEL_RELPATH},
                    "zones": zones_desc,
                }
            }

        profile_sha256 = write_cognitive_profile(
            tmp_dir / COGNITIVE_PROFILE_FILENAME, sliced_profile
        )

        integrity = compute_package_integrity(
            tmp_dir,
            base_model_digest=str(
                (src_manifest.get("integrity") or {}).get(
                    "base_model_digest", "unverified_offline"
                )
            ),
        )

        depth_summary = {
            zn: cells[-1].cell.level for zn, cells in zone_cells_selected.items() if cells
        }
        lora_rank = int(src_cfg.get("lora_rank", 8))

        # 先构造含临时 actual_size_bytes 的 manifest 以测量真实落盘体积，再做精确回填并签名
        provisional_slice = build_slice_descriptor(
            parent_manifest_sha256=parent_manifest_sha256,
            selected_domains=selected_domains,
            selected_edition=chosen_edition,
            target_device_class=resolved_device_class,
            actual_size_bytes=_dir_total_bytes(tmp_dir),
            max_size_mb=max_size_mb,
            max_depth_by_domain=depth_summary,
            effective_lora_rank=lora_rank,
        )
        provisional_manifest = build_manifest(
            training_backend=str(src_manifest.get("training_backend", "stub")),
            base_model=str(src_cfg.get("base_model", src_manifest.get("base_model", "qwen3.5-0.8B"))),
            editions=(chosen_edition,),
            version=str(src_manifest.get("version", "0.2.0")),
            grid=grid_payload,
            fidelity_report=(
                dict(src_manifest["fidelity_report"])
                if isinstance(src_manifest.get("fidelity_report"), Mapping)
                else None
            ),
            integrity=integrity,
            slice_descriptor=provisional_slice,
            cognitive_profile_sha256=profile_sha256,
        )
        signed_provisional = sign_manifest(
            provisional_manifest,
            signer_fay_id=resolved_fay_id,
            subkey=subkey_bytes,
        )
        _write_json(tmp_dir / MANIFEST_FILENAME, signed_provisional)

        actual_bytes = _dir_total_bytes(tmp_dir)
        if limit_bytes is not None and actual_bytes > limit_bytes:
            actual_mb = actual_bytes / (1024 * 1024)
            raise VaultError(
                f"切片最小必含体积（Kernel + 元数据 = {actual_mb:.4f} MB）超出 "
                f"--max-size-mb={max_size_mb} MB 上限"
            )

        final_slice_desc = build_slice_descriptor(
            parent_manifest_sha256=parent_manifest_sha256,
            selected_domains=selected_domains,
            selected_edition=chosen_edition,
            target_device_class=resolved_device_class,
            actual_size_bytes=actual_bytes,
            max_size_mb=max_size_mb,
            max_depth_by_domain=depth_summary,
            effective_lora_rank=lora_rank,
        )
        final_manifest_unsigned = dict(provisional_manifest)
        final_manifest_unsigned["slice_descriptor"] = final_slice_desc
        final_manifest_signed = sign_manifest(
            final_manifest_unsigned,
            signer_fay_id=resolved_fay_id,
            subkey=subkey_bytes,
        )
        _write_json(tmp_dir / MANIFEST_FILENAME, final_manifest_signed)
        actual_bytes = _dir_total_bytes(tmp_dir)
        if limit_bytes is not None and actual_bytes > limit_bytes:
            actual_mb = actual_bytes / (1024 * 1024)
            raise VaultError(
                f"切片总体积 {actual_mb:.4f} MB 超出 --max-size-mb={max_size_mb} MB 上限"
            )

        _atomic_replace_dir(tmp_ed, dest_root / chosen_edition)
        os.replace(
            tmp_dir / COGNITIVE_PROFILE_FILENAME,
            dest_root / COGNITIVE_PROFILE_FILENAME,
        )
        os.replace(tmp_dir / MANIFEST_FILENAME, dest_root / MANIFEST_FILENAME)
        _remove_orphans(dest_root, keep_editions=(chosen_edition,))
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    ver = verify_package_signature(dest_root)
    return SliceArtifacts(
        root=dest_root.resolve(),
        manifest_path=(dest_root / MANIFEST_FILENAME).resolve(),
        edition=chosen_edition,
        selected_domains=tuple(selected_domains),
        included_cells=tuple(included_cell_labels),
        actual_size_bytes=_dir_total_bytes(dest_root),
        signer_fay_id=ver.signer_fay_id or resolved_fay_id,
        public_key_hex=ver.public_key_hex or "",
    )
