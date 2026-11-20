"""基于 FayID 派生子密钥的两级（Tier 1 摘要 + Tier 2 JCS Ed25519）模型包签名与验签（ADR 0003）。

本模块实现与 ``FayGer`` 跨仓逐字节对齐的 ``Ego_Package`` 供应链防篡改契约：

1. **子密钥派生（:func:`derive_ego_signing_subkey`）**：经 HKDF-SHA256
   （``salt=b"ifay.fayid.hkdf.v1"``，``info=b"ifay.ego.package-signing.v1"``）从
   32 字节主种子确定性派生 32 字节 Ed25519 签名子密钥，确保人类原型根私钥永不入包。
2. **RFC 8785 JCS 规范化（:func:`canonicalize_jcs`）**：键按 UTF-16 码元升序排列、
   紧凑分隔符 ``(",", ":")``、非 ASCII UTF-8 直出、数值遵循 ES6 / RFC 8785 规范化。
3. **Tier 1 完整性清单（:func:`compute_package_integrity`）**：记录相对包根的 POSIX
   路径 ``cell_digests``（``sha256:<64-lowercase-hex>``）与 ``base_model_digest``。
4. **Tier 2 Ed25519 签名与按需子集验签（:func:`sign_manifest` /
   :func:`verify_package_signature`）**：对 ``b"ifay.ego.package-signing.v1:" + jcs_bytes``
   签名/验签，并支持 Sparse Grid 按需加载仅校验 ``selected_cell_rel_paths`` 指定的适配器。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ego_pod.exit_codes import VaultError

__all__ = [
    "INTEGRITY_SCHEMA_VERSION",
    "SIGNING_SCHEMA_VERSION",
    "SIGNING_ALG",
    "SIGNING_DOMAIN",
    "HKDF_SALT",
    "HKDF_INFO",
    "KEY_ID_SUFFIX",
    "ENV_SIGNING_SEED_HEX",
    "ENV_SIGNER_FAY_ID",
    "ENV_TRUSTED_PUBKEY_HEX",
    "SignatureVerificationResult",
    "derive_ego_signing_subkey",
    "canonicalize_jcs",
    "compute_package_integrity",
    "sign_manifest",
    "verify_package_signature",
]

INTEGRITY_SCHEMA_VERSION = "1.0"
SIGNING_SCHEMA_VERSION = "1.0"
SIGNING_ALG = "Ed25519"
SIGNING_DOMAIN = "ifay.ego.package-signing.v1"

HKDF_SALT = b"ifay.fayid.hkdf.v1"
HKDF_INFO = b"ifay.ego.package-signing.v1"
KEY_ID_SUFFIX = "#ego-sign-1"

ENV_SIGNING_SEED_HEX = "EGO_SIGNING_SEED_HEX"
ENV_SIGNER_FAY_ID = "EGO_SIGNER_FAY_ID"
ENV_TRUSTED_PUBKEY_HEX = "EGO_TRUSTED_PUBKEY_HEX"

_HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")
_HEX_128_RE = re.compile(r"^[0-9a-f]{128}$")
_SHA256_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
#: Canonical iFay FayID: FayID schema ``^ifay_[a-z2-7]+$`` (lowercase RFC 4648 base32,
#: no padding), plus the >=16-char body FayGer's loader enforces. Real IDs derived
#: from 32-byte digests are 52 chars, so both rules agree on them.
_CANONICAL_IFAY_ID_RE = re.compile(r"^ifay_[a-z2-7]{16,}$")


def _require_canonical_ifay_id(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not _CANONICAL_IFAY_ID_RE.match(value):
        raise VaultError(
            f"{label} 不是规范 iFay FayID（须为 'ifay_' + 至少 16 位小写 base32 [a-z2-7]），"
            f"实际为 {value!r}"
        )
    return value


@dataclass(frozen=True)
class SignatureVerificationResult:
    """一次 ``Ego_Package`` 验签与完整性核验的结构化结果。"""

    status: Literal["verified", "unsigned"]
    signer_fay_id: str | None = None
    key_id: str | None = None
    public_key_hex: str | None = None
    payload_jcs_sha256: str | None = None
    verified_cells: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# 1. 字节编解码与 HKDF-SHA256 子密钥派生
# ---------------------------------------------------------------------------


def _coerce_32_bytes(material: bytes | str, *, label: str) -> bytes:
    """将 32 字节 ``bytes`` 或 64 位十六进制字符串规整为 32 字节 ``bytes``。"""

    if isinstance(material, str):
        cleaned = material.strip()
        try:
            raw = bytes.fromhex(cleaned)
        except ValueError as exc:
            raise VaultError(f"{label} 不是合法的十六进制字符串") from exc
    elif isinstance(material, (bytes, bytearray, memoryview)):
        raw = bytes(material)
    else:
        raise VaultError(f"{label} 必须为 32 字节 bytes 或 64 字符十六进制串")

    if len(raw) != 32:
        raise VaultError(f"{label} 长度必须恰好为 32 字节，实际为 {len(raw)} 字节")
    return raw


def derive_ego_signing_subkey(master_seed: bytes | str) -> bytes:
    """从 32 字节 FayID 主种子经 HKDF-SHA256 派生 32 字节 Ed25519 包签名子密钥。"""

    seed_bytes = _coerce_32_bytes(master_seed, label="master_seed")
    try:
        from cryptography.hazmat.primitives import hashes  # noqa: PLC0415
        from cryptography.hazmat.primitives.kdf.hkdf import HKDF  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - cryptography 为必装依赖
        raise VaultError(f"无法派生签名子密钥：缺少 cryptography 依赖（{exc}）") from exc

    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=HKDF_SALT,
        info=HKDF_INFO,
    )
    return hkdf.derive(seed_bytes)


# ---------------------------------------------------------------------------
# 2. RFC 8785 JSON Canonicalization Scheme (JCS)
# ---------------------------------------------------------------------------


def _format_jcs_float(value: float) -> str:
    """按 RFC 8785 §3.2.2.3（ECMA-262 Number::toString）格式化有限 IEEE-754 浮点数。"""

    if math.isnan(value) or math.isinf(value):
        raise VaultError("RFC 8785 JCS 规范化不允许 NaN 或 Inf 浮点数")
    if value == 0.0:
        return "0"

    sign = "-" if value < 0.0 else ""
    text = repr(abs(value)).lower()
    if "e" in text:
        mantissa, exp_str = text.split("e", 1)
        exp = int(exp_str)
    else:
        mantissa, exp = text, 0

    if "." in mantissa:
        int_part, frac_part = mantissa.split(".", 1)
        raw_digits = int_part + frac_part
        exp -= len(frac_part)
    else:
        raw_digits = mantissa

    # 剥离前导零与尾随零，得到有效数字串 digits（长度 k）与十进制阶 n（满足 |v| = int(digits) * 10^(n - k)）。
    stripped_leading = raw_digits.lstrip("0")
    digits = stripped_leading.rstrip("0")
    trailing_zeros = len(stripped_leading) - len(digits)
    exp += trailing_zeros
    k = len(digits)
    n = exp + k

    if k <= n <= 21:
        body = digits + ("0" * (n - k))
    elif 0 < n <= 21:
        body = f"{digits[:n]}.{digits[n:]}"
    elif -6 < n <= 0:
        body = "0." + ("0" * (-n)) + digits
    else:
        exponent_val = n - 1
        exp_sign = "+" if exponent_val >= 0 else "-"
        coeff = digits if k == 1 else f"{digits[0]}.{digits[1:]}"
        body = f"{coeff}e{exp_sign}{abs(exponent_val)}"

    return f"{sign}{body}"


def _serialize_jcs_value(value: Any) -> str:
    """递归将 JSON 兼容 Python 值序列化为紧凑 RFC 8785 JCS 字符串。"""

    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _format_jcs_float(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_serialize_jcs_value(item) for item in value) + "]"
    if isinstance(value, Mapping):
        items: list[tuple[bytes, str, Any]] = []
        for k, v in value.items():
            if not isinstance(k, str):
                raise VaultError(f"JCS 对象键必须为字符串，实际为 {type(k).__name__}")
            items.append((k.encode("utf-16-be"), k, v))
        items.sort(key=lambda entry: entry[0])
        members = [
            f"{json.dumps(k, ensure_ascii=False)}:{_serialize_jcs_value(v)}"
            for _sort_key, k, v in items
        ]
        return "{" + ",".join(members) + "}"
    raise VaultError(f"JCS 不支持序列化类型：{type(value).__name__}")


def canonicalize_jcs(obj: Any) -> bytes:
    """将 JSON 兼容对象按 RFC 8785 JCS 规范化为 UTF-8 字节串。"""

    return _serialize_jcs_value(obj).encode("utf-8")


# ---------------------------------------------------------------------------
# 3. Tier 1 文件摘要计算（compute_package_integrity）
# ---------------------------------------------------------------------------


def _sha256_file_prefixed(path: Path) -> str:
    """流式计算文件 SHA-256 并返回 ``sha256:<64-lowercase-hex>``。"""

    hasher = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(65536)
                if not chunk:
                    break
                hasher.update(chunk)
    except OSError as exc:
        raise VaultError(f"计算适配器摘要失败：{path}；原因：{exc}") from exc
    return f"sha256:{hasher.hexdigest()}"


def _validate_base_model_digest(digest: str) -> None:
    if not isinstance(digest, str) or (
        digest != "unverified_offline" and not _SHA256_DIGEST_RE.match(digest)
    ):
        raise VaultError(
            "integrity.base_model_digest 必须为 'unverified_offline' 或 "
            f"'sha256:<64-lowercase-hex>'，实际为 {digest!r}"
        )


def compute_package_integrity(
    package_root: str | Path,
    *,
    base_model_digest: str = "unverified_offline",
) -> dict[str, Any]:
    """扫描 ``package_root`` 下全部 ``.safetensors`` 文件，构造 Tier 1 ``integrity`` 字典。"""

    _validate_base_model_digest(base_model_digest)
    root = Path(package_root)
    if not root.is_dir():
        raise VaultError(f"计算包完整性失败，目录不存在：{root}")

    safetensor_files: list[tuple[str, Path]] = []
    for candidate in root.rglob("*.safetensors"):
        rel_parts = candidate.relative_to(root).parts
        if ".tmp" in rel_parts or ".history" in rel_parts:
            continue
        if candidate.is_file() and not candidate.is_symlink():
            rel_posix = candidate.relative_to(root).as_posix()
            safetensor_files.append((rel_posix, candidate))

    if not safetensor_files:
        raise VaultError(f"Ego_Package 下未找到任何 .safetensors 适配器文件：{root}")

    safetensor_files.sort(key=lambda pair: pair[0])
    cell_digests = {
        rel_posix: _sha256_file_prefixed(file_path)
        for rel_posix, file_path in safetensor_files
    }
    return {
        "schema_version": INTEGRITY_SCHEMA_VERSION,
        "base_model_digest": base_model_digest,
        "cell_digests": cell_digests,
    }


# ---------------------------------------------------------------------------
# 4. Tier 2 签名与按需子集验签
# ---------------------------------------------------------------------------


def _validate_integrity_block(integrity: Any) -> Mapping[str, str]:
    """校验 ``manifest.integrity`` 结构并返回 ``cell_digests`` 映射。"""

    if not isinstance(integrity, Mapping):
        raise VaultError("manifest 缺少合法的 integrity 对象")
    if integrity.get("schema_version") != INTEGRITY_SCHEMA_VERSION:
        raise VaultError(
            f"integrity.schema_version 非法：{integrity.get('schema_version')!r}"
        )
    _validate_base_model_digest(integrity.get("base_model_digest"))  # type: ignore[arg-type]

    cell_digests = integrity.get("cell_digests")
    if not isinstance(cell_digests, Mapping) or not cell_digests:
        raise VaultError("integrity.cell_digests 必须为非空对象")

    for rel_path, digest in cell_digests.items():
        if (
            not isinstance(rel_path, str)
            or not rel_path
            or rel_path.startswith("/")
            or "\\" in rel_path
            or ".." in Path(rel_path).parts
        ):
            raise VaultError(f"integrity.cell_digests 含非法相对路径键：{rel_path!r}")
        if not isinstance(digest, str) or not _SHA256_DIGEST_RE.match(digest):
            raise VaultError(
                f"integrity.cell_digests[{rel_path!r}] 摘要格式非法：{digest!r}"
            )
    return cell_digests


def sign_manifest(
    manifest: Mapping[str, Any],
    *,
    signer_fay_id: str,
    master_seed: bytes | str | None = None,
    subkey: bytes | str | None = None,
    key_id: str | None = None,
) -> dict[str, Any]:
    """对已含 ``integrity`` 的 ``manifest`` 计算 JCS + Ed25519 签名并返回附带 ``signature`` 的新字典。"""

    _require_canonical_ifay_id(signer_fay_id, label="signer_fay_id")
    if (master_seed is None) == (subkey is None):
        raise VaultError("sign_manifest 必须且只能提供 master_seed 或 subkey 之一")

    _validate_integrity_block(manifest.get("integrity"))

    priv_bytes = (
        derive_ego_signing_subkey(master_seed)
        if master_seed is not None
        else _coerce_32_bytes(subkey, label="subkey")  # type: ignore[arg-type]
    )

    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (  # noqa: PLC0415
            Ed25519PrivateKey,
        )
    except ImportError as exc:  # pragma: no cover
        raise VaultError(f"无法执行 Ed25519 签名：缺少 cryptography 依赖（{exc}）") from exc

    private_key = Ed25519PrivateKey.from_private_bytes(priv_bytes)
    public_key_hex = private_key.public_key().public_bytes_raw().hex()

    payload_obj: dict[str, Any] = {
        k: v for k, v in manifest.items() if k != "signature"
    }
    jcs_bytes = canonicalize_jcs(payload_obj)
    payload_jcs_sha256 = hashlib.sha256(jcs_bytes).hexdigest()

    signing_message = SIGNING_DOMAIN.encode("ascii") + b":" + jcs_bytes
    sig_hex = private_key.sign(signing_message).hex()

    resolved_key_id = (
        key_id if key_id is not None else f"{signer_fay_id}{KEY_ID_SUFFIX}"
    )

    signed = dict(payload_obj)
    signed["signature"] = {
        "schema_version": SIGNING_SCHEMA_VERSION,
        "alg": SIGNING_ALG,
        "domain": SIGNING_DOMAIN,
        "signer_fay_id": signer_fay_id,
        "key_id": resolved_key_id,
        "public_key_hex": public_key_hex,
        "payload_jcs_sha256": payload_jcs_sha256,
        "sig_hex": sig_hex,
    }
    return signed


def verify_package_signature(
    package_root: str | Path,
    trusted_public_key_hex: str | None = None,
    selected_cell_rel_paths: Sequence[str] | None = None,
) -> SignatureVerificationResult:
    """校验 ``Ego_Package`` 的 Ed25519 签名与（全部或按需选定的）Cell 适配器 SHA-256 摘要。"""

    root = Path(package_root)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise VaultError(f"未找到 manifest.json：{manifest_path}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise VaultError(f"读取或解析 manifest.json 失败：{manifest_path}；{exc}") from exc

    if not isinstance(manifest, Mapping):
        raise VaultError(f"manifest.json 顶层必须为 JSON 对象：{manifest_path}")

    sig_obj = manifest.get("signature")
    if sig_obj is None:
        _verify_optional_manifest_extensions(root, manifest)
        if trusted_public_key_hex is None:
            return SignatureVerificationResult(status="unsigned")
        raise VaultError(
            "Ego_Package 未包含 signature，但当前策略要求校验受信任公钥"
        )

    if not isinstance(sig_obj, Mapping):
        raise VaultError("manifest.signature 必须为 JSON 对象")
    if sig_obj.get("schema_version") != SIGNING_SCHEMA_VERSION:
        raise VaultError(
            f"signature.schema_version 非法：{sig_obj.get('schema_version')!r}"
        )
    if sig_obj.get("alg") != SIGNING_ALG:
        raise VaultError(f"signature.alg 不受支持：{sig_obj.get('alg')!r}")
    if sig_obj.get("domain") != SIGNING_DOMAIN:
        raise VaultError(f"signature.domain 不匹配：{sig_obj.get('domain')!r}")

    signer_fay_id = sig_obj.get("signer_fay_id")
    key_id = sig_obj.get("key_id")
    public_key_hex = sig_obj.get("public_key_hex")
    payload_jcs_sha256 = sig_obj.get("payload_jcs_sha256")
    sig_hex = sig_obj.get("sig_hex")

    _require_canonical_ifay_id(signer_fay_id, label="signature.signer_fay_id")
    if not isinstance(key_id, str) or not key_id:
        raise VaultError("signature.key_id 必须为非空字符串")
    if not isinstance(public_key_hex, str) or not _HEX_64_RE.match(public_key_hex):
        raise VaultError("signature.public_key_hex 必须为 64 位小写十六进制字符串")
    if not isinstance(payload_jcs_sha256, str) or not _HEX_64_RE.match(
        payload_jcs_sha256
    ):
        raise VaultError("signature.payload_jcs_sha256 必须为 64 位小写十六进制字符串")
    if not isinstance(sig_hex, str) or not _HEX_128_RE.match(sig_hex):
        raise VaultError("signature.sig_hex 必须为 128 位小写十六进制字符串")

    if trusted_public_key_hex is not None:
        normalized_trusted = trusted_public_key_hex.strip().lower()
        if not _HEX_64_RE.match(normalized_trusted):
            raise VaultError("trusted_public_key_hex 必须为 64 位十六进制字符串")
        if public_key_hex != normalized_trusted:
            raise VaultError(
                "Ego_Package 签名公钥与受信任公钥不匹配："
                f"package={public_key_hex} != trusted={normalized_trusted}"
            )

    payload_obj = {k: v for k, v in manifest.items() if k != "signature"}
    jcs_bytes = canonicalize_jcs(payload_obj)
    actual_jcs_sha256 = hashlib.sha256(jcs_bytes).hexdigest()
    if actual_jcs_sha256 != payload_jcs_sha256:
        raise VaultError(
            "manifest.json 内容已被篡改：payload_jcs_sha256 不匹配"
        )

    try:
        from cryptography.exceptions import InvalidSignature  # noqa: PLC0415
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (  # noqa: PLC0415
            Ed25519PublicKey,
        )
    except ImportError as exc:  # pragma: no cover
        raise VaultError(f"无法执行 Ed25519 验签：缺少 cryptography 依赖（{exc}）") from exc

    pub_key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex))
    signing_message = SIGNING_DOMAIN.encode("ascii") + b":" + jcs_bytes
    try:
        pub_key.verify(bytes.fromhex(sig_hex), signing_message)
    except InvalidSignature as exc:
        raise VaultError("Ego_Package Ed25519 签名校验失败（签名无效或内容被篡改）") from exc

    # Tier 1 适配器文件 SHA-256 校验（全量或按需子集）
    cell_digests = _validate_integrity_block(manifest.get("integrity"))

    if selected_cell_rel_paths is not None:
        normalized_targets = [
            Path(p).as_posix().lstrip("/") for p in selected_cell_rel_paths
        ]
        if not normalized_targets:
            raise VaultError("selected_cell_rel_paths 不能为空序列")
        for rel_posix in normalized_targets:
            if ".." in Path(rel_posix).parts:
                raise VaultError(f"非法的适配器相对路径：{rel_posix!r}")
            expected_digest = cell_digests.get(rel_posix)
            if expected_digest is None:
                raise VaultError(
                    f"选定的 Cell 未在 integrity.cell_digests 中受保护：{rel_posix}"
                )
            cell_file = root / rel_posix
            if not cell_file.is_file():
                raise VaultError(f"选定的已签名适配器文件缺失：{cell_file}")
            actual_digest = _sha256_file_prefixed(cell_file)
            if actual_digest != expected_digest:
                raise VaultError(
                    f"适配器文件已被篡改（SHA-256 不匹配）：{rel_posix}"
                )
        verified = tuple(normalized_targets)
    else:
        for rel_posix, expected_digest in cell_digests.items():
            cell_file = root / rel_posix
            if not cell_file.is_file():
                raise VaultError(f"integrity.cell_digests 所列适配器缺失：{cell_file}")
            actual_digest = _sha256_file_prefixed(cell_file)
            if actual_digest != expected_digest:
                raise VaultError(
                    f"适配器文件已被篡改（SHA-256 不匹配）：{rel_posix}"
                )
        # 全量校验时确保包内不存在未登记的孤儿 .safetensors 文件（忽略 .tmp 与 .history）
        on_disk = {
            p.relative_to(root).as_posix()
            for p in root.rglob("*.safetensors")
            if ".tmp" not in p.relative_to(root).parts
            and ".history" not in p.relative_to(root).parts
            and p.is_file()
        }
        unlisted = sorted(on_disk - set(cell_digests.keys()))
        if unlisted:
            raise VaultError(
                f"发现未纳入签名摘要清单的 .safetensors 文件：{unlisted}"
            )
        verified = tuple(cell_digests.keys())

    _verify_optional_manifest_extensions(root, manifest)

    return SignatureVerificationResult(
        status="verified",
        signer_fay_id=signer_fay_id,
        key_id=key_id,
        public_key_hex=public_key_hex,
        payload_jcs_sha256=actual_jcs_sha256,
        verified_cells=verified,
    )


def _verify_optional_manifest_extensions(
    root: Path, manifest: Mapping[str, Any]
) -> None:
    """校验可选的 ``slice_descriptor`` 协议字段与 ``cognitive_profile.json`` 摘要完整性。"""

    if "slice_descriptor" in manifest and manifest["slice_descriptor"] is not None:
        from ego_pod.vault.submodel_slice import validate_slice_descriptor

        validate_slice_descriptor(manifest["slice_descriptor"])

    expected_prof_sha = manifest.get("cognitive_profile_sha256")
    if expected_prof_sha is not None:
        if not isinstance(expected_prof_sha, str) or not _SHA256_DIGEST_RE.match(
            expected_prof_sha
        ):
            raise VaultError(
                f"manifest.cognitive_profile_sha256 格式非法：{expected_prof_sha!r}"
            )
        prof_path = root / "cognitive_profile.json"
        if not prof_path.is_file():
            raise VaultError(
                f"manifest 声明了 cognitive_profile_sha256，但文件缺失：{prof_path}"
            )
        actual_prof_sha = _sha256_file_prefixed(prof_path)
        if actual_prof_sha != expected_prof_sha:
            raise VaultError(
                "cognitive_profile.json 已被篡改（SHA-256 不匹配）"
            )
