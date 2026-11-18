"""严格基于环境变量凭证与 AWS SigV4 签名的 S3 / OSS 云存储数据源采集器（ADR 0006 §5）。

安全不变量：
1. 仅从环境变量 ``EGO_CLOUD_ACCESS_KEY_ID`` / ``EGO_CLOUD_SECRET_ACCESS_KEY`` /
   ``EGO_CLOUD_ENDPOINT_URL`` 读取云存储访问凭证；
2. 离线模式（``--offline`` 或 ``EGO_OFFLINE=1``）启用时立即抛 :class:`NetworkBlockedError`；
3. 对下载落盘到 ``.ego/remote_sources/`` 的文本执行主动环境变量密钥擦除，杜绝云端凭证
   进入采集文件或下游编译语料。
"""

from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import os
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen

from ego_pod.collector.base import Category, CollectedFile, SourceAdapter
from ego_pod.collector.url import (
    CHECKPOINTS_FILENAME,
    _load_checkpoints,
    _save_checkpoints,
    _utc_now_iso,
)
from ego_pod.exit_codes import CollectorError
from ego_pod.privacy.offline import (
    OfflineMode,
    assert_request_allowed,
    get_active_offline_mode,
)

if TYPE_CHECKING:
    from ego_pod.config.schema import SourceEntry

__all__ = [
    "ENV_CLOUD_ACCESS_KEY_ID",
    "ENV_CLOUD_SECRET_ACCESS_KEY",
    "ENV_CLOUD_ENDPOINT_URL",
    "CloudSourceAdapter",
]

ENV_CLOUD_ACCESS_KEY_ID = "_".join(("EGO", "CLOUD", "ACCESS", "KEY", "ID"))
ENV_CLOUD_SECRET_ACCESS_KEY = "_".join(("EGO", "CLOUD", "SECRET", "ACCESS", "KEY"))
ENV_CLOUD_ENDPOINT_URL = "_".join(("EGO", "CLOUD", "ENDPOINT", "URL"))

_EMPTY_PAYLOAD_SHA256 = hashlib.sha256(b"").hexdigest()


def _sign_hmac(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def _build_sigv4_headers(
    *,
    method: str,
    url: str,
    access_key: str,
    secret_key: str,
    region: str = "us-east-1",
    service: str = "s3",
) -> dict[str, str]:
    """使用 Python 标准库构造 S3 / S3-compatible OSS 的 AWS4-HMAC-SHA256 请求头。"""

    now = dt.datetime.now(dt.timezone.utc)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = now.strftime("%Y%m%d")

    parsed = urlparse(url)
    host = parsed.netloc
    canonical_uri = parsed.path or "/"
    canonical_querystring = parsed.query

    canonical_headers = (
        f"host:{host}\n"
        f"x-amz-content-sha256:{_EMPTY_PAYLOAD_SHA256}\n"
        f"x-amz-date:{amz_date}\n"
    )
    signed_headers = "host;x-amz-content-sha256;x-amz-date"
    canonical_request = "\n".join(
        [
            method,
            canonical_uri,
            canonical_querystring,
            canonical_headers,
            signed_headers,
            _EMPTY_PAYLOAD_SHA256,
        ]
    )
    credential_scope = f"{date_stamp}/{region}/{service}/aws4_request"
    string_to_sign = "\n".join(
        [
            "AWS4-HMAC-SHA256",
            amz_date,
            credential_scope,
            hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
        ]
    )

    k_date = _sign_hmac(("AWS4" + secret_key).encode("utf-8"), date_stamp)
    k_region = _sign_hmac(k_date, region)
    k_service = _sign_hmac(k_region, service)
    k_signing = _sign_hmac(k_service, "aws4_request")
    signature = hmac.new(
        k_signing, string_to_sign.encode("utf-8"), hashlib.sha256
    ).hexdigest()

    auth_header = (
        f"AWS4-HMAC-SHA256 Credential={access_key}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )
    return {
        "Host": host,
        "x-amz-date": amz_date,
        "x-amz-content-sha256": _EMPTY_PAYLOAD_SHA256,
        "Authorization": auth_header,
    }


def _parse_list_objects_xml(xml_bytes: bytes) -> list[tuple[str, str]]:
    """解析 S3 ListObjectsV2 XML 响应，返回 ``[(key, etag), ...]``。"""

    try:
        root = ET.fromstring(xml_bytes)  # noqa: S314 - 受控端点返回的清单元数据
    except ET.ParseError as exc:
        raise CollectorError(f"[COLLECTOR] 云存储对象列表 XML 解析失败：{exc}") from exc

    items: list[tuple[str, str]] = []
    for elem in root.iter():
        if elem.tag.endswith("Contents"):
            key_text = ""
            etag_text = ""
            for child in elem:
                if child.tag.endswith("Key") and child.text:
                    key_text = child.text
                elif child.tag.endswith("ETag") and child.text:
                    etag_text = child.text
            if key_text and not key_text.endswith("/"):
                items.append((key_text, etag_text))
    items.sort(key=lambda pair: pair[0])
    return items


def _scrub_cloud_secrets(raw_bytes: bytes, secrets: tuple[str, ...]) -> bytes:
    """确保任何意外出现在下载内容中的环境变量云密钥被强制抹除。"""

    for secret in secrets:
        if secret and len(secret) >= 4:
            sec_bytes = secret.encode("utf-8")
            if sec_bytes in raw_bytes:
                raw_bytes = raw_bytes.replace(sec_bytes, b"[REDACTED_CLOUD_CREDENTIAL]")
    return raw_bytes


class CloudSourceAdapter(SourceAdapter):
    """兼容 S3 与阿里云 OSS（S3 兼容端点）的受控云端数据源适配器。"""

    def __init__(
        self,
        *,
        cwd: Path | None = None,
        offline: OfflineMode | None = None,
        env: Mapping[str, str] | None = None,
        opener: Callable[[Request], Any] | None = None,
    ) -> None:
        self._cwd = Path.cwd() if cwd is None else Path(cwd)
        self._offline = offline
        self._env = env
        self._opener = opener

    def _resolve_bucket_and_prefix(self, entry: "SourceEntry") -> tuple[str, str]:
        if entry.bucket and entry.bucket.strip():
            return entry.bucket.strip(), (entry.prefix or "").lstrip("/")
        raw_uri = (entry.url or "").strip()
        if raw_uri.startswith(("s3://", "oss://")):
            parsed = urlparse(raw_uri)
            return parsed.netloc, parsed.path.lstrip("/")
        raise CollectorError(
            f"[COLLECTOR] 云存储数据源 (type={entry.type!r}) 缺少有效 bucket"
        )

    def collect(
        self, entry: "SourceEntry", category: Category
    ) -> Iterator[CollectedFile]:
        bucket, prefix = self._resolve_bucket_and_prefix(entry)
        env_map = os.environ if self._env is None else self._env

        endpoint = (env_map.get(ENV_CLOUD_ENDPOINT_URL) or "").strip().rstrip("/")
        if not endpoint:
            default_domain = (
                "oss-cn-hangzhou.aliyuncs.com"
                if entry.type == "oss"
                else "s3.amazonaws.com"
            )
            endpoint = f"https://{bucket}.{default_domain}"
            base_object_prefix = f"{endpoint}"
        else:
            base_object_prefix = f"{endpoint}/{quote(bucket)}"

        endpoint_host = (urlparse(endpoint).hostname or "").lower()
        active_offline = (
            self._offline if self._offline is not None else get_active_offline_mode()
        )
        if active_offline is not None and active_offline.enabled:
            from ego_pod.privacy.offline import (
                NETWORK_OFFLINE_BLOCKED_MESSAGE,
                NetworkBlockedError,
            )

            raise NetworkBlockedError(NETWORK_OFFLINE_BLOCKED_MESSAGE)

        # 严格仅从环境变量读取云凭证
        access_key = (env_map.get(ENV_CLOUD_ACCESS_KEY_ID) or "").strip()
        secret_key = (env_map.get(ENV_CLOUD_SECRET_ACCESS_KEY) or "").strip()
        if not access_key or not secret_key:
            raise CollectorError(
                f"[COLLECTOR] 云存储数据源 (type={entry.type!r}) 缺少环境变量凭证："
                f"请设置 {ENV_CLOUD_ACCESS_KEY_ID} 与 {ENV_CLOUD_SECRET_ACCESS_KEY}"
            )

        effective_hosts = list(entry.allowed_hosts)
        if not effective_hosts:
            effective_hosts.extend(["amazonaws.com", "aliyuncs.com"])
        if endpoint_host and endpoint_host in ("127.0.0.1", "localhost", "::1"):
            effective_hosts.append(endpoint_host)

        assert_request_allowed(
            endpoint,
            offline=active_offline,
            extra_allowed_hosts=effective_hosts,
        )

        query_params = [("list-type", "2")]
        if prefix:
            query_params.append(("prefix", prefix))
        list_url = f"{base_object_prefix}?{urlencode(sorted(query_params))}"

        open_fn = self._opener if self._opener is not None else urlopen
        list_headers = _build_sigv4_headers(
            method="GET",
            url=list_url,
            access_key=access_key,
            secret_key=secret_key,
        )
        try:
            resp = open_fn(Request(list_url, headers=list_headers, method="GET"))  # noqa: S310
            xml_bytes = resp.read()
            if callable(getattr(resp, "close", None)):
                resp.close()
        except (HTTPError, URLError, OSError) as exc:
            raise CollectorError(
                f"[COLLECTOR] 列举云存储对象失败（bucket={bucket!r}）：{exc}"
            ) from exc

        objects = _parse_list_objects_xml(xml_bytes)
        ego_dir = self._cwd / ".ego"
        ckpt_path = ego_dir / CHECKPOINTS_FILENAME
        checkpoints = _load_checkpoints(ckpt_path)

        for obj_key, etag in objects:
            ckpt_key = f"{entry.type}:{bucket}:{obj_key}"
            key_hash = hashlib.sha256(obj_key.encode("utf-8")).hexdigest()[:16]
            safe_name = Path(obj_key).name or "object.dat"
            cache_file = (
                ego_dir
                / "remote_sources"
                / entry.type
                / category
                / f"{key_hash}_{safe_name}"
            )
            cache_file.parent.mkdir(parents=True, exist_ok=True)

            prev = checkpoints.get(ckpt_key, {})
            if (
                isinstance(prev, dict)
                and etag
                and prev.get("etag") == etag
                and cache_file.is_file()
            ):
                size_bytes = cache_file.stat().st_size
                yield CollectedFile(
                    absolute_path=str(cache_file.resolve()),
                    size_bytes=size_bytes,
                    category=category,
                    read_at=_utc_now_iso(),
                )
                continue

            encoded_key = "/".join(quote(seg, safe="") for seg in obj_key.split("/"))
            obj_url = f"{base_object_prefix}/{encoded_key}"
            obj_headers = _build_sigv4_headers(
                method="GET",
                url=obj_url,
                access_key=access_key,
                secret_key=secret_key,
            )
            try:
                obj_resp = open_fn(Request(obj_url, headers=obj_headers, method="GET"))  # noqa: S310
                raw_data = obj_resp.read()
                if callable(getattr(obj_resp, "close", None)):
                    obj_resp.close()
            except (HTTPError, URLError, OSError) as exc:
                raise CollectorError(
                    f"[COLLECTOR] 下载云存储对象失败（{bucket}/{obj_key}）：{exc}"
                ) from exc

            scrubbed = _scrub_cloud_secrets(raw_data, (secret_key, access_key))
            cache_file.write_bytes(scrubbed)
            size_bytes = cache_file.stat().st_size
            checkpoints[ckpt_key] = {
                "etag": etag,
                "offset": size_bytes,
                "cached_path": str(cache_file.resolve()),
                "updated_at": _utc_now_iso(),
            }
            _save_checkpoints(ckpt_path, checkpoints)

            yield CollectedFile(
                absolute_path=str(cache_file.resolve()),
                size_bytes=size_bytes,
                category=category,
                read_at=_utc_now_iso(),
            )
