"""受离线守卫与主机白名单保护、支持 ETag / Range 断点续传的 URL 数据源采集器（ADR 0006 §5）。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from ego_pod.collector.base import Category, CollectedFile, SourceAdapter
from ego_pod.exit_codes import CollectorError
from ego_pod.privacy.offline import (
    OfflineMode,
    assert_request_allowed,
    get_active_offline_mode,
)

if TYPE_CHECKING:
    from ego_pod.config.schema import SourceEntry

__all__ = ["CHECKPOINTS_FILENAME", "UrlSourceAdapter"]

CHECKPOINTS_FILENAME = "collector-checkpoints.json"
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .strftime("%Y-%m-%dT%H:%M:%SZ")
    )


def _load_checkpoints(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return dict(obj) if isinstance(obj, dict) else {}
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}


def _save_checkpoints(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


class UrlSourceAdapter(SourceAdapter):
    """支持 HTTPS（及本地测试环回 HTTP）拉取与断点续传的远程 URL 适配器。"""

    def __init__(
        self,
        *,
        cwd: Path | None = None,
        offline: OfflineMode | None = None,
        opener: Callable[[Request], Any] | None = None,
    ) -> None:
        self._cwd = Path.cwd() if cwd is None else Path(cwd)
        self._offline = offline
        self._opener = opener

    def collect(
        self, entry: "SourceEntry", category: Category
    ) -> Iterator[CollectedFile]:
        raw_url = (entry.url or "").strip()
        if not raw_url:
            raise CollectorError(
                f"[COLLECTOR] Data_Category {category} 的 url 数据源缺少有效 url"
            )

        parsed = urlparse(raw_url)
        scheme = (parsed.scheme or "").lower()
        host = (parsed.hostname or "").lower()
        if scheme != "https" and not (scheme == "http" and host in _LOOPBACK_HOSTS):
            raise CollectorError(
                f"[COLLECTOR] URL 数据源必须使用 https:// 或本地环回 http://：{raw_url}"
            )

        active_offline = (
            self._offline if self._offline is not None else get_active_offline_mode()
        )
        assert_request_allowed(
            raw_url,
            offline=active_offline,
            extra_allowed_hosts=entry.allowed_hosts,
        )

        ego_dir = self._cwd / ".ego"
        ckpt_path = ego_dir / CHECKPOINTS_FILENAME
        checkpoints = _load_checkpoints(ckpt_path)
        ckpt_key = f"url:{category}:{raw_url}"
        prev = checkpoints.get(ckpt_key, {}) if isinstance(checkpoints.get(ckpt_key), dict) else {}

        url_hash = hashlib.sha256(raw_url.encode("utf-8")).hexdigest()[:16]
        cache_file = ego_dir / "remote_sources" / "url" / category / f"{url_hash}.dat"
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        headers: dict[str, str] = {"User-Agent": "ifay-ego-collector/0.2"}
        prev_etag = prev.get("etag")
        prev_offset = int(prev.get("offset", 0)) if cache_file.is_file() else 0

        if isinstance(prev_etag, str) and prev_etag and cache_file.is_file():
            headers["If-None-Match"] = prev_etag
        if prev_offset > 0 and cache_file.is_file():
            headers["Range"] = f"bytes={prev_offset}-"

        req = Request(raw_url, headers=headers, method="GET")  # noqa: S310
        open_fn = self._opener if self._opener is not None else urlopen

        try:
            resp = open_fn(req)
            status = getattr(resp, "status", None) or resp.getcode()
            resp_headers = getattr(resp, "headers", {}) or {}
            new_etag = resp_headers.get("ETag") or prev_etag
            body = resp.read()
            close_fn = getattr(resp, "close", None)
            if callable(close_fn):
                close_fn()

            if status == 206 and prev_offset > 0 and cache_file.is_file():
                with cache_file.open("ab") as fh:
                    fh.write(body)
            elif status == 304 and cache_file.is_file():
                pass
            else:
                cache_file.write_bytes(body)
        except HTTPError as http_err:
            if http_err.code == 304 and cache_file.is_file():
                new_etag = http_err.headers.get("ETag") if http_err.headers else prev_etag
            elif http_err.code == 416 and cache_file.is_file():
                # Range 已在文件末尾（无新内容）
                new_etag = http_err.headers.get("ETag") if http_err.headers else prev_etag
            else:
                raise CollectorError(
                    f"[COLLECTOR] 拉取 URL 数据源失败（HTTP {http_err.code}）：{raw_url}"
                ) from http_err
        except URLError as url_err:
            raise CollectorError(
                f"[COLLECTOR] 连接 URL 数据源失败：{raw_url}（{url_err.reason}）"
            ) from url_err
        except OSError as os_err:
            raise CollectorError(
                f"[COLLECTOR] 读取或缓存 URL 数据源失败：{raw_url}（{os_err}）"
            ) from os_err

        if not cache_file.is_file():
            return

        size_bytes = cache_file.stat().st_size
        checkpoints[ckpt_key] = {
            "etag": new_etag,
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
