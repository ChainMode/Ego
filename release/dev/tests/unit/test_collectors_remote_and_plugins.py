"""可插拔数据源、URL 断点续传采集器、S3/OSS 云采集器与增量采集缓存单测（ADR 0006 §5）。"""

from __future__ import annotations

import http.server
import io
import os
import socketserver
import threading
from pathlib import Path
from typing import Iterator

import pytest

from ego_pod.collector import pipeline as collector_pipeline
from ego_pod.collector.base import Category, CollectedFile
from ego_pod.collector.cloud import (
    ENV_CLOUD_ACCESS_KEY_ID,
    ENV_CLOUD_ENDPOINT_URL,
    ENV_CLOUD_SECRET_ACCESS_KEY,
    CloudSourceAdapter,
    _parse_list_objects_xml,
)
from ego_pod.collector.incremental import compile_sources_incrementally
from ego_pod.collector.plugins import (
    clear_collector_plugins,
    get_collector_plugin,
    is_registered_collector_plugin,
    register_collector_plugin,
    unregister_collector_plugin,
)
from ego_pod.collector.url import UrlSourceAdapter
from ego_pod.config.schema import (
    EditionDef,
    EgoConfig,
    SourceEntry,
    TrainingParams,
)
from ego_pod.exit_codes import CollectorError
from ego_pod.privacy.offline import NetworkBlockedError, OfflineMode


def _base_training() -> TrainingParams:
    return TrainingParams(
        method="qlora",
        quantization="4bit",
        lora_rank=8,
        lora_alpha=16,
        learning_rate=2e-4,
        epochs=1,
        batch_size=2,
    )


def test_url_collector_offline_guard_and_allowlist(tmp_path: Path) -> None:
    """URL 采集器在离线模式下立即以 NetworkBlockedError 阻断；非白名单主机亦被拒绝。"""
    adapter = UrlSourceAdapter(cwd=tmp_path, offline=OfflineMode(enabled=True))
    entry = SourceEntry(
        type="url",
        url="https://127.0.0.1:9999/notes.txt",
        allowed_hosts=("127.0.0.1",),
    )
    with pytest.raises(NetworkBlockedError):
        list(adapter.collect(entry, "thought"))

    online_adapter = UrlSourceAdapter(cwd=tmp_path, offline=OfflineMode(enabled=False))
    disallowed_entry = SourceEntry(
        type="url",
        url="https://evil.example.org/notes.txt",
        allowed_hosts=("trusted.example.org",),
    )
    with pytest.raises(NetworkBlockedError):
        list(online_adapter.collect(disallowed_entry, "thought"))


def test_url_collector_loopback_http_and_range_checkpoint_resume(
    tmp_path: Path,
) -> None:
    """URL 采集器支持 ETag / Range 断点续传（200 首采 -> 206 续传追加 -> 304 缓存命中）。"""
    payload_holder = {"body": b"Part1: initial thought record.\n", "etag": '"v1"', "requests": []}

    class _RangeHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            rng = self.headers.get("Range")
            inm = self.headers.get("If-None-Match")
            payload_holder["requests"].append((rng, inm))
            full_body: bytes = payload_holder["body"]  # type: ignore[assignment]
            cur_etag: str = payload_holder["etag"]  # type: ignore[assignment]

            if inm == cur_etag:
                self.send_response(304)
                self.end_headers()
                return

            if rng and rng.startswith("bytes="):
                start = int(rng.split("=", 1)[1].split("-", 1)[0])
                if 0 < start < len(full_body):
                    chunk = full_body[start:]
                    self.send_response(206)
                    self.send_header("ETag", cur_etag)
                    self.send_header(
                        "Content-Range", f"bytes {start}-{len(full_body)-1}/{len(full_body)}"
                    )
                    self.send_header("Content-Length", str(len(chunk)))
                    self.end_headers()
                    self.wfile.write(chunk)
                    return

            self.send_response(200)
            self.send_header("ETag", cur_etag)
            self.send_header("Content-Length", str(len(full_body)))
            self.end_headers()
            self.wfile.write(full_body)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            return

    with socketserver.TCPServer(("127.0.0.1", 0), _RangeHandler) as httpd:
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{port}/thoughts.txt"
            entry = SourceEntry(type="url", url=url, allowed_hosts=("127.0.0.1",))
            adapter = UrlSourceAdapter(cwd=tmp_path, offline=OfflineMode(enabled=False))

            # 第 1 次：200 全量拉取
            files_1 = list(adapter.collect(entry, "thought"))
            assert len(files_1) == 1
            assert Path(files_1[0].absolute_path).read_bytes() == payload_holder["body"]

            # 第 2 次：内容未变，ETag 命中 304 复用缓存
            files_2 = list(adapter.collect(entry, "thought"))
            assert len(files_2) == 1
            assert files_2[0].absolute_path == files_1[0].absolute_path

            # 第 3 次：远端文件增长且 ETag 升级，Range 请求触发 206 续传追加
            payload_holder["body"] = (
                b"Part1: initial thought record.\nPart2: resumed tail line.\n"
            )
            payload_holder["etag"] = '"v2"'
            files_3 = list(adapter.collect(entry, "thought"))
            assert Path(files_3[0].absolute_path).read_bytes() == payload_holder["body"]
        finally:
            httpd.shutdown()


def test_cloud_s3_collector_env_credentials_and_secret_scrubbing(
    tmp_path: Path,
) -> None:
    """S3/OSS 采集器仅从环境变量读取凭证、支持离线阻断，且绝不让云凭证落入采集样本。"""
    entry = SourceEntry(
        type="s3",
        bucket="ego-vault-bucket",
        prefix="skill/",
        allowed_hosts=("127.0.0.1",),
    )

    # 离线模式阻断
    offline_cloud = CloudSourceAdapter(
        cwd=tmp_path,
        offline=OfflineMode(enabled=True),
        env={
            ENV_CLOUD_ACCESS_KEY_ID: "<example-cloud-id>",
            ENV_CLOUD_SECRET_ACCESS_KEY: "<example-cloud-placeholder>",
        },
    )
    with pytest.raises(NetworkBlockedError):
        list(offline_cloud.collect(entry, "skill"))

    # 缺失环境变量时报错点名环境变量名
    missing_cred_cloud = CloudSourceAdapter(
        cwd=tmp_path,
        offline=OfflineMode(enabled=False),
        env={},
    )
    with pytest.raises(CollectorError) as exc:
        list(missing_cred_cloud.collect(entry, "skill"))
    assert ENV_CLOUD_ACCESS_KEY_ID in str(exc.value)

    # 模拟 S3 HTTP 服务端：验证 SigV4 请求头存在，且若远端文本意外混入凭证占位符会被自动擦除
    placeholder_cred = "example-placeholder-cloud-cred-98765"
    auth_headers: list[str] = []

    class _FakeS3Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            auth_headers.append(self.headers.get("Authorization", ""))
            if "list-type=2" in self.path:
                xml_body = (
                    '<?xml version="1.0" encoding="UTF-8"?>'
                    "<ListBucketResult>"
                    "<Contents><Key>skill/python_tips.txt</Key><ETag>\"etag1\"</ETag></Contents>"
                    "</ListBucketResult>"
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/xml")
                self.send_header("Content-Length", str(len(xml_body)))
                self.end_headers()
                self.wfile.write(xml_body)
            else:
                obj_body = (
                    f"编写模块化函数记录。意外混入环境变量：{placeholder_cred} 结束。".encode(
                        "utf-8"
                    )
                )
                self.send_response(200)
                self.send_header("ETag", '"etag1"')
                self.send_header("Content-Length", str(len(obj_body)))
                self.end_headers()
                self.wfile.write(obj_body)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            return

    with socketserver.TCPServer(("127.0.0.1", 0), _FakeS3Handler) as httpd:
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            cloud = CloudSourceAdapter(
                cwd=tmp_path,
                offline=OfflineMode(enabled=False),
                env={
                    ENV_CLOUD_ACCESS_KEY_ID: "example-placeholder-id-12345",
                    ENV_CLOUD_SECRET_ACCESS_KEY: placeholder_cred,
                    ENV_CLOUD_ENDPOINT_URL: f"http://127.0.0.1:{port}",
                },
            )
            collected = list(cloud.collect(entry, "skill"))
            assert len(collected) == 1
            assert any("AWS4-HMAC-SHA256" in h for h in auth_headers)
            saved_text = Path(collected[0].absolute_path).read_text(encoding="utf-8")
            assert placeholder_cred not in saved_text
            assert "example-placeholder-id-12345" not in saved_text
        finally:
            httpd.shutdown()


def test_plugin_registry_validates_interface_and_dispatches_in_pipeline(
    tmp_path: Path,
) -> None:
    """插件注册表对缺失 collect 方法的对象前置报错，合法插件可通过 collector.pipeline 调度。"""
    clear_collector_plugins()
    try:
        class _BrokenPlugin:
            pass

        with pytest.raises(CollectorError) as exc:
            register_collector_plugin("broken-src", _BrokenPlugin)
        assert "collect" in str(exc.value)

        note_file = tmp_path / "plugin_note.txt"
        note_file.write_text("来自自定义插件的数据源笔记记录。", encoding="utf-8")

        class _MemoryNotePlugin:
            def collect(
                self, entry: SourceEntry, category: Category
            ) -> Iterator[CollectedFile]:
                yield CollectedFile(
                    absolute_path=str(note_file.resolve()),
                    size_bytes=note_file.stat().st_size,
                    category=category,
                    read_at="2026-11-18T00:00:00Z",
                )

        register_collector_plugin("custom-notes", _MemoryNotePlugin())

        cfg = EgoConfig(
            sources={
                "sense": SourceEntry(type="custom-notes"),
                "action": SourceEntry(type="local", path=str(tmp_path / "missing_a")),
                "thought": SourceEntry(type="local", path=str(tmp_path / "missing_t")),
                "skill": SourceEntry(type="local", path=str(tmp_path / "missing_s")),
                "credentials": SourceEntry(type="local", path=str(tmp_path / "missing_c")),
            },
            training=_base_training(),
            editions={"normal": EditionDef(base="qwen3.5-0.8B")},
            output={},
        )
        res = collector_pipeline.run(cfg, cwd=tmp_path)
        assert len(res.files) == 1
        assert res.summary["sense"][0] == 1
    finally:
        clear_collector_plugins()


def test_incremental_collection_manifest_reuses_unchanged_and_tracks_diffs(
    tmp_path: Path,
) -> None:
    """增量采集清单基于 SHA-256 复用未变更记录，并准确统计新增/变更/移除文件数。"""
    f1 = tmp_path / "s1.txt"
    f2 = tmp_path / "s2.txt"
    f1.write_text("第一条不变的思想随笔记录内容。", encoding="utf-8")
    f2.write_text("第二条稍后会被修改的技能代码笔记。", encoding="utf-8")

    cf1 = CollectedFile(str(f1), f1.stat().st_size, "thought", "2026-11-18T00:00:00Z")
    cf2 = CollectedFile(str(f2), f2.stat().st_size, "skill", "2026-11-18T00:00:00Z")

    plan_init = compile_sources_incrementally([cf1, cf2], cwd=tmp_path, update=False)
    assert plan_init.changed_or_new_count == 2
    assert plan_init.unchanged_count == 0
    assert (tmp_path / ".ego" / "collection-manifest.json").is_file()

    # 修改 f2，保持 f1 不变，新增 f3
    f2.write_text("第二条已被更新的技能代码笔记 sk-abcdef1234567890abcdef1234。", encoding="utf-8")
    f3 = tmp_path / "s3.txt"
    f3.write_text("第三条全新的日常行动记录。", encoding="utf-8")
    cf2_mod = CollectedFile(str(f2), f2.stat().st_size, "skill", "2026-11-18T00:01:00Z")
    cf3_new = CollectedFile(str(f3), f3.stat().st_size, "action", "2026-11-18T00:01:00Z")

    plan_upd = compile_sources_incrementally(
        [cf1, cf2_mod, cf3_new], cwd=tmp_path, update=True
    )
    assert plan_upd.unchanged_count == 1
    assert plan_upd.changed_or_new_count == 2
    assert plan_upd.removed_count == 0
    assert len(plan_upd.source_samples) == 3
    assert plan_upd.redact_counters.total >= 1

    # 删除 f3 后再次 --update：removed_count == 1，unchanged_count == 2
    plan_del = compile_sources_incrementally([cf1, cf2_mod], cwd=tmp_path, update=True)
    assert plan_del.unchanged_count == 2
    assert plan_del.changed_or_new_count == 0
    assert plan_del.removed_count == 1


@pytest.mark.parametrize(
    "bad_url",
    [
        "",
        "http://example.com/insecure.txt",
        "ftp://files.example.com/archive.txt",
    ],
)
def test_url_collector_rejects_empty_or_insecure_non_loopback_urls(
    tmp_path: Path, bad_url: str
) -> None:
    adapter = UrlSourceAdapter(cwd=tmp_path, offline=OfflineMode(enabled=False))
    entry = SourceEntry(type="url", url=bad_url, allowed_hosts=("example.com", "files.example.com"))
    with pytest.raises(CollectorError):
        list(adapter.collect(entry, "sense"))


def test_cloud_xml_parser_skips_folders_and_rejects_malformed_xml() -> None:
    xml_ok = (
        b"<ListBucketResult>"
        b"<Contents><Key>prefix/</Key><ETag>\"dir\"</ETag></Contents>"
        b"<Contents><Key>prefix/b.txt</Key><ETag>\"e2\"</ETag></Contents>"
        b"<Contents><Key>prefix/a.txt</Key><ETag>\"e1\"</ETag></Contents>"
        b"</ListBucketResult>"
    )
    assert _parse_list_objects_xml(xml_ok) == [
        ("prefix/a.txt", '"e1"'),
        ("prefix/b.txt", '"e2"'),
    ]

    with pytest.raises(CollectorError):
        _parse_list_objects_xml(b"<ListBucketResult><unclosed>")


def test_cloud_oss_uri_and_etag_cache_hit(tmp_path: Path) -> None:
    """OSS URI (oss://bucket/prefix) 解析与二次采集 ETag 命中免重复下载验证。"""
    get_object_calls = 0

    class _OssHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            nonlocal get_object_calls
            if "list-type=2" in self.path:
                body = (
                    b"<ListBucketResult>"
                    b"<Contents><Key>notes/n1.txt</Key><ETag>\"v1\"</ETag></Contents>"
                    b"</ListBucketResult>"
                )
            else:
                get_object_calls += 1
                body = "来自 OSS 的笔记内容记录。".encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            return

    with socketserver.TCPServer(("127.0.0.1", 0), _OssHandler) as httpd:
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            adapter = CloudSourceAdapter(
                cwd=tmp_path,
                offline=OfflineMode(enabled=False),
                env={
                    ENV_CLOUD_ACCESS_KEY_ID: "example-oss-placeholder-id",
                    ENV_CLOUD_SECRET_ACCESS_KEY: "example-oss-placeholder-cred",
                    ENV_CLOUD_ENDPOINT_URL: f"http://127.0.0.1:{port}",
                },
            )
            entry = SourceEntry(type="oss", url="oss://my-oss-bucket/notes/")
            first = list(adapter.collect(entry, "thought"))
            second = list(adapter.collect(entry, "thought"))
            assert len(first) == len(second) == 1
            assert get_object_calls == 1  # 第二次命中 ETag 直接复用缓存
        finally:
            httpd.shutdown()


def test_entry_point_plugin_discovery_and_unregister(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """验证 entry_points 发现与缺失 collect 方法插件的加载期拦截。"""
    clear_collector_plugins()

    class _FakeEntryPoint:
        def __init__(self, name: str, obj: object) -> None:
            self.name = name
            self._obj = obj

        def load(self) -> object:
            return self._obj

    class _ValidEpAdapter:
        def collect(self, entry: SourceEntry, category: Category) -> Iterator[CollectedFile]:
            return iter(())

    class _InvalidEpAdapter:
        pass

    monkeypatch.setattr(
        "importlib.metadata.entry_points",
        lambda group=None: [
            _FakeEntryPoint("ep-valid", _ValidEpAdapter),
            _FakeEntryPoint("ep-broken", _InvalidEpAdapter),
        ],
    )

    assert is_registered_collector_plugin("ep-valid") is True
    loaded = get_collector_plugin("ep-valid")
    assert loaded is not None
    assert list(loaded.collect(SourceEntry(type="ep-valid"), "sense")) == []

    with pytest.raises(CollectorError) as exc:
        get_collector_plugin("ep-broken")
    assert "collect" in str(exc.value)

    register_collector_plugin("temp-plug", _ValidEpAdapter)
    unregister_collector_plugin("temp-plug")
    assert get_collector_plugin("unknown-plug") is None
    clear_collector_plugins()
