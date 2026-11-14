"""`ego_pod.privacy.offline` 网络守卫的属性测试（hypothesis）。

对应设计文档 "Correctness Properties" 章节的 **Property 15: privacy 离线无外发**，
验证需求 R9（隐私优先与离线运行）的核心不变量：

- **离线模式（R9-AC6）**：``--offline`` / ``EGO_OFFLINE=1`` 启用时，对任意主机
  （即便是白名单内的 ``huggingface.co`` / ``hf.co`` 及其子域名）的任何外发请求都
  被守卫拒绝并抛 :class:`NetworkBlockedError`。
- **在线模式（R9-AC5）**：非白名单主机在任何模式下都被拒绝；白名单主机
  （``huggingface.co`` / ``hf.co`` 及其子域名）通过守卫，委托底层 requester 发起
  请求。

策略层面用 hypothesis 生成任意 scheme / 子域名 / 路径 / 查询参数组合的 URL，
分别覆盖"白名单主机"与"非白名单主机"两类输入空间。所有请求都通过注入的
``requester`` 桩完成，守卫拒绝时 requester 不会被调用，因此不触达真实网络。

**Validates: Requirements 9.1, 9.3, 9.4, 9.5, 9.6, 9.7**
"""

from __future__ import annotations

import string

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ego_pod.privacy.offline import (
    ALLOWED_HOSTS,
    NetworkBlockedError,
    OfflineMode,
    assert_request_allowed,
    guarded_get,
    guarded_post,
    guarded_urlopen,
    is_allowed_host,
    set_active_offline_mode,
)

# ---------------------------------------------------------------------------
# fixture：隔离进程内"活动离线模式"全局状态
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_active_offline_mode():
    """每个用例前后把活动离线模式重置为未初始化，避免跨用例污染。

    本文件所有断言都显式传入 ``offline=`` 覆盖离线状态，不依赖全局活动模式；
    重置仅为防御性隔离。
    """
    set_active_offline_mode(None)
    yield
    set_active_offline_mode(None)


# ---------------------------------------------------------------------------
# 公共策略：scheme / 路径 / 域名标签
# ---------------------------------------------------------------------------

#: 仅针对 HTTP/HTTPS 外发请求施加守卫（R9 网络外发约束）。
schemes = st.sampled_from(["http", "https"])

#: URL 路径与查询片段，覆盖空路径、根路径、带查询参数等情形。
paths = st.sampled_from(
    ["", "/", "/models/foo", "/a/b/c", "/resolve/main/model.safetensors", "/x?y=1&z=2"]
)

#: 单个域名标签字符集 [a-z0-9-]。
_LABEL_ALPHABET = string.ascii_lowercase + string.digits + "-"

#: 单个域名标签（1–12 字符）。
domain_label = st.text(alphabet=_LABEL_ALPHABET, min_size=1, max_size=12)


# ---------------------------------------------------------------------------
# 白名单主机策略：huggingface.co / hf.co 及其任意层级子域名
# ---------------------------------------------------------------------------

allowlisted_host = st.builds(
    lambda labels, base: ".".join([*labels, base]),
    st.lists(domain_label, min_size=0, max_size=3),
    st.sampled_from(ALLOWED_HOSTS),
)


# ---------------------------------------------------------------------------
# 非白名单主机策略：任意域名，过滤掉恰好落入白名单的情形
# ---------------------------------------------------------------------------

non_allowlisted_host = (
    st.lists(domain_label, min_size=1, max_size=4)
    .map(lambda labels: ".".join(labels))
    .filter(lambda host: not is_allowed_host(host))
)


def _build_url(scheme: str, host: str, path: str) -> str:
    """由 scheme/host/path 片段拼出完整 URL。"""
    return f"{scheme}://{host}{path}"


# 任意主机（白名单或非白名单），用于"离线模式拒绝一切"的全覆盖断言。
any_host = st.one_of(allowlisted_host, non_allowlisted_host)


# ---------------------------------------------------------------------------
# Property 15 (a): 离线模式下任何外发请求都被拒绝（含白名单主机）
# ---------------------------------------------------------------------------


@settings(max_examples=300, deadline=None)
@given(scheme=schemes, host=any_host, path=paths)
def test_offline_blocks_every_request(scheme: str, host: str, path: str) -> None:
    """离线模式（R9-AC6）：对任意主机的请求都抛 ``NetworkBlockedError``。

    即便目标主机在白名单内，离线模式也一律拒绝；守卫在委托底层 requester 前抛错，
    因此不触达真实网络。

    **Validates: Requirements 9.1, 9.3, 9.4, 9.5, 9.6, 9.7**
    """
    offline = OfflineMode(enabled=True)
    url = _build_url(scheme, host, path)

    # 直接守卫：无条件拒绝。
    with pytest.raises(NetworkBlockedError):
        assert_request_allowed(url, offline=offline)

    # 经 guarded_* 包装：requester 绝不应被调用（守卫先抛错）。
    sentinel_calls: list[str] = []

    def _requester(called_url: str, *args, **kwargs):
        sentinel_calls.append(called_url)
        return "SHOULD_NOT_HAPPEN"

    for guard in (guarded_get, guarded_post, guarded_urlopen):
        with pytest.raises(NetworkBlockedError):
            guard(url, offline=offline, requester=_requester)
    assert sentinel_calls == [], "离线模式下 requester 不应被调用"


# ---------------------------------------------------------------------------
# Property 15 (b): 在线模式下非白名单主机被拒绝
# ---------------------------------------------------------------------------


@settings(max_examples=300, deadline=None)
@given(scheme=schemes, host=non_allowlisted_host, path=paths)
def test_online_blocks_non_allowlisted_host(
    scheme: str, host: str, path: str
) -> None:
    """在线模式（R9-AC5）：非白名单主机被拒绝，requester 不被调用。

    **Validates: Requirements 9.1, 9.3, 9.4, 9.5, 9.6, 9.7**
    """
    online = OfflineMode(enabled=False)
    url = _build_url(scheme, host, path)

    with pytest.raises(NetworkBlockedError):
        assert_request_allowed(url, offline=online)

    sentinel_calls: list[str] = []

    def _requester(called_url: str, *args, **kwargs):
        sentinel_calls.append(called_url)
        return "SHOULD_NOT_HAPPEN"

    for guard in (guarded_get, guarded_post, guarded_urlopen):
        with pytest.raises(NetworkBlockedError):
            guard(url, offline=online, requester=_requester)
    assert sentinel_calls == [], "非白名单主机的 requester 不应被调用"


# ---------------------------------------------------------------------------
# Property 15 (c): 在线模式下白名单主机通过守卫并委托 requester
# ---------------------------------------------------------------------------


@settings(max_examples=300, deadline=None)
@given(scheme=schemes, host=allowlisted_host, path=paths)
def test_online_allows_allowlisted_host(
    scheme: str, host: str, path: str
) -> None:
    """在线模式（R9-AC5）：白名单主机通过守卫，委托注入的 requester 发起请求。

    通过注入 requester 桩验证守卫放行后确实把调用转交给底层，且不触达真实网络。

    **Validates: Requirements 9.1, 9.3, 9.4, 9.5, 9.6, 9.7**
    """
    online = OfflineMode(enabled=False)
    url = _build_url(scheme, host, path)

    # 守卫本身应静默通过（返回 None，不抛异常）。
    assert assert_request_allowed(url, offline=online) is None

    # guarded_get / guarded_post：requester 被调用且 URL 透传、返回值回传。
    for guard, tag in ((guarded_get, "GET"), (guarded_post, "POST")):
        recorded: list[str] = []

        def _requester(called_url: str, *args, _tag=tag, **kwargs):
            recorded.append(called_url)
            return _tag

        result = guard(url, offline=online, requester=_requester)
        assert result == tag
        assert recorded == [url]

    # guarded_urlopen：同样委托注入的 requester。
    urlopen_calls: list[str] = []

    def _urlopen_requester(called_url: str, *args, **kwargs):
        urlopen_calls.append(called_url)
        return "OPENED"

    assert (
        guarded_urlopen(url, offline=online, requester=_urlopen_requester)
        == "OPENED"
    )
    assert urlopen_calls == [url]
