"""Privacy 模块 — 隐私优先与离线运行。

实现 Requirement 9 的离线模式开关与网络外发拦截。本模块导出：

- 离线模式开关相关能力（任务 3.1）：``OfflineMode`` / ``init_offline_mode`` /
  ``is_offline_requested`` / ``OfflineMissingWeightsError`` 等。
- 网络白名单包装函数（任务 3.2）：``guarded_get`` / ``guarded_post`` /
  ``guarded_urlopen`` 及守卫原语 ``assert_request_allowed`` / ``is_allowed_host``，
  仅放行 ``huggingface.co``、``hf.co``（含子域名）；离线模式下任何外发请求、
  以及任何模式下非白名单 host 均抛 ``NetworkBlockedError``。
"""

from ego_pod.privacy.offline import (
    ALLOWED_HOSTS,
    NETWORK_HOST_NOT_ALLOWED_MESSAGE,
    NETWORK_OFFLINE_BLOCKED_MESSAGE,
    OFFLINE_ENABLED_MESSAGE,
    OFFLINE_MISSING_WEIGHTS_MESSAGE,
    NetworkBlockedError,
    OfflineMissingWeightsError,
    OfflineMode,
    assert_request_allowed,
    get_active_offline_mode,
    guarded_get,
    guarded_post,
    guarded_urlopen,
    init_offline_mode,
    is_allowed_host,
    is_offline_requested,
    set_active_offline_mode,
)

__all__ = [
    "ALLOWED_HOSTS",
    "NETWORK_HOST_NOT_ALLOWED_MESSAGE",
    "NETWORK_OFFLINE_BLOCKED_MESSAGE",
    "OFFLINE_ENABLED_MESSAGE",
    "OFFLINE_MISSING_WEIGHTS_MESSAGE",
    "NetworkBlockedError",
    "OfflineMissingWeightsError",
    "OfflineMode",
    "assert_request_allowed",
    "get_active_offline_mode",
    "guarded_get",
    "guarded_post",
    "guarded_urlopen",
    "init_offline_mode",
    "is_allowed_host",
    "is_offline_requested",
    "set_active_offline_mode",
]
