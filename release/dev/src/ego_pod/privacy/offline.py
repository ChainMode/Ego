"""离线模式开关。

实现 Requirement 9.6 / 9.7 中与「离线模式开关」相关的部分：

- 进程启动时根据 ``--offline`` CLI 参数与 ``EGO_OFFLINE=1`` 环境变量决定是否
  启用离线模式；二者任一启用即进入离线模式（R9-AC6）。
- 启用时在进程环境中设置 ``HF_HUB_OFFLINE=1`` 与 ``TRANSFORMERS_OFFLINE=1``，
  让 huggingface_hub / transformers 等上游库主动拒绝外发网络请求。
- 未显式启用时（含网络故障、缺失凭证、DNS 失败等情形）不做任何环境变更，即
  不自动进入离线行为（R9-AC6）。
- 定义 :class:`OfflineMissingWeightsError`，供执行流程在离线模式下检测到
  Base_Model 权重不可用时抛出。CLI 顶层将其按 R9-AC7 映射为
  (a) 在标准错误输出 "离线模式下缺失 Base_Model 权重"；
  (b) 以非零退出码（1，即 ``EXIT_RUNTIME_FAILURE``）结束。

任务 3.2 在本模块追加网络白名单包装函数（``guarded_get`` / ``guarded_post`` /
``guarded_urlopen`` 等）：

- 仅放行 ``huggingface.co``、``hf.co`` 及其子域名两个主机族（R9-AC5 中
  "仅向预先声明的模型权重源端点发起下载请求" 的内部白名单实现），其余 host 在
  任何模式下都抛 :class:`NetworkBlockedError`（``EgoError`` 子类）。
- 离线模式启用时（R9-AC6）拒绝执行任何外发网络请求——即便目标在白名单内也一律
  抛 :class:`NetworkBlockedError`。
- 守卫检查与离线状态通过进程内的 :class:`OfflineMode` 集成：``init_offline_mode``
  在进程启动时把解析出的 :class:`OfflineMode` 注册为"当前活动离线模式"，
  ``guarded_*`` 默认据此判断；测试可显式传入 ``offline=`` 覆盖。

设计取舍（见 design.md §8）：不全局劫持 socket，而是"信任内部模块自律调用
``guarded_*`` 包装函数 + 设置 HF 离线环境变量"，因为全局劫持会误伤测试对
``localhost`` 的连接，反而更难测。
"""

from __future__ import annotations

import sys
from collections.abc import Callable, MutableMapping
from dataclasses import dataclass
from typing import Any, TextIO
from urllib.parse import urlsplit

# exit_codes 模块由任务 1.2 并行创建；按设计章节 §Error Handling 的命名约定导入。
from ego_pod.exit_codes import EXIT_RUNTIME_FAILURE, EgoError, RuntimeError_

__all__ = [
    "ALLOWED_HOSTS",
    "ENV_EGO_OFFLINE",
    "ENV_HF_HUB_OFFLINE",
    "ENV_TRANSFORMERS_OFFLINE",
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

# --- 环境变量名常量 -----------------------------------------------------------

#: 用户显式启用离线模式的环境变量，按 R9-AC6 取值为 ``"1"`` 时启用。
ENV_EGO_OFFLINE = "EGO_OFFLINE"
#: huggingface_hub 离线开关；离线模式启用时被设为 ``"1"``。
ENV_HF_HUB_OFFLINE = "HF_HUB_OFFLINE"
#: transformers 离线开关；离线模式启用时被设为 ``"1"``。
ENV_TRANSFORMERS_OFFLINE = "TRANSFORMERS_OFFLINE"

# --- 固定文案常量 -------------------------------------------------------------

#: 离线模式启用时在标准错误输出的提示（R9-AC6）。
OFFLINE_ENABLED_MESSAGE = "离线模式已启用"
#: 离线模式下缺失 Base_Model 权重时的错误文案（R9-AC7）。
OFFLINE_MISSING_WEIGHTS_MESSAGE = "离线模式下缺失 Base_Model 权重"
#: 离线模式下任何外发请求被拒绝时的错误文案（R9-AC6）。
NETWORK_OFFLINE_BLOCKED_MESSAGE = "离线模式已启用，拒绝外发网络请求"
#: 非白名单 host 在任何模式下被拒绝时的错误文案（R9-AC5）。
NETWORK_HOST_NOT_ALLOWED_MESSAGE = "非白名单主机的外发网络请求被拒绝"

# --- 网络白名单 ---------------------------------------------------------------

#: 允许外发请求的主机族（R9-AC5 内部白名单）。MVP 中实际下载请求只走
#: HuggingFace Hub 端点，因此白名单仅 ``huggingface.co`` 与 ``hf.co``，并按设计
#: 一并放行其子域名（如 ``cdn-lfs.huggingface.co``）。
ALLOWED_HOSTS: tuple[str, ...] = ("huggingface.co", "hf.co")


class OfflineMissingWeightsError(RuntimeError_):
    """离线模式下 Base_Model 权重不可用。

    R9-AC7：当 ``--offline`` / ``EGO_OFFLINE=1`` 启用且基座权重缓存缺失时抛出。
    CLI 顶层的统一异常处理会将其映射为单一不可拆分的合规要求：
    (a) 在标准错误输出 :data:`OFFLINE_MISSING_WEIGHTS_MESSAGE`；
    (b) 以非零退出码 :data:`~ego_pod.exit_codes.EXIT_RUNTIME_FAILURE` (=1) 结束。

    在线模式下的基座加载失败由 R8-AC7 单独处理，不走本异常路径。
    """

    #: 显式声明退出码，确保即便 ``RuntimeError_`` 的默认值变化也固定为 1。
    exit_code = EXIT_RUNTIME_FAILURE

    def __init__(self, message: str = OFFLINE_MISSING_WEIGHTS_MESSAGE) -> None:
        super().__init__(message)


class NetworkBlockedError(EgoError):
    """外发网络请求被隐私守卫拦截。

    两种触发场景（任一即抛）：

    * **离线模式拦截（R9-AC6）**：``--offline`` / ``EGO_OFFLINE=1`` 启用时，任何
      外发请求都被拒绝——即便目标主机在 :data:`ALLOWED_HOSTS` 白名单内。
    * **非白名单主机拦截（R9-AC5）**：无论在线还是离线模式，目标主机不属于
      :data:`ALLOWED_HOSTS` 时一律拒绝。

    继承 :class:`~ego_pod.exit_codes.EgoError`，沿用其
    :data:`~ego_pod.exit_codes.EXIT_RUNTIME_FAILURE` (=1) 退出码，由 CLI 顶层
    统一捕获并打印消息到标准错误。
    """


def is_offline_requested(
    cli_offline: bool,
    env: MutableMapping[str, str],
) -> bool:
    """判断离线模式是否被显式启用。

    ``--offline`` CLI 参数（``cli_offline``）或 ``EGO_OFFLINE=1`` 环境变量任一
    启用即视为请求离线模式。仅 ``EGO_OFFLINE`` 取值严格等于 ``"1"`` 时视为启用，
    与 R9-AC6 的 ``EGO_OFFLINE=1`` 约定一致；其它取值（含空串、``"0"``、``"true"``）
    一律不启用，避免误入离线行为。
    """
    if cli_offline:
        return True
    return env.get(ENV_EGO_OFFLINE) == "1"


@dataclass(frozen=True)
class OfflineMode:
    """离线模式状态对象。

    由 :func:`init_offline_mode` 在进程启动时解析得到，贯穿整个 Ego_Pod 进程
    生命周期使用。``enabled`` 为 ``True`` 表示已进入离线模式。
    """

    enabled: bool

    def require_base_model_available(self, available: bool) -> None:
        """R9-AC7 强制点：离线模式下若基座权重不可用则抛异常。

        仅在 :attr:`enabled` 为真时生效。``available`` 由调用方（Trainer）在加载
        基座前对本地 Base_Model 权重缓存的存在性检查给出。当离线模式启用且权重
        不可用时抛出 :class:`OfflineMissingWeightsError`，由 CLI 顶层按 R9-AC7
        映射为固定 stderr 文案与退出码 1。

        在线模式下（``enabled`` 为假）本方法不做任何动作，基座加载失败交由
        R8-AC7 处理。
        """
        if self.enabled and not available:
            raise OfflineMissingWeightsError()


def init_offline_mode(
    cli_offline: bool = False,
    env: MutableMapping[str, str] | None = None,
    stream: TextIO | None = None,
) -> OfflineMode:
    """进程启动时解析并应用离线模式开关。

    读取 ``--offline`` CLI 参数（``cli_offline``）与 ``EGO_OFFLINE=1`` 环境变量；
    二者任一启用即进入离线模式（R9-AC6）。启用时：

    1. 在进程环境中设置 ``HF_HUB_OFFLINE=1`` 与 ``TRANSFORMERS_OFFLINE=1``，让
       上游库主动拒绝外发网络请求。
    2. 在标准错误输出 :data:`OFFLINE_ENABLED_MESSAGE` 提示（R9-AC6）。

    未显式启用时不做任何环境变更，也不打印提示，即不自动进入离线行为。

    参数
    ----
    cli_offline:
        来自 CLI ``--offline`` 标志的布尔值。
    env:
        目标环境映射，默认对 ``os.environ`` 生效；测试可注入临时字典。
    stream:
        离线提示的输出流，默认 ``sys.stderr``。

    返回
    ----
    解析得到的 :class:`OfflineMode`。
    """
    # 延迟到运行期再取 os.environ，避免在导入期固化引用，便于测试注入。
    if env is None:
        import os

        target_env: MutableMapping[str, str] = os.environ
    else:
        target_env = env

    enabled = is_offline_requested(cli_offline, target_env)
    if enabled:
        target_env[ENV_HF_HUB_OFFLINE] = "1"
        target_env[ENV_TRANSFORMERS_OFFLINE] = "1"
        out = sys.stderr if stream is None else stream
        print(OFFLINE_ENABLED_MESSAGE, file=out)

    mode = OfflineMode(enabled=enabled)
    # 注册为进程内"当前活动离线模式"，供 guarded_* 包装函数默认据此判断（R9-AC6）。
    set_active_offline_mode(mode)
    return mode


# ---------------------------------------------------------------------------
# 网络白名单包装（任务 3.2）
# ---------------------------------------------------------------------------

#: 进程内当前活动的离线模式。由 :func:`init_offline_mode` 在启动时注册；
#: ``guarded_*`` 包装函数在未显式传入 ``offline=`` 时据此判断是否处于离线模式。
#: 默认 ``None`` 表示尚未初始化，等价于"在线模式"（仅施加白名单约束）。
_ACTIVE_OFFLINE_MODE: OfflineMode | None = None


def set_active_offline_mode(mode: OfflineMode | None) -> None:
    """注册进程内当前活动的 :class:`OfflineMode`。

    通常由 :func:`init_offline_mode` 在进程启动时调用。测试可显式调用以重置或
    注入特定状态（传入 ``None`` 表示恢复未初始化状态）。
    """
    global _ACTIVE_OFFLINE_MODE
    _ACTIVE_OFFLINE_MODE = mode


def get_active_offline_mode() -> OfflineMode | None:
    """返回进程内当前活动的 :class:`OfflineMode`，未初始化时返回 ``None``。"""
    return _ACTIVE_OFFLINE_MODE


def is_allowed_host(host: str | None) -> bool:
    """判断主机名是否属于白名单 :data:`ALLOWED_HOSTS`（含子域名）。

    比较前对主机名做大小写归一化并去除末尾的根域点 ``.``。``host`` 等于白名单
    条目本身，或以 ``"." + 条目`` 结尾（即其子域名，如
    ``cdn-lfs.huggingface.co``）时视为允许。``None``、空串、形如纯端口/IP 字面量
    等无法匹配的输入一律视为不允许。
    """
    if not host:
        return False
    normalized = host.strip().lower().rstrip(".")
    if not normalized:
        return False
    for allowed in ALLOWED_HOSTS:
        if normalized == allowed or normalized.endswith("." + allowed):
            return True
    return False


def _extract_host(url: str) -> str | None:
    """从 URL 中解析主机名（不含端口、用户信息），无法解析时返回 ``None``。"""
    try:
        return urlsplit(url).hostname
    except ValueError:
        # 形如非法 IPv6 字面量等极端输入：视为无主机，交由调用方按非白名单拒绝。
        return None


def assert_request_allowed(
    url: str,
    *,
    offline: OfflineMode | None = None,
) -> None:
    """守卫：校验对 ``url`` 的外发请求是否被允许，不允许则抛异常。

    判定顺序与语义：

    1. **离线模式拦截（R9-AC6）**：若处于离线模式（``offline`` 显式给出，或回退到
       :func:`get_active_offline_mode` 注册的活动模式，其 ``enabled`` 为真），无论
       目标 host 是否在白名单内，一律抛 :class:`NetworkBlockedError`。
    2. **白名单拦截（R9-AC5）**：否则校验目标 host，不属于 :data:`ALLOWED_HOSTS`
       （含子域名）时抛 :class:`NetworkBlockedError`。

    校验通过则正常返回 ``None``。

    参数
    ----
    url:
        即将请求的目标 URL。
    offline:
        显式指定的离线模式状态；为 ``None`` 时回退到进程内活动离线模式。
    """
    mode = offline if offline is not None else get_active_offline_mode()
    if mode is not None and mode.enabled:
        raise NetworkBlockedError(NETWORK_OFFLINE_BLOCKED_MESSAGE)

    host = _extract_host(url)
    if not is_allowed_host(host):
        raise NetworkBlockedError(
            f"{NETWORK_HOST_NOT_ALLOWED_MESSAGE}: {host or url}"
        )


def guarded_get(
    url: str,
    *args: Any,
    offline: OfflineMode | None = None,
    requester: Callable[..., Any] | None = None,
    **kwargs: Any,
) -> Any:
    """对 GET 请求的薄包装：先施加隐私守卫，再委托底层发起请求。

    在调用底层 HTTP 客户端前调用 :func:`assert_request_allowed`：离线模式下任何
    外发请求被拒（R9-AC6），非白名单 host 在任何模式下被拒（R9-AC5）。守卫通过后
    才委托 ``requester`` 实际发起请求。

    ``requester`` 默认延迟导入 ``requests.get``；调用方/测试可注入自定义可调用对象
    （签名兼容 ``requester(url, *args, **kwargs)``）。``offline`` 用于显式覆盖离线
    状态，默认回退到进程内活动离线模式。
    """
    assert_request_allowed(url, offline=offline)
    call = requester if requester is not None else _default_requester("get")
    return call(url, *args, **kwargs)


def guarded_post(
    url: str,
    *args: Any,
    offline: OfflineMode | None = None,
    requester: Callable[..., Any] | None = None,
    **kwargs: Any,
) -> Any:
    """对 POST 请求的薄包装，语义同 :func:`guarded_get`。

    ``requester`` 默认延迟导入 ``requests.post``。
    """
    assert_request_allowed(url, offline=offline)
    call = requester if requester is not None else _default_requester("post")
    return call(url, *args, **kwargs)


def guarded_urlopen(
    url: str,
    *args: Any,
    offline: OfflineMode | None = None,
    requester: Callable[..., Any] | None = None,
    **kwargs: Any,
) -> Any:
    """对 ``urllib.request.urlopen`` 的薄包装，语义同 :func:`guarded_get`。

    仅支持字符串 URL；守卫据其 host 判定白名单。``requester`` 默认延迟导入
    ``urllib.request.urlopen``。
    """
    assert_request_allowed(url, offline=offline)
    if requester is not None:
        return requester(url, *args, **kwargs)
    from urllib.request import urlopen

    return urlopen(url, *args, **kwargs)  # noqa: S310 - host 已通过白名单守卫校验


def _default_requester(method: str) -> Callable[..., Any]:
    """延迟导入 ``requests`` 并返回对应的方法可调用对象。

    将 ``requests`` 作为运行期可选依赖延迟导入，避免在无网络栈的测试/CI 环境因缺包
    而在导入期失败。守卫逻辑本身不依赖 ``requests``。
    """
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - 取决于环境是否安装 requests
        raise EgoError(
            "外发网络请求需要 'requests' 依赖，但未安装"
        ) from exc
    return getattr(requests, method)
