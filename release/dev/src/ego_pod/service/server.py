"""基于标准库 `http.server.ThreadingHTTPServer` 的 `ego serve` 本地 HTTP JSON 服务（零第三方 Web 依赖）。"""

from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from typing import Any
from urllib.parse import urlsplit

from ego_pod.service.engine import EgoCognitiveEngine

__all__ = [
    "DEFAULT_SERVE_HOST",
    "DEFAULT_SERVE_PORT",
    "MAX_REQUEST_BODY_BYTES",
    "EgoServiceServer",
]

DEFAULT_SERVE_HOST = "127.0.0.1"
DEFAULT_SERVE_PORT = 7980
MAX_REQUEST_BODY_BYTES = 1024 * 1024  # 1 MiB 请求体安全上限


def _make_handler_class(
    engine: EgoCognitiveEngine, lock: threading.RLock
) -> type[BaseHTTPRequestHandler]:
    class _EgoHTTPHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            # 保持安静，避免打印含用户 prompt 的请求日志以守护本地隐私
            return None

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            raw = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def _read_json_body(self) -> dict[str, Any] | None:
            raw_len = self.headers.get("Content-Length")
            if raw_len is None:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "missing_content_length"},
                )
                return None
            try:
                content_len = int(raw_len)
            except ValueError:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "invalid_content_length"},
                )
                return None
            if content_len < 0 or content_len > MAX_REQUEST_BODY_BYTES:
                self._send_json(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    {"error": "payload_too_large"},
                )
                return None
            body = self.rfile.read(content_len) if content_len > 0 else b"{}"
            try:
                parsed = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "invalid_json"},
                )
                return None
            if not isinstance(parsed, dict):
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "json_body_must_be_object"},
                )
                return None
            return parsed

        def do_GET(self) -> None:  # noqa: N802
            route = urlsplit(self.path).path.rstrip("/") or "/"
            if route == "/health":
                self._send_json(HTTPStatus.OK, engine.health())
                return
            self._send_json(
                HTTPStatus.NOT_FOUND,
                {"error": "not_found", "path": route},
            )

        def do_POST(self) -> None:  # noqa: N802
            route = urlsplit(self.path).path.rstrip("/")
            if route not in (
                "/v1/intuition",
                "/v1/dynamic-context",
                "/v1/guard-rewrite",
            ):
                self._send_json(
                    HTTPStatus.NOT_FOUND,
                    {"error": "not_found", "path": route},
                )
                return

            payload = self._read_json_body()
            if payload is None:
                return

            try:
                if route == "/v1/intuition":
                    prompt = payload.get("prompt")
                    if not isinstance(prompt, str):
                        self._send_json(
                            HTTPStatus.BAD_REQUEST,
                            {"error": "field 'prompt' must be a string"},
                        )
                        return
                    ctx = payload.get("context")
                    if ctx is not None and not isinstance(ctx, dict):
                        self._send_json(
                            HTTPStatus.BAD_REQUEST,
                            {"error": "field 'context' must be an object when provided"},
                        )
                        return
                    if engine.neural_session is not None:
                        with lock:
                            res = engine.intuition(prompt, context=ctx)
                    else:
                        res = engine.intuition(prompt, context=ctx)
                    self._send_json(HTTPStatus.OK, res)
                    return

                if route == "/v1/dynamic-context":
                    prompt = payload.get("prompt", "")
                    if not isinstance(prompt, str):
                        self._send_json(
                            HTTPStatus.BAD_REQUEST,
                            {"error": "field 'prompt' must be a string"},
                        )
                        return
                    task_type = payload.get("task_type")
                    if task_type is not None and not isinstance(task_type, str):
                        self._send_json(
                            HTTPStatus.BAD_REQUEST,
                            {"error": "field 'task_type' must be a string when provided"},
                        )
                        return
                    self._send_json(
                        HTTPStatus.OK,
                        engine.dynamic_context(prompt, task_type=task_type),
                    )
                    return

                if route == "/v1/guard-rewrite":
                    prompt = payload.get("prompt", "")
                    draft_reply = payload.get("draft_reply")
                    if not isinstance(prompt, str) or not isinstance(draft_reply, str):
                        self._send_json(
                            HTTPStatus.BAD_REQUEST,
                            {
                                "error": "fields 'prompt' and 'draft_reply' must be strings"
                            },
                        )
                        return
                    self._send_json(
                        HTTPStatus.OK,
                        engine.guard_rewrite(prompt, draft_reply),
                    )
                    return
            except Exception as exc:  # noqa: BLE001
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": "internal_engine_error", "detail": str(exc)},
                )

    return _EgoHTTPHandler


class EgoServiceServer:
    """封装 ``ThreadingHTTPServer`` 的本地 `Ego` 认知服务实例（支持阻塞运行与后台线程托管）。"""

    def __init__(
        self,
        engine: EgoCognitiveEngine,
        *,
        host: str = DEFAULT_SERVE_HOST,
        port: int = DEFAULT_SERVE_PORT,
    ) -> None:
        self.engine = engine
        self._lock = threading.RLock()
        handler_cls = _make_handler_class(engine, self._lock)
        self._httpd = ThreadingHTTPServer((host, int(port)), handler_cls)
        bound_addr = self._httpd.server_address
        self.host: str = str(bound_addr[0])
        self.port: int = int(bound_addr[1])
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def serve_forever(self) -> None:
        """在当前线程阻塞服务，直至收到 ``shutdown()`` 或中断信号。"""

        try:
            self._httpd.serve_forever(poll_interval=0.1)
        finally:
            self._httpd.server_close()
            self.engine.close()

    def start_background(self) -> "EgoServiceServer":
        """在守护线程中启动 HTTP 服务并立即返回（便于嵌入式宿主与集成测试调用）。"""

        if self._thread is not None and self._thread.is_alive():
            return self
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            kwargs={"poll_interval": 0.05},
            name=f"ego-serve-{self.port}",
            daemon=True,
        )
        self._thread.start()
        return self

    def shutdown(self) -> None:
        """优雅关闭后台 HTTP 服务并释放模型会话。"""

        if self._thread is not None and self._thread.is_alive():
            self._httpd.shutdown()
            self._thread.join(timeout=5.0)
        self._httpd.server_close()
        self.engine.close()

    def __enter__(self) -> "EgoServiceServer":
        return self.start_background()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.shutdown()
