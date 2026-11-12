"""Per-launch token check for the desktop backend.

The desktop shell starts the backend on a random 127.0.0.1 port with a fresh token and
hands the token only to its own webviews (the main window and the quick-search window).
Any web page can still send requests to localhost, so every request must prove it came
from the shell: without the token the backend answers 401 and does nothing.

A request that carries an ``Origin`` other than the shell's own is refused too (403), token
or not: browsers always name the page that sent a cross-site request, so a web page cannot
use the backend even if the token ever leaked (a log, a screenshot). Requests without an
``Origin`` (the shell's health check, scripts on this machine) are judged by the token alone.
"""

from __future__ import annotations

import hmac
import json
from typing import Callable

from starlette.types import ASGIApp, Receive, Scope, Send


TOKEN_HEADER = "x-app-token"
PUBLIC_PATHS = frozenset({"/api/v1/health/live"})
# Where the Tauri webviews serve the page from: WebView2 on Windows (http, or https with
# useHttpsScheme), and the custom scheme used on other platforms.
DESKTOP_ORIGINS = ("http://tauri.localhost", "https://tauri.localhost", "tauri://localhost")


def request_token(scope: Scope) -> str:
    """Token from ``X-App-Token`` or ``Authorization: Bearer``."""
    headers = dict(scope.get("headers") or [])
    token = headers.get(TOKEN_HEADER.encode("latin-1"), b"").decode("latin-1")
    if token:
        return token
    authorization = headers.get(b"authorization", b"").decode("latin-1")
    scheme, _, value = authorization.partition(" ")
    return value.strip() if scheme.lower() == "bearer" else ""


def request_origin(scope: Scope) -> str | None:
    """The ``Origin`` header, or None when the request has none."""
    for name, value in scope.get("headers") or []:
        if name == b"origin":
            return value.decode("latin-1").strip()
    return None


class TokenAuthMiddleware:
    """Pure ASGI middleware (keeps SSE streaming untouched).

    ``token_provider`` is read per request, so an empty token (development) disables the
    check and tests can switch it on for the real app.
    """

    def __init__(self, app: ASGIApp, token_provider: Callable[[], str], origins: tuple[str, ...] = DESKTOP_ORIGINS):
        self.app = app
        self.token_provider = token_provider
        self.origins = frozenset(origins)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        expected = self.token_provider()
        if not expected or scope["type"] not in ("http", "websocket") or scope.get("path") in PUBLIC_PATHS:
            await self.app(scope, receive, send)
            return
        origin = request_origin(scope)
        if origin is not None and origin not in self.origins:
            await self._refuse(scope, send, 403, "只接受寻寻窗口发来的请求。")
            return
        if hmac.compare_digest(request_token(scope).encode("utf-8"), expected.encode("utf-8")):
            await self.app(scope, receive, send)
            return
        await self._refuse(scope, send, 401, "缺少或错误的访问令牌。")

    @staticmethod
    async def _refuse(scope: Scope, send: Send, status: int, detail: str) -> None:
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 4000 + status})
            return
        body = json.dumps({"detail": detail}, ensure_ascii=False).encode("utf-8")
        headers = [
            (b"content-type", b"application/json; charset=utf-8"),
            (b"content-length", str(len(body)).encode("latin-1")),
            (b"cache-control", b"no-store"),
        ]
        if status == 401:
            headers.append((b"www-authenticate", b"Bearer"))
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": body})
