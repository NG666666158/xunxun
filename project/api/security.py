"""Per-launch token check for the desktop backend.

The desktop shell starts the backend on a random 127.0.0.1 port with a fresh token and
hands the token only to its own webview. Any web page can still send requests to
localhost, so every request must prove it came from the shell: without the token the
backend answers 401 and does nothing.
"""

from __future__ import annotations

import hmac
import json
from typing import Callable

from starlette.types import ASGIApp, Receive, Scope, Send


TOKEN_HEADER = "x-app-token"
PUBLIC_PATHS = frozenset({"/api/v1/health/live"})


def request_token(scope: Scope) -> str:
    """Token from ``X-App-Token`` or ``Authorization: Bearer``."""
    headers = dict(scope.get("headers") or [])
    token = headers.get(TOKEN_HEADER.encode("latin-1"), b"").decode("latin-1")
    if token:
        return token
    authorization = headers.get(b"authorization", b"").decode("latin-1")
    scheme, _, value = authorization.partition(" ")
    return value.strip() if scheme.lower() == "bearer" else ""


class TokenAuthMiddleware:
    """Pure ASGI middleware (keeps SSE streaming untouched).

    ``token_provider`` is read per request, so an empty token (development) disables the
    check and tests can switch it on for the real app.
    """

    def __init__(self, app: ASGIApp, token_provider: Callable[[], str]):
        self.app = app
        self.token_provider = token_provider

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        expected = self.token_provider()
        if not expected or scope["type"] not in ("http", "websocket") or scope.get("path") in PUBLIC_PATHS:
            await self.app(scope, receive, send)
            return
        if hmac.compare_digest(request_token(scope).encode("utf-8"), expected.encode("utf-8")):
            await self.app(scope, receive, send)
            return
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 4401})
            return
        body = json.dumps({"detail": "缺少或错误的访问令牌。"}, ensure_ascii=False).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json; charset=utf-8"),
                    (b"content-length", str(len(body)).encode("latin-1")),
                    (b"www-authenticate", b"Bearer"),
                    (b"cache-control", b"no-store"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
