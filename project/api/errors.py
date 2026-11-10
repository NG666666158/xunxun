"""Unhandled errors become a JSON ``{"detail": ...}`` with the real reason.

Starlette answers an unhandled exception from its outermost layer, outside the CORS
middleware, so the desktop webview (another origin) cannot read that response and only
sees "Failed to fetch". This middleware sits inside CORS and answers the request itself.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from typing import Callable

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from config import MissingApiKeyError
from core.data_safety import DAMAGED_MESSAGE, DatabaseDamaged
from services.model_clients import MODEL_SERVICE_ERRORS, describe_model_error

logger = logging.getLogger(__name__)


def is_damage(exc: BaseException) -> bool:
    """SQLite's words for a damaged file (locked or busy files are something else)."""
    text = str(exc).lower()
    return isinstance(exc, sqlite3.DatabaseError) and ("malformed" in text or "not a database" in text)


def describe_error(exc: BaseException) -> tuple[int, str]:
    """``(status, message)`` for an exception no endpoint handled."""
    if isinstance(exc, (MissingApiKeyError, DatabaseDamaged)):
        return 503, str(exc)
    if is_damage(exc):
        return 503, DAMAGED_MESSAGE
    if isinstance(exc, MODEL_SERVICE_ERRORS):
        return 502, describe_model_error(exc)
    if isinstance(exc, PermissionError):
        return 500, f"没有权限访问文件：{exc.filename or exc}"
    if isinstance(exc, FileNotFoundError):
        return 500, f"找不到文件：{exc.filename or exc}"
    if isinstance(exc, OSError):
        return 500, f"文件或磁盘操作失败：{exc}"
    text = str(exc).strip()
    return 500, f"服务内部错误（{type(exc).__name__}{'：' + text[:300] if text else ''}），详情见日志。"


class ErrorDetailMiddleware:
    def __init__(self, app: ASGIApp, on_damage: Callable[[BaseException], None] | None = None):
        self.app = app
        self.on_damage = on_damage

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = False

        async def tracking_send(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, receive, tracking_send)
        except Exception as exc:
            logger.exception("Unhandled error in %s %s", scope.get("method"), scope.get("path"))
            if self.on_damage and is_damage(exc):
                self.on_damage(exc)
            if started:
                # Headers are out (a stream broke off): nothing more can be said over HTTP.
                raise
            status, message = describe_error(exc)
            body = json.dumps({"detail": message}, ensure_ascii=False).encode("utf-8")
            await send({
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"application/json; charset=utf-8"),
                    (b"content-length", str(len(body)).encode("latin-1")),
                ],
            })
            await send({"type": "http.response.body", "body": body})
