"""The desktop backend listens on 127.0.0.1, where any web page in any browser can send it
requests. Only the shell's own windows may use it (api/security.py): requests without the
launch token, or sent by a page from another origin, are refused; the main window and the
quick-search window keep working."""

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from api import main
from api.main import DEVELOPMENT_ORIGINS
from api.security import DESKTOP_ORIGINS, TokenAuthMiddleware

TOKEN = "launch-token-0123456789abcdef"
DESKTOP = "http://tauri.localhost"
WEB_PAGES = ("https://evil.example", "http://127.0.0.1:5500", "http://localhost:8000", "null", "http://tauri.localhost.evil.example")


def _unit_client() -> TestClient:
    app = FastAPI()
    app.add_middleware(TokenAuthMiddleware, token_provider=lambda: TOKEN)

    @app.get("/api/v1/health/live")
    def live():
        return {"status": "alive"}

    @app.post("/api/v1/thing")
    def thing():
        return {"ok": True}

    return TestClient(app)


@pytest.mark.parametrize("origin", WEB_PAGES)
def test_web_page_origins_are_refused_with_or_without_the_token(origin):
    client = _unit_client()
    for headers in ({"Origin": origin}, {"Origin": origin, "X-App-Token": TOKEN}, {"Origin": origin, "Authorization": f"Bearer {TOKEN}"}):
        refused = client.post("/api/v1/thing", headers=headers)
        assert refused.status_code == 403, headers
        assert "ok" not in refused.json()


def test_desktop_origins_and_local_scripts_need_only_the_token():
    client = _unit_client()
    for origin in DESKTOP_ORIGINS:
        assert client.post("/api/v1/thing", headers={"Origin": origin, "X-App-Token": TOKEN}).json() == {"ok": True}
        assert client.post("/api/v1/thing", headers={"Origin": origin}).status_code == 401
    # No Origin: the shell's own checks, scripts on this machine.
    assert client.post("/api/v1/thing", headers={"X-App-Token": TOKEN}).status_code == 200
    assert client.post("/api/v1/thing").status_code == 401
    # Liveness stays public; a web page cannot read it (no CORS header for its origin).
    assert client.get("/api/v1/health/live", headers={"Origin": "https://evil.example"}).status_code == 200


def test_websocket_from_a_web_page_is_closed():
    from fastapi import WebSocket
    from starlette.websockets import WebSocketDisconnect

    app = FastAPI()
    app.add_middleware(TokenAuthMiddleware, token_provider=lambda: TOKEN)

    @app.websocket("/ws")
    async def ws(socket: WebSocket):
        await socket.accept()
        await socket.send_text("hi")

    client = TestClient(app)
    with pytest.raises(WebSocketDisconnect) as closed:
        with client.websocket_connect("/ws", headers={"Origin": "https://evil.example", "X-App-Token": TOKEN}) as socket:
            socket.receive_text()
    assert closed.value.code == 4403
    with client.websocket_connect("/ws", headers={"Origin": DESKTOP, "X-App-Token": TOKEN}) as socket:
        assert socket.receive_text() == "hi"


# --- the real app -----------------------------------------------------------------


class _FakeRegistry:
    def get_file(self, file_id):
        return None


class _FakeLibrary:
    registry = _FakeRegistry()

    def status(self, since=None):
        return {"busy": False, "queued": 0, "counts": {"indexed": 1}}

    def list_roots(self):
        return [{"root_id": "r1", "path": "D:/资料"}]

    def sync(self):
        return {"checked": 1}


class _FakeRag:
    def search_files(self, q, limit=8, filters=None, scope=None):
        return {"query": q, "total": 1, "files": [{"file_id": "f1", "name": "季度报告.docx"}]}


@pytest.fixture
def desktop(monkeypatch):
    monkeypatch.setattr(main.settings, "app_token", TOKEN)
    monkeypatch.setattr(main, "_library", lambda: _FakeLibrary())
    monkeypatch.setattr(main, "_ready_system", lambda: _FakeRag())
    monkeypatch.setattr(main, "preview_prefetcher", lambda: SimpleNamespace(start=lambda: None, request=lambda doc_id: "skip"))
    monkeypatch.setattr(main, "slides_prefetcher", lambda: SimpleNamespace(start=lambda: None, request=lambda doc_id: "skip"))
    return TestClient(main.app)


# (method, path, json body): what the main window and the quick-search window call.
MAIN_WINDOW = [
    ("GET", "/api/v1/settings", None),
    ("GET", "/api/v1/library/status", None),
    ("GET", "/api/v1/watch-roots", None),
    ("POST", "/api/v1/library/sync", {}),
]
QUICK_WINDOW = [
    ("GET", "/api/v1/files/search?q=%E5%AD%A3%E5%BA%A6&limit=8", None),
    ("POST", "/api/v1/documents/prepare", {"doc_ids": ["f1"]}),
]


@pytest.mark.parametrize("method,path,body", MAIN_WINDOW + QUICK_WINDOW)
def test_desktop_windows_still_work(desktop, method, path, body):
    preflight = desktop.options(
        path,
        headers={"Origin": DESKTOP, "Access-Control-Request-Method": method, "Access-Control-Request-Headers": "x-app-token,content-type"},
    )
    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == DESKTOP

    answer = desktop.request(method, path, json=body, headers={"Origin": DESKTOP, "X-App-Token": TOKEN})
    assert answer.status_code == 200, answer.text
    assert answer.headers["access-control-allow-origin"] == DESKTOP


@pytest.mark.parametrize("method,path,body", MAIN_WINDOW + QUICK_WINDOW)
def test_requests_without_the_token_are_refused(desktop, method, path, body):
    for headers in ({}, {"Origin": DESKTOP}, {"X-App-Token": "not-the-token"}, {"Authorization": "Basic " + TOKEN}):
        assert desktop.request(method, path, json=body, headers=headers).status_code == 401, headers


@pytest.mark.parametrize("origin", WEB_PAGES)
def test_a_web_page_calling_127_0_0_1_is_refused(desktop, origin):
    # A simple request needs no preflight: it reaches the backend, which must refuse it
    # itself, even when the page somehow knows the token.
    for headers in ({"Origin": origin}, {"Origin": origin, "X-App-Token": TOKEN}):
        for method, path, body in MAIN_WINDOW + QUICK_WINDOW:
            refused = desktop.request(method, path, json=body, headers=headers)
            assert refused.status_code == 403, (method, path, headers)
            if origin not in DEVELOPMENT_ORIGINS:  # the packaged app drops these from CORS too
                assert "access-control-allow-origin" not in refused.headers
    sync = desktop.post("/api/v1/library/sync", content="{}", headers={"Origin": origin, "Content-Type": "text/plain"})
    assert sync.status_code == 403
    # A preflighted request does not get as far.
    preflight = desktop.options(
        "/api/v1/settings",
        headers={"Origin": origin, "Access-Control-Request-Method": "PUT", "Access-Control-Request-Headers": "x-app-token"},
    )
    if origin not in DEVELOPMENT_ORIGINS:
        assert preflight.status_code == 400
        assert "access-control-allow-origin" not in preflight.headers


def test_packaged_app_allows_only_the_desktop_origins_in_cors():
    assert main.cors_origins(token="launch-token") == list(DESKTOP_ORIGINS)
    assert main.cors_origins(token="") == [*DEVELOPMENT_ORIGINS, *DESKTOP_ORIGINS]


def test_page_files_are_not_served_to_a_browser(desktop):
    assert desktop.get("/ui/").status_code == 401
    assert desktop.get("/ui/", headers={"Origin": "https://evil.example", "X-App-Token": TOKEN}).status_code == 403
    assert desktop.get("/ui/app.js", headers={"X-App-Token": TOKEN}).status_code == 200


def test_development_mode_without_token_is_unchanged(monkeypatch):
    monkeypatch.setattr(main.settings, "app_token", "")
    client = TestClient(main.app)
    assert client.get("/api/v1/settings").status_code == 200
    assert client.get("/api/v1/settings", headers={"Origin": "http://127.0.0.1:8000"}).status_code == 200
