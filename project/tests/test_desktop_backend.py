"""Desktop backend: per-launch token, data-dir resolution, keyring-backed settings."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

import app as backend_app
import config
from api import main
from api.security import TokenAuthMiddleware
from config import Settings, default_data_dir
from core import user_settings


# --- token middleware -------------------------------------------------------------


def _token_app(token: str) -> TestClient:
    app = FastAPI()
    app.add_middleware(TokenAuthMiddleware, token_provider=lambda: token)

    @app.get("/api/v1/health/live")
    def live():
        return {"status": "alive"}

    @app.get("/api/v1/thing")
    def thing():
        return {"ok": True}

    return TestClient(app)


def test_token_required_when_configured():
    client = _token_app("s3cret-token")

    missing = client.get("/api/v1/thing")
    assert missing.status_code == 401
    assert missing.headers["www-authenticate"] == "Bearer"
    assert client.get("/api/v1/thing", headers={"X-App-Token": "wrong"}).status_code == 401
    assert client.get("/api/v1/thing", headers={"Authorization": "Basic s3cret-token"}).status_code == 401
    assert client.get("/api/v1/thing", headers={"X-App-Token": "s3cret-token"}).json() == {"ok": True}
    assert client.get("/api/v1/thing", headers={"Authorization": "Bearer s3cret-token"}).status_code == 200


def test_liveness_is_public_and_empty_token_disables_check():
    assert _token_app("s3cret-token").get("/api/v1/health/live").status_code == 200
    assert _token_app("").get("/api/v1/thing").status_code == 200


def test_real_app_rejects_requests_without_token(monkeypatch):
    monkeypatch.setattr(main.settings, "app_token", "launch-token")
    client = TestClient(main.app)

    assert client.get("/api/v1/health/live").status_code == 200
    assert client.get("/api/v1/settings").status_code == 401
    assert client.get("/ui/").status_code == 401
    assert client.post("/api/v1/chat", json={"question": "hi"}).status_code == 401
    assert client.get("/api/v1/settings", headers={"X-App-Token": "launch-token"}).status_code == 200


def test_cors_preflight_from_desktop_webview_needs_no_token(monkeypatch):
    monkeypatch.setattr(main.settings, "app_token", "launch-token")
    client = TestClient(main.app)
    preflight = client.options(
        "/api/v1/settings",
        headers={
            "Origin": "http://tauri.localhost",
            "Access-Control-Request-Method": "PUT",
            "Access-Control-Request-Headers": "x-app-token,content-type",
        },
    )
    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == "http://tauri.localhost"

    denied = client.get("/api/v1/settings", headers={"Origin": "http://tauri.localhost"})
    assert denied.status_code == 401
    assert denied.headers["access-control-allow-origin"] == "http://tauri.localhost"


# --- data directory ---------------------------------------------------------------


def test_development_data_dir_is_repository_data():
    assert default_data_dir(frozen=False) == config.REPO_DIR / "data"


def test_packaged_data_dir_is_under_appdata():
    path = default_data_dir(frozen=True, environ={"APPDATA": r"C:\Users\me\AppData\Roaming"})
    assert path == Path(r"C:\Users\me\AppData\Roaming") / config.APP_DATA_NAME


def test_all_app_files_follow_data_dir(tmp_path):
    settings = Settings(data_dir=tmp_path / "kb")
    root = tmp_path / "kb"
    assert settings.embedded_db_path == root / "knowledge.sqlite3"
    assert settings.registry_path == root / "knowledge.sqlite3"
    assert settings.parsed_dir == root / "parsed"
    assert settings.parent_store_dir == root / "parents"
    assert settings.upload_dir == root / "uploads"
    assert settings.cache_dir == root / "cache"
    assert settings.log_dir == root / "logs"
    assert settings.conversations_db_path == root / "conversations.sqlite3"
    # The enterprise-demo stores are gone from the product.
    assert not hasattr(settings, "structured_db_path") and not hasattr(settings, "business_data_path")


def test_explicit_paths_and_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path / "from-env"))
    settings = Settings(parsed_dir=tmp_path / "elsewhere")
    assert settings.data_dir == tmp_path / "from-env"
    assert settings.upload_dir == tmp_path / "from-env" / "uploads"
    assert settings.parsed_dir == tmp_path / "elsewhere"


def test_frozen_build_keeps_evaluation_output_in_data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config.sys, "frozen", True, raising=False)
    assert Settings(data_dir=tmp_path).evaluation_dir == tmp_path / "evaluation"


def test_cli_arguments_reach_settings(tmp_path):
    args = backend_app.parse_args(["--port", "51234", "--token", "abc", "--data-dir", str(tmp_path)])
    environ: dict[str, str] = {}
    backend_app.apply_environment(args, environ)

    assert args.port == 51234
    assert environ == {"RAG_APP_TOKEN": "abc", "RAG_DATA_DIR": str(tmp_path.resolve())}
    assert backend_app.bind_host(args, frozen=False, token="abc") == "127.0.0.1"
    assert backend_app.bind_host(backend_app.parse_args([]), frozen=True, token="") == "127.0.0.1"
    assert backend_app.bind_host(backend_app.parse_args([]), frozen=False, token="") == "0.0.0.0"


# --- keyring + preferences --------------------------------------------------------


def test_api_key_round_trips_through_keyring(memory_keyring):
    assert user_settings.load_api_key() == ""
    user_settings.save_api_key("sk-abcdefghijklmnop")

    assert memory_keyring.passwords == {
        (user_settings.KEYRING_SERVICE, user_settings.KEYRING_USERNAME): "sk-abcdefghijklmnop"
    }
    assert user_settings.load_api_key() == "sk-abcdefghijklmnop"
    user_settings.delete_api_key()
    user_settings.delete_api_key()  # deleting twice is fine
    assert user_settings.load_api_key() == ""


def test_stored_key_and_model_override_env(tmp_path):
    settings = Settings(data_dir=tmp_path, siliconflow_api_key="sk-from-env-file")
    user_settings.save_api_key("sk-from-credential-manager")
    user_settings.save_preferences(settings, {"llm_model": "Qwen/Qwen3-32B"})

    user_settings.apply_user_settings(settings)
    assert settings.siliconflow_api_key == "sk-from-credential-manager"
    assert settings.llm_model == "Qwen/Qwen3-32B"
    assert "sk-from" not in (tmp_path / "preferences.json").read_text(encoding="utf-8")


def test_unavailable_keyring_falls_back_to_env(monkeypatch):
    import keyring

    def broken(*_args):
        raise RuntimeError("no backend")

    monkeypatch.setattr(keyring, "get_password", broken)
    monkeypatch.setattr(keyring, "set_password", broken)
    assert user_settings.load_api_key() == ""
    with pytest.raises(user_settings.SecretStoreError):
        user_settings.save_api_key("sk-abcdefghijklmnop")


def test_mask_key_hides_middle():
    assert user_settings.mask_key("sk-abcdefghijklmnop") == "sk-ab…mnop"
    assert "abc" not in user_settings.mask_key("sk-abc")


class RecordingSystem:
    initialized = False

    def __init__(self):
        self.reloads = 0

    def reload_models(self):
        self.reloads += 1


def test_settings_endpoint_saves_key_to_keyring_not_disk(tmp_path, monkeypatch):
    settings = Settings(data_dir=tmp_path, siliconflow_api_key="")
    system = RecordingSystem()
    monkeypatch.setattr(main, "settings", settings)
    monkeypatch.setattr(main, "system", system)
    client = TestClient(main.app)

    before = client.get("/api/v1/settings").json()
    assert before["api_key"]["configured"] is False
    assert before["data_dir"] == str(tmp_path)

    saved = client.put("/api/v1/settings", json={"api_key": "sk-newkey-1234567890", "llm_model": "Qwen/Qwen3-14B"})
    assert saved.status_code == 200
    body = saved.json()
    assert body["api_key"] == {"configured": True, "masked": "sk-ne…7890", "source": "keyring"}
    assert body["llm_model"] == "Qwen/Qwen3-14B"
    assert settings.siliconflow_api_key == "sk-newkey-1234567890"
    assert user_settings.load_api_key() == "sk-newkey-1234567890"
    assert system.reloads == 1
    written = [path.read_text(encoding="utf-8", errors="ignore") for path in tmp_path.rglob("*") if path.is_file()]
    assert not any("sk-newkey" in text for text in written)

    assert client.put("/api/v1/settings", json={"api_key": "sk new key with spaces"}).status_code == 400
    assert client.put("/api/v1/settings", json={"api_key": ""}).status_code == 422

    removed = client.delete("/api/v1/settings/api-key").json()
    assert user_settings.load_api_key() == ""
    assert removed["api_key"]["source"] != "keyring"


def test_deleting_last_key_after_initialization_does_not_fail(tmp_path, monkeypatch):
    class InitializedSystem:
        def reload_models(self):
            raise RuntimeError("缺少 SILICONFLOW_API_KEY")

    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setattr(main, "Settings", lambda: Settings(_env_file=None, data_dir=tmp_path))
    monkeypatch.setattr(main, "settings", Settings(_env_file=None, data_dir=tmp_path, siliconflow_api_key="sk-old-key-123456"))
    monkeypatch.setattr(main, "system", InitializedSystem())
    user_settings.save_api_key("sk-old-key-123456")

    response = TestClient(main.app).delete("/api/v1/settings/api-key")
    assert response.status_code == 200
    assert response.json()["api_key"]["configured"] is False


def test_image_understanding_is_saved_and_requeues_pdfs(tmp_path, monkeypatch):
    class Library:
        def __init__(self):
            self.calls = 0

        def requeue_outdated(self):
            self.calls += 1
            return 3

    settings = Settings(data_dir=tmp_path, siliconflow_api_key="sk-test-1234567890")
    system = RecordingSystem()
    system.library = Library()
    monkeypatch.setattr(main, "settings", settings)
    monkeypatch.setattr(main, "system", system)
    client = TestClient(main.app)

    assert client.get("/api/v1/settings").json()["image_understanding"] == "off"
    body = client.put("/api/v1/settings", json={"image_understanding": "local"}).json()
    assert body["image_understanding"] == "local" and body["requeued"] == 3
    assert settings.image_understanding == "local"
    # Unchanged mode: nothing is re-queued.
    assert client.put("/api/v1/settings", json={"image_understanding": "local"}).json()["requeued"] == 0
    assert system.library.calls == 1
    assert client.put("/api/v1/settings", json={"image_understanding": "gpu"}).status_code == 422

    restored = user_settings.apply_user_settings(Settings(data_dir=tmp_path))
    assert restored.image_understanding == "local"
