"""Settings the user changes from the UI: the SiliconFlow key, chat model and image mode.

The API key goes to the OS credential store through ``keyring`` (Windows Credential
Manager on Windows) and is never written to a plain-text file. Non-secret preferences
live in ``<data_dir>/preferences.json``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from config import APP_DATA_NAME, Settings


logger = logging.getLogger(__name__)

KEYRING_SERVICE = APP_DATA_NAME
KEYRING_USERNAME = "siliconflow_api_key"
PREFERENCES_FILE = "preferences.json"

# Offered on the Settings page; any other SiliconFlow chat model id can be typed in.
LLM_MODEL_CHOICES = (
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen2.5-72B-Instruct",
    "deepseek-ai/DeepSeek-V3",
    "deepseek-ai/DeepSeek-V4-Flash",
    "THUDM/GLM-4-9B-0414",
)


# "off" | "local" (RapidOCR on this machine) | "cloud" (SiliconFlow vision models).
IMAGE_MODES = ("off", "local", "cloud")
# Image files: "ask" once per import, or always "local" / "cloud" / "name" (file name only).
IMAGE_IMPORT_MODES = ("ask", "local", "cloud", "name")
# "fast" checks only the citations; "strict" also has a model check each cited statement.
ANSWER_MODES = ("fast", "strict")
# Reaching the model service: "auto" (direct when it connects, else the system proxy), "direct", "proxy".
MODEL_CONNECTIONS = ("auto", "direct", "proxy")


class SecretStoreError(RuntimeError):
    """The OS credential store is unavailable or refused the operation."""


def load_api_key() -> str:
    """Return the stored key, or "" when none is stored or the store is unavailable."""
    try:
        import keyring

        return keyring.get_password(KEYRING_SERVICE, KEYRING_USERNAME) or ""
    except Exception:
        logger.warning("Credential store unavailable; falling back to SILICONFLOW_API_KEY", exc_info=True)
        return ""


def save_api_key(value: str) -> None:
    try:
        import keyring

        keyring.set_password(KEYRING_SERVICE, KEYRING_USERNAME, value)
    except Exception as exc:
        raise SecretStoreError("无法写入系统凭据管理器。") from exc


def delete_api_key() -> None:
    try:
        import keyring
        from keyring.errors import PasswordDeleteError

        try:
            keyring.delete_password(KEYRING_SERVICE, KEYRING_USERNAME)
        except PasswordDeleteError:
            pass
    except Exception as exc:
        raise SecretStoreError("无法访问系统凭据管理器。") from exc


class KeyCheckError(RuntimeError):
    """The key could not be confirmed; the message says why, for the Settings page."""


def check_api_key(
    key: str, base_url: str, model: str = "", *, embedding_model: str = "", transport=None, timeout: float = 10.0
) -> dict[str, Any]:
    """Ask the model service which chat models the key may use. Answering at all proves the
    key is valid; the list also tells whether the chosen chat model is among them. Listing
    works on an empty account, so with ``embedding_model`` one word is also embedded (the
    first thing indexing does, and it costs next to nothing)."""
    import time

    import httpx

    def reject(status: int) -> None:
        if status in (401, 403):
            raise KeyCheckError("Key 无效或已被停用。")
        if status == 402:
            raise KeyCheckError("Key 有效，但硅基流动账户余额不足，请先充值。")
        if status >= 400:
            raise KeyCheckError(f"模型服务返回 {status}，请稍后再试。")

    started = time.perf_counter()
    headers = {"Authorization": f"Bearer {key}"}
    try:
        with httpx.Client(base_url=base_url.rstrip("/") + "/", timeout=timeout, transport=transport) as client:
            res = client.get("models", params={"type": "text", "sub_type": "chat"}, headers=headers)
            reject(res.status_code)
            if embedding_model:
                reject(client.post("embeddings", headers=headers, json={"model": embedding_model, "input": ["你好"]}).status_code)
    except httpx.HTTPError as exc:
        raise KeyCheckError(f"连不上模型服务（{type(exc).__name__}），请检查网络或代理。") from exc
    try:
        items = res.json().get("data") or []
    except ValueError:
        items = []
    ids = {item.get("id") for item in items if isinstance(item, dict)}
    return {
        "ok": True,
        "elapsed_ms": round((time.perf_counter() - started) * 1000),
        "model": model,
        "model_available": (model in ids) if model and ids else None,
    }


def mask_key(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 10:
        return "•" * len(value)
    return f"{value[:5]}…{value[-4:]}"


def preferences_path(settings: Settings) -> Path:
    return Path(settings.data_dir) / PREFERENCES_FILE


def load_preferences(settings: Settings) -> dict[str, Any]:
    path = preferences_path(settings)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError):
        logger.warning("Ignoring unreadable %s", path, exc_info=True)
        return {}
    return payload if isinstance(payload, dict) else {}


def save_preferences(settings: Settings, updates: dict[str, Any]) -> dict[str, Any]:
    preferences = {**load_preferences(settings), **updates}
    path = preferences_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(preferences, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return preferences


def apply_user_settings(settings: Settings) -> Settings:
    """Overlay the stored key and preferences onto settings built from env/.env."""
    key = load_api_key()
    if key:
        settings.siliconflow_api_key = key
    preferences = load_preferences(settings)
    model = preferences.get("llm_model")
    if isinstance(model, str) and model.strip():
        settings.llm_model = model.strip()
    mode = preferences.get("image_understanding")
    if mode in IMAGE_MODES:
        settings.image_understanding = mode
    if preferences.get("image_import_mode") in IMAGE_IMPORT_MODES:
        settings.image_import_mode = preferences["image_import_mode"]
    if isinstance(preferences.get("local_formula"), bool):
        settings.local_formula = preferences["local_formula"]
    if preferences.get("answer_mode") in ANSWER_MODES:
        settings.answer_mode = preferences["answer_mode"]
    if preferences.get("model_connection") in MODEL_CONNECTIONS:
        settings.model_connection = preferences["model_connection"]
    for switch in ("doc_cards", "doc_type_search", "doc_card_owner"):
        if isinstance(preferences.get(switch), bool):
            setattr(settings, switch, preferences[switch])
    owner = preferences.get("owner_name")
    if isinstance(owner, str) and owner.strip():
        settings.owner_name = owner.strip()[:40]
    return settings
