"""FastAPI app: document management, chat/SSE and health."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
import json
import logging
import mimetypes
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from typing import Annotated, Any, Iterator
import uuid

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import MissingApiKeyError, Settings, get_settings
from core import diagnostics, user_settings
from core.data_safety import DatabaseDamaged, RecoveryError
from core.rag_system import RAGSystem
from ingestion import ocr_runtime
from ingestion.images import load_pages as load_image_pages
from ingestion.slides import deck_preview, media_part
from services.model_clients import describe_model_error
from library import FILE_STATUSES, LibraryError
from library.registry import SEARCHABLE_STATUSES
from library.preview import (
    CONVERTED_IMAGES, PowerPointPdfConverter, PreviewError, PreviewPrefetcher, WordPdfConverter, preview_kind, preview_plan,
)
from retrieval.doc_cards import OTHER_TYPE, file_card, name_doc_type
from retrieval.file_filters import FILE_TYPES, parse_query, resolve_time
from suggestions import privacy as suggestion_privacy
from suggestions.service import overview as suggestion_overview
from .schemas import (
    BlockFileRequest, ChatRequest, ChatResponse, ChatScope, ConversationIn, ConversationsImport, DeleteDocumentResponse,
    ContentModeRequest, FolderModeRequest, FollowupRequest, ImageModeRequest, ImportRequest, RuleRequest, KeyTestRequest, PathRequest, PathsRequest, RestoreRequest,
    SettingsUpdate, SuggestionEvent,
)
from .errors import ErrorDetailMiddleware
from .security import TokenAuthMiddleware

settings = get_settings()
logger = logging.getLogger(__name__)
UI_DIR = Path(__file__).resolve().parents[1] / "ui" / "static"
system = RAGSystem(settings)


def _start_library() -> None:
    try:
        system.safety.prepare()
        system.safety.start_schedule()
        system.start_library()
    except DatabaseDamaged:
        logger.warning("File library not started: the index failed its check")
    except MissingApiKeyError:
        logger.info("File library waits for an API key")
    except Exception:
        logger.exception("File library failed to start; it will start on first use")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Reconciliation scans the watched folders, so it runs off the startup path.
    threading.Thread(target=_start_library, name="library-start", daemon=True).start()
    yield
    if system.library:
        system.library.stop()


app = FastAPI(
    lifespan=lifespan,
    title=settings.app_name,
    version=settings.app_version,
    description="寻寻的本机后台：LangGraph + 混合检索（SQLite 嵌入式）+ BGE 重排，回答并给出引用",
)
# Innermost: unhandled errors answer with their reason, still inside CORS (see errors.py).
app.add_middleware(ErrorDetailMiddleware, on_damage=lambda exc: system.safety.note_damage(exc))
# Added before CORS so it runs inside CORS: preflights are answered without a token, and 401s
# still carry CORS headers the desktop webview can read.
app.add_middleware(TokenAuthMiddleware, token_provider=lambda: settings.app_token)
app.add_middleware(
    CORSMiddleware,
    # The Tauri webview serves the UI from its own origin and calls this backend.
    allow_origins=[
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "http://tauri.localhost",
        "https://tauri.localhost",
        "tauri://localhost",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
    # Range requests let PDF.js load large PDFs page by page; it reads these headers.
    expose_headers=["Content-Disposition", "Accept-Ranges", "Content-Range", "Content-Length", "X-Preview-Cache"],
)


def _ready_system() -> RAGSystem:
    try:
        return system.initialize()
    except (MissingApiKeyError, DatabaseDamaged) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("System initialization failed")
        raise HTTPException(
            status_code=503,
            detail="系统依赖尚未就绪，请检查服务端日志。",
        ) from exc


def _health_payload(status: str) -> dict:
    return {
        "status": status,
        "version": settings.app_version,
        "models": {
            "llm": settings.llm_model,
            "embedding": settings.embedding_model,
            "reranker": settings.rerank_model,
        },
    }


@app.get("/api/v1/health")
def health():
    """Return a lightweight application status summary."""
    return _health_payload("ready" if system.initialized else "starting")


@app.get("/api/v1/health/live")
def health_live():
    """Liveness proves the HTTP process can serve requests without touching dependencies."""
    return {"status": "alive", "version": settings.app_version}


@app.get("/api/v1/health/ready")
def health_ready():
    """Readiness initializes the application and performs a lightweight vector-store query."""
    try:
        rag = system.initialize()
        if rag.store is None:
            raise RuntimeError("Vector store was not initialized")
        rag.store.list_documents()
    except Exception as exc:
        logger.exception("Readiness probe failed")
        raise HTTPException(
            status_code=503,
            detail="系统依赖尚未就绪，请检查服务端日志。",
        ) from exc
    payload = _health_payload("ready")
    backend = getattr(rag.store, "backend", settings.vector_backend)
    payload["checks"] = {"vector_store": "ok", "vector_backend": backend, "model_api": "configured"}
    return payload


def _library():
    rag = _ready_system()
    try:
        return rag.start_library()
    except Exception as exc:
        logger.exception("File library initialization failed")
        raise HTTPException(status_code=503, detail="文件库尚未就绪，请检查服务端日志。") from exc


def _local_only(request: Request) -> None:
    """Folder paths and Explorer refer to this machine; refuse them from other hosts."""
    host = request.client.host if request.client else ""
    if host not in ("127.0.0.1", "::1", "localhost", "testclient"):
        raise HTTPException(status_code=403, detail="只能在本机上管理文件夹。")


DEFAULT_QUICK_SEARCH_SHORTCUT = "Alt+Space"


def _settings_payload() -> dict:
    preferences = user_settings.load_preferences(settings)
    stored = user_settings.load_api_key()
    key = settings.siliconflow_api_key
    configured = bool(key) and not key.startswith("sk-your-")
    return {
        "api_key": {
            "configured": configured,
            "masked": user_settings.mask_key(key) if configured else "",
            "source": ("keyring" if stored and stored == key else "env") if configured else "",
        },
        "llm_model": settings.llm_model,
        "llm_model_choices": list(user_settings.LLM_MODEL_CHOICES),
        "embedding_model": settings.embedding_model,
        "rerank_model": settings.rerank_model,
        "image_understanding": settings.image_understanding,
        "image_import_mode": settings.image_import_mode,
        "ocr_runtime": ocr_runtime.installer(settings.data_dir).status(),
        "formula_runtime": ocr_runtime.installer(settings.data_dir, "formula").status(),
        "local_formula": settings.local_formula,
        "answer_mode": settings.answer_mode,
        "close_to_tray": user_settings.load_preferences(settings).get("close_to_tray", True) is not False,
        "quick_search_shortcut": user_settings.load_preferences(settings).get("quick_search_shortcut")
        or DEFAULT_QUICK_SEARCH_SHORTCUT,
        "vision_model": settings.vision_model,
        "vision_ocr_model": settings.vision_ocr_model,
        "data_dir": str(settings.data_dir),
        "log_dir": str(settings.log_dir),
        "database": str(settings.embedded_db_path),
        "desktop": bool(settings.app_token),
        "version": settings.app_version,
        "suggestions": _suggestion_settings(),
        "followups_enabled": _followups_enabled(),
        "data_safety": _safety_status(),
        # The guided start: what is left to show (static/onboarding.js decides from these).
        "onboarding_done": preferences.get("onboarding_done") is True,
        "data_notice_seen": preferences.get("data_notice_seen") is True,
    }


def _safety_status() -> dict:
    safety = getattr(system, "safety", None)
    return safety.status() if safety else {"state": "ok", "backups": []}


def _followups_enabled() -> bool:
    return user_settings.load_preferences(settings).get("followups_enabled", True) is not False


def _suggestions():
    return getattr(system, "suggestions", None)


def _suggestion_store():
    """The suggestion tables, also before the system is initialized (no API key yet)."""
    service = _suggestions()
    if service is not None:
        return service.store
    try:
        return getattr(system, "suggestion_store", None)
    except DatabaseDamaged:
        return None


def _suggestion_settings() -> dict:
    service = _suggestions()
    if service is not None:
        return service.overview()
    return suggestion_overview(settings, _suggestion_store())


@app.get("/api/v1/settings")
def get_app_settings():
    return _settings_payload()


@app.put("/api/v1/settings")
def update_app_settings(body: SettingsUpdate):
    if body.api_key is not None:
        key = body.api_key.strip()
        if any(char.isspace() for char in key):
            raise HTTPException(status_code=400, detail="API Key 不能包含空格。")
        try:
            user_settings.save_api_key(key)
        except user_settings.SecretStoreError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        settings.siliconflow_api_key = key
        if not system.initialized:
            # The first key: start indexing the watched folders now, not on first use.
            threading.Thread(target=_start_library, name="library-start", daemon=True).start()
    if body.llm_model is not None and body.llm_model.strip() != settings.llm_model:
        settings.llm_model = body.llm_model.strip()
        user_settings.save_preferences(settings, {"llm_model": settings.llm_model})
        system.reload_models()
    if body.answer_mode is not None and body.answer_mode != settings.answer_mode:
        # Read by the graph on every turn: applies to the next question.
        settings.answer_mode = body.answer_mode
        user_settings.save_preferences(settings, {"answer_mode": body.answer_mode})
    if body.close_to_tray is not None:
        # The desktop shell applies it; the backend remembers it for the next launch.
        user_settings.save_preferences(settings, {"close_to_tray": body.close_to_tray})
    suggestion_changes = {
        "enabled": body.suggestions_enabled,
        "personalize": body.suggestions_personalize,
        "ai_questions": body.suggestions_ai,
        "sensitive_words": body.suggestions_sensitive_words,
        "excluded_folders": body.suggestions_excluded_folders,
    }
    if any(value is not None for value in suggestion_changes.values()):
        suggestion_privacy.save_prefs(settings, **suggestion_changes)
        if _suggestions() is not None:
            # Sensitive words, folders and the AI switch change which files get questions.
            threading.Thread(target=_suggestions().sync, name="suggestions-sync", daemon=True).start()
    if body.followups_enabled is not None:
        user_settings.save_preferences(settings, {"followups_enabled": body.followups_enabled})
    if body.onboarding_done is not None:
        user_settings.save_preferences(settings, {"onboarding_done": body.onboarding_done})
    if body.data_notice_seen is not None:
        user_settings.save_preferences(settings, {"data_notice_seen": body.data_notice_seen})
    if body.image_import_mode is not None:
        _remember_image_mode(body.image_import_mode)
    if body.quick_search_shortcut is not None:
        # The desktop shell registers it; the backend only remembers the choice.
        user_settings.save_preferences(settings, {"quick_search_shortcut": body.quick_search_shortcut})
    requeued = 0
    if body.image_understanding == "local" and not ocr_runtime.activate(settings.data_dir):
        raise HTTPException(status_code=409, detail="请先下载本地识别组件。")
    if body.local_formula and not ocr_runtime.activate(settings.data_dir, "formula"):
        raise HTTPException(status_code=409, detail="请先下载本地公式识别组件。")
    image_changed = False
    if body.image_understanding is not None and body.image_understanding != settings.image_understanding:
        settings.image_understanding = body.image_understanding
        user_settings.save_preferences(settings, {"image_understanding": body.image_understanding})
        image_changed = True
    if body.local_formula is not None and body.local_formula != settings.local_formula:
        settings.local_formula = body.local_formula
        user_settings.save_preferences(settings, {"local_formula": body.local_formula})
        image_changed = True
    if image_changed:
        # PDFs indexed under the other mode are parsed again in the background.
        library = getattr(system, "library", None)
        requeued = library.requeue_outdated() if library else 0
    return {**_settings_payload(), "requeued": requeued}


def _component(component: str) -> str:
    if component not in ocr_runtime.COMPONENTS:
        raise HTTPException(status_code=404, detail=f"没有这个组件：{component}")
    return component


@app.get("/api/v1/ocr-runtime")
def ocr_runtime_status(component: str = "ocr"):
    return ocr_runtime.installer(settings.data_dir, _component(component)).status()


@app.post("/api/v1/ocr-runtime/install")
def install_ocr_runtime(component: str = "ocr"):
    """Download a local recognition component in the background; poll GET /ocr-runtime."""
    return ocr_runtime.installer(settings.data_dir, _component(component)).start()


@app.post("/api/v1/settings/test-key")
def test_api_key(body: KeyTestRequest):
    """Check a key against the model service before (or after) saving it."""
    key = (body.api_key or "").strip() or settings.siliconflow_api_key
    if not key or key.startswith("sk-your-"):
        raise HTTPException(status_code=400, detail="还没有可测试的 Key：请先粘贴或保存一个。")
    try:
        return user_settings.check_api_key(
            key, settings.siliconflow_base_url, settings.llm_model, embedding_model=settings.embedding_model
        )
    except user_settings.KeyCheckError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/v1/settings/api-key")
def delete_api_key():
    try:
        user_settings.delete_api_key()
    except user_settings.SecretStoreError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    # Fall back to SILICONFLOW_API_KEY from the environment / project/.env, if any. The
    # clients read the key per request, so the next request already goes without it.
    settings.siliconflow_api_key = Settings().siliconflow_api_key
    return _settings_payload()


@app.get("/api/v1/data-safety")
def data_safety_status():
    return system.safety.status()


@app.post("/api/v1/data-safety/backup")
def backup_now():
    system.safety.prepare()
    if system.safety.problem:
        raise HTTPException(status_code=409, detail="索引库已损坏，不能再备份。请先恢复。")
    backup = system.safety.backup(reason="manual")
    if backup is None:
        raise HTTPException(status_code=500, detail=system.safety.backup_note or "没有可备份的数据。")
    return system.safety.status()


def _after_recovery(result: dict) -> dict:
    if not result["restart_required"]:
        threading.Thread(target=_start_library, name="library-start", daemon=True).start()
    return {**result, "data_safety": system.safety.status()}


@app.post("/api/v1/data-safety/restore")
def restore_backup(body: RestoreRequest):
    """Put a backup back. When the databases are in use it waits for the next start
    (``restart_required``); the desktop shell can restart right away."""
    try:
        return _after_recovery(system.safety.restore(body.name))
    except RecoveryError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/v1/data-safety/rebuild")
def rebuild_index():
    """Start over from the file list: the index is emptied and every file is read again."""
    try:
        return _after_recovery(system.safety.rebuild())
    except RecoveryError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def _file_counts() -> dict:
    # Only when the library is already open: this must also work on a damaged index.
    library = getattr(system, "library", None)
    return library.registry.status_counts() if library else {}


def _show_in_explorer(path: Path) -> None:
    if sys.platform == "win32":
        # explorer parses "/select," itself; Windows paths cannot contain quotes.
        subprocess.Popen(f'explorer /select,"{path}"')


@app.post("/api/v1/diagnostics/export")
def export_diagnostics(request: Request):
    """关于 › 导出诊断信息: logs and a summary, without keys or document text. Shown in Explorer."""
    _local_only(request)
    path = diagnostics.export(settings, system.safety.status(), _file_counts())
    _show_in_explorer(path)
    return {"path": str(path), "size": path.stat().st_size}


@app.post("/api/v1/settings/open-data-dir")
def open_data_dir(request: Request):
    _local_only(request)
    if sys.platform != "win32":
        raise HTTPException(status_code=501, detail="仅支持 Windows。")
    os.startfile(settings.data_dir)  # noqa: S606 - opens our own data folder in Explorer
    return {"opened": str(settings.data_dir)}


@app.get("/api/v1/suggestions")
def get_suggestions(n: int = Query(4, ge=1, le=8), exclude: str = Query("", max_length=2000)):
    """Home-page questions from the local pool; see suggestions/sampler.py for how they are drawn."""
    prefs = suggestion_privacy.load_prefs(settings)
    if not prefs.enabled:
        return {"enabled": False, "ready": True, "file_count": 0, "library_empty": False, "cards": []}
    try:
        system.initialize()
    except MissingApiKeyError:
        return {"enabled": True, "ready": False, "file_count": 0, "library_empty": False, "cards": []}
    _library()
    payload = system.suggestions.suggest(n=n, exclude=[item for item in exclude.split(",") if item])
    return {**payload, "ready": True}


@app.post("/api/v1/suggestions/events")
def suggestion_events(body: SuggestionEvent):
    """Impressions, clicks and "不感兴趣" on suggestion cards: the only feedback recorded."""
    return {"updated": _suggestion_store().record(body.action, body.ids)}


@app.post("/api/v1/suggestions/blocked")
def block_suggestion_file(body: BlockFileRequest):
    """"不再推荐这个文件": the file stays searchable, it just gets no cards."""
    _ready_system()
    _library()
    try:
        return system.suggestions.block_file(body.doc_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="文件不存在。") from exc


@app.delete("/api/v1/suggestions/blocked/{doc_id}")
def unblock_suggestion_file(doc_id: str):
    _suggestion_store().unblock_file(doc_id)
    return _suggestion_settings()


@app.delete("/api/v1/suggestions/history")
def clear_suggestion_history():
    removed = _suggestion_store().clear_history()
    return {**_suggestion_settings(), "removed": removed}


@app.post("/api/v1/warm")
def warm():
    """The question box is being typed in: reopen idle connections to the model service now,
    so the question does not wait for them (at most every 20 s; free requests only)."""
    system.warm_up_soon()
    return {"ok": True}


@app.post("/api/v1/followups")
def followups(body: FollowupRequest):
    """Next questions after an answer, written from this turn only (see suggestions/followups.py).

    The UI asks once `final` has arrived and drops the request when the user starts typing
    or leaves the conversation; switched off in Settings, nothing is generated.
    """
    if not _followups_enabled():
        return {"enabled": False, "scenario": "", "items": []}
    if not system.initialized:
        try:
            # Answers from before a restart have their state too, so this may be the first request.
            system.initialize()
        except Exception as exc:
            logger.info("No follow-ups, the system is not ready: %s", exc)
            return {"enabled": True, "scenario": "", "items": []}
    return {"enabled": True, **system.followups(body.thread_id, body.question, n=body.n)}


@app.get("/api/v1/conversations")
def list_conversations():
    """The conversation history, newest first and without its messages."""
    return {"conversations": system.conversations.list()}


@app.post("/api/v1/conversations/import")
def import_conversations(body: ConversationsImport):
    """Conversations the UI kept in localStorage before they moved here. Safe to send twice;
    ``ids`` lists every one now stored, so the UI knows when its own copy may go."""
    return system.conversations.import_many(body.conversations)


@app.get("/api/v1/conversations/{conversation_id}")
def get_conversation(conversation_id: str):
    conversation = system.conversations.get(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="对话不存在。")
    return conversation


@app.put("/api/v1/conversations/{conversation_id}")
def save_conversation(conversation_id: str, body: ConversationIn):
    system.conversations.save({**body.model_dump(), "id": conversation_id})
    return {"id": conversation_id, "saved": True}


@app.delete("/api/v1/conversations/{conversation_id}")
def delete_conversation(conversation_id: str):
    """The conversation and the agent state behind it. An answer still running stops being stored."""
    return {"id": conversation_id, "deleted": system.delete_conversation(conversation_id)}


def _remember_question(question: str) -> None:
    if _suggestions() is not None:
        try:
            _suggestions().question_asked(question)
        except Exception as exc:
            logger.warning("Question not kept for suggestions: %s", exc)


@app.get("/api/v1/documents")
def list_documents():
    library = _library()
    return {
        "documents": [
            library.file_view(record)
            for record in library.registry.list_files()
            if library.file_view(record)["searchable"]
        ]
    }


@app.get("/api/v1/library/status")
def library_status(since: int | None = Query(None, ge=0)):
    """Queue progress, file counts and watched folders. With ``since`` (the ``changes.seq`` of
    the previous answer) also what changed meanwhile: new, moved, missing, recycled ... files."""
    library = _library()
    return {**library.status(since=since), "roots": library.list_roots()}


def _remember_image_mode(mode: str) -> None:
    settings.image_import_mode = mode
    user_settings.save_preferences(settings, {"image_import_mode": mode})


@app.post("/api/v1/library/inspect")
def inspect_import(body: PathsRequest, request: Request):
    """Before adding files/folders: counts by type, the images among them and an estimate,
    so the UI can ask once how to read the images. Registers nothing."""
    _local_only(request)
    return _library().inspect(body.paths)


@app.post("/api/v1/library/import")
def import_paths(body: ImportRequest, request: Request):
    """Add local files (registered in place, never copied) and folders (watched). Each path
    gets its own result; files are findable by name at once and read in the background."""
    _local_only(request)
    library = _library()
    if body.remember and body.image_mode:
        _remember_image_mode(body.image_mode)
    try:
        return library.import_paths(body.paths, image_mode=body.image_mode)
    except LibraryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/v1/library/sync")
def sync_library(request: Request):
    """Look at the disk now (the window regained focus): single files are stat-ed, lost ones
    looked up by file ID, watched folders checked. Cheap; repeated calls within 3 s are cached."""
    _local_only(request)
    return _library().sync()


@app.get("/api/v1/library/lost")
def lost_files(limit: int = Query(500, ge=1, le=5000)):
    """"找不到的文件": missing, offline (drive not connected) and in the Recycle Bin."""
    return {"files": _library().lost_files(limit=limit)}


@app.get("/api/v1/watch-roots")
def list_watch_roots():
    return {"roots": _library().list_roots()}


@app.post("/api/v1/watch-roots/preview")
def preview_watch_root(body: PathRequest, request: Request):
    _local_only(request)
    try:
        return _library().preview_root(body.path)
    except LibraryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/v1/watch-roots")
def add_watch_root(body: PathRequest, request: Request):
    _local_only(request)
    try:
        return _library().add_root(body.path)
    except LibraryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/v1/watch-roots/{root_id}/rescan")
def rescan_watch_root(root_id: str):
    try:
        return _library().rescan_root(root_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="监控目录不存在。") from exc


@app.delete("/api/v1/watch-roots/{root_id}")
def remove_watch_root(root_id: str):
    try:
        removed = _library().remove_root(root_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="监控目录不存在。") from exc
    return {"root_id": root_id, "removed_files": removed}


@app.get("/api/v1/files")
def list_files(
    root_id: str | None = None,
    status: str = "",
    q: str = "",
    limit: int = Query(500, ge=1, le=5000),
    offset: int = Query(0, ge=0),
    doc_type: str = "",
):
    library = _library()
    statuses = [item for item in status.split(",") if item]
    if any(item not in FILE_STATUSES for item in statuses):
        raise HTTPException(status_code=400, detail="未知的文件状态。")
    if doc_type:
        # Only files of this type (as the card and the file's own name say): the type is not
        # a registry column, so the matching files are found first, then paged.
        records = library.registry.list_files(root_id=root_id, statuses=statuses, search=q)
        cards = _file_cards(library, records)
        records = [record for record in records if (cards.get(record["file_id"]) or {}).get("type") == doc_type]
        total = len(records)
        records = records[offset : offset + limit]
    else:
        records = library.registry.list_files(root_id=root_id, statuses=statuses, limit=limit, offset=offset, search=q)
        cards = _file_cards(library, records)
        total = library.registry.count_files(root_id=root_id, statuses=statuses, search=q)
    files = []
    for record in records:
        view = library.file_view(record)
        view["card"] = cards.get(record["file_id"])
        files.append(view)
    return {"files": files, "total": total}


def _file_cards(library, records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Each file's type and document card for the library page (doc_cards.file_card)."""
    store = getattr(library, "store", None)
    stored: dict[str, dict[str, Any]] = {}
    if records and hasattr(store, "doc_cards"):
        try:
            stored = store.doc_cards(doc_ids=[record["file_id"] for record in records])
        except Exception as exc:
            logger.warning("Document cards unavailable for the file list: %s", exc)
    out = {}
    for record in records:
        card = file_card(str(record.get("name", "")), stored.get(record["file_id"]))
        if card:
            out[record["file_id"]] = card
    return out


@app.get("/api/v1/library/types")
def library_types():
    """Document types in the library with how many files each, for the type filter; and how
    many readable files still wait for their card (they are filled in in the background; a
    file whose own name gives the type needs none), with the time left at the current pace."""
    library = _library()
    records = library.registry.list_files(statuses=SEARCHABLE_STATUSES)
    cards = _file_cards(library, records)
    counts = Counter(card["type"] for card in cards.values())
    counts[OTHER_TYPE] += sum(1 for record in records if record["file_id"] not in cards)
    store = getattr(library, "store", None)
    enabled = bool(getattr(settings, "doc_cards", False))
    pending = 0
    if enabled and hasattr(store, "doc_card_hashes"):
        try:
            have = store.doc_card_hashes()
            pending = sum(1 for record in records
                          if record["file_id"] not in have and record.get("content_mode") != "name"
                          and not name_doc_type(str(record.get("name", ""))))
        except Exception as exc:
            logger.warning("Document cards unavailable for the type list: %s", exc)
    seconds_left = None
    builder = getattr(system, "doc_cards", None)
    if enabled and pending and builder is not None:
        try:
            pace = builder.progress()
            if pace.get("tokens_per_minute"):
                seconds_left = round(pending * pace["tokens_per_file"] / pace["tokens_per_minute"] * 60)
        except Exception as exc:
            logger.debug("Document card progress unavailable: %s", exc)
    types = sorted(((name, n) for name, n in counts.items() if n and name != OTHER_TYPE), key=lambda item: (-item[1], item[0]))
    if counts[OTHER_TYPE]:
        types.append((OTHER_TYPE, counts[OTHER_TYPE]))
    return {"types": [{"type": name, "count": n} for name, n in types], "pending": pending, "enabled": enabled,
            "seconds_left": seconds_left}


@app.post("/api/v1/files/{file_id}/relocate")
def relocate_file(file_id: str, body: PathRequest, request: Request):
    _local_only(request)
    library = _library()
    try:
        return library.file_view(library.relocate(file_id, body.path))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="文件不存在。") from exc
    except LibraryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/v1/files/{file_id}/reindex")
def reindex_file(file_id: str):
    """Queue a file to be parsed and embedded again, even if it has not changed."""
    library = _library()
    try:
        return library.file_view(library.reindex(file_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="文件不存在。") from exc


@app.post("/api/v1/files/{file_id}/check")
def check_file(file_id: str, request: Request):
    """Check one file against the disk before it is shown or opened (moved, recycled, changed)."""
    _local_only(request)
    library = _library()
    try:
        return library.file_view(library.check_file(file_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="文件不存在。") from exc


@app.post("/api/v1/files/{file_id}/image-mode")
def set_file_image_mode(file_id: str, body: ImageModeRequest):
    """Read one file's images another way (re-reads it): an image file, or a scanned PDF
    ("off" answers "no" to "这份 PDF 是扫描件，要识别文字吗？")."""
    library = _library()
    try:
        return library.file_view(library.set_image_mode(file_id, body.mode))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="文件不存在。") from exc
    except LibraryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/v1/files/{file_id}/content-mode")
def set_file_content_mode(file_id: str, body: ContentModeRequest):
    """只按文件名收录 one file ("name": its text leaves the index and the backups and is never
    read again), or read it again ("full"). With ``dry_run`` only says what would change."""
    library = _library()
    try:
        return library.set_content_mode(file_id, body.mode, dry_run=body.dry_run)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="文件不存在。") from exc
    except LibraryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/v1/library/folder-mode")
def set_folder_content_mode(body: FolderModeRequest, request: Request):
    """The same for a watched folder or one of its sub-folders, files added later included."""
    _local_only(request)
    try:
        return _library().set_folder_mode(body.path, body.mode, dry_run=body.dry_run)
    except LibraryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/v1/library/privacy")
def library_privacy():
    """What the library may not read: every rule, the files kept by name only, the last scrub."""
    library = _library()
    scrubber = getattr(system, "scrubber", None)
    return {**library.privacy_summary(), "scrub": scrubber.status() if scrubber is not None else None}


@app.post("/api/v1/library/rules")
def add_library_rule(body: RuleRequest, request: Request):
    """Exclude a folder, a file or a name pattern; or keep a name pattern by name only. Files
    now excluded leave the library and their text is purged. ``dry_run`` counts them."""
    _local_only(request)
    if bool(body.path) == bool(body.pattern):
        raise HTTPException(status_code=400, detail="请给出一个路径或一个名字。")
    try:
        return _library().add_rule(
            action=body.action, path=body.path, pattern=body.pattern, root_id=body.root_id, dry_run=body.dry_run
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="监控目录不存在。") from exc
    except LibraryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/v1/library/rules/{rule_id}")
def remove_library_rule(rule_id: str, dry_run: bool = False):
    """Take a rule back. Files it kept out are added and read again (``files`` says how many)."""
    try:
        return _library().remove_rule(rule_id, dry_run=dry_run)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="规则不存在。") from exc
    except LibraryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/v1/files/{file_id}/reveal")
def reveal_file(file_id: str, request: Request):
    _local_only(request)
    try:
        return {"opened": _library().reveal(file_id)}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="文件不存在。") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="原文件及其所在文件夹都已不存在。") from exc


@app.get("/api/v1/files/search")
def search_files(
    q: str = Query("", max_length=500),
    limit: int = Query(8, ge=1, le=200),
    types: str = "",
    folder: str = Query("", max_length=4096),
    time: str = Query("", max_length=40),
    time_field: str = Query("", pattern="^(|added|modified|any)$"),
    doc_ids: str = "",
):
    """Find files: rules parse the query (no LLM); explicit parameters override them."""
    rag = _ready_system()
    _library()
    filters = parse_query(q)
    labels = [label for label in types.split(",") if label]
    if any(label not in FILE_TYPES for label in labels):
        raise HTTPException(status_code=400, detail="未知的文件类型。")
    for label in labels:
        if label not in filters.file_types:
            filters.file_types.append(label)
    if folder:
        filters.folder = folder
    if time:
        resolved = resolve_time(time)
        if resolved is None:
            raise HTTPException(status_code=400, detail="无法识别的时间条件。")
        filters.time_range, filters.time_phrase = resolved, time
    if time_field:
        filters.time_field = time_field
    if not (q.strip() or filters.has_metadata):
        return {"query": q, "filters": filters.to_dict(), "elapsed_ms": 0, "total": 0, "total_candidates": 0, "unreadable": 0, "files": []}
    scope = [item for item in doc_ids.split(",") if item] or None
    return rag.search_files(q, limit=limit, filters=filters, scope=scope)


@app.post("/api/v1/files/{file_id}/open")
def open_file(file_id: str, request: Request):
    """Open the original with its default program (this machine only)."""
    _local_only(request)
    record = _library().registry.get_file(file_id)
    if record is None:
        raise HTTPException(status_code=404, detail="文件不存在。")
    path = record["path"]
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="原文件已不存在，可以在阅读面板中查看已索引的内容。")
    _looked_at(file_id)
    if sys.platform == "win32":
        os.startfile(path)  # noqa: S606 - a registered file the user asked to open
    else:
        import subprocess

        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", path])
    return {"opened": path}


@app.delete("/api/v1/files/{file_id}", response_model=DeleteDocumentResponse)
def remove_file(file_id: str):
    library = _library()
    if library.registry.get_file(file_id) is None:
        raise HTTPException(status_code=404, detail="文件不存在。")
    library.remove_file(file_id)
    return DeleteDocumentResponse(doc_id=file_id, deleted=True)


@app.post("/api/v1/documents")
def upload_documents(files: list[UploadFile] = File(...), image_mode: Annotated[str | None, Form()] = None):
    """Import single files from the browser. The browser cannot reveal a local path, so
    unlike watched folders these are copied to ``upload_dir/<file_id>/``. Indexing runs
    in the background; poll ``/library/status`` or ``/files`` for progress. ``image_mode``
    (local / cloud / name) is how uploaded images are read."""
    library = _library()
    if image_mode is not None and image_mode not in ("local", "cloud", "name"):
        raise HTTPException(status_code=400, detail="未知的图片识别方式。")
    results = []
    for upload in files:
        filename = Path(upload.filename or "").name
        suffix = Path(filename).suffix.lower()
        if not filename or suffix not in settings.allowed_extensions:
            raise HTTPException(status_code=400, detail=f"不支持的文件：{filename}")
        if upload.size is not None and upload.size > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"{filename} 超过 {settings.max_upload_mb} MB 限制")
        # Stage under a unique directory, then file the copy under its new file_id:
        # same-named uploads never overwrite each other.
        staging = settings.upload_dir / ".staging" / uuid.uuid4().hex
        staged = staging / filename
        try:
            staging.mkdir(parents=True)
            written = 0
            with staged.open("wb") as target:
                while block := upload.file.read(1024 * 1024):
                    written += len(block)
                    if written > settings.max_upload_mb * 1024 * 1024:
                        raise HTTPException(
                            status_code=413,
                            detail=f"{filename} 超过 {settings.max_upload_mb} MB 限制",
                        )
                    target.write(block)
            record = library.import_upload(staged, filename, image_mode=image_mode)
            results.append(
                {"doc_id": record["file_id"], "source": filename, "chunks": 0, "status": record["status"]}
            )
        except Exception as exc:
            if isinstance(exc, HTTPException):
                raise
            logger.exception("Document import failed: %s", filename)
            raise HTTPException(
                status_code=422,
                detail=f"{filename} 导入失败，请检查文件格式或服务端日志。",
            ) from exc
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    return {"results": results}


_word_converter: WordPdfConverter | None = None
_word_converter_lock = threading.Lock()


def word_converter() -> WordPdfConverter:
    global _word_converter
    with _word_converter_lock:
        if _word_converter is None:
            _word_converter = WordPdfConverter(settings.data_dir / "preview_cache" / "word")
        return _word_converter


_slides_converter: PowerPointPdfConverter | None = None


def slides_converter() -> PowerPointPdfConverter:
    global _slides_converter
    with _word_converter_lock:
        if _slides_converter is None:
            _slides_converter = PowerPointPdfConverter(settings.data_dir / "preview_cache" / "slides")
        return _slides_converter


_prefetcher: PreviewPrefetcher | None = None
_slides_prefetcher: PreviewPrefetcher | None = None
RECENT_WORD_DAYS = 14
WORD_KINDS = ("docx", "doc")
SLIDE_KINDS = ("pptx", "ppt")


def _recent_word_files() -> list[str]:
    """Word files added to the library lately, newest first: converted while the app is idle."""
    return _recent_files(".doc", WORD_KINDS)


def _recent_slide_files() -> list[str]:
    return _recent_files(".ppt", SLIDE_KINDS)


def _recent_files(search: str, kinds: tuple[str, ...]) -> list[str]:
    library = system.library
    if library is None:
        return []
    cutoff = time.time() - RECENT_WORD_DAYS * 86400
    records = [
        r for r in library.registry.list_files(statuses=("indexed",), search=search)
        if preview_kind(Path(r["path"]).suffix) in kinds
    ]

    def added(record) -> float:
        value = record.get("created_at")
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return datetime.fromisoformat(str(value)).timestamp()
        except ValueError:
            return 0.0

    records = [r for r in records if added(r) >= cutoff]
    records.sort(key=added, reverse=True)
    return [r["file_id"] for r in records[:20]]


def preview_prefetcher() -> PreviewPrefetcher:
    global _prefetcher
    converter = word_converter()
    with _word_converter_lock:
        if _prefetcher is None:
            _prefetcher = PreviewPrefetcher(converter, lambda doc_id: system.original_file(doc_id, ""), _recent_word_files)
        _prefetcher.converter = converter
        return _prefetcher


def slides_prefetcher() -> PreviewPrefetcher:
    global _slides_prefetcher
    converter = slides_converter()
    with _word_converter_lock:
        if _slides_prefetcher is None:
            _slides_prefetcher = PreviewPrefetcher(
                converter, lambda doc_id: system.original_file(doc_id, ""), _recent_slide_files, kinds=SLIDE_KINDS,
            )
        _slides_prefetcher.converter = converter
        return _slides_prefetcher


class PrepareRequest(BaseModel):
    doc_ids: list[str] = Field(default_factory=list, max_length=50)


def _unparsed_document(doc_id: str) -> dict | None:
    """A registered file without parsed text (failed, unsupported, still queued): the reader
    can still show the original, which is what tells the user whether it is the right file."""
    record = system.library.registry.get_file(doc_id) if system.library else None
    if record is None:
        return None
    return {
        "doc_id": doc_id,
        "source": record["name"],
        "file_type": Path(record["name"]).suffix.lower(),
        "units": [],
        "parents": [],
        "characters": 0,
        "image_only_pages": [],
        "has_file": system.original_file(doc_id, record["name"]) is not None,
        "path": record["path"],
        "status": record["status"],
        "size": record["size"],
        "error": record.get("error"),
        "image_mode": record.get("image_mode"),
        "notice": record.get("notice"),
        "parsed": False,
    }


def _looked_at(doc_id: str) -> None:
    """A file opened or read in the app gets its document card first."""
    cards = getattr(system, "doc_cards", None)
    if cards is not None and getattr(settings, "doc_cards", False):
        try:
            cards.prioritize([doc_id])
        except Exception as exc:
            logger.debug("Document card not queued for %s: %s", doc_id, exc)


@app.get("/api/v1/documents/{doc_id}")
def get_document(doc_id: str):
    document = system.get_document(doc_id) or _unparsed_document(doc_id)
    if document is None:
        raise HTTPException(status_code=404, detail="文档不存在或尚未完成解析。")
    _looked_at(doc_id)
    ext = Path(document.get("path") or document["source"]).suffix or document.get("file_type", "")
    kind = preview_kind(ext)
    word = kind in WORD_KINDS and document["has_file"] and word_converter().available()
    powerpoint = kind in SLIDE_KINDS and document["has_file"] and slides_converter().available()
    document["preview"] = preview_plan(ext, document["has_file"], word, powerpoint)
    return document


def _original_or_404(doc_id: str) -> Path:
    document = system.get_document(doc_id) or _unparsed_document(doc_id)
    path = system.original_file(doc_id, document["source"]) if document else None
    if path is None:
        raise HTTPException(status_code=404, detail="没有可读取的原文件。")
    return path


@app.get("/api/v1/documents/{doc_id}/file")
def download_document(doc_id: str, inline: bool = False):
    path = _original_or_404(doc_id)
    return FileResponse(path, filename=path.name, content_disposition_type="inline" if inline else "attachment")


@app.get("/api/v1/documents/{doc_id}/image")
def document_image(doc_id: str):
    """An image file as the reader shows it: the original, or a PNG of a TIFF's first page."""
    path = _original_or_404(doc_id)
    if preview_kind(path.suffix) != "image":
        raise HTTPException(status_code=400, detail="这不是图片文件。")
    if path.suffix.lower() not in CONVERTED_IMAGES:
        return FileResponse(path, content_disposition_type="inline")
    try:
        data, mime = load_image_pages(path, 2400)[0]
    except Exception as exc:
        raise HTTPException(status_code=422, detail="暂时无法显示这张图片。") from exc
    return Response(data, media_type=mime)


@app.post("/api/v1/documents/prepare")
def prepare_documents(body: PrepareRequest):
    """Files about to be shown (answer cards, find-file rows): convert Word and PowerPoint
    originals in the background, one at a time, so opening them later does not wait. Also
    starts the idle pre-conversion of recently added ones."""
    words, slides = preview_prefetcher(), slides_prefetcher()
    words.start()
    slides.start()
    states = {}
    for doc_id in dict.fromkeys(body.doc_ids):
        state = words.request(doc_id)
        states[doc_id] = state if state != "skip" else slides.request(doc_id)
    return {"states": states}


@app.get("/api/v1/documents/{doc_id}/pdf")
def document_as_pdf(doc_id: str, wait: float = Query(0, ge=0, le=10)):
    """A Word or PowerPoint original converted to PDF by the local application (cached per
    file version).

    With ``wait``, the conversion goes to the front of the background queue and the request
    waits at most that many seconds: 202 means it is still running (the reader shows another
    rendering meanwhile; the next open gets the PDF)."""
    path = _original_or_404(doc_id)
    kind = preview_kind(path.suffix)
    if kind not in (*WORD_KINDS, *SLIDE_KINDS):
        raise HTTPException(status_code=400, detail="只有 Word 和 PowerPoint 文件需要转换。")
    converter = word_converter() if kind in WORD_KINDS else slides_converter()
    started = time.perf_counter()
    if wait:
        if not converter.cached(path):
            if not converter.available():
                raise HTTPException(status_code=409, detail=converter.missing)
            reason = converter.failure(path)
            if reason:
                raise HTTPException(status_code=422, detail=reason)
            prefetcher = preview_prefetcher() if kind in WORD_KINDS else slides_prefetcher()
            prefetcher.request(doc_id, urgent=True)
            if prefetcher.wait(path, wait) is None:
                return JSONResponse({"state": "preparing"}, status_code=202)
    try:
        pdf, hit = converter.convert(path)
    except PreviewError as exc:
        raise HTTPException(status_code=exc.status, detail=exc.reason) from exc
    elapsed = int((time.perf_counter() - started) * 1000)
    logger.info("%s preview %s: %s in %d ms", converter.application, "cache hit" if hit else "converted", path.name, elapsed)
    return FileResponse(
        pdf,
        media_type="application/pdf",
        headers={"X-Preview-Cache": f"{'hit' if hit else 'miss'}; {elapsed} ms"},
    )


@app.get("/api/v1/documents/{doc_id}/slides")
def document_slides(doc_id: str):
    """A deck's slides as shapes, for the reader when PowerPoint cannot convert it."""
    path = _original_or_404(doc_id)
    if preview_kind(path.suffix) != "pptx":
        raise HTTPException(status_code=400, detail="只有 .pptx 文件可以这样显示。")
    try:
        return deck_preview(str(path))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/v1/documents/{doc_id}/slides/media")
def document_slide_media(doc_id: str, part: str = Query(..., max_length=300)):
    """One picture of a deck, from under ppt/media/ only."""
    path = _original_or_404(doc_id)
    if preview_kind(path.suffix) != "pptx":
        raise HTTPException(status_code=400, detail="只有 .pptx 文件可以这样显示。")
    try:
        data = media_part(str(path), part)
    except (KeyError, OSError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="没有这张图片。") from exc
    mime = mimetypes.guess_type(part)[0] or "application/octet-stream"
    if not mime.startswith("image/"):
        raise HTTPException(status_code=404, detail="没有这张图片。")
    return Response(data, media_type=mime, headers={"Cache-Control": "private, max-age=3600"})


@app.delete("/api/v1/documents/{doc_id}", response_model=DeleteDocumentResponse)
def delete_document(doc_id: str):
    rag = _ready_system()
    rag.delete_document(doc_id)
    return DeleteDocumentResponse(doc_id=doc_id, deleted=True)


def _chat_scope(rag: RAGSystem, scope: ChatScope | None) -> list[str] | None:
    if scope is None or not (scope.doc_ids or scope.folder):
        return None
    if scope.folder:
        _library()
    try:
        return rag.resolve_scope(scope.doc_ids, scope.folder)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/v1/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    # The answer's timings start here, so a wait for start-up is counted too.
    received = time.perf_counter()
    rag = _ready_system()
    thread_id = request.thread_id or rag.new_thread_id()
    scope = _chat_scope(rag, request.scope)
    try:
        return ChatResponse(**rag.chat(request.question, thread_id, scope=scope, received_at=received))
    finally:
        # After the answer: the suggestion store may be busy with the background builder.
        _remember_question(request.question)


@app.post("/api/v1/chat/stream")
def chat_stream(request: ChatRequest):
    received = time.perf_counter()
    rag = _ready_system()
    thread_id = request.thread_id or rag.new_thread_id()
    scope = _chat_scope(rag, request.scope)
    return _sse_response(rag, request.question, thread_id, scope, received_at=received)


def _sse_response(
    rag: RAGSystem, question: str, thread_id: str, scope: list[str] | None = None, received_at: float | None = None
) -> StreamingResponse:

    def events() -> Iterator[str]:
        try:
            options: dict = {}
            if scope:
                options["scope"] = scope
            if received_at is not None:
                options["received_at"] = received_at
            stream = rag.stream(question, thread_id, **options)
            for event in stream:
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:
            logger.exception("Streaming chat failed")
            error = {
                "event": "error",
                "message": describe_model_error(exc),
                "thread_id": thread_id,
            }
            yield f"data: {json.dumps(error, ensure_ascii=False)}\n\n"
        finally:
            # After the answer: the suggestion store may be busy with the background builder,
            # and the answer must not wait for it.
            _remember_question(question)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/", include_in_schema=False)
def index():
    return RedirectResponse("/ui/")


app.mount("/ui", StaticFiles(directory=UI_DIR, html=True), name="ui")
