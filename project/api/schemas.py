from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatScope(BaseModel):
    """Ask within these files and/or every indexed file under a folder."""

    doc_ids: list[str] = Field(default_factory=list, max_length=500)
    folder: str = Field(default="", max_length=4096)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    thread_id: str | None = None
    scope: ChatScope | None = None


class ConversationIn(BaseModel):
    """A conversation as the UI keeps it; its messages are stored as they are."""

    title: str = ""
    updated: float = 0
    scope: dict[str, Any] | None = None
    messages: list[Any] = Field(default_factory=list)


class ConversationsImport(BaseModel):
    conversations: list[dict[str, Any]] = Field(default_factory=list)


class ChatResponse(BaseModel):
    thread_id: str
    answer: str
    intent: str
    rewritten_query: str
    citations: list[dict[str, str]]
    grounded: bool
    refused: bool
    retrieval_attempts: int
    # True when the answer is shown unverified because the fact checker was unreachable.
    fact_check_skipped: bool = False
    check_mode: str = ""
    # find_file answers: ranked files with their best passages.
    files: list[dict[str, Any]] = Field(default_factory=list)
    file_search: dict[str, Any] = Field(default_factory=dict)
    # Questions about a whole file: what was read of it.
    coverage: dict[str, Any] = Field(default_factory=dict)
    # Where the answer's time went (core/timings.py); numbers only.
    timings: dict[str, Any] = Field(default_factory=dict)


class PathRequest(BaseModel):
    path: str = Field(min_length=1, max_length=4096)


class PathsRequest(BaseModel):
    """Local files and/or folders, e.g. picked in the file dialog or dropped on the window."""

    paths: list[str] = Field(min_length=1, max_length=1000)


class ImportRequest(PathsRequest):
    # How images among them are read; left out: the remembered choice, else local OCR.
    image_mode: Literal["local", "cloud", "name"] | None = None
    # "记住我的选择": later imports use image_mode without asking.
    remember: bool = False


class ImageModeRequest(BaseModel):
    """An image file: local / cloud / name. A scanned PDF: local / cloud, or off (don't read)."""

    mode: Literal["local", "cloud", "name", "off"]


class ContentModeRequest(BaseModel):
    """How a file or folder is kept: "name" (只按文件名收录: name, folder and dates only, nothing
    read or uploaded) or "full" (read and indexed). ``dry_run`` only reports what would change."""

    mode: Literal["name", "full"]
    dry_run: bool = False


class FolderModeRequest(ContentModeRequest):
    path: str = Field(..., min_length=1, max_length=4096)


class RuleRequest(BaseModel):
    """A new rule: exclude a folder or file inside a watched folder (``path``), or exclude / keep
    by name only every file or folder with this name (``pattern``; ``root_id`` one watched
    folder, None all of them)."""

    action: Literal["name", "exclude"]
    path: str = Field("", max_length=4096)
    pattern: str = Field("", max_length=200)
    root_id: str | None = Field(None, max_length=64)
    dry_run: bool = False


class DeleteDocumentResponse(BaseModel):
    doc_id: str
    deleted: bool


class SettingsUpdate(BaseModel):
    """Fields left out (None) are unchanged; an empty api_key is rejected."""

    api_key: str | None = Field(default=None, min_length=8, max_length=256)
    llm_model: str | None = Field(default=None, min_length=1, max_length=200)
    image_understanding: Literal["off", "local", "cloud"] | None = None
    # Image files: ask once per import, or always read them this way.
    image_import_mode: Literal["ask", "local", "cloud", "name"] | None = None
    local_formula: bool | None = None
    answer_mode: Literal["fast", "strict"] | None = None
    # Desktop: closing the main window hides it to the tray instead of quitting.
    close_to_tray: bool | None = None
    # Global shortcut of the desktop quick-search window, e.g. "Alt+Space".
    quick_search_shortcut: str | None = Field(
        default=None,
        pattern=r"^(?:(?:Ctrl|Alt|Shift|Super)\+){1,3}(?:[A-Z0-9]|Space|F(?:[1-9]|1[0-2])|Enter|Tab|Backquote)$",
    )
    # Home-page suggestions ("首页推荐").
    suggestions_enabled: bool | None = None
    suggestions_personalize: Literal["history", "library"] | None = None
    suggestions_ai: bool | None = None
    suggestions_sensitive_words: list[str] | None = Field(default=None, max_length=200)
    suggestions_excluded_folders: list[str] | None = Field(default=None, max_length=100)
    # "回答后推荐追问": the likeliest next question in the input box, alternatives below the answer.
    followups_enabled: bool | None = None
    # The guided start was finished or skipped; the note on where data goes was read.
    onboarding_done: bool | None = None
    data_notice_seen: bool | None = None


class RestoreRequest(BaseModel):
    name: str = Field(min_length=1, max_length=64, pattern=r"^[0-9-]+$")


class KeyTestRequest(BaseModel):
    """The key typed on the Settings page; left out, the saved key is tested."""

    api_key: str | None = Field(default=None, max_length=256)


class FollowupRequest(BaseModel):
    """Follow-ups for the last finished turn of ``thread_id``; ``question`` is that turn's question."""

    thread_id: str = Field(min_length=1, max_length=100)
    question: str = Field(default="", max_length=2000)
    n: int = Field(default=4, ge=1, le=6)


class SuggestionEvent(BaseModel):
    action: Literal["shown", "click", "dismiss"]
    ids: list[str] = Field(min_length=1, max_length=50)


class BlockFileRequest(BaseModel):
    doc_id: str = Field(min_length=24, max_length=32, pattern=r"^[0-9a-f]+$")
