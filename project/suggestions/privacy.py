"""What the home-page suggestions may use, and which files they leave alone.

Suggestions are kept modest on purpose:

* the only personal signals are questions the user sent and what they did with the
  suggestion cards (click, "不感兴趣", "不再推荐这个文件") — never which files were
  opened, for how long, or in what order;
* files whose name or path contains a sensitive word (体检, 工资, 合同 …) or that lie in
  an excluded folder get no suggested questions; they stay searchable as before;
* nothing leaves the machine unless the user turns on AI-written questions.

The settings live in ``preferences.json`` next to the other UI preferences.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import re
from typing import Any, Iterable

from config import Settings
from core import user_settings

# "A&B" matches only when both words appear: company test reports are fine to suggest,
# a personal medical one is not.
DEFAULT_SENSITIVE_WORDS: tuple[str, ...] = (
    "体检", "病历", "诊断", "身份证", "护照", "工资", "薪资", "合同", "简历", "银行", "密码",
    "离婚", "律师函", "检测报告&个人", "检测报告&姓名", "化验", "核酸",
)
PERSONALIZE_MODES = ("history", "library")  # "同时参考我的提问" | "仅根据知识库内容"
MAX_WORDS = 200
MAX_WORD_LENGTH = 40
MAX_EXCLUDED_FOLDERS = 100

_PREFIX = "suggestions_"


@dataclass
class SuggestionPrefs:
    enabled: bool = True
    personalize: str = "history"
    ai_questions: bool = False
    sensitive_words: list[str] = field(default_factory=lambda: list(DEFAULT_SENSITIVE_WORDS))
    excluded_folders: list[str] = field(default_factory=list)

    @property
    def uses_history(self) -> bool:
        return self.enabled and self.personalize == "history"

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "personalize": self.personalize,
            "ai_questions": self.ai_questions,
            "sensitive_words": list(self.sensitive_words),
            "default_sensitive_words": list(DEFAULT_SENSITIVE_WORDS),
            "excluded_folders": list(self.excluded_folders),
        }


def clean_words(words: Iterable[str]) -> list[str]:
    seen: list[str] = []
    for word in words:
        word = re.sub(r"\s+", "", str(word))[:MAX_WORD_LENGTH].strip("&")
        if word and word not in seen:
            seen.append(word)
    return seen[:MAX_WORDS]


def clean_folders(folders: Iterable[str]) -> list[str]:
    seen: list[str] = []
    for folder in folders:
        folder = str(folder).strip().strip('"')
        if folder:
            folder = os.path.normpath(folder)
            if folder not in seen:
                seen.append(folder)
    return seen[:MAX_EXCLUDED_FOLDERS]


def load_prefs(settings: Settings) -> SuggestionPrefs:
    stored = user_settings.load_preferences(settings)
    prefs = SuggestionPrefs()
    if isinstance(stored.get(_PREFIX + "enabled"), bool):
        prefs.enabled = stored[_PREFIX + "enabled"]
    if stored.get(_PREFIX + "personalize") in PERSONALIZE_MODES:
        prefs.personalize = stored[_PREFIX + "personalize"]
    if isinstance(stored.get(_PREFIX + "ai_questions"), bool):
        prefs.ai_questions = stored[_PREFIX + "ai_questions"]
    if isinstance(stored.get(_PREFIX + "sensitive_words"), list):
        prefs.sensitive_words = clean_words(stored[_PREFIX + "sensitive_words"])
    if isinstance(stored.get(_PREFIX + "excluded_folders"), list):
        prefs.excluded_folders = clean_folders(stored[_PREFIX + "excluded_folders"])
    return prefs


def save_prefs(settings: Settings, **changes: Any) -> SuggestionPrefs:
    updates: dict[str, Any] = {}
    for key, value in changes.items():
        if value is None:
            continue
        if key == "sensitive_words":
            value = clean_words(value)
        elif key == "excluded_folders":
            value = clean_folders(value)
        elif key == "personalize" and value not in PERSONALIZE_MODES:
            raise ValueError(f"unknown personalization: {value}")
        updates[_PREFIX + key] = value
    if updates:
        user_settings.save_preferences(settings, updates)
    return load_prefs(settings)


def _matches(word: str, text: str) -> bool:
    return all(part and part in text for part in word.lower().split("&"))


def sensitive_word(path: str, name: str, words: Iterable[str]) -> str:
    """The first sensitive word found in the file's name or path, or ""."""
    text = f"{path}\n{name}".lower()
    for word in words:
        if _matches(word, text):
            return word
    return ""


def in_folder(path: str, folders: Iterable[str]) -> str:
    """The excluded folder containing ``path``, or ""."""
    key = os.path.normcase(os.path.normpath(path))
    for folder in folders:
        prefix = os.path.normcase(os.path.normpath(folder)).rstrip("\\/")
        if prefix and (key == prefix or key.startswith(prefix + os.sep)):
            return folder
    return ""


def exclusion_reason(record: dict[str, Any], prefs: SuggestionPrefs) -> str:
    """Why a file gets no suggestions ("" when it may have some)."""
    word = sensitive_word(record.get("path", ""), record.get("name", ""), prefs.sensitive_words)
    if word:
        return f"sensitive:{word}"
    folder = in_folder(record.get("path", ""), prefs.excluded_folders)
    if folder:
        return f"folder:{folder}"
    return ""
