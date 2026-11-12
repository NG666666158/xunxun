"""Shared helpers for the doc_cards_after_import scripts (public data only).

* ``copy_data``: a working copy of a data directory (SQLite backup API, the source read only),
  with follow-ups and home-page questions off and, unless kept, no document cards;
* ``CachedChat``: the card model behind a reply cache, so a re-run costs nothing; live calls
  keep their tokens and seconds;
* ``load_key``: the SiliconFlow key from an .env file into the environment (never printed).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import threading
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROJECT = ROOT / "project"


def use_project(project: Path = PROJECT) -> None:
    sys.path.insert(0, str(project.resolve()))
    os.environ["PYTHON_KEYRING_BACKEND"] = "keyring.backends.null.Keyring"


def load_key(env_file: Path | None) -> None:
    if os.environ.get("SILICONFLOW_API_KEY"):
        return
    if env_file and env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "SILICONFLOW_API_KEY":
                os.environ["SILICONFLOW_API_KEY"] = value.strip().strip('"').strip("'")


def copy_data(source: Path, target: Path, keep_cards: bool = False) -> Path:
    """A copy of a data directory to work on; the source is only read."""
    target.mkdir(parents=True, exist_ok=False)
    src = sqlite3.connect(f"file:{(source / 'knowledge.sqlite3').as_posix()}?mode=ro", uri=True)
    dst = sqlite3.connect(str(target / "knowledge.sqlite3"))
    src.backup(dst)
    src.close()
    if not keep_cards:
        tables = {row[0] for row in dst.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "doc_cards" in tables:
            dst.execute("DELETE FROM doc_cards")
            dst.commit()
    dst.close()
    for folder in ("parsed", "parents", "cache"):
        if (source / folder).is_dir():
            shutil.copytree(source / folder, target / folder)
    prefs = {}
    if (source / "preferences.json").is_file():
        prefs = json.loads((source / "preferences.json").read_text(encoding="utf-8"))
    prefs.update({"followups_enabled": False, "suggestions_enabled": False,
                  "doc_cards": True, "doc_type_search": True})
    (target / "preferences.json").write_text(json.dumps(prefs, ensure_ascii=False), encoding="utf-8")
    return target


class CachedChat:
    """The card model behind a reply cache (keyed by model, prompt and reply room)."""

    def __init__(self, model, model_name: str, path: Path, max_tokens: int | None = None):
        self.model, self.model_name, self.path, self.max_tokens = model, model_name, path, max_tokens
        self.cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        self.lock = threading.Lock()
        self.live: list[dict] = []
        # Every reply used, live or from the cache (a re-run reports the first run's cost).
        self.used: list[dict] = []
        self.hits = 0

    def bind(self, **kwargs):
        bound = CachedChat.__new__(CachedChat)
        bound.__dict__.update(self.__dict__)
        bound.model = self.model.bind(**kwargs) if hasattr(self.model, "bind") else self.model
        bound.max_tokens = kwargs.get("max_tokens", self.max_tokens)
        bound._root = getattr(self, "_root", self)
        return bound

    def invoke(self, messages):
        root = getattr(self, "_root", self)
        prompt = messages[0].content
        key = hashlib.sha1(json.dumps([self.model_name, prompt, self.max_tokens], ensure_ascii=False)
                           .encode("utf-8")).hexdigest()
        with root.lock:
            entry = root.cache.get(key)
        if entry is not None:
            root.hits += 1
            root.used.append(entry)
            return SimpleNamespace(content=entry["content"], usage_metadata={"total_tokens": entry["tokens"]})
        started = time.perf_counter()
        message = self.model.invoke(messages)
        usage = getattr(message, "usage_metadata", None) or {}
        entry = {"content": str(message.content), "tokens": int(usage.get("total_tokens") or 0),
                 "input_tokens": int(usage.get("input_tokens") or 0),
                 "output_tokens": int(usage.get("output_tokens") or 0),
                 "seconds": round(time.perf_counter() - started, 2), "at": time.time(), "files": prompt.count("【文件 ") or 1}
        with root.lock:
            root.cache[key] = entry
            root.live.append(entry)
            root.used.append(entry)
            root.path.parent.mkdir(parents=True, exist_ok=True)
            root.path.write_text(json.dumps(root.cache, ensure_ascii=False), encoding="utf-8")
        return SimpleNamespace(content=entry["content"], usage_metadata={"total_tokens": entry["tokens"]})


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
