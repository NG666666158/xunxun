"""Type questions right after a big import, before most files have a card (public data only).

The library: 1,500 synthetic files made from CRUD-RAG news (import_speed_v1/make_corpus.py,
already indexed with real bge-m3 by import_speed_v1) plus the 110 public files of doc_cards_v1
(20 resumes, 12 contracts, 16 paper abstracts, 62 news, all named 资料_0001.txt …), imported
into it with the app's own library (real bge-m3). No file has a card: the state right after a
big import.

    # 1. the library (copies the 1,500-file data directory, imports the 110 files)
    python -I evaluation/doc_cards_after_import/big_library.py build --base <real_final_small data> \
        --files <doc_cards_v1 work/files> --data <new folder> --env project/.env
    # 2. how many files of each type a type question reaches before any card exists (no chat model)
    python -I evaluation/doc_cards_after_import/big_library.py probe --data <folder> \
        --questions <questions.jsonl> --manifest <manifest.json> --env project/.env
    # 3. ask, let the builder write the cards of the files that question moved up (real
    #    Qwen3-8B, the app's pace), ask again
    python -I evaluation/doc_cards_after_import/big_library.py lazy --data <folder> ... --batch 1

Writes results/big_library_<step>.json (scores and counts only).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I leaves the script's folder out
from common import HERE, CachedChat, copy_data, load_key, read_jsonl, use_project  # noqa: E402
from questions import Recorder, score  # noqa: E402

NOW = datetime(2026, 11, 9, 12, 0)
PURE = "pure"


def open_system(data: Path, **overrides):
    from config import Settings
    from core.user_settings import apply_user_settings

    settings = apply_user_settings(Settings(_env_file=None, data_dir=data.resolve(), watch_enabled=False))
    for key, value in overrides.items():
        setattr(settings, key, value)
    return settings


def build(args) -> None:
    from core.rag_system import RAGSystem

    copy_data(args.base, args.data)
    prefs = json.loads((args.data / "preferences.json").read_text(encoding="utf-8"))
    prefs["doc_cards"] = False  # no cards while building: the experiment starts without any
    (args.data / "preferences.json").write_text(json.dumps(prefs), encoding="utf-8")
    settings = open_system(args.data)
    settings.ensure_directories()
    system = RAGSystem(settings)
    system.start_library()
    library = system.library
    started = time.monotonic()
    library.import_paths([str(args.files.resolve())])
    while True:
        status = library.status()
        counts = status.get("counts") or {}
        busy = (counts.get("pending") or 0) + (counts.get("indexing") or 0)
        if not busy and time.monotonic() - started > 10:
            break
        time.sleep(2)
    print("imported in", round(time.monotonic() - started), "s", counts, flush=True)
    library.stop()
    prefs["doc_cards"] = True
    (args.data / "preferences.json").write_text(json.dumps(prefs), encoding="utf-8")


def service_for(settings):
    from library import FileRegistry
    from retrieval.embedded_store import EmbeddedStore
    from retrieval.file_search import FileSearchService
    from services.model_clients import SiliconFlowEmbeddings, SiliconFlowReranker

    store = EmbeddedStore(settings, db_path=settings.embedded_db_path)
    store.ensure_collection()
    registry = FileRegistry(settings.registry_path)
    service = FileSearchService(store, SiliconFlowEmbeddings(settings), SiliconFlowReranker(settings), registry,
                                settings, clock=lambda: NOW)
    return service, store, registry


def probe(args) -> None:
    """Before any card: which files of the type a question reaches (listed, the search's own
    pool, the type probe), and how other probe texts would do."""
    from library.registry import SEARCHABLE_STATUSES
    from retrieval.doc_cards import TYPES_BY_NAME, name_doc_type, type_probe

    settings = open_system(args.data)
    service, store, registry = service_for(settings)
    names = {row["file_id"]: row["name"] for row in registry.list_files(statuses=SEARCHABLE_STATUSES)}
    ids = {name: fid for fid, name in names.items()}
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    gold_type = {item["name"]: item["type"] for item in manifest}
    of_type: dict[str, set[str]] = defaultdict(set)
    for name, kind in gold_type.items():
        if name in ids:
            of_type[kind].add(ids[name])
    waiting = {fid for fid, name in names.items() if not name_doc_type(name)}
    questions = [q for q in read_jsonl(args.questions) if q["category"] == PURE]
    service.probe_in_background = False
    out: dict = {"files": len(names), "without_card": len(waiting), "by_question": [], "variants": {}}
    for question in questions:
        recorder = Recorder()
        service.cards = recorder
        started = time.perf_counter()
        result = service.search(question["question"], limit=100)
        elapsed = round((time.perf_counter() - started) * 1000)
        listed = [f["file_id"] for f in result["files"]]
        asked = list(dict.fromkeys(recorder.asked))
        want = of_type[question["doc_type"]]
        out["by_question"].append({
            "id": question["id"], "doc_type": question["doc_type"], "elapsed_ms_with_probe": elapsed,
            "untyped": result.get("untyped"), "moved_up": len(asked),
            "of_type_listed_in_top8": len(want & set(listed[:8])),
            "of_type_moved_up": len(want & set(asked)), "of_type": len(want),
        })
        print(out["by_question"][-1], flush=True)
    # Probe texts and pool sizes (file entries of the files without a card).
    for kind in ("简历", "合同", "论文"):
        item = TYPES_BY_NAME[kind]
        texts = {"type_probe": type_probe(kind), "type_word": kind,
                 "words": " ".join(item.words[:6])}
        want = of_type[kind]
        for label, text in texts.items():
            vector = service._embed(text)
            entries = store.search_file_entries(text, vector, limit=64, doc_ids=waiting)
            order = [entry["doc_id"] for entry in entries]
            chunks = service._search_chunks(text, vector, waiting, 64)
            chunk_order = list(dict.fromkeys(hit.doc_id for hit in chunks))
            out["variants"].setdefault(kind, {})[label] = {
                "entries": {k: len(want & set(order[:k])) for k in (8, 16, 32, 64)},
                "chunks": {k: len(want & set(chunk_order[:k])) for k in (8, 16, 32, 64)},
                "of_type": len(want),
            }
        print(kind, json.dumps(out["variants"][kind], ensure_ascii=False), flush=True)
    write(args, "probe", out)


def lazy(args) -> None:
    """Ask a type question, let the app's builder write the cards that question moved up (real
    Qwen3-8B, the app's pace), ask again; the next type starts from the cards written so far."""
    from library.registry import SEARCHABLE_STATUSES
    from retrieval.doc_cards import DocCardBuilder
    from services.model_clients import build_chat_model

    settings = open_system(args.data)
    service, store, registry = service_for(settings)
    rows = {row["file_id"]: row for row in registry.list_files(statuses=SEARCHABLE_STATUSES)}
    files = {fid: row["name"] for fid, row in rows.items() if row.get("content_mode") != "name"}
    model = build_chat_model(settings, max_tokens=200, max_retries=0)
    model.hedge_after = 0.0
    chat = CachedChat(model, settings.llm_model, args.cache, max_tokens=200)
    builder = DocCardBuilder(store, settings, chat_model=lambda: chat, files=lambda: dict(files), batch=args.batch)
    builder.tokens_per_minute = builder.idle_tokens_per_minute = args.rate
    service.cards = builder
    service.probe_in_background = False
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    types = {item["name"]: item["type"] for item in manifest}
    mentions = {item["name"]: item["mentions"] for item in manifest if item.get("mentions")}
    questions = [q for q in read_jsonl(args.questions) if q["category"] == PURE]
    first = {}
    for question in questions:
        first.setdefault(question["doc_type"], question)
    out: dict = {"files": len(files), "batch": args.batch, "tokens_per_minute": args.rate, "steps": []}
    for kind, question in first.items():
        before = ask(service, question, types, mentions)
        waiting = builder.pending()
        used = len(chat.used)
        started = time.monotonic()
        outcomes = builder.run_pending()
        seconds = time.monotonic() - started
        calls = chat.used[used:]
        after = ask(service, question, types, mentions)
        # The other ways of asking for the same type, now.
        others = [ask(service, q, types, mentions) for q in questions if q["doc_type"] == kind and q is not question]
        step = {"doc_type": kind, "question": question["id"], "first_ask": before, "moved_up": waiting,
                "outcomes": outcomes, "build_seconds": round(seconds, 1), "calls": len(calls),
                "tokens": sum(e["tokens"] for e in calls), "second_ask": after,
                "other_questions_now": {key: round(statistics.mean(o[key] for o in others), 3)
                                        for key in ("hit1", "type5", "type_recall")} if others else None}
        out["steps"].append(step)
        print(json.dumps(step, ensure_ascii=False), flush=True)
    out["cards_written"] = len(store.doc_card_hashes())
    out["rate_limited"] = builder.stats["rate_limited"]
    write(args, f"lazy_b{args.batch}", out)


def ask(service, question: dict, types: dict, mentions: dict) -> dict:
    started = time.perf_counter()
    result = service.search(question["question"], limit=100)
    elapsed = round((time.perf_counter() - started) * 1000)
    return {"elapsed_ms": elapsed, "untyped": result.get("untyped"), "typed": result.get("typed"),
            **{k: v for k, v in score(question, result["files"], types, mentions).items() if k != "rank"}}


def write(args, step: str, out: dict) -> None:
    (HERE / "results").mkdir(exist_ok=True)
    path = HERE / "results" / f"big_library_{step}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", path, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=("build", "probe", "lazy"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--files", type=Path)
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--env", type=Path)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--rate", type=float, default=9000.0)
    parser.add_argument("--cache", type=Path, default=HERE / "work" / "card_cache.json")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    use_project()
    load_key(args.env)
    {"build": build, "probe": probe, "lazy": lazy}[args.step](args)


if __name__ == "__main__":
    main()
