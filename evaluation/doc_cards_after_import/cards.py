"""Write document cards for a data directory with the app's own builder; measure cost and accuracy.

    python -I evaluation/doc_cards_after_import/cards.py --data <data copy> --batch 8 \
        --out evaluation/doc_cards_after_import/results/public_batch8.json \
        --manifest <doc_cards_v1 manifest.json> --reference <doc_cards_v1 public_cards.json> --env project/.env

* ``retrieval.doc_cards.DocCardBuilder`` as the app runs it (prompt, pacing, waits out 限流), on
  Qwen/Qwen3-8B with thinking off; ``--batch`` files share one call (1 = one call per file).
* Replies are cached (``--cache``): a re-run costs nothing; the first run's tokens and seconds
  are kept with each entry.
* Writes counts, tokens, time and type accuracy against the manifest (and agreement with cards
  written one file per call, ``--reference``). File names in the public set are anonymous.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I leaves the script's folder out
from common import HERE, CachedChat, load_key, use_project  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--reference", type=Path, help="cards written one file per call (name -> doc_type, subject)")
    parser.add_argument("--env", type=Path)
    parser.add_argument("--cache", type=Path, default=HERE / "work" / "card_cache.json")
    parser.add_argument("--rate", type=float, default=0.0, help="tokens a minute (default: the app's setting)")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    use_project()
    load_key(args.env)

    from config import Settings
    from core.user_settings import apply_user_settings
    from library import FileRegistry
    from library.registry import SEARCHABLE_STATUSES
    from retrieval.doc_cards import DocCardBuilder, name_doc_type
    from retrieval.embedded_store import EmbeddedStore
    from services.model_clients import build_chat_model

    settings = apply_user_settings(Settings(_env_file=None, data_dir=args.data.resolve()))
    if settings.llm_model != "Qwen/Qwen3-8B":
        raise SystemExit(f"model is {settings.llm_model}, expected Qwen/Qwen3-8B")
    store = EmbeddedStore(settings, db_path=settings.embedded_db_path)
    store.ensure_collection()
    registry = FileRegistry(settings.registry_path)
    rows = {row["file_id"]: row for row in registry.list_files(statuses=SEARCHABLE_STATUSES)}
    files = {fid: row["name"] for fid, row in rows.items() if row.get("content_mode") != "name"}

    model = build_chat_model(settings, max_tokens=200, max_retries=0)
    model.hedge_after = 0.0
    chat = CachedChat(model, settings.llm_model, args.cache, max_tokens=200)
    builder = DocCardBuilder(store, settings, chat_model=lambda: chat, files=lambda: dict(files), batch=args.batch)
    if args.rate:
        builder.tokens_per_minute = builder.idle_tokens_per_minute = args.rate
    started = time.perf_counter()
    queued = builder.sync()
    outcomes = builder.run_pending()
    wall = time.perf_counter() - started

    cards = store.doc_cards(doc_ids=files)
    by_name = {files[fid]: card for fid, card in cards.items()}
    live = chat.used  # live calls, or the cached replies of the run that made them
    tokens = sum(entry["tokens"] for entry in live)
    built = outcomes.get("built", 0)
    if live and chat.hits:
        # A re-run from the cache: the first run's time, from its calls' clock times.
        wall = max(e["at"] for e in live) - min(e["at"] - e["seconds"] for e in live)
    summary = {
        "batch": args.batch, "files": len(files), "queued": queued, "outcomes": outcomes,
        "named_by_file_name": sum(1 for name in files.values() if name_doc_type(name)),
        "cards": len(cards), "unknown_type": sum(1 for card in cards.values() if not card["doc_type"]),
        "calls": builder.stats["calls"], "live_calls": len(chat.live), "cache_hits": chat.hits,
        "files_per_call": round(sum(entry["files"] for entry in live) / len(live), 2) if live else None,
        "seconds_per_call": round(sum(e["seconds"] for e in live) / len(live), 2) if live else None,
        "tokens_total": tokens,
        "tokens_per_file": round(tokens / built, 1) if built and live else None,
        "input_tokens_per_file": round(sum(e["input_tokens"] for e in live) / built, 1) if built and live else None,
        "output_tokens_per_file": round(sum(e["output_tokens"] for e in live) / built, 1) if built and live else None,
        "wall_seconds": round(wall, 1),
        "files_per_minute": round(built / wall * 60, 2) if wall else None,
        "tokens_per_minute": round(tokens / wall * 60) if wall and live else None,
        "budget_tokens_per_minute": builder.tokens_per_minute,
        "rate_limited": builder.stats["rate_limited"], "failed": builder.stats.get("failed", 0),
        "asked_alone_after_batch": builder.stats.get("retry", 0),
        "types": Counter(card["doc_type"] or "（其他）" for card in cards.values()).most_common(),
    }
    if args.manifest:
        manifest = {item["name"]: item for item in json.loads(args.manifest.read_text(encoding="utf-8"))}
        pairs = [(manifest[name]["type"], card["doc_type"]) for name, card in by_name.items() if name in manifest]
        summary["type_accuracy"] = round(sum(g == p for g, p in pairs) / len(pairs), 3) if pairs else None
        by_gold: dict[str, list[int]] = {}
        for gold, got in pairs:
            by_gold.setdefault(gold, [0, 0])
            by_gold[gold][0] += gold == got
            by_gold[gold][1] += 1
        summary["type_accuracy_by_gold"] = {gold: f"{ok}/{n}" for gold, (ok, n) in by_gold.items()}
        summary["confusions"] = Counter(f"{g}→{p or '（其他）'}" for g, p in pairs if g != p).most_common()
        mentions = [(item["mentions"], by_name.get(name, {}).get("doc_type")) for name, item in manifest.items()
                    if item.get("mentions")]
        summary["mention_only_typed_as_mentioned"] = f"{sum(1 for m, got in mentions if got == m)}/{len(mentions)}"
        # 简历: does the card name the person (the manifest's subject)?
        def person(text: str) -> str:
            return re.sub(r"先生|女士|\s", "", text or "")

        people = [(person(item["subject"]), person(by_name.get(name, {}).get("subject", "")))
                  for name, item in manifest.items() if item["type"] == "简历" and item.get("subject")]
        summary["resume_subject_named"] = f"{sum(1 for want, got in people if want and want in got)}/{len(people)}"
    if args.reference:
        reference = {item["name"]: item for item in json.loads(args.reference.read_text(encoding="utf-8"))}
        common = [name for name in by_name if name in reference]
        summary["same_type_as_one_per_call"] = (
            f"{sum(1 for name in common if by_name[name]['doc_type'] == reference[name]['doc_type'])}/{len(common)}")
        summary["same_period_as_one_per_call"] = (
            f"{sum(1 for name in common if by_name[name]['period'] == reference[name]['period'])}/{len(common)}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
