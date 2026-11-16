"""Does a Qwen3-8B rewrite add anything on top of the synonym table (step 2)?

  python evaluation/recall_wording_v1/rewrite_eval.py [--pause 4] [--set field=value ...]

Two rewrites, one model call each per question, replies cached in work/rewrites.json:

* hyde         a sentence written the way a policy paper would answer the question
* paraphrase   the question said three other, more formal ways

Each rewrite is searched (BM25 + vector) and its first new candidates join the pool of the
configuration given with --set (default: wording_expansion on), exactly as the synonym
expansion's do; every candidate is scored against the question as asked and cut by the app's
own ``select_hits``. Reported: pool recall, gold rank and kept, with and without each
rewrite, and what the rewrite costs (tokens, seconds per call).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))

import harness  # noqa: E402

from rag_agent.answering import classify_question, retrieval_limits  # noqa: E402
from retrieval.selection import select_hits  # noqa: E402
from retrieval.types import SearchHit  # noqa: E402

PROMPTS = {
    "hyde": (
        "你帮助检索政策文件。假设资料里有一段话正好回答用户的问题，请用政策文件、申报指南的书面用语把这段话写出来，"
        "60字以内。不知道的数字、日期一律写“若干”，不要编造。只输出这段话。"
    ),
    "paraphrase": (
        "你帮助检索政策文件。把用户的问题用政策文件、申报指南里常见的书面说法改写成3个不同的检索问句，"
        "每行一个，不要编号，不要解释。"
    ),
}
ADD = 20  # new candidates a rewrite may add, as wording_top


def rewrites(questions: list[dict], pause: float) -> dict:
    from langchain_core.messages import HumanMessage, SystemMessage
    from services.model_clients import build_chat_model

    path = harness.WORK / "rewrites.json"
    cache = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    llm = None
    for question in questions:
        for kind, prompt in PROMPTS.items():
            key = f"{kind}\n{question['question']}"
            if key in cache:
                continue
            if llm is None:
                llm = build_chat_model(harness.settings(), max_tokens=200)
            for attempt in range(6):
                started = time.perf_counter()
                try:
                    reply = llm.invoke([SystemMessage(content=prompt), HumanMessage(content=question["question"])])
                    break
                except Exception as exc:  # rate limit: wait and try again
                    print(f"retry {attempt + 1}: {type(exc).__name__}", flush=True)
                    time.sleep(30)
            else:
                raise SystemExit("the chat model stayed unavailable")
            usage = getattr(reply, "usage_metadata", None) or {}
            cache[key] = {"text": re.sub(r"<think>.*?</think>", "", str(reply.content), flags=re.S).strip(),
                          "seconds": round(time.perf_counter() - started, 2),
                          "tokens": int(usage.get("total_tokens") or 0)}
            path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
            time.sleep(pause)
    return cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pause", type=float, default=4.0)
    parser.add_argument("--index", default="base")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--name", default="rewrite")
    args = parser.parse_args()
    overrides = harness.parse_overrides(args.overrides) or {"wording_expansion": True}
    questions = harness.load_questions()
    cache = rewrites(questions, args.pause)
    retriever, recorder, counter, store, model_cache = harness.open_retriever(args.index, **overrides)
    chunks = harness.all_chunks(harness.index_dir(args.index) / "knowledge.sqlite3")
    by_text = {c["text"]: c for c in chunks}
    variants = ("none", "hyde", "paraphrase")
    stats = {v: {t: {"gold": 0, "pool": 0, "kept": 0, "ranks": [], "docs": []} for t in ("fact", "generic", "survey")} for v in variants}
    for question in questions:
        text = question["question"]
        limits = retrieval_limits(classify_question(text))
        recorder.reset()
        retriever.retrieve(text, mode="hybrid_rerank", **limits)
        pool = list(dict.fromkeys(t for call in recorder.calls for t in call["texts"] if t in by_text))
        gold = [harness.gold_chunk_ids(chunks, g) for g in question["gold"]]
        for variant in variants:
            texts = list(pool)
            if variant != "none":
                lines = [line.strip(" -·0123456789.、") for line in cache[f"{variant}\n{text}"]["text"].splitlines() if line.strip()]
                lines = lines[:3] if variant == "paraphrase" else [" ".join(lines)]
                share = max(1, ADD // max(1, len(lines)))
                for line in lines:
                    added = 0
                    for found in store.search_hybrid(line, counter.embed_query(line), limit=40):
                        if found.text not in texts and found.text in by_text and added < share:
                            texts.append(found.text)
                            added += 1
            scores = recorder.inner.scores(text, texts)
            ranked = sorted(zip(texts, scores), key=lambda pair: -pair[1])
            hits = [SearchHit(id=by_text[t]["id"], score=s, text=t, doc_id=by_text[t]["doc_id"], parent_id="", source=by_text[t]["source"],
                              heading="", locator="", retrieval_method="", rerank_score=s) for t, s in ranked]
            kept = {hit.id for hit in select_hits(hits, retriever.policy(limits.get("max_keep"), limits.get("per_document")))}
            order = [hit.id for hit in hits]
            block = stats[variant][question["type"]]
            block["docs"].append(len(texts))
            for ids in gold:
                block["gold"] += 1
                ranks = [order.index(i) + 1 for i in ids if i in order]
                block["pool"] += bool(ranks)
                block["kept"] += bool(ids & kept)
                if ranks:
                    block["ranks"].append(min(ranks))
    out: dict = {"overrides": overrides, "add": ADD}
    for variant in variants:
        out[variant] = {
            kind: {"pool": round(b["pool"] / b["gold"], 3), "kept": round(b["kept"] / b["gold"], 3),
                   "rank_median": statistics.median(b["ranks"]) if b["ranks"] else None,
                   "scored": round(statistics.fmean(b["docs"]), 1)}
            for kind, b in stats[variant].items()
        }
    for kind in PROMPTS:
        calls = [value for key, value in cache.items() if key.startswith(kind + "\n")]
        out[f"{kind}_call"] = {"calls": len(calls), "seconds_median": statistics.median(c["seconds"] for c in calls),
                               "tokens_mean": round(statistics.fmean(c["tokens"] for c in calls))}
    harness.RESULTS.mkdir(exist_ok=True)
    (harness.RESULTS / f"{args.name}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    store.close()
    model_cache.close()


if __name__ == "__main__":
    main()
