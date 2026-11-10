"""Adapters: public datasets (and the repository's own sets) into the suite's schema.

``prepare(name)`` reads the downloaded files (evaluation/external/<name>/, see fetch.py),
samples with a fixed seed and writes ``<work>/<name>/prepared/``:

* ``questions.jsonl`` and ``corpus.jsonl`` in the schema of schema.py;
* ``PREPARED.json``: sizes, seed, the source checksum (FETCHED.json) and a checksum of
  the two files, recorded by every result.

Choices shared by the external sets (see REPORT.md):

* The corpus is every gold document of the sampled questions plus random documents of
  the same dataset ("distractors"), up to the target size.
* Held-out negatives: a seeded share of the sampled questions has its gold documents
  left out of the corpus. Those questions are kept with ``answerable: false`` and type
  ``heldout_negative``: the index has no labelled answer for them.
* A distractor that is a near copy of any gold text (half of its sampled 16-character
  windows occur in gold texts) is left out: it would be an unlabelled duplicate of the
  answer. Rewrites of the same story by another outlet stay: they are the hard cases.
* Files have neutral names (``doc000123.txt``), except MultiHop-RAG articles, which are
  named by their title as a user would save them. The chunker puts the name in front
  of every child (``chunk_context_prefix``), so meaningless names keep it neutral.

Run with -I (the inputs are untrusted data):
  python -I evaluation/suite/adapters.py crud_rag duretrieval t2retrieval multihop_rag
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
from typing import Any, Iterable
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I leaves the script's folder out

from common import EXTERNAL, WORK, read_jsonl, sha256_file, write_jsonl
import schema

SEED = 20261107
PYARROW_DIR = os.environ.get("EVAL_PYARROW_DIR", "")  # pyarrow outside the app's venv (optional)

# Tags assigned from the query text (all datasets).
DIGITS = re.compile(r"\d")
# Model numbers, codes, versions: "iPhone 14", "GB/T 2260", "rtx3060", "v2.1", "4G".
CODE = re.compile(r"[A-Za-z]+[-_/ ]?\d|\d+[A-Za-z]{1,4}\b|\d+\.\d+|[A-Z]{2,}")
# In English text capitals are ordinary words ("FTX", "CEO"): only letter-digit mixes count.
CODE_EN = re.compile(r"[A-Za-z]+[-_/]?\d|\d+[A-Za-z]{1,4}\b|\d+\.\d+")
COLLOQUIAL = re.compile(r"怎么|咋|啥|吗|呢|嘛|么|哪个好|多少钱|靠谱|行不行|能不能|是不是|好不好|有没有|怎样|如何|为什么")
# Generation failures shipped as questions ("由于两篇新闻内容相同，无法设计出…的问题。"): kept
# in the files with this tag and left out of every metric (report.py).
DEFECT = re.compile(r"无法设计|无法生成|无法提出|不能设计|无法构造")
TEMPORAL = re.compile(r"\d{4}年|\d{1,2}月|\d{1,2}日|今年|去年|明年|最近|最新|近期|当时|之前|之后|以前|以后|期间|何时|什么时候|哪一年|哪年")


def query_tags(query: str, language: str) -> list[str]:
    tags = []
    if DIGITS.search(query):
        tags.append("digits")
    if (CODE if language == "zh" else CODE_EN).search(query):
        tags.append("code_or_model")
    if language == "zh":
        if COLLOQUIAL.search(query):
            tags.append("colloquial")
        if len(query) <= 10:
            tags.append("short_query")
        if TEMPORAL.search(query):
            tags.append("temporal_wording")
    return tags


# ---------------------------------------------------------------- near copies
WINDOW = 16


def _fingerprints(text: str) -> set[int]:
    """Hashes of 16-character windows (whitespace removed), keeping those ≡ 0 mod 4:
    the same window is kept in every text, so overlaps can be compared on the sample."""
    compact = re.sub(r"\s+", "", text)
    out = set()
    for start in range(0, max(0, len(compact) - WINDOW + 1)):
        value = zlib.crc32(compact[start:start + WINDOW].encode("utf-8"))
        if value % 4 == 0:
            out.add(value)
    return out


class NearCopies:
    def __init__(self, texts: Iterable[str], threshold: float = 0.5):
        self.gold: set[int] = set()
        for text in texts:
            self.gold |= _fingerprints(text)
        self.threshold = threshold

    def is_copy(self, text: str) -> bool:
        prints = _fingerprints(text)
        return bool(prints) and len(prints & self.gold) / len(prints) >= self.threshold


# ---------------------------------------------------------------- shared helpers
def _fetched(name: str) -> dict:
    path = EXTERNAL / name / "FETCHED.json"
    if not path.is_file():
        raise SystemExit(f"{path} missing: run python -I evaluation/suite/fetch.py {name}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write(name: str, questions: list[dict], corpus: list[dict], info: dict[str, Any]) -> dict:
    for row in questions:
        schema.validate_question(row)
    seen = set()
    for row in corpus:
        schema.validate_corpus_row(row)
        if row["doc_id"] in seen:
            raise ValueError(f"duplicate doc_id {row['doc_id']}")
        seen.add(row["doc_id"])
    in_corpus = seen
    for row in questions:
        missing = [d for d in row["gold_doc_ids"] if d not in in_corpus]
        if row["answerable"] and missing:
            raise ValueError(f"{row['id']}: gold {missing[:2]} not in the corpus")
        if not row["answerable"] and any(d in in_corpus for d in row["gold_doc_ids"]):
            raise ValueError(f"{row['id']}: a held-out gold is in the corpus")
    folder = WORK / name / "prepared"
    write_jsonl(folder / "questions.jsonl", questions)
    write_jsonl(folder / "corpus.jsonl", corpus)
    fetched = _fetched(name) if (EXTERNAL / name / "FETCHED.json").is_file() else {}
    manifest = {
        "dataset": name,
        "seed": SEED,
        "source_checksum": fetched.get("checksum"),
        "source_revisions": sorted({f["revision"] for f in fetched.get("files", [])}),
        "license": fetched.get("license"),
        "questions": len(questions),
        "answerable": sum(q["answerable"] for q in questions),
        "types": dict(Counter(q["type"] for q in questions)),
        "tags": dict(Counter(t for q in questions for t in q["tags"])),
        "corpus_docs": len(corpus),
        "corpus_chars": sum(len(d["text"]) for d in corpus),
        "questions_sha256": sha256_file(folder / "questions.jsonl"),
        "corpus_sha256": sha256_file(folder / "corpus.jsonl"),
        **info,
    }
    (folder / "PREPARED.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _clean(text: str) -> str:
    """What the app's .txt parser does (parsers._clean_text), plus web markup in T2Ranking."""
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<img[^>]*>", "", text, flags=re.IGNORECASE)
    text = text.replace("\x00", " ").replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------- C-MTEB (DuRetrieval, T2Retrieval)
def _parquet(path: Path) -> list[dict]:
    if PYARROW_DIR and PYARROW_DIR not in sys.path:
        sys.path.append(PYARROW_DIR)
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("reading the C-MTEB files needs pyarrow (pip install pyarrow, or set EVAL_PYARROW_DIR)") from exc
    return pq.read_table(path).to_pylist()


def prepare_cmteb(name: str, queries_n: int, corpus_n: int, heldout_share: float = 0.1) -> dict:
    folder = EXTERNAL / name
    fetched = _fetched(name)
    rng = random.Random(f"{SEED}:{name}")
    queries = {str(r["id"]): r["text"] for r in _parquet(folder / "queries.parquet")}
    positives: dict[str, list[str]] = defaultdict(list)
    for row in _parquet(folder / "qrels.parquet"):
        if int(row["score"]) > 0:
            positives[str(row["qid"])].append(str(row["pid"]))
    corpus_rows = _parquet(folder / "corpus.parquet")
    texts = {str(r["id"]): _clean(r["text"]) for r in corpus_rows}
    del corpus_rows
    qids = sorted(q for q in positives if q in queries and all(_nonempty(texts.get(p)) for p in positives[q]))
    sampled = rng.sample(qids, min(queries_n, len(qids)))
    # Held out: questions whose positives no other sampled question shares.
    owners = Counter(p for q in sampled for p in set(positives[q]))
    exclusive = [q for q in sampled if all(owners[p] == 1 for p in positives[q])]
    heldout = set(rng.sample(exclusive, min(len(exclusive), round(heldout_share * len(sampled)))))
    gold = {p for q in sampled if q not in heldout for p in positives[q]}
    removed = {p for q in heldout for p in positives[q]}
    near = NearCopies([texts[p] for p in gold | removed])
    candidates = [pid for pid in sorted(texts) if pid not in gold and pid not in removed and _nonempty(texts[pid])]
    rng.shuffle(candidates)
    distractors, copies = [], 0
    for pid in candidates:
        if len(gold) + len(distractors) >= corpus_n:
            break
        if near.is_copy(texts[pid]):
            copies += 1
            continue
        distractors.append(pid)
    doc_ids = sorted(gold) + distractors
    names = {pid: f"doc{n:06d}.txt" for n, pid in enumerate(sorted(doc_ids, key=lambda p: hashlib.sha1(f'{SEED}{p}'.encode()).hexdigest()))}
    corpus = [{"doc_id": pid, "title": "", "text": texts[pid], "source": names[pid], "metadata": {}} for pid in doc_ids]
    version = ",".join(sorted({f["revision"][:12] for f in fetched["files"]}))
    questions = []
    for qid in sampled:
        query = queries[qid].strip()
        held = qid in heldout
        questions.append(schema.question(
            id=f"{name}:{qid}", query=query, language="zh",
            type="heldout_negative" if held else "single",
            tags=query_tags(query, "zh") + (["multi_positive"] if len(positives[qid]) > 1 else []),
            gold_doc_ids=sorted(set(positives[qid])),
            gold_passages=[],
            answerable=not held,
            needs_all=False,
            source={"dataset": name, "version": version, "original_id": qid},
        ))
    return _write(name, questions, corpus, {"heldout_questions": len(heldout), "near_copies_dropped": copies,
                                             "gold_docs": len(gold), "distractors": len(distractors)})


def _nonempty(text: str | None) -> bool:
    return bool(text and text.strip() and len(text.strip()) >= 5)


# ---------------------------------------------------------------- CRUD-RAG
CRUD_SUBSETS = {"questanswer_1doc": ("single", 1), "questanswer_2docs": ("multi_2doc", 2), "questanswer_3docs": ("multi_3doc", 3)}


def _crud_corpus(folder: Path) -> list[str]:
    docs = []
    for path in sorted((folder / "80000_docs").glob("documents_dup_part_*")):
        for line in path.read_text(encoding="utf-8").split("\n"):
            line = _clean(line)
            if _nonempty(line):
                docs.append(line)
    return docs


def prepare_crud(per_subset: dict[str, int] | None = None, corpus_n: int = 10000, heldout_events: int = 40) -> dict:
    name = "crud_rag"
    folder = EXTERNAL / name
    fetched = _fetched(name)
    rng = random.Random(f"{SEED}:{name}")
    per_subset = per_subset or {"questanswer_1doc": 400, "questanswer_2docs": 300, "questanswer_3docs": 300}
    data = json.loads((folder / "split_merged.json").read_text(encoding="utf-8"))
    # One document per distinct news text (the same report backs several questions).
    doc_of: dict[str, str] = {}

    def gold_id(text: str) -> str:
        key = hashlib.sha1(re.sub(r"\s+", "", text).encode("utf-8")).hexdigest()[:16]
        doc_of.setdefault(key, text)
        return key

    sampled: list[tuple[str, dict]] = []
    for subset, n in per_subset.items():
        rows = [r for r in data[subset] if r.get("questions", "").strip()]
        sampled += [(subset, r) for r in rng.sample(rows, min(n, len(rows)))]
    events = sorted({r["event"] for _, r in sampled})
    held_events = set(rng.sample(events, min(heldout_events, len(events))))
    # Every news text of a held-out event leaves the corpus, from all subsets (sampled or not).
    removed_texts = [_clean(r[f"news{i}"]) for subset in CRUD_SUBSETS for r in data[subset] if r["event"] in held_events
                     for i in (1, 2, 3) if r.get(f"news{i}")]
    questions = []
    gold_docs: set[str] = set()
    for subset, row in sampled:
        kind, count = CRUD_SUBSETS[subset]
        golds = list(dict.fromkeys(gold_id(_clean(row[f"news{i}"])) for i in range(1, count + 1) if _nonempty(row.get(f"news{i}"))))
        held = row["event"] in held_events
        if not held:
            gold_docs.update(golds)
        query = row["questions"].strip()
        questions.append(schema.question(
            id=f"crud_rag:{subset.split('_')[1]}:{row['ID']}", query=query, language="zh",
            type="heldout_negative" if held else kind,
            tags=query_tags(query, "zh") + ([] if kind == "single" else ["multi_doc"])
            + (["dataset_defect"] if DEFECT.search(query) else []),
            gold_doc_ids=golds, gold_answer=str(row.get("answers") or ""),
            answerable=not held, needs_all=len(golds) > 1,
            source={"dataset": name, "version": fetched["files"][0]["revision"][:12], "original_id": row["ID"], "subset": subset},
        ))
    held_ids = {g for q in questions if not q["answerable"] for g in q["gold_doc_ids"]}
    # A held-out document can also back a kept question (same text): then it is not held out.
    clash = held_ids & gold_docs
    if clash:
        questions = [q for q in questions if q["answerable"] or not set(q["gold_doc_ids"]) & clash]
    near = NearCopies([doc_of[g] for g in gold_docs] + removed_texts)
    pool = _crud_corpus(folder)
    order = list(range(len(pool)))
    rng.shuffle(order)
    distractors, copies = [], 0
    for index in order:
        if len(gold_docs) + len(distractors) >= corpus_n:
            break
        if near.is_copy(pool[index]):
            copies += 1
            continue
        distractors.append(index)
    corpus = [{"doc_id": g, "title": "", "text": doc_of[g], "source": "", "metadata": {"gold": True}} for g in sorted(gold_docs)]
    corpus += [{"doc_id": f"d{index:06d}", "title": "", "text": pool[index], "source": "", "metadata": {}} for index in distractors]
    order_names = sorted(range(len(corpus)), key=lambda i: hashlib.sha1(f"{SEED}{corpus[i]['doc_id']}".encode()).hexdigest())
    for n, i in enumerate(order_names):
        corpus[i]["source"] = f"news{n:05d}.txt"
    return _write(name, questions, corpus, {"heldout_events": len(held_events), "near_copies_dropped": copies,
                                             "gold_docs": len(gold_docs), "distractors": len(distractors),
                                             "distractor_pool": len(pool)})


# ---------------------------------------------------------------- MultiHop-RAG
MH_TYPES = {"inference_query": "inference", "comparison_query": "comparison", "temporal_query": "temporal", "null_query": "null"}


def prepare_multihop(per_type: dict[str, int] | None = None) -> dict:
    name = "multihop_rag"
    folder = EXTERNAL / name
    fetched = _fetched(name)
    rng = random.Random(f"{SEED}:{name}")
    per_type = per_type or {"inference_query": 250, "comparison_query": 250, "temporal_query": 250, "null_query": 10**6}
    articles = json.loads((folder / "corpus.json").read_text(encoding="utf-8"))
    by_title: dict[str, list[int]] = defaultdict(list)
    for index, article in enumerate(articles):
        by_title[article["title"].strip()].append(index)
    corpus = []
    for index, article in enumerate(articles):
        title = article["title"].strip()
        safe = re.sub(r'[\\/:*?"<>|]+', " ", title).strip()[:120] or f"article{index:04d}"
        text = _clean(f"{title}\n\n{article['body']}")
        corpus.append({"doc_id": f"a{index:04d}", "title": title, "text": text, "source": f"{safe}.txt",
                       "metadata": {k: article.get(k) for k in ("source", "published_at", "category", "url")}})
    url_of = {a.get("url"): f"a{i:04d}" for i, a in enumerate(articles)}
    queries = json.loads((folder / "MultiHopRAG.json").read_text(encoding="utf-8"))
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(queries):
        groups[row["question_type"]].append(index)
    questions, unmatched = [], 0
    for qtype, indexes in sorted(groups.items()):
        for index in sorted(rng.sample(indexes, min(per_type.get(qtype, 0), len(indexes)))):
            row = queries[index]
            golds, passages = [], []
            for evidence in row.get("evidence_list") or []:
                doc = url_of.get(evidence.get("url"))
                if doc is None:
                    hits = by_title.get((evidence.get("title") or "").strip(), [])
                    doc = f"a{hits[0]:04d}" if len(hits) == 1 else None
                if doc is None:
                    unmatched += 1
                    continue
                golds.append(doc)
                if evidence.get("fact"):
                    passages.append({"doc_id": doc, "text": evidence["fact"], "match": "text"})
            golds = list(dict.fromkeys(golds))
            kind = MH_TYPES[qtype]
            answerable = kind != "null" and bool(golds)
            query = row["query"].strip()
            questions.append(schema.question(
                id=f"multihop_rag:{index}", query=query, language="en", type=kind,
                tags=query_tags(query, "en") + (["multi_doc"] if len(golds) > 1 else []),
                gold_doc_ids=golds, gold_passages=passages, gold_answer=str(row.get("answer") or ""),
                answerable=answerable, needs_all=len(golds) > 1,
                source={"dataset": name, "version": fetched["files"][0]["revision"][:12], "original_id": str(index)},
            ))
    return _write(name, questions, corpus, {"evidence_unmatched": unmatched})


PREPARE = {
    "crud_rag": prepare_crud,
    "duretrieval": lambda: prepare_cmteb("duretrieval", queries_n=1500, corpus_n=25000),
    "t2retrieval": lambda: prepare_cmteb("t2retrieval", queries_n=1500, corpus_n=20000),
    "multihop_rag": prepare_multihop,
}


def load_prepared(name: str) -> tuple[list[dict], dict]:
    folder = WORK / name / "prepared"
    return read_jsonl(folder / "questions.jsonl"), json.loads((folder / "PREPARED.json").read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", choices=sorted(PREPARE))
    args = parser.parse_args()
    for name in args.datasets:
        manifest = PREPARE[name]()
        print(name, json.dumps({k: manifest[k] for k in manifest if k not in ("tags",)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
