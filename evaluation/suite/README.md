# evaluation/suite — one evaluation path for every question set

Retrieval evaluation of the app's real pipeline (hybrid BM25 + dense, RRF, bge-reranker over the
whole pool, score cutoffs, per-question-type caps, per-document cap, parent expansion) on public
datasets and on the repository's own sets. Results and findings: `evaluation/suite_v1/REPORT.md`.

| File | What it does |
|---|---|
| `sources.json`, `fetch.py` | Public datasets, pinned revisions and checksums; downloads into `evaluation/external/<dataset>/` (gitignored). |
| `schema.py` | The JSONL schema of questions and corpora. |
| `adapters.py` | CRUD-RAG, DuRetrieval, T2Retrieval, MultiHop-RAG into the schema (fixed seed, held-out negatives). books / complex / benchmark are read in `runner.Legacy`. |
| `index.py` | Isolated index per dataset through `IngestionPipeline._commit` (no file registry). |
| `cache.py` | bge-m3 vectors and reranker scores keyed by text; offline mode. |
| `runner.py` | `HybridRetriever.retrieve` + `build_context` per question, every stage recorded and timed. `parts` in each record is the multi-part trace (`retrieval/multipart.py`): parts, per-part pools and scores, the adaptive round, every rerank request with its estimated tokens (see `evaluation/multidoc_v1`). |
| `metrics.py` | Recall/Precision/Hit/Success@K, MRR, nDCG@10 per stage (before and after the cut), negatives, latency percentiles. |
| `report.py` | Tables per type and tag, loss attribution, latency, where the app would route each question; `--compare` two runs; `evaluation/suite_v1/results/*.json`. |
| `replay.py` | Selection what-ifs (cutoffs, caps, per-document cap) replayed on recorded pools with the app's own `select_hits` / `diversify`. |
| `decompose.py` | What-if: multi-part questions reranked part by part (same reranker). |
| `embed_latency.py` | Live query-embedding latency on a fixed sample. |
| `examples.py` | Failure examples per type / tag / loss (prints dataset text: read locally, do not commit). |
| `ci_tier.py`, `fixtures/ci/` | The offline CI tier (no network, no key). `update` rewrites `expected.json` from the cached fixtures after a change that needs no new vectors or scores. |
| `fixes_compare.py` | Before / after the three fixes of REPORT.md section 5, as one table: retrieval, routing (questions the find-file cue sends away count as misses), negatives, file_search_v1 cues. |

## Running it

Python that reads downloaded data runs with `-I`. The C-MTEB files are parquet: install `pyarrow`
or point `EVAL_PYARROW_DIR` at a folder that has it.

```bash
PY=.venv/Scripts/python.exe
$PY -I evaluation/suite/fetch.py                       # download + verify (hf-mirror by default)
$PY -I evaluation/suite/adapters.py crud_rag duretrieval t2retrieval multihop_rag
$PY -I evaluation/suite/index.py crud_rag duretrieval t2retrieval multihop_rag
$PY -I evaluation/suite/runner.py crud_rag --name base --workers 3          # live: fills the caches
$PY -I evaluation/suite/runner.py crud_rag --name base --offline --workers 1 --tag latency
$PY -I evaluation/suite/report.py crud_rag:base duretrieval:base --latency latency
$PY -I evaluation/suite/runner.py books --name base                         # books_v1 on its index
$PY -I evaluation/suite/ci_tier.py check                                    # what CI runs
$PY -I evaluation/suite/runner.py crud_rag --name fixes --offline             # after a selection change: cached pools
$PY -I evaluation/suite/fixes_compare.py --after fixes                       # before / after table
```

Every run writes `<run>.meta.json` next to its records: commit (and whether the tree was dirty),
models, retrieval settings, dataset revision and checksums, seed. Only the SiliconFlow key and URL
are read from `project/.env`; models and retrieval settings are the code defaults (the suite stops
if a model differs from bge-m3 / bge-reranker-v2-m3 / Qwen3-8B).

`--set-option field=value` runs an ablation (e.g. `rerank_min_score=0.3`); a pool that changes is
scored from cached pairs plus one request for the new passages.

## Caveats

* The reranker scores a pair slightly differently depending on the rest of the request (up to
  ~0.005 in a check of five passages); the first live run scores each pool in one request, as the
  app does, and later runs reuse those scores.
* Query embedding and children embedding go through the same text-keyed cache, as the app embeds
  both with the same call; batch composition changes a vector by ~1e-3.
