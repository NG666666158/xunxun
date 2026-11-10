"""Paths, JSONL helpers, settings and provenance shared by the evaluation suite.

Every result file records ``provenance()``: the commit (and whether the tree had
uncommitted changes), the three models, the retrieval settings, the dataset's version
and checksum, and the sample seed.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Iterable, Iterator

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROJECT = ROOT / "project"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

EXTERNAL = Path(os.environ.get("EVAL_EXTERNAL_DIR", ROOT / "evaluation" / "external"))
# Indexes, caches and per-question records: large and regenerable, never committed.
WORK = Path(os.environ.get("EVAL_WORK_DIR", ROOT / "evaluation" / "suite_v1" / "work"))
RESULTS = ROOT / "evaluation" / "suite_v1" / "results"

# The pipeline is measured with these models; the suite refuses to run with others.
MODELS = {
    "embedding_model": "BAAI/bge-m3",
    "rerank_model": "BAAI/bge-reranker-v2-m3",
    "llm_model": "Qwen/Qwen3-8B",
}
# Settings that shape retrieval, recorded with every result.
RETRIEVAL_FIELDS = (
    "parent_chunk_size", "child_chunk_size", "child_chunk_overlap", "chunk_context_prefix",
    "dense_top_k", "sparse_top_k", "fusion_top_k", "candidate_auto_scale", "candidate_pool_max",
    "rerank_top_k", "rerank_min_keep", "rerank_min_score", "rerank_relative_score",
    "rerank_per_document", "rerank_no_evidence_below", "rrf_k", "context_token_budget", "embedding_dimension",
    "part_retrieval", "part_rerank", "part_search_top", "part_rerank_top", "part_min_score", "part_quota",
    "multipart_doc_first", "multipart_doc_floor", "bridge_terms", "bridge_top", "neighbor_parents", "prefer_named_files",
)


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def iter_jsonl(path: Path) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def text_key(model: str, text: str) -> str:
    return hashlib.sha1(f"{model}\n{text}".encode("utf-8")).hexdigest()


def index_doc_id(dataset: str, doc_id: str) -> str:
    """The store's document id (32 hex characters, as the app's) for a dataset's document."""
    return hashlib.sha256(f"{dataset}\n{doc_id}".encode("utf-8")).hexdigest()[:32]


def settings(data_dir: Path | None = None, **overrides: Any):
    """The app's Settings with its code defaults for retrieval and the pinned models.

    Only the SiliconFlow key and URL come from project/.env (or the environment); the
    rest of .env (a different model, a smaller budget) is not applied, so a run measures
    the pipeline as shipped. ``overrides`` changes single fields (e.g. for an ablation).
    """
    from config import Settings

    env = Settings()
    values: dict[str, Any] = {
        "siliconflow_api_key": env.siliconflow_api_key,
        "siliconflow_base_url": env.siliconflow_base_url,
        **MODELS,
    }
    if data_dir is not None:
        values["data_dir"] = Path(data_dir)
    values.update(overrides)
    return Settings(_env_file=None, **values)


def git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            return ""

    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("status", "--porcelain", "--untracked-files=no", "--", "project", "evaluation/suite"))}


def provenance(config, dataset: dict[str, Any] | None = None, **extra: Any) -> dict[str, Any]:
    """What a result depends on. ``dataset``: name, version, checksum, seed, sizes."""
    return {
        "git": git_state(),
        "models": {key: getattr(config, key) for key in MODELS},
        "retrieval_settings": {field: getattr(config, field) for field in RETRIEVAL_FIELDS},
        "dataset": dataset or {},
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        **extra,
    }


def check_models(config) -> None:
    for key, expected in MODELS.items():
        if getattr(config, key) != expected:
            raise SystemExit(f"{key} is {getattr(config, key)!r}; the suite measures {expected!r}")
