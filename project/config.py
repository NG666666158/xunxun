"""Central, environment-driven configuration for the enterprise RAG demo.

Keeping settings here makes every model, storage and retrieval choice visible
in one place. Secrets are read from ``project/.env`` and never hard-coded.
"""

from functools import lru_cache
import os
from pathlib import Path
import sys
from typing import Literal

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent

# Folder name under %APPDATA% for the packaged desktop app; also the keyring service name.
APP_DATA_NAME = "EnterpriseKnowledgeBase"


def is_frozen() -> bool:
    """True inside the PyInstaller build, where the repository layout does not exist."""
    return bool(getattr(sys, "frozen", False))


def default_data_dir(frozen: bool | None = None, environ: dict[str, str] | None = None) -> Path:
    """Repository ``data/`` in development; ``%APPDATA%/<APP_DATA_NAME>`` when packaged."""
    environ = os.environ if environ is None else environ
    if not (is_frozen() if frozen is None else frozen):
        return REPO_DIR / "data"
    base = environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    return Path(base) / APP_DATA_NAME


# Paths that follow ``data_dir`` unless set explicitly (field -> name inside data_dir).
_DATA_DIR_PATHS = {
    "embedded_db_path": "knowledge.sqlite3",
    "conversations_db_path": "conversations.sqlite3",
    "upload_dir": "uploads",
    "parsed_dir": "parsed",
    "parent_store_dir": "parents",
    "cache_dir": "cache",
    "log_dir": "logs",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "寻寻"
    app_version: str = "1.0.2"
    log_level: str = "INFO"

    # Desktop shell: every request except the liveness probe must carry this per-launch
    # token. Empty (development) disables the check.
    app_token: str = Field(default="", repr=False, validation_alias=AliasChoices("RAG_APP_TOKEN"))

    siliconflow_api_key: str = Field(
        default="",
        repr=False,
        validation_alias=AliasChoices("SILICONFLOW_API_KEY", "LLM_API_KEY"),
    )
    siliconflow_base_url: str = Field(
        default="https://api.siliconflow.cn/v1",
        validation_alias=AliasChoices("SILICONFLOW_BASE_URL", "LLM_BASE_URL"),
    )
    llm_model: str = "Qwen/Qwen3-8B"
    embedding_model: str = "BAAI/bge-m3"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    embedding_dimension: int = 1024
    llm_temperature: float = 0.0
    llm_max_tokens: int = 3000
    request_timeout_seconds: float = 30.0
    request_max_retries: int = 1
    # How the model service is reached when a system proxy is set (services/model_clients.py):
    # "auto" (direct when it connects, else the proxy), "direct" or "proxy".
    model_connection: Literal["auto", "direct", "proxy"] = "auto"
    # A chat call with no first token after this many seconds is sent again alongside; the
    # first to answer is used (0 = wait for the timeout, then retry).
    first_token_hedge_seconds: float = 8.0

    # "embedded" keeps chunks, BM25 postings and vectors in one SQLite file (no Docker);
    # "milvus" needs a running Milvus Standalone and the optional pymilvus package.
    vector_backend: Literal["embedded", "milvus"] = "embedded"
    embedded_db_path: Path = Path("knowledge.sqlite3")
    # Conversation history and the agent state behind each conversation.
    conversations_db_path: Path = Path("conversations.sqlite3")

    # Watched-folder file registry. Defaults to the embedded store's SQLite file so the
    # file rows and their chunks live (and later sync) together.
    file_registry_path: Path | None = None
    watch_enabled: bool = True
    watch_debounce_seconds: float = 2.0
    watch_max_files: int = 50000
    library_monitor_seconds: float = 30.0
    # Single imported files are not watched: a stat sweep checks them this often (and the
    # UI asks for a check when the window regains focus).
    library_sweep_seconds: float = 180.0
    # While a question is being answered the indexing queue waits, at most this long per pause.
    index_yield_max_seconds: float = 120.0
    # Files indexed side by side. Recognition (local OCR, vision models) is capped: at most
    # index_ocr_jobs such files at once, of which index_long_jobs long ones (scanned PDFs);
    # the other workers keep reading text files. Background calls to SiliconFlow (embeddings,
    # vision) are limited to index_api_concurrency at once across all of them.
    index_workers: int = 4
    index_ocr_jobs: int = 2
    index_long_jobs: int = 1
    index_api_concurrency: int = 4
    # Embedding tokens a minute the library starts at, and the most it may grow to
    # (ingestion/embed_pacing.py): SiliconFlow refuses past the account's limit (about 500k a
    # minute for bge-m3 on the account measured, evaluation/import_speed_v1); the pacer finds it.
    index_embed_tpm: float = 450_000
    index_embed_tpm_max: float = 4_000_000
    index_embed_burst: float = 1_500_000

    milvus_uri: str = "http://127.0.0.1:19530"
    milvus_token: str = "root:Milvus"
    milvus_collection: str = "enterprise_knowledge_chunks"

    # Everything the app writes lives under data_dir; the paths below default to
    # data_dir/<name> (see _DATA_DIR_PATHS) unless configured explicitly.
    data_dir: Path = Field(
        default_factory=default_data_dir,
        validation_alias=AliasChoices("RAG_DATA_DIR", "DATA_DIR"),
    )
    upload_dir: Path = Path("uploads")
    parsed_dir: Path = Path("parsed")
    parent_store_dir: Path = Path("parents")
    cache_dir: Path = Path("cache")
    log_dir: Path = Path("logs")
    evaluation_dir: Path = REPO_DIR / "evaluation"

    parent_chunk_size: int = 1800
    child_chunk_size: int = 700
    child_chunk_overlap: int = 120
    # Prefix every retrieval child with "document · section" so searches that name the
    # document or chapter ("面试题里有哪些…") can match; parents stay plain text.
    chunk_context_prefix: bool = True
    # Figures and image-only PDF pages: "off", "local" (RapidOCR, nothing is uploaded) or
    # "cloud" (SiliconFlow vision models; page images are sent to the API). See figures.py.
    image_understanding: Literal["off", "local", "cloud"] = "off"
    # With "local": also read formulas with RapidLaTeXOCR (an optional ~180 MB download).
    local_formula: bool = False
    vision_model: str = "Qwen/Qwen3-VL-32B-Instruct"
    vision_ocr_model: str = "PaddlePaddle/PaddleOCR-VL-1.5"
    vision_concurrency: int = 6
    # Image files (.png, .jpg, ...) are read per file in the mode chosen when they were
    # added: "local" (RapidOCR), "cloud" (vision model; the image is uploaded) or "name"
    # (file name and date only). image_import_mode is the remembered choice; "ask" lets the
    # UI ask once per import. Images are scaled to image_max_side pixels before reading.
    image_import_mode: Literal["ask", "local", "cloud", "name"] = "ask"
    image_max_side: int = 2000
    # Candidates: each route (vector, BM25) returns at least dense/sparse_top_k, and the
    # fused best fusion_top_k go to the reranker. With candidate_auto_scale the pool grows
    # by 25% per doubling of the library beyond 20k chunks, up to candidate_pool_max.
    # Chosen in evaluation/retrieval_tuning_v1/REPORT.md.
    dense_top_k: int = 40
    sparse_top_k: int = 40
    fusion_top_k: int = 40
    candidate_auto_scale: bool = True
    candidate_pool_max: int = 80
    # After reranking, keep the hits scoring at least rerank_min_score and at least
    # rerank_relative_score x the best score: no fewer than rerank_min_keep, no more than
    # rerank_top_k (each question type has its own lower maximum, see answering.PROFILES),
    # and at most rerank_per_document from one document while other documents qualify (0: no cap).
    rerank_top_k: int = 12
    rerank_min_keep: int = 3
    rerank_min_score: float = 0.1
    rerank_relative_score: float = 0.1
    rerank_per_document: int = 4
    # When even the best reranked passage scores below this, there is no evidence: nothing
    # is kept (rerank_min_keep does not apply) and the answer says so plainly
    # (evaluation/suite_v1/REPORT.md, fix 3). 0 turns it off.
    rerank_no_evidence_below: float = 0.1
    # Questions of several parts, answers that span documents (retrieval/multipart.py;
    # measured in evaluation/multidoc_v1/REPORT.md). Defaults are the report's recommendation.
    # * multipart_doc_first: for a multi-part question, the best passage of each of the first
    #   N documents above multipart_doc_floor is kept first (no extra call).
    # * part_retrieval "adaptive": when no kept passage mentions a part's own words, that part
    #   is searched on its own (one more round, only then; "always" searches every part, in
    #   parallel with the question's own search). part_rerank "whole": the part's first
    #   part_rerank_top new candidates are scored against the question itself and join its
    #   ranking, so the documents-first cut picks them up; "part": they are scored against the
    #   part, and those scoring at least part_min_score are kept, part_quota per part.
    # * bridge_terms (names / numbers / titles of the first passages searched again with BM25)
    #   and neighbor_parents (the parents next to the best N documents' hits) measured little
    #   gain and stay off.
    part_retrieval: Literal["off", "always", "adaptive"] = "adaptive"
    part_rerank: Literal["part", "whole"] = "whole"
    part_search_top: int = 20
    part_rerank_top: int = 6
    part_min_score: float = 0.1
    part_quota: int = 2
    multipart_doc_first: int = 8
    multipart_doc_floor: float = 0.01
    bridge_terms: bool = False
    bridge_top: int = 6
    neighbor_parents: int = 0
    # A file the question names ("答辩PPT用了哪几步制备工艺", "《X》里…") answers it when it holds
    # evidence, other files only when it does not; a multi-part question keeps the named
    # file's best passage among the others (retrieval/naming.py).
    prefer_named_files: bool = True
    # A follow-up question ("那它的工艺有几步？") also searches the files the previous answer cited
    # and adds history_prior_boost to their passages' rerank scores.
    history_prior: bool = True
    history_prior_boost: float = 0.05
    rrf_k: int = 60
    relevance_threshold: float = 0.20
    max_retrieval_attempts: int = 2
    context_token_budget: int = 9000
    sse_chunk_size: int = 12
    # "strict": a model checks every cited statement against its passages (a failed check
    # searches again); "fast": only the citations are checked, with no extra model call.
    answer_mode: Literal["fast", "strict"] = "strict"
    # Stream the answer draft while it is generated; `final` is still sent after fact check.
    stream_draft: bool = True
    # ---- Answer levers for weak answer models (evaluation/generation_v1/REPORT.md); each is
    # switched on or off on its own.
    # Rate limits, timeouts and 5xx are waited out, up to this many seconds per call (plus one
    # request), for the answer and its fact check, with a progress note; a check that still
    # cannot run is replaced by the rule check (numbers, model names, 《titles》 against the
    # passages), and the answer says it was only checked that way. 0: three short retries on a
    # rate limit, then the answer is shown unchecked (1.2.0).
    fact_check_patience_seconds: float = 75.0
    # The rule check before the model check: a cited statement whose number or model name is
    # in no passage fails the check without a model call.
    rule_precheck: bool = False
    # A key term of the question the evidence only has as a near term ("质保期" asked,
    # "保质期" written): the answer opens with one line saying which word the documents use.
    # Retrieval and passage selection are unchanged (near terms are found on purpose).
    term_note: bool = True
    # Fact questions: most reranked passages kept, and the fraction of the best rerank score a
    # passage needs (0: rerank_relative_score, as for every question type).
    fact_max_passages: int = 6
    fact_relative_score: float = 0.0
    # Fact questions quote the evidence sentence first, then answer.
    extract_first: bool = False
    # The answer points out passages that contradict each other instead of picking one (a rule in
    # the answer prompt; the model's judgment).
    conflict_note: bool = False
    # The same found by the code (guards.conflicting_values): the value the answer gives for the
    # question's fact (a number with its unit, a date, a name) against a different one that
    # another passage about the same subject gives. The answer opens with one line naming both
    # and where each is written; no model call.
    conflict_finder: bool = True
    # The answer model's own "nothing in the evidence" reply uses the app's wording
    # ("资料里没有找到…").
    plain_refusal: bool = False
    # The answer and the fact check require the evidence to be about the subject asked (the same
    # place, year, project, product), not a similar one.
    subject_rule: bool = False
    # A place, year, name or 《title》 of the question that the answer talks about but none of its
    # cited passages mention: the answer opens with one caution line naming it.
    anchor_note: bool = True
    # The library has the question's topic but not its answer (guards.unanswered,
    # evaluation/rejection_v1/REPORT.md): the answer opens with one line saying what the documents
    # lack ("资料里没有找到2022年的内容，下面的回答讲的可能是2023年的情况…"), and stays below it.
    # "all": a year, month, numbered subject ("第七届"), place, person, company or 《title》 of the
    # question that no passage names; "cited": the same in the passages the answer cites; "value":
    # "all", and the answer's own value in no passage, or only where the passages talk about another
    # year / place / edition, or not as the role or measure asked (导演, 冠军, 增长…); "off".
    rejection_note: Literal["off", "all", "cited", "value"] = "value"
    # With rejection_note: when nothing the question names is in the evidence, the reply is that
    # line alone ("资料里没有找到…，所以没有作答") instead of an answer below it.
    rejection_refusal: bool = False

    # ---- Answers drawing on several documents (rag_agent/multidoc_answer.py;
    # evaluation/multidoc_answer_v1/REPORT.md); each on or off on its own. Defaults are the
    # report's recommendation: check + citation repair + dates, 23 better / 0 worse over 260
    # end-to-end questions (MultiHop temporal 10 -> 30 / 50, comparison 27 -> 30 / 50).
    # A question asking several things (or a yes / no question of several parts) gets its
    # evidence grouped per part in the one answer call and is answered part by part. Off: with
    # the other three it cost MultiHop 30 -> 27 on both types.
    part_answer: bool = False
    # A question about time ("before / after", "10月12日的报道") sees each passage's file date.
    dated_evidence: bool = True
    # Yes / no and either-or questions get a verdict: "prompt" asks the answer to open with one
    # conclusion line ("结论：是"), checked and normalised by the code; "check" leaves the prompt as
    # it is and only reads the answer's own verdict. Either way an answer that gives none costs one
    # short model call that names it, and the answer opens with that line. "prompt" changed
    # answers that were right (comparison 8 better / 10 worse); "check" did not.
    conclusion_line: Literal["off", "prompt", "check"] = "check"
    # A cited statement whose passage lacks its numbers or names cites the passage that has them.
    citation_repair: bool = True

    # ---- Finding files by document type ("找一下我的简历": the word 简历 is in neither the file
    # name nor its text). Measured in evaluation/doc_cards_v1/REPORT.md; each is its own switch.
    # * doc_cards: a chat-model call reads a file's name and first doc_card_chars characters
    #   and writes its card: 类型 / 对象 / 时间 / 摘要 (retrieval/doc_cards.py). Built in the
    #   background, waiting while a question is answered; never for files kept by name only,
    #   nor for those whose own name gives the type ("01 申报书.pdf"). Files the user is looking
    #   at (listed by a search, opened, likely matches of a type question) go first, then the
    #   newest. doc_card_batch > 1 lets that many files share one call: about 35 % fewer tokens,
    #   but Qwen3-8B then got the type right for 0.70 to 0.75 of the public set instead of 0.89,
    #   so it stays 1. Paced by tokens: doc_card_tokens_per_minute of the account's chat-model
    #   budget, doc_card_idle_tokens_per_minute once nobody has asked anything for ten minutes,
    #   nothing for doc_card_cool_down_seconds after a question; at least
    #   doc_card_interval_seconds between calls. Measured in evaluation/doc_cards_after_import.
    # * doc_type_search: a question naming a type ranks files whose card has that type first;
    #   files that only mention the word come after them (needs the cards).
    # * doc_card_owner: "我的简历" puts cards about owner_name (the user's own name) first.
    doc_cards: bool = True
    doc_card_chars: int = 900
    doc_card_interval_seconds: float = 3.0
    doc_card_batch: int = 1
    doc_card_tokens_per_minute: float = 9000.0
    doc_card_idle_tokens_per_minute: float = 15000.0
    doc_card_cool_down_seconds: float = 60.0
    doc_type_search: bool = True
    doc_card_owner: bool = False
    owner_name: str = ""
    # Follow-up questions after an answer: one short model call ("" = the answer model),
    # abandoned for local templates after this many seconds.
    followup_model: str = ""
    followup_timeout_seconds: float = 4.0
    followup_max_tokens: int = 200
    # "总结一下《X》" reads the whole file (rag_agent/whole_doc.py). summary_model writes the
    # notes ("" = the answer model); the answer itself always comes from the answer model.
    # Above summary_confirm_tokens of unread text the app says what it will cost and asks.
    summary_model: str = ""
    summary_concurrency: int = 6
    summary_batch_tokens: int = 4000
    summary_reduce_tokens: int = 12000
    summary_confirm_tokens: int = 150000

    # Verified copies of the databases in data_dir/backups (core/data_safety.py).
    backup_keep: int = 3
    backup_interval_hours: float = 24.0
    backup_first_delay_seconds: float = 600.0

    api_base_url: str = "http://127.0.0.1:8000"
    # Browser uploads are copied into data_dir, so they stay small.
    max_upload_mb: int = 20
    # Files registered in place are never copied. Chosen from time and memory on big files
    # (evaluation/doc_capabilities_v1/REPORT.md).
    max_local_file_mb: int = 1024
    max_local_text_mb: int = 50
    allowed_extensions: tuple[str, ...] = (
        ".pdf", ".docx", ".xlsx", ".pptx", ".ppt", ".md", ".txt",
        ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff",
    )

    langfuse_enabled: bool = False
    langfuse_public_key: str = Field(default="", repr=False)
    langfuse_secret_key: str = Field(default="", repr=False)
    langfuse_base_url: str = "https://cloud.langfuse.com"

    @model_validator(mode="after")
    def derive_data_paths(self):
        self.data_dir = Path(self.data_dir).expanduser()
        for field, name in _DATA_DIR_PATHS.items():
            if field not in self.model_fields_set:
                setattr(self, field, self.data_dir / name)
        if "evaluation_dir" not in self.model_fields_set and is_frozen():
            self.evaluation_dir = self.data_dir / "evaluation"
        return self

    @field_validator("child_chunk_overlap")
    @classmethod
    def validate_overlap(cls, value: int, info):
        size = info.data.get("child_chunk_size", 700)
        if value < 0 or value >= size:
            raise ValueError("child_chunk_overlap must be >= 0 and smaller than child_chunk_size")
        return value

    @field_validator(
        "parent_chunk_size",
        "child_chunk_size",
        "context_token_budget",
        "max_retrieval_attempts",
        "sse_chunk_size",
        "max_upload_mb",
        "backup_keep",
        "max_local_file_mb",
        "max_local_text_mb",
        "dense_top_k",
        "sparse_top_k",
        "fusion_top_k",
        "candidate_pool_max",
        "rerank_top_k",
        "rerank_min_keep",
    )
    @classmethod
    def validate_positive_integer(cls, value: int, info):
        if value <= 0:
            raise ValueError(f"{info.field_name} must be greater than zero")
        return value

    def ensure_directories(self) -> None:
        for path in (
            self.data_dir,
            self.upload_dir,
            self.parsed_dir,
            self.parent_store_dir,
            self.cache_dir,
            self.log_dir,
            self.evaluation_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    @property
    def registry_path(self) -> Path:
        return self.file_registry_path or self.embedded_db_path

    def require_api_key(self) -> str:
        """The current key. Model clients call this per request, so a key saved or removed
        in Settings applies to the next request without rebuilding anything."""
        if not self.siliconflow_api_key or self.siliconflow_api_key.startswith("sk-your-"):
            raise MissingApiKeyError(
                "缺少 SILICONFLOW_API_KEY。请在“设置”页填写，或复制 project/.env.example 为 project/.env 后填写密钥。"
            )
        return self.siliconflow_api_key


class MissingApiKeyError(RuntimeError):
    """No SiliconFlow key is configured (never entered, or removed in Settings)."""


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    from core.user_settings import apply_user_settings

    settings = Settings()
    settings.ensure_directories()
    # The key saved from the Settings page (Windows Credential Manager) and the chosen
    # model override project/.env.
    apply_user_settings(settings)
    return settings
