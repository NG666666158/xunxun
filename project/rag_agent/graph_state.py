from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class State(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    intent: str
    rewritten_query: str
    plan: dict[str, Any]
    retrieval_attempts: int
    retrieve_chunks: list[dict[str, Any]]
    documents: list[dict[str, Any]]
    context_docs: list[dict[str, Any]]
    evidence: str
    answer: str
    citations: list[dict[str, str]]
    grounded: bool
    refused: bool
    unsupported_claims: list[str]
    last_error: str
    # User-facing reason when the model service (not the evidence) failed.
    model_error: str
    # The fact check could not run (rate limit, network); the answer is shown unverified.
    check_skipped: bool
    # How the answer was checked: "cited" (cited statements vs. the passages they cite),
    # "full" (whole answer vs. all evidence) or "citations" (fast mode: citations only).
    check_mode: str
    # Lines opening the answer about near terms (config term_note): "资料里写的是“保质期…”，没有提到“质保期”。"
    term_notes: list[str]
    anchor_notes: list[str]
    # Two passages giving the question's fact different values (config conflict_finder):
    # "注意：资料里的说法不一致，《a》写的是“X”，《b》写的是“Y”，请点开出处核对。"
    conflict_notes: list[str]
    # What the question pins down that the evidence lacks (config rejection_note): the answer opens
    # with "资料里没有找到2022年的内容，下面的回答讲的可能是2023年的情况，请点开出处核对。"
    rejection_notes: list[str]
    # Nothing the question names is in the evidence and rejection_refusal is on: the answer is
    # that line alone, with no fact check.
    unanswered_refusal: bool
    # What the multi-document answer steps did (rag_agent/multidoc_answer.py): "shape",
    # "parts" (answered part by part), "dated" (passages with a file date), "conclusion"
    # (model / derived / asked / missing / …), "repaired" (statements whose citation moved).
    answer_steps: dict[str, Any]
    # "fact" | "multi" | "explain" | "compare" ("summary" for a whole file): how many passages
    # are kept, the context budget and the answer length (answering.PROFILES).
    answer_profile: str
    # "Ask within this file/folder": retrieval is limited to these documents (empty = all).
    # Sent with every request, never carried over from an earlier turn.
    scope_doc_ids: list[str]
    # Result of a find_file question: ranked files with their best passages.
    files: list[dict[str, Any]]
    file_search: dict[str, Any]
    # A question about one file as a whole (rag_agent/document_nodes.py): the file; what was
    # read of it; and a question waiting for the reply on the next turn ("哪一份？").
    target_file: dict[str, Any]
    coverage: dict[str, Any]
    pending_file: dict[str, Any]
    # The node already wrote the whole reply (a question back, or why it cannot read the file).
    turn_done: bool
    # Replies offered as follow-ups for a question this turn asked back.
    choices: list[str]
    table_query: dict[str, Any]
    table_fallback: bool
    # Files the previous turn cited, found or summarized: a follow-up question searches them
    # as well and slightly prefers them (settings.history_prior; retrieval/multipart.py).
    recent_doc_ids: list[str]
