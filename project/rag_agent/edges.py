from __future__ import annotations

from config import Settings, get_settings
from .graph_state import State


def route_after_analysis(state: State) -> str:
    if state.get("temporal") and not state["temporal"]["doc_ids"]:
        return "refuse"
    if state.get("plan", {}).get("needs_clarification"):
        return "clarification"
    return {
        "knowledge": "retrieve",
        "find_file": "search_files",
        "chitchat": "chitchat",
        "summarize_file": "read_whole_file",
        "table_query": "query_table",
    }.get(state.get("intent", "knowledge"), "retrieve")


def route_after_reading(state: State) -> str:
    if state.get("turn_done"):
        return "__end__"
    return "generate_answer" if state.get("evidence") else "refuse"


def route_after_table(state: State) -> str:
    if state.get("table_fallback"):
        return "retrieve"
    if state.get("model_error"):
        return "refuse"
    return "finish"


def route_after_retrieval(state: State, settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    documents = state.get("documents", [])
    if documents:
        top_score = documents[0].get("rerank_score")
        if top_score is None or float(top_score) >= settings.relevance_threshold:
            return "build_context"
    if state.get("retrieval_attempts", 0) < settings.max_retrieval_attempts:
        return "rewrite_search"
    return "refuse"


def route_after_generation(state: State) -> str:
    # A model-service failure is not an evidence problem; retrying the search won't help.
    if state.get("model_error"):
        return "refuse"
    # The code found nothing the question names in the evidence (rejection_refusal): nothing to check.
    return "finish" if state.get("unanswered_refusal") else "fact_check"


def route_after_fact_check(state: State, settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    # A summary is written from the whole file: searching again could only make it narrower.
    if state.get("intent") == "summarize_file":
        return "finish"
    # Fast mode only checks the citations and never re-runs the search.
    if state.get("grounded") or state.get("check_skipped") or state.get("check_mode") == "citations":
        return "finish"
    if state.get("retrieval_attempts", 0) < settings.max_retrieval_attempts:
        return "rewrite_search"
    return "refuse"
