"""Explicit LangGraph workflow for bounded, evidence-grounded question answering."""

from __future__ import annotations

from functools import partial

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from config import Settings, get_settings
from core.timings import timed_node
from .edges import (
    route_after_analysis,
    route_after_fact_check,
    route_after_generation,
    route_after_reading,
    route_after_retrieval,
    route_after_table,
)
from .graph_state import State
from .nodes import AgentNodes


def create_agent_graph(
    llm,
    retriever,
    settings: Settings | None = None,
    checkpointer=None,
    parent_store=None,
    file_search=None,
    documents=None,
    summary_llm=None,
    document_dates=None,
):
    settings = settings or get_settings()
    nodes = AgentNodes(
        llm=llm,
        retriever=retriever,
        settings=settings,
        parent_store=parent_store,
        file_search=file_search,
        documents=documents,
        summary_llm=summary_llm,
        document_dates=document_dates,
    )
    graph = StateGraph(State)

    graph.add_node("analyze_query", timed_node("analyze_query", nodes.analyze_query))
    graph.add_node("clarification", timed_node("clarification", nodes.clarification))
    graph.add_node("chitchat", timed_node("chitchat", nodes.chitchat))
    graph.add_node("retrieve", timed_node("retrieve", nodes.retrieve))
    graph.add_node("search_files", timed_node("search_files", nodes.search_files))
    graph.add_node("read_whole_file", timed_node("read_whole_file", nodes.read_whole_file))
    graph.add_node("query_table", timed_node("query_table", nodes.query_table))
    graph.add_node("build_context", timed_node("build_context", nodes.build_context))
    graph.add_node("rewrite_search", timed_node("rewrite_search", nodes.rewrite_search))
    graph.add_node("refuse", timed_node("refuse", nodes.refuse))
    graph.add_node("generate_answer", timed_node("generate_answer", nodes.generate_answer))
    graph.add_node("fact_check", timed_node("fact_check", nodes.fact_check))
    graph.add_node("finish", timed_node("finish", nodes.finish))

    graph.add_edge(START, "analyze_query")
    graph.add_conditional_edges("analyze_query", route_after_analysis)
    graph.add_edge("clarification", END)
    graph.add_edge("chitchat", END)
    # Finding files needs no generation or fact check.
    graph.add_edge("search_files", END)
    graph.add_conditional_edges("read_whole_file", route_after_reading)
    graph.add_conditional_edges("query_table", route_after_table)
    graph.add_conditional_edges(
        "retrieve",
        partial(route_after_retrieval, settings=settings),
    )
    graph.add_edge("rewrite_search", "retrieve")
    graph.add_edge("build_context", "generate_answer")
    graph.add_edge("refuse", END)
    graph.add_conditional_edges("generate_answer", route_after_generation)
    graph.add_conditional_edges(
        "fact_check",
        partial(route_after_fact_check, settings=settings),
    )
    graph.add_edge("finish", END)

    return graph.compile(checkpointer=checkpointer or InMemorySaver())
