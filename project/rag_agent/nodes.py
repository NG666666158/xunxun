"""LangGraph nodes for planning, retrieval, generation and verification."""

from __future__ import annotations

import json
import logging
import math
import re
import time
from datetime import datetime
from pathlib import Path, PureWindowsPath
from typing import Any

import openai

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.config import get_stream_writer

from config import Settings, get_settings
from core import timings
from retrieval.file_filters import (
    HISTORY_REFERENCE,
    QUESTION_WORDS,
    FileFilters,
    find_file_cue,
    merge_planner_filter,
    parse_query,
)
from retrieval.hybrid_retriever import HybridRetriever
from services.model_clients import MODEL_SERVICE_ERRORS, describe_model_error, transient_error
from .answering import (
    cited_claims,
    classify_question,
    evidence_for,
    literally_supported,
    profile_for,
    retrieval_limits,
)
from .document_nodes import DOCUMENT_INTENTS, DocumentNodes
from .graph_state import State
from . import multidoc_answer as multidoc
from .guards import (
    anchor_line,
    conflicting_values,
    hint as term_hint,
    missing_anchors,
    near_term_notes,
    note_present,
    rule_check,
    unanswered,
)
from .prompts import (
    CITATION_REMINDER,
    EXTRACT_FIRST_GUIDANCE,
    QUERY_PLAN_PROMPT,
    SAME_NAME_RULE,
    SEARCH_REWRITE_PROMPT,
    answer_prompt,
    fact_check_prompt,
)
from .schemas import FaithfulnessCheck, QueryPlan, SearchRewrite

logger = logging.getLogger(__name__)

SMALL_TALK = re.compile(
    r"^(?:你好|您好|嗨|哈喽|hi|hello|hey|早上好|晚上好|在吗|谢谢|多谢|感谢|再见|拜拜|好的|ok"
    r"|你是谁|你能做什么|你可以做什么|你会什么|你有什么功能)"
    r"[\s!！。,.，~?？]*(?:呀|啊|呢|啦|你)?[\s!！。,.，~?？]*$",
    re.IGNORECASE,
)

# A file kept by name only (library/privacy.py): its name, path and text never go to the model.
NAME_ONLY = "named"
NAME_ONLY_PLACEHOLDER = "〔只按文件名收录的文件〕"

# Files in a find_file answer: a topic's best matches, or a list by conditions alone.
MATCHED_FILES = 8
LISTED_FILES = 100
TYPE_NAMES = {"pdf": "PDF", "word": "Word", "excel": "Excel", "ppt": "PPT", "markdown": "Markdown",
              "text": "TXT", "image": "图片"}
# "共 3 篇论文", "共 5 张发票"; other document types are counted in 份.
TYPE_MEASURES = {"论文": "篇", "新闻": "篇", "照片": "张", "发票": "张", "书籍": "本"}

# Wording that may need the planner: a reference to an earlier turn.
AMBIGUOUS_REFERENCE = re.compile(r"(?:它|那个|上一个|上一款|这款|该产品|该服务)")


class AgentNodes(DocumentNodes):
    def __init__(
        self,
        llm,
        retriever: HybridRetriever,
        settings: Settings | None = None,
        parent_store=None,
        file_search=None,
        documents=None,
        summary_llm=None,
        document_dates=None,
    ):
        self.llm = llm
        self.retriever = retriever
        self.settings = settings or get_settings()
        self.parent_store = parent_store
        self.file_search = file_search
        self.documents = documents
        self.summary_llm = summary_llm
        # doc ids -> {doc id: "YYYY-MM-DD"} (dated_evidence); None: the library's file dates.
        self.document_dates = document_dates

    def analyze_query(self, state: State) -> dict[str, Any]:
        question = self._current_question(state)
        history = self._conversation_context(state.get("messages", []))
        scope = state.get("scope_doc_ids") or []
        document = self._document_turn(question, state)
        if document is not None:
            plan = document.plan
        elif SMALL_TALK.match(question.strip()):
            # "你好" / "谢谢" / "你能做什么": the planner could only agree (see _enforce_plan_policy).
            plan = QueryPlan(intent="chitchat", rewritten_query=question[:500])
        elif self.file_search is not None and len(scope) != 1 and self._obviously_find_file(question):
            # "…在哪" / "上周加的…" / a bare file title: unambiguous and self-contained, so the
            # rules alone extract the filters and the planner's LLM round trip is skipped.
            plan = QueryPlan(intent="find_file", rewritten_query=question[:500])
        elif not history and self._plain_knowledge_question(question):
            # A first-turn question with no file or reference cue is searched as
            # asked: the planner would only copy it, at the cost of a model round trip.
            plan = QueryPlan(intent="knowledge", rewritten_query=question[:500])
        else:
            plan = self._plan(question, history)
            if plan.intent in DOCUMENT_INTENTS:
                document = self._planner_document_turn(plan, question)
                plan = document.plan if document else plan.model_copy(update={"intent": "knowledge"})
        if plan.intent == "find_file" and (len(scope) == 1 or self.file_search is None):
            # Asking within one file is always about its content.
            plan = plan.model_copy(update={"intent": "knowledge"})
        profile = classify_question(question)
        if profile == "fact" and plan.rewritten_query != question:
            # "展开讲讲" is classified by what it expands into.
            profile = classify_question(plan.rewritten_query)
        return {
            "question": question,
            "intent": plan.intent,
            "rewritten_query": plan.rewritten_query,
            "plan": plan.model_dump(),
            "answer_profile": profile,
            "retrieval_attempts": 0,
            "retrieve_chunks": [],
            "documents": [],
            "context_docs": [],
            "evidence": "",
            "answer": "",
            "citations": [],
            "grounded": False,
            "refused": False,
            "unsupported_claims": [],
            "last_error": "",
            "model_error": "",
            "check_skipped": False,
            "check_mode": "",
            "term_notes": [],
            "anchor_notes": [],
            "conflict_notes": [],
            "rejection_notes": [],
            "unanswered_refusal": False,
            "answer_steps": {},
            "files": [],
            "file_search": {},
            "target_file": document.target if document else {},
            "pending_file": document.pending if document else {},
            "choices": document.choices if document else [],
            "coverage": {},
            "turn_done": False,
            "table_query": {},
            "table_fallback": False,
            "recent_doc_ids": self._recent_doc_ids(state) if history else [],
        }

    @staticmethod
    def _recent_doc_ids(state: State) -> list[str]:
        """The files the previous turn cited, found or summarized (still in the state now)."""
        ids = [str(item.get("doc_id") or "") for item in state.get("citations") or []]
        ids += [str(item.get("file_id") or "") for item in state.get("files") or []]
        ids.append(str((state.get("target_file") or {}).get("file_id") or ""))
        return [i for i in dict.fromkeys(ids) if i][:8]

    # A plan is a few dozen tokens; the cap only stops a runaway reply.
    plan_max_tokens = 300

    def _plan(self, question: str, history: str) -> QueryPlan:
        try:
            plan = self._invoke_structured(
                QueryPlan,
                [
                    SystemMessage(content=QUERY_PLAN_PROMPT),
                    HumanMessage(content=f"最近对话：\n{history or '(无)'}\n\n当前问题：\n{question}"),
                ],
                compact=True,
                max_tokens=self.plan_max_tokens,
            )
        except ValueError as exc:
            # A reply that doesn't parse says nothing about the question: search it as asked.
            logger.warning("Query plan unusable, searching the question as asked: %s", exc)
            return QueryPlan(intent="knowledge", rewritten_query=question[:500])
        except MODEL_SERVICE_ERRORS as exc:
            # The planner is an optional step: a rate limit or a dropped connection here must not
            # end the turn (the answer step waits or reports the failure itself).
            logger.warning("Query planner unavailable, searching the question as asked: %r", exc)
            return QueryPlan(intent="knowledge", rewritten_query=question[:500])
        return self._enforce_plan_policy(question, plan)

    @staticmethod
    def _plain_knowledge_question(question: str) -> bool:
        if len(question) > 500 or find_file_cue(question) is not None or HISTORY_REFERENCE.search(question):
            return False
        if len(question) <= 30 and not QUESTION_WORDS.search(question):
            # "北方区考勤办法": a bare title may be a file lookup; the planner decides.
            return False
        return not (SMALL_TALK.match(question.strip()) or AMBIGUOUS_REFERENCE.search(question))

    def _obviously_find_file(self, question: str) -> bool:
        if HISTORY_REFERENCE.search(question):
            return False
        cue = find_file_cue(question)
        if cue == "strong":
            return True
        # "面试题200问", "北方区考勤办法": a short title, not a question, naming an existing file.
        if len(question) > 30 or QUESTION_WORDS.search(question):
            return False
        # "Word 文档", "上周的 PDF": nothing but conditions on files.
        filters = parse_query(question)
        if cue and filters.has_metadata and not filters.topic:
            return True
        try:
            return self.file_search.best_name_match(question) >= 0.75
        except Exception as exc:
            logger.warning("File name lookup failed: %s", exc)
            return False

    @staticmethod
    def _enforce_plan_policy(question: str, plan: QueryPlan) -> QueryPlan:
        """Clamp probabilistic planner output to what the question's wording supports."""
        updates: dict[str, Any] = {}

        # Finding files needs some sign of it in the question ("在哪", "哪份", "PDF", "资料");
        # "KV Cache 的原理是什么" is a knowledge question whatever the planner says.
        cue = find_file_cue(question)
        if plan.intent == "find_file" and cue is None:
            updates.update(intent="knowledge")
        elif plan.intent == "knowledge" and cue == "strong":
            updates.update(intent="find_file")

        # Only plain greetings, thanks and "what can you do" skip retrieval;
        # anything else the planner calls chitchat is searched anyway.
        if plan.intent == "chitchat" and not SMALL_TALK.match(question.strip()):
            updates.update(intent="knowledge")

        ambiguous_reference = AMBIGUOUS_REFERENCE.search(question)
        if plan.needs_clarification and not ambiguous_reference:
            updates.update(needs_clarification=False, clarification_question="")
        return plan.model_copy(update=updates) if updates else plan

    def clarification(self, state: State) -> dict[str, Any]:
        message = state.get("plan", {}).get("clarification_question") or "请补充你所指的具体制度、产品或服务名称。"
        return {"answer": message, "messages": [AIMessage(content=message)]}

    def chitchat(self, state: State) -> dict[str, Any]:
        message = (
            "你好！我会基于你知识库里的文档回答问题，并标出每条内容的出处。"
            "可以直接问文档里的概念和细节，也可以让我总结、对比某个主题，或者对上一条回答继续追问。"
        )
        return {
            "answer": message,
            "messages": [AIMessage(content=message)],
            "grounded": False,
            "refused": False,
        }

    def retrieve(self, state: State) -> dict[str, Any]:
        attempts = state.get("retrieval_attempts", 0) + 1
        scope = state.get("scope_doc_ids") or []
        # The question type caps how many reranked passages are kept (score cutoffs may keep
        # fewer) and, for questions of several parts, how many may come from one document.
        limits = self._retrieval_limits(state.get("answer_profile"))
        recent = state.get("recent_doc_ids") or []
        if recent and not scope and getattr(self.settings, "history_prior", False) and self._follow_up(state):
            # "那它的工艺有几步？": the files of the previous answer are searched too and preferred.
            limits["prefer_doc_ids"] = list(recent)
            if self._elliptical(state.get("question", "")):
                limits["prefer_only"] = True
        try:
            if scope:
                hits = self.retriever.retrieve(
                    state["rewritten_query"], mode="hybrid_rerank", doc_ids=set(scope), **limits
                )
            else:
                hits = self.retriever.retrieve(state["rewritten_query"], mode="hybrid_rerank", **limits)
            documents = [hit.to_dict() for hit in hits]
            return {
                "retrieval_attempts": attempts,
                "retrieve_chunks": documents,
                "documents": documents,
                "context_docs": [],
                "last_error": "",
            }
        except Exception as exc:
            logger.exception("Knowledge retrieval failed")
            return {
                "retrieval_attempts": attempts,
                "retrieve_chunks": [],
                "documents": [],
                "context_docs": [],
                "last_error": f"{type(exc).__name__}: {exc}",
            }

    def _retrieval_limits(self, profile: str | None) -> dict[str, Any]:
        """``retrieval_limits`` with the fact-question levers (fact_max_passages, fact_relative_score)."""
        limits: dict[str, Any] = retrieval_limits(profile)
        if profile == "fact":
            keep = int(getattr(self.settings, "fact_max_passages", 0) or 0)
            if keep > 0 and keep != limits["max_keep"]:
                limits["max_keep"] = keep
            relative = float(getattr(self.settings, "fact_relative_score", 0.0) or 0.0)
            if relative > 0:
                limits["relative"] = relative
        return limits

    @staticmethod
    def _follow_up(state: State) -> bool:
        """A question that leans on the previous turn: a reference to it, a question the
        planner completed from it, or a short one."""
        question = state.get("question", "")
        return bool(
            HISTORY_REFERENCE.search(question)
            or AMBIGUOUS_REFERENCE.search(question)
            or (state.get("rewritten_query") and state.get("rewritten_query") != question)
            or len(question) <= 20
        )

    @staticmethod
    def _elliptical(question: str) -> bool:
        """A follow-up that names nothing of its own: a reference ("它 / 这份 / 刚才"), or a short
        question without a proper name ("用了哪几步制备工艺？", not "深圳的人口是多少？")."""
        from retrieval.question_parts import _has_name

        if HISTORY_REFERENCE.search(question) or AMBIGUOUS_REFERENCE.search(question):
            return True
        return len(question) <= 20 and not _has_name(question)

    def search_files(self, state: State) -> dict[str, Any]:
        """Rank files for a find_file question; the answer is the list itself."""
        question = state.get("question", "")
        if self.file_search is None:
            message = "当前没有可用的文件检索服务。"
            return {"answer": message, "messages": [AIMessage(content=message)], "refused": True, "files": []}
        # Relative dates ("上周") are resolved against the search service's clock.
        now = getattr(self.file_search, "clock", datetime.now)()
        filters = merge_planner_filter(parse_query(question, now), state.get("plan", {}).get("file_filter"), now)
        scope = set(state.get("scope_doc_ids") or []) or None
        # Only conditions, or only a document type ("有哪些合同"): the list is the answer, so it
        # can run long (the UI folds it).
        listing = bool(getattr(self.file_search, "lists_type", lambda _: False)(filters))
        if hasattr(self.file_search, "as_searched"):
            filters = self.file_search.as_searched(filters)
        limit = LISTED_FILES if listing or not filters.topic.strip() else MATCHED_FILES
        try:
            result = self.file_search.search(question, filters, limit=limit, scope=scope)
        except Exception as exc:
            logger.exception("File search failed")
            message = "文件检索暂时不可用，请稍后再试。"
            return {
                "answer": message,
                "messages": [AIMessage(content=message)],
                "refused": True,
                "files": [],
                "last_error": f"{type(exc).__name__}: {exc}",
            }
        files = result["files"]
        message = self._describe_files(files, filters, result["total"], result["unreadable"],
                                       typed=result.get("typed") if listing else None)
        return {
            "answer": message,
            # The answer is shown as it is; what later turns send to the model leaves out the
            # names and paths of files kept by name only (design/PRIVACY.md).
            "messages": [AIMessage(content=self._without_name_only(message, files))],
            "files": files,
            "file_search": {key: result[key] for key in ("filters", "elapsed_ms", "total", "total_candidates", "unreadable")},
            "grounded": bool(files),
            "refused": not files,
        }

    @staticmethod
    def _without_name_only(message: str, files: list[dict[str, Any]]) -> str:
        for item in files:
            if item.get("status") == NAME_ONLY:
                for text in (item.get("path"), item.get("name"), Path(str(item.get("name") or "")).stem):
                    if text:
                        message = message.replace(str(text), NAME_ONLY_PLACEHOLDER)
        return message

    @staticmethod
    def _describe_files(
        files: list[dict[str, Any]],
        filters: FileFilters,
        total: int | None = None,
        unreadable: int = 0,
        typed: int | None = None,
    ) -> str:
        """The answer's text. ``typed``: a list of every file of a document type ("有哪些合同"),
        of which that many are of the type."""
        total = len(files) if total is None else total
        conditions = []
        if filters.time_range:
            start, end = filters.to_dict()["time_range"]
            field = {"added": "加入时间", "modified": "修改时间"}.get(filters.time_field, "时间")
            conditions.append(f"{field} {start} 至 {end}")
        kinds = "/".join(TYPE_NAMES.get(label, label.upper()) for label in filters.file_types)
        if filters.folder:
            conditions.append(f"文件夹含“{filters.folder}”")
        skipped = f"，另有 {unreadable} 个没能读取" if unreadable else ""
        if typed:
            # "有哪些合同": how many there are, not "related files"; the files that only came
            # up by their content are listed after them. None of the type: the usual wording.
            word = filters.doc_type_word if re.search(r"[一-鿿]", filters.doc_type_word) else filters.doc_type
            noun = f" {kinds} {word}" if kinds else word
            condition = f"（{'；'.join(conditions)}）" if conditions else ""
            head = f"共 {typed} {TYPE_MEASURES.get(filters.doc_type, '份')}{noun}{condition}{skipped}"
            if filters.newest_first and typed > 1:
                head += f"，按{'加入' if filters.time_field == 'added' else '修改'}时间从新到旧"
            if total > typed:
                head += f"，另有 {total - typed} 个相关文件"
            head += f"，列出 {len(files)} 个：" if total > len(files) else "："
        elif not filters.topic.strip():
            # Only conditions: the exact count of what meets them, then the newest of those.
            noun = "图片" if kinds == "图片" else f" {kinds} 文件" if kinds else "文件"
            condition = f"（{'；'.join(conditions)}）" if conditions else ""
            if not files:
                return f"没有符合条件的{noun}{condition}{skipped}。"
            head = f"共 {total} 个{noun}{condition}{skipped}"
            head += f"，列出最新的 {len(files)} 个：" if total > len(files) else "："
        else:
            if kinds:
                conditions.insert(1 if filters.time_range else 0, f"类型 {kinds}")
            condition = f"（条件：{'；'.join(conditions)}）" if conditions else ""
            if not files:
                return f"没有找到符合条件的文件{condition}。可以换个说法或放宽条件再试。"
            head = f"找到 {total} 个相关文件{condition}"
            head += f"，列出最相关的 {len(files)} 个：" if total > len(files) else "："
        lines = [head]
        for index, item in enumerate(files[:MATCHED_FILES], start=1):
            locator = item["snippets"][0].get("locator", "") if item.get("snippets") else ""
            page = re.fullmatch(r"page:(\d+(?:-\d+)?)", locator)
            where = f"（第 {page.group(1)} 页）" if page else (f"（{locator}）" if locator else "")
            by = item.get("matched_by") or []
            if "type" in by:
                # "找一下我的简历": the file is one (its card says so), whatever its name.
                where = f"（类型：{item.get('type_label') or item.get('doc_type') or filters.doc_type}）"
            elif "mention" in by and filters.doc_type_word:
                where += f"（提到“{filters.doc_type_word}”）"
            lines.append(f"{index}. {item['name']}{where} — {item.get('path') or ''}".rstrip(" —"))
        return "\n".join(lines)

    def rewrite_search(self, state: State) -> dict[str, Any]:
        try:
            result = self._invoke_structured(
                SearchRewrite,
                [
                    SystemMessage(content=SEARCH_REWRITE_PROMPT),
                    HumanMessage(
                        content=f"原问题：{state['question']}\n初次查询：{state['rewritten_query']}"
                    ),
                ],
            )
            query = result.query
        except Exception as exc:
            logger.warning("Search rewrite failed, using original question: %s", exc)
            query = state["question"]
        return {
            "rewritten_query": query,
            "retrieve_chunks": [],
            "documents": [],
            "context_docs": [],
            "last_error": "",
        }

    def build_context(self, state: State) -> dict[str, Any]:
        """Expand child hits to unique parents and enforce the context budget."""
        children = state.get("retrieve_chunks") or state.get("documents", [])
        selected: list[dict[str, Any]] = []
        seen: set[str] = set()
        used_tokens = 0
        # Fact lookups get the top passages only; explanations and comparisons get more.
        budget = min(self.settings.context_token_budget, profile_for(state.get("answer_profile")).context_budget)

        for child in children:
            parent_id = str(child.get("parent_id", ""))
            if parent_id in seen:
                continue
            seen.add(parent_id)
            parent = None
            if self.parent_store and parent_id:
                try:
                    parent = self.parent_store.get(parent_id)
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    logger.warning("Failed to load parent %s: %s", parent_id, exc)

            context = dict(child)
            context["child_text"] = str(child.get("text", ""))
            if parent:
                context.update(parent)
                context["score"] = child.get("score", 0.0)
                context["rerank_score"] = child.get("rerank_score")
                context["retrieval_method"] = child.get("retrieval_method", "")

            token_count = self._estimate_tokens(str(context.get("text", "")))
            if selected and used_tokens + token_count > budget:
                continue
            selected.append(context)
            used_tokens += token_count
            if used_tokens >= budget:
                break

        selected = self._label_same_names(selected)
        if getattr(self.settings, "dated_evidence", False) and multidoc.temporal_question(state.get("question", "")):
            selected = self._with_dates(selected)
        return {"context_docs": selected}

    def _with_dates(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Each passage with its file's date ("date"), shown in the evidence for questions about
        time order (evaluation/multidoc_answer_v1/REPORT.md, candidate 2)."""
        ids = {str(document.get("doc_id", "")) for document in documents}
        try:
            dates = self._document_dates(ids)
        except Exception as exc:
            logger.warning("File dates unavailable: %s", exc)
            return documents
        return [{**document, "date": dates[key]} if (key := str(document.get("doc_id", ""))) in dates else document
                for document in documents]

    def _document_dates(self, doc_ids: set[str]) -> dict[str, str]:
        """{doc id: "YYYY-MM-DD"}: the given lookup (a corpus's publish dates), else the library's
        file modification dates."""
        if self.document_dates is not None:
            found = self.document_dates(doc_ids) or {}
        elif getattr(self, "documents", None) is not None:
            found = {str(row["file_id"]): row.get("mtime") for row in self.documents.rows() if str(row["file_id"]) in doc_ids}
        else:
            return {}
        return {key: text for key, value in found.items() if (text := multidoc.date_text(value))}

    def _label_same_names(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Same-named files from different folders ("04 答辩PPT.pptx" of two projects) get their
        folder after the name, so the answer can tell them apart instead of mixing them
        (evaluation/multidoc_v1/REPORT.md, method 5). The citation keeps the doc id."""
        by_name: dict[str, set[str]] = {}
        for document in documents:
            by_name.setdefault(str(document.get("source", "")), set()).add(str(document.get("doc_id", "")))
        shared = {name for name, ids in by_name.items() if len(ids) > 1}
        access = getattr(self, "documents", None)
        if not shared or access is None:
            return documents
        try:
            paths = {str(row["file_id"]): str(row.get("path") or "") for row in access.rows()}
        except Exception as exc:
            logger.warning("Folders of same-named files unavailable: %s", exc)
            return documents
        labelled = []
        for document in documents:
            name = str(document.get("source", ""))
            folder = PureWindowsPath(paths.get(str(document.get("doc_id", "")), "")).parent.name if name in shared else ""
            if folder:
                document = {**document, "source": f"{name}（{folder}）", "same_name": True}
            labelled.append(document)
        return labelled

    def refuse(self, state: State) -> dict[str, Any]:
        if state.get("model_error"):
            message = f"这次没能生成回答。{state['model_error']}"
        elif state.get("last_error"):
            message = "检索服务暂时不可用，这次没能查资料。请稍后再试。"
        elif self._relevant_evidence(state):
            # Passages were found, but no answer written from them passed the fact check.
            message = "资料里的内容不足以可靠地回答这个问题，所以没有作答。可以换个更具体的问法再试。"
        else:
            message = self._nothing_found(state)
        return {
            "answer": message,
            "messages": [AIMessage(content=message)],
            "grounded": False,
            "refused": True,
        }

    def _relevant_evidence(self, state: State) -> bool:
        """Did the last search return passages relevant enough to answer from (route_after_retrieval)?"""
        documents = state.get("documents") or []
        if not documents:
            return False
        top = documents[0].get("rerank_score")
        return top is None or float(top) >= getattr(self.settings, "relevance_threshold", 0.0)

    @staticmethod
    def _nothing_found(state: State) -> str:
        """The plain reply when the search found nothing that answers the question."""
        if state.get("table_fallback") and (state.get("target_file") or {}).get("name"):
            return f"《{Path(state['target_file']['name']).stem}》里没有找到能回答这个问题的内容。"
        if state.get("scope_doc_ids"):
            return "在限定的文件范围里没有找到能回答这个问题的内容。可以取消范围限定后再问。"
        return "资料里没有找到能回答这个问题的内容。可以换个说法再问，或者把相关的文件加进资料库。"

    def generate_answer(self, state: State) -> dict[str, Any]:
        evidence, citations = self._build_evidence(state)
        if not evidence:
            message = self._nothing_found(state)
            return {
                "answer": message,
                "messages": [AIMessage(content=message)],
                "grounded": False,
                "refused": True,
            }
        history = self._conversation_context(state.get("messages", []))
        profile = profile_for(state.get("answer_profile"))
        settings = self.settings
        guidance = profile.guidance
        if profile.name == "fact" and getattr(settings, "extract_first", False):
            guidance = EXTRACT_FIRST_GUIDANCE
        question = state["question"]
        whole_file = state.get("intent") == "summarize_file"
        verdicts = str(getattr(settings, "conclusion_line", "off") or "off")
        multi_steps = not whole_file and (verdicts != "off" or any(
            getattr(settings, name, False) for name in ("part_answer", "citation_repair")
        ))
        shape = multidoc.question_shape(question) if multi_steps else ""
        steps: dict[str, Any] = {"shape": shape} if shape else {}
        if any(document.get("date") for document in state.get("context_docs") or []):
            steps["dated"] = sum(bool(document.get("date")) for document in state.get("context_docs") or [])
        if getattr(settings, "part_answer", False) and not whole_file:
            parts = multidoc.part_mode(question, shape)
            if parts:
                # The evidence grouped per part, in the same single call (candidate 1).
                blocks, _ = self._evidence_parts(state)
                groups = multidoc.group_passages(parts, [block["text"] for block in self._evidence_blocks(state)])
                evidence = multidoc.grouped_evidence(groups, blocks)
                guidance = multidoc.PART_GUIDANCE_YESNO if shape == "yesno" else multidoc.PART_GUIDANCE
                steps["parts"] = len(parts)
        conclusion = verdicts != "off" and shape in ("yesno", "choice")
        if conclusion and verdicts == "prompt":
            guidance += multidoc.conclusion_rule(question, shape)
        # The citation rule is restated last: with the length guidance alone, small models
        # often dropped the citations of shorter answers.
        prompt = (
            f"问题：\n{question}\n\n证据：\n{evidence}\n\n篇幅要求：{guidance}\n"
            f"引用要求：{CITATION_REMINDER}"
        )
        notes = []
        if getattr(settings, "term_note", False) and state.get("intent") != "summarize_file":
            notes = near_term_notes(state["question"], [block["text"] for block in self._evidence_blocks(state)])
            if notes:
                prompt += "\n" + term_hint(notes)
        if any(document.get("same_name") for document in state.get("context_docs") or []):
            prompt += f"\n{SAME_NAME_RULE}"
        if history:
            prompt = f"最近对话：\n{history}\n\n{prompt}"
        system = answer_prompt(plain_refusal=bool(getattr(settings, "plain_refusal", False)),
                               conflict_note=bool(getattr(settings, "conflict_note", False)),
                               subject_rule=bool(getattr(settings, "subject_rule", False)))
        try:
            response = self._call_llm(
                [SystemMessage(content=system), HumanMessage(content=prompt)], max_tokens=profile.max_tokens,
                patience=float(getattr(settings, "fact_check_patience_seconds", 0.0) or 0.0),
            )
        except MODEL_SERVICE_ERRORS as exc:
            logger.warning("Answer generation failed: %r", exc)
            return {"answer": "", "evidence": evidence, "citations": citations, "model_error": describe_model_error(exc),
                    "answer_steps": steps}
        answer = str(response.content).strip()
        if getattr(settings, "citation_repair", False) and not whole_file:
            # A statement citing a passage without its numbers or names cites the one that has
            # them (candidate 4; no model call). The fact check then compares the right passages.
            answer, repaired = multidoc.repair_citations(answer, self._evidence_blocks(state), question)
            if repaired:
                steps["repaired"] = repaired
        # The checks below read the model's own answer, not the conclusion line added to it
        # (each was measured that way: evaluation/rejection_v1, generation_v1).
        body = answer
        if conclusion:
            answer, steps["conclusion"] = self._with_conclusion(question, answer, shape, verdicts == "check")
        lines: list[str] = []
        anchors: list[str] = []
        conflicts: list[str] = []
        gaps = None
        scope = str(getattr(settings, "rejection_note", "off") or "off")
        if scope != "off" and state.get("intent") != "summarize_file":
            # What the question pins down (a year, place, numbered subject, name) or the answer's
            # own value, missing from the evidence: the documents have the topic, not the answer.
            gaps = unanswered(state["question"], body, self._evidence_blocks(state), scope=scope)
            if gaps is not None and gaps.every and getattr(settings, "rejection_refusal", False):
                # Nothing the question names is in the evidence: say so instead of answering.
                message = gaps.refusal()
                return {"answer": message, "evidence": evidence, "citations": [], "rejection_notes": [message],
                        "grounded": False, "refused": True, "unanswered_refusal": True}
        if getattr(settings, "conflict_finder", False) and state.get("intent") != "summarize_file":
            # The answer's value for the question's fact against a different value another passage
            # about the same subject gives ("10.48‰" / "5.48‰"): found by the code, not the model.
            blocks = self._evidence_blocks(state)
            found = conflicting_values(state["question"], body, blocks)
            if found:
                # One plain line added by the code: no model call, and the fact check does not see
                # it (it cites nothing). Having the model write the answer again with both values
                # was measured and did worse (evaluation/generation_v1/REPORT.md).
                conflicts = [conflict.line(blocks) for conflict in found]
                answer = "\n".join(conflicts) + "\n\n" + answer
        if getattr(settings, "anchor_note", False) and state.get("intent") != "summarize_file":
            # A place, year, name or title the answer talks about that none of its cited passages
            # mention: the answer may be about a similar but different subject. Said, not hidden.
            missing = missing_anchors(state["question"], body, self._evidence_blocks(state))
            if missing:
                anchors = [anchor_line(missing)]
                answer = anchors[0] + "\n\n" + answer
        if notes:
            answer, lines = self._with_term_notes(answer, notes, self._evidence_blocks(state))
        rejection: list[str] = []
        if gaps is not None:
            # The first sentence says what the documents lack; the answer stays below it, so the
            # user sees what the documents do say. It replaces the anchor line, which says less.
            rejection = [gaps.line()]
            if anchors:
                answer = answer.replace(anchors[0] + "\n\n", "", 1)
                anchors = []
            answer = rejection[0] + "\n\n" + answer
        return {"answer": answer, "evidence": evidence, "citations": citations, "term_notes": lines,
                "anchor_notes": anchors, "conflict_notes": conflicts, "rejection_notes": rejection,
                "answer_steps": steps}

    # The verdict call reads only the question and the answer: a few hundred tokens in, one word out.
    verdict_max_tokens = 10

    def _with_conclusion(self, question: str, answer: str, shape: str, check_only: bool = False) -> tuple[str, str]:
        """The answer opening with its normalised conclusion line (candidate 3): the model's own
        line, or one taken from its first sentence; an answer that gives no verdict at all costs
        one short call that reads the answer and names it (是 / 否 / 资料不足). ``check_only``: an
        answer that already gives its verdict in its own words is left as it is."""
        if check_only and multidoc.gives_verdict(question, answer, shape):
            return answer, "had"
        result = multidoc.check_conclusion(question, answer, shape)
        if result.status != "missing" or (shape == "choice" and len(multidoc.choice_options(question)) != 2):
            return result.answer, result.status
        try:
            # Not streamed: the draft shows the answer itself, the final answer has the line.
            llm = self.llm.with_config(tags=["nostream"]) if hasattr(self.llm, "with_config") else self.llm
            reply = self._call_llm(
                [SystemMessage(content=multidoc.VERDICT_PROMPT),
                 HumanMessage(content=multidoc.verdict_request(question, answer, shape))],
                max_tokens=self.verdict_max_tokens, llm=llm,
                patience=float(getattr(self.settings, "fact_check_patience_seconds", 0.0) or 0.0),
            )
        except MODEL_SERVICE_ERRORS as exc:
            logger.warning("Verdict call failed, the answer keeps its own wording: %r", exc)
            return answer, "missing"
        asked = multidoc.with_verdict(question, answer, str(reply.content), shape)
        return asked.answer, asked.status

    @staticmethod
    def _with_term_notes(answer: str, notes, blocks: list[dict[str, str]]) -> tuple[str, list[str]]:
        """Open the answer with the line of each near term found in a passage it cites (in any
        passage when it cites none), unless the answer already says so itself."""
        cited = {i for _, refs in cited_claims(answer) for i in evidence_for(refs, blocks)[0]}
        lines = [note.line() for note in notes if (not cited or note.passage in cited) and not note_present(answer, note)]
        if not lines:
            return answer, []
        return "\n".join(lines) + "\n\n" + answer, lines

    def fact_check(self, state: State) -> dict[str, Any]:
        answer = state.get("answer", "")
        blocks = self._evidence_blocks(state)
        claims = cited_claims(answer)
        resolved = [(claim, *evidence_for(refs, blocks)) for claim, refs in claims]
        unresolved = [claim for claim, _, ok in resolved if not ok]
        if getattr(self.settings, "answer_mode", "strict") == "fast":
            # Fast mode: no model call; every citation must point at evidence that was given.
            grounded = bool(claims) and not unresolved
            missing = unresolved or ([] if claims else ["回答没有标注引用出处"])
            return {"grounded": grounded, "unsupported_claims": missing, "check_skipped": False,
                    "check_mode": "citations"}
        patience = float(getattr(self.settings, "fact_check_patience_seconds", 0.0) or 0.0)
        subject = bool(getattr(self.settings, "subject_rule", False))
        question = state.get("question", "")
        if getattr(self.settings, "rule_precheck", False):
            # A number or model name that is in no passage needs no model to be found out.
            problems = rule_check(answer, [block["text"] for block in blocks], question)
            if problems:
                return {"grounded": False, "unsupported_claims": problems, "check_skipped": False, "check_mode": "rules"}
        if claims:
            # A statement that repeats the passage it cites, numbers and negations as they are,
            # needs no model to confirm it; only the others are sent, each with its passages.
            literal = {
                n for n, (claim, indexes, ok) in enumerate(resolved)
                if ok and indexes and literally_supported(claim, [blocks[i]["text"] for i in indexes])
            }
            pending = [item for n, item in enumerate(resolved) if n not in literal]
            timings.note(claims=len(resolved), literal_claims=len(literal))
            if not pending:
                return {"grounded": True, "unsupported_claims": [], "check_skipped": False, "check_mode": "literal"}
            used = sorted({i for _, indexes, _ in pending for i in indexes})
            number = {index: n for n, index in enumerate(used, start=1)}
            evidence = "\n\n".join(
                f"[E{number[i]}｜来源：{blocks[i]['source']}｜位置：{blocks[i]['locator']}]\n{blocks[i]['text']}" for i in used
            )
            lines = [
                f"{n}. {claim}（依据：{'、'.join(f'E{number[i]}' for i in indexes) or '无'}）"
                for n, (claim, indexes, _) in enumerate(pending, start=1)
            ]
            system, content, mode = (
                fact_check_prompt(True, subject),
                f"证据：\n{evidence or '(无)'}\n\n待核对的陈述：\n" + "\n".join(lines),
                "cited",
            )
        else:
            # No citations at all (a "没有找到依据" reply, or the model forgot them): check it whole.
            system, content, mode = (
                fact_check_prompt(False, subject), f"证据：\n{state.get('evidence', '')}\n\n回答：\n{answer}", "full",
            )
        try:
            result = self._invoke_structured(
                FaithfulnessCheck, [SystemMessage(content=system), HumanMessage(content=content)], patience=patience
            )
            if mode == "cited" and not result.grounded and result.unsupported_claims:
                # A second look before the answer is thrown away: the flagged statements
                # against all evidence, as the whole-answer check judged them. A statement
                # that cites the wrong passage of the evidence is not an invented fact.
                everything = "\n\n".join(
                    f"[E{n}｜来源：{b['source']}｜位置：{b['locator']}]\n{b['text']}" for n, b in enumerate(blocks, start=1)
                )
                flagged = "\n".join(f"{n}. {claim}" for n, claim in enumerate(result.unsupported_claims, start=1))
                result = self._invoke_structured(FaithfulnessCheck, [
                    SystemMessage(content=fact_check_prompt(True, subject)),
                    HumanMessage(content=f"证据：\n{everything}\n\n待核对的陈述（依据：以上全部证据）：\n{flagged}"),
                ], patience=patience)
                mode = "cited+all"
            unsupported = [*result.unsupported_claims, *(f"引用的出处不在检索证据中：{c[:80]}" for c in unresolved)]
            return {
                "grounded": result.grounded and not unresolved,
                "unsupported_claims": unsupported,
                "check_skipped": False,
                "check_mode": mode,
            }
        except MODEL_SERVICE_ERRORS as exc:
            # The checker was unreachable, which says nothing about the answer:
            # show it marked as unverified instead of re-running the whole search.
            logger.warning("Fact check skipped, model service failed: %r", exc)
            if patience > 0:
                # Still not possible after waiting: the rule check decides what it can (a number
                # or model name in no passage fails the answer), and the answer says how it was checked.
                problems = rule_check(answer, [block["text"] for block in blocks], question)
                if problems:
                    return {"grounded": False, "unsupported_claims": problems, "check_skipped": False,
                            "check_mode": "rules"}
                return {
                    "grounded": False,
                    "unsupported_claims": ["事实校验未完成，只按规则核对了数字和名称：" + describe_model_error(exc)],
                    "check_skipped": True,
                    "check_mode": "rules",
                }
            return {
                "grounded": False,
                "unsupported_claims": ["事实校验未完成：" + describe_model_error(exc)],
                "check_skipped": True,
            }
        except Exception as exc:
            logger.warning("Fact check failed closed: %s", exc)
            return {"grounded": False, "unsupported_claims": ["事实校验服务未成功完成"], "check_skipped": False}

    def finish(self, state: State) -> dict[str, Any]:
        answer = state.get("answer", "")
        if state.get("intent") == "summarize_file":
            return self._finish_summary(state, answer)
        return {"messages": [AIMessage(content=answer)]}

    # Rate limits are per minute, so waiting a few seconds usually clears them.
    rate_limit_backoff: tuple[float, ...] = (4.0, 8.0, 16.0)
    # With patience (fact_check_patience_seconds): the longest single wait after those.
    patience_step: float = 20.0
    # With patience, the wall time allowed beyond it: about one timed-out request.
    patience_slack: float = 30.0

    def _backoff(self, patience: float) -> list[float]:
        """Waits between attempts: the usual three, then (with patience) steps of up to
        ``patience_step`` until ``patience`` seconds are used up (a per-minute token limit frees
        up within a minute)."""
        delays = list(self.rate_limit_backoff)
        while patience > 0 and sum(delays) < patience:
            delays.append(min(self.patience_step, patience - sum(delays)))
        return delays

    def _call_llm(self, messages: list[Any], max_tokens: int | None = None, llm: Any = None, patience: float = 0.0):
        """Rate limits are retried (4 / 8 / 16 s). With ``patience`` the retries go on until that
        many seconds have passed, and a timed-out or dropped request or a 5xx is retried too:
        the check is only given up when the service stayed unusable the whole time."""
        llm = llm if llm is not None else self.llm
        llm = llm.bind(max_tokens=max_tokens) if max_tokens and hasattr(llm, "bind") else llm
        waited = 0.0
        started = time.monotonic()
        for delay in (*self._backoff(patience), None):
            try:
                return llm.invoke(messages)
            except Exception as exc:
                limited = isinstance(exc, openai.RateLimitError)
                if not (limited or (patience > 0 and transient_error(exc))) or delay is None:
                    raise
                if patience > 0 and time.monotonic() - started + delay > patience + self.patience_slack:
                    raise
                logger.warning("Model %s, retrying in %.0fs", "rate limited" if limited else f"failed ({exc!r})", delay)
                if patience > 0:
                    self._progress({"kind": "waiting", "reason": "rate_limit" if limited else "busy",
                                    "waited": round(waited), "seconds": round(delay)})
                time.sleep(delay)
                waited += delay

    @staticmethod
    def _progress(event: dict[str, Any]) -> None:
        """A progress event for the UI (nothing outside a streamed graph run)."""
        try:
            get_stream_writer()(event)
        except Exception:
            pass

    @staticmethod
    def _current_question(state: State) -> str:
        for message in reversed(state.get("messages", [])):
            if isinstance(message, HumanMessage):
                return str(message.content).strip()
        return state.get("question", "").strip()

    @staticmethod
    def _conversation_context(messages: list[Any]) -> str:
        lines = []
        for message in messages[-6:-1]:
            role = "用户" if isinstance(message, HumanMessage) else "助手"
            content = str(getattr(message, "content", "")).strip()
            if content:
                lines.append(f"{role}: {content[:500]}")
        return "\n".join(lines)

    def _invoke_structured(
        self, schema, messages: list[Any], *, compact: bool = False, max_tokens: int | None = None, patience: float = 0.0
    ):
        """Request JSON once and always validate it with Pydantic.

        SiliconFlow is OpenAI-compatible, but free models differ in native
        function-call support. Prompted JSON avoids a failed tool-call request
        plus retry while retaining strict structured-output validation.

        ``compact`` asks the model to leave out fields at their defaults: every output token
        costs time (a full query plan is ~90 tokens, most of them empty fields), and the
        schema fills the defaults back in.
        """
        rule = "只输出一个 JSON 对象，不要 Markdown 代码块。"
        if compact:
            rule += "取默认值的字段（空字符串、空列表、false）不要输出，只写用到的字段。"
        json_instruction = HumanMessage(
            content=rule + "JSON 必须符合这个 schema：\n" + json.dumps(schema.model_json_schema(), ensure_ascii=False)
        )
        response = self._call_llm([*messages, json_instruction], max_tokens=max_tokens, patience=patience)
        content = str(response.content).strip()
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            raise ValueError(f"模型没有返回可解析的 {schema.__name__} JSON")
        return schema.model_validate_json(match.group(0))

    @staticmethod
    def _build_evidence(state: State) -> tuple[str, list[dict[str, str]]]:
        # A summary is written from the notes read_whole_file made; only the fact check reads
        # the passages. Any other question builds its evidence anew: a retry searched again.
        if state.get("intent") == "summarize_file" and state.get("evidence"):
            return state["evidence"], state.get("citations", [])
        blocks, citations = AgentNodes._evidence_parts(state)
        return "\n\n".join(blocks), citations

    @staticmethod
    def _evidence_parts(state: State) -> tuple[list[str], list[dict[str, str]]]:
        """Each passage as the answer model sees it ("[证据 n｜来源：…｜位置：…]" and its text, with
        its file date first when it has one), and the citations."""
        blocks: list[str] = []
        citations: list[dict[str, str]] = []
        documents = state.get("context_docs") or state.get("documents", [])
        for index, document in enumerate(documents, start=1):
            source = str(document.get("source", "unknown"))
            locator = str(document.get("locator") or document.get("heading") or "未标注位置")
            text = multidoc.dated_text(str(document.get("text", "")), str(document.get("date") or ""))
            blocks.append(f"[证据 {index}｜来源：{source}｜位置：{locator}]\n{text}")
            citation = {"source": source, "locator": locator, "doc_id": str(document.get("doc_id", ""))}
            if not any(all(seen.get(key) == value for key, value in citation.items()) for seen in citations):
                # The passage's opening lets the reader highlight it in the original.
                citations.append({**citation, "quote": str(document.get("text", ""))[:200]})
        return blocks, citations

    @staticmethod
    def _evidence_blocks(state: State) -> list[dict[str, str]]:
        """The passages the answer was generated from, as the evidence numbered them (with the
        file date the answer model saw, so the checks see it too)."""
        documents = state.get("context_docs") or state.get("documents", [])
        return [
            {
                "source": str(d.get("source", "unknown")),
                "locator": str(d.get("locator") or d.get("heading") or "未标注位置"),
                "text": multidoc.dated_text(str(d.get("text", "")), str(d.get("date") or "")),
            }
            for d in documents
        ]

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Conservative tokenizer-free estimate for mixed Chinese/Latin text."""
        cjk = len(re.findall(r"[\u3400-\u9fff]", text))
        non_cjk = max(0, len(text) - cjk)
        return cjk + math.ceil(non_cjk / 4)
