"""Questions about one file as a whole: "总结一下《X》" and "《X》这张表一共多少条".

Both read the file itself instead of searching passages (``whole_doc`` and
``services.table_query``). Which file is decided here, from the question and the registry,
without a model call: 《书名号》, a file's name written out, or the file the user is asking
within. Same-named files are asked about; the reply is read on the next turn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
import re
from typing import Any, Callable

import openai
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.config import get_stream_writer
from pydantic import ValidationError

from services.model_clients import MODEL_SERVICE_ERRORS, describe_model_error
from services.table_query import (
    TableError,
    TableQuery,
    WorkbookCache,
    answer_text,
    is_table_file,
    run_query,
    schema_text,
    sheet_asked_for,
)
from .answering import cited_claims, evidence_for
from .graph_state import State
from .prompts import TABLE_PLAN_PROMPT
from .schemas import QueryPlan
from .whole_doc import (
    CONFIRM,
    SUMMARY_CUE,
    NoteStore,
    compress_notes,
    describe_coverage,
    distinct_files,
    estimate_reading,
    files_named_in,
    folder_label,
    match_files,
    narrow_by_folder,
    pick_choice,
    read_passages,
    wants_whole_file,
)

logger = logging.getLogger(__name__)

SUMMARIZE, TABLE = "summarize_file", "table_query"
DOCUMENT_INTENTS = (SUMMARIZE, TABLE)
TABLE_CUE = re.compile(
    r"多少|几条|几行|几个|几项|几家|几种|总数|总共|一共|合计|总计|求和|总和|平均|均值|最大|最小|最高|最低|"
    r"统计|分组|分类汇总|分别|各有|每个|每类|每种|列出|列一下|筛选|有哪些|哪些|排序|前\s*\d+|分布|占比"
)
# "这份 / 这个 / 该 / 刚才那份": the file of the previous turn.
THIS_FILE = re.compile(r"这份|这个|这篇|这本|这张|该|此|刚才|刚刚|上面|上述|那份|那个")
# The kind of file a question names, by extension.
KIND_WORDS = {
    "ppt": (".ppt", ".pptx"), "演示文稿": (".ppt", ".pptx"), "幻灯片": (".ppt", ".pptx"),
    "pdf": (".pdf",), "word": (".doc", ".docx"), "表格": (".xls", ".xlsx", ".csv"), "excel": (".xls", ".xlsx"),
}


def _folders(row: dict[str, Any]) -> list[str]:
    """The last folders of a registry row's path ("02 精密级合金粉末宏量开发")."""
    from pathlib import PureWindowsPath

    return [part for part in PureWindowsPath(str(row.get("path") or "")).parent.parts[-3:] if len(part) >= 2]


# Opening a workbook takes about 3 s and 100 MB of memory per MB of .xlsx: past this size a
# question would wait half a minute and hold a gigabyte (evaluation/doc_capabilities_v1/REPORT.md).
TABLE_MAX_MB = 10


@dataclass
class DocumentAccess:
    """What these questions need from the app: the registry's files, a file's parsed passages
    and metadata, and the path of its original."""

    rows: Callable[[], list[dict[str, Any]]]
    parents: Callable[[str], list[dict[str, Any]]]
    metadata: Callable[[str], dict[str, Any]]
    original: Callable[[str], Path | None]
    notes: NoteStore
    workbooks: WorkbookCache = field(default_factory=WorkbookCache)


@dataclass
class DocumentTurn:
    """How a document question goes: its plan, the file, and what to remember for the reply."""

    plan: QueryPlan
    target: dict[str, Any] = field(default_factory=dict)
    pending: dict[str, Any] = field(default_factory=dict)
    choices: list[str] = field(default_factory=list)


class DocumentNodes:
    """Mixed into ``AgentNodes``, which provides llm, settings and the model-call helpers."""

    documents: DocumentAccess | None = None
    summary_llm: Any = None

    # ---------------- which file ----------------

    def _document_turn(self, question: str, state: State) -> DocumentTurn | None:
        """A document question this turn, decided by rules; None lets the planner look at it."""
        if self.documents is None:
            return None
        pending = state.get("pending_file") or {}
        if pending:
            turn = self._answer_to_pending(question, pending)
            if turn is not None:
                return turn
        scope = state.get("scope_doc_ids") or []
        if len(scope) > 1:
            return None
        table_cue = TABLE_CUE.search(question) is not None
        if len(scope) == 1:
            row = next((r for r in self._rows() if r["file_id"] == scope[0]), None)
            if row is None:
                return None
            if wants_whole_file(question):
                return self._target_turn(SUMMARIZE, question, row)
            if table_cue and is_table_file(row["name"]):
                return self._target_turn(TABLE, question, row)
            return None
        # Cheap wording checks first: most questions never list the registry.
        if not (table_cue or SUMMARY_CUE.search(question)):
            return None
        rows, names = files_named_in(question, self._rows())
        if not rows:
            # "总结一下这份PPT" right after a turn that found or cited it: the file meant is there.
            if SUMMARY_CUE.search(question) and THIS_FILE.search(question) and wants_whole_file(question):
                recent = self._recent_files(question, state)
                if recent:
                    return self._resolve(SUMMARIZE, question, recent)
            return None
        folders = [part for row in rows for part in _folders(row)]
        if wants_whole_file(question, names, folders):
            return self._resolve(SUMMARIZE, question, rows)
        tables = [row for row in rows if is_table_file(row["name"])]
        if table_cue and tables:
            return self._resolve(TABLE, question, tables)
        return None

    def _planner_document_turn(self, plan: QueryPlan, question: str) -> DocumentTurn | None:
        """The planner called it a document question (no 《》, fuzzy name): it must name a file."""
        if self.documents is None or plan.intent not in DOCUMENT_INTENTS or not plan.file_name.strip():
            return None
        rows = match_files(plan.file_name, self._rows())
        if plan.intent == TABLE:
            rows = [row for row in rows if is_table_file(row["name"])]
        return self._resolve(plan.intent, question, rows) if rows else None

    def _rows(self) -> list[dict[str, Any]]:
        return self.documents.rows() if self.documents else []

    def _recent_files(self, question: str, state: State) -> list[dict[str, Any]]:
        """Registry rows of the files the previous turn found, cited or summarized, narrowed to
        the kind the question names ("这份PPT" -> .ppt/.pptx); [] when nothing fits."""
        ids: list[str] = []
        target = state.get("target_file") or {}
        if target.get("file_id"):
            ids.append(str(target["file_id"]))
        ids += [str(item.get("file_id") or "") for item in state.get("files") or []]
        ids += [str(item.get("doc_id") or "") for item in state.get("citations") or []]
        ids = [i for i in dict.fromkeys(ids) if i]
        if not ids:
            return []
        rows = {row["file_id"]: row for row in self._rows()}
        recent = [rows[i] for i in ids if i in rows]
        kinds = [suffixes for word, suffixes in KIND_WORDS.items() if word in question.lower()]
        if kinds:
            recent = [row for row in recent if any(row["name"].lower().endswith(s) for group in kinds for s in group)]
        return recent

    def _target_turn(self, intent: str, question: str, row: dict[str, Any], *, confirmed: bool = False) -> DocumentTurn:
        plan = QueryPlan(intent=intent, rewritten_query=question[:500], file_name=row["name"])
        target = {"file_id": row["file_id"], "name": row["name"], "path": row.get("path") or "", "confirmed": confirmed}
        return DocumentTurn(plan, target)

    def _resolve(self, intent: str, question: str, rows: list[dict[str, Any]]) -> DocumentTurn:
        candidates = distinct_files(rows)
        if len(candidates) > 1:
            candidates = narrow_by_folder(question, candidates) or candidates
        if len(candidates) == 1:
            return self._target_turn(intent, question, candidates[0])
        return self._ask_which(intent, question, candidates)

    def _ask_which(self, intent: str, question: str, candidates: list[dict[str, Any]]) -> DocumentTurn:
        candidates = candidates[:8]
        name = Path(candidates[0]["name"]).stem
        verb = "总结" if intent == SUMMARIZE else "统计"
        same = len({row["name"] for row in candidates}) == 1
        head = f"有 {len(candidates)} 份《{name}》" if same else f"有 {len(candidates)} 份文件符合"
        lines = [f"{head}，要{verb}哪一份？"]
        lines += [f"{n}. {folder_label(row)}" for n, row in enumerate(candidates, start=1)]
        plan = QueryPlan(
            intent=intent, rewritten_query=question[:500], needs_clarification=True,
            clarification_question="\n".join(lines),
        )
        pending = {
            "kind": "choose", "intent": intent, "question": question,
            "candidates": [{"file_id": r["file_id"], "name": r["name"], "path": r.get("path") or ""} for r in candidates],
        }
        return DocumentTurn(plan, pending=pending, choices=[f"第 {n} 份" for n in range(1, len(candidates) + 1)])

    def _answer_to_pending(self, reply: str, pending: dict[str, Any]) -> DocumentTurn | None:
        """The reply to "哪一份？" or "要开始吗？"; anything else is a new question."""
        question = pending.get("question") or reply
        if pending.get("kind") == "choose":
            row = pick_choice(reply, pending.get("candidates") or [])
            return self._target_turn(pending["intent"], question, row) if row else None
        if pending.get("kind") == "confirm" and CONFIRM.match(reply):
            return self._target_turn(SUMMARIZE, question, pending["file"], confirmed=True)
        return None

    # ---------------- reading the whole file ----------------

    def read_whole_file(self, state: State) -> dict[str, Any]:
        """Notes on every passage of the file; generate_answer then writes from all of them."""
        target = state.get("target_file") or {}
        doc_id = target.get("file_id", "")
        source = target.get("name", "")
        parents = self.documents.parents(doc_id) if self.documents else []
        if not parents:
            return self._reply(f"《{Path(source).stem}》还没有读取完成，暂时无法总结。", refused=True)
        store = self.documents.notes
        settings = self.settings
        model = getattr(settings, "summary_model", "") or getattr(settings, "llm_model", "")
        concurrency = int(getattr(settings, "summary_concurrency", 6))
        if not target.get("confirmed"):
            estimate = estimate_reading(parents, store, doc_id, model, concurrency=concurrency)
            if estimate["tokens"] > int(getattr(settings, "summary_confirm_tokens", 150000)):
                return self._ask_before_reading(state, target, estimate)
        writer = get_stream_writer()

        def progress(done: int, total: int) -> None:
            writer({"kind": "reading", "done": done, "total": total, "file": source})

        result = read_passages(
            doc_id, parents, self._summary_call, store,
            batch_tokens=int(getattr(settings, "summary_batch_tokens", 4000)),
            concurrency=concurrency, progress=progress,
        )
        coverage = describe_coverage(parents, result, self.documents.metadata(doc_id))
        if not result.notes:
            if result.error:
                return {"coverage": coverage, "model_error": result.error, "answer": ""}
            return self._reply(f"《{Path(source).stem}》没有可读的文字内容。", refused=True, coverage=coverage)
        notes = result.notes
        original = {note.index: str(parents[note.index].get("text", "")) for note in notes}
        limit = int(getattr(settings, "summary_reduce_tokens", 12000))
        try:
            notes = compress_notes(doc_id, notes, self._summary_call, store, result, limit_tokens=limit, concurrency=concurrency)
        except MODEL_SERVICE_ERRORS as exc:
            logger.warning("Compressing notes failed, using them as they are: %r", exc)
        coverage.update(calls=result.calls, input_tokens=result.input_tokens, output_tokens=result.output_tokens)
        coverage["notes"] = len(notes)
        # The answer is written from the notes and checked against the passages themselves, so a
        # slip in a note is caught too. Merged notes span many passages: too much to send, so a
        # very long file's answer is checked against its notes, and the coverage says so.
        checked_on_passages = notes is result.notes
        coverage["checked_against"] = "passages" if checked_on_passages else "notes"
        documents = [
            {"doc_id": doc_id, "source": source, "locator": note.locator, "heading": note.heading,
             "text": original[note.index] if checked_on_passages else note.text,
             "parent_id": f"{doc_id}-p{note.index}", "quote": note.quote}
            for note in notes
        ]
        blocks = [
            f"[证据 {n}｜来源：{source}｜位置：{note.locator}]\n{note.text}" for n, note in enumerate(notes, start=1)
        ]
        seen: set[str] = set()
        citations = []
        for note in notes:
            if note.locator not in seen:
                seen.add(note.locator)
                citations.append({"source": source, "locator": note.locator, "doc_id": doc_id, "quote": note.quote})
        return {
            "context_docs": documents,
            "documents": documents,
            "evidence": "\n\n".join(blocks),
            "citations": citations,
            "answer_profile": "summary",
            "coverage": coverage,
            "retrieval_attempts": 1,
        }

    def _summary_call(self, messages: list[Any], max_tokens: int | None = None):
        if self.summary_llm is None:
            return self._call_llm(messages, max_tokens=max_tokens)
        try:
            return self._call_llm(messages, max_tokens=max_tokens, llm=self.summary_llm)
        except (openai.NotFoundError, openai.BadRequestError) as exc:
            # The notes model was retired or renamed: the answer model can write them too.
            logger.warning("Notes model refused the request, using the answer model: %r", exc)
            return self._call_llm(messages, max_tokens=max_tokens)

    def _ask_before_reading(self, state: State, target: dict[str, Any], estimate: dict[str, Any]) -> dict[str, Any]:
        """Very long files cost real money and minutes: say how much, and read on a yes."""
        words = estimate["characters"] / 10000
        minutes = estimate["seconds"] / 60
        cost = f"、约 ¥{max(estimate['yuan'], 0.01):.2f}" if estimate.get("yuan") else ""
        name = Path(target["name"]).stem
        time_text = f"约 {max(1, round(minutes))} 分钟"
        message = (
            f"《{name}》约 {words:.0f} 万字，从头读完{time_text}{cost}（之后再问不用重读）。\n"
            f"要开始吗？也可以直接问具体章节。"
        )
        pending = {"kind": "confirm", "question": state.get("question", ""), "file": target}
        return self._reply(message, pending_file=pending, choices=["开始通读"])

    @staticmethod
    def _reply(message: str, **extra: Any) -> dict[str, Any]:
        return {"answer": message, "messages": [AIMessage(content=message)], "turn_done": True, **extra}

    def _finish_summary(self, state: State, answer: str) -> dict[str, Any]:
        """Keep the sources the answer cites (a long file has hundreds of notes) and say what
        was covered, under the answer so it is never mistaken for part of the summary."""
        citations = list(state.get("citations") or [])
        refs = [ref for _, claim_refs in cited_claims(answer) for ref in claim_refs]
        kept = citations
        if refs:
            used, _ = evidence_for(refs, [{"source": c["source"], "locator": c["locator"]} for c in citations])
            kept = [citations[i] for i in sorted(used)]
        lines = [answer]
        if not state.get("grounded") and not state.get("check_skipped") and state.get("unsupported_claims"):
            lines.append(f"有 {len(state['unsupported_claims'])} 处没能和文件内容对上，请点开出处确认。")
        note = (state.get("coverage") or {}).get("text")
        if note:
            lines.append(note)
        final = "\n\n".join(lines)
        return {"answer": final, "citations": kept, "messages": [AIMessage(content=final)]}

    # ---------------- table questions ----------------

    def query_table(self, state: State) -> dict[str, Any]:
        """Turn the question into a fixed query, run it locally, write the answer from the result."""
        target = state.get("target_file") or {}
        doc_id, source = target.get("file_id", ""), target.get("name", "")
        # Not state["question"]: after "哪一份？" that is only the reply.
        question = state.get("rewritten_query") or state.get("question", "")
        fallback = {"intent": "knowledge", "scope_doc_ids": [doc_id], "table_fallback": True}
        path = self.documents.original(doc_id) if self.documents else None
        if path is None:
            return self._plain_refusal(f"找不到《{Path(source).stem}》的原文件，没法统计。可以先确认文件还在原来的位置。")
        if os.path.getsize(path) > TABLE_MAX_MB * 1024 * 1024:
            return self._plain_refusal(
                f"《{Path(source).stem}》超过 {TABLE_MAX_MB} MB，暂时不能整表统计。可以问具体内容，我从已读取的部分里找。"
            )
        try:
            workbook = self.documents.workbooks.get(path)
        except Exception as exc:
            logger.warning("Reading workbook %s failed: %s", path, exc)
            return self._plain_refusal(f"《{Path(source).stem}》打不开，没法统计（可能有密码保护或已损坏）。")
        if not workbook.sheets:
            return self._plain_refusal(f"《{Path(source).stem}》里没有可统计的表格。")
        try:
            query = self._invoke_structured(TableQuery, [
                SystemMessage(content=TABLE_PLAN_PROMPT),
                HumanMessage(content=f"工作簿：{source}\n{schema_text(workbook)}\n\n问题：{question}"),
            ])
        except MODEL_SERVICE_ERRORS as exc:
            logger.warning("Table query planning failed: %r", exc)
            return {"model_error": describe_model_error(exc), "answer": ""}
        except (ValueError, ValidationError, json.JSONDecodeError) as exc:
            logger.warning("Table query not understood: %s", exc)
            return fallback
        if query.op == "none":
            return fallback
        sheet = sheet_asked_for(question, query.sheet, workbook, source)
        if sheet != query.sheet:
            # "《0.政策补贴汇总》一共多少条": the small model puts the first sheet (or the file's
            # name) where the question named none; the question is about every sheet.
            logger.info("Table query: sheet %r is not named in the question, using every sheet", query.sheet)
            query = query.model_copy(update={"sheet": sheet})
        try:
            result = run_query(query, workbook)
        except TableError as exc:
            return self._plain_refusal(str(exc), table_query=query.model_dump())
        answer, citations = answer_text(result, source, question)
        for citation in citations:
            citation["doc_id"] = doc_id
        return {
            "answer": answer,
            "citations": citations,
            "grounded": True,
            "refused": False,
            "check_mode": "table",
            "table_query": query.model_dump(),
        }

    @staticmethod
    def _plain_refusal(message: str, **extra: Any) -> dict[str, Any]:
        return {"answer": message, "refused": True, "grounded": False, **extra}
