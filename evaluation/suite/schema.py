"""One JSONL schema for every question set, and one for corpora.

Question (one JSON object per line)::

    {
      "id": "duretrieval:edb58f52",          # unique within the set, prefixed by the set
      "query": "国家法定节假日共多少天",
      "language": "zh",                      # "zh" | "en"
      "type": "single",                      # primary category, one per question
      "tags": ["colloquial", "digits"],      # any number of secondary labels
      "gold_doc_ids": ["2c4fe63d..."],       # dataset document ids (corpus.jsonl "doc_id")
      "gold_passages": [                     # optional, finer than documents
        {"doc_id": "...", "text": "...", "locator": "page:3", "keywords": ["..."], "match": "doc"}
      ],
      "gold_answer": "11天",                  # "" when the set has none
      "answerable": true,                    # false: no gold document is in the index
      "needs_all": false,                    # true: every gold document is needed (multi-doc)
      "source": {"dataset": "duretrieval", "version": "<revision>", "original_id": "..."}
    }

``gold_passages[].match`` says how a retrieved passage is judged against it:

* ``doc``          the passage belongs to the gold document (``doc_id``);
* ``text``         the gold text (an evidence sentence) is inside the passage, compared
                   without spacing, punctuation or case;
* ``books_strict`` books_v1 / complex_v1's strict rule (gold file, page overlap, keywords);
* ``chunk``        the passage's chunk id is one of ``chunk_ids`` (benchmark_v1).

Corpus (one JSON object per line)::

    {"doc_id": "...", "title": "...", "text": "...", "source": "doc000123.txt", "metadata": {...}}

``source`` is the file name the app would show; it is also the "【title】" prefix the
chunker puts in front of every child (``chunk_context_prefix``).
"""

from __future__ import annotations

from typing import Any

QUESTION_FIELDS = {
    "id": str, "query": str, "language": str, "type": str, "tags": list, "gold_doc_ids": list,
    "gold_passages": list, "gold_answer": str, "answerable": bool, "needs_all": bool, "source": dict,
}
MATCH_RULES = ("doc", "text", "books_strict", "chunk")
LANGUAGES = ("zh", "en")


def question(**fields: Any) -> dict[str, Any]:
    """A question with defaults filled in, validated."""
    row = {"tags": [], "gold_doc_ids": [], "gold_passages": [], "gold_answer": "", "answerable": True,
           "needs_all": False, "language": "zh", **fields}
    validate_question(row)
    return row


def validate_question(row: dict[str, Any]) -> None:
    for name, kind in QUESTION_FIELDS.items():
        if name not in row:
            raise ValueError(f"{row.get('id')}: missing {name}")
        if not isinstance(row[name], kind):
            raise ValueError(f"{row.get('id')}: {name} must be {kind.__name__}")
    if not row["query"].strip():
        raise ValueError(f"{row['id']}: empty query")
    if row["language"] not in LANGUAGES:
        raise ValueError(f"{row['id']}: language {row['language']!r}")
    if row["answerable"] and not (row["gold_doc_ids"] or row["gold_passages"]):
        raise ValueError(f"{row['id']}: answerable without gold")
    for passage in row["gold_passages"]:
        if passage.get("match", "doc") not in MATCH_RULES:
            raise ValueError(f"{row['id']}: unknown match rule {passage.get('match')!r}")
    if "dataset" not in row["source"]:
        raise ValueError(f"{row['id']}: source.dataset missing")


def validate_corpus_row(row: dict[str, Any]) -> None:
    for name in ("doc_id", "text", "source"):
        if not isinstance(row.get(name), str) or not row[name]:
            raise ValueError(f"corpus row {row.get('doc_id')}: {name} missing")
