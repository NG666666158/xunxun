from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class FileFilter(BaseModel):
    """Conditions of a "find a file" question; dates are resolved in code, not by the model."""

    time_phrase: str = Field(default="", max_length=40)
    time_field: Literal["added", "modified", "any"] = "any"
    file_types: list[str] = Field(default_factory=list, max_length=6)
    folder: str = Field(default="", max_length=200)
    name_keywords: list[str] = Field(default_factory=list, max_length=8)
    topic: str = Field(default="", max_length=200)


class QueryPlan(BaseModel):
    intent: Literal["knowledge", "find_file", "chitchat", "summarize_file", "table_query"]
    rewritten_query: str = Field(min_length=1, max_length=500)
    needs_clarification: bool = False
    clarification_question: str = ""
    file_filter: FileFilter = Field(default_factory=FileFilter)
    file_name: str = Field(default="", max_length=200)


class SearchRewrite(BaseModel):
    query: str = Field(min_length=1, max_length=500)


class FaithfulnessCheck(BaseModel):
    grounded: bool
    unsupported_claims: list[str] = Field(default_factory=list)
