import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from api import main
from core.rag_system import RAGSystem
from db.parent_store_manager import ParentStoreManager
from ingestion.models import ParentChunk


DOC_ID = "b" * 24


def _settings(tmp_path):
    settings = SimpleNamespace(
        parsed_dir=tmp_path / "parsed",
        parent_store_dir=tmp_path / "parents",
        upload_dir=tmp_path / "uploads",
    )
    for path in (settings.parsed_dir, settings.upload_dir):
        path.mkdir()
    return settings


def _parent(number, text):
    return ParentChunk(
        parent_id=f"{DOC_ID}-p{number}",
        doc_id=DOC_ID,
        source="policy.md",
        heading=f"第{number}节",
        locator="",
        text=text,
    )


def test_list_parents_keeps_numeric_order(tmp_path):
    store = ParentStoreManager(SimpleNamespace(parent_store_dir=tmp_path))
    store.save_document(DOC_ID, [_parent(10, "十"), _parent(2, "二"), _parent(0, "零")])

    assert [p["text"] for p in store.list_parents(DOC_ID)] == ["零", "二", "十"]


def test_get_document_returns_units_parents_and_file_flag(tmp_path):
    settings = _settings(tmp_path)
    (settings.parsed_dir / f"{DOC_ID}.json").write_text(
        json.dumps(
            {
                "doc_id": DOC_ID,
                "source": "policy.md",
                "file_type": ".md",
                "units": [{"heading": "总则", "locator": "", "text": "第一条", "metadata": {}}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    ParentStoreManager(settings).save_document(DOC_ID, [_parent(0, "完整父块")])
    rag = RAGSystem(settings)

    document = rag.get_document(DOC_ID)
    assert document["units"] == [{"heading": "总则", "locator": "", "text": "第一条"}]
    assert document["parents"][0]["text"] == "完整父块"
    assert document["characters"] == 3
    assert document["has_file"] is False

    (settings.upload_dir / DOC_ID).mkdir()
    (settings.upload_dir / DOC_ID / "policy.md").write_text("原文", encoding="utf-8")
    assert rag.get_document(DOC_ID)["has_file"] is True


@pytest.mark.parametrize("doc_id", ["../secret", "B" * 24, ""])
def test_get_document_rejects_unsafe_or_unknown_ids(tmp_path, doc_id):
    assert RAGSystem(_settings(tmp_path)).get_document(doc_id) is None


def test_document_endpoint_returns_404_for_missing_document(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "system", RAGSystem(_settings(tmp_path)))
    with pytest.raises(HTTPException) as error:
        main.get_document(DOC_ID)
    assert error.value.status_code == 404
